"""
PPG signal processor using Wavelet Transform.

Alternative implementation using multi-band wavelet analysis instead of
Butterworth filtering. Provides more robust frequency detection across
different heart rate ranges.

Algorithm
---------
1. Extract the mean green-channel intensity from each incoming frame.
2. Keep a rolling buffer of the last ``window_seconds`` samples.
3. Apply Continuous Wavelet Transform (CWT) with 6 frequency bands.
4. Calculate power in each band to identify the dominant heart rate band.
5. Find peak frequency within the dominant band for precise BPM estimate.

References
----------
- Addison P.S., "Wavelet transforms and the ECG: a review."
  Physiol. Meas., 2005.
- Peng et al., "Extracting heart rate variability using wavelet transform."
  IEEE EMBS, 2006.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import pywt

logger = logging.getLogger(__name__)


class SignalProcessorWavelet:
    """
    Rolling PPG signal analyser using Wavelet Transform.

    This alternative implementation uses wavelet decomposition into 6 frequency
    bands to robustly identify heart rate, especially useful when the signal
    contains multiple frequency components or noise.

    Parameters
    ----------
    fps:
        Frames-per-second of the incoming video stream.
    window_seconds:
        Length of the analysis window in seconds.
        Recommended: 10 – 15 s.
    bpm_low:
        Lower BPM boundary (default 45 BPM).
    bpm_high:
        Upper BPM boundary (default 240 BPM).
    n_bands:
        Number of frequency bands for wavelet analysis (default 6).
    wavelet:
        Wavelet family to use (default 'morl' - Morlet wavelet).
    min_samples:
        Minimum number of samples required before a BPM estimate.
    """

    def __init__(
        self,
        fps: float = 30.0,
        window_seconds: float = 12.0,
        bpm_low: float = 45.0,
        bpm_high: float = 240.0,
        n_bands: int = 6,
        wavelet: str = "morl",
        min_samples: int | None = None,
    ) -> None:
        self.fps = fps
        self.window_seconds = window_seconds
        self.bpm_low = bpm_low
        self.bpm_high = bpm_high
        self.n_bands = n_bands
        self.wavelet = wavelet

        maxlen = int(fps * window_seconds)
        self._buffer: Deque[float] = deque(maxlen=maxlen)
        self._buffer_red: Deque[float] = deque(maxlen=maxlen)  # Red channel for SpO2
        self.min_samples: int = min_samples if min_samples is not None else int(2 * fps)

        # Define frequency bands (in Hz)
        low_hz = self.bpm_low / 60.0
        high_hz = self.bpm_high / 60.0
        self.band_edges = np.linspace(low_hz, high_hz, n_bands + 1)

        # Pre-compute scales for CWT
        self._scales = self._compute_scales()

        # Last computed results
        self._last_bpm: float = 0.0
        self._last_confidence: float = 0.0
        self._last_spo2: float = 0.0
        self._last_band_powers: np.ndarray = np.zeros(n_bands)
        self._last_dominant_band: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(self, frame: "np.ndarray") -> None:
        """
        Extract the mean green and red intensity from *frame* and append to buffers.

        Parameters
        ----------
        frame:
            BGR image array (H × W × 3, uint8).
        """
        green_mean = float(np.mean(frame[:, :, 1]))  # channel 1 = Green in BGR
        red_mean = float(np.mean(frame[:, :, 2]))    # channel 2 = Red in BGR
        self._buffer.append(green_mean)
        self._buffer_red.append(red_mean)

    def compute_bpm(self) -> Tuple[float, float]:
        """
        Return ``(bpm, confidence)`` from the current signal buffer using wavelet analysis.

        The confidence score is the ratio of power in the dominant band to the
        total power across all bands (0 – 1).

        Returns (0.0, 0.0) when there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return 0.0, 0.0

        try:
            signal = np.array(self._buffer, dtype=np.float64)
            # Detrend (remove slow drift / DC offset)
            signal = signal - np.mean(signal)

            # Continuous Wavelet Transform
            coefficients, frequencies = pywt.cwt(
                signal, self._scales, self.wavelet, sampling_period=1.0 / self.fps
            )

            # Calculate power spectrum (squared magnitude of coefficients)
            power_spectrum = np.abs(coefficients) ** 2

            # Calculate power in each frequency band
            band_powers = np.zeros(self.n_bands)
            for i in range(self.n_bands):
                low_f = self.band_edges[i]
                high_f = self.band_edges[i + 1]
                band_mask = (frequencies >= low_f) & (frequencies < high_f)
                if band_mask.any():
                    band_powers[i] = np.sum(power_spectrum[band_mask, :])

            # Find dominant band (highest power)
            dominant_band = int(np.argmax(band_powers))
            self._last_dominant_band = dominant_band
            self._last_band_powers = band_powers

            # Extract frequencies and power within dominant band
            low_f = self.band_edges[dominant_band]
            high_f = self.band_edges[dominant_band + 1]
            band_mask = (frequencies >= low_f) & (frequencies < high_f)

            if not band_mask.any():
                return 0.0, 0.0

            # Average power across time for each frequency in the band
            band_power_profile = np.mean(power_spectrum[band_mask, :], axis=1)
            band_freqs = frequencies[band_mask]

            # Find peak frequency in the band
            peak_idx = int(np.argmax(band_power_profile))
            peak_freq = band_freqs[peak_idx]

            # Parabolic interpolation for sub-bin precision
            if 0 < peak_idx < len(band_power_profile) - 1:
                alpha = band_power_profile[peak_idx - 1]
                beta = band_power_profile[peak_idx]
                gamma = band_power_profile[peak_idx + 1]

                if alpha - 2 * beta + gamma != 0:
                    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                    freq_step = band_freqs[1] - band_freqs[0] if len(band_freqs) > 1 else 0
                    peak_freq = band_freqs[peak_idx] + p * freq_step

            bpm = peak_freq * 60.0

            # Confidence: dominant band power / total power
            total_power = float(band_powers.sum())
            confidence = float(band_powers[dominant_band] / total_power) if total_power > 0 else 0.0
            
            # Penalty for edge cases to avoid false confidence
            # 1. Reduce confidence if stuck at lowest frequency (likely noise/DC)
            if bpm < self.bpm_low + 5:  # Within 5 BPM of lower bound
                confidence *= 0.5
            
            # 2. Reduce confidence if dominant band is lowest and has >80% of power (likely no real signal)
            if dominant_band == 0 and confidence > 0.8:
                confidence *= 0.6

            self._last_bpm = bpm
            self._last_confidence = confidence
            return bpm, confidence

        except Exception as e:
            logger.warning("Wavelet BPM calculation failed: %s", e)
            return 0.0, 0.0

    @property
    def buffer_fill_ratio(self) -> float:
        """How full the rolling buffer is (0 – 1)."""
        return len(self._buffer) / self._buffer.maxlen

    @property
    def last_bpm(self) -> float:
        return self._last_bpm

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def last_spo2(self) -> float:
        return self._last_spo2

    @property
    def band_powers(self) -> np.ndarray:
        """Power in each frequency band (useful for visualization)."""
        return self._last_band_powers

    @property
    def dominant_band(self) -> int:
        """Index of the dominant frequency band."""
        return self._last_dominant_band

    def compute_spo2(self) -> float:
        """
        Estimate oxygen saturation (SpO2) using red and green channels.

        Same implementation as the Butterworth version for consistency.

        Returns
        -------
        spo2:
            Estimated oxygen saturation percentage (0-100).
            Returns 0.0 when there is insufficient data or calculation fails.
        """
        if len(self._buffer) < self.min_samples or len(self._buffer_red) < self.min_samples:
            return 0.0

        try:
            # Extract signals
            green_signal = np.array(self._buffer, dtype=np.float64)
            red_signal = np.array(self._buffer_red, dtype=np.float64)

            # Calculate DC components (mean values)
            dc_green = np.mean(green_signal)
            dc_red = np.mean(red_signal)

            if dc_green == 0 or dc_red == 0:
                return 0.0

            # Detrend signals
            green_detrended = green_signal - dc_green
            red_detrended = red_signal - dc_red

            # Use wavelet transform to extract AC components
            green_coeffs, _ = pywt.cwt(
                green_detrended, self._scales, self.wavelet, sampling_period=1.0 / self.fps
            )
            red_coeffs, _ = pywt.cwt(
                red_detrended, self._scales, self.wavelet, sampling_period=1.0 / self.fps
            )

            # Calculate AC components (RMS of wavelet coefficients in heart rate band)
            # Use dominant band for more robust estimation
            low_f = self.band_edges[self._last_dominant_band]
            high_f = self.band_edges[self._last_dominant_band + 1]
            band_mask = (self._get_frequencies() >= low_f) & (self._get_frequencies() < high_f)

            if not band_mask.any():
                return 0.0

            ac_green = np.sqrt(np.mean(np.abs(green_coeffs[band_mask, :]) ** 2))
            ac_red = np.sqrt(np.mean(np.abs(red_coeffs[band_mask, :]) ** 2))

            if ac_green == 0:
                return 0.0

            # Calculate the ratio of ratios (R)
            ratio_red = ac_red / dc_red
            ratio_green = ac_green / dc_green
            R = ratio_red / ratio_green

            # Empirical calibration formula
            spo2 = 110 - 25 * R

            # Clamp to physiologically plausible range
            spo2 = max(70.0, min(100.0, spo2))

            self._last_spo2 = spo2
            return spo2

        except Exception as e:
            logger.warning("SpO2 calculation failed: %s", e)
            return 0.0

    def get_filtered_signal(self) -> np.ndarray:
        """
        Return the current wavelet-reconstructed PPG waveform (for plotting).
        Returns an empty array if there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return np.array([])

        try:
            signal = np.array(self._buffer, dtype=np.float64)
            signal -= np.mean(signal)

            # Perform CWT and reconstruct using only heart rate frequencies
            coefficients, frequencies = pywt.cwt(
                signal, self._scales, self.wavelet, sampling_period=1.0 / self.fps
            )

            # Keep only heart rate band
            low_hz = self.bpm_low / 60.0
            high_hz = self.bpm_high / 60.0
            band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)

            # Zero out coefficients outside the band
            filtered_coeffs = coefficients.copy()
            filtered_coeffs[~band_mask, :] = 0

            # Simple reconstruction by averaging across scales
            reconstructed = np.mean(filtered_coeffs, axis=0)
            return reconstructed

        except Exception as e:
            logger.warning("Signal reconstruction failed: %s", e)
            return np.array([])

    def get_fft_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the wavelet power spectrum (frequencies in BPM, power).
        Returns empty arrays if there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return np.array([]), np.array([])

        try:
            signal = np.array(self._buffer, dtype=np.float64)
            signal = signal - np.mean(signal)

            # CWT
            coefficients, frequencies = pywt.cwt(
                signal, self._scales, self.wavelet, sampling_period=1.0 / self.fps
            )

            # Average power across time
            power_profile = np.mean(np.abs(coefficients) ** 2, axis=1)

            # Convert frequencies to BPM
            freqs_bpm = frequencies * 60.0

            # Restrict to valid BPM band
            band_mask = (freqs_bpm >= self.bpm_low) & (freqs_bpm <= self.bpm_high)
            return freqs_bpm[band_mask], power_profile[band_mask]

        except Exception as e:
            logger.warning("FFT data extraction failed: %s", e)
            return np.array([]), np.array([])

    def reset(self) -> None:
        """Clear the signal buffers."""
        self._buffer.clear()
        self._buffer_red.clear()
        self._last_bpm = 0.0
        self._last_confidence = 0.0
        self._last_spo2 = 0.0
        self._last_band_powers = np.zeros(self.n_bands)
        self._last_dominant_band = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_scales(self) -> np.ndarray:
        """
        Compute wavelet scales corresponding to heart rate frequencies.

        Returns array of scales for CWT analysis.
        """
        # Frequency range in Hz
        low_hz = self.bpm_low / 60.0
        high_hz = self.bpm_high / 60.0

        # Get center frequency of the wavelet
        center_freq = pywt.central_frequency(self.wavelet)

        # Compute scales: scale = fc / (freq * dt)
        # Use logarithmic spacing for better frequency resolution
        n_scales = 64  # Number of scales
        frequencies = np.logspace(np.log10(low_hz), np.log10(high_hz), n_scales)
        scales = center_freq / (frequencies * (1.0 / self.fps))

        return scales

    def _get_frequencies(self) -> np.ndarray:
        """Get frequencies corresponding to the current scales."""
        center_freq = pywt.central_frequency(self.wavelet)
        frequencies = center_freq / (self._scales * (1.0 / self.fps))
        return frequencies
