"""
PPG signal processor.

Algorithm
---------
1. Extract the mean green-channel intensity from each incoming frame.
   (Green is most sensitive to haemoglobin absorption changes.)
2. Keep a rolling buffer of the last ``window_seconds`` samples.
3. Apply a Butterworth bandpass filter (default: 0.75 – 4.0 Hz = 45 – 240 BPM).
4. Compute the FFT of the filtered signal; the dominant peak gives the
   instantaneous heart-rate estimate.

References
----------
- Verkruysse W. et al., "Remote plethysmographic imaging using ambient light."
  Opt Express, 2008.
- De Haan G. et al., "Robust pulse rate from chrominance-based rPPG." 2013.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Rolling PPG signal analyser.

    Parameters
    ----------
    fps:
        Frames-per-second of the incoming video stream.  Must match the
        camera's actual capture rate for accurate BPM computation.
    window_seconds:
        Length of the analysis window in seconds.  Longer windows give
        smoother BPM estimates but slower response to changes.
        Recommended: 10 – 15 s.
    bpm_low:
        Lower BPM boundary for the bandpass filter (default 45 BPM).
    bpm_high:
        Upper BPM boundary for the bandpass filter (default 240 BPM).
    filter_order:
        Order of the Butterworth filter (default 4).
    min_samples:
        Minimum number of samples required before a BPM estimate is
        emitted.  Defaults to 2 × fps (≈ 2 seconds of data).
    """

    def __init__(
        self,
        fps: float = 30.0,
        window_seconds: float = 12.0,
        bpm_low: float = 45.0,
        bpm_high: float = 240.0,
        filter_order: int = 4,
        min_samples: int | None = None,
    ) -> None:
        self.fps = fps
        self.window_seconds = window_seconds
        self.bpm_low = bpm_low
        self.bpm_high = bpm_high
        self.filter_order = filter_order

        maxlen = int(fps * window_seconds)
        self._buffer: Deque[float] = deque(maxlen=maxlen)
        self._buffer_red: Deque[float] = deque(maxlen=maxlen)  # Red channel for SpO2
        self.min_samples: int = min_samples if min_samples is not None else int(2 * fps)

        # Pre-build the bandpass filter (second-order sections)
        self._sos = self._build_filter()

        # Last computed result
        self._last_bpm: float = 0.0
        self._last_confidence: float = 0.0   # peak power / total power
        self._last_spo2: float = 0.0         # oxygen saturation %

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
        Return ``(bpm, confidence)`` from the current signal buffer.

        The confidence score is the ratio of spectral power at the dominant
        peak to the total power within the valid frequency band (0 – 1).

        Returns (0.0, 0.0) when there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return 0.0, 0.0

        signal = np.array(self._buffer, dtype=np.float64)
        # Detrend (remove slow drift / DC offset)
        signal = signal - np.mean(signal)

        # Bandpass filter
        filtered = sosfilt(self._sos, signal)

        # FFT
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)   # Hz
        power = np.abs(np.fft.rfft(filtered)) ** 2

        # Restrict to valid BPM band
        low_hz = self.bpm_low / 60.0
        high_hz = self.bpm_high / 60.0
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)

        if not band_mask.any():
            return 0.0, 0.0

        band_power = power[band_mask]
        band_freqs = freqs[band_mask]

        peak_idx = int(np.argmax(band_power))
        peak_freq = band_freqs[peak_idx]
        
        # Parabolic interpolation for sub-bin frequency resolution
        # This refines the peak frequency using neighboring bins
        if 0 < peak_idx < len(band_power) - 1:
            # Use three points around the peak for parabolic fit
            alpha = band_power[peak_idx - 1]
            beta = band_power[peak_idx]
            gamma = band_power[peak_idx + 1]
            
            # Parabolic interpolation formula
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            
            # Refined frequency with sub-bin precision
            freq_step = band_freqs[1] - band_freqs[0] if len(band_freqs) > 1 else 0
            peak_freq = band_freqs[peak_idx] + p * freq_step
        
        bpm = peak_freq * 60.0

        # Confidence: peak / total band power
        total = float(band_power.sum())
        confidence = float(band_power[peak_idx] / total) if total > 0 else 0.0

        self._last_bpm = bpm
        self._last_confidence = confidence
        return bpm, confidence

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

    def compute_spo2(self) -> float:
        """
        Estimate oxygen saturation (SpO2) using red and green channels.

        The algorithm computes the ratio of AC/DC components for both
        red and green channels, then applies an empirical calibration formula.

        Formula: SpO2 ≈ 110 - 25 × (AC_red/DC_red) / (AC_green/DC_green)

        Returns
        -------
        spo2:
            Estimated oxygen saturation percentage (0-100).
            Returns 0.0 when there is insufficient data or calculation fails.

        Notes
        -----
        - This is an approximation using visible spectrum channels
        - True pulse oximetry uses red (~660nm) and infrared (~940nm)
        - Results should be considered indicative, not clinical-grade
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

            # Apply bandpass filter to get pulsatile (AC) components
            green_filtered = sosfilt(self._sos, green_detrended)
            red_filtered = sosfilt(self._sos, red_detrended)

            # Calculate AC components (RMS of filtered signals)
            ac_green = np.sqrt(np.mean(green_filtered ** 2))
            ac_red = np.sqrt(np.mean(red_filtered ** 2))

            if ac_green == 0:
                return 0.0

            # Calculate the ratio of ratios (R)
            ratio_red = ac_red / dc_red
            ratio_green = ac_green / dc_green
            R = ratio_red / ratio_green

            # Empirical calibration formula for camera-based PPG
            # Standard formula: SpO2 = 110 - 25 * R
            # Adjusted for visible spectrum (red/green instead of red/IR)
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
        Return the current bandpass-filtered PPG waveform (for plotting).
        Returns an empty array if there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return np.array([])
        signal = np.array(self._buffer, dtype=np.float64)
        signal -= np.mean(signal)
        return sosfilt(self._sos, signal)

    def get_fft_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the current FFT spectrum (frequencies in BPM, power).
        Returns empty arrays if there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return np.array([]), np.array([])

        signal = np.array(self._buffer, dtype=np.float64)
        signal = signal - np.mean(signal)
        filtered = sosfilt(self._sos, signal)

        # FFT
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)  # Hz
        power = np.abs(np.fft.rfft(filtered)) ** 2

        # Convert frequencies to BPM
        freqs_bpm = freqs * 60.0

        # Restrict to valid BPM band
        band_mask = (freqs_bpm >= self.bpm_low) & (freqs_bpm <= self.bpm_high)
        return freqs_bpm[band_mask], power[band_mask]

    def reset(self) -> None:
        """Clear the signal buffers."""
        self._buffer.clear()
        self._buffer_red.clear()
        self._last_bpm = 0.0
        self._last_confidence = 0.0
        self._last_spo2 = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_filter(self) -> np.ndarray:
        """Construct a Butterworth bandpass filter (SOS form)."""
        nyq = self.fps / 2.0
        low = (self.bpm_low / 60.0) / nyq
        high = (self.bpm_high / 60.0) / nyq
        # Clamp to valid range
        low = max(1e-4, min(low, 0.999))
        high = max(low + 1e-4, min(high, 0.999))
        sos = butter(self.filter_order, [low, high], btype="bandpass", output="sos")
        return sos
