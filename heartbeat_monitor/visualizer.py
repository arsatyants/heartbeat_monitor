"""
Real-time overlay visualiser.

Draws the following elements onto each video frame:
  • A highlighted "region of interest" rectangle in the frame centre.
  • A scrolling waveform strip (PPG signal).
  • BPM readout with colour-coded confidence indicator.
  • Finger-placement status hint.
  • Optional frame-rate counter.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_GREEN  = (0, 220,  80)
_RED    = (0,  50, 220)
_YELLOW = (0, 210, 210)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_CYAN   = (220, 200,  0)
_PURPLE = (150, 100, 180)
_DARK   = (30, 30, 30)


class Visualizer:
    """
    Draws heartbeat monitoring UI onto OpenCV frames in-place.

    Parameters
    ----------
    resolution:
        (width, height) of the video frame.
    waveform_height:
        Pixel height of the scrolling waveform panel at the bottom of the frame.
    roi_fraction:
        Fraction of the shorter frame dimension used for the ROI square.
    show_fps:
        Whether to overlay computed FPS in the top-right corner.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        waveform_height: int = 80,
        roi_fraction: float = 0.35,
        show_fps: bool = True,
        show_fft: bool = True,
    ) -> None:
        self.w, self.h = resolution
        self.waveform_height = waveform_height
        self.roi_fraction = roi_fraction
        self.show_fps = show_fps
        self.show_fft = show_fft

        # Pre-compute ROI rectangle
        side = int(min(self.w, self.h) * roi_fraction)
        cx, cy = self.w // 2, self.h // 2
        self.roi = (cx - side // 2, cy - side // 2, side, side)  # x, y, w, h

        # Waveform scroll buffer
        self._wave_buf = np.zeros(self.w, dtype=np.float64)

        # FPS tracking
        self._fps_tick = cv2.getTickCount()
        self._fps_display: float = 0.0

        # Seconds clock (resets every minute)
        self._start_time = cv2.getTickCount()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_roi(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) of the region of interest."""
        return self.roi

    def update_waveform(self, value: float) -> None:
        """Push a new PPG sample into the scrolling waveform buffer."""
        self._wave_buf = np.roll(self._wave_buf, -1)
        self._wave_buf[-1] = value

    def draw(
        self,
        frame: np.ndarray,
        bpm: float,
        confidence: float,
        buffer_fill: float,
        finger_detected: bool,
        filtered_signal: Optional[np.ndarray] = None,
        fft_freqs: Optional[np.ndarray] = None,
        fft_power: Optional[np.ndarray] = None,
        spo2: float = 0.0,
    ) -> np.ndarray:
        """
        Annotate *frame* in-place and return it.

        Parameters
        ----------
        frame:
            BGR frame from the camera.
        bpm:
            Current heart-rate estimate in BPM.
        confidence:
            Spectral confidence score (0 – 1).
        buffer_fill:
            How full the signal buffer is (0 – 1).  Drives the loading bar.
        finger_detected:
            Whether a finger is currently covering the lens.
        filtered_signal:
            Optional 1-D array of the latest filtered PPG waveform to plot.
        fft_freqs:
            Optional 1-D array of FFT frequencies in BPM.
        fft_power:
            Optional 1-D array of FFT power spectrum.
        spo2:
            Oxygen saturation percentage (70-100).
        """
        self._update_fps()

        x, y, rw, rh = self.roi

        # --- ROI highlight box -----------------------------------------------
        color = _GREEN if finger_detected else _YELLOW
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)
        cv2.putText(
            frame, "Place finger here" if not finger_detected else "Scanning...",
            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

        # --- BPM readout -------------------------------------------------------
        self._draw_bpm(frame, bpm, confidence, finger_detected)

        # --- SpO2 readout ------------------------------------------------------
        if spo2 > 0:
            self._draw_spo2(frame, spo2, finger_detected)

        # --- Buffer fill bar ---------------------------------------------------
        self._draw_fill_bar(frame, buffer_fill)

        # --- Waveform strip at the bottom ------------------------------------
        if filtered_signal is not None and len(filtered_signal) > 1:
            self._draw_waveform(frame, filtered_signal)

        # --- FFT spectrum panel (right side) ----------------------------------
        if self.show_fft and fft_freqs is not None and fft_power is not None:
            if len(fft_freqs) > 0 and len(fft_power) > 0:
                self._draw_fft_spectrum(frame, fft_freqs, fft_power, bpm)

        # --- FPS counter -------------------------------------------------------
        if self.show_fps:
            cv2.putText(
                frame,
                f"FPS {self._fps_display:.1f}",
                (self.w - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA,
            )

        # --- Seconds clock (resets every minute) ------------------------------
        elapsed = (cv2.getTickCount() - self._start_time) / cv2.getTickFrequency()
        seconds = int(elapsed) % 60
        cv2.putText(
            frame,
            f"{seconds:02d}s",
            (self.w - 100, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA,
        )

        return frame

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_bpm(
        self,
        frame: np.ndarray,
        bpm: float,
        confidence: float,
        finger_detected: bool,
    ) -> None:
        if bpm > 0 and finger_detected:
            # Colour: green (high confidence) → yellow → red (low)
            if confidence >= 0.5:
                col = _GREEN
            elif confidence >= 0.3:
                col = _YELLOW
            else:
                col = _RED

            cv2.putText(
                frame, f"{bpm:.0f} BPM",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, _BLACK, 5, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{bpm:.0f} BPM",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 3, cv2.LINE_AA,
            )
            # Confidence mini-bar
            bar_w = int(120 * confidence)
            cv2.rectangle(frame, (16, 60), (136, 72), _DARK, -1)
            cv2.rectangle(frame, (16, 60), (16 + bar_w, 72), col, -1)
            cv2.putText(
                frame, f"conf {confidence * 100:.0f}%",
                (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
            )
        else:
            status = "Warming up..." if finger_detected else "No finger detected"
            cv2.putText(
                frame, status,
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2, cv2.LINE_AA,
            )

    def _draw_spo2(
        self,
        frame: np.ndarray,
        spo2: float,
        finger_detected: bool,
    ) -> None:
        """Draw SpO2 percentage below the BPM readout."""
        if spo2 > 0 and finger_detected:
            # Color coding: green (healthy), yellow (moderate), red (low)
            if spo2 >= 95:
                col = _GREEN
            elif spo2 >= 90:
                col = _YELLOW
            else:
                col = _RED

            cv2.putText(
                frame, f"SpO2: {spo2:.0f}%",
                (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BLACK, 3, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"SpO2: {spo2:.0f}%",
                (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA,
            )

    def _draw_fill_bar(self, frame: np.ndarray, fill: float) -> None:
        bar_w = int((self.w - 32) * min(fill, 1.0))
        y0, y1 = self.h - self.waveform_height - 12, self.h - self.waveform_height - 4
        cv2.rectangle(frame, (16, y0), (self.w - 16, y1), _DARK, -1)
        cv2.rectangle(frame, (16, y0), (16 + bar_w, y1), _CYAN, -1)
        cv2.putText(
            frame, "buffer",
            (16, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _CYAN, 1, cv2.LINE_AA,
        )

    def _draw_waveform(self, frame: np.ndarray, signal: np.ndarray) -> None:
        """Draw a scrolling waveform in a dark strip at the bottom of the frame."""
        panel_top = self.h - self.waveform_height
        # Dark background
        cv2.rectangle(
            frame,
            (0, panel_top),
            (self.w, self.h),
            _DARK,
            -1,
        )

        # Normalise signal to [0, 1]
        sig = signal[-self.w:] if len(signal) >= self.w else signal
        mn, mx = sig.min(), sig.max()
        rng = mx - mn if mx != mn else 1.0
        norm = (sig - mn) / rng

        margin = 6
        plot_h = self.waveform_height - 2 * margin
        xs = np.linspace(0, self.w - 1, len(norm)).astype(int)
        ys = (panel_top + margin + (1.0 - norm) * plot_h).astype(int)

        pts = np.column_stack([xs, ys])
        cv2.polylines(frame, [pts[:, None, :]], False, _GREEN, 1, cv2.LINE_AA)

        cv2.putText(
            frame, "PPG",
            (4, panel_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
        )

    def _draw_fft_spectrum(
        self,
        frame: np.ndarray,
        freqs: np.ndarray,
        power: np.ndarray,
        peak_bpm: float,
    ) -> None:
        """Draw FFT spectrum as a vertical bar graph on the right side."""
        panel_w = 160
        panel_x = self.w - panel_w - 10
        panel_y = 100
        panel_h = self.h - panel_y - self.waveform_height - 20

        # Dark background
        cv2.rectangle(frame, (panel_x, panel_y), (self.w - 10, panel_y + panel_h), _DARK, -1)

        # Normalize power
        if power.max() > 0:
            norm_power = power / power.max()
        else:
            norm_power = power

        # Draw bars
        num_bars = min(len(freqs), 100)
        if num_bars > 0:
            step = len(freqs) // num_bars
            bar_width = max(1, (panel_w - 20) // num_bars)

            for i in range(num_bars):
                idx = i * step
                if idx >= len(freqs):
                    break
                freq_bpm = freqs[idx]
                pwr = norm_power[idx]
                bar_h = int(pwr * (panel_h - 20))

                x = panel_x + 10 + i * bar_width
                y_bottom = panel_y + panel_h - 10
                y_top = y_bottom - bar_h

                # Highlight peak frequency
                color = _YELLOW if abs(freq_bpm - peak_bpm) < 2.0 else _PURPLE
                cv2.rectangle(frame, (x, y_top), (x + bar_width - 1, y_bottom), color, -1)

        # Label
        cv2.putText(
            frame, "FFT",
            (panel_x + 10, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Spectrum",
            (panel_x + 10, panel_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1, cv2.LINE_AA,
        )

    def _update_fps(self) -> None:
        """Compute rolling FPS."""
        now = cv2.getTickCount()
        elapsed = (now - self._fps_tick) / cv2.getTickFrequency()
        if elapsed > 0:
            self._fps_display = 1.0 / elapsed
        self._fps_tick = now
