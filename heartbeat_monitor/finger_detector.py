"""
Finger-on-lens detector.

When a finger covers the IMX500 camera, the frame becomes:
  - Dominated by reddish / pinkish tones (blood tissue).
  - Much darker than a normal scene.
  - Low in spatial variance (uniform colour, no edges).

This module provides a lightweight heuristic check used to gate the
PPG signal processor so it does not emit spurious BPM values when the
camera is uncovered.
"""

from __future__ import annotations

import numpy as np


class FingerDetector:
    """
    Heuristic detector: is the camera lens covered by a finger?

    Parameters
    ----------
    brightness_threshold:
        Maximum allowed *mean* pixel brightness (0 – 255).  A covered
        lens is much darker than an open scene.  Default: 100.
    variance_threshold:
        Maximum allowed *spatial variance* of green channel intensity.
        A covered lens yields a nearly uniform field.  Default: 800.
    red_dominance:
        Minimum ratio ``mean_red / mean_green`` required to confirm
        skin tone is present.  Default: 1.05 (red must be ≥ 5 % brighter
        than green).
    """

    def __init__(
        self,
        brightness_threshold: float = 100.0,
        variance_threshold: float = 800.0,
        red_dominance: float = 1.05,
    ) -> None:
        self.brightness_threshold = brightness_threshold
        self.variance_threshold = variance_threshold
        self.red_dominance = red_dominance

    def is_finger(self, frame: np.ndarray) -> bool:
        """
        Return *True* if *frame* looks like a finger covering the lens.

        Parameters
        ----------
        frame:
            BGR image array (H × W × 3, uint8).
        """
        b_ch = frame[:, :, 0].astype(np.float64)
        g_ch = frame[:, :, 1].astype(np.float64)
        r_ch = frame[:, :, 2].astype(np.float64)

        mean_r = float(r_ch.mean())
        mean_g = float(g_ch.mean())
        mean_b = float(b_ch.mean())
        brightness = (mean_r + mean_g + mean_b) / 3.0
        variance = float(g_ch.var())

        red_ratio = mean_r / (mean_g + 1e-6)

        dark_enough     = brightness < self.brightness_threshold
        uniform_enough  = variance < self.variance_threshold
        skin_tone       = red_ratio >= self.red_dominance

        return dark_enough and uniform_enough and skin_tone
