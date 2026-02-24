"""
Unit tests for SignalProcessor and FingerDetector.
Run with:  pytest tests/
"""

from __future__ import annotations

import numpy as np
import pytest

from heartbeat_monitor.signal_processor import SignalProcessor
from heartbeat_monitor.finger_detector import FingerDetector


# ---------------------------------------------------------------------------
# SignalProcessor tests
# ---------------------------------------------------------------------------

class TestSignalProcessor:

    def test_no_data_returns_zero(self):
        sp = SignalProcessor(fps=30.0, window_seconds=10.0)
        bpm, conf = sp.compute_bpm()
        assert bpm == 0.0
        assert conf == 0.0

    def test_buffer_fill_ratio_grows(self):
        sp = SignalProcessor(fps=10.0, window_seconds=5.0)
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        assert sp.buffer_fill_ratio == 0.0
        sp.push_frame(frame)
        assert sp.buffer_fill_ratio > 0.0

    def test_reset_clears_buffer(self):
        sp = SignalProcessor(fps=10.0, window_seconds=5.0)
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        for _ in range(20):
            sp.push_frame(frame)
        sp.reset()
        assert sp.buffer_fill_ratio == 0.0
        assert sp.last_bpm == 0.0

    def test_synthetic_sine_detected(self):
        """Feed a pure 1.2 Hz sine (72 BPM) and verify we get a close estimate."""
        fps = 30.0
        target_hz = 1.2           # 72 BPM
        sp = SignalProcessor(fps=fps, window_seconds=15.0, min_samples=int(2 * fps))

        # Synthesise: mean brightness = 100, PPG oscillation amplitude = 5
        t = np.arange(int(fps * 15)) / fps
        green_signal = 100 + 5 * np.sin(2 * np.pi * target_hz * t)

        # Push as fake frames where only green channel matters
        for val in green_signal:
            f = np.zeros((4, 4, 3), dtype=np.uint8)
            f[:, :, 1] = int(np.clip(val, 0, 255))
            sp.push_frame(f)

        bpm, conf = sp.compute_bpm()
        expected_bpm = target_hz * 60.0
        assert abs(bpm - expected_bpm) < 5.0, f"Expected ~{expected_bpm} BPM, got {bpm:.1f}"
        assert conf > 0.5, f"Confidence too low: {conf:.2f}"

    def test_filtered_signal_shape(self):
        sp = SignalProcessor(fps=10.0, window_seconds=5.0, min_samples=5)
        frame = np.full((8, 8, 3), 80, dtype=np.uint8)
        for _ in range(10):
            sp.push_frame(frame)
        sig = sp.get_filtered_signal()
        assert len(sig) == 10


# ---------------------------------------------------------------------------
# FingerDetector tests
# ---------------------------------------------------------------------------

class TestFingerDetector:

    def _make_frame(self, r, g, b, noise=0) -> np.ndarray:
        """Create a uniform-colour frame with optional Gaussian noise."""
        rng = np.random.default_rng(42)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = np.clip(r + rng.integers(-noise, noise + 1, (480, 640)), 0, 255)
        frame[:, :, 1] = np.clip(g + rng.integers(-noise, noise + 1, (480, 640)), 0, 255)
        frame[:, :, 0] = np.clip(b + rng.integers(-noise, noise + 1, (480, 640)), 0, 255)
        return frame

    def test_finger_dark_reddish(self):
        fd = FingerDetector(brightness_threshold=100, variance_threshold=800)
        # Dark reddish frame (finger on lens)
        frame = self._make_frame(r=80, g=40, b=30, noise=3)
        assert fd.is_finger(frame) is True

    def test_no_finger_bright_scene(self):
        fd = FingerDetector()
        # Bright daylight-like frame
        frame = self._make_frame(r=200, g=180, b=160, noise=20)
        assert fd.is_finger(frame) is False

    def test_no_finger_uniform_but_bright(self):
        fd = FingerDetector(brightness_threshold=100)
        # Very uniform but too bright
        frame = self._make_frame(r=150, g=140, b=130, noise=0)
        assert fd.is_finger(frame) is False
