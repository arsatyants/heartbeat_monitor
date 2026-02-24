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

    def test_spo2_no_data_returns_zero(self):
        """SpO2 should return 0.0 when there is insufficient data."""
        sp = SignalProcessor(fps=30.0, window_seconds=10.0)
        spo2 = sp.compute_spo2()
        assert spo2 == 0.0

    def test_spo2_synthetic_signal(self):
        """Test SpO2 calculation with synthetic pulsatile signals."""
        fps = 30.0
        target_hz = 1.2  # 72 BPM
        sp = SignalProcessor(fps=fps, window_seconds=15.0, min_samples=int(2 * fps))

        # Synthesise red and green channels with pulsatile components
        # Simulating healthy oxygenated blood (SpO2 ~96-98%)
        t = np.arange(int(fps * 15)) / fps
        
        # Green channel: higher mean, moderate pulsatile component
        green_signal = 120 + 8 * np.sin(2 * np.pi * target_hz * t)
        
        # Red channel: lower mean, smaller pulsatile component (oxygenated blood absorbs more IR/less red modulation)
        red_signal = 100 + 3 * np.sin(2 * np.pi * target_hz * t)

        # Push as fake frames
        for g_val, r_val in zip(green_signal, red_signal):
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            frame[:, :, 1] = int(np.clip(g_val, 0, 255))  # Green
            frame[:, :, 2] = int(np.clip(r_val, 0, 255))  # Red
            sp.push_frame(frame)

        spo2 = sp.compute_spo2()
        # Should be in valid physiological range
        assert 70.0 <= spo2 <= 100.0, f"SpO2 {spo2:.1f}% out of valid range"
        assert spo2 > 0.0, "SpO2 should be computed"

    def test_spo2_clamped_to_range(self):
        """SpO2 should be clamped to 70-100% range."""
        fps = 30.0
        sp = SignalProcessor(fps=fps, window_seconds=10.0, min_samples=int(2 * fps))

        # Create pathological signal that would give extreme ratios
        t = np.arange(int(fps * 10)) / fps
        target_hz = 1.0
        
        # Extreme case: very high red pulsation, low green
        green_signal = 150 + 2 * np.sin(2 * np.pi * target_hz * t)
        red_signal = 80 + 20 * np.sin(2 * np.pi * target_hz * t)

        for g_val, r_val in zip(green_signal, red_signal):
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            frame[:, :, 1] = int(np.clip(g_val, 0, 255))
            frame[:, :, 2] = int(np.clip(r_val, 0, 255))
            sp.push_frame(frame)

        spo2 = sp.compute_spo2()
        # Should be clamped
        assert 70.0 <= spo2 <= 100.0, f"SpO2 {spo2:.1f}% not properly clamped"

    def test_spo2_reset_clears_value(self):
        """Reset should clear the last SpO2 value."""
        sp = SignalProcessor(fps=10.0, window_seconds=5.0)
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        for _ in range(60):
            sp.push_frame(frame)
        sp.compute_spo2()  # Compute to set last_spo2
        sp.reset()
        assert sp.last_spo2 == 0.0


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
