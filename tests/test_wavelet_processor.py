"""
Unit tests for SignalProcessorWavelet.
Run with:  pytest tests/test_wavelet_processor.py
"""

from __future__ import annotations

import numpy as np
import pytest

from heartbeat_monitor.signal_processor_wavelet import SignalProcessorWavelet


class TestSignalProcessorWavelet:

    def test_no_data_returns_zero(self):
        """Should return 0.0 BPM when there is no data."""
        sp = SignalProcessorWavelet(fps=30.0, window_seconds=10.0)
        bpm, conf = sp.compute_bpm()
        assert bpm == 0.0
        assert conf == 0.0

    def test_buffer_fill_ratio_grows(self):
        """Buffer fill ratio should increase as frames are added."""
        sp = SignalProcessorWavelet(fps=10.0, window_seconds=5.0)
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        assert sp.buffer_fill_ratio == 0.0
        sp.push_frame(frame)
        assert sp.buffer_fill_ratio > 0.0

    def test_reset_clears_buffer(self):
        """Reset should clear all buffers and state."""
        sp = SignalProcessorWavelet(fps=10.0, window_seconds=5.0)
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        for _ in range(20):
            sp.push_frame(frame)
        sp.reset()
        assert sp.buffer_fill_ratio == 0.0
        assert sp.last_bpm == 0.0
        assert sp.last_spo2 == 0.0

    def test_synthetic_sine_detected(self):
        """Feed a pure 1.2 Hz sine (72 BPM) and verify detection."""
        fps = 30.0
        target_hz = 1.2  # 72 BPM
        sp = SignalProcessorWavelet(fps=fps, window_seconds=15.0, min_samples=int(2 * fps))

        # Synthesise: mean brightness = 100, PPG oscillation amplitude = 5
        t = np.arange(int(fps * 15)) / fps
        green_signal = 100 + 5 * np.sin(2 * np.pi * target_hz * t)

        # Push as fake frames
        for val in green_signal:
            f = np.zeros((4, 4, 3), dtype=np.uint8)
            f[:, :, 1] = int(np.clip(val, 0, 255))
            sp.push_frame(f)

        bpm, conf = sp.compute_bpm()
        expected_bpm = target_hz * 60.0
        assert abs(bpm - expected_bpm) < 5.0, f"Expected ~{expected_bpm} BPM, got {bpm:.1f}"
        assert conf > 0.3, f"Confidence too low: {conf:.2f}"

    def test_band_powers_computed(self):
        """Band powers should be computed and accessible."""
        sp = SignalProcessorWavelet(fps=30.0, window_seconds=10.0, n_bands=6)
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        
        # Add enough frames
        for _ in range(int(30 * 10)):
            sp.push_frame(frame)
        
        sp.compute_bpm()
        
        # Should have 6 band powers
        assert len(sp.band_powers) == 6
        assert sp.dominant_band >= 0
        assert sp.dominant_band < 6

    def test_spo2_no_data_returns_zero(self):
        """SpO2 should return 0.0 when there is insufficient data."""
        sp = SignalProcessorWavelet(fps=30.0, window_seconds=10.0)
        spo2 = sp.compute_spo2()
        assert spo2 == 0.0

    def test_spo2_synthetic_signal(self):
        """Test SpO2 calculation with synthetic pulsatile signals."""
        fps = 30.0
        target_hz = 1.2  # 72 BPM
        sp = SignalProcessorWavelet(fps=fps, window_seconds=15.0, min_samples=int(2 * fps))

        # Synthesise red and green channels with pulsatile components
        t = np.arange(int(fps * 15)) / fps
        green_signal = 120 + 8 * np.sin(2 * np.pi * target_hz * t)
        red_signal = 100 + 3 * np.sin(2 * np.pi * target_hz * t)

        # Push as fake frames
        for g_val, r_val in zip(green_signal, red_signal):
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            frame[:, :, 1] = int(np.clip(g_val, 0, 255))
            frame[:, :, 2] = int(np.clip(r_val, 0, 255))
            sp.push_frame(frame)

        # Compute BPM first (to set dominant band)
        sp.compute_bpm()
        
        spo2 = sp.compute_spo2()
        # Should be in valid physiological range
        assert 70.0 <= spo2 <= 100.0, f"SpO2 {spo2:.1f}% out of valid range"

    def test_filtered_signal_shape(self):
        """Filtered signal should have correct length."""
        sp = SignalProcessorWavelet(fps=10.0, window_seconds=5.0, min_samples=5)
        frame = np.full((8, 8, 3), 80, dtype=np.uint8)
        for _ in range(10):
            sp.push_frame(frame)
        sig = sp.get_filtered_signal()
        assert len(sig) > 0  # Should return reconstructed signal

    def test_wavelet_fft_data(self):
        """Should return wavelet power spectrum data."""
        sp = SignalProcessorWavelet(fps=30.0, window_seconds=10.0, min_samples=60)
        frame = np.full((8, 8, 3), 100, dtype=np.uint8)
        
        # Add enough frames
        for _ in range(300):
            sp.push_frame(frame)
        
        freqs, power = sp.get_fft_data()
        
        # Should have data
        assert len(freqs) > 0
        assert len(power) > 0
        assert len(freqs) == len(power)
        
        # Frequencies should be in BPM range
        assert np.all(freqs >= 45)
        assert np.all(freqs <= 240)

    def test_different_band_counts(self):
        """Should work with different number of bands."""
        for n_bands in [4, 6, 8]:
            sp = SignalProcessorWavelet(fps=30.0, window_seconds=10.0, n_bands=n_bands)
            assert sp.n_bands == n_bands
            assert len(sp.band_edges) == n_bands + 1
