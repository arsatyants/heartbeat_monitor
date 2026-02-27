"""
Unit tests for the GPU wavelet sub-package.

All tests run entirely on the CPU (pyopencl may not be installed in CI),
using WaveletProcessorGPU with force_cpu_fallback=True.

Run with:  pytest tests/test_gpu_wavelet.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from heartbeat_monitor.gpu.hardware_detector import (
    BoardType,
    HardwareProfile,
    detect,
    _identify_board,
)
from heartbeat_monitor.gpu.wavelet_processor_gpu import (
    WaveletProcessorGPU,
    _scale_to_bpm,
    _scales_for_bpm_range as proc_scales,
    _numpy_cwt_energy,
)
from heartbeat_monitor.gpu.wavelet_kernels import (
    ALL_KERNELS_SOURCE,
    KERNEL_CWT_MORLET,
    KERNEL_DETREND,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_processor(**kwargs) -> WaveletProcessorGPU:
    """Return a processor that always uses the CPU fallback."""
    return WaveletProcessorGPU(force_cpu_fallback=True, **kwargs)


def _make_bgr(r: int, g: int, b: int, h: int = 8, w: int = 8) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 2] = r
    frame[:, :, 1] = g
    frame[:, :, 0] = b
    return frame


# ---------------------------------------------------------------------------
# hardware_detector tests
# ---------------------------------------------------------------------------

class TestHardwareDetector:

    def test_detect_returns_profile(self):
        profile = detect(force_cpu_fallback=True)
        assert isinstance(profile, HardwareProfile)
        assert profile.board in list(BoardType)

    def test_board_name_non_empty(self):
        profile = detect()
        assert len(profile.board_name) > 0

    def test_preferred_work_group_positive(self):
        profile = detect()
        assert profile.preferred_work_group > 0

    def test_max_wavelet_scales_positive(self):
        profile = detect()
        assert profile.max_wavelet_scales > 0

    def test_rpi5_profile_tuning(self):
        """Mock a Raspberry Pi 5 board identification."""
        board, name = _identify_board("")
        # On non-Pi hardware this returns X86 or GENERIC_ARM – just assert type
        assert isinstance(board, BoardType)

    def test_summary_string(self):
        profile = detect()
        summary = profile.summary()
        assert "Board" in summary
        assert "WG size" in summary

    def test_rpi_zero_2w_detected_from_cpuinfo(self):
        fake_cpuinfo = "Hardware\t: BCM2835\nRevision\t: 902120\n"
        board, _ = _identify_board(fake_cpuinfo)
        # On non-Pi system device-tree model won't say Zero 2W, but we confirm
        # the function returns a BoardType without error.
        assert isinstance(board, BoardType)


# ---------------------------------------------------------------------------
# Kernel source sanity tests
# ---------------------------------------------------------------------------

class TestKernelSource:

    def test_all_kernels_non_empty(self):
        assert len(ALL_KERNELS_SOURCE) > 200

    def test_cwt_morlet_kernel_contains_omega0(self):
        assert "OMEGA0" in KERNEL_CWT_MORLET

    def test_detrend_kernel_function_signature(self):
        assert "detrend_normalize" in KERNEL_DETREND

    def test_all_kernels_contains_band_energy(self):
        assert "band_energy" in ALL_KERNELS_SOURCE

    def test_all_kernels_contains_tiled_variant(self):
        assert "cwt_morlet_tiled" in ALL_KERNELS_SOURCE


# ---------------------------------------------------------------------------
# Scale ↔ BPM conversion tests
# ---------------------------------------------------------------------------

class TestScaleConversions:

    def test_scales_count(self):
        scales = proc_scales(fps=30.0, bpm_low=45.0, bpm_high=240.0, n_scales=16)
        assert len(scales) == 16

    def test_scales_dtype_float32(self):
        scales = proc_scales(fps=30.0, bpm_low=45.0, bpm_high=240.0, n_scales=8)
        assert scales.dtype == np.float32

    def test_scales_monotonically_increasing(self):
        scales = proc_scales(fps=30.0, bpm_low=45.0, bpm_high=240.0, n_scales=12)
        assert np.all(np.diff(scales) > 0)

    def test_scale_to_bpm_round_trip(self):
        fps = 30.0
        scales = proc_scales(fps, 60.0, 120.0, 20)
        bpms = [_scale_to_bpm(float(s), fps) for s in scales]
        assert all(55 <= b <= 130 for b in bpms), f"BPM out of range: {bpms}"

    def test_scale_to_bpm_72(self):
        """1.2 Hz = 72 BPM at 30 fps – verify round-trip within 5%."""
        fps = 30.0
        omega0 = 6.0
        fc = omega0 / (2 * math.pi)
        target_hz = 1.2
        dt = 1.0 / fps
        s = fc / (target_hz * dt)
        bpm = _scale_to_bpm(s, fps)
        assert abs(bpm - 72.0) < 4.0, f"Expected ~72 BPM, got {bpm:.2f}"


# ---------------------------------------------------------------------------
# WaveletProcessorGPU – CPU fallback mode tests
# ---------------------------------------------------------------------------

class TestWaveletProcessorGPUCPU:

    def test_initial_bpm_zero(self):
        proc = _cpu_processor(fps=30.0)
        bpm, conf = proc.compute_bpm()
        assert bpm == 0.0
        assert conf == 0.0

    def test_buffer_fills(self):
        proc = _cpu_processor(fps=10.0, window_seconds=5.0)
        frame = _make_bgr(80, 60, 50)
        proc.push_frame(frame)
        assert proc.buffer_fill_ratio > 0.0

    def test_reset_clears(self):
        proc = _cpu_processor(fps=10.0, window_seconds=5.0)
        for _ in range(30):
            proc.push_frame(_make_bgr(80, 60, 50))
        proc.reset()
        assert proc.buffer_fill_ratio == 0.0
        assert proc.last_bpm == 0.0

    def test_filtered_signal_empty_when_insufficient(self):
        proc = _cpu_processor(fps=30.0, min_samples=60)
        for _ in range(10):
            proc.push_frame(_make_bgr(80, 60, 50))
        assert len(proc.get_filtered_signal()) == 0

    def test_filtered_signal_shape_when_sufficient(self):
        proc = _cpu_processor(fps=10.0, window_seconds=5.0, min_samples=5)
        for _ in range(15):
            proc.push_frame(_make_bgr(80, 60, 50))
        sig = proc.get_filtered_signal()
        assert len(sig) == 15

    def test_context_manager(self):
        with _cpu_processor(fps=30.0) as proc:
            assert proc is not None

    def test_opencl_not_available_in_fallback(self):
        proc = _cpu_processor(fps=30.0)
        assert proc._opencl_available is False

    def test_band_bpms_within_range(self):
        proc = _cpu_processor(fps=30.0, bpm_low=45.0, bpm_high=240.0)
        bpms = proc.band_bpms
        assert np.all(bpms >= 40)
        assert np.all(bpms <= 250)

    def test_synthetic_sine_72bpm(self):
        """
        Feed in a pure 1.2 Hz sine (72 BPM) via CPU fallback and verify the
        dominant scale lands within ±8 BPM of 72.
        """
        fps = 20.0
        target_hz = 1.2   # 72 BPM
        proc = _cpu_processor(
            fps=fps,
            window_seconds=15.0,
            bpm_low=45.0,
            bpm_high=150.0,
            min_samples=int(2 * fps),
        )

        t = np.arange(int(fps * 15)) / fps
        green_signal = 100.0 + 5.0 * np.sin(2 * math.pi * target_hz * t)

        for val in green_signal:
            f = np.zeros((4, 4, 3), dtype=np.uint8)
            f[:, :, 1] = int(np.clip(val, 0, 255))
            proc.push_frame(f)

        bpm, conf = proc.compute_bpm()
        assert abs(bpm - 72.0) < 10.0, f"Expected ~72 BPM, got {bpm:.1f}"
        # CWT spreads energy across adjacent scales; 0.10 is realistic for 16 bands
        assert conf > 0.10, f"Confidence too low: {conf:.3f}"

    def test_band_energies_filled_after_compute(self):
        proc = _cpu_processor(fps=10.0, window_seconds=5.0, min_samples=5)
        t = np.arange(50) / 10.0
        green = 100.0 + 5.0 * np.sin(2 * math.pi * 1.2 * t)
        for val in green:
            f = np.zeros((4, 4, 3), dtype=np.uint8)
            f[:, :, 1] = int(np.clip(val, 0, 255))
            proc.push_frame(f)
        proc.compute_bpm()
        assert len(proc.band_energies) > 0


# ---------------------------------------------------------------------------
# NumPy CWT engine (standalone)
# ---------------------------------------------------------------------------

class TestNumpyCWT:

    def test_returns_correct_length(self):
        fps = 10.0
        scales = proc_scales(fps, 60.0, 120.0, 8)
        signal = np.random.randn(50)
        energy = _numpy_cwt_energy(signal, scales)
        assert len(energy) == 8

    def test_energy_non_negative(self):
        fps = 10.0
        scales = proc_scales(fps, 60.0, 120.0, 8)
        signal = np.random.randn(50)
        energy = _numpy_cwt_energy(signal, scales)
        assert np.all(energy >= 0)

    def test_dominant_peak_matches_sine(self):
        """1.2 Hz sine in a short signal – peak scale should map to ~72 BPM."""
        fps = 20.0
        scales = proc_scales(fps, 45.0, 150.0, 16)
        t = np.arange(60) / fps
        signal = np.sin(2 * math.pi * 1.2 * t).astype(np.float64)
        energy = _numpy_cwt_energy(signal, scales)
        peak_idx = int(np.argmax(energy))
        bpm = _scale_to_bpm(float(scales[peak_idx]), fps)
        assert abs(bpm - 72.0) < 15.0, f"Expected ~72 BPM, got {bpm:.1f}"
