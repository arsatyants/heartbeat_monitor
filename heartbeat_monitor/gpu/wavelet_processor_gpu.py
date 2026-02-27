"""
GPU-accelerated PPG signal processor using OpenCL parallel CWT.

This module is a fully independent reimplementation of the heartbeat
measurement pipeline.  It does **not** import any sibling module.

Architecture
------------

1.  :class:`WaveletProcessorGPU` acts as the single public object.
2.  On first call to :meth:`WaveletProcessorGPU.open`, the OpenCL context,
    command queue, and compiled kernels are created.  Persistent device
    buffers are allocated once and reused every frame to avoid PCIe / DMA
    overhead.
3.  Per-frame flow:

    .. code-block:: text

        push_frame(bgr_frame)
           │
           ▼
        Extract mean green channel → append to deque
           │
           ▼ (every 3rd frame, when buffer has enough data)
        compute_bpm()
           │
           ├─ Host: detrend mean/std  (numpy, ~μs)
           │
           ├─ H→D: upload float32 signal to GPU buffer
           │
           ├─ GPU kernel: detrend_normalize      (parallel, O(N))
           │
           ├─ GPU kernel: cwt_morlet             (parallel, O(n_scales × N))
           │     or cwt_morlet_tiled for RPi Zero 2W
           │
           ├─ GPU kernel: band_energy            (parallel, O(n_scales))
           │
           ├─ D→H: download energy vector        (~n_scales × 4 bytes)
           │
           └─ Host: pick dominant scale → BPM, confidence  (numpy, ~μs)

Graceful degradation
--------------------
When ``pyopencl`` is not installed, or no OpenCL platform is found, the
class falls back to a pure-NumPy CWT implementation (identical algorithm,
CPU only).  The public API is unchanged.

Thread safety
-------------
Not thread-safe.  Use one instance per thread / process.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Tuple

import numpy as np

from heartbeat_monitor.gpu.hardware_detector import (
    HardwareProfile,
    BoardType,
    detect as detect_hardware,
)
from heartbeat_monitor.gpu.wavelet_kernels import (
    ALL_KERNELS_SOURCE,
    KERNEL_BAND_ENERGY,
    KERNEL_CWT_MORLET,
    KERNEL_CWT_MORLET_TILED,
    KERNEL_DETREND,
)

logger = logging.getLogger(__name__)

# Centre frequency of the Morlet wavelet (must match OMEGA0 in kernel source)
_MORLET_OMEGA0: float = 6.0
# Morlet centre frequency in normalised units: f_c = omega_0 / (2 * pi)
_MORLET_FC: float = _MORLET_OMEGA0 / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Scale ↔ frequency conversions
# ---------------------------------------------------------------------------

def _scales_for_bpm_range(
    fps: float,
    bpm_low: float,
    bpm_high: float,
    n_scales: int,
) -> np.ndarray:
    """
    Return ``n_scales`` logarithmically-spaced CWT scales that cover the
    heart-rate band [bpm_low, bpm_high] BPM at a given sample-rate.

    Morlet scale ↔ frequency relation:  f = f_c / (s × dt)  →  s = f_c / (f × dt)
    """
    dt = 1.0 / fps
    f_low  = bpm_low  / 60.0   # Hz
    f_high = bpm_high / 60.0   # Hz
    s_min  = _MORLET_FC / (f_high * dt)
    s_max  = _MORLET_FC / (f_low  * dt)
    scales = np.geomspace(s_min, s_max, n_scales).astype(np.float32)
    return scales


def _scale_to_bpm(scale: float, fps: float) -> float:
    """Convert a Morlet CWT scale to BPM."""
    dt = 1.0 / fps
    return (_MORLET_FC / (scale * dt)) * 60.0


# Physiological BPM band edges (matches CPU SignalProcessorWavelet n_bands=6)
_BAND_EDGES_BPM = np.array([45, 60, 90, 120, 150, 180, 240], dtype=np.float64)


def _compute_bands(
    energy: np.ndarray,
    scales: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, int, float]:
    """
    Group per-scale energy into 6 physiological BPM bands.

    Returns
    -------
    band_powers : ndarray shape (6,)
    dominant_band : int
    confidence : float  – dominant_band_power / total_power
    """
    bpms = np.array([_scale_to_bpm(float(s), fps) for s in scales])
    n_bands = len(_BAND_EDGES_BPM) - 1
    band_powers = np.zeros(n_bands, dtype=np.float64)

    for i in range(n_bands):
        mask = (bpms >= _BAND_EDGES_BPM[i]) & (bpms < _BAND_EDGES_BPM[i + 1])
        if mask.any():
            band_powers[i] = energy[mask].sum()

    total = band_powers.sum()
    if total == 0:
        return band_powers, 0, 0.0

    dominant = int(np.argmax(band_powers))
    return band_powers, dominant, float(band_powers[dominant] / total)


# ---------------------------------------------------------------------------
# NumPy (CPU) fallback CWT
# ---------------------------------------------------------------------------

def _numpy_cwt_energy(
    signal: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Morlet CWT band-energy via np.convolve (CPU fallback).

    Replaces the original O(N^2 * n_scales) pure-Python double-loop with an
    O(N * n_scales) pass using NumPy convolution – roughly 100-300x faster
    for typical PPG buffer sizes (N ~= 300-400, n_scales ~= 32).

    Returns an array of shape (n_scales,) with summed power per scale.
    """
    sig = signal.astype(np.float64)
    n = len(sig)
    energy = np.zeros(len(scales), dtype=np.float64)

    for i, s in enumerate(scales.astype(np.float64)):
        # Build a finite Morlet wavelet kernel centred at 0 for this scale.
        # Truncate at ±4σ to keep the kernel compact.
        half = min(int(4.0 * s), n)
        t = np.arange(-half, half + 1, dtype=np.float64)
        dt = t / s
        # Standard Morlet norm: (π·s)^(-1/4) / √s so energy is scale-invariant
        norm = (math.pi * s) ** (-0.25) / math.sqrt(s)
        env = norm * np.exp(-0.5 * dt ** 2)
        kernel_re = env * np.cos(_MORLET_OMEGA0 * dt)
        kernel_im = env * np.sin(_MORLET_OMEGA0 * dt)

        # Correlate signal with wavelet (flip kernel = convolution → correlation)
        conv_re = np.convolve(sig, kernel_re[::-1], mode='same')
        conv_im = np.convolve(sig, kernel_im[::-1], mode='same')
        energy[i] = (conv_re ** 2 + conv_im ** 2).sum()

    return energy


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class WaveletProcessorGPU:
    """
    GPU-accelerated PPG analyser using OpenCL parallel Morlet CWT.

    Parameters
    ----------
    fps:
        Capture frame rate (must match the actual camera FPS for correct BPM).
    window_seconds:
        Rolling analysis window length in seconds.
    bpm_low / bpm_high:
        Heart-rate search band in BPM.
    hardware_profile:
        Pre-detected hardware profile.  When *None*, :func:`detect_hardware`
        is called automatically.
    force_cpu_fallback:
        Skip GPU and use the NumPy CPU fallback regardless of hardware.
    min_samples:
        Minimum buffer fill (samples) before a BPM estimate is emitted.
    """

    def __init__(
        self,
        fps: float = 30.0,
        window_seconds: float = 12.0,
        bpm_low: float = 45.0,
        bpm_high: float = 240.0,
        hardware_profile: HardwareProfile | None = None,
        force_cpu_fallback: bool = False,
        min_samples: int | None = None,
    ) -> None:
        self.fps            = fps
        self.window_seconds = window_seconds
        self.bpm_low        = bpm_low
        self.bpm_high       = bpm_high
        self._force_cpu     = force_cpu_fallback

        # Detect hardware if not supplied
        if hardware_profile is None:
            hardware_profile = detect_hardware(force_cpu_fallback=force_cpu_fallback)
        self.hw = hardware_profile
        logger.info("WaveletProcessorGPU hardware profile:\n%s", self.hw.summary())

        # Rolling signal buffer
        maxlen = min(
            int(fps * window_seconds),
            self.hw.signal_buffer_limit,
        )
        self._buffer: Deque[float] = deque(maxlen=maxlen)

        self.min_samples: int = (
            min_samples if min_samples is not None else int(2 * fps)
        )

        # CWT scales
        n_scales = self.hw.max_wavelet_scales
        self._scales = _scales_for_bpm_range(fps, bpm_low, bpm_high, n_scales)
        logger.info(
            "CWT scales: n=%d  s_min=%.2f (%.0f BPM)  s_max=%.2f (%.0f BPM)",
            n_scales,
            self._scales[0],  _scale_to_bpm(float(self._scales[0]), fps),
            self._scales[-1], _scale_to_bpm(float(self._scales[-1]), fps),
        )

        # OpenCL state (populated in open())
        self._cl_ctx:   "pyopencl.Context  | None" = None
        self._cl_queue: "pyopencl.CommandQueue | None" = None
        self._cl_prog:  "pyopencl.Program   | None" = None
        self._use_tiled: bool = (self.hw.board == BoardType.RPI_ZERO_2W)

        # Persistent device buffers (reallocated if signal length changes)
        self._d_signal:     "pyopencl.Buffer | None" = None
        self._d_scales:     "pyopencl.Buffer | None" = None
        self._d_power:      "pyopencl.Buffer | None" = None
        self._d_energy:     "pyopencl.Buffer | None" = None
        self._last_sig_len: int = 0

        # Last results
        self._last_bpm:          float = 0.0
        self._last_confidence:   float = 0.0
        self._last_energy:       np.ndarray = np.array([])
        self._last_band_powers:  np.ndarray = np.zeros(len(_BAND_EDGES_BPM) - 1)
        self._last_dominant_band: int = 0

        # Temporal smoothing for BPM stability
        self._bpm_history: Deque[Tuple[float, float]] = deque(maxlen=5)  # (bpm, confidence)
        self._smoothed_bpm: float = 0.0

        # Open OpenCL context
        if not force_cpu_fallback:
            self._opencl_available = self.open()
        else:
            self._opencl_available = False
            logger.info("CPU fallback mode: OpenCL disabled.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Initialise the OpenCL context, queue, and compile kernels.

        Returns *True* on success, *False* if OpenCL is unavailable.
        """
        try:
            import pyopencl as cl
        except ImportError:
            logger.warning("pyopencl not installed.  Using CPU fallback.")
            return False

        dev_info = self.hw.opencl_device
        if dev_info is None:
            logger.warning("No OpenCL device found.  Using CPU fallback.")
            return False

        try:
            platforms = cl.get_platforms()
            platform  = platforms[dev_info.platform_index]
            devices   = platform.get_devices()
            device    = devices[dev_info.device_index]

            self._cl_ctx   = cl.Context([device])
            self._cl_queue = cl.CommandQueue(
                self._cl_ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE,
            )

            # Compile all kernels
            self._cl_prog = cl.Program(self._cl_ctx, ALL_KERNELS_SOURCE).build(
                options="-cl-fast-relaxed-math -cl-mad-enable"
            )

            # Upload scales to device (constant across frames)
            import pyopencl as cl
            mf = cl.mem_flags
            self._d_scales = cl.Buffer(
                self._cl_ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=self._scales,
            )

            logger.info(
                "OpenCL context created on device: %s (tiled=%s)",
                dev_info.device_name,
                self._use_tiled,
            )
            return True

        except Exception as exc:                            # noqa: BLE001
            logger.error("OpenCL init failed: %s.  Falling back to CPU.", exc)
            self._cl_ctx = self._cl_queue = self._cl_prog = None
            return False

    def close(self) -> None:
        """Release all OpenCL resources."""
        for attr in ("_d_signal", "_d_scales", "_d_power", "_d_energy"):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:                          # noqa: BLE001
                    pass
            setattr(self, attr, None)
        self._cl_queue = None
        self._cl_ctx   = None
        self._cl_prog  = None
        logger.info("WaveletProcessorGPU closed.")

    def __enter__(self) -> "WaveletProcessorGPU":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public frame interface
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Extract mean green channel from *frame* and push to rolling buffer.

        Parameters
        ----------
        frame:
            BGR image array (H × W × 3, uint8).
        """
        green_mean = float(np.mean(frame[:, :, 1]))
        self._buffer.append(green_mean)

    def compute_bpm(self) -> Tuple[float, float]:
        """
        Compute heart rate from the current buffer using GPU-parallel CWT.

        Returns
        -------
        (bpm, confidence)
            Confidence is peak-band energy / total energy (0 – 1).
            Returns ``(0.0, 0.0)`` when there is insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return 0.0, 0.0

        signal = np.array(self._buffer, dtype=np.float32)

        if self._opencl_available and self._cl_ctx is not None:
            energy = self._compute_energy_gpu(signal)
        else:
            # NumPy fallback – linear detrend only, no std normalisation
            sig64 = signal.astype(np.float64)
            t = np.arange(len(sig64), dtype=np.float64)
            p = np.polyfit(t, sig64, 1)
            sig64 -= np.polyval(p, t)
            energy = _numpy_cwt_energy(sig64, self._scales)

        if energy.sum() == 0:
            return 0.0, 0.0

        self._last_energy = energy

        # ── Step A: group scales into 6 physiological bands (same as CPU) ──
        band_powers, dominant_band, confidence = _compute_bands(
            energy, self._scales, self.fps
        )
        self._last_band_powers   = band_powers
        self._last_dominant_band = dominant_band

        # ── Step B: find peak scale *within* the dominant band ──
        bpms = np.array([_scale_to_bpm(float(s), self.fps) for s in self._scales])
        band_mask = (
            (bpms >= _BAND_EDGES_BPM[dominant_band]) &
            (bpms <  _BAND_EDGES_BPM[dominant_band + 1])
        )
        if not band_mask.any():
            band_mask = np.ones(len(self._scales), dtype=bool)  # fallback: all scales

        band_energy = energy.copy()
        band_energy[~band_mask] = 0.0
        peak_idx   = int(np.argmax(band_energy))
        peak_scale = float(self._scales[peak_idx])

        # ── Step C: parabolic interpolation for sub-bin precision ──
        local_indices = np.where(band_mask)[0]
        if len(local_indices) >= 3:
            local_pos = np.searchsorted(local_indices, peak_idx)
            if 0 < local_pos < len(local_indices) - 1:
                i_prev = local_indices[local_pos - 1]
                i_curr = local_indices[local_pos]
                i_next = local_indices[local_pos + 1]
                alpha = float(energy[i_prev])
                beta  = float(energy[i_curr])
                gamma = float(energy[i_next])
                denom = alpha - 2 * beta + gamma
                if abs(denom) > 1e-9:
                    p_off = 0.5 * (alpha - gamma) / denom
                    if abs(p_off) < 1.0:
                        s_lo = float(self._scales[i_curr])
                        s_hi = float(self._scales[i_next if p_off > 0 else i_prev])
                        peak_scale = s_lo * (s_hi / s_lo) ** abs(p_off)

        bpm = _scale_to_bpm(peak_scale, self.fps)

        # ── Step D: confidence penalties (same as CPU) ──
        # 1. Near lower bound → likely DC drift
        if bpm < self.bpm_low + 5:
            confidence *= 0.5
        # 2. Near upper bound → likely harmonics
        if bpm > self.bpm_high - 10:
            confidence *= 0.6
        # 3. Lowest band with near-total dominance → no real signal yet
        if dominant_band == 0 and confidence > 0.8:
            confidence *= 0.6

        # Apply temporal smoothing for stability
        self._bpm_history.append((bpm, confidence))
        
        # Use weighted median filter to reject outliers
        if len(self._bpm_history) >= 3:
            # Extract BPMs and confidences
            bpms = np.array([b for b, c in self._bpm_history])
            confs = np.array([c for b, c in self._bpm_history])
            
            # Only smooth if current confidence is reasonable
            if confidence > 0.15:
                # Weighted median: give more weight to high-confidence readings
                weights = confs / confs.sum() if confs.sum() > 0 else np.ones_like(confs) / len(confs)
                
                # Calculate weighted median
                sorted_indices = np.argsort(bpms)
                sorted_bpms = bpms[sorted_indices]
                sorted_weights = weights[sorted_indices]
                cumsum = np.cumsum(sorted_weights)
                median_idx = np.searchsorted(cumsum, 0.5)
                
                smoothed_bpm = sorted_bpms[median_idx]
                
                # Exponential moving average for final smoothing (alpha=0.4)
                if self._smoothed_bpm > 0:
                    smoothed_bpm = 0.4 * smoothed_bpm + 0.6 * self._smoothed_bpm
                
                self._smoothed_bpm = smoothed_bpm
                bpm = smoothed_bpm

        self._last_bpm        = bpm
        self._last_confidence = confidence
        return bpm, confidence

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def buffer_fill_ratio(self) -> float:
        return len(self._buffer) / self._buffer.maxlen

    @property
    def last_bpm(self) -> float:
        return self._last_bpm

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def band_energies(self) -> np.ndarray:
        """Latest per-scale energy vector (for plotting / debugging)."""
        return self._last_energy.copy() if len(self._last_energy) else np.array([])

    @property
    def band_powers(self) -> np.ndarray:
        """Per-band summed energy (6 physiological bands) – mirrors CPU API."""
        return self._last_band_powers.copy()

    @property
    def dominant_band(self) -> int:
        """Index of the dominant frequency band (0–5)."""
        return self._last_dominant_band

    @property
    def band_edges(self) -> np.ndarray:
        """Band edge frequencies in Hz (mirrors CPU API)."""
        return _BAND_EDGES_BPM / 60.0

    @property
    def band_bpms(self) -> np.ndarray:
        """BPM value corresponding to each scale (same length as band_energies)."""
        return np.array(
            [_scale_to_bpm(float(s), self.fps) for s in self._scales],
            dtype=np.float32,
        )

    def get_filtered_signal(self) -> np.ndarray:
        """
        Return the current (linearly detrended) signal buffer for plotting.
        Returns empty array if insufficient data.
        """
        if len(self._buffer) < self.min_samples:
            return np.array([])
        sig = np.array(self._buffer, dtype=np.float64)
        t = np.arange(len(sig), dtype=np.float64)
        p = np.polyfit(t, sig, 1)
        sig -= np.polyval(p, t)
        return sig

    def get_fft_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the wavelet power spectrum as (freqs_bpm, power) arrays.

        Reuses the energy vector computed during the last :meth:`compute_bpm`
        call, so there is zero extra computation cost.

        Returns
        -------
        freqs_bpm : ndarray
            Centre frequency of each wavelet scale in BPM, restricted to
            [bpm_low, bpm_high].
        power : ndarray
            Mean squared wavelet energy at each scale (same length).
        """
        if len(self._last_energy) == 0:
            return np.array([]), np.array([])

        bpms = np.array(
            [_scale_to_bpm(float(s), self.fps) for s in self._scales],
            dtype=np.float64,
        )
        band_mask = (bpms >= self.bpm_low) & (bpms <= self.bpm_high)
        freqs = bpms[band_mask]
        power = self._last_energy.astype(np.float64)[band_mask]
        return freqs, power

    def reset(self) -> None:
        """Clear rolling buffer and last results."""
        self._buffer.clear()
        self._last_bpm           = 0.0
        self._last_confidence    = 0.0
        self._last_energy        = np.array([])
        self._last_band_powers   = np.zeros(len(_BAND_EDGES_BPM) - 1)
        self._last_dominant_band = 0
        self._bpm_history.clear()
        self._smoothed_bpm       = 0.0
        logger.debug("WaveletProcessorGPU buffer reset.")

    # ------------------------------------------------------------------
    # Private – GPU execution
    # ------------------------------------------------------------------

    def _compute_energy_gpu(self, signal: np.ndarray) -> np.ndarray:
        """
        Run the full GPU pipeline and return the per-scale energy vector.

        Parameters
        ----------
        signal : np.ndarray
            Float32 raw PPG signal of length N.
        """
        import pyopencl as cl
        mf = cl.mem_flags

        n_scales   = len(self._scales)
        sig_len    = len(signal)
        wg_size    = self.hw.preferred_work_group
        max_half   = min(int(4.0 * float(self._scales[-1])), sig_len // 2)

        # -----------------------------------------------------------------
        # Step 1 – linear detrend on host (removes DC + slow drift)
        # -----------------------------------------------------------------
        sig64 = signal.astype(np.float64)
        # Fit and subtract a straight line to remove slow drift / DC
        t = np.arange(len(sig64), dtype=np.float64)
        p = np.polyfit(t, sig64, 1)
        sig64 -= np.polyval(p, t)
        signal_norm = sig64.astype(np.float32)

        # -----------------------------------------------------------------
        # Step 2 – (re)allocate device buffers if signal length changed
        # -----------------------------------------------------------------
        if sig_len != self._last_sig_len:
            self._reallocate_buffers(sig_len, n_scales)
            self._last_sig_len = sig_len

        # -----------------------------------------------------------------
        # Step 3 – upload signal
        # -----------------------------------------------------------------
        cl.enqueue_copy(self._cl_queue, self._d_signal, signal_norm)

        # -----------------------------------------------------------------
        # Step 4 – CWT kernel
        # -----------------------------------------------------------------
        if self._use_tiled:
            # Tiled kernel for RPi Zero 2W (small local mem)
            tile_wg    = 64
            global_tau = int(math.ceil(sig_len / tile_wg)) * tile_wg
            evt = self._cl_prog.cwt_morlet_tiled(
                self._cl_queue,
                (n_scales, global_tau),
                (1, tile_wg),
                self._d_signal,
                self._d_power,
                self._d_scales,
                np.int32(sig_len),
                np.int32(n_scales),
                np.int32(max_half),
                cl.LocalMemory(tile_wg * 4),    # float per item
            )
        else:
            # Standard kernel – work-group shape along tau axis
            global_tau = int(math.ceil(sig_len / wg_size)) * wg_size
            evt = self._cl_prog.cwt_morlet(
                self._cl_queue,
                (n_scales, global_tau),
                (1, wg_size),
                self._d_signal,
                self._d_power,
                self._d_scales,
                np.int32(sig_len),
                np.int32(n_scales),
                np.int32(max_half),
            )
        evt.wait()

        # -----------------------------------------------------------------
        # Step 5 – band energy summation
        # -----------------------------------------------------------------
        global_scales = int(math.ceil(n_scales / max(wg_size, 1))) * max(wg_size, 1)
        evt2 = self._cl_prog.band_energy(
            self._cl_queue,
            (global_scales,),
            (min(wg_size, n_scales),),
            self._d_power,
            self._d_energy,
            np.int32(sig_len),
            np.int32(n_scales),
        )
        evt2.wait()

        # -----------------------------------------------------------------
        # Step 6 – download energy
        # -----------------------------------------------------------------
        energy_host = np.empty(n_scales, dtype=np.float32)
        cl.enqueue_copy(self._cl_queue, energy_host, self._d_energy)
        self._cl_queue.finish()

        return energy_host.astype(np.float64)

    def _reallocate_buffers(self, sig_len: int, n_scales: int) -> None:
        """(Re)allocate persistent GPU device buffers for a new signal length."""
        import pyopencl as cl
        mf = cl.mem_flags

        # Release old buffers
        for attr in ("_d_signal", "_d_power", "_d_energy"):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:                          # noqa: BLE001
                    pass

        self._d_signal = cl.Buffer(
            self._cl_ctx,
            mf.READ_WRITE,
            size=sig_len * 4,                             # float32
        )
        self._d_power = cl.Buffer(
            self._cl_ctx,
            mf.READ_WRITE,
            size=n_scales * sig_len * 4,
        )
        self._d_energy = cl.Buffer(
            self._cl_ctx,
            mf.READ_WRITE,
            size=n_scales * 4,
        )
        logger.debug(
            "GPU buffers reallocated: sig_len=%d  n_scales=%d  "
            "power_buf=%.2f MB",
            sig_len, n_scales, (n_scales * sig_len * 4) / 1e6,
        )
