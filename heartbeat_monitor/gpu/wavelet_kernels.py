"""
OpenCL kernel source strings for GPU-accelerated CWT-based PPG processing.

All kernels operate on 1-D float arrays.  They are compiled at runtime by
:mod:`heartbeat_monitor.gpu.wavelet_processor_gpu`.

Kernel overview
---------------

``cwt_morlet``
    2-D kernel: global_size = (n_scales, signal_len).
    Each work item computes ONE (scale_idx, time_idx) CWT coefficient using a
    discretised real Morlet wavelet convolution.  Output is the instantaneous
    power: ``power[scale_idx * signal_len + time_idx] = Re² + Im²``.

``band_energy``
    1-D kernel: global_size = (n_scales,).
    Each work item accumulates the summed power over all time positions for one
    scale, writing a single ``energy[scale_idx]`` value.

``find_peak_bpm``
    Single work item (global_size = (1,)).
    Reads ``energy[0..n_scales-1]`` and the scale→Hz lookup table, and writes
    the dominant BPM and confidence score.

``detrend_normalize``
    1-D kernel: global_size = (signal_len,).
    Subtracts the pre-computed mean and divides by the standard deviation in
    parallel (mean and std are passed as scalar kernel arguments).

Design notes
------------
* All kernels use ``float`` (fp32).  The processor can optionally reinterpret
  data when the device profile requests fp16 by quantising on the Python side
  before upload (OpenCL fp16 support varies widely on VideoCore).
* The Morlet wavelet support is truncated to ±3σ scaled by the current scale
  to keep convolution tractable on constrained GPUs.
* A work-group local-memory tile is used in ``cwt_morlet`` to cache the signal
  tile for each time window, avoiding redundant global reads.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helper: shared preamble
# ---------------------------------------------------------------------------

_PREAMBLE = r"""
/* ---- floating-point constants ---- */
#define PI_F   3.14159265358979323846f
#define TWO_PI (2.0f * PI_F)

/* Morlet wavelet parameter (omega_0).
   6.0 is the standard choice that satisfies the admissibility condition. */
#define OMEGA0  6.0f
"""

# ---------------------------------------------------------------------------
# Kernel 1 – detrend + normalize
# ---------------------------------------------------------------------------

KERNEL_DETREND = (
    _PREAMBLE
    + r"""
/**
 * Subtract mean and divide by std-dev, in parallel.
 *
 * @param signal   Input/output float array of length signal_len.
 * @param mean     Pre-computed mean of the input signal.
 * @param inv_std  1.0 / std_dev  (compute on host to avoid division in kernel).
 * @param len      Number of elements.
 */
__kernel void detrend_normalize(
    __global float* signal,
    const    float  mean,
    const    float  inv_std,
    const    int    len)
{
    int t = get_global_id(0);
    if (t >= len) return;
    signal[t] = (signal[t] - mean) * inv_std;
}
"""
)

# ---------------------------------------------------------------------------
# Kernel 2 – CWT with complex Morlet wavelet (outputs instantaneous power)
# ---------------------------------------------------------------------------

KERNEL_CWT_MORLET = (
    _PREAMBLE
    + r"""
/**
 * Compute the instantaneous power of the Complex Morlet CWT for all
 * (scale, time) pairs in parallel.
 *
 * Global work size: (n_scales, signal_len).
 * One work item handles ONE (scale_idx, time_idx) pair.
 *
 * The discretised Morlet wavelet at scale s and translation tau:
 *   psi(t; s, tau) = (pi*s)^(-0.25) * exp(-((t-tau)^2)/(2*s^2))
 *                    * exp(i * OMEGA0 * (t-tau) / s)
 *
 * Real CWT convolution:
 *   W_re = sum_t signal[t] * psi_re(t; s, tau)
 *   W_im = sum_t signal[t] * psi_im(t; s, tau)
 *   power = W_re^2 + W_im^2
 *
 * To keep work per item bounded, the summation window is clamped to
 * ±3*s samples around the central time index (compact support approximation).
 *
 * @param signal      Input signal (float, length signal_len) – detrended.
 * @param power_out   Output power array (float, n_scales × signal_len).
 * @param scales      Array of scale values s[0..n_scales-1] (float).
 * @param signal_len  Length of signal.
 * @param n_scales    Number of scales.
 * @param max_half_support  Maximum half-window (samples) to cap convolution.
 */
__kernel void cwt_morlet(
    __global const float*  signal,
    __global       float*  power_out,
    __global const float*  scales,
    const          int     signal_len,
    const          int     n_scales,
    const          int     max_half_support)
{
    int s_idx = get_global_id(0);   /* scale index  */
    int tau   = get_global_id(1);   /* time index   */

    if (s_idx >= n_scales || tau >= signal_len) return;

    float s = scales[s_idx];

    /* Normalisation factor: (pi * s)^{-0.25} / sqrt(s)  – L² (energy) normalisation.
       Matches pywt and the NumPy fallback.  Without the /sqrt(s) term, larger scales
       (lower frequencies) accumulate artificially more energy, biasing confidence
       toward low-frequency bands. */
    float norm = native_powr(PI_F * s, -0.25f) / native_sqrt(s);

    /* Support: ±4s samples, capped at max_half_support (matches CPU fallback) */
    int half = (int)(4.0f * s);
    if (half > max_half_support) half = max_half_support;

    int t_start = tau - half;
    int t_end   = tau + half;

    float w_re = 0.0f;
    float w_im = 0.0f;

    for (int t = t_start; t <= t_end; t++) {
        /* Boundary: zero-pad */
        float x = (t >= 0 && t < signal_len) ? signal[t] : 0.0f;

        float dt    = (float)(t - tau);
        float dt_s  = dt / s;

        /* Gaussian envelope */
        float envelope = norm * native_exp(-0.5f * dt_s * dt_s);

        /* Oscillatory part */
        float phase = OMEGA0 * dt_s;
        float c, sv;
        c  = native_cos(phase);
        sv = native_sin(phase);

        w_re += x * envelope * c;
        w_im += x * envelope * sv;
    }

    power_out[s_idx * signal_len + tau] = w_re * w_re + w_im * w_im;
}
"""
)

# ---------------------------------------------------------------------------
# Kernel 3 – per-band energy accumulation
# ---------------------------------------------------------------------------

KERNEL_BAND_ENERGY = (
    _PREAMBLE
    + r"""
/**
 * Sum power over all time positions for each scale band.
 *
 * Global work size: (n_scales,).
 *
 * @param power_in   CWT power array (float, n_scales × signal_len).
 * @param energy_out Per-scale energy output (float, n_scales).
 * @param signal_len Number of time samples.
 * @param n_scales   Number of scales.
 */
__kernel void band_energy(
    __global const float* power_in,
    __global       float* energy_out,
    const          int    signal_len,
    const          int    n_scales)
{
    int s_idx = get_global_id(0);
    if (s_idx >= n_scales) return;

    float acc = 0.0f;
    int   base = s_idx * signal_len;
    for (int t = 0; t < signal_len; t++) {
        acc += power_in[base + t];
    }
    energy_out[s_idx] = acc;
}
"""
)

# ---------------------------------------------------------------------------
# Kernel 4 – local-memory-optimised CWT for devices with small global memory
# (used on RPi Zero 2W; tiles the signal into __local memory)
# ---------------------------------------------------------------------------

KERNEL_CWT_MORLET_TILED = (
    _PREAMBLE
    + r"""
#define TILE_SIZE 64

/**
 * Tiled variant of cwt_morlet for constrained GPUs.
 *
 * Each work-group loads a tile of the signal into local memory, then each
 * work-item (= one tau) computes its CWT coefficient by reading from local.
 * Because the wavelet support may exceed the tile, partial sums from adjacent
 * tiles are accumulated, but this version keeps a simpler structure:
 * it loads a centred window around each tau lazily.
 *
 * Work-group size: (1, TILE_SIZE).
 * Global size:     (n_scales, ceil(signal_len / TILE_SIZE) * TILE_SIZE).
 *
 * Parameters identical to cwt_morlet.
 */
__kernel __attribute__((reqd_work_group_size(1, TILE_SIZE, 1)))
void cwt_morlet_tiled(
    __global const float*  signal,
    __global       float*  power_out,
    __global const float*  scales,
    const          int     signal_len,
    const          int     n_scales,
    const          int     max_half_support,
    __local        float*  local_tile)   /* size = TILE_SIZE floats */
{
    int s_idx   = get_global_id(0);
    int tau     = get_global_id(1);
    int local_t = get_local_id(1);

    float s = (s_idx < n_scales) ? scales[s_idx] : 1.0f;

    int half = (int)(4.0f * s);
    if (half > max_half_support) half = max_half_support;

    /* Co-operative load of a signal tile into local memory */
    int tile_base = tau - local_t;            /* beginning of tile */
    int load_idx  = tile_base + local_t;
    local_tile[local_t] = (load_idx >= 0 && load_idx < signal_len)
                          ? signal[load_idx]
                          : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (s_idx >= n_scales || tau >= signal_len) return;

    float norm  = native_powr(PI_F * s, -0.25f) / native_sqrt(s);  /* L² normalisation */
    float w_re  = 0.0f;
    float w_im  = 0.0f;
    int   t_start = tau - half;
    int   t_end   = tau + half;

    for (int t = t_start; t <= t_end; t++) {
        /* Use local memory when inside tile, global otherwise */
        float x;
        int lt = t - tile_base;
        if (lt >= 0 && lt < TILE_SIZE) {
            x = local_tile[lt];
        } else {
            x = (t >= 0 && t < signal_len) ? signal[t] : 0.0f;
        }

        float dt_s  = (float)(t - tau) / s;
        float envelope = norm * native_exp(-0.5f * dt_s * dt_s);
        float phase    = OMEGA0 * dt_s;
        float c  = native_cos(phase);
        float sv = native_sin(phase);
        w_re += x * envelope * c;
        w_im += x * envelope * sv;
    }

    power_out[s_idx * signal_len + tau] = w_re * w_re + w_im * w_im;
}
"""
)

# ---------------------------------------------------------------------------
# Combined source (used when compiling all kernels at once)
# ---------------------------------------------------------------------------

ALL_KERNELS_SOURCE: str = "\n\n".join([
    KERNEL_DETREND,
    KERNEL_CWT_MORLET,
    KERNEL_CWT_MORLET_TILED,
    KERNEL_BAND_ENERGY,
])
