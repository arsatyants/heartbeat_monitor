# Wavelet-Based Signal Processor

## Recent Optimizations (Latest)

**Major performance and accuracy improvements applied to GPU implementation:**
- **Reduced scale count**: 16-24 scales (was 32-48) for 2× faster computation and better frequency resolution  
- **Parabolic interpolation**: Sub-bin precision for <1% BPM error (was ~5-10% quantization error)
- **Confidence penalties**: Realistic confidence scoring for edge cases
- **Temporal smoothing**: Weighted median filter + EMA reduces jitter to <3 BPM std dev
- **Extended wavelet support**: ±4σ (was ±3σ) for better accuracy matching CPU implementation

See [GPU_WAVELET_OPTIMIZATIONS.md](GPU_WAVELET_OPTIMIZATIONS.md) for detailed analysis and benchmarks.

## Overview

This is an alternative implementation of the heartbeat detection algorithm using **Continuous Wavelet Transform (CWT)** instead of Butterworth filtering and FFT. 

## Key Differences from FFT Implementation

| Feature | FFT Implementation | Wavelet Implementation |
|---------|-------------------|------------------------|
| **Filtering** | Butterworth bandpass filter | Continuous Wavelet Transform |
| **Frequency Analysis** | Single FFT pass | Multi-band wavelet decomposition |
| **Band Detection** | Single frequency band | 6 configurable frequency bands |
| **Peak Finding** | FFT peak with interpolation | Power-based band selection + refinement |
| **Robustness** | Good for clean signals | Better for noisy/multi-component signals |

## Algorithm

### 1. Multi-Band Wavelet Decomposition
The heart rate range (45-240 BPM or 0.75-4 Hz) is divided into **6 frequency bands**:
- Band 0: 45-77.5 BPM (0.75-1.29 Hz)
- Band 1: 77.5-110 BPM (1.29-1.83 Hz)
- Band 2: 110-142.5 BPM (1.83-2.38 Hz)
- Band 3: 142.5-175 BPM (2.38-2.92 Hz)
- Band 4: 175-207.5 BPM (2.92-3.46 Hz)
- Band 5: 207.5-240 BPM (3.46-4.00 Hz)

### 2. Power Calculation
For each band, the total wavelet power is calculated:
```
P_i = Σ |W(scale, time)|² for scales in band_i
```

### 3. Dominant Band Selection
The band with highest power is selected as the dominant heart rate band:
```
dominant_band = argmax(P_0, P_1, ..., P_5)
```

### 4. Peak Frequency Refinement
Within the dominant band:
1. Average wavelet coefficients across time
2. Find peak in the power spectrum
3. Apply parabolic interpolation for sub-bin precision

### 5. Confidence Score
```
confidence = P_dominant / Σ P_i
```

## Advantages

✅ **Multi-Scale Analysis**: Analyzes signal at multiple time-frequency resolutions simultaneously

✅ **Noise Robustness**: Better handles motion artifacts and ambient light variations

✅ **Frequency Localization**: Excellent time-frequency localization properties

✅ **Adaptive**: Automatically identifies the most prominent heart rate band

✅ **Non-stationary Signals**: Better suited for heart rate variability and transient changes

## Disadvantages

⚠️ **Computational Cost**: More CPU-intensive than FFT (requires multiple wavelet transforms)

⚠️ **Memory Usage**: Stores 2D time-scale representation (vs 1D frequency spectrum)

⚠️ **Parameter Tuning**: Additional parameters (wavelet type, number of bands, scale range)

⚠️ **Startup Time**: May take slightly longer to produce first BPM estimate

## Usage

### Command Line
```bash
python main_wavelet.py [OPTIONS]

Options:
  --bands INT       Number of frequency bands (default: 6)
  --wavelet STR     Wavelet type (default: morl)
  --window FLOAT    Analysis window in seconds (default: 12)
  --strategy STR    CWT backend: 'pywt' (default) or 'numpy'
```

### Programmatic
```python
from heartbeat_monitor.signal_processor_wavelet import SignalProcessorWavelet

processor = SignalProcessorWavelet(
    fps=30.0,
    window_seconds=12.0,
    n_bands=6,              # Number of frequency bands
    wavelet='morl',         # Morlet wavelet (good for oscillatory signals)
    strategy='pywt',        # 'pywt' (default) or 'numpy'
)

# Use exactly like the standard processor
processor.push_frame(frame)
bpm, confidence = processor.compute_bpm()
spo2 = processor.compute_spo2()  # only available in 'pywt' strategy
```

## CWT Strategy Comparison

`SignalProcessorWavelet` supports two independent CWT backends, selectable via
the `strategy` constructor argument or `--strategy` on the command line.

### `pywt` (default)

Uses **PyWavelets** (`pywt.cwt`) to compute the full 2-D coefficient matrix.

| Property | Detail |
|----------|--------|
| Wavelet  | Any PyWavelets family (`morl`, `mexh`, `gaus1`, … – controlled by `--wavelet`) |
| Scales   | 64, logarithmically spaced |
| Detrend  | Linear (polynomial order 1) |
| Smoothing | None – each call returns the raw CWT result |
| SpO₂     | ✅ Yes (second CWT pass on the red channel) |
| Filtered waveform | CWT bandpass reconstruction (third CWT pass) |
| FFT panel | Mean power per scale – requires a fresh CWT run |
| Compute cost | ~11 ms per call on RPi 5 (360-sample buffer) |

**Best for:** highest frequency resolution, SpO₂ readout, debugging with
different wavelet families.

---

### `numpy`

Uses a **vectorised NumPy Morlet convolution** – the same algorithm as the
GPU processor's CPU fallback (`_numpy_cwt_energy`).  Adds temporal smoothing
identical to the GPU path.

| Property | Detail |
|----------|--------|
| Wavelet  | Fixed complex Morlet (ω₀ = 6, f_c ≈ 0.955 Hz) – `--wavelet` ignored |
| Scales   | 64, geometrically spaced via `np.geomspace` |
| Detrend  | Linear (polynomial order 1) |
| Smoothing | **Weighted median** over last 5 BPM readings (weight = confidence) → **EMA** (α = 0.4, 60 % inertia) |
| SpO₂     | ❌ Not computed (returns `0.0`) |
| Filtered waveform | Linearly detrended raw signal (zero extra cost) |
| FFT panel | Reuses energy vector from `compute_bpm()` – zero extra cost |
| Compute cost | ~13 ms per call on RPi 5 (360-sample buffer) |

**Best for:** stable live display, headless logging with low jitter, hardware
where pywt is unavailable or slow.

---

### Side-by-side benchmark (72 BPM synthetic signal, 360 samples)

```
True BPM : 72.0

strategy='pywt'  → BPM=72.3  error=0.3  confidence=0.842  time=10.9 ms
strategy='numpy' → BPM=71.7  error=0.3  confidence=0.897  time=12.5 ms
```

### Temporal stability (72 BPM live stream, updates every 10 frames)

```
frame    pywt BPM    numpy BPM
------   ---------   ---------
   60       70.9        71.2    ← settling
   70       72.7        72.1
   80       73.8        72.1    pywt swings ±3 BPM while settling
   90       72.3        72.1    numpy already locked
  ...        ..          ..
  150       71.9        71.5    pywt ±0.5 BPM  /  numpy ±0.1 BPM
  240       72.2        71.6    ← phase jump (finger micro-shift)
  250       72.2        71.6    both absorb the jump cleanly
```

The `numpy` strategy's two-layer filter (weighted median then EMA) is the
reason the digit on screen barely moves once the signal is established –
exactly the behaviour of the GPU version.

---

### When to use which

| Situation | Recommended strategy |
|-----------|---------------------|
| Need SpO₂ overlay | `pywt` |
| Testing different wavelet families | `pywt` |
| Stable live BPM display on screen | `numpy` |
| Headless logging / alerting | `numpy` |
| Raspberry Pi with limited RAM | `numpy` (no 2-D coefficient matrix) |
| GPU mode not available, matching GPU behaviour | `numpy` |

The implementation supports various wavelet families from PyWavelets:

| Wavelet | Description | Best For |
|---------|-------------|----------|
| `morl` | Morlet (default) | Oscillatory signals, heart rate |
| `mexh` | Mexican Hat | Sharp transients |
| `gaus1` | Gaussian 1st derivative | Smooth signals |
| `cgau1` | Complex Gaussian | Phase information |

## Wavelet Types

> **Note:** the `--wavelet` option is only respected by the `pywt` strategy.
> The `numpy` strategy always uses the standard complex Morlet (ω₀ = 6).

## Performance Characteristics

Tested on Raspberry Pi 5 with IMX500 camera:

- **Processing Time**: ~15-25ms per frame (vs ~8-12ms for FFT)
- **Memory**: ~8-12 MB additional (wavelet coefficients)
- **Accuracy**: Comparable to FFT for clean signals, better in noisy conditions
- **Startup**: 2-3 seconds for stable readings (same as FFT)

## Testing

Run the wavelet-specific tests:
```bash
pytest tests/test_wavelet_processor.py -v
```

All standard tests from `test_heartbeat.py` also apply.

## Visualizations

The wavelet implementation includes additional visualization:
- **Band Power Bars**: Shows power distribution across 6 frequency bands
- **Dominant Band**: Highlighted in yellow
- **Wavelet Spectrum**: Optional frequency-time representation

## When to Use Wavelet vs FFT

### Use Wavelet When:
- Signal has multiple frequency components
- Dealing with motion artifacts or noise
- Heart rate varies significantly during measurement
- Need better time-frequency localization
- Working in challenging lighting conditions

### Use FFT When:
- Clean, stationary signals
- Computational resources are limited
- Real-time processing on low-power devices
- Simpler implementation is preferred
- Signal quality is high

## Configuration Tips

### For Stable Conditions
```python
SignalProcessorWavelet(
    n_bands=4,              # Fewer bands = faster
    window_seconds=15.0,    # Longer window = more stable
)
```

### For Dynamic Conditions
```python
SignalProcessorWavelet(
    n_bands=8,              # More bands = better resolution
    window_seconds=10.0,    # Shorter window = faster response
)
```

## Implementation Files

- `signal_processor_wavelet.py` – Core wavelet processor (both strategies)
- `main_wavelet.py` – Entry point (pywt strategy, default)
- `run_wavelet.sh` – Launcher for `pywt` strategy
- `run_wavelet_numpy.sh` – Launcher for `numpy` strategy (temporal smoothing)
- `test_wavelet_processor.py` – Unit tests

## References

1. Addison P.S., "Wavelet transforms and the ECG: a review." Physiological Measurement, 2005.
2. Peng et al., "Extracting heart rate variability using wavelet transform." IEEE EMBS, 2006.
3. Torrence & Compo, "A Practical Guide to Wavelet Analysis." BAMS, 1998.

## Future Enhancements

- [ ] Wavelet packet decomposition for even finer frequency resolution
- [ ] Adaptive band selection based on signal quality
- [ ] GPU acceleration for real-time performance
- [ ] Time-frequency visualization in UI
- [ ] Automatic wavelet selection based on signal characteristics
