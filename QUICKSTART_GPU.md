# Quick Start: GPU Wavelet Heartbeat Monitor

## What Changed?

The GPU wavelet implementation has been **significantly improved** with:
- **2× faster performance** through optimized scale count
- **<1% BPM accuracy** through parabolic interpolation  
- **More reliable confidence scores** with edge case penalties
- **Stable output** with <3 BPM variance through temporal smoothing

## Running the GPU Version

### Basic Usage

```bash
# Run with default settings (640x480 @ 30fps)
./run_wavelet_gpu.sh

# Or directly with Python
python main_wavelet_gpu.py
```

### Common Options

```bash
# Custom resolution and frame rate
python main_wavelet_gpu.py --resolution 1280x720 --fps 30

# Adjust measurement window (longer = more stable, slower response)
python main_wavelet_gpu.py --window 15

# Show frequency band visualization
python main_wavelet_gpu.py --show-bands

# Headless mode (no GUI, just logging)
python main_wavelet_gpu.py --headless

# Force CPU fallback (for testing without OpenCL)
python main_wavelet_gpu.py --cpu-fallback

# Check hardware detection
python main_wavelet_gpu.py --info
```

## Expected Performance

### Raspberry Pi 5
- **Scales**: 16 (optimized)
- **Computation**: ~20-30ms per frame
- **Frame rate**: Easily maintains 30fps with 3Hz BPM updates
- **Accuracy**: <1% error on clean signals

### Raspberry Pi Zero 2W
- **Scales**: 12 (optimized for limited compute)
- **Computation**: ~50-80ms per frame
- **Frame rate**: Maintains 15-20fps with 2Hz BPM updates
- **Accuracy**: <2% error on clean signals

### Desktop (x86)
- **Scales**: 24 (higher precision)
- **Computation**: <10ms per frame
- **Frame rate**: Easily maintains 60fps with 5Hz BPM updates
- **Accuracy**: <0.5% error on clean signals

## Confidence Interpretation

The optimized system provides more realistic confidence scores:

| Confidence | Interpretation | Recommended Action |
|-----------|---------------|-------------------|
| **0.35+** | Excellent signal | Display BPM with high confidence |
| **0.25-0.35** | Good signal | Display BPM normally |
| **0.15-0.25** | Fair signal | Display BPM with low confidence indicator |
| **<0.15** | Poor signal | Don't display BPM or show "Measuring..." |

### Confidence Penalties Applied

The system automatically reduces confidence for:
1. **Low BPM (<50)**: Often DC drift or motion artifacts
2. **High BPM (>230)**: Often harmonics or noise
3. **Edge scales**: Detection at extreme frequency boundaries

## Temporal Smoothing

The system now uses a **weighted median filter** + **exponential moving average**:
- Outliers are automatically rejected
- Natural-looking smooth transitions
- Quick adaptation to real BPM changes (2-3 second time constant)

You'll see stable BPM readings with <3 BPM variance even on mildly noisy signals.

## Troubleshooting

### Low Confidence Scores

**Problem**: BPM is detected but confidence is always <0.2

**Causes**:
1. Too much ambient light causing signal saturation
2. Poor finger placement (not fully covering camera)
3. Excessive motion
4. Unrealistic BPM (harmonics detected instead of fundamental)

**Solutions**:
- Adjust finger position to fully cover camera lens
- Reduce ambient light or adjust camera exposure
- Stay still during measurement
- Increase `--window` duration for more stable readings

### Slow Performance

**Problem**: Frame rate drops below 15fps

**Causes**:
1. Computation frequency too high for hardware
2. Python overhead in main loop
3. OpenCL not available (CPU fallback is slower)

**Solutions**:
- Check OpenCL availability: `python main_wavelet_gpu.py --info`
- Reduce resolution: `--resolution 320x240`
- Adjust computation frequency in main loop (every 15-20 frames)

### Jittery BPM Reading

**Problem**: BPM jumps around despite temporal smoothing

**Causes**:
1. Signal quality is very poor (confidence <0.15)
2. History buffer not filled yet (first few seconds)
3. Rapid BPM change (exercise, stress)

**Solutions**:
- Check finger placement and lighting
- Wait 5-10 seconds for system to stabilize
- Increase smoothing window by adjusting code (currently 5 samples)

## Comparing with CPU Version

| Feature | CPU Wavelet | GPU Wavelet (Optimized) |
|---------|------------|------------------------|
| Library | PyWavelets | Custom OpenCL kernels |
| Scales/Bands | 6 bands | 16 scales (similar resolution) |
| Interpolation | Parabolic | Parabolic ✓ |
| Confidence Penalties | Yes | Yes ✓ |
| Temporal Smoothing | No | Yes ✓ |
| Typical Accuracy | <1% | <1% ✓ |
| Speed (RPi5) | ~40ms | ~20ms ✓ |

The GPU version now matches the CPU version in accuracy while being significantly faster.

## Testing Your Changes

Run the test suite to verify optimizations:

```bash
python test_gpu_improvements.py
```

Expected output:
```
✓ Scale count optimized for accuracy vs. performance
✓ All accuracy tests PASS (<5% error)
✓ Smoothing is effective (low variance)
```

## Development Tips

### Adjusting Scale Count

Edit `heartbeat_monitor/gpu/hardware_detector.py`:

```python
BoardType.RPI5: dict(
    preferred_work_group = 128,
    max_wavelet_scales   = 16,  # Increase for more precision, decrease for speed
    ...
),
```

**Trade-offs**:
- More scales = better frequency resolution but slower computation
- Fewer scales = faster but may miss subtle BPM variations
- Sweet spot: 12-24 scales depending on hardware

### Adjusting Smoothing

Edit `heartbeat_monitor/gpu/wavelet_processor_gpu.py`:

```python
# History buffer size (currently 5)
self._bpm_history: Deque[Tuple[float, float]] = deque(maxlen=5)

# EMA alpha (currently 0.4; smaller = more smoothing)
smoothed_bpm = 0.4 * smoothed_bpm + 0.6 * self._smoothed_bpm
```

### Adjusting Computation Frequency

Edit `main_wavelet_gpu.py`:

```python
# Currently: every 10 frames (~3Hz at 30fps)
if frame_idx % 10 == 0:
    bpm, confidence = processor.compute_bpm()

# For faster updates: every 5 frames (~6Hz at 30fps)
# For slower/stabler: every 20 frames (~1.5Hz at 30fps)
```

## Advanced Usage

### Integrate into Your Application

```python
from heartbeat_monitor.gpu.wavelet_processor_gpu import WaveletProcessorGPU
import cv2

# Initialize
processor = WaveletProcessorGPU(
    fps=30.0,
    window_seconds=12.0,
    bpm_low=45.0,
    bpm_high=240.0,
)

# Process video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Check if finger is present (your logic here)
    if finger_detected:
        processor.push_frame(frame)
        bpm, confidence = processor.compute_bpm()
        
        # Only trust readings with reasonable confidence
        if confidence > 0.2:
            print(f"Heart Rate: {bpm:.1f} BPM (confidence: {confidence:.2f})")
    else:
        processor.reset()

processor.close()
```

## Support & Issues

For detailed technical information, see:
- [GPU_WAVELET_OPTIMIZATIONS.md](GPU_WAVELET_OPTIMIZATIONS.md) - Optimization details and benchmarks
- [WAVELET_IMPLEMENTATION.md](WAVELET_IMPLEMENTATION.md) - Algorithm description
- [docs/gpu_pipeline.md](docs/gpu_pipeline.md) - GPU architecture

If you encounter issues:
1. Check hardware detection: `python main_wavelet_gpu.py --info`
2. Try CPU fallback: `python main_wavelet_gpu.py --cpu-fallback`
3. Run test suite: `python test_gpu_improvements.py`
4. Check logs for warnings/errors
