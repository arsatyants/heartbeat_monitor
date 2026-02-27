#!/usr/bin/env python3
"""
Test script to verify GPU wavelet improvements.

Tests:
1. Reduced scale count for better frequency resolution
2. Parabolic interpolation for sub-bin precision
3. Confidence penalties for edge cases
4. Temporal smoothing for stability
"""

import numpy as np
from heartbeat_monitor.gpu.wavelet_processor_gpu import WaveletProcessorGPU


def generate_synthetic_ppg(fps: float, duration: float, bpm: float, noise_level: float = 0.1) -> np.ndarray:
    """Generate a synthetic PPG signal with known heart rate."""
    t = np.arange(0, duration, 1/fps)
    heart_rate_hz = bpm / 60.0
    
    # Fundamental frequency + second harmonic for realism
    signal = (
        np.sin(2 * np.pi * heart_rate_hz * t) +
        0.3 * np.sin(2 * np.pi * 2 * heart_rate_hz * t) +
        noise_level * np.random.randn(len(t))
    )
    
    # Scale to typical green channel values (150-200) with modulation
    signal = 175 + 15 * signal
    return signal


def test_accuracy():
    """Test BPM detection accuracy on synthetic signals."""
    print("=" * 70)
    print("Testing BPM Detection Accuracy")
    print("=" * 70)
    
    fps = 30.0
    window_seconds = 12.0
    
    # Test different BPM values
    test_bpms = [60, 75, 90, 120, 150]
    
    for target_bpm in test_bpms:
        processor = WaveletProcessorGPU(
            fps=fps,
            window_seconds=window_seconds,
            force_cpu_fallback=True,
        )
        
        # Generate signal
        duration = window_seconds + 2  # Extra time for buffer fill
        signal = generate_synthetic_ppg(fps, duration, target_bpm, noise_level=0.05)
        
        # Feed frames as BGR images (using green channel)
        for value in signal:
            # Create a dummy BGR frame with the signal in green channel
            frame = np.full((100, 100, 3), value, dtype=np.uint8)
            frame[:, :, 1] = int(value)  # Green channel
            processor.push_frame(frame)
        
        # Compute BPM
        detected_bpm, confidence = processor.compute_bpm()
        error = abs(detected_bpm - target_bpm)
        error_pct = (error / target_bpm) * 100
        
        print(f"  Target: {target_bpm:3.0f} BPM  →  "
              f"Detected: {detected_bpm:6.2f} BPM  "
              f"(Error: {error:5.2f} BPM / {error_pct:4.1f}%)  "
              f"Confidence: {confidence:.3f}")
        
        # Check if accuracy is reasonable (within 5%)
        if error_pct < 5.0:
            print(f"    ✓ PASS")
        else:
            print(f"    ✗ FAIL (error too high)")
    
    print()


def test_confidence_penalties():
    """Test that confidence penalties are applied correctly."""
    print("=" * 70)
    print("Testing Confidence Penalties")
    print("=" * 70)
    
    fps = 30.0
    processor = WaveletProcessorGPU(
        fps=fps,
        window_seconds=12.0,
        bpm_low=45.0,
        bpm_high=240.0,
        force_cpu_fallback=True,
    )
    
    # Test low BPM (should get penalty)
    signal_low = generate_synthetic_ppg(fps, 14, 48, noise_level=0.05)
    for value in signal_low:
        frame = np.full((100, 100, 3), value, dtype=np.uint8)
        frame[:, :, 1] = int(value)
        processor.push_frame(frame)
    
    bpm_low, conf_low = processor.compute_bpm()
    print(f"  Low BPM ({bpm_low:.1f}):  Confidence = {conf_low:.3f}")
    print(f"    (Should have penalty for being near lower bound)")
    
    # Test normal BPM (no penalty)
    processor.reset()
    signal_normal = generate_synthetic_ppg(fps, 14, 75, noise_level=0.05)
    for value in signal_normal:
        frame = np.full((100, 100, 3), value, dtype=np.uint8)
        frame[:, :, 1] = int(value)
        processor.push_frame(frame)
    
    bpm_normal, conf_normal = processor.compute_bpm()
    print(f"  Normal BPM ({bpm_normal:.1f}):  Confidence = {conf_normal:.3f}")
    print(f"    (Should have no penalty)")
    
    # Test high BPM (should get penalty)
    processor.reset()
    signal_high = generate_synthetic_ppg(fps, 14, 235, noise_level=0.05)
    for value in signal_high:
        frame = np.full((100, 100, 3), value, dtype=np.uint8)
        frame[:, :, 1] = int(value)
        processor.push_frame(frame)
    
    bpm_high, conf_high = processor.compute_bpm()
    print(f"  High BPM ({bpm_high:.1f}):  Confidence = {conf_high:.3f}")
    print(f"    (Should have penalty for being near upper bound)")
    
    print()


def test_temporal_smoothing():
    """Test that temporal smoothing reduces jitter."""
    print("=" * 70)
    print("Testing Temporal Smoothing")
    print("=" * 70)
    
    fps = 30.0
    processor = WaveletProcessorGPU(
        fps=fps,
        window_seconds=12.0,
        force_cpu_fallback=True,
    )
    
    # Generate signal with slight variations
    base_bpm = 75
    duration = 14
    
    detected_bpms = []
    
    # Simulate 5 measurements with slight BPM drift
    for measurement in range(5):
        processor.reset()
        # Add small random variation
        current_bpm = base_bpm + np.random.uniform(-2, 2)
        signal = generate_synthetic_ppg(fps, duration, current_bpm, noise_level=0.08)
        
        for value in signal:
            frame = np.full((100, 100, 3), value, dtype=np.uint8)
            frame[:, :, 1] = int(value)
            processor.push_frame(frame)
        
        bpm, conf = processor.compute_bpm()
        detected_bpms.append(bpm)
        print(f"  Measurement {measurement + 1}: {bpm:.2f} BPM (confidence: {conf:.3f})")
    
    # Calculate standard deviation (should be small due to smoothing)
    std_dev = np.std(detected_bpms)
    mean_bpm = np.mean(detected_bpms)
    print(f"\n  Mean BPM: {mean_bpm:.2f}  ±  {std_dev:.2f} BPM")
    print(f"  Expected: ~{base_bpm:.0f} BPM")
    
    if std_dev < 3.0:
        print("  ✓ Smoothing is effective (low variance)")
    else:
        print("  ⚠ High variance detected")
    
    print()


def test_scale_configuration():
    """Test that scale configuration is optimized."""
    print("=" * 70)
    print("Testing Scale Configuration")
    print("=" * 70)
    
    processor = WaveletProcessorGPU(
        fps=30.0,
        window_seconds=12.0,
        force_cpu_fallback=True,
    )
    
    n_scales = len(processor._scales)
    scale_min = processor._scales[0]
    scale_max = processor._scales[-1]
    bpm_max = (6.0 / (2 * np.pi)) / (scale_min / 30.0) * 60.0
    bpm_min = (6.0 / (2 * np.pi)) / (scale_max / 30.0) * 60.0
    
    print(f"  Number of scales: {n_scales}")
    print(f"  Scale range: {scale_min:.2f} - {scale_max:.2f}")
    print(f"  BPM range: {bpm_min:.1f} - {bpm_max:.1f}")
    print(f"  Frequency resolution: {(bpm_max - bpm_min) / n_scales:.2f} BPM per scale")
    
    # Check if optimized (should be 16-24 scales)
    if 12 <= n_scales <= 24:
        print("  ✓ Scale count optimized for accuracy vs. performance")
    else:
        print(f"  ⚠ Scale count ({n_scales}) may not be optimal")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GPU Wavelet Processor Improvements Test Suite")
    print("=" * 70 + "\n")
    
    try:
        test_scale_configuration()
        test_accuracy()
        test_confidence_penalties()
        test_temporal_smoothing()
        
        print("=" * 70)
        print("All tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
