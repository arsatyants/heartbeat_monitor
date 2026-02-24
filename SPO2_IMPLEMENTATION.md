# SpO2 Implementation

## Overview
Oxygen saturation (SpO2) measurement has been added to the heartbeat monitor using photoplethysmography (PPG) analysis.

## How It Works

### Algorithm
1. **Dual-Channel Capture**: Both red and green color channels are captured from each video frame
2. **AC/DC Component Extraction**:
   - **DC Component**: Mean value of each channel (baseline signal)
   - **AC Component**: RMS of filtered pulsatile signal (heartbeat oscillations)
3. **Ratio Calculation**: Compute ratio of ratios: R = (AC_red/DC_red) / (AC_green/DC_green)
4. **Calibration Formula**: SpO2 ≈ 110 - 25 × R
5. **Clamping**: Result is clamped to physiologically plausible range (70-100%)

### Technical Details
- Uses the same bandpass filter as heart rate detection (45-240 BPM)
- Calculates RMS (Root Mean Square) of AC components for robust amplitude estimation
- Requires minimum sample buffer before computing (same as BPM)
- Handles division-by-zero and error cases gracefully

## Important Notes

### Accuracy Limitations
⚠️ **This is NOT clinical-grade measurement**

- Standard pulse oximeters use red (~660nm) and infrared (~940nm) LEDs
- Camera-based approach uses visible spectrum (red/green channels)
- Results should be considered **indicative only, not for medical diagnosis**
- Factors affecting accuracy:
  - Ambient lighting conditions
  - Skin tone and pigmentation
  - Camera sensor quality
  - Finger pressure on lens

### Typical Values
- **Healthy**: 95-100%
- **Acceptable**: 90-94%
- **Low**: Below 90%

## Files Modified

1. **heartbeat_monitor/signal_processor.py**
   - Added `_buffer_red` for red channel data
   - Added `compute_spo2()` method
   - Added `last_spo2` property
   - Updated `push_frame()` to capture both channels
   - Updated `reset()` to clear both buffers

2. **heartbeat_monitor/visualizer.py**
   - Added `spo2` parameter to `draw()` method
   - Added `_draw_spo2()` method with color-coded display
   - Color coding: Green (≥95%), Yellow (90-94%), Red (<90%)

3. **main.py**
   - Added `spo2` variable tracking
   - Calls `compute_spo2()` alongside `compute_bpm()`
   - Passes SpO2 to visualizer
   - Includes SpO2 in console logs

4. **tests/test_heartbeat.py**
   - Added 4 new test cases for SpO2 functionality

## Usage

Run the monitor as before:
```bash
python main.py
```

SpO2 will be displayed below the BPM readout on screen and included in console output when in headless mode.

## Future Improvements

- Multi-wavelength LED integration for better accuracy
- Calibration against reference pulse oximeter
- Motion artifact detection and compensation
- Adaptive filtering based on signal quality
