# Heartbeat Monitor — IMX500 rPPG

Real-time **heartbeat frequency detection** using a Raspberry Pi Camera Module
(IMX500) and remote photoplethysmography (rPPG).

Place your fingertip directly on the camera lens. The system measures subtle
changes in the green-channel brightness caused by blood-volume pulsation, applies
a bandpass filter, and computes your heart rate via FFT.

---

## Hardware

| Component | Requirement |
|-----------|-------------|
| SBC | Raspberry Pi 4 / 5, Zero 2W|
| Camera | Raspberry Pi AI Camera (IMX500) or any picamera2-compatible module |
| OS | Raspberry Pi OS Bookworm (64-bit) |

---

## Quick start

```bash
git clone https://github.com/arsatyants/heartbeat_monitor.git
cd heartbeat_monitor
chmod +x setup.sh run.sh
./setup.sh          # creates .venv with system-site-packages, installs deps
./run.sh            # opens live camera window
```

> **Note:** `picamera2` and `libcamera` are installed system-wide via Raspberry Pi OS.  
> The setup script creates the venv with `--system-site-packages` so they are accessible.

### Usage — place your finger on the lens
1. Run `./run.sh`
2. A window **"Heartbeat Monitor"** opens showing the live camera feed
3. Place your **fingertip firmly over the camera lens** — the ROI box in the centre turns green
4. Hold still for ~12 seconds while the buffer fills (watch the cyan progress bar)
5. Your **BPM** appears top-left, colour-coded by confidence (green = high, yellow = medium, red = low)
6. A scrolling **PPG waveform** is drawn at the bottom of the frame

### Headless (no display)
```bash
./run.sh --headless
```

### Save video
```bash
./run.sh --save recording.mp4
```

---

## CLI options

```
--resolution WxH     Camera resolution  (default: 640x480)
--fps INT            Target frame rate  (default: 30)
--window FLOAT       PPG analysis window seconds (default: 12)
--no-flip            Disable horizontal mirror
--camera-index INT   OpenCV fallback camera index (default: 0)
--save PATH          Save annotated video to PATH
--headless           No window; print BPM to stdout
```

### Keyboard shortcuts
| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset signal buffer |
| `s` | Save PNG snapshot |

---

## How it works

```
Camera frame
     │
     ▼
FingerDetector ──(no finger)──► reset buffer
     │ finger present
     ▼
SignalProcessor.push_frame()
  • Extract mean green channel value
  • Append to rolling buffer (default 12 s)
     │
     ▼
SignalProcessor.compute_bpm()
  • Detrend (remove DC)
  • Butterworth bandpass 0.75–4 Hz (45–240 BPM)
  • FFT → dominant peak → BPM
  • Confidence = peak power / total band power
     │
     ▼
Visualizer.draw()
  • ROI highlight box
  • BPM readout (colour-coded by confidence)
  • Scrolling PPG waveform strip
  • Buffer fill bar
```

---

## Project structure

```
heartbeat_monitor/
├── main.py                   # Entry point & main loop
├── requirements.txt
├── setup.py
├── setup.sh                  # First-time environment setup
├── run.sh                    # Quick launcher
├── heartbeat_monitor/
│   ├── camera.py             # IMX500 / OpenCV camera wrapper
│   ├── finger_detector.py    # Heuristic finger-on-lens detector
│   ├── signal_processor.py   # PPG extraction, filter, FFT → BPM
│   └── visualizer.py         # OpenCV overlay rendering
└── tests/
    └── test_heartbeat.py     # Unit tests (pytest)
```

---

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video capture & display |
| `numpy` | Array operations |
| `scipy` | Butterworth bandpass filter |
| `picamera2` | Raspberry Pi camera interface (Pi only) |

---

## References

- Verkruysse W. et al., *Remote plethysmographic imaging using ambient light*, Opt. Express 2008.
- De Haan G. & Jeanne V., *Robust pulse rate from chrominance-based rPPG*, IEEE TBME 2013.
