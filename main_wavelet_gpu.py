#!/usr/bin/env python3
"""
Heartbeat Monitor – GPU wavelet version (OpenCL + RPi IMX500).

Fully independent entry point.  Does NOT import from main.py, main_wavelet.py
or any non-GPU processor.  Uses:
  heartbeat_monitor.gpu.wavelet_processor_gpu.WaveletProcessorGPU
  heartbeat_monitor.gpu.hardware_detector.detect

Usage
-----
    python main_wavelet_gpu.py [OPTIONS]

Options
-------
    --resolution WxH        Camera resolution      (default: 640x480)
    --fps INT               Target frame rate       (default: 30)
    --window FLOAT          Analysis window seconds (default: 12)
    --bpm-low INT           Minimum BPM search     (default: 45)
    --bpm-high INT          Maximum BPM search     (default: 240)
    --no-flip               Disable horizontal mirror
    --camera-index INT      OpenCV camera index     (default: 0)
    --save PATH             Save annotated video to file
    --headless              No window; log BPM to stdout
    --cpu-fallback          Force CPU-only mode (no OpenCL)
    --show-bands            Overlay per-scale band-energy bar chart
    --info                  Print hardware/OpenCL info and exit

Keyboard shortcuts (when a window is open)
------------------------------------------
    q / ESC  – quit
    r        – reset signal buffer
    s        – save PNG snapshot
    b        – toggle band-energy overlay
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Must be set before cv2 is imported so Qt5 uses X11/XWayland instead of
# looking for a Wayland plugin that is not bundled with pip-installed opencv.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

from heartbeat_monitor.camera import IMX500Camera
from heartbeat_monitor.finger_detector import FingerDetector
from heartbeat_monitor.gpu.hardware_detector import detect as detect_hardware
from heartbeat_monitor.gpu.wavelet_processor_gpu import WaveletProcessorGPU
from heartbeat_monitor.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heartbeat_gpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heartbeat monitor – GPU-accelerated OpenCL CWT (IMX500)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resolution",    default="640x480")
    p.add_argument("--fps",           type=int,   default=30)
    p.add_argument("--window",        type=float, default=12.0)
    p.add_argument("--bpm-low",       type=float, default=45.0)
    p.add_argument("--bpm-high",      type=float, default=240.0)
    p.add_argument("--no-flip",       action="store_true")
    p.add_argument("--camera-index",  type=int,   default=0)
    p.add_argument("--save",          type=Path,  default=None)
    p.add_argument("--headless",      action="store_true")
    p.add_argument("--cpu-fallback",  action="store_true",
                   help="Disable OpenCL; use NumPy CPU fallback")
    p.add_argument("--info",          action="store_true",
                   help="Print hardware/OpenCL info and exit")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Band-energy overlay helper
# ---------------------------------------------------------------------------

def _draw_band_powers(
    frame: np.ndarray,
    band_powers: np.ndarray,
    dominant_band: int,
    band_edges: np.ndarray,   # Hz
) -> None:
    """Draw frequency band power bars with BPM ranges – identical to CPU version."""
    h, w = frame.shape[:2]
    bar_width   = 20
    bar_spacing = 30
    start_x     = 15
    max_height  = 80
    waveform_h  = 80
    start_y     = h - waveform_h - 20 - max_height - 30

    norm_powers = band_powers / band_powers.max() if band_powers.max() > 0 else band_powers

    for i, power in enumerate(norm_powers):
        bar_h   = int(power * max_height)
        x       = start_x + i * bar_spacing
        y_bot   = start_y + max_height
        y_top   = y_bot - bar_h

        color = (0, 255, 255) if i == dominant_band else (180, 180, 180)
        cv2.rectangle(frame, (x, y_top), (x + bar_width, y_bot), color, -1)
        cv2.putText(frame, str(i), (x + 5, y_bot + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        if i < len(band_edges) - 1:
            low_bpm  = int(band_edges[i] * 60)
            high_bpm = int(band_edges[i + 1] * 60)
            label    = f"{low_bpm}-{high_bpm}"
            font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
            tsz  = cv2.getTextSize(label, font, fscale, thick)[0]
            timg = np.zeros((tsz[1] + 4, tsz[0] + 4, 3), dtype=np.uint8)
            cv2.putText(timg, label, (2, tsz[1] + 2), font, fscale,
                        (128, 128, 128), thick, cv2.LINE_AA)
            rot  = cv2.rotate(timg, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tx   = x + bar_width + 2
            ty   = max(0, y_bot - rot.shape[0])
            rh, rw = rot.shape[:2]
            if ty + rh <= frame.shape[0] and tx + rw <= frame.shape[1]:
                roi  = frame[ty:ty + rh, tx:tx + rw]
                mask = (rot > 0).any(axis=2)
                roi[mask] = rot[mask]

    cv2.putText(frame, "Bands (BPM)", (start_x, start_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    # ------------------------------------------------------------------
    # Hardware info / early exit
    # ------------------------------------------------------------------
    hw = detect_hardware(force_cpu_fallback=args.cpu_fallback)
    if args.info:
        print(hw.summary())
        return 0

    print(hw.summary())

    # ------------------------------------------------------------------
    # Parse resolution
    # ------------------------------------------------------------------
    try:
        res_w, res_h = (int(v) for v in args.resolution.lower().split("x"))
    except ValueError:
        logger.error("Invalid --resolution '%s'.  Use WxH, e.g. 640x480.",
                     args.resolution)
        return 1

    resolution = (res_w, res_h)

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    camera = IMX500Camera(
        resolution=resolution,
        fps=args.fps,
        flip_horizontal=not args.no_flip,
        camera_index=args.camera_index,
    )

    processor = WaveletProcessorGPU(
        fps=float(args.fps),
        window_seconds=args.window,
        bpm_low=args.bpm_low,
        bpm_high=args.bpm_high,
        hardware_profile=hw,
        force_cpu_fallback=args.cpu_fallback,
    )

    # IMX500 auto-exposure brightens a covered lens above the default
    # threshold of 100.  Loosen both brightness and variance limits so the
    # GPU version actually fires on a real finger placement.
    detector = FingerDetector(
        brightness_threshold=160,
        variance_threshold=1500,
        red_dominance=1.02,
    )
    vis      = Visualizer(resolution=resolution, show_fps=not args.headless)

    writer: cv2.VideoWriter | None = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, args.fps, resolution)
        logger.info("Saving video → %s", args.save)

    bpm        = 0.0
    confidence = 0.0
    fft_freqs: np.ndarray = np.array([])
    fft_power: np.ndarray = np.array([])
    frame_idx  = 0
    log_every  = args.fps   # print once per second in headless mode

    logger.info(
        "GPU Heartbeat Monitor started.  OpenCL=%s  board=%s",
        processor._opencl_available,
        hw.board.name,
    )

    if not args.headless:
        cv2.namedWindow("Heartbeat GPU Monitor — IMX500", cv2.WINDOW_NORMAL)

    try:
        with camera:
            for frame in camera.frames():
                finger = detector.is_finger(frame)

                if finger:
                    processor.push_frame(frame)
                    # Compute BPM at ~3 Hz (every 10 frames at 30 fps).
                    # With optimized scale count (16 instead of 32), the CWT
                    # is now 2× faster and can be called more frequently for
                    # better responsiveness.
                    if frame_idx % 10 == 0:
                        bpm, confidence = processor.compute_bpm()
                        fft_freqs, fft_power = processor.get_fft_data()
                else:
                    processor.reset()
                    bpm, confidence = 0.0, 0.0
                    fft_freqs, fft_power = np.array([]), np.array([])

                filtered = processor.get_filtered_signal()

                annotated = vis.draw(
                    frame,
                    bpm=bpm,
                    confidence=confidence,
                    buffer_fill=processor.buffer_fill_ratio,
                    finger_detected=finger,
                    filtered_signal=filtered if len(filtered) > 0 else None,
                    fft_freqs=fft_freqs if len(fft_freqs) > 0 else None,
                    fft_power=fft_power if len(fft_power) > 0 else None,
                )

                # Band-power panel (always visible, same layout as CPU version)
                if processor.band_powers.sum() > 0:
                    _draw_band_powers(
                        annotated,
                        processor.band_powers,
                        processor.dominant_band,
                        processor.band_edges,
                    )

                # GPU/CPU mode badge
                mode_txt = (
                    f"GPU | {hw.board.name}"
                    if processor._opencl_available
                    else f"CPU | {hw.board.name}"
                )
                cv2.putText(
                    annotated, mode_txt,
                    (8, annotated.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 200, 255) if processor._opencl_available else (80, 80, 200),
                    1, cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(annotated)

                if args.headless and frame_idx % log_every == 0:
                    ts = time.strftime("%H:%M:%S")
                    if bpm > 0:
                        print(f"[{ts}] BPM={bpm:.1f}  conf={confidence:.2f}"
                              f"  finger={finger}"
                              f"  gpu={processor._opencl_available}")
                    else:
                        print(f"[{ts}] Waiting…  finger={finger}")

                if not args.headless:
                    cv2.imshow("Heartbeat GPU Monitor — IMX500", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logger.info("Quit by user.")
                        break
                    elif key == ord("r"):
                        processor.reset()
                        logger.info("Buffer reset.")
                    elif key == ord("s"):
                        fname = f"snapshot_gpu_{int(time.time())}.png"
                        cv2.imwrite(fname, annotated)
                        logger.info("Saved %s", fname)

                frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        processor.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run(parse_args()))
