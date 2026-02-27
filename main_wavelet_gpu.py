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
    p.add_argument("--show-bands",    action="store_true",
                   help="Overlay band-energy histogram")
    p.add_argument("--info",          action="store_true",
                   help="Print hardware/OpenCL info and exit")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Band-energy overlay helper
# ---------------------------------------------------------------------------

def _draw_band_overlay(
    frame: np.ndarray,
    energies: np.ndarray,
    bpms: np.ndarray,
    peak_bpm: float,
) -> None:
    """Draw a small bar-chart of CWT band energies in the top-right corner."""
    if len(energies) == 0:
        return

    h, w = frame.shape[:2]
    bar_area_w = 160
    bar_area_h = 80
    x0 = w - bar_area_w - 8
    y0 = 30

    # Dark semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bar_area_w, y0 + bar_area_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    total = energies.sum()
    if total == 0:
        return

    n = len(energies)
    bar_w = max(1, (bar_area_w - 4) // n)
    max_e = energies.max()

    for i, (e, bpm) in enumerate(zip(energies, bpms)):
        bar_h = int((e / max_e) * (bar_area_h - 12))
        bx = x0 + 2 + i * bar_w
        by_top = y0 + bar_area_h - 10 - bar_h
        by_bot = y0 + bar_area_h - 10

        # Highlight peak band
        col = (0, 200, 60) if abs(bpm - peak_bpm) < 10 else (80, 80, 180)
        cv2.rectangle(frame, (bx, by_top), (bx + bar_w - 1, by_bot), col, -1)

    cv2.putText(frame, "Band energy",
                (x0 + 2, y0 + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)


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

    show_bands: bool = args.show_bands
    bpm        = 0.0
    confidence = 0.0
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
                    # Compute BPM at ~2 Hz (every 15 frames at 30 fps).
                    # The CPU-fallback CWT is still O(N·n_scales) and calling
                    # it 10×/s at 30 fps causes visible UI lag.
                    if frame_idx % 15 == 0:
                        bpm, confidence = processor.compute_bpm()
                else:
                    processor.reset()
                    bpm, confidence = 0.0, 0.0

                filtered = processor.get_filtered_signal()

                annotated = vis.draw(
                    frame,
                    bpm=bpm,
                    confidence=confidence,
                    buffer_fill=processor.buffer_fill_ratio,
                    finger_detected=finger,
                    filtered_signal=filtered if len(filtered) > 0 else None,
                )

                # Band-energy overlay (optional)
                if show_bands:
                    _draw_band_overlay(
                        annotated,
                        processor.band_energies,
                        processor.band_bpms,
                        bpm,
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
                    elif key == ord("b"):
                        show_bands = not show_bands

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
