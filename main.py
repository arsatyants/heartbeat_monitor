#!/usr/bin/env python3
"""
Heartbeat Monitor – main entry point.

Usage
-----
    python main.py [OPTIONS]

Options
-------
    --resolution WxH     Camera resolution (default: 640x480)
    --fps INT            Target frame rate  (default: 30)
    --window FLOAT       PPG analysis window in seconds (default: 12)
    --no-flip            Disable horizontal mirror
    --camera-index INT   OpenCV camera index (fallback, default: 0)
    --save PATH          Save annotated video to file (optional)
    --headless           Run without display window (log BPM to stdout)

Keyboard shortcuts (when a window is open)
------------------------------------------
    q / ESC  – quit
    r        – reset signal buffer
    s        – save a single annotated frame as PNG
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
from heartbeat_monitor.signal_processor import SignalProcessor
from heartbeat_monitor.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heartbeat_monitor")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heartbeat monitor via Pi Camera IMX500 (rPPG)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resolution", default="640x480",
                        help="Camera resolution, e.g. 640x480")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target capture frame rate")
    parser.add_argument("--window", type=float, default=12.0,
                        help="PPG analysis window in seconds")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal image flip")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV VideoCapture index (fallback)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save annotated video to this file path")
    parser.add_argument("--headless", action="store_true",
                        help="No display window; log BPM to stdout only")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    # Parse resolution
    try:
        res_w, res_h = (int(v) for v in args.resolution.lower().split("x"))
    except ValueError:
        logger.error("Invalid --resolution format.  Use WxH, e.g. 640x480.")
        return 1

    resolution = (res_w, res_h)

    # Initialise components
    camera    = IMX500Camera(
        resolution=resolution,
        fps=args.fps,
        flip_horizontal=not args.no_flip,
        camera_index=args.camera_index,
    )
    processor = SignalProcessor(fps=float(args.fps), window_seconds=args.window)
    detector  = FingerDetector()
    vis       = Visualizer(resolution=resolution, show_fps=not args.headless)

    # Optional video writer
    writer: cv2.VideoWriter | None = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, args.fps, resolution)
        logger.info("Saving video to %s", args.save)

    logger.info("Starting heartbeat monitor.  Press 'q' or ESC to quit.")

    if not args.headless:
        cv2.namedWindow("Heartbeat Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Heartbeat Monitor", res_w, res_h)

    bpm        = 0.0
    confidence = 0.0
    spo2       = 0.0
    frame_idx  = 0
    bpm_log_interval = args.fps  # log to stdout every ~1 second

    try:
        with camera:
            roi_x, roi_y, roi_w, roi_h = vis.get_roi()
            for frame in camera.frames():
                roi_patch = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                finger_present = detector.is_finger(roi_patch)

                if finger_present:
                    processor.push_frame(roi_patch)
                    if frame_idx % 3 == 0:          # recompute every 3rd frame
                        bpm, confidence = processor.compute_bpm()
                        spo2 = processor.compute_spo2()
                else:
                    # Reset buffer when finger is removed
                    processor.reset()
                    bpm, confidence, spo2 = 0.0, 0.0, 0.0

                filtered = processor.get_filtered_signal()
                fft_freqs, fft_power = processor.get_fft_data()

                annotated = vis.draw(
                    frame,
                    bpm=bpm,
                    confidence=confidence,
                    buffer_fill=processor.buffer_fill_ratio,
                    finger_detected=finger_present,
                    filtered_signal=filtered if len(filtered) > 0 else None,
                    fft_freqs=fft_freqs if len(fft_freqs) > 0 else None,
                    spo2=spo2,
                    fft_power=fft_power if len(fft_power) > 0 else None,
                )

                if writer is not None:
                    writer.write(annotated)

                # Stdout log
                if args.headless and frame_idx % bpm_log_interval == 0:
                    ts = time.strftime("%H:%M:%S")
                    if bpm > 0:
                        print(f"[{ts}] BPM={bpm:.1f}  SpO2={spo2:.0f}%  conf={confidence:.2f}  finger={finger_present}")
                    else:
                        print(f"[{ts}] Waiting for signal…  finger={finger_present}")

                if not args.headless:
                    cv2.imshow("Heartbeat Monitor", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):          # q or ESC
                        logger.info("Quit requested by user.")
                        break
                    elif key == ord("r"):
                        processor.reset()
                        logger.info("Signal buffer reset.")
                    elif key == ord("s"):
                        fname = f"snapshot_{int(time.time())}.png"
                        cv2.imwrite(fname, annotated)
                        logger.info("Saved snapshot: %s", fname)

                frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
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
