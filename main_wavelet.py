#!/usr/bin/env python3
"""
Heartbeat Monitor – Wavelet version.

This is an alternative implementation using wavelet transform instead of
Butterworth filtering. Use this to compare performance with the standard
FFT-based approach.

Usage
-----
    python main_wavelet.py [OPTIONS]

Options
-------
    --resolution WxH     Camera resolution (default: 640x480)
    --fps INT            Target frame rate  (default: 30)
    --window FLOAT       PPG analysis window in seconds (default: 12)
    --bands INT          Number of frequency bands (default: 6)
    --wavelet STR        Wavelet type (default: morl)
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

import cv2
import numpy as np

from heartbeat_monitor.camera import IMX500Camera
from heartbeat_monitor.finger_detector import FingerDetector
from heartbeat_monitor.signal_processor_wavelet import SignalProcessorWavelet
from heartbeat_monitor.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heartbeat_monitor_wavelet")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heartbeat monitor via Pi Camera IMX500 (wavelet-based rPPG)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resolution", default="640x480",
                        help="Camera resolution, e.g. 640x480")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target capture frame rate")
    parser.add_argument("--window", type=float, default=12.0,
                        help="PPG analysis window in seconds")
    parser.add_argument("--bands", type=int, default=6,
                        help="Number of frequency bands for wavelet analysis")
    parser.add_argument("--wavelet", type=str, default="morl",
                        help="Wavelet type (morl, mexh, gaus1, etc.)")
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

    # Initialise components - using wavelet processor
    camera    = IMX500Camera(
        resolution=resolution,
        fps=args.fps,
        flip_horizontal=not args.no_flip,
        camera_index=args.camera_index,
    )
    processor = SignalProcessorWavelet(
        fps=float(args.fps),
        window_seconds=args.window,
        n_bands=args.bands,
        wavelet=args.wavelet,
    )
    detector  = FingerDetector()
    vis       = Visualizer(resolution=resolution, show_fps=not args.headless)

    # Optional video writer
    writer: cv2.VideoWriter | None = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, args.fps, resolution)
        logger.info("Saving video to %s", args.save)

    logger.info("Starting wavelet-based heartbeat monitor (bands=%d, wavelet=%s).", 
                args.bands, args.wavelet)
    logger.info("Press 'q' or ESC to quit.")

    if not args.headless:
        cv2.namedWindow("Heartbeat Monitor (Wavelet)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Heartbeat Monitor (Wavelet)", res_w, res_h)

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
                    fft_power=fft_power if len(fft_power) > 0 else None,
                    spo2=spo2,
                )

                # Add wavelet info to frame
                if processor.band_powers.sum() > 0:
                    _draw_band_powers(annotated, processor.band_powers, 
                                     processor.dominant_band, processor.band_edges)

                if writer is not None:
                    writer.write(annotated)

                # Stdout log
                if args.headless and frame_idx % bpm_log_interval == 0:
                    ts = time.strftime("%H:%M:%S")
                    if bpm > 0:
                        band_info = f"band={processor.dominant_band}"
                        print(f"[{ts}] BPM={bpm:.1f}  SpO2={spo2:.0f}%  conf={confidence:.2f}  {band_info}  finger={finger_present}")
                    else:
                        print(f"[{ts}] Waiting for signal…  finger={finger_present}")

                if not args.headless:
                    cv2.imshow("Heartbeat Monitor (Wavelet)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):          # q or ESC
                        logger.info("Quit requested by user.")
                        break
                    elif key == ord("r"):
                        processor.reset()
                        logger.info("Signal buffer reset.")
                    elif key == ord("s"):
                        fname = f"snapshot_wavelet_{int(time.time())}.png"
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


def _draw_band_powers(frame: np.ndarray, band_powers: np.ndarray, dominant_band: int, band_edges: np.ndarray) -> None:
    """Draw frequency band power bars with frequency ranges."""
    h, w = frame.shape[:2]
    bar_width = 20
    bar_spacing = 30
    start_x = 15  # Left side of screen
    max_height = 80
    waveform_height = 80  # Match visualizer waveform height
    # Align bottom with FFT panel: end above waveform, raised by 30px to avoid cyan bar
    start_y = h - waveform_height - 20 - max_height - 30  # Raised 30px higher
    
    # Normalize powers
    if band_powers.max() > 0:
        norm_powers = band_powers / band_powers.max()
    else:
        norm_powers = band_powers
    
    for i, power in enumerate(norm_powers):
        bar_h = int(power * max_height)
        x = start_x + i * bar_spacing
        y_bottom = start_y + max_height
        y_top = y_bottom - bar_h
        
        # Highlight dominant band
        color = (0, 255, 255) if i == dominant_band else (180, 180, 180)
        cv2.rectangle(frame, (x, y_top), (x + bar_width, y_bottom), color, -1)
        
        # Band number label
        cv2.putText(frame, f"{i}", (x + 5, y_bottom + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Frequency range label (BPM) - vertical text along the bar
        if i < len(band_edges) - 1:
            low_bpm = int(band_edges[i] * 60)
            high_bpm = int(band_edges[i + 1] * 60)
            freq_label = f"{low_bpm}-{high_bpm}"
            
            # Create temporary image for rotated text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            text_size = cv2.getTextSize(freq_label, font, font_scale, thickness)[0]
            
            # Create image with text
            text_img = np.zeros((text_size[1] + 4, text_size[0] + 4, 3), dtype=np.uint8)
            cv2.putText(text_img, freq_label, (2, text_size[1] + 2), 
                       font, font_scale, (128, 128, 128), thickness, cv2.LINE_AA)
            
            # Rotate 90 degrees counter-clockwise
            rotated = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Place on frame alongside the bar
            text_x = x + bar_width + 2
            text_y = y_bottom - rotated.shape[0]
            if text_y < 0:
                text_y = 0
            
            # Blend the text onto frame
            h_rot, w_rot = rotated.shape[:2]
            if text_y + h_rot <= frame.shape[0] and text_x + w_rot <= frame.shape[1]:
                roi = frame[text_y:text_y + h_rot, text_x:text_x + w_rot]
                mask = (rotated > 0).any(axis=2)
                roi[mask] = rotated[mask]
    
    # Title
    cv2.putText(frame, "Bands (BPM)", (start_x, start_y - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run(parse_args()))
