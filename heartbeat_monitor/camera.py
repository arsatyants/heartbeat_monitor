"""
Camera module for Raspberry Pi IMX500.

Wraps picamera2 to provide a simple iterator of OpenCV-compatible BGR frames.
Falls back to OpenCV VideoCapture (any webcam) when picamera2 is unavailable,
which is handy for development on non-Pi hardware.
"""

from __future__ import annotations

import logging
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing picamera2 (only available on Raspberry Pi OS)
# ---------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    from libcamera import Transform          # optional horizontal flip
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False
    logger.warning("picamera2 not found – falling back to OpenCV VideoCapture.")


class IMX500Camera:
    """
    Thin wrapper around the Raspberry Pi Camera Module / IMX500.

    Parameters
    ----------
    resolution:
        (width, height) of captured frames.
    fps:
        Target frame rate.  Actual rate may differ slightly.
    flip_horizontal:
        Mirror the image left-to-right (useful for selfie-style use).
    camera_index:
        Fallback OpenCV camera index when picamera2 is unavailable.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        flip_horizontal: bool = False,
        camera_index: int = 0,
    ) -> None:
        self.resolution = resolution
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.camera_index = camera_index

        self._cam: "Picamera2 | cv2.VideoCapture | None" = None
        self._use_picamera2 = _PICAMERA2_AVAILABLE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialise and start the camera."""
        if self._use_picamera2:
            self._open_picamera2()
        else:
            self._open_opencv()
        logger.info(
            "Camera opened – backend=%s resolution=%s fps=%d",
            "picamera2" if self._use_picamera2 else "opencv",
            self.resolution,
            self.fps,
        )

    def close(self) -> None:
        """Stop and release the camera."""
        if self._cam is None:
            return
        if self._use_picamera2:
            self._cam.stop()
            self._cam.close()
        else:
            self._cam.release()
        self._cam = None
        logger.info("Camera closed.")

    # Context-manager support
    def __enter__(self) -> "IMX500Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read_frame(self) -> np.ndarray | None:
        """
        Capture a single frame.

        Returns
        -------
        numpy.ndarray
            BGR image array (H × W × 3, dtype uint8), or *None* on failure.
        """
        if self._cam is None:
            raise RuntimeError("Camera is not open.  Call open() first.")

        if self._use_picamera2:
            return self._read_picamera2()
        return self._read_opencv()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield frames indefinitely until the camera is closed or an error occurs.

        Usage::

            with IMX500Camera() as cam:
                for frame in cam.frames():
                    process(frame)
        """
        _null_streak = 0
        while self._cam is not None:
            frame = self.read_frame()
            if frame is None:
                _null_streak += 1
                if _null_streak >= 10:
                    logger.error(
                        "Camera returned 10 consecutive None frames – aborting."
                    )
                    break
                continue
            _null_streak = 0
            yield frame

    # ------------------------------------------------------------------
    # Private helpers – picamera2
    # ------------------------------------------------------------------

    def _open_picamera2(self) -> None:
        cam = Picamera2()
        w, h = self.resolution
        transform = Transform(hflip=self.flip_horizontal)
        # RGB888 is the safest 3-channel format across all Pi camera models
        # (IMX500, Camera Module 3, HQ Camera, etc.) and all picamera2 versions.
        config = cam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},
            transform=transform,
            buffer_count=4,
        )
        cam.configure(config)
        frame_duration = int(1_000_000 / self.fps)   # microseconds
        try:
            cam.set_controls({
                "FrameDurationLimits": (frame_duration, frame_duration),
            })
        except Exception as exc:                         # noqa: BLE001
            logger.warning("Could not set FrameDurationLimits: %s", exc)
        cam.start()
        # Discard the first few frames so auto-exposure/white-balance can settle.
        # Without this the IMX500 often returns under-exposed (black) frames.
        for _ in range(8):
            cam.capture_array("main")
        self._cam = cam

    def _read_picamera2(self) -> np.ndarray | None:
        # Always specify the stream name for compatibility with newer picamera2
        frame = self._cam.capture_array("main")
        if frame is None:
            logger.warning("capture_array returned None.")
            return None
        # Drop alpha channel if camera returned 4-channel XRGB/RGBA
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        # Validate: warn if the frame is completely black (all zeros)
        if frame.max() == 0:
            logger.warning(
                "Camera returned an all-black frame "
                "(shape=%s dtype=%s) – check lens cap / exposure.",
                frame.shape, frame.dtype,
            )
        # picamera2 RGB888 → OpenCV BGR
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Private helpers – OpenCV fallback
    # ------------------------------------------------------------------

    def _open_opencv(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video capture device index={self.camera_index}"
            )
        w, h = self.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cam = cap

    def _read_opencv(self) -> np.ndarray | None:
        ok, frame = self._cam.read()
        if not ok:
            logger.warning("VideoCapture.read() returned False.")
            return None
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame
