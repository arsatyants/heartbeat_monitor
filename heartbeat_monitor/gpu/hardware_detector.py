"""
Hardware detector for OpenCL GPU selection.

Identifies the host platform (Raspberry Pi 5 / Zero 2W / other) by reading
``/proc/cpuinfo``, then enumerates all available OpenCL platforms and devices.

The public function :func:`detect` returns a :class:`HardwareProfile` that the
GPU wavelet processor uses to pick optimal OpenCL work-group sizes, memory
strategies and wavelet scale counts for the detected device.

Supported targets
-----------------
* Raspberry Pi 5  – BCM2712, VideoCore VII (OpenCL via Mesa/v3dv or Clover)
* Raspberry Pi Zero 2W – BCM2710A1, VideoCore VI (limited OpenCL, <= 64 WG)
* Generic x86/ARM GPU – any device reported by the installed ICD
* CPU fallback – OpenCL CPU device when no GPU is found
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Board enumeration
# ---------------------------------------------------------------------------

class BoardType(Enum):
    RPI5          = auto()   # BCM2712  – VideoCore VII
    RPI_ZERO_2W   = auto()   # BCM2710A1 – VideoCore VI (same die as RPi 3)
    RPI_OTHER     = auto()   # older Pi with VideoCore IV/VI
    GENERIC_ARM   = auto()   # non-Pi ARM SBC
    X86           = auto()   # desktop / server
    UNKNOWN       = auto()


# ---------------------------------------------------------------------------
# OpenCL device profile
# ---------------------------------------------------------------------------

@dataclass
class OpenCLDeviceInfo:
    platform_name:  str
    device_name:    str
    device_type:    str          # "GPU", "CPU", "ACCELERATOR", "ALL"
    max_work_group: int          # CL_DEVICE_MAX_WORK_GROUP_SIZE
    global_mem_mb:  int
    local_mem_kb:   int
    compute_units:  int
    opencl_version: str
    platform_index: int
    device_index:   int

    def __str__(self) -> str:
        return (
            f"[{self.platform_index}:{self.device_index}] "
            f"{self.device_type} | {self.device_name} | "
            f"{self.platform_name} | "
            f"CUs={self.compute_units} WGmax={self.max_work_group} "
            f"gmem={self.global_mem_mb} MB lmem={self.local_mem_kb} KB"
        )


# ---------------------------------------------------------------------------
# Hardware profile – the single object returned to callers
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    board:          BoardType
    board_name:     str          # human-readable, e.g. "Raspberry Pi 5 Model B"
    opencl_device:  Optional[OpenCLDeviceInfo]

    # Tuning hints derived from hardware
    preferred_work_group:  int   # safe CL work-group size
    max_wavelet_scales:    int   # how many CWT scales to compute
    signal_buffer_limit:   int   # max samples to hold in GPU buffer
    use_float16:           bool  # True if fp16 is faster on this device
    has_gpu:               bool

    def summary(self) -> str:
        lines = [
            f"Board      : {self.board_name}  ({self.board.name})",
            f"Has GPU    : {self.has_gpu}",
            f"WG size    : {self.preferred_work_group}",
            f"CWT scales : {self.max_wavelet_scales}",
            f"Buf limit  : {self.signal_buffer_limit} samples",
            f"fp16       : {self.use_float16}",
        ]
        if self.opencl_device:
            lines.append(f"OpenCL dev : {self.opencl_device}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Board identification
# ---------------------------------------------------------------------------

def _read_cpuinfo() -> str:
    try:
        return Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        return ""


def _identify_board(cpuinfo: str) -> tuple[BoardType, str]:
    """Return (BoardType, human-readable model string)."""
    # Try /proc/device-tree/model first (most reliable on Pi)
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            model = model_path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
        except OSError:
            model = ""
    else:
        m = re.search(r"^Model\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        model = m.group(1).strip() if m else ""

    model_lower = model.lower()

    if "raspberry pi 5" in model_lower:
        return BoardType.RPI5, model or "Raspberry Pi 5"
    if ("raspberry pi zero 2" in model_lower or
            "bcm2710a1" in cpuinfo.lower()):
        return BoardType.RPI_ZERO_2W, model or "Raspberry Pi Zero 2W"
    if "raspberry pi" in model_lower:
        return BoardType.RPI_OTHER, model or "Raspberry Pi (unknown revision)"

    machine = platform.machine().lower()
    if machine.startswith(("arm", "aarch")):
        return BoardType.GENERIC_ARM, f"Generic ARM ({machine})"
    return BoardType.X86, f"x86 ({platform.machine()})"


# ---------------------------------------------------------------------------
# OpenCL enumeration
# ---------------------------------------------------------------------------

def _enumerate_opencl_devices() -> list[OpenCLDeviceInfo]:
    """Return all available OpenCL devices via pyopencl."""
    try:
        import pyopencl as cl
    except ImportError:
        logger.warning("pyopencl not installed – OpenCL unavailable.")
        return []

    devices: list[OpenCLDeviceInfo] = []
    try:
        platforms = cl.get_platforms()
    except cl.Error as exc:
        logger.warning("OpenCL platform enumeration failed: %s", exc)
        return []

    for p_idx, platform in enumerate(platforms):
        try:
            cl_devices = platform.get_devices()
        except cl.Error as exc:
            logger.warning("Cannot list devices on platform %s: %s", platform.name, exc)
            continue

        for d_idx, dev in enumerate(cl_devices):
            try:
                dtype_raw = dev.type
                if dtype_raw == cl.device_type.GPU:
                    dtype_str = "GPU"
                elif dtype_raw == cl.device_type.CPU:
                    dtype_str = "CPU"
                elif dtype_raw == cl.device_type.ACCELERATOR:
                    dtype_str = "ACCELERATOR"
                else:
                    dtype_str = "ALL"

                info = OpenCLDeviceInfo(
                    platform_name  = platform.name.strip(),
                    device_name    = dev.name.strip(),
                    device_type    = dtype_str,
                    max_work_group = int(dev.max_work_group_size),
                    global_mem_mb  = int(dev.global_mem_size // (1024 * 1024)),
                    local_mem_kb   = int(dev.local_mem_size // 1024),
                    compute_units  = int(dev.max_compute_units),
                    opencl_version = dev.version.strip(),
                    platform_index = p_idx,
                    device_index   = d_idx,
                )
                devices.append(info)
            except cl.Error as exc:
                logger.warning("Error reading device %d on platform %d: %s",
                               d_idx, p_idx, exc)

    return devices


def _pick_best_device(
    devices: list[OpenCLDeviceInfo],
    board: BoardType,
) -> Optional[OpenCLDeviceInfo]:
    """
    Choose the most suitable device.

    Priority: GPU > ACCELERATOR > CPU.
    On RPi, prefer the first GPU (VideoCore) exactly as-is.
    """
    if not devices:
        return None

    # Prefer GPU, then Accelerator, then CPU
    for preferred_type in ("GPU", "ACCELERATOR", "CPU"):
        subset = [d for d in devices if d.device_type == preferred_type]
        if subset:
            return subset[0]

    return devices[0]


# ---------------------------------------------------------------------------
# Tuning table
# ---------------------------------------------------------------------------

_BOARD_TUNING: dict[BoardType, dict] = {
    # RPi 5 – VideoCore VII – 4 shader slices, higher bandwidth
    BoardType.RPI5: dict(
        preferred_work_group = 128,
        max_wavelet_scales   = 16,  # Reduced for better frequency resolution
        signal_buffer_limit  = 1024,
        use_float16          = False,   # VC7 handles fp32 natively
    ),
    # RPi Zero 2W – VideoCore VI – 1 shader slice, constrained memory
    BoardType.RPI_ZERO_2W: dict(
        preferred_work_group = 64,
        max_wavelet_scales   = 12,  # Reduced for better frequency resolution
        signal_buffer_limit  = 512,
        use_float16          = True,    # saves memory bus bandwidth
    ),
    # Older Pi (3B, 4B, etc.)
    BoardType.RPI_OTHER: dict(
        preferred_work_group = 64,
        max_wavelet_scales   = 16,  # Reduced for better frequency resolution
        signal_buffer_limit  = 512,
        use_float16          = False,
    ),
    # Generic ARM / x86
    BoardType.GENERIC_ARM: dict(
        preferred_work_group = 128,
        max_wavelet_scales   = 16,  # Reduced for better frequency resolution
        signal_buffer_limit  = 1024,
        use_float16          = False,
    ),
    BoardType.X86: dict(
        preferred_work_group = 256,
        max_wavelet_scales   = 24,  # Reduced for better frequency resolution
        signal_buffer_limit  = 2048,
        use_float16          = False,
    ),
    BoardType.UNKNOWN: dict(
        preferred_work_group = 64,
        max_wavelet_scales   = 12,  # Reduced for better frequency resolution
        signal_buffer_limit  = 512,
        use_float16          = False,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(force_cpu_fallback: bool = False) -> HardwareProfile:
    """
    Detect hardware and return a populated :class:`HardwareProfile`.

    Parameters
    ----------
    force_cpu_fallback:
        When *True*, skip GPU selection and use an OpenCL CPU device if
        available (useful for testing or when the GPU driver is unstable).
    """
    cpuinfo = _read_cpuinfo()
    board, board_name = _identify_board(cpuinfo)
    logger.info("Board detected: %s (%s)", board_name, board.name)

    devices = _enumerate_opencl_devices()
    for dev in devices:
        logger.info("OpenCL device found: %s", dev)

    if force_cpu_fallback:
        cpu_devices = [d for d in devices if d.device_type == "CPU"]
        chosen = cpu_devices[0] if cpu_devices else None
        logger.info("CPU fallback forced – using: %s", chosen)
    else:
        chosen = _pick_best_device(devices, board)
        logger.info("Selected OpenCL device: %s", chosen)

    tuning = _BOARD_TUNING.get(board, _BOARD_TUNING[BoardType.UNKNOWN])

    # Clamp work-group to device maximum
    if chosen is not None:
        wg = min(tuning["preferred_work_group"], chosen.max_work_group)
    else:
        wg = tuning["preferred_work_group"]

    has_gpu = (chosen is not None and chosen.device_type == "GPU") or (
        chosen is not None and not force_cpu_fallback
    )

    return HardwareProfile(
        board                = board,
        board_name           = board_name,
        opencl_device        = chosen,
        preferred_work_group = wg,
        max_wavelet_scales   = tuning["max_wavelet_scales"],
        signal_buffer_limit  = tuning["signal_buffer_limit"],
        use_float16          = tuning["use_float16"],
        has_gpu              = has_gpu,
    )
