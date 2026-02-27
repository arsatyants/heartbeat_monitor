#!/usr/bin/env bash
# Launch the GPU-accelerated wavelet heartbeat monitor.
# Pass extra arguments directly to main_wavelet_gpu.py, e.g.:
#   ./run_wavelet_gpu.sh --headless
#   ./run_wavelet_gpu.sh --show-bands --resolution 320x240
#   ./run_wavelet_gpu.sh --cpu-fallback   # force CPU-only (no OpenCL)
#   ./run_wavelet_gpu.sh --info           # print hardware info and exit

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# Use X11/XWayland for OpenCV imshow (bundled Qt5 has no Wayland plugin)
export QT_QPA_PLATFORM=xcb

exec python main_wavelet_gpu.py "$@"
