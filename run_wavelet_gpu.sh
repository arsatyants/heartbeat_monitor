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
# Point Qt5 at system fonts – pip opencv-python ships no fonts of its own
export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu

# Release any stale camera lock left by a previous crashed run
fuser -k /dev/media0 /dev/media1 2>/dev/null || true
sleep 0.3

# Run python as a child so the trap can kill it on Ctrl-C / TERM / EXIT
trap 'kill "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; exit' INT TERM EXIT
python main_wavelet_gpu.py "$@" &
CHILD=$!
wait "$CHILD"
