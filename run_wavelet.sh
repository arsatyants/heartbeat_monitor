#!/usr/bin/env bash
# Quick-start launcher for wavelet-based heartbeat monitor

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick Python: prefer venv if present, otherwise system python3
if [[ -x .venv/bin/python ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

# Use X11/XWayland for OpenCV imshow (bundled Qt5 has no Wayland plugin)
export QT_QPA_PLATFORM=xcb
# Point Qt5 at system fonts
export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu

# Kill any leftover camera process on exit so /dev/media0 is always released
trap 'kill $(jobs -p) 2>/dev/null; exit' INT TERM EXIT

"$PYTHON" main_wavelet.py "$@"
