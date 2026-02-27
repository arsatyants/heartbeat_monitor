#!/usr/bin/env bash
# Quick-start launcher

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# Use X11/XWayland for OpenCV imshow (bundled Qt5 has no Wayland plugin)
export QT_QPA_PLATFORM=xcb

# Kill any leftover camera process on exit so /dev/media0 is always released
trap 'kill $(jobs -p) 2>/dev/null; exit' INT TERM EXIT

python main.py "$@"
