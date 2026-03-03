#!/usr/bin/env bash
# Launcher for the wavelet heartbeat monitor using the NumPy Morlet strategy.
#
# Differences from run_wavelet.sh (pywt strategy):
#   • Fixed complex Morlet wavelet (ω₀ = 6) – ignores --wavelet flag.
#   • Weighted-median + EMA temporal smoothing: displayed BPM is visibly
#     more stable once the signal is established (<±0.2 BPM jitter vs
#     ±0.5 BPM for the pywt strategy).
#   • get_filtered_signal() and get_fft_data() reuse the last compute_bpm()
#     energy vector – no extra CWT passes per frame.
#   • SpO₂ is NOT computed (always shows 0 %).
#
# Pass extra arguments directly to main_wavelet.py, e.g.:
#   ./run_wavelet_numpy.sh --headless
#   ./run_wavelet_numpy.sh --window 15 --bands 8
#   ./run_wavelet_numpy.sh --fps 25 --resolution 320x240

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

"$PYTHON" main_wavelet.py --strategy numpy "$@"
