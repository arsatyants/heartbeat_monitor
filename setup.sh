#!/usr/bin/env bash
# First-time setup: creates a venv and installs dependencies.
# On Raspberry Pi OS picamera2 is installed system-wide; use --system-site-packages.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Pi
IS_PI=false
if grep -qi "raspberry" /proc/cpuinfo 2>/dev/null || command -v picamera2 &>/dev/null; then
    IS_PI=true
fi

echo "=== Heartbeat Monitor Setup ==="
echo "Raspberry Pi detected: $IS_PI"

if $IS_PI; then
    echo "Creating venv with --system-site-packages (for picamera2 / libcamera)..."
    python3 -m venv .venv --system-site-packages
else
    echo "Creating isolated venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Run the monitor with:"
echo "  ./run.sh"
echo "  ./run.sh --headless          # no display, log BPM to stdout"
echo "  ./run.sh --resolution 320x240 --fps 60"
