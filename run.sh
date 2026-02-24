#!/usr/bin/env bash
# Quick-start launcher

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

python main.py "$@"
