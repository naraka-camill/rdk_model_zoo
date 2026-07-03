#!/bin/bash

# Real-time ASR startup script (arecord version)
# Checks for arecord, then runs realtime_asr.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Checking dependencies ==="
if ! command -v arecord &>/dev/null; then
    echo "Error: arecord not found. Install it with:"
    echo "    sudo apt install alsa-utils"
    exit 1
fi
echo "  arecord found: $(arecord --version | head -1)"
echo ""

echo "=== Available recording devices ==="
arecord -l 2>/dev/null || echo "(no devices listed)"
echo ""

echo "=== Starting real-time ASR ==="
echo "Speak into your microphone. Press Ctrl+C to stop."
echo ""

python3 realtime_asr.py "$@"
