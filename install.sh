#!/bin/bash
# Raman Spectrum Analyzer – macOS installer
set -e
echo "=== Raman Spectrum Analyzer – macOS Setup ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install from https://python.org"
    exit 1
fi

PY=$(python3 --version)
echo "Using: $PY"

# Create virtual environment
VENV="venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "Created virtual environment."
fi

source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "=== Installation complete ==="
echo "Run the app with:  ./run.sh"
echo ""
echo "Camera note: FLIR/PointGrey capture requires the Spinnaker SDK and PySpin"
echo "installed into the Python used by run.sh. File viewing/export works without PySpin."
