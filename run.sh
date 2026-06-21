#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Ensure Spinnaker libraries and GenICam producers are found
export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"
export GENICAM_GENTL64_PATH="/usr/local/lib/spinnaker-gentl"

if [ -n "$RAMAN_PYTHON" ]; then
    exec "$RAMAN_PYTHON" main_qt.py
fi

# Prefer the local venv for normal dependencies, unless the known FLIR env exists.
FLIR_PYTHON="/opt/anaconda3/envs/raman_flir/bin/python3"
if [ -f "$FLIR_PYTHON" ] && "$FLIR_PYTHON" -c "import PySpin" >/dev/null 2>&1; then
    exec "$FLIR_PYTHON" main_qt.py
elif [ -f "venv/bin/python" ]; then
    exec "venv/bin/python" main_qt.py
else
    exec python3 main_qt.py
fi
