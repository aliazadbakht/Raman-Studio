#!/bin/bash
# Raman Studio — launcher (macOS / Linux)
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Point the optional FLIR Spinnaker SDK at its libraries. The variable differs
# per platform; main_qt.py does the same thing for anyone launching it directly.
case "$(uname -s)" in
    Darwin)
        [ -d /usr/local/lib ] && export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"
        [ -d /usr/local/lib/spinnaker-gentl ] && \
            export GENICAM_GENTL64_PATH="${GENICAM_GENTL64_PATH:-/usr/local/lib/spinnaker-gentl}"
        ;;
    Linux)
        [ -d /opt/spinnaker/lib ] && export LD_LIBRARY_PATH="/opt/spinnaker/lib:$LD_LIBRARY_PATH"
        [ -d /opt/spinnaker/lib/flir-gentl ] && \
            export GENICAM_GENTL64_PATH="${GENICAM_GENTL64_PATH:-/opt/spinnaker/lib/flir-gentl}"
        ;;
esac

# Interpreter: $RAMAN_PYTHON wins, then a Spinnaker-capable env, then the venv.
if [ -n "$RAMAN_PYTHON" ]; then
    exec "$RAMAN_PYTHON" main_qt.py
fi

FLIR_PYTHON="${RAMAN_FLIR_PYTHON:-/opt/anaconda3/envs/raman_flir/bin/python3}"
if [ -f "$FLIR_PYTHON" ] && "$FLIR_PYTHON" -c "import PySpin" >/dev/null 2>&1; then
    exec "$FLIR_PYTHON" main_qt.py
elif [ -f "venv/bin/python" ]; then
    exec "venv/bin/python" main_qt.py
else
    exec python3 main_qt.py
fi
