#!/usr/bin/env bash
# Launch the IQ waterfall viewer with the correct interpreter, from anywhere.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Build the native lib if it isn't there yet.
if [ ! -f "$HERE/libiqwaterfall.dylib" ] && [ ! -f "$HERE/libiqwaterfall.so" ]; then
    "$HERE/build.sh"
fi

# Prefer the project's .venv python (has dearpygui/numpy); fall back to python3.
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi

exec "$PY" "$HERE/iq_waterfall_app.py" "$@"
