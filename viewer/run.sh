#!/usr/bin/env bash
# Launch the IQ waterfall viewer with the correct interpreter, from anywhere.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Build when absent or stale.  The Python wrapper and native functions share a
# C ABI; loading yesterday's dylib after changing an argument list corrupts the
# call frame instead of raising a Python exception.
case "$(uname -s)" in
    Darwin) LIB="$HERE/libiqwaterfall.dylib" ;;
    *)      LIB="$HERE/libiqwaterfall.so" ;;
esac
STALE=""
if [ -f "$LIB" ]; then
    STALE="$(find "$HERE" "$ROOT/include" "$ROOT/src" \
        -type f \( -name '*.cpp' -o -name '*.c' -o -name '*.h' \
        -o -name '*.hpp' \) -newer "$LIB" -print -quit)"
fi
if [ ! -f "$LIB" ] || [ -n "$STALE" ]; then
    "$HERE/build.sh"
fi

# Prefer the project's .venv python (has dearpygui/numpy); fall back to python3.
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi

exec "$PY" "$HERE/iq_waterfall_app.py" "$@"
