#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$project_dir/.venv/bin/tlvector-gui" ]; then
    exec "$project_dir/.venv/bin/tlvector-gui"
fi

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m tlvector.web_gui
