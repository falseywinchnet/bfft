#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
v1_dir=$(CDPATH= cd -- "$project_dir/../svg_converter" && pwd)
if [ -x "$v1_dir/.venv/bin/tlvector-v2-gui" ]; then
    exec "$v1_dir/.venv/bin/tlvector-v2-gui"
fi

export PYTHONPATH="$project_dir/src:$v1_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m tlvector_v2.web_gui
