#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
v1_dir=$(CDPATH= cd -- "$project_dir/../svg_converter" && pwd)
if [ -x "$v1_dir/.venv/bin/posterizer-gui" ]; then
    exec "$v1_dir/.venv/bin/posterizer-gui"
fi

export PYTHONPATH="$project_dir/src:$project_dir/../svg_converter_v2/src:$v1_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m posterizer.web_gui
