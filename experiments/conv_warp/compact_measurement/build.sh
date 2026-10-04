#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sdk=${WASI_SDK_PATH:-$HOME/.cache/wasi-sdk/wasi-sdk-33.0-arm64-macos}
"$sdk/bin/clang" --target=wasm32-wasip1 -O3 -msimd128 -ffp-contract=off -fno-builtin -nostdlib \
  -Wl,--no-entry -Wl,--export-memory -Wl,--initial-memory=131072 \
  -Wl,--max-memory=4294967296 -Wl,--strip-all -o "$root/compact.wasm" "$root/compact.c"
