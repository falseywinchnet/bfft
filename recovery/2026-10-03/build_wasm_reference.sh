#!/bin/sh
# Recovery-only reproduction of the standalone reference from retained sources.
# Run at repository root on the Neo, where the existing WASI SDK is available.
set -eu
sdk=${WASI_SDK_PATH:-$HOME/.cache/wasi-sdk/wasi-sdk-33.0-arm64-macos}
source_dir=experiments/conv_warp/compact_measurement
destination=${1:-/tmp/bfft-recovery-reference.wasm}
"$sdk/bin/clang" --target=wasm32-wasip1 -O3 -msimd128 -ffp-contract=off \
  -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export-memory \
  -Wl,--initial-memory=131072 -Wl,--max-memory=4294967296 -Wl,--strip-all \
  -include "$source_dir/reference_shell.h" "$source_dir/reference_source.c" \
  -o "$destination"
