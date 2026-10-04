#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p /tmp/conv_fast_aa_build
xcrun clang++ -std=c++17 -O3 -fobjc-arc gpu.mm -framework Foundation -framework Metal -o /tmp/conv_fast_aa_build/gpu
xcrun clang++ -std=c++17 -O3 -fobjc-arc geometry_gpu.mm -framework Foundation -framework Metal -o /tmp/conv_fast_aa_build/geometry_gpu
xcrun clang++ -std=c++17 -O3 -fobjc-arc visibility_gpu.mm -framework Foundation -framework Metal -o /tmp/conv_fast_aa_build/visibility_gpu
