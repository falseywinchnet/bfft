#!/usr/bin/env bash
# Build the monolithic IQ-waterfall backend library.
set -e
cd "$(dirname "$0")"

ROOT=..
INC=$ROOT/include
BFFT_BUILD=$ROOT/build-viewer-release
BFFT_LIB=$BFFT_BUILD/libbfft.a

# The intrinsic FCT is branch-and-bound heavy enough that accidentally linking
# a Debug libbfft makes the viewer appear unusable.  Keep a small, isolated
# Release build so this script cannot inherit the configuration of a developer
# test tree in ../build.
cmake -S "$ROOT" -B "$BFFT_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBFFT_BUILD_SHARED=OFF \
    -DBFFT_BUILD_EXAMPLES=OFF \
    -DBFFT_BUILD_TESTS=OFF \
    -DBFFT_BUILD_PROBES=OFF
cmake --build "$BFFT_BUILD" --target bfft_static --parallel

case "$(uname -s)" in
    Darwin) OUT=libiqwaterfall.dylib; FRAMEWORKS="-framework Accelerate" ;;
    *)      OUT=libiqwaterfall.so;    FRAMEWORKS="" ;;
esac

c++ -O3 -std=c++17 -fPIC -shared -Wall \
    -I"$INC" \
    iqwaterfall.cpp dip_algo.cpp \
    "$BFFT_LIB" \
    $FRAMEWORKS -lm \
    -o "$OUT"

echo "built $OUT"
