#!/usr/bin/env bash
# Build the monolithic IQ-waterfall backend library.
set -e
cd "$(dirname "$0")"

ROOT=..
INC=$ROOT/include
BFFT_LIB=$ROOT/build/libbfft.a

if [ ! -f "$BFFT_LIB" ]; then
    echo "libbfft.a not found; building it first..."
    (cd $ROOT && make lib)
fi

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
