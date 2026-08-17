#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
OBS_SOURCE_DIR=${OBS_SOURCE_DIR:-/tmp/obs-studio-32.2.1}
OBS_CONFIG_INCLUDE_DIR=${OBS_CONFIG_INCLUDE_DIR:-/tmp}
SIMDE_INCLUDE_DIR=${SIMDE_INCLUDE_DIR:-/tmp/simde-0.8.2}
OBS_APP=${OBS_APP:-/Applications/OBS.app}
RVFX_SMOKE_WIDTH=${RVFX_SMOKE_WIDTH:-640}
RVFX_SMOKE_HEIGHT=${RVFX_SMOKE_HEIGHT:-360}
RVFX_CXX=${RVFX_CXX:-c++}
BUILD_DIR=${RVFX_OBS_BUILD_DIR:-$PROJECT_DIR/build/obs-macos}
BUNDLE_DIR=$PROJECT_DIR/dist/realtime-vector-fx.plugin
FRAMEWORK=$OBS_APP/Contents/Frameworks/libobs.framework/libobs

test -f "$OBS_SOURCE_DIR/libobs/obs-module.h" || {
    echo "OBS_SOURCE_DIR must point to matching OBS source headers" >&2
    exit 2
}
test -f "$OBS_CONFIG_INCLUDE_DIR/obsconfig.h" || {
    echo "OBS_CONFIG_INCLUDE_DIR must contain the generated obsconfig.h" >&2
    exit 2
}
test -d "$SIMDE_INCLUDE_DIR/simde" || {
    echo "SIMDE_INCLUDE_DIR must contain simde/" >&2
    exit 2
}
test -f "$FRAMEWORK" || {
    echo "OBS_APP does not contain libobs.framework" >&2
    exit 2
}

mkdir -p "$BUILD_DIR" "$BUNDLE_DIR/Contents/MacOS" "$BUNDLE_DIR/Contents/Resources"

COMMON_FLAGS="-std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wno-gnu-anonymous-struct -Wno-nested-anon-types -fPIC"
INCLUDES="-I$PROJECT_DIR/include -I$OBS_CONFIG_INCLUDE_DIR -I$OBS_SOURCE_DIR/libobs -I$OBS_SOURCE_DIR/libobs/util -I$SIMDE_INCLUDE_DIR"

# shellcheck disable=SC2086
$RVFX_CXX $INCLUDES $COMMON_FLAGS -c "$PROJECT_DIR/src/engine.cpp" -o "$BUILD_DIR/engine.o"
# shellcheck disable=SC2086
$RVFX_CXX $INCLUDES $COMMON_FLAGS -c "$PROJECT_DIR/obs/plugin-main.cpp" -o "$BUILD_DIR/plugin-main.o"
# shellcheck disable=SC2086
$RVFX_CXX $INCLUDES $COMMON_FLAGS -c "$PROJECT_DIR/obs/gpu-filter.cpp" -o "$BUILD_DIR/gpu-filter.o"

$RVFX_CXX -bundle "$BUILD_DIR/engine.o" "$BUILD_DIR/plugin-main.o" "$BUILD_DIR/gpu-filter.o" \
    "$FRAMEWORK" -Wl,-rpath,"$OBS_APP/Contents/Frameworks" \
    -o "$BUNDLE_DIR/Contents/MacOS/realtime-vector-fx"
sed 's/@PROJECT_VERSION@/0.1.0/g' "$PROJECT_DIR/obs/Info.plist.in" > "$BUNDLE_DIR/Contents/Info.plist"
/usr/bin/codesign --force --sign - "$BUNDLE_DIR"
/usr/bin/codesign --verify --strict --verbose=2 "$BUNDLE_DIR"

if test "${RVFX_RUN_OBS_SMOKE:-0}" = 1; then
    clang++ -x objective-c++ -std=c++17 -Wall -Wextra -Wpedantic \
        -Wno-gnu-anonymous-struct -Wno-nested-anon-types \
        -DRVFX_SMOKE_WIDTH="$RVFX_SMOKE_WIDTH" -DRVFX_SMOKE_HEIGHT="$RVFX_SMOKE_HEIGHT" \
        -I"$OBS_CONFIG_INCLUDE_DIR" -I"$OBS_SOURCE_DIR/libobs" \
        -I"$OBS_SOURCE_DIR/libobs/util" -I"$SIMDE_INCLUDE_DIR" \
        -c "$PROJECT_DIR/tools/obs_smoke.mm" -o "$BUILD_DIR/obs-smoke.o"
    clang++ "$BUILD_DIR/obs-smoke.o" "$FRAMEWORK" -framework AppKit \
        -Wl,-rpath,"$OBS_APP/Contents/Frameworks" -o "$BUILD_DIR/obs-smoke"
    "$BUILD_DIR/obs-smoke" "$BUNDLE_DIR/Contents/MacOS/realtime-vector-fx" \
        "$OBS_APP/Contents/Frameworks/libobs-metal.dylib" "$BUILD_DIR/obs-smoke.ppm"
fi

echo "$BUNDLE_DIR"
