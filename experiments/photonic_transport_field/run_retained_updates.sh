#!/bin/sh
# Authoritative sources remain on the MacBook; compile and execute on the Mini.
# Executables live outside the shared mirror so another task's sync cannot
# replace them with a stale local binary between build and validation.
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_updates_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -mcpu=native \
    -fPIC -fvisibility=hidden -dynamiclib "$native/retained_transport.cpp" \
    -o /tmp/lib_retained_updates.dylib
  c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -mcpu=native \
    "$native/retained_transport_test.cpp" "$native/retained_transport.cpp" -o /tmp/retained_updates_kernel_test
  c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -mcpu=native \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_updates_conv.o -pthread -o /tmp/retained_scene_updates_native
  /tmp/retained_updates_kernel_test
  /tmp/retained_scene_updates_native --self-test
  /tmp/retained_scene_updates_native --update-self-test
  PFT_RETAINED_LIBRARY=/tmp/lib_retained_updates.dylib python3 -m unittest discover \
    -s experiments/photonic_transport_field -t . -p "test_*.py" -q
  /tmp/retained_scene_updates_native --benchmark-updates /tmp/retained_updates_m4.json --width 96 --height 64
  /tmp/retained_scene_updates_native --benchmark-updates /tmp/retained_updates_800x600_m4.json --width 800 --height 600
  /tmp/retained_scene_updates_native --benchmark-geometry-updates /tmp/retained_geometry_m4.json
  cc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -pthread -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_updates_conv_sanitized.o
  c++ -O1 -g -std=c++20 -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_updates_conv_sanitized.o -pthread -o /tmp/retained_updates_sanitized
  /tmp/retained_updates_sanitized --update-self-test
'
mkdir -p experiments/photonic_transport_field/retained_updates_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/retained_updates_m4.json*" \
  experiments/photonic_transport_field/retained_updates_m4/
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/retained_geometry_m4.json" \
  experiments/photonic_transport_field/retained_updates_m4/

scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/retained_updates_800x600_m4.json*" \
  experiments/photonic_transport_field/retained_updates_m4/
