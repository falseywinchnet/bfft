#!/bin/sh
# Compile authoritative sources on the selected M4 CPU and retain /tmp receipts.
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_boundary_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -mcpu=native \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_boundary_conv.o -pthread -o /tmp/retained_boundary_native
  /tmp/retained_boundary_native --boundary-self-test
  /tmp/retained_boundary_native --self-test
  /tmp/retained_boundary_native --update-self-test
  /tmp/retained_boundary_native --camera-self-test
  /tmp/retained_boundary_native --benchmark-boundary /tmp/boundary_discovery_benchmark.json --width 800 --height 600
  /tmp/retained_boundary_native --benchmark-updates /tmp/boundary_discovery_updates.json --width 800 --height 600
  for scene in standard aperture-canyon mirror-relay; do
    /tmp/retained_boundary_native --width 800 --height 600 --scene "$scene" --boundary-audit \
      --boundary-filter independent --boundary-path-cache 0 --out /tmp/boundary_diagnostic.ppm \
      > "/tmp/boundary_baseline_${scene}.json" 2> "/tmp/boundary_baseline_${scene}_audit.json"
  done
  cc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -pthread -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_boundary_conv_sanitized.o
  c++ -O1 -g -std=c++20 -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_boundary_conv_sanitized.o -pthread -o /tmp/retained_boundary_sanitized
  /tmp/retained_boundary_sanitized --boundary-self-test
  cc -O1 -g -std=c11 -fsanitize=thread -pthread -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_boundary_conv_tsan.o
  c++ -O1 -g -std=c++20 -fsanitize=thread "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_boundary_conv_tsan.o -pthread -o /tmp/retained_boundary_tsan
  /tmp/retained_boundary_tsan --boundary-self-test
'
mkdir -p experiments/photonic_transport_field/boundary_discovery_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/boundary_discovery_*.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/boundary_baseline_*.json" \
  experiments/photonic_transport_field/boundary_discovery_m4/
