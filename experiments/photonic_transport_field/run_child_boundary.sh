#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/child_boundary_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native \
    "$native/child_boundary_probe.cpp" "$native/retained_transport.cpp" \
    /tmp/child_boundary_conv.o -pthread -o /tmp/child_boundary_probe
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/child_boundary_conv.o -pthread -o /tmp/child_boundary_native
  /tmp/child_boundary_native --self-test > /tmp/child_boundary_regressions.log
  /tmp/child_boundary_native --update-self-test >> /tmp/child_boundary_regressions.log
  /tmp/child_boundary_native --camera-self-test >> /tmp/child_boundary_regressions.log
  /tmp/child_boundary_native --boundary-self-test >> /tmp/child_boundary_regressions.log
  /tmp/child_boundary_native --source-self-test >> /tmp/child_boundary_regressions.log
  /usr/bin/time -l /tmp/child_boundary_probe /tmp/child_boundary_screen.json \
    > /tmp/child_boundary_tests.log 2> /tmp/child_boundary_resources.log
'
mkdir -p experiments/photonic_transport_field/child_boundary_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/child_boundary_screen.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/child_boundary_*.log" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/child_boundary_*.ppm" \
  experiments/photonic_transport_field/child_boundary_m4/
