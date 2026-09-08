#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/pixel_redundancy_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native \
    "$native/pixel_redundancy_probe.cpp" "$native/retained_transport.cpp" \
    /tmp/pixel_redundancy_conv.o -pthread -o /tmp/pixel_redundancy_probe
  /tmp/pixel_redundancy_probe /tmp/pixel_redundancy.json
'
mkdir -p experiments/photonic_transport_field/pixel_redundancy_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/pixel_redundancy.json" \
  experiments/photonic_transport_field/pixel_redundancy_m4/
