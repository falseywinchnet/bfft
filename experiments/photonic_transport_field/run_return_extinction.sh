#!/bin/sh
# Authoritative local source; compile and measure on the selected M4 host.
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/return_extinction_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native \
    "$native/return_extinction_probe.cpp" "$native/retained_transport.cpp" \
    /tmp/return_extinction_conv.o -pthread -o /tmp/return_extinction_probe
  /tmp/return_extinction_probe > /tmp/return_extinction_tests.log
  /tmp/return_extinction_probe /tmp/return_extinction_benchmark.json
'
mkdir -p experiments/photonic_transport_field/return_extinction_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/return_extinction_benchmark.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/return_extinction_tests.log" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/return_*.ppm" \
  experiments/photonic_transport_field/return_extinction_m4/
