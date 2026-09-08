#!/bin/sh
# CPU source expansion diagnosis and exact support-elision validation.
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/source_expansion_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -mcpu=native \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/source_expansion_conv.o -pthread -o /tmp/source_expansion_native
  /tmp/source_expansion_native --source-self-test
  /tmp/source_expansion_native --self-test
  /tmp/source_expansion_native --update-self-test
  /tmp/source_expansion_native --camera-self-test
  /tmp/source_expansion_native --boundary-self-test
  /tmp/source_expansion_native --benchmark-source /tmp/source_expansion_benchmark.json --width 800 --height 600
  /tmp/source_expansion_native --width 800 --height 600 --scene aperture-canyon \
    --expansion-audit /tmp/source_expansion_optimized_audit.json --out /tmp/source_expansion_optimized.ppm \
    > /tmp/source_expansion_optimized_frame.json
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native \
    "$native/source_expansion_probe.cpp" "$native/retained_transport.cpp" \
    /tmp/source_expansion_conv.o -pthread -o /tmp/source_expansion_probe
  /tmp/source_expansion_probe 275 324 > /tmp/source_expansion_words.txt
'
mkdir -p experiments/photonic_transport_field/source_expansion_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/source_expansion_*.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/source_expansion_words.txt" \
  experiments/photonic_transport_field/source_expansion_m4/
