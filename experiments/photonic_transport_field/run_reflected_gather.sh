#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 native=experiments/photonic_transport_field/native
 python3 experiments/photonic_transport_field/prepare_direct_spectral.py /tmp/direct_spectral_scene.cpp
 python3 experiments/photonic_transport_field/prepare_reflected_gather.py /tmp/reflected_gather_scene.cpp
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/reflected_gather_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native -I "$native" "$native/reflected_gather_probe.cpp" "$native/retained_transport.cpp" /tmp/reflected_gather_conv.o -pthread -o /tmp/reflected_gather_probe
'
output=experiments/photonic_transport_field/reflected_gather_m4
mkdir -p "$output"
for scene in mirror-relay aperture-canyon standard occlusion-garden; do
 ssh "$(/Users/ultimussecundai/.local/bin/m4host)" "/tmp/reflected_gather_probe /tmp/reflected_gather_${scene}.json $scene"
 scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/reflected_gather_${scene}.json*" "$output/"
done
