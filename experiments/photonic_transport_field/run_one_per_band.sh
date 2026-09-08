#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 native=experiments/photonic_transport_field/native
 python3 experiments/photonic_transport_field/prepare_direct_spectral.py /tmp/direct_spectral_scene.cpp
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/direct_spectral_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native -I "$native" "$native/direct_spectral_probe.cpp" "$native/retained_transport.cpp" /tmp/direct_spectral_conv.o -pthread -o /tmp/direct_spectral_probe
'
output=experiments/photonic_transport_field/direct_spectral_m4
mkdir -p "$output"
for scene in aperture-canyon standard mirror-relay occlusion-garden; do
 ssh "$(/Users/ultimussecundai/.local/bin/m4host)" "/tmp/direct_spectral_probe /tmp/one_per_band_${scene}.json $scene 1 3"
 scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/one_per_band_${scene}.json*" "$output/"
done
