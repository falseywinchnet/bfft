#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 native=experiments/photonic_transport_field/native
 python3 experiments/photonic_transport_field/prepare_direct_spectral.py /tmp/direct_spectral_scene.cpp
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/direct_spectral_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native -I "$native" "$native/direct_spectral_probe.cpp" "$native/retained_transport.cpp" /tmp/direct_spectral_conv.o -pthread -o /tmp/direct_spectral_probe
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native -I "$native" "$native/direct_spectral_ray_probe.cpp" "$native/retained_transport.cpp" /tmp/direct_spectral_conv.o -pthread -o /tmp/direct_spectral_ray_probe
'
output=experiments/photonic_transport_field/direct_spectral_m4
mkdir -p "$output"
for scene in aperture-canyon standard mirror-relay occlusion-garden; do
 stem=$scene
 if [ "$scene" = aperture-canyon ]; then stem=aperture; fi
 ssh "$(/Users/ultimussecundai/.local/bin/m4host)" "/tmp/direct_spectral_probe /tmp/direct_spectral_${stem}.json $scene"
 scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/direct_spectral_${stem}.json*" "$output/"
done
ssh "$(/Users/ultimussecundai/.local/bin/m4host)" '/tmp/direct_spectral_probe /tmp/direct_spectral_aperture256.json aperture-canyon 256'
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/direct_spectral_aperture256.json*" "$output/"
ssh "$(/Users/ultimussecundai/.local/bin/m4host)" '/tmp/direct_spectral_ray_probe /tmp/direct_spectral_rays.json'
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/direct_spectral_rays.json" "$output/"
