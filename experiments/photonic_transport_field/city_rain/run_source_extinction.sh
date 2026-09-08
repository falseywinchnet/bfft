#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 root=experiments/photonic_transport_field
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/night_screen_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native "$root/city_rain/source_extinction_study.cpp" "$root/native/retained_transport.cpp" /tmp/night_screen_conv.o -pthread -o /tmp/source_extinction_study
 /tmp/source_extinction_study > /tmp/source_extinction_study.json
 /tmp/source_extinction_study --native-regressions > /tmp/source_extinction_regressions.log
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native "$root/city_rain/night_screen.cpp" "$root/native/retained_transport.cpp" /tmp/night_screen_conv.o -pthread -o /tmp/night_screen_native
 /tmp/night_screen_native test
 /tmp/night_screen_native bake /tmp/night_screen_extinction
'
compute_host=$(/Users/ultimussecundai/.local/bin/m4host)
output=experiments/photonic_transport_field/city_rain/output/source_extinction
mkdir -p "$output"
scp "$compute_host:/tmp/source_extinction_study.json" "$compute_host:/tmp/source_extinction_regressions.log" "$compute_host:/tmp/night_screen_extinction/bake.json" "$compute_host:/tmp/night_screen_extinction/illumination.screen" "$compute_host:/tmp/night_screen_extinction/night_clear.ppm" "$output/"
