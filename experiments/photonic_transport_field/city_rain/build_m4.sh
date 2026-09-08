#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 root=experiments/photonic_transport_field
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/city_rain_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native "$root/city_rain/city_rain.cpp" "$root/native/retained_transport.cpp" /tmp/city_rain_conv.o -pthread -o /tmp/city_rain_native
'
