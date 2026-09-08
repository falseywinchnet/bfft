#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/observer_registration_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native \
    "$native/observer_registration_probe.cpp" "$native/retained_transport.cpp" \
    /tmp/observer_registration_conv.o -pthread -o /tmp/observer_registration_probe
  /tmp/observer_registration_probe /tmp/observer_registration_screen.json > /tmp/observer_registration_tests.log
'
mkdir -p experiments/photonic_transport_field/observer_registration_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/observer_registration_screen.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/observer_registration_tests.log" \
  experiments/photonic_transport_field/observer_registration_m4/
