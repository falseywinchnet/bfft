#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 mkdir -p /tmp/conifer_projectors
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/conifer_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native experiments/photonic_transport_field/conifer_projectors/conifer_demo.cpp experiments/photonic_transport_field/native/retained_transport.cpp /tmp/conifer_conv.o -pthread -o /tmp/conifer_demo
 /tmp/conifer_demo test > /tmp/conifer_projectors/tests.json
 /tmp/conifer_demo bake /tmp/conifer_projectors 1280
 /tmp/conifer_demo play /tmp/conifer_projectors/conifer.sheet /tmp/conifer_projectors
'
conifer_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/photonic_transport_field/conifer_projectors/output
scp -r "$conifer_host:/tmp/conifer_projectors/." experiments/photonic_transport_field/conifer_projectors/output/
