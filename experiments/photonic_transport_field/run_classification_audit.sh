#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  python3 experiments/photonic_transport_field/audit_classification.py /tmp/classification_audit.cpp
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/classification_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -mcpu=native -I "$native" /tmp/classification_audit.cpp "$native/retained_transport.cpp" /tmp/classification_conv.o -pthread -o /tmp/classification_audit
  /tmp/classification_audit > /tmp/classification_audit.json
'
mkdir -p experiments/photonic_transport_field/classification_audit_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/classification_audit.json" experiments/photonic_transport_field/classification_audit_m4/
