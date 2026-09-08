#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
c++ -O3 -DNDEBUG -std=c++17 -fPIC -shared -Iinclude \
 experiments/meyer_transport_audit/native.cpp -Lbuild -lbfft \
 -Wl,-rpath,$PWD/build -o /tmp/meyer_audit_settle.dylib
python3 -m unittest experiments.meyer_transport_audit.test_audit -v
python3 -c "from experiments.meyer_transport_audit.native_audit import test,library; test(library(\"/tmp/meyer_audit_settle.dylib\"))"
python3 experiments/meyer_transport_audit/validate.py --size 256 --threads 1 \
 --repeats 15 --mode 3 --lib /tmp/meyer_audit_settle.dylib \
 --out /tmp/meyer_fused_validate256.json
python3 experiments/meyer_transport_audit/stress.py \
 --lib /tmp/meyer_audit_settle.dylib --out /tmp/meyer_audit_stress.json
python3 experiments/meyer_transport_audit/settle_probe.py \
 --arrays /tmp/meyer_fused_validate256.npz --out /tmp/meyer_settle_probe.json
'
remote_host=$(/Users/ultimussecundai/.local/bin/m4host)
for artifact in meyer_fused_validate256.json meyer_fused_validate256.npz meyer_audit_stress.json meyer_settle_probe.json; do
 scp "$remote_host:/tmp/$artifact" experiments/meyer_transport_audit/
done
.venv-jpeg/bin/python experiments/meyer_transport_audit/plot_results.py
