#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.civilian_transport.test_transport experiments.civilian_transport.test_io experiments.civilian_transport.test_certificates experiments.civilian_transport.test_degradation -v &&
   mkdir -p /tmp/civilian_transport_degradation_workspace/experiments /tmp/civilian_transport_degradation &&
   cp -R experiments/civilian_transport /tmp/civilian_transport_degradation_workspace/experiments/ &&
   cd /tmp/civilian_transport_degradation_workspace &&
   python3 -m experiments.civilian_transport.degradation --seeds 6 --particles 6144 --draws 1024 \
     --out /tmp/civilian_transport_degradation > /tmp/civilian_transport_degradation.log 2>&1'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/degradation_results
rsync -az "$civilian_host:/tmp/civilian_transport_degradation/" experiments/civilian_transport/degradation_results/
scp "$civilian_host:/tmp/civilian_transport_degradation.log" experiments/civilian_transport/degradation_results/run.log
