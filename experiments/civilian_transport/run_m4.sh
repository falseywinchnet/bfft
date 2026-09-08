#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.civilian_transport.test_transport experiments.civilian_transport.test_io -v &&
   mkdir -p /tmp/civilian_transport_workspace/experiments &&
   cp -R experiments/civilian_transport /tmp/civilian_transport_workspace/experiments/ &&
   cd /tmp/civilian_transport_workspace &&
   python3 -m experiments.civilian_transport.study --seeds 8 --particles 1536 \
     --out /tmp/civilian_transport_final > /tmp/civilian_transport_final.log 2>&1'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/results
rsync -az "$civilian_host:/tmp/civilian_transport_final/" \
  experiments/civilian_transport/results/
scp "$civilian_host:/tmp/civilian_transport_final.log" \
  experiments/civilian_transport/results/run.log
