#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.civilian_transport.test_witness_geometry -v &&
   python3 -m experiments.civilian_transport.witness_study \
   --seeds 12 --out /tmp/civilian_witness_geometry > /tmp/civilian_witness_geometry.log 2>&1'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/witness_results
rsync -az "$civilian_host:/tmp/civilian_witness_geometry/" experiments/civilian_transport/witness_results/
scp "$civilian_host:/tmp/civilian_witness_geometry.log" experiments/civilian_transport/witness_results/run.log
