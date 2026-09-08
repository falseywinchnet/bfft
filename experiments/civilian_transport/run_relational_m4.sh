#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.civilian_transport.test_relational_kalman -v &&
   python3 -m experiments.civilian_transport.relational_study \
   --seeds 6 --out /tmp/civilian_relational_kalman > /tmp/civilian_relational_kalman.log 2>&1'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/relational_results
rsync -az "$civilian_host:/tmp/civilian_relational_kalman/" experiments/civilian_transport/relational_results/
scp "$civilian_host:/tmp/civilian_relational_kalman.log" experiments/civilian_transport/relational_results/run.log
