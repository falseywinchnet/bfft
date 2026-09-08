#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.civilian_transport.test_adaptive_acquisition \
       experiments.civilian_transport.test_relational_kalman -v &&
   python3 -m experiments.civilian_transport.acquisition_study \
       --seeds 8 --out /tmp/civilian_adaptive_acquisition > /tmp/civilian_adaptive_acquisition.log 2>&1'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/acquisition_results
rsync -az "$civilian_host:/tmp/civilian_adaptive_acquisition/" experiments/civilian_transport/acquisition_results/
scp "$civilian_host:/tmp/civilian_adaptive_acquisition.log" experiments/civilian_transport/acquisition_results/run.log
