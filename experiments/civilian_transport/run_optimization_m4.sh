#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'make -C experiments/civilian_transport/native libmarkov.so libcv_control.so &&
   python3 -m unittest experiments.civilian_transport.test_relational_markov \
       experiments.civilian_transport.test_relational_kalman \
       experiments.civilian_transport.test_adaptive_acquisition -v &&
   python3 -m experiments.civilian_transport.optimization_study \
       --out /tmp/civilian_exact_optimization --seeds 8 --repeats 31 \
       > /tmp/civilian_exact_optimization.log 2>&1 &&
   python3 -m experiments.civilian_transport.cost_controls \
       /tmp/civilian_cv_cost_controls.json'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/optimization_results
rsync -az "$civilian_host:/tmp/civilian_exact_optimization/" experiments/civilian_transport/optimization_results/
scp "$civilian_host:/tmp/civilian_exact_optimization.log" experiments/civilian_transport/optimization_results/run.log
scp "$civilian_host:/tmp/civilian_cv_cost_controls.json" experiments/civilian_transport/optimization_results/cv_cost_controls.json
