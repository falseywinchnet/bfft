#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.projectile_transport.test_core -v &&
   python3 -m experiments.projectile_transport.study \
     --seeds 8 --start-seed 100 --out /tmp/projectile_transport_first'
projectile_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/projectile_transport/results
rsync -az "$projectile_host:/tmp/projectile_transport_first/" \
  experiments/projectile_transport/results/
