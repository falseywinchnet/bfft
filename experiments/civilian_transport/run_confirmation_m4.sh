#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
  'python3 -m unittest experiments.civilian_transport.test_transport experiments.civilian_transport.test_io -v &&
   mkdir -p /tmp/civilian_transport_confirmation_workspace/experiments /tmp/civilian_transport_confirmation &&
   cp -R experiments/civilian_transport /tmp/civilian_transport_confirmation_workspace/experiments/ &&
   cp experiments/civilian_transport/results/development.json /tmp/civilian_transport_confirmation/development.json &&
   cd /tmp/civilian_transport_confirmation_workspace &&
   python3 -m experiments.civilian_transport.study --stage test --seeds 8 \
     --particles 6144 --test-seed-base 20000 \
     --out /tmp/civilian_transport_confirmation > /tmp/civilian_transport_confirmation.log 2>&1'
civilian_host=$(/Users/ultimussecundai/.local/bin/m4host)
mkdir -p experiments/civilian_transport/confirmation
rsync -az "$civilian_host:/tmp/civilian_transport_confirmation/" \
  experiments/civilian_transport/confirmation/
scp "$civilian_host:/tmp/civilian_transport_confirmation.log" \
  experiments/civilian_transport/confirmation/run.log
