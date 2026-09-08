#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  python3 -m unittest discover -s experiments/fourier_twos -p "test_*.py" -v > /tmp/fourier_twos_tests.txt 2>&1
  python3 experiments/fourier_twos/run_experiment.py --out /tmp/fourier_twos_results.json
'
fourier_twos_host="$(/Users/ultimussecundai/.local/bin/m4host)"
mkdir -p experiments/fourier_twos/m4_results
scp "$fourier_twos_host:/tmp/fourier_twos_tests.txt" \
    "$fourier_twos_host:/tmp/fourier_twos_results.json" \
    experiments/fourier_twos/m4_results/
python3 experiments/fourier_twos/report.py
