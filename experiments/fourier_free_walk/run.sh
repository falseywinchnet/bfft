#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 mkdir -p /tmp/fourier_free_walk_final
 python3 -m unittest discover -s experiments/fourier_free_walk -p "test_*.py" -v > /tmp/fourier_free_walk_final/tests.txt 2>&1
 python3 experiments/fourier_free_walk/run.py --sizes 4,6,8,12,16 --seeds 3 --out /tmp/fourier_free_walk_final/initial
 python3 experiments/fourier_free_walk/run_turnover.py --sizes 4,6,8,10,12,16,24,32 --seeds 3 --steps 12000 --out /tmp/fourier_free_walk_final/turnovers
 python3 experiments/fourier_free_walk/run_gauges.py /tmp/fourier_free_walk_final/turnovers --out /tmp/fourier_free_walk_final/gauges
 python3 experiments/fourier_free_walk/run_joint.py --sizes 4,6,8,12,16 --seeds 3 --steps 4000 --out /tmp/fourier_free_walk_final/joint
 python3 experiments/fourier_free_walk/run_joint.py --excursions --sizes 4,6,8,12,16 --seeds 3 --steps 4000 --out /tmp/fourier_free_walk_final/continuous
 python3 experiments/fourier_free_walk/projection_rank.py --sizes 4,8,16,32,64 --out /tmp/fourier_free_walk_final/projection_rank.json
 python3 experiments/fourier_free_walk/certificate.py --plan /tmp/fourier_free_walk_final/joint/n8_seed0.json --out /tmp/fourier_free_walk_final/certificate8.json
 python3 experiments/fourier_free_walk/exact8.py 1 2 3 4 5 6 7 8 > /tmp/fourier_free_walk_final/example.json
'
fourier_host="$(/Users/ultimussecundai/.local/bin/m4host)"
mkdir -p experiments/fourier_free_walk/reproduced
scp -r "$fourier_host:/tmp/fourier_free_walk_final/." experiments/fourier_free_walk/reproduced/
