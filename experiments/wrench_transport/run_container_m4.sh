#!/bin/sh
# Runs the container benchmark on the M4 Mini for this engine, MuJoCo and
# Rapier, scores every result with container_report.mjs, and copies the
# records back into results/container/. Usage: run_container_m4.sh [seed]
set -eu

SEED="${1:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
M4BUILD=/Users/ultimussecundai/.local/bin/m4build
HOST="$(/Users/ultimussecundai/.local/bin/m4host)"
NODE=/opt/homebrew/bin/node
OUT="/tmp/wrench_container_s${SEED}"
EXTERNAL=/Users/joshuahkuttenkuler/Developer/CodexBuilds/wrench_external
DIR=experiments/wrench_transport

"$M4BUILD" -- sh -c "
  rm -rf $OUT && mkdir -p $OUT &&
  $NODE $DIR/container.mjs --seed $SEED --export $OUT/scene.json --run wrench --hz 60 --out $OUT/wrench_60.json &&
  $NODE $DIR/container.mjs --seed $SEED --run wrench --hz 120 --out $OUT/wrench_120.json &&
  $NODE $DIR/container.mjs --seed $SEED --run soft_step_best --hz 60 --out $OUT/soft_step_best_60.json &&
  $NODE $DIR/container.mjs --seed $SEED --run soft_step_16 --hz 60 --out $OUT/soft_step_16_60.json &&
  python3 $DIR/external/run_mujoco.py $OUT/scene.json $OUT/mujoco_480_default.json --steps-per-sample 8 &&
  python3 $DIR/external/run_mujoco.py $OUT/scene.json $OUT/mujoco_480_multi.json --steps-per-sample 8 --multiccd &&
  python3 $DIR/external/run_mujoco.py $OUT/scene.json $OUT/mujoco_480_stiff.json --steps-per-sample 8 --multiccd --solref 0.005 &&
  python3 $DIR/external/run_mujoco.py $OUT/scene.json $OUT/mujoco_120_multi.json --steps-per-sample 2 --multiccd &&
  $NODE $DIR/external/run_rapier.mjs $OUT/scene.json $OUT/rapier_60_default.json --steps-per-sample 1 --iterations 4 --modules $EXTERNAL &&
  $NODE $DIR/external/run_rapier.mjs $OUT/scene.json $OUT/rapier_60_scaled.json --steps-per-sample 1 --iterations 4 --length-unit 0.05 --modules $EXTERNAL &&
  $NODE $DIR/external/run_rapier.mjs $OUT/scene.json $OUT/rapier_240_scaled.json --steps-per-sample 4 --iterations 8 --length-unit 0.05 --modules $EXTERNAL &&
  $NODE $DIR/container_report.mjs --seed $SEED --out $OUT/report.json \
     $OUT/wrench_60.json $OUT/wrench_120.json $OUT/soft_step_best_60.json $OUT/soft_step_16_60.json \
     $OUT/mujoco_480_default.json $OUT/mujoco_480_multi.json $OUT/mujoco_480_stiff.json $OUT/mujoco_120_multi.json \
     $OUT/rapier_60_default.json $OUT/rapier_60_scaled.json $OUT/rapier_240_scaled.json \
     > $OUT/report.md && cat $OUT/report.md
"
mkdir -p "$HERE/results/container/seed${SEED}"
scp -q "$HOST:$OUT/*" "$HERE/results/container/seed${SEED}/"
echo "copied to $HERE/results/container/seed${SEED}"
