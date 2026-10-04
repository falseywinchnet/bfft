# Hollow-object packing and collision accuracy

A reproducible reconstruction of the task in Lucky Iyinbor's
[collision-accuracy video](https://x.com/Luckyballa/status/2104797946281894190),
plus a fixed-configuration 32–512-object scaling experiment. It tests the
standalone C++ Wrench engine against Rapier 0.21.0 and MuJoCo 3.2.3. This is a
new workload; it does not replace the older 64-convex-body experiment.

* [Protocol and source boundaries](PROTOCOL.md)
* [Measured results](RESULTS.md)
* [Interactive replay](viewer/index.html), with engine, size and seed selection
* [Machine-readable summary](viewer/summary.json)
* [Receipt and determinism audit](audits.json)
* `results/countN/`: common scenes, all warmup/measured outputs, process logs,
  independent geometric scores, failure/timeout records, SHA-256 manifest
* `results/controls/`: free fall, open-cavity and thin-slab checks, with poses
  exported at every outer simulation step

## Run

Build on the Mini using the authoritative local tree:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  '/opt/homebrew/bin/cmake -S experiments/wrench_transport/packing -B /tmp/wrench_packing -DCMAKE_BUILD_TYPE=Release && \
   /opt/homebrew/bin/cmake --build /tmp/wrench_packing -j4 && \
   /opt/homebrew/bin/ctest --test-dir /tmp/wrench_packing/wrench --output-on-failure'
python3 experiments/wrench_transport/packing/run_m4.py --counts 32,64,128,256,512 --sync
```

The Mini already has MuJoCo and the Rapier JS package used by the previous
study. No new software installation is required. The wrapper runs engines
sequentially, copies each completed size home, verifies every receipt hash,
and only then removes that run's temporary remote result directory. Each run
has a fixed 180-second process budget; scoring has a separate budget. A failed
scene/configuration is not retried during its measured repetitions.

To regenerate the report and inspect it locally:

```sh
python3 experiments/wrench_transport/packing/validate_results.py
node experiments/wrench_transport/packing/test_score.mjs
/Users/ultimussecundai/bfft/.venv-jpeg/bin/python experiments/wrench_transport/packing/report.py
python3 -m http.server 62839 --bind 127.0.0.1 --directory experiments/wrench_transport/packing
```

Open `http://127.0.0.1:62839/viewer/`. The browser reads compressed recorded
poses; its drawing speed is not a physics benchmark. The plot is also saved
as `viewer/scaling.svg` for reuse without rasterization.

## What is measured

Geometry is shared at the convex-part level: no engine may fill a cup or bowl
with one outer convex hull. Density and friction are shared; MuJoCo receives
explicit common mass/inertia and the other two cook the same component
geometry. Measured mass agreement is checked independently. All bodies stay
awake and all step calls advance a fixed amount of simulated time.

The main scaling cost is the eight-second interval after the final release,
when every engine has all bodies active. MuJoCo preallocates its future bodies
with collision disabled and gravity compensation until release; Rapier and
Wrench add bodies on release. Activation and initialization are outside the
step timer. This distinction is why release-phase cost is retained separately.

An independent separating-axis implementation checks every face normal and
every pair of edge directions for each candidate convex-part pair, using only
bounding boxes to reject distant pairs. It samples the pile at 10 Hz. Final
error, sampled motion error, escape counts and sustained motion are separate
outcomes. Small controlled impact tests export every outer integration step.

A successful process is not a physically successful run. Escapes, numerical
warnings, nonconverged native solves, missed slab collisions and unsettled
states remain in the outputs. A fast simulation that has lost its pile does
not establish efficient accurate packing. The report's plots mark escapes or
reported numerical instability explicitly and omit incomplete timing groups.

The native engine's source and solver settings are unchanged by this study.
`native.cpp` is a benchmark client of the public library API, built by the
small CMake project in this directory. It does not replace engine internals.
