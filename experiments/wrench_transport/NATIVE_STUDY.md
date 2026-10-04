# Native C++ study: same trajectory, less work around the arithmetic

Date: 2026-10-03. Author/conceptual direction: Joshuah Rainstar.

## Baseline and retained change

Baseline commit: `fcdb0a3` (the current C++ game engine, not the older JavaScript).
`cpp-baseline.json` records SHA-256 hashes of its source files. The baseline already
includes compounds, pair caching, deterministic replay, and island sleep/wake fixes.

The retained optimization moves the existing short vector/quaternion definitions
from `hull.cpp` into `physics.hpp` as inline functions. This exposes scalar
operations to the compiler across source-file boundaries. It preserves the
expressions, ordering, collision rules, solver settings, and floating-point
contraction policy. The application does not enable fast-math or approximate reciprocals.

An attempted six-right-hand-side block solve was discarded: alone it raised
mean time from 0.8778 to 0.9030 ms. Combining it with inlining took 0.7398 ms,
versus 0.7184 ms for inlining alone. A performance-looking rewrite is not a win
until the measurements agree.

## Protocol

Apple M4 Mac mini, macOS 26.5 arm64, AppleClang 21.0.0.21000101, CMake Release,
C++20, `-O3 -DNDEBUG -ffp-contract=off`. No link-time optimization. The local
checkout is authoritative; compilation and simulation run through `m4build`.

Each scene contains 64 mixed convex dynamic bodies and static container walls.
Shapes include boxes, sphere approximations, cylinders, cones, thin sheets,
prisms, polyhedra, and irregular hulls. Seeds: 1, 7, 19, 101, 271, 314159,
8675309. Seed 1 was used during exploration; the other six entered the repeated study
without per-seed tuning.
Eight simulated seconds per run; 60 Hz; 480 steps; sleeping disabled; damping
and ground rolling resistance disabled as in the existing container comparison.

There is one warm-up run per binary/seed, then seven measured repetitions.
Four variants rotate execution order each repetition. The reported overall mean
averages the per-run mean step times (49 measured runs per variant). Timing
surrounds `World::step()` only: cooking, scene parsing, checksumming, output, and
rendering are excluded. The host was not a dedicated real-time benchmark machine;
variation and observed maxima are retained, not hidden. These are empirical
workload results, not timing guarantees or a general engine leaderboard.

| Variant | Overall mean step (ms) | Decision |
| --- | ---: | --- |
| Current C++ baseline | 0.8778 | Reference |
| Batched block solve | 0.9030 | Discard |
| Inlining + batched block solve | 0.7398 | Discard block rewrite |
| Inlining only | 0.7184 | Retain |

This is **18.2% less mean step time** (1.22x throughput). Moving frames average
3.0065 -> 2.7550 ms, an 8.4% reduction. Quiet but awake frames average
0.4018 -> 0.2631 ms, a 34.5% reduction. Those phase means pool the corresponding
frames. Mean per-run 95th percentile is 3.9900 -> 3.7691 ms. Observed worst steps
are 9.2517 and 9.0637 ms respectively; these are not worst-case bounds.

Per-seed median of the seven mean-step measurements:

| Seed | Baseline (ms) | Retained (ms) | Less time |
| --- | ---: | ---: | ---: |
| 1 | 0.8397 | 0.6876 | 18.1% |
| 7 | 0.7954 | 0.6647 | 16.4% |
| 19 | 0.9485 | 0.7618 | 19.7% |
| 101 | 0.7648 | 0.6267 | 18.1% |
| 271 | 0.9547 | 0.8139 | 14.7% |
| 314159 | 0.8528 | 0.7000 | 17.9% |
| 8675309 | 0.9860 | 0.7729 | 21.6% |

## Correctness and scope

Every measured and warm-up run matched the baseline's FNV-based aggregate of all
480 exact-state checksums, final poses, sampled speed/energy, floor load, and
reported Newton/factorization/pass counters. A matching finite checksum is strong
regression evidence, not a proof or cross-platform determinism guarantee.

Both CTest suites pass. The acceptance suite covers the original sixteen scenarios
plus static tables, restitution, rest reports, and stable sets. Independent new
checks compare 80 sparse SPD systems with dense Gaussian elimination (maximum
residual 2.398e-14; maximum solution difference 3.575e-14), and 6,000 convex pairs
with exhaustive face and edge SAT axes (zero missed overlaps, false contacts
beyond the margin, or invalid normals). The latter permits the pre-existing
5 mm lateral witness reach; observed maximum surface excursion was 0.7814 mm.
It does not establish exact surface witnesses or continuous collision detection.

The library handles convex hulls and compounds, not arbitrary deformable bodies
or arbitrary joints. There is no CCD; extreme impacts, thin geometry, and very
large graphs remain limitations. Broad phase and symbolic adjacency still use
quadratic structures. The finite outer iteration budget is not a convergence
proof. A native replay in the site exhibit displays recorded C++ results; it is
not a live browser simulation. The older JavaScript viewer is explicitly separate.

## Reproduction

Build the baseline commit in a separate checkout. Copy the current
`phys/tools/phys_bench.cpp` into its `phys/tools/` before compiling, so both
libraries use the same measurement/checksum harness. Build the retained version
in another build directory. On the Mini, run:

```sh
python3 experiments/wrench_transport/run_native_study.py \
  --node /opt/homebrew/bin/node \
  --binary baseline=/tmp/wrench_baseline/phys_bench \
  --binary inline=/tmp/wrench_current/phys_bench \
  --repeats 7 --output /tmp/wrench-study.json
```

Copy results back immediately. `results/native/study.json` preserves the four-variant
study, per-run times, trajectory hashes, counters, and process summaries.
`results/native/final-validation.txt` preserves acceptance, installation, and example
checks. The benchmark's optional fifth positional argument writes a native replay:

```sh
node experiments/wrench_transport/container.mjs --seed 1 --export-text /tmp/scene.txt
build/wrench/phys_bench /tmp/scene.txt /tmp/result.json 60 8 /tmp/replay.json
```

Historical MuJoCo/Rapier comparisons are retained under `results/container/`;
those configurations and earlier JavaScript/native snapshots are not used to
claim superiority of this optimization over other engines.
