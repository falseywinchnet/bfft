# Wrench transport: a rigid-body contact solver for rock stacking

A JavaScript prototype of a 3-D rigid-body engine for convex rocks (gravity,
unilateral contact, Coulomb friction, sleeping), written to the scope of the
Zen Construction specification (`~/solitaire_sounds/zen/PHYSICS_SPEC.md`) and
measured against that specification's own solver. No dependencies.

Measured results, the design's derivation, the approaches that were tried and
dropped, and the open limits are in [FINDINGS.md](FINDINGS.md). This file says
what the pieces are and how to run them.

## The idea in one paragraph

Contact is treated as transport of momentum through a retained field, after
the photonic transport engine in `experiments/photonic_transport_field/`. Each
contact point keeps the impulse it carried last frame. A frame runs a few
passes; each pass solves one convex problem for the twists of all awake bodies
with Newton's method, and each Newton system is solved exactly by eliminating
bodies one at a time over the contact graph. Eliminating a body folds its
inertia and load into the bodies it touches, so what a lower rock "sees" is the
macro-frame above it; one up sweep and one down sweep carry a load change
through the whole pile inside the frame. Each contact's compliance is set
against the mass it carries, which is what keeps the passes converging at the
same rate for a 30-box column or a 5000:1 mass ratio as for a single rock. A
pile at rest is a fixed point of all of this: the frame costs one gradient
evaluation and nothing moves.

## Files

| File | Role |
| --- | --- |
| `math3.mjs` | vectors, quaternions, 3x3 matrices |
| `hull.mjs` | convex hull, coplanar-face merging, exact mass properties (`cook`) |
| `shapes.mjs` | the specification's test shapes: boxes and seeded rocks |
| `collide.mjs` | separating-axis test with Gauss-map edge pruning, face clipping, witness validation, ground contact, four-point reduction |
| `world.mjs` | bodies, once-per-frame collision, impulse persistence, islands, sleeping |
| `solver_soft_step.mjs` | the comparator: substepped soft-step sequential impulses, as the specification's section 7 prescribes |
| `solver_wrench.mjs` | the experiment: retained impulse field, proximal passes, Newton with exact line search, carried-mass compliance |
| `block_ldl.mjs` | macro-frame elimination: sparse 6x6-block Cholesky over the contact graph |
| `scenes.mjs` | benchmark scenes after the specification's acceptance tests |
| `run_study.mjs` | runs every scene for every solver configuration, writes JSON |
| `test_basic.mjs`, `test_collide.mjs` | cooking, rest and stack checks; narrow phase against a brute-force oracle |
| `test_features.mjs` | static bodies, restitution, rolling resistance, crane hold, rest report, stable-set query |
| `container.mjs`, `container_report.mjs` | the general benchmark: 64 mixed shapes packed in a box; one scorer for every engine |
| `external/run_mujoco.py`, `external/run_rapier.mjs` | the same scene in MuJoCo and Rapier |
| `run_container_m4.sh` | runs the container benchmark for all engines on the Mini and copies results back |
| `rockgen.mjs` | rock generator: geometry below a cutoff wavelength, texture above it as friction and a deformation map |
| `summarize.mjs` | prints the earlier study record as a table |
| `viewer.html`, `viewer.mjs` | WebGL viewer: pick a scene and a solver, orbit, drop generated rocks |
| `results/` | retained study records |

## Running

Tests and the study run on the M4 Mini (Node 25 at `/opt/homebrew/bin/node`):

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  '/opt/homebrew/bin/node experiments/wrench_transport/test_collide.mjs && \
   /opt/homebrew/bin/node experiments/wrench_transport/test_basic.mjs'

/Users/ultimussecundai/.local/bin/m4build -- \
  /opt/homebrew/bin/node experiments/wrench_transport/run_study.mjs \
  --out /tmp/wrench_transport_full.json
```

Copy `/tmp/wrench_transport_full.json` back into `results/` immediately. Add
`--quick` for a two-minute version and `--configs wrench,soft_step_best` to
select solver configurations. `test_features.mjs` runs the same way.

The container benchmark against MuJoCo and Rapier, for one seed:

```sh
experiments/wrench_transport/run_container_m4.sh 1
```

It writes and copies back `results/container/seed1/` (scene, one result per
engine setting, `report.md`, `report.json`). MuJoCo 3.2.3 is installed in the
Mini's system Python 3.9 user site; Rapier 0.21 is in
`~/Developer/CodexBuilds/wrench_external/node_modules` on the Mini, outside
the mirror.

`test_basic.mjs` reports, without asserting, whether the specification's
solver holds a 10-box and a 30-box column; at the specification's parameters
it does not (see FINDINGS.md).

The viewer needs a static file server because it loads ES modules:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory experiments/wrench_transport
```

then open `http://localhost:8765/viewer.html`. In the Claude desktop app the
`wrench-transport-viewer` entry in `.claude/launch.json` starts the same server.

## Solver configurations compared

| Name | What it is |
| --- | --- |
| `soft_step_spec` | the specification as written: 30 Hz soft contacts, damping ratio 10, 8 substeps, four-point manifolds, rolling resistance after each substep |
| `soft_step_best` | the same solver at its strongest measured setting for 8 substeps: 120 Hz contacts, every manifold point kept, no post-hoc rolling resistance |
| `soft_step_16` | 16 substeps, 240 Hz contacts, otherwise as `soft_step_best` |
| `wrench` | wrench transport, one solve per frame, no substeps |

Both solvers share collision, persistence, islands and sleeping. The
comparator is an in-house implementation of the published soft-step method; no
external engine has been run against these scenes yet.

## Scope and status

This is the demonstration stage. It has contact, friction, sleeping, static
bodies, restitution, rolling resistance, the crane hold, a rest report, a
stable-set query, a rock generator and the benchmarks. It does not yet have
compound shapes in scenes, determinism checks, or the C++ implementation with
the specification's exact API. Timings are JavaScript timings and say nothing
yet about the 1.5 ms C++ budget.
