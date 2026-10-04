# Obligation Dynamics

An independent C++20 rigid-body engine organized around retained collision and
support obligations. Every body keeps a predicted trajectory and a deadline.
Impacts transmit equal-and-opposite linear and angular impulses through contacts.
Standing contacts retain support and friction forces. A verified stationary
force field produces no periodic physics work until something changes it.

The model is developed alongside Wrench, with the same convex/compound geometry
cooking and contact manifold routines. Wrench dynamics are linked only into the
comparison executable. The library itself has no Wrench world or solver dependency.

Read [MODEL.md](MODEL.md) for the atomic equations, numerical approximations,
clock behavior, and equilibrium tolerances. Development measurements are kept in
`results/development*`; they are diagnostic records, not a frozen release study.

## Build and use

On a machine with a C++20 compiler and CMake:

```sh
cmake -S experiments/obligation_dynamics -B build/obligation -DCMAKE_BUILD_TYPE=Release
cmake --build build/obligation -j4
ctest --test-dir build/obligation --output-on-failure
build/obligation/obligation_minimal
```

In this workspace, run release builds and sustained tests through `m4build`, as
specified in the repository's root `AGENTS.md`. [minimal.cpp](minimal.cpp) is a
complete application: construct a convex shape, drop it at 30 m/s, inspect its
equilibrium residual, then verify that another ten stationary seconds add no
contact work. To embed the engine, add this directory with `add_subdirectory`
and link `obligation_dynamics`. Include [engine.hpp](engine.hpp).

The public operations are `add_body`, `add_static_body`, `advance`, `state`,
`apply_impulse`, `set_pose`, `remove_body`, `contacts`, `statistics`,
`kinetic_energy`, and `equilibrium_residual`. Time and length use seconds and
metres; density is kg/m³. Poses are at the cooked centre of mass. Geometry that
was authored about another origin provides its offset in `Shape::com_offset`.

`advance(dt)` advances simulated time. It processes scheduled events up to the
requested observation time; splitting an observation into smaller calls does
not request a different solver frequency. `apply_impulse` takes momentum in
kg m/s and a world-space point. `set_pose` resets motion and invalidates affected
support. Shape/material editing, joints, soft bodies, and fluids are outside
this implementation.

## Reproducible collision and scaling fixtures

`fixtures/` contains the exact common geometry used by the preceding Wrench,
Rapier and MuJoCo packing study, with SHA-256 provenance. It includes five body
counts (32–512), three fixed seeds, elevated-slab/cavity/freefall controls, and
a separate narrow-bin reconstruction. Cups and bowls are compounds with open
cavities; they are not replaced by their convex hulls for the solver.

`bench.cpp` reads the shared `PACK1` format and writes poses, timing, work counts,
certificate residuals and removed equilibrium energy. `scoring/score.mjs` is the
unchanged independent convex SAT scoring routine. Its known-case tests are in
`scoring/test_score.mjs`. Geometry scoring time is outside simulation timing.

The Mini runner uses a fresh output directory, one warm-up and three measured
repeats by default, and preserves failures and timeouts. It hashes source,
shared geometry and the executable. It reuses an expensive geometry score only
when the complete pose trajectory and scene hashes match; each repetition keeps
its own simulation timing and raw result. Run it sequentially, without other
timed workloads on the Mini:

```sh
python3 experiments/obligation_dynamics/run_study.py \
  --counts 32,64,128,256,512 --seeds 1,7,19 --repeats 3 \
  --output /tmp/obligation-study
```

`--controls` selects the small accuracy controls; `--demo` selects the narrow
bin; `--modes adaptive,awake` adds an equilibrium-certificate ablation. The
`obligation_activity` executable compares 512 isolated resting bodies and one
changing body against Wrench with sleeping both enabled and disabled. Its first
round is a warm-up, followed by three measured rounds in rotated engine order.

Do not infer correctness at arbitrary speed from a finite collision sweep.
Packing overlap is a geometric error measure, not ground-truth motion accuracy.
Position projection and finite-horizon support are approximations; unconverged
updates and tiny energy removals are reported rather than hidden.
