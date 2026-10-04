# zc::phys notes

State on 2026-10-03: the engine core is ported from the JavaScript prototype
one directory up and runs the container benchmark. The specification's
acceptance tests are not ported yet; `tests/phys_tests.cpp` is one smoke check.

## Build and run (on the Mini)

```sh
/opt/homebrew/bin/cmake -S experiments/wrench_transport/phys -B /tmp/zc_phys_build
/opt/homebrew/bin/cmake --build /tmp/zc_phys_build -j4
/tmp/zc_phys_build/phys_tests
node experiments/wrench_transport/container.mjs --seed 1 --export-text /tmp/zc_scene.txt
/tmp/zc_phys_build/phys_bench /tmp/zc_scene.txt /tmp/zc_cpp60.json 60 8
```

`phys_bench` writes the same result record as the other engines' runners, so
`container_report.mjs` scores it with them.

## Where this departs from the specification

- **Solver.** Section 7's soft-step sequential impulses are replaced by the
  wrench-transport solve (`solver.cpp`; method and measurements in
  `../FINDINGS.md`). `substeps`, `contact_hertz` and `contact_damping` in
  `WorldParams` are kept for the signature and unused.
- **Manifolds** are not reduced to four points.
- **Collision** adds two checks to section 6.2: a face manifold must hold the
  depth its axis measured, and an edge axis must match the hulls' true
  separation along it.
- **Rolling resistance** is a row of the solve, not a correction after it.
  The ground has its own lever arm (`SolverParams::ground_rolling_resistance`).
- **Restitution** is asked for one frame after a gap closes, at the surface.
- **Additions to the API:** `add_static_body`, `SolverParams` and its
  accessors, `rest_report`, `stable_set`, `solve_stats`, `guard_count`,
  `HoldState::lateral_force`, `Shape::hulls`.
- `cook` throws on degenerate input; the specification asks for an assert in
  debug and a fallback tetrahedron in release. Hull simplification above 48
  vertices is not implemented (and no vertex cap is enforced).
- `step()` still allocates when its vectors grow and in `stable_set`,
  `contacts` and `bodies`; steady-state frames of a fixed scene reuse storage,
  but that has not been audited.
- `Contact::a < b` is not guaranteed: `a` is always a free body and `b` may
  be a static body with a lower id.

## First measurements

Container benchmark, seed 1, 64 bodies, sleeping off, M4 Mini, Release:

| | 60 Hz | 120 Hz |
| --- | ---: | ---: |
| Mean step | 1.46 ms | 1.14 ms |
| Worst step | 9.13 ms | 6.27 ms |
| Mean step while anything moves faster than 4 mm/s | 3.75 ms | 2.67 ms |
| Mean step once quiet (still awake, colliding every pair) | 1.07 ms | 0.91 ms |
| Newton solves / factorisations | 3,237 / 3,232 | 5,717 / 5,712 |
| Wall-clock for 8 s | 0.70 s | 1.09 s |

The JavaScript prototype took 2.00 s and Rapier 0.22 s for the same 8 s. Final
state: no overlapping pair, deepest overlap 0.003 mm, quiet after 1.20 s.

Nothing here is optimised. The quiet-frame cost is almost all narrow phase:
every pair runs the full separating-axis test with a quadratic edge search
each frame, with no cached axis and no broad phase beyond bounding spheres.
The moving-frame cost is about 45 Newton solves per frame at about 60
microseconds each. Both are the next things to work on, against the
specification's 1.5 ms mean for 75 awake bodies.
