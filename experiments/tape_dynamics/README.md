# Atomic tape dynamics

A separate C++20 rigid-body experiment using the existing Wrench convex/compound geometry and the unchanged persistent atomic worker pool from File Manager. Body state is carried between physical intervals. Contact groups perform coherent two-body momentum exchanges, while disjoint contacts execute concurrently. This is a budgeted sequential-impulse method with parallel transactions; it does not establish a new force law.

## State and execution

Each body carries pose, linear velocity, world angular momentum, inertia, and a position in a mutable tape. Contact witnesses retain normal and tangential impulses for warm starting. Geometry transforms and pair manifold/row construction run through the worker pool. The spatial tree update and candidate collection are currently serial.

A contact task locks its two dynamic bodies in ascending ID order and reads/updates both records while holding those locks. Equal and opposite impulses use a shared world contact point, preserving total linear and angular momentum during that exchange. Disjoint tasks execute concurrently; tasks sharing a body serialize. This avoids torn component snapshots or lost downstream additions. The public World interface itself requires a single caller; callers must not mutate a world while advance is executing.

The default numerical budget is 32 alternating tape traversals per 1/120-second interval. With more than 640 contact groups, all traversals are submitted in one batch without global barriers between traversals. Shared-body locks retain local dependencies. Thread scheduling can change contact order and therefore the finite-budget trajectory. Smaller batches use traversal barriers and can finish early. The pool selects its own serial fallback for small workloads; workers=1 supplies the same algorithm as a serial control.

After a batch, measured contact corrections can move a body's strongest downstream neighbor beside it in the tape. A contact group whose complete last visit produced zero corrections can skip another visit only while both bodies' velocity revision counters remain unchanged. Groups are reconstructed every physical interval. This is an unchanged-input optimization, not sleeping or an equilibrium certificate.

Friction projects onto the Coulomb disk in the two-dimensional effective-mass metric. Speculative contacts use a velocity/rotation-expanded margin, followed by four position-correction passes. Each body independently substeps its angular integration; one fast spinner does not shorten the entire world's interval. Free flight uses the exact constant-gravity position integral when no contacts are present.

## Geometry and limits

Open cups, bowls, bottles, plates, rods, and small convex shapes retain their original compound pieces. Cavities are not filled and geometry is not simplified. Default bodies never sleep. The saved fixture contains 1,000 bodies and 5,342 convex parts, with 14 seconds of motion and all objects released by 6 seconds.

The contact margin is not certified continuous collision detection. Finite passes can leave constraint residuals and transient penetration. Position correction is an approximate stabilization operation, not an energy-conserving dynamics update. The 12-box stack regression explicitly uses 128 traversals; passing that regression does not establish stack convergence at the default budget of 32. The 100 m/s thin-wall check covers one case, not all rotational or compound tunneling cases. Measured overlap is an independent SAT check at saved 10 Hz poses; larger excursions can occur between those samples.

## Build and verify on the Mini

Run from the repository checkout. The helper mirrors authoritative local files using the user's host selector.

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  /opt/homebrew/bin/cmake -S experiments/tape_dynamics -B /tmp/tape_dynamics -DCMAKE_BUILD_TYPE=Release &&
  /opt/homebrew/bin/cmake --build /tmp/tape_dynamics -j4 &&
  /opt/homebrew/bin/ctest --test-dir /tmp/tape_dynamics --output-on-failure'
```

The native checks cover free flight, parallel pair momentum, 800 simultaneous inputs into a shared dynamic body, angular momentum, energy dissipation, friction, a thin wall, a stack, an open compound cavity, support removal, and exact serial equivalence with unchanged-input skipping disabled. The unchanged Wrench acceptance and independent numerical checks are also built and run.

ThreadSanitizer uses a separate build:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  /opt/homebrew/bin/cmake -S experiments/tape_dynamics -B /tmp/tape_dynamics_tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTAPE_BUILD_BASELINE=OFF -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread &&
  /opt/homebrew/bin/cmake --build /tmp/tape_dynamics_tsan -j4 &&
  TSAN_OPTIONS=halt_on_error=1 /tmp/tape_dynamics_tsan/tape_tests'
```

## Matched native measurements

```sh
/Users/ultimussecundai/.local/bin/m4build -- /tmp/tape_dynamics/tape_bench experiments/tape_dynamics/fixtures/count1000-seed1.txt /tmp/tape1000.json 60 --workers=10
/Users/ultimussecundai/.local/bin/m4build -- /tmp/tape_dynamics/tape_legacy_bench experiments/tape_dynamics/fixtures/count1000-seed1.txt /tmp/tape1000-legacy.json 60 120
```

Repeat the first command with workers=1 and workers=4 for controls. Copy outputs home immediately using the route selected by m4host. Run timed jobs sequentially. The benchmark measures native advance only, excluding scene cooking, activation, JSON output, rendering, and independent scoring; those exclusions must accompany performance claims. Active timing covers the eight seconds after all bodies have been released. The unchanged legacy Wrench benchmark also advances internally at 120 Hz and observes at 60 Hz.

The independent JavaScript scorer is copied unchanged from the earlier packing study:

```sh
node experiments/tape_dynamics/scoring/test_score.mjs
node experiments/tape_dynamics/scoring/score.mjs experiments/tape_dynamics/fixtures/count1000-seed1.json RESULT.json SCORE.json
```

Saved records may be gzip-compressed. Decompress before invoking this scorer. The browser viewer reads compressed records directly and replays native poses at 10 Hz with visual interpolation. It is not a browser physics implementation, and its playback speed is not the native simulation performance.

See FINDINGS.md for retained measurements and remaining failures. Development outputs include rejected approaches and one truncated output caused by disk exhaustion; they are not all accepted results.
