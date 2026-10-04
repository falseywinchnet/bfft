# Wrench transport: findings

State as of 2026-10-03, second pass. Everything here was measured with the
JavaScript prototype in this directory on the M4 Mini (Node 25). Retained
records are under `results/`.

## Scope

The engine's job is correct, efficient rigid-body physics for arbitrary convex
geometry. Stacking rocks is the game's job. The general test is therefore a
container packed with mixed shapes, scored against outside engines (below);
the stacking scenes that follow it are demonstrations of the solver at rest,
not requirements on what shapes must stack.

## Summary

In a container of 64 mixed convex bodies, over three seeds:

- **Speed to settle.** This solver is quiet (fastest body below 4 mm/s and
  staying there) 1.0 to 2.1 s after release at 60 Hz. Rapier at its defaults
  takes 1.1 to 3.4 s. MuJoCo did not become quiet within 8 s in any of the
  four configurations tried.
- **Quality.** At the end this solver has no pair overlapping by more than
  0.001 mm, and the floor and walls return 0.999998 to 1.000000 of the weight.
  Rapier leaves 76 to 99 pairs overlapping by 0.6 to 1.2 mm on average (3 to 12
  mm at worst). MuJoCo's best configuration leaves 75 to 85 pairs at 0.04 mm
  average, 0.2 mm worst, with bodies still moving at 16 to 64 mm/s.
- **Work.** This solver averages about 9 Newton solves (each one
  factorisation) per frame over the 8 s run, most of them while the pile is
  moving: 1.6 to 2.0 s of JavaScript for 8 simulated seconds. Rapier spends 4 relaxation sweeps
  per frame and 0.22 s as compiled WebAssembly, seven to nine times less
  wall-clock. MuJoCo at 480 Hz spends 29 to 39 thousand Newton iterations
  and 12 to 31 s.

So: equal or faster to settle than the engines it was compared with, more
exact at rest than both, and slower per frame than a compiled game engine by
a factor that the C++ port has yet to measure.

Not established: the cost in C++ against the 1.5 ms budget; whether an expert
could configure Rapier or MuJoCo to do better than my settings; whether any
part of the method is new.

## What the solver is

Unknowns are the twists V of the awake bodies. V* is the free twist after
gravity and damping. Each contact point c has a 3-row Jacobian J_c (two
tangents, one normal), a friction coefficient, a target velocity vhat_c, a
compliance R_c, and a retained impulse lambda_c inherited from the last frame
by position matching. One pass minimises

    l(V) = sum_b 1/2 (V_b - V*_b)^T M_b (V_b - V*_b)
         + sum_c R_c/2 | P_K( lambda_c - (J_c V - vhat_c) / R_c ) |^2

where P_K is the Euclidean projection onto the friction cone. The projected
vector is the pass's impulse gamma_c; its gradient contribution is -J_c^T
gamma_c, and the Hessian contribution is J_c^T (dP/dy / R_c) J_c. After a
pass, lambda_c becomes gamma_c. The passes stop when the field changes by less
than 1e-3 relative (R-weighted) and the sliding shifts by less than 1e-3 m/s,
or after six.

- **Normal target.** A gap may close exactly within the frame (vhat_n =
  -gap/h); an overlap decays with a 0.1 s time constant and never faster than
  0.5 m/s. vhat_n also carries de Saxce's shift -mu |v_t|, updated each pass,
  which removes the convex model's lift of sliding contacts.
- **Inner solve.** Newton's method with an exact line search on the convex
  cost. A pass is solved only as accurately as the field is still changing
  (0.1 times the last relative change, between 1e-9 and 1e-3), but a start
  that misses 1e-9 always takes one step.
- **Linear solve.** The Hessian is M plus one relation per touching body pair,
  6x6 blocks on the contact graph. `block_ldl.mjs` chooses a minimum-degree
  order when the graph changes and performs a block Cholesky factorisation:
  eliminating a body adds its Schur complement to the bodies it touches. The
  factor is reused within a frame while no point slides and the pressing set
  is unchanged.
- **Compliance.** R_c = 0.03 / max(local effective mass, mass carried through
  c's pair, mass implied by last frame's load on c). Carried mass comes from
  one sweep over bodies from the highest centre of mass down, splitting each
  body's own-plus-carried mass among the pairs that support it.

At the fixed point lambda stops changing, the compliance term is inert, and
the impulses satisfy rigid unilateral contact with Coulomb friction. A pile at
rest starts each frame at that fixed point: zero Newton iterations, one
gradient evaluation.

Lineage, stated plainly: the inner problem is the compliant convex contact
potential of Castro et al. (Drake's SAP, 2022), in the family of Anitescu's
convex model and MuJoCo's solver; wrapping it in multiplier updates is the
proximal method of multipliers; the block elimination is sparse Cholesky, the
same algebra as Featherstone's and Baraff's linear-time methods on trees.
What this experiment adds is the combination for stacking, the retained
impulse field as the frame-to-frame state, and the carried-mass compliance. I
have not searched the literature for the last item; treat its novelty as
unknown.

## How it relates to the photonic engine and the macro-frame idea

| Photonic transport field | Here |
| --- | --- |
| retained operator plus its current response | retained impulse field plus this frame's Hessian factor |
| rank-one relation block `v (u^T L)` | per-pair relation `X_a^T K X_b`: any number of contact points between two bodies acts through twelve twist coordinates |
| child-volume response `R_V`, emission `E_V` | Schur complement of an eliminated body: the inertia and load its supporters see |
| signed residual march after an edit | proximal pass from the retained field |
| invalidation by dependency | not implemented: only whole-frame factor reuse so far |

The analogy fails at one point, and the failure is what shaped the solver. An
optical residual march contracts because every bounce absorbs (`q < 1`).
Static load is transmitted without loss: the same march over contacts has a
contraction factor that tends to 1 with pile height, which is why relaxation
solvers need sleeping to look still. The macro-frame is the cure in both of
its roles: exact elimination replaces the march inside a pass, and
carried-mass compliance keeps the pass-to-pass contraction from collapsing.
For a column of N equal bodies the slowest error mode decays per pass like
1 / (1 + 1 / (0.8 eps N^2)) with local compliance (0.96 for N = 30 at eps =
0.03) and like 1 / (1 + 1 / (c eps N)) with carried-mass compliance, c below
1. These are estimates from the chain's Delassus spectrum, not measured rates.

## Container benchmark against outside engines

`container.mjs` releases 64 bodies (8 each of cube, 42-vertex sphere,
16-sided cylinder, cone, 8 mm sheet, random prism, random polyhedron, jagged
rock; 4 to 9 cm) on a grid inside an open box with static walls, friction 0.5
everywhere, no restitution, no damping, for 8 s. Every engine gets the same
hull vertices and density. `container_report.mjs` scores every engine's final
poses with this experiment's own narrow phase, so overlap is measured with one
ruler. `run_container_m4.sh SEED` runs everything on the Mini.

Three seeds; each cell lists seed 1, 2, 3.

| Engine and setting | Quiet after (s) | Fastest body in last second (m/s) | Deepest overlap (mm) | Overlapping pairs, mean depth (mm) | Vertical support / weight | Wall-clock for 8 s (s) |
| --- | --- | --- | --- | --- | --- | --- |
| wrench, 60 Hz | 2.07, 0.98, 1.27 | 1.1e-3, 6.9e-7, 3.9e-7 | 6.5e-4, 2.4e-4, 2.9e-4 | none | 0.999998, 1.000000, 1.000000 | 2.00, 1.58, 1.66 |
| wrench, 120 Hz | 1.25, 1.02, 1.70 | 7.6e-8, 2.2e-9, 9.0e-9 | 8.5e-5, 3.9e-8, 1.0e-6 | none | 1.000000 x3 | 2.96, 2.82, 3.11 |
| soft-step (best), 60 Hz | 5.83, 6.52, 2.53 | 4.1e-4, 1.1e-3, 1.1e-4 | 1.39, 0.97, 0.85 | 81 to 92, 0.53 to 0.56 | 1.000000, 0.999976, 1.000001 | 1.41, 1.49, 1.48 |
| soft-step, 16 substeps | 6.33, 3.90, 2.10 | 2.8e-4, 1.0e-3, 5.2e-4 | 0.74, 0.66, 0.73 | 87 to 94, 0.47 to 0.50 | 1.000088, 0.999998, 0.999902 | 1.96, 2.01, 2.01 |
| MuJoCo 3.2.3, 480 Hz, defaults | never x3 | 0.36, 0.15, 0.17 | 10.2, 7.1, 6.9 | 89 to 114, 0.57 to 0.67 | 1.0037, 1.0036, 0.9773 | 30.4, 29.5, 24.7 |
| MuJoCo, 480 Hz, multiccd | never x3 | 0.18, 0.50, 0.12 | 10.0, 1.1, 1.4 | 92 to 114, 0.32 to 0.43 | 0.9988, 0.9985, 0.9929 | 28.2, 20.5, 11.8 |
| MuJoCo, 480 Hz, multiccd, solref 5 ms | never x3 | 0.037, 0.016, 0.064 | 0.24, 0.19, 0.20 | 75 to 85, 0.04 to 0.05 | 1.0038, 0.9932, 0.9982 | 18.4, 20.9, 30.6 |
| MuJoCo, 120 Hz, multiccd | never x3 | 0.84, 0.48, 0.80 | 3.6, 6.7, 7.8 | 102 to 114, 0.45 to 0.53 | 0.9856, 1.0160, 1.0221 | 11.9, 8.5, 10.5 |
| Rapier 0.21, 60 Hz, defaults | 1.72, 3.43, 1.08 | 1.3e-5, 4.3e-6, 1.8e-6 | 7.0, 12.0, 7.2 | 76 to 87, 1.11 to 1.24 | not extracted | 0.22, 0.23, 0.23 |
| Rapier, 60 Hz, length unit 5 cm | never, 5.05, never | 0.17, 1.4e-6, 0.22 | 6.8, 3.2, 7.4 | 86 to 99, 0.63 to 1.15 | not extracted | 0.17, 0.16, 0.17 |
| Rapier, 240 Hz, 8 iterations, length unit 5 cm | 6.87, 1.07, never | 2.1e-3, 5.1e-5, 0.29 | 5.4, 4.5, 5.6 | 88 to 92, 0.61 to 0.91 | not extracted | 0.66, 0.66, 0.66 |

Work in seed 1: wrench 60 Hz used 4,237 Newton solves, 4,232 factorisations
and 1,293 passes over 480 frames with about 399 contact points; MuJoCo used
28,947 to 39,286 Newton iterations over 3,840 steps with 190 to 241 contact
points; Rapier used 4 solver iterations per step.

What to hold against these numbers.

- The wrench and soft-step rows are JavaScript; Rapier is compiled Rust in
  WebAssembly; MuJoCo is compiled C driven from Python. Wall-clock compares
  implementations as they stand, not methods.
- MuJoCo and Rapier were run as I configured them. For MuJoCo I tried the
  default, multiple contact points per convex pair, a stiffer contact time
  constant, and a lower rate; for Rapier the default, its length-unit scale
  for small objects, and a higher rate with more iterations. Someone who
  knows either engine well may do better. MuJoCo is built for articulated
  robots and soft contacts; piles of convex meshes are not its home ground.
  One body left the container in two MuJoCo configurations on seed 1.
- One MuJoCo run at 60 Hz was also tried (seed 1, defaults) and diverged; it
  is not in the table.
- Bullet could not be built: the Mini's disk has about 0.6 GB free and the
  build needs more.
- At 60 Hz one seed of this solver ended with a body still moving at 1.1
  mm/s, below the quiet threshold but not at rest. At 120 Hz none did.
- On seed 1 the mean centre height of the pile is 51 mm for this solver, 43
  to 54 mm for Rapier and 34 to 44 mm for MuJoCo. The lower piles are the
  ones with overlap; I did not separate packing differences from overlap.

## Demonstrations at rest (earlier study)

These rows were measured before the hold, restitution and rolling-resistance
work, with `results/full_study.json`. The 15-stone rows use flat stones cut
level top and bottom, which makes stacking easy; they show the solver holding
a stack, not that arbitrary stones stack.

Sleeping is off unless a row says otherwise. Worst case over seeds (5 for
rest, 3 for stacks and drops, 1 for the heap). 60 Hz frames.

| Scene and measure | soft_step_spec | soft_step_best | soft_step_16 | wrench |
| --- | ---: | ---: | ---: | ---: |
| Rest, round rock: drift after 3 s (mm) | 0.500 | 0.220 | 0.273 | 7.6e-10 |
| Rest, round rock: overlap at rest (mm) | 0.731 | 0.514 | 0.504 | 1.6e-10 |
| Rest, flat rock: drift after 3 s (mm) | 5.311 | 4.1e-7 | 0.063 | 3.8e-11 |
| Rest, flat rock: overlap at rest (mm) | 3.474 | 0.518 | 0.505 | 3.3e-11 |
| Rest, jagged rock: drift after 3 s (mm) | 0.373 | 0.043 | 0.137 | 1.0e-9 |
| Rest, jagged rock: overlap at rest (mm) | 0.932 | 0.527 | 0.507 | 8.2e-11 |
| Column of 10 boxes | falls | stands | stands | stands |
| Column of 10: top height error (mm) | - | -7.127 | -5.566 | -2.5e-10 |
| Column of 20 boxes | falls | falls | stands | stands |
| Column of 20: top height error (mm) | - | - | -12.009 | -1.9e-10 |
| Column of 40 boxes | falls | falls | falls | stands |
| Column of 40: top height error (mm) | - | - | - | 8.2e-10 |
| 15 flat stones: stacks built of 3 seeds | 0 of 3 (fewest placed 0) | 0 of 3 (fewest placed 3) | 0 of 3 (fewest placed 6) | 3 of 3 (fewest placed 15) |
| 15 flat stones: worst drift over 60 s (mm) | - | - | - | 2.9e-4 |
| 15 flat stones, sleeping on: stacks built | 0 of 3 (fewest placed 0) | 0 of 3 (fewest placed 3) | 0 of 3 (fewest placed 4) | 3 of 3 (fewest placed 15) |
| 15 flat stones, sleeping on: asleep after (s) | - | - | - | 0.52 |
| Slab on three pebbles, 500:1: slab sinks (mm) | 20.074 | 6.948 | 2.488 | 1.8e-13 |
| Pebble on slab, 500:1: overlap (mm) | 0.968 | 0.529 | 0.507 | 2.6e-10 |
| Slab on three pebbles, 5000:1: slab sinks (mm) | 20.158 | 19.720 | 19.528 | 2.3e-13 |
| Pebble on slab, 5000:1: overlap (mm) | 0.967 | 0.529 | 0.507 | 5.5e-10 |
| Overhang, centre of mass 5% inside: rotation (deg) | 0.3031 | 0.0384 | 0.0292 | 0.0000 |
| Overhang, centre of mass 2% inside: rotation (deg) | 0.3571 | 0.0382 | 0.0299 | 0.0000 |
| Overhang, centre of mass 1% inside: rotation (deg) | 0.3761 | 0.0369 | 0.0305 | 0.0000 |
| Overhang, 1% outside: tips after (s) | 0.35 | 0.20 | 0.20 | 0.23 |
| Overhang, 2% outside: tips after (s) | 0.20 | 0.15 | 0.15 | 0.17 |
| Overhang, 5% outside: tips after (s) | 0.12 | 0.12 | 0.12 | 0.12 |
| Incline below friction limit: creep (mm) | 0.245 | 0.358 | 0.427 | 3.2e-6 |
| Incline above friction limit: slide in 3 s (mm) | 2292.791 | 2293.964 | 2251.127 | 2253.296 |
| Box on slab, below limit: creep (mm) | 0.266 | 0.450 | 0.480 | 2.8e-5 |
| Drop from 0.3 m: worst bounce (mm) | 1.311 | 0.503 | 0.068 | 0.389 |
| Drop from 0.3 m: deepest overlap (mm) | 1.665 | 0.519 | 0.505 | 8.9e-5 |
| Drop from 0.3 m: at rest after (s) | 1.317 | 1.433 | 1.667 | 1.667 |
| Heap of 30: at rest after last drop (s) | not within 12 s | 1.850 | 0.983 | 0.800 |
| Heap of 30: final overlap (mm) | 3.797 | 0.751 | 0.524 | 3.9e-10 |
| Heap of 30: deepest overlap while pouring (mm) | 27.051 | 23.973 | 20.243 | 13.315 |
| Heap of 30: mean step (ms, JavaScript) | 1.44 | 1.36 | 1.45 | 1.74 |
| Heap of 30: worst step (ms, JavaScript) | 2.73 | 2.98 | 3.11 | 7.45 |

Reading notes.

- The incline's analytic slide with the scenes' 0.02/s linear damping is about
  2255 mm; both families are within 2% of it.
- The heap's 1,260 frames used 11,187 Newton iterations and 4,015 passes; 509
  frames stopped at the six-pass limit rather than at tolerance. The heap
  still came to rest because the field keeps converging across frames.
- Stack-building uses a placement helper (set each stone over the previous
  stone's centre of mass, 2 mm up; accept if everything settles, the new stone
  ends within 3 cm, and no older stone shifts more than 5 mm; otherwise try 18
  nearby offsets). Its thresholds were chosen while developing the wrench
  solver. The comparator was not given a separately tuned helper, so "0 of 3"
  measures that solver under this helper, not the best it could do with one
  written for it.
- Without sleeping, two of the three stacks keep a residual motion of up to
  8e-6 m/s during the watch (worst drift 2.9e-4 mm in 60 s) and take one
  Newton iteration per frame instead of none. I did not find which mode is
  still converging.

### Why the specification's solver sinks

A soft contact of frequency f and effective mass m_eff is a spring of
stiffness m_eff (2 pi f)^2 whatever its damping ratio, so a point carrying
force F rests F / (m_eff (2 pi f)^2) below the slop. For a box corner m_eff is
about m / 5.8. At the specification's 30 Hz that is 0.4 mm for one box, about
4 mm under ten, and tens of millimetres for a 0.02 kg pebble under a 10 kg
slab. The measured sinks match. This is a property of the parameters in the
specification, checked by hand against the implementation here; it does not
say what Box2D or any shipping engine does.

## What was tried and dropped

Each of these was implemented and measured in this session; numbers are from
those runs, most on an earlier heap scene (rocks released from up to 1.45 m).

1. **Stiff cone potential, one pass per frame** (compliance 1e-6 of local
   inverse mass). Exact at rest, but one round rock's impact frame took 69
   Newton iterations. A stiff penalty on a curved (conic) constraint leaves
   Newton a narrow curved valley.
2. **Stiff normal plus friction disks with lagged bounds** (bound = mu times
   last pass's normal impulse). Good on single rocks (19 iterations worst).
   In the heap the bound iteration did not converge within six passes in 251
   of 360 frames, and cost 38 to 70 Newton iterations per frame. Friction near
   0.8 with wedged rocks couples normal and tangential impulses too strongly
   for a lagged bound.
3. **Unit Newton steps** in place of the line search. Most frames converged
   in under ten iterations, but 181 solves in 360 frames cycled without
   converging. With a watchdog that
   falls back to a line search, a 30-box column took 3,073 iterations and left
   64 frames unconverged, against 185 iterations with the line search alone.
4. **Soft-to-stiff continuation** for frames that stall. Worse than not doing
   it: 130 to 190 iterations per frame.
5. **Proximal passes with compliance scaled by the touching pair only.** The
   10-box column was exact; the 30-box column bounced indefinitely (late
   kinetic energy 2 J, pass limit reached every frame) and the heap never came
   to rest. Scaling by carried mass fixed both. Adding last-frame load removed
   intermittent re-activations of a settled heap.
6. **Rolling resistance applied to angular velocity after the solve.** It
   turned a rock rocking about an edge into a rock translating at a constant
   3 mm/s forever, because it removes rotation and leaves the matching
   translation. It must be a term inside the solve.
7. **Per-pair friction anchors** (a retained relative pose supplying a
   restoring target). Wrong for rolling contact, and unnecessary once the
   retained field carries static friction.

## Collision findings

Both solvers share the narrow phase, and three of its faults first looked like
solver failures. `test_collide.mjs` now checks 3,000 random near-contact pairs
and 3,000 resting face contacts against a brute-force support-function oracle.

- **Face bias.** The specification's 1 mm preference for face axes let a face
  manifold stand in for a deeper edge contact; one case missed a 2 mm overlap.
  The bias is kept (without it resting face contacts flicker to single-point
  edge contacts), but a face manifold is now accepted only if it holds the
  depth its axis measured, with ordered fallbacks.
- **Edge axes.** With nearly parallel faces the Gauss-map arc test admits edge
  pairs that are not faces of the Minkowski difference, with meaningless
  separations. One such value exceeded the margin and made a resting contact
  in a 15-stone stack vanish for a frame; the stones above dropped 2.7 mm.
  Each improving edge candidate is now verified against the hulls' true
  separation along its axis.
- **Four-point reduction.** Reducing ground manifolds to four points makes a
  resting flat stone buzz at up to 5e-2 m/s under the soft-step solver;
  keeping every point gives 2.5e-17 m/s. The wrench solver keeps every point
  at no cost in system size.
- **Fast first impacts.** Collision runs once per frame against a margin
  capped at 5 cm. A rock arriving at 3 to 5 m/s can meet features other than
  the ones its speculative points guard, leaving up to 13 mm of overlap that
  then decays over about ten frames. The comparator shows 20 to 27 mm in the
  same scene. Unsolved.

## Features around the solve

All checked by `test_features.mjs` (17 checks, passing on the Mini).

- **Static bodies.** Bodies that never move, colliding like the ground.
- **Restitution.** A point closing faster than the threshold first closes
  its gap; the rebound is asked for on the next frame, at the surface. A box
  dropped 0.3 m with e = 0.5 rises 75.9 mm (ideal 75).
- **Rolling resistance** is a row of the solve: a torque on the pair's
  relative angular velocity, bounded by a lever arm times the pair's normal
  impulse. With a 2 mm lever arm a slab whose centre of mass is 1 mm past
  the edge stays and one 4 mm past tips; with none, 1 mm tips.
- **Crane hold.** A linear spring with gravity feed-forward, wires that only
  pull (capped), capped side force, and a capped angular spring. Hanging
  error 0.009 mm, tension 0.9997; a 10 cm move ends without overshoot;
  setting a 2 kg box on a five-box stack moves the stack 3e-11 mm while
  tension falls monotonically to slack; a sideways shove is capped at exactly
  0.35 of the held weight.
- **Rest report.** `world.rest` holds quiet/not, seconds quiet, kinetic
  energy, fastest speed and awake count, refreshed inside the step at no
  extra pass. Reading it is constant time.
- **Stable set.** `stableSet(world)` walks the retained contact manifolds
  breadth-first from the ground and static bodies through bodies that have
  been quiet for a quarter second, and returns the settled part of a pile
  while other rocks are still moving. One pass over the manifolds.

## Rock generator

`rockgen.mjs` builds a rock as a radius field over directions: an ellipsoid,
a few fracture planes, and noise octaves whose amplitude falls with frequency.
The octaves are split at a cutoff wavelength equal to twice the mean spacing
of the collision hull's vertices.

- Bands below the cutoff are geometry: sampled at the vertex budget, pushed
  out until the hull and the surface agree on average, and cooked as the
  convex collision hull.
- Bands above the cutoff never reach the physics as shape. They are kept as
  a deformation map (six faces of 32 x 32 heights over directions) and as one
  number, the roughness angle: half the texture's root-mean-square slope
  angle, capped at 15 degrees. Friction is tan(basic friction angle +
  roughness angle), Patton's law for rough rock joints, with a basic angle
  of 31 degrees.
- The render mesh (2,562 vertices) is the collision hull's radius plus the
  texture height, so what is drawn touching is what the solver has touching,
  to within the texture's height.

Mean over 20 seeds of 9 cm rocks (`results/rockgen_budget.json`):

| Hull vertex budget | Geometric octaves | Texture rms height (mm) | Friction | Hull vs designed surface, mean (mm) | worst (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 1 | 0.91 | 0.881 | 1.18 | 13.6 |
| 96 | 2 | 0.49 | 0.864 | 0.71 | 10.3 |
| 192 | 2 | 0.49 | 0.864 | 0.43 | 8.5 |

The worst-case figures are concavities and fracture edges that one convex
hull bridges. The roughness gain, cap and basic angle are modelling choices,
not measurements. Rocks are single convex hulls; concave rocks as compounds
of hulls, and friction that depends on which two textures meet, are not done.

## Floor absorption

With no rolling resistance and no damping (the cross-engine setting), a
settled container pile kept a group of bodies rocking at up to 0.4 mm/s
indefinitely: a marginal mode with nothing to dissipate it. Giving the ground
alone a rolling-resistance lever arm of 4 mm (`groundRollingResistance`, no
resistance between bodies) removed it: no body above 0.01 mm/s from 5 s on,
including bodies that do not touch the floor. This is the transport picture
again: an absorbing boundary makes the whole field contract. The lever arm is
now a world parameter with that default; the cross-engine benchmark keeps it
at zero for parity.

## Rock mix

Five photographs of balanced stones and a beach cobble field (Wikimedia
Commons, category "Rock balancing") show: mostly water-rounded discs and
ovoid cobbles in warm tans, creams, pinks and greys; blocky sub-angular
stones with flat fracture faces and rounded edges; flat blades and slabs
stood on edge; sizes spanning about three to one within one stack, largest
at the bottom. `generateMixedRock` draws from five classes on that basis.
Over 300 seeds:

| Class | Share | Shape | Mean friction | Texture rms (mm) |
| --- | ---: | --- | ---: | ---: |
| river disc | 30% | smooth, thin axis 0.4 | 0.61 | 0.33 |
| cobble | 21% | smooth, near-equant | 0.65 | 0.43 |
| block | 18% | seven fracture faces | 0.81 | 0.75 |
| slab | 19% | two bedding planes, thin axis 0.3 | 0.79 | 0.83 |
| shard | 12% | 3 to 7 cm, eight deep fractures | about 0.9 (capped) | 0.64 |

Five photographs are a look, not a survey; the shares and ratios are my
reading of them, not measurements. The render mesh now sits midway between
the hull's facets and the designed surface, which softens the outline of
large smooth stones.

## C++ port: first measurements

`phys/` holds the engine core in C++20 with the specification's API plus
additions (`phys/NOTES.md`). It builds without warnings and reproduces the
prototype's behaviour on the container benchmark (seed 1, sleeping off):
quiet after 1.20 s, no overlapping pair, deepest overlap 0.003 mm.

| | C++, 60 Hz | JavaScript, 60 Hz | Rapier, 60 Hz |
| --- | ---: | ---: | ---: |
| Wall-clock for 8 s | 0.70 s | 2.00 s | 0.22 s |
| Mean step | 1.46 ms | 4.2 ms | 0.46 ms |
| Worst step | 9.1 ms | 22.6 ms | 26.7 ms |

Unoptimised: about 1 ms of every frame is the narrow phase run on every pair
with no cached axis, and moving frames average 3.75 ms. The specification's
budget is 1.5 ms mean for 75 awake bodies and 0.1 ms asleep.

## Limits

- The C++ port has no acceptance tests yet and is not optimised; its mean
  step meets the budget on this scene only because most frames are quiet.
- A fast first impact can leave up to 13 mm of overlap for about ten frames
  (see Collision findings). Unsolved.
- Compound (concave) shapes, determinism checks and wake-on-remove tests are
  not exercised.
- Frame-to-frame factor reuse and dirty-set refactorisation are not
  implemented.
- The pass loop reaches its limit in most frames of a pouring heap.
