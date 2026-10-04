# Discoverable polynomial transport in mirror geometry

Independent research branch: `codex/krylov-bregman-research`.
Manuscript: `paper/krylov_bregman/main.tex`.

This study asks when a small Krylov representation of an iteration's local
propagator can advance its complete state through many ordinary steps at lower
complete execution cost. It treats smooth mirror descent, Bregman proximal
gradient, and memory-bearing splitting maps with their distinct assumptions.
The finite polynomial itself is not claimed as new: the paper discusses close
trajectory-following and Anderson/Bregman precedents explicitly.

## Mathematical results

- The smooth dual mirror-descent Jacobian is self-adjoint in the inverse
  mirror-Hessian metric at every interior anchor. A corresponding statement
  holds for a differentiable, positive-definite pre-proximal Bregman chart.
- Weighted Arnoldi produces a finite polynomial without a stationary inverse.
  It exactly reproduces affine trajectories when the polynomial fits the
  Krylov space, including closed spaces with a unit eigenvalue.
- For quadratic mirrors and objectives, an energy-orthogonal decomposition
  relates the finite update to the classical CG Galerkin correction and gives
  an objective contraction bound. This does not improve on CG for fixed SPD
  quadratic problems.
- An exact discrete trajectory-defect identity separates frozen-geometry
  error from Krylov compression error. A conditional cost inequality states
  how small their accumulated effect must be for computational acceleration.
- Boundary entropy and scalar quartic examples separate exact finite
  transport, existence of a finite dual fixed point, and actual speedup.

These are local or explicitly conditional statements, not a globally
convergent accelerator for arbitrary mirror/Bregman problems.

## Implementation and retained evidence

`core.py` implements weighted Arnoldi, the finite and stationary corrections,
and the regularized Anderson comparator. `problems.py` defines analytic maps,
Jacobian actions, metrics, and known-optimum synthetic objectives.
`study.py` measures complete solve cost and separately diagnoses one-block
prediction error. `run_suite.py` and `shadow_study.py` define the fixed runs.
`report.py` regenerates manuscript tables and figures from retained JSON.

The fixed finite schedule uses four ordinary prefix steps, depth four,
horizon sixteen, and one ordinary settling step per proposal. It is unguarded.
The ADMM shadow retains the full proximal relation through `s=d+b`; its
ordinary trajectory agrees with the full `(d,b)` representation. It is not a
primal-only memory removal.

Protocol v2 records are in `results/suite_v2` and `results/shadow_v2`.
Primary runs use three seeds and five timing repeats; penalty/shadow runs use
three seeds and three repeats. The known-optimum objective-gap target is
1e-6. Budget failures remain visible. Timing includes basis construction,
metric applications, tangent actions, reconstruction, settling, and scheduled
objective checks. Setup is excluded and recorded separately. Map-plus-Jacobian
counts are reported alongside time, not treated as equal physical costs.
The full-state residual is a separate diagnostic from the objective target.

At n=512, finite Euclidean Arnoldi measured median paired speedups of 2.11x
on quadratic mirrors and 2.32x on coupled entropy-simplex problems versus
ordinary iteration. Mirror weighting measured 2.08x on both, with no additional
work-count advantage in this screen. Anderson was substantially faster on
these smooth cases. At small dimensions and on cheap exact affine controls,
Arnoldi overhead can erase the reduced operator count.

At ADMM rho=1, all three finite-flow seeds failed to reach the target within
budget in both full and shadow coordinates; ordinary ADMM and Anderson
succeeded. Increasing Krylov depth can shrink compression error while the
frozen-geometry error remains large. Representation fidelity is necessary
for the claimed trajectory, but is not sufficient for acceleration.

Exploratory v1 files are retained separately for provenance. They are excluded
from manuscript tables: v2 removes an unnecessary finite-polynomial evaluation
from the stationary comparator and adds a residual-scale ridge floor to
Anderson. The finite schedule was unchanged. These comparators are research
implementations, not reproductions of guarded AA-BPG or A2FoM, and the results
do not establish superiority to those published algorithms.

## Reproduce on the M4 Mini

From this authoritative worktree:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.krylov_bregman.test_contracts -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.run_suite \
  --out /tmp/krylov_bregman_suite_v2

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.shadow_study \
  --out /tmp/krylov_bregman_shadow_v2
```

Copy the remote outputs immediately into `results/suite_v2` and
`results/shadow_v2`, respectively, using the route selected by `m4host`.
The initial retained verification run passed all 17 tests, covering tangent actions,
metric symmetry and covariance, affine polynomial exactness, the quadratic
bound, nonlinear defects, boundary drift, shadow equivalence, and work counts.

Render locally using the existing NumPy/Matplotlib environment:

```sh
/Users/ultimussecundai/bfft/.venv-jpeg/bin/python \
  -m experiments.krylov_bregman.report
tectonic --keep-logs --outdir output/pdf paper/krylov_bregman/main.tex
```

The TeX source uses `results.tex` and two PDF figures. The source archive
contains all manuscript dependencies; no numerical rerun is needed to compile.

## Next research questions

1. Derive an inexpensive predictor of frozen-geometry error that does not
   evaluate every skipped step. Prove what it controls and charge its cost.
2. Use the derivative's mirror symmetry to investigate short-recurrence
   Lanczos or polynomial schedules while controlling changes between metrics.
3. Extend the defect analysis across proximal active-set changes without
   assuming a fixed mask makes every projection branch affine.
4. Compare faithful guarded AA-BPG and trajectory-following implementations
   under matched stopping rules and complete cost accounting.
5. Establish a theorem with checkable assumptions under which nonlinear
   geometry variation is small enough for a genuine cost advantage, or prove
   a useful obstruction. Larger synthetic sweeps alone cannot supply this.

The draft provides a foundation and falsifiable experiments for these questions;
the general nonlinear acceleration problem remains open here.

## September 29 continuation: discoverable polynomial transport

The research object now explicitly includes **discovery** of coordinates,
relations, order, and useful horizon, with all acquisition cost charged.
`paper/krylov_bregman/discovery.tex` develops the connection to Tomić,
Widdershoven, and De Lathauwer, *Sparse Approximation via Polynomial Equations*,
https://arxiv.org/abs/2609.11215. Their sparse-root formulation motivates
algebraic relations, but is not itself a transport-acceleration theorem.

Three distinct constructions are retained:

- `discovery.py`: incremental current-map Arnoldi and sampled live-map
  falsification. No objective, optimum, or future reference is used to select
  the order or horizon. Samples do not constitute a path certificate.
- `test_discovery.py`: exact ESP/rank identities, a conditioning-sensitive
  singular-tail bound, an identifiable finite polynomial-observable control,
  and a counterexample showing why a few accurate probes cannot certify an
  unrestricted nonlinear future. The lifted control is classical Koopman
  identification, not a claimed mirror-descent example or new algorithm.
- `certified_piece.py`: actual finite-path certification for Huber mirror
  descent, using the feature-space constraints of its current affine piece.
  Cached `BQ` columns already acquired during tangent actions check every
  predicted input. The sum of compressed Arnoldi defects bounds the actual
  path error by global nonexpansivity. Exact closure makes the jump exact.
  This is an exact-arithmetic theorem with floating-point verification,
  not an interval-arithmetic certificate.

The sparse sampled scan accepts 35 of 66 anchors; all accepted cases have
independently measured relative trajectory error below 0.0200. It rejects all
natural/permuted Meyer anchors and is too costly as a repeated generic gate.
That does not invalidate fixed finite Meyer acceleration: matching the full
ordinary trajectory closely is not necessary for improving its objective.
This failed gate is not promoted as a replacement accelerator.

The Huber class permits a stronger result. In a saturated region it discovers
`J=I`, the rank-one residual space, and the first piece-crossing time. Positive
camera-derived sensing controls at n=64 have first jumps of 392, 366, and 386
steps, with one map and one tangent action. Independent ordinary replay
matches to about 1e-11. Anderson with a history confined to an exactly constant
residual region has zero residual differences and takes only an ordinary step
under the implemented positive-ridge convention. This establishes a local
collapse of acceleration, not a claim that Anderson fails on all structured
problems or an exact reproduction of the earlier subproblem Meyer stall.

The Huber data use noiseless synthetic sensing matrices and a downsampled
camera source; these are camera-derived inverse problems, not a natural noisy
benchmark. The objective has known optimum zero. The target ratio is 1e-3.
The fixed polynomial, ordinary method, certified discovery, and full-state
Anderson share the same map. The certified method tries depths 2/4/8, caps
horizons at 4096, uses a 2% displacement-relative certificate, and takes 16
ordinary steps after an unprofitable attempt. This explicit acquisition
schedule is experimental. All checks and failed attempts are timed.

`results/discovery_v1.json` holds the sampled screen and 512-work-unit solves.
`results/certified_huber_v1.json` retains the initial scalar reduced-path loop;
`certified_huber_v2.json` uses block doubling and batched feature checks with
unchanged mathematical admission conditions. `certified_huber_v2_n256.json`
extends v2 to n=256 without changing its accelerator parameters. All reported
speed comparisons must include full-solve time and cannot be inferred from
the first exact event jump alone.

Reproduce on the selected Mini (copy each /tmp result back immediately):

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.krylov_bregman.test_contracts \
  experiments.krylov_bregman.test_discovery \
  experiments.krylov_bregman.test_certified_piece -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.discovery_study \
  --out /tmp/krylov_discovery_v1.json --size 32 --seeds 2

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.certified_study \
  --out /tmp/krylov_certified_huber_v2.json --seeds 3 --repeats 3

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.certified_study \
  --out /tmp/krylov_certified_huber_v2_n256.json --side 16 \
  --seeds 3 --repeats 3 --budget 80000
```

The next major question at that stage was a discovery certificate for curved proximal pieces
and lifted observables. An unchanged exterior disk mask is not an affine
piece. Extending the Huber theorem to that case requires new geometry, not
relabeling an existing mask check as a certificate.

The complete continuation suite passes **32 tests** on the M4 Mini. At n=256,
positive sensing, protocol v2 measures paired median speedups of **3.47x**
versus ordinary iteration and **1.42x** versus the fixed finite schedule.
Anderson wins these complete solves after the initial drift phase. On signed
sensing the certificate is expensive and the discovered method loses; both
results remain in the manuscript table. Rebuild the continuation tables and
figure with `.venv-jpeg/bin/python -m experiments.krylov_bregman.discovery_report`
(using the MacBook's existing environment as above) before compiling TeX.

At n=256 the first positive-sensing skips are 838, 831, and 821 ordinary
steps for two map/tangent work units, with absolute replay error below 3.7e-11.
The recorded Anderson trajectories include gap excursions of 25.26x and
7.35x the initial gap on seeds zero and one before recovering. Final time
alone omits this behavior; the paper preserves it explicitly. Recorded
checkpoints are not a bound on every intermediate objective value.

## Curved Meyer continuation: exact validity beyond affine pieces

`curved_meyer.py` applies to the original coupled recurrence in
`experiments/meyer_transport_audit/model.py`. It does not replace either disk
projection or remove a vector memory. The exact future-driving state is
`(s=u+w, t_u, t_w)`. The emitted texture is already `f-s`; the current primal
difference is forgotten, but an actual next pass recovers both next primals
exactly. Independent tests compare complete future outputs.

The quotient is globally nonexpansive, and in fact 2/3-averaged, in
`||s||² + 2||t_u||² + 10||t_w||²`. The proof factors each branch into a firmly
nonexpansive graph/projection map and then combines their energy inequalities.
It holds for arbitrary input states, not only a proximal trajectory.

For a reduced direction, every disk's squared radius is a quadratic form in
Krylov coordinates. The implementation prepares normal/tangential linear
forms and evaluates the exact projection remainder, including both inward
and outward crossings. Exterior radial motion is distinguished from turning.
The full map's nonlinear remainder is a fixed linear response to these two
projection remainders. Its Fourier operator norm is computed once from the
grid and parameters (1.14819 on the retained 64-square grid), independently of
the image. Path validation therefore needs **zero additional screened solves**.
It still has pointwise/coordinate evaluation cost, which is timed.

The sum of curvature and Arnoldi compression bounds controls the actual
finite trajectory in the nonexpansive quotient norm and bounds emitted
texture error. This is an exact-arithmetic theorem with floating-point
verification, not an outward-rounded numerical certificate.

Retained evidence:

- `curved_meyer_v1.json`: 240 predictions on camera, Barbara, a permuted
  camera, a straight carrier, and crossing carriers/edge, size 64; ordinary
  prefixes 4/32/128/512; depths 2/4/8; horizons 8/16/32/64. No measured bound
  violation. Acquisition plus full-horizon checks/reconstruction are timed.
- `curved_discovery_v1.json`: live order/horizon discovery at the twenty
  source/prefix anchors, five timing repeats. Seven anchors admitted. Late
  camera selects depth 2/horizon 16 and is 1.39x faster than replay; coherent
  carrier anchors select depth 2/horizon 64 and achieve 3.47x/3.48x. Three
  other admitted blocks fail to amortize in time despite nominal work savings.
- `curved_solve_v1.json`: complete same-map solves at a primal-dual gap ratio
  target of 1e-4, budget 4096 map/tangent units, three shuffled timings.
  The discovered policy loses to ordinary iteration on all five sources.
  It uses a 10% displacement-relative sufficient path bound and a 32-step
  ordinary cooldown after failed acquisition. Failed probes, geometry checks,
  output-recovery steps, and gap checks are charged. The six-field fixed
  polynomial and fixed quotient polynomial remain separate comparators.

These results establish the curved validity description and some local
amortization; they do not establish a generally faster complete solver.
The conservative gate is a research instrument, not a promoted replacement
for the native Meyer operator. Whole-trajectory fidelity remains sufficient
for this certificate but is not necessary for useful objective acceleration.

The complete research suite now passes **41 tests**, including sharp
attainment of the Fourier gain, the averaged inequality, crossing formulas,
quotient output equivalence, and actual trajectory-error bounds.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.krylov_bregman.test_curved_meyer \
  experiments.krylov_bregman.test_contracts \
  experiments.krylov_bregman.test_discovery \
  experiments.krylov_bregman.test_certified_piece -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.curved_study \
  --out /tmp/krylov_curved_meyer_v1.json --size 64 --repeats 3

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.curved_discovery_study \
  --out /tmp/krylov_curved_discovery_v1.json --size 64 --repeats 5

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.curved_solve \
  --out /tmp/krylov_curved_solve_v1.json --size 64 --repeats 3 \
  --budget 4096 --target 1e-4
```

Copy each remote JSON immediately into this experiment's `results/` folder.
Render locally with the existing environment:

```sh
/Users/ultimussecundai/bfft/.venv-jpeg/bin/python \
  -m experiments.krylov_bregman.curved_report
tectonic --keep-logs --outdir output/pdf paper/krylov_bregman/main.tex
```

The next cost questions are acquisition scheduling and how much structure is
lost when spatial curvature responses are reduced to norm bounds. Capturing
the direction and cancellation of those responses may improve the certificate,
but its extra discovery cost must be measured; it is not implemented here.

## General geometry and observable discovery

Section 10 now separates geometry compatibility, observable recurrence, and
acquisition cost. For smooth interior mirror descent a prescribed map F admits
an objective precisely when H_h(F(x)) DF(x) is symmetric (on a simply connected
domain). Convexity additionally requires H_h(x)-H_h(F(x)) DF(x) to be positive
semidefinite on a convex domain. This is a general test, independent of Meyer.

`general_geometry.py` implements a compatible nonlinear rational dual family,
with four mirrors and dense primal mixing. It supports snapshot recurrence
identification and synthesis of cross-ratio coordinates from degree-(1,1)
rational relations. The latter does not receive the rates or fixed points.
It does receive the known mirror factorization and a finite rational grammar.
Structural identities certify the family; fitting alone is not certification.

Eight transitions, 24 dimensions, three seeds, and four geometries give a
maximum relative primal error of 2.88e-11 at horizon 128 for synthesized
cross-ratio transport. The supplied reciprocal dictionary recovers rank four;
its maximum error is 2.45e-8. Raw dual fitting reaches 4.53 relative error.
These are constructed controls, not general coverage or speed claims. Timings
are recorded for diagnostics without a repeated performance gate. The complete
research suite passes 51 tests. `general_geometry_v1.json` precedes chart
synthesis; use `general_geometry_v2.json` for the paper table.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest discover -s experiments/krylov_bregman -t . -p 'test_*.py' -q
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.general_geometry_study \
  --out /tmp/krylov_general_geometry_v2.json
```

Copy the JSON immediately to `experiments/krylov_bregman/results/`, then run
`python3 -m experiments.krylov_bregman.general_geometry_report` locally.

## Sinkhorn transfer experiment

The real-problem transfer test is entropic optimal transport on irregular point
supports. See [SINKHORN_FINDINGS.md](SINKHORN_FINDINGS.md) for the exact objective,
protocol, full timing scope, all successes and losses, and comparison with
Anderson. Fixed polynomial transport speeds up all 24 tested cases (median
2.40x); existing discovery speeds up 19 (median 1.18x). The coordinatewise
rational chart does not transfer reliably. This is a separate experiment;
it does not change the open paper or the original Meyer solver.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest discover -s experiments/krylov_bregman -t . -p 'test_*.py' -q
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.sinkhorn_study \
  --out /tmp/krylov_sinkhorn_full.json
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.sinkhorn_chart_probe \
  --out /tmp/krylov_sinkhorn_chart_probe.json
```

Copy each JSON immediately to this experiment's `results/` directory as
`sinkhorn_full.json` and `sinkhorn_chart_probe.json`. Render with:

```sh
/Users/ultimussecundai/bfft/.venv-jpeg/bin/python \
  -m experiments.krylov_bregman.sinkhorn_report
```

## Sinkhorn/Meyer structural difference and mutations

See [MUTATION_FINDINGS.md](MUTATION_FINDINGS.md). The positive Sinkhorn tangent
is contrasted with the rotating primal/vector-memory branches of Meyer.
Implemented mutations retain exact disk nonlinearity in a reduced recurrence,
enrich its basis from a live defect, or infer finite coupled increment transport
from ordinary history. None is an accepted faster Meyer implementation.
Defect enrichment improves local prediction but loses complete cost; increment
learning speeds up two Sinkhorn controls and rejects nearly all Meyer proposals.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest discover -s experiments/krylov_bregman -t . -p 'test_*.py' -q
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.structure_probe --out /tmp/krylov_structure_probe.json
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.phase_probe --out /tmp/krylov_phase_probe.json
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.factor_study --repeats 3 --out /tmp/krylov_factor_full.json
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.krylov_bregman.increment_study --repeats 3 --out /tmp/krylov_increment_full.json
```

Copy each JSON immediately into this experiment's `results/` directory,
retaining the filename. Render with the MacBook `.venv-jpeg` Python using
`-m experiments.krylov_bregman.mutation_report`.
