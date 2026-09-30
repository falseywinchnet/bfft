# Krylov transport in mirror geometry

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
The retained verification run passed all 17 tests, covering tangent actions,
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
