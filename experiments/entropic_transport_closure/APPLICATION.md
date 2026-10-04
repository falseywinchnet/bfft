# Applying evolving transport closure to Sinkhorn

September 30, 2026. This applies the preceding formal analysis to actual
positive-kernel Sinkhorn problems. It preserves the ordinary Meyer
construction as the reference; no Meyer code or schedule is changed.

## Measured result

The exact proportional-block family supports a useful projective-power
jump: 4.7–6.9x faster for block coupling .03 and 56.9–73.8x faster for
coupling .001 than ordinary iteration using the SAME exact block reduction.
These are specialized structured problems, not a general dense-kernel claim.

For general kernels, a reduced quadratic model carries the leading change
of the local transport rule. It improves state prediction in 39/52 paired
diagnostics and derivative prediction in 36/52. Two unrestricted quadratic
rollouts do not complete; the bounded solver does not admit those advances.
The general model is approximate and its error includes subspace escape.

The two-direction changing-rule model is 1.13–1.33x faster than ordinary
Sinkhorn on the three narrow-kernel cases at size 1024, and 1.16–1.28x at
size 2048. It loses on the 256-point cases. The four-direction frozen-rule
control is faster than either changing-rule version on every nontrivial
2048-point case. The evidence supports a useful evolving model and some
acceleration over ordinary iteration, but not a cost advantage for carrying
the changing rule over the matched frozen-rule approach.

All 720 timed general solves and 120 structured solves reach actual relative
marginal tolerance 1e-9. The worst general final residual is
9.995434480103427e-10. Sixteen mathematical and application tests pass on the
M4 Mini system Python. There is no fixed-point oracle in any timed solver.

## Algorithms that were actually implemented

`sinkhorn.solve(K,a,b,method=...)` returns log column scaling, row and column
scaling factors, a measured marginal residual, costs, and a convergence flag.
It returns factors of a coupling, not a materialized dense coupling. All
methods use the same row-normalized output phase and stopping measurement.

* `ordinary`: the original alternating row/column scaling map.
* `linear`: a four-direction finite polynomial of the frozen local rule.
* `quadratic`: four retained directions plus the actual conditional-moment
  tensor that changes the rule as the model state moves.
* `quadratic2`: the same changing-rule construction with two directions,
  reducing acquisition cost. This is an explicit capacity/cost variant.

The default remains `ordinary`; no automatic universal selector is claimed.

Every model uses the current coupling column masses c as the metric.
Starting with the current displacement, weighted Krylov acquisition builds
U with U^T diag(c/sum(c)) U=I and c^T U=0. Conditional products use the
original K; no dense Jacobian or conditional matrix is formed.

Let J=QP at the anchor. For directions h,k, the second derivative is

    D²F[h,k] = Q P(h*k) - 2 Q[(P h)*(P k)]
                + (Q P h)*(Q P k).                 (1)

Every vector on the right is computed with the actual current conditional
operators. Retain the complete output vectors of J U and D²F[U_i,U_j],
as well as their weighted projections. The reduced model is

    z_next = r + H z + 1/2 G[z,z],
    D(model)(z) = H + G[z,.].                       (2)

The rule therefore changes during the finite advance. The covariance terms
are not replaced by a scalar relaxation factor. The state and derivative
remain approximate: (2) truncates the nonlinear primitive at second order
and restricts the motion to U. The complete-vector defect is charged below.

The linear ablation sets G=0 with the same four-direction acquisition. It
is a numerical control, not a substitute for the coupled principle being
tested. `quadratic2` and `linear` have different capacities; comparisons of
the effect of curvature itself use `quadratic` versus `linear` at depth four.

## A whole-state bound, including both sources of approximation

For an arbitrary displacement h, put R=osc(h). Along y+t h, each row
half-step x_i has

    x_i' in [-max(h),-min(h)],
    x_i'' in [-R²/4,0],
    |x_i'''| <= R³/4.

The last inequality uses |third central moment| <= R*variance <= R³/4.
Differentiating the second log-sum-exp gives three terms: an expected third
derivative, three times a covariance of first and second derivatives, and
a third central moment of first derivatives. The covariance is bounded by
the product of the two ranges divided by four. Consequently

    |d³ F_i(y+t h)/dt³| <= R³/4 + 3R³/16 + R³/4
                         = 11R³/16.

Taylor's integral remainder yields the GLOBAL bound

    ||F(y+h)-F(y)-J h-(1/2)D²F[h,h]||_infinity
       <= 11 osc(h)^3/96.                           (3)

The linear model uses the previously proved osc(h)^2/8 bound.
Use the gauge quotient norm ||h||_q=osc(h)/2. F is nonexpansive in that
norm because its derivative is stochastic. At each modeled step, compute
the complete-vector polynomial result and its difference from the lifted
reduced result U z_next. The quotient norm of that difference is the
projection/escape defect, delta_j. For s_j=U z_j,

    ||F^H(y)-(y+s_H)||_q
       <= sum_(j=0)^(H-1) [delta_j + remainder(s_j)]. (4)

This bounds escape as well as primitive truncation; it does not assume
that the retained subspace is invariant. The inner steps require vector
and small-tensor arithmetic but no K applications. They are not free, and
all are timed. Floating-point rounding is not included in the mathematical
bound; final marginal measurement always uses the actual original kernel.

## Admission, refresh, and cost accounting

Fixed settings selected in the seed-zero screen:

* maximum horizon 128;
* accumulated bound at most .05 times predicted quotient displacement;
* at least 16 modeled passes for `linear`/`quadratic2`, 32 for `quadratic`;
* eight ordinary passes initially and between failed acquisitions;
* immediately refresh after an accepted advance.

The current actual F(y) is already available. If acquisition cannot justify
a long enough advance, the solver takes that ordinary step. A proposed
advance is evaluated with the actual kernel and accepted only if its actual
maximum relative marginal error decreases. Otherwise it takes the cached
ordinary step. Rejected evaluations, acquisitions without proposals, tensor
construction, per-step bounds, reconstruction, and stopping are all charged.
The test suite explicitly exercises rejection rather than relying on every
proposal being successful in the measured examples.

No solve of (I-J)s=d is used. No late solution, true H-step trajectory, or
future geometry is available to the timed model. It learns a current
response representation from actual conditional moments and composes its
finite nonlinear rule. It does not prove autonomous learned closure for
all entropic problems.

The general solver uses normalized exponentials and ordinary scaling
arithmetic for the tested positive kernels. It is an experimental kernel,
not a fully stabilized log-domain implementation for arbitrary extreme
costs, vanishing masses, sparse supports, or underflowed kernels. Nonfinite
scaling raises an error. Unrestricted diagnostic rollouts can diverge; a
numerical guard stops before their cubic bound overflows. That guard was
added after the timed validation and does not change any admitted horizons,
kernel operations, or numerical outcomes in it; the tests were rerun.

## General-kernel validation

M4 CPU, macOS 26.5, Python 3.9.6, NumPy 1.26.4;
OPENBLAS_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. Each method is warmed,
then five repetitions per seed are shuffled within each problem. Validation
seeds are 1 and 2, distinct from the seed-zero timing/configuration screen.
Sizes are 256,1024,2048. The six families are a positive lognormal matrix,
two line Gaussian kernels, and three two-dimensional Gaussian kernels.
Marginals differ between seeds. Kernel creation is common input work and
excluded; kernel validation and all method-specific setup are included.

Stop when the returned row-normalized coupling satisfies

    max_j |c_j/b_j-1| <= 1e-9.

Rows equal a by construction, up to floating-point rounding. This is a
matched marginal-accuracy comparison, not a certified objective-gap or
transport-plan-distance comparison. Raw measurements retain convergence,
residual, evaluations, acquisition count, accepted horizons, rejection
count, kernel calls, and kernel-vector-product equivalents.

Median milliseconds across seed medians at size 2048:

| Problem | Ordinary | Frozen, 4 | Changing, 4 | Changing, 2 |
|---|---:|---:|---:|---:|
| Line, epsilon .02 | 85.66 | 62.29 | 80.11 | 80.22 |
| Line, epsilon .003 | 547.80 | 349.35 | 493.13 | 426.58 |
| Plane, epsilon .03 | 57.37 | 49.11 | 58.56 | 61.41 |
| Plane, epsilon .006 | 281.61 | 191.12 | 276.25 | 243.55 |
| Plane, epsilon .003 | 564.03 | 380.83 | 536.28 | 459.07 |

The positive lognormal control converges before acquisition begins; all
methods execute ordinary iteration there. Its small timing differences are
noise, not acceleration. Ratios in `summary.json` are medians of within-seed
ratios, so dividing the displayed aggregated milliseconds need not give
exactly the same last digit.

At 256 points every changing-rule nontrivial case is slower. At 1024 and
2048, costly kernel applications allow some setup to amortize. For the
2048-point narrow line, median actual evaluations are 198 for frozen,
186 for changing-four, and 193 for changing-two. However their total
kernel-vector products are respectively 648,1114,686. Fewer full passes
alone is not the cost result. Batched matrix products can have a different
cost per vector than separate products; measured complete time is decisive.

## Does the model really improve prediction of the changing rule?

`probe_sinkhorn.py` holds the four-direction basis and anchor fixed. It
compares the linear and quadratic models with independent true trajectories
at horizons 16 and 32, anchors 8,32,128, size 128, seeds 0 and 1. Already
converged anchors are skipped. These diagnostics are entirely outside the
timed solvers and never inform a proposal.

There are 52 paired comparisons. Fifty complete for both models; two
unrestricted quadratic rollouts fail to complete. Among all 52 pairs the
quadratic model improves state error in 39 and the projected evolving
derivative error in 36. No computed finite-trajectory bound is violated.
The derivative comparison uses the same anchor coordinate basis and metric
on the true endpoint derivative. These are projected derivative errors,
not full operator-norm errors in the original dimension.

This demonstrates useful transport-rule information, together with real
failure modes when the model is run outside its admission bound. It does
not show that acquiring that information is cheaper than the frozen model.

## Exact structured application

`solve_blocks` assumes supplied exact structure K_ij=r_i C_g(i),h(j) s_j
with two groups on each side. It aggregates marginals and the initial
column scaling, enters the invariant block manifold, and either:

* iterates the exact reduced two-dimensional ordinary map, or
* advances its projective 2x2 matrix by normalized binary powers.

Both methods pay aggregation, actual reduced marginal checks, and full
scaling-factor reconstruction. The benchmark compares them directly;
neither performs unnecessary dense-kernel operations. The matrix-power
method increases its finite horizon geometrically and stops only after the
actual marginal check. It does not require a fixed point or a root solve.

Across original dimensions 256 through 8192, block coupling .03 gives
4.7–6.9x speedup and .001 gives 56.9–73.8x. The latter ordinary reduced
iterations take roughly 14–15 milliseconds, versus .19–.26 milliseconds
for powers and reconstruction. Relative gains shrink with original size
as reconstruction becomes a larger part of both costs. These are arithmetic
and interpreter timings of this implementation, not universal complexity
constants or comparisons with every available 2x2 solver.

The structured family validates the exact closure mechanism. It does not
demonstrate discovery of block structure on general kernels. The generic
quadratic model uses no supplied block partitions.

## Retained files and reproduction

Implementation: `sinkhorn.py`; tests: `test_sinkhorn.py`, `test_closure.py`.
Runner: `run_sinkhorn.py`; untimed probe: `probe_sinkhorn.py`.
Local report: `report_sinkhorn.py`. Formal groundwork: `FORMAL_ANALYSIS.md`.

Raw validation and plots are under
`experiments/entropic_transport_closure/results/`:
`entropic_sinkhorn_final256.json`, `entropic_sinkhorn_final1024.json`,
`entropic_sinkhorn_final2048.json`, `models.json`, `summary.json`,
`tests.txt`, and `application.png`/`.svg`.
The seed-zero screens remain here as `screen256.json`, `screen1024.json`,
and `screen_depth2.json`; their single repetitions are exploratory only.

Root AGENTS.md records the m4build commands. The retained runs copied only
the authoritative Python files to `/tmp/entropic_sinkhorn_application` on
the host selected by m4host, ran there with the stated thread environment,
and copied each completed result back immediately. No source was edited on
the Mini and no remote software was installed.

Render existing results locally with:

```sh
.venv-jpeg/bin/python -m experiments.entropic_transport_closure.report_sinkhorn \
  experiments/entropic_transport_closure/results
```

The next unresolved cost is acquisition and refresh of useful nonlinear
response directions. The result does not justify throwing away the changing
rule, nor promoting it as the fastest general implementation. The existing
Meyer construction remains unchanged.
