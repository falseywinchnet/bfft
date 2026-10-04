# Restart from what actually transports

The user's September 30 correction rejects BFGS and potential preconditioning
as the active approach. An optimizer helped by a local response model is not
an answer to the transport question. The unfinished amortization sources were
removed from active source and archived, including their negative screens.
Committed earlier reports remain historical records, not endorsed directions.

## What the Meyer construction carries

The early record `notes/meyer_bregman_ladder.md`, section 3, distinguishes the
moving ROF target and its current flow from warm Split-Bregman `(d,b)` state.
Carrying momentum across the changing outer target was harmful; retaining
compatible split/flux state and interleaving work survived. The learned state
is not a predictor of a fixed target that the outer pass has already replaced.

The later accelerator in `notes/meyer_finite_flow_jump.md` carries the full
coupled state `(u,w,t_u,t_w)` and advances a finite displacement of the fused
map. Its local approximation is `sum_{j=0}^{H-1} A^j r`, with ordinary passes
refreshing the nonlinear projection geometry. It targets a finite fused
state, not a stationary solution of a surrogate. These are related but
separate mechanisms; neither licenses fitting the entropic potential with BFGS.

## Entropic distinction and immediate experiment

The ordinary recurrence is

    u = a / (K v),
    v_next = b / (K^T u).

The two reciprocals are evaluated in full at every pass. The current
normalized conditionals and `J=QP` change as the state changes. A previous
local rule cannot simply retain its authority. The fixed problem relation K
and its measured linear actions do persist.

The experiment therefore stores exact-arithmetic action pairs of the fixed
primitive, while every new request and every normalization remains a full
vector. No gradient, Hessian, BFGS memory, objective line search, optimum,
or future trajectory is used. This is a first test of the distinction, not a
claim that primitive caching already reproduces Meyer's finite-horizon jump.
It still executes each cheap nonlinear pass; it tries to avoid repeated dense
kernel products inside those passes.

A diagnostic uses the same ordinary trajectory: after 16 passes it compares
the initial `J0` and current `J16` acting on the CURRENT update direction.
Across the six n=128 cases the relative discrepancy is 9.3–23.0%; across the
six n=1024 cases it is 6.5–16.0%. These are particular directional defects,
not universal lower bounds or proofs that J cannot be evolved. The exact
finite scaling relation below reproduces the current P and Q to at most
5.56e-16 across all twelve cases. Those full-matrix diagnostics are offline
and are not charged as free algorithmic observations.

## Carried primitive response and its changing frame

At a positive anchor x0 let k0=K x0 and define the stochastic operator

    P0 = diag(1/k0) K diag(x0).

Retain orthonormal input columns Q and measured output columns Y=P0 Q.
For a new positive request x, normalize v=x/x0 by a common scalar and write

    v = Q c + r.

The represented response is Y c. If the missing input r is too large, measure
its response directly with one K product, add its normalized direction and
response, and reconstruct the current product. This is an operator action
measurement, not a secant fit of the evolving nonlinear map. Measuring the
missing action directly avoids subtracting nearly equal positive outputs.
A finite cache and rebasing keep the representation bounded.

At a new anchor x1 with known k1=K x1, the fixed relation gives exactly

    P1 = diag(k0/k1) P0 diag(x1/x0),
    Q1 = diag(x0/x1) Q,
    P1 Q1 = diag(k0/k1) Y.

Thus the response pairs themselves can be transferred between frames with
no new kernel product. Orthogonalizing a transferred input column requires
applying the same linear combinations and normalization to its output column.
The optional `--transport` arm tests this whole-basis transfer. Exact algebraic
transfer does not ensure that the old span covers new requests economically.
The default arm rebuilds its cache when its input log-ratio range exceeds 8
or the rank budget of 32 is reached. This fixed range prevents catastrophic
cancellation in unbalanced representations; it is not a classification of the
source or a change to the problem.

## A finite-trajectory enclosure

Let vhat=Q c. Compute the entirely observed input residual and

    eta = max_j |v_j-vhat_j| / v_j.

Positivity implies

    |P0 v-P0 vhat| <= eta P0 v.

For eta<1, the Hilbert log-distance between the exact and represented output
is at most `log(1+eta)-log(1-eta)`. Positive linear maps are nonexpansive in
that distance; componentwise reciprocals and multiplication by a fixed
positive marginal preserve it. Summing the two half-pass bounds therefore
bounds the difference from the ORIGINAL ordinary finite trajectory. This is
not a new objective or a trajectory redefined to fit the representation.

The per-query threshold is 1e-8 in the retained experiment. Acquisition steps
are exact in real arithmetic and contribute zero approximation error. Every
accepted reuse contributes its measured bound. The implementation uses
ordinary float64, not outward-rounded intervals. Rounding error and retained
response conditioning are not included in this analytic bound, so it is not
a rigorous machine certificate. Independent final trajectory comparisons and
focused representation tests check its numerical behavior.

## Retained results

Both arms execute 256 passes from v=1 on identical problems. Ordinary uses
512 kernel products. The kernel is materialized identically before timing;
all cache setup, frame handling, measurements, reuses, reciprocal updates,
and trace storage are inside the carried timer. The independent geometry and
final trajectory scoring are outside both timers. Three same-Mini repeats
with one BLAS thread yield these medians at n=1024:

| Seed | epsilon | Kernel products / 512 | Ordinary ms | Carried ms | Speed ratio | Final log-state error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01 | 94 | 59.339 | 31.595 | 1.878x | 3.05e-08 |
| 0 | 0.003 | 253 | 59.630 | 52.756 | 1.130x | 1.89e-08 |
| 0 | 0.001 | 442 | 62.454 | 77.979 | 0.801x | 1.65e-08 |
| 1 | 0.01 | 62 | 59.539 | 29.255 | 2.035x | 1.37e-08 |
| 1 | 0.003 | 193 | 60.891 | 45.789 | 1.330x | 3.18e-08 |
| 1 | 0.001 | 433 | 59.432 | 73.115 | 0.813x | 2e-08 |

The error is `osc(log(v_carried)-log(v_ordinary))` at pass 256, not a marginal
residual, objective gap, or error to an unknown optimizer. Every error is below
its accumulated exact-arithmetic trajectory envelope in these floating tests.
Four cases improve complete time; the two epsilon=.001 cases regress. At
n=128 every case is slower despite saving products: the cache overhead costs
more than the small dense products it avoids. This is not a general speedup
and is not a convergence-to-1e-9 timing comparison.

The initial unbalanced carrier suffered cancellation and produced NaNs on
four cases; that failed screen is preserved as `primitive_transport_first`.
The first balanced implementation derived a new response by subtracting two
nearby outputs; this wasted reuse near numerical dependence. The final code
measures the missing action itself. Intermediate screens are retained with
distinct filenames and are not used for the final table.

## Rejected whole-basis frame transfer

The final `--transport` arm was also run at n=1024 with the direct missing-action
measurement. It is NOT validated by the exact-arithmetic identity alone. Four
hard cases acquired 992–1002 products, repeatedly hit the negative-output
fallback, and took 1.37–1.40 seconds instead of about .06 seconds. Their finite
state discrepancies ranged from 3.64e-7 to 1.94e-3 despite zero recorded reuse
error. These are failures of the floating representation, not successful
zero-error runs. Near-dependent basis transformations amplify existing image
roundoff; transferring that corrupted image again perpetuates the error.
The simple random frame test passes but does not cover those ill-conditioned
sequences. The default cache-reset arm does not carry those corrupt response
images through successive frame changes.

This failed arm is retained explicitly for diagnosis in
`primitive_transport_carried_final1024.json`. Its `trajectory_bound` covers
only omitted input residuals in exact arithmetic and cannot certify its
roundoff-corrupted acquired responses. It must not be promoted or used as the
accuracy evidence for the default arm. Stable economical relation transport
through strong frame changes remains an open part of this construction.

## Tests and scope

Three focused tests passed on the M4 Mini: exact response transfer between
frames and orthogonality; an accepted reuse and its output enclosure; and
complete finite trajectories with both cache-reset and frame-transfer arms.
No production optimizer or Meyer schedule was modified.

Code: `primitive_transport.py`. Reproduction: `probe_primitive_transport.py`.
Tests: `test_primitive_transport.py`. Final records: `results/theory/primitive_transport_final128.json`
and `primitive_transport_final1024.json`.

The surviving research question is now representational: which carried
primitive-response directions continue to cover the FULL newly rebuilt
requests across strong scaling changes? The frozen potential rule, approximate
inverse, and generic descent controller are no longer the active objects.
