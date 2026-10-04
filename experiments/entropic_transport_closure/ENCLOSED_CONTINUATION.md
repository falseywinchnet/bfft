# Joint response and remainder enclosures for finite continuation

This is the requested implementation of the idea prompted by *The Zonotopic
Mixture Filter* (Gonzalez, Cedeno, Puig, arXiv:2608.17897v1). It carries a
measured reduced response together with an inclusion-preserving enclosure of
its unresolved nonlinear change, and admits internal moves using the worst
allowed response. No exact gradient or objective is evaluated at the internal
points to authorize them. Full evaluations occur at block endpoints, where the
representation is refreshed and the original marginal stopping test is applied.

This is an experimental optimizer for the original transport objective. Its
trajectory changes; it does not reproduce a specified number of ordinary
Sinkhorn iterations. It provides an exact-arithmetic descent inequality, not
an outward-rounded floating-point certificate. The distinction is visible in
the retained independent numerical audit and is not hidden by its tolerances.

## 1. The retained higher state

At an exactly evaluated anchor y0, define

    P_ij = K_ij exp(y0_j) / sum_l K_il exp(y0_l),
    F(y) = sum_i a_i log sum_j K_ij exp(y_j) - b.y.

Use mass coordinates z=sqrt(b)*y, and choose a gauge-free orthonormal chart
U with r columns. The default chart includes the current mass-coordinate
gradient, then orthogonalized recent state and gradient differences. It uses
already acquired motion, not an optimum, spectrum, or future trajectory.

Let V=diag(1/sqrt(b)) U. The row vectors v_j of V are the feature points of
the possible destination indices. Define

    f(s) = F(y0+V s),
    g0 = V^T(c-b),       c=P^T a,
    A = V^T diag(c) V - (P V)^T diag(a) (P V).

Thus g0 and A are the actual gradient and Hessian of f at zero. The only new
kernel products needed to acquire A in this chart are P V, one product per
column, which the implementation batches. No inverse Hessian is formed.

The higher state is the chart, coordinates s, measured response (g0,A), and
a geometric enclosure of the feature cloud. Its central gradient prediction
is g0+A s; the changing set around that prediction carries the nonlinear part
that the retained response has not resolved. The representation is exact at
s=0 and its uncertainty grows with displacement.

## 2. A nonlinear gradient enclosure

Write

    R(s) = max_j (V s)_j - min_j (V s)_j,
    E(s) = s^T A s,
    phi2(R) = (exp(R)-1-R)/R^2,        phi2(0)=1/2,
    lambda(s) = E(s) phi2(R(s)).

Then for every chart direction d,

    [grad f(s) - (g0+A s)].d <= lambda(s) osc(V d).        (1)

This is a global finite-displacement inequality; it does not assume that the
actual gradient remainder is Gaussian or that the current leading mode keeps
its identity.

Proof. For a single conditional row, put h_j=(V s)_j and w_j=(V d)_j.
Along its exponential tilt p_t,j proportional to P_ij exp(t h_j), the third
mixed derivative of its log partition function is

    E_t[(w-E_t w)(h-E_t h)^2].

Its absolute value is bounded by osc(w) Var_t(h). The density ratio between
p_t and p_0 is at most exp(t osc(h)); minimizing the squared deviation over
its center therefore gives

    Var_t(h) <= exp(t R) Var_0(h).

Sum the inequalities with the authentic source weights a_i and integrate
(1-t) from 0 to 1. The integral of (1-t)exp(tR) is phi2(R), while the sum of
anchor variances is E(s). This proves (1), including its negative-direction
counterpart. No runtime reference solution is involved.

Equivalently, with D=conv{v_j}-conv{v_j},

    grad f(s) in g0+A s + lambda(s) D.                    (2)

The support function of D is precisely osc(Vd). This gives a compact way to
retain all feature points as a control, without building convex-hull facets.

## 3. Inclusion-preserving compression

If the feature cloud is enclosed in the zonotope c+G[-1,1]^q, then

    D subset 2G[-1,1]^q,
    grad f(s) in g0+A s + 2 lambda(s) G[-1,1]^q.           (3)

We compare three implementations:

* **Exact feature support:** evaluate osc(Vd) using all feature points. This
  retains the exact support of D, but (1) is still an upper bound on the true
  nonlinear remainder. 'Exact' never means the true future gradient is known.
* **Box:** enclose the cloud by coordinate minima and maxima, using r diagonal
  generators. The support penalty is 2 lambda ||G^T d||_1.
* **Merged zonotope:** merge point zonotopes in a balanced tree, pairing
  neighbors in the first chart coordinate. The enclosing merge is equation
  (41) of the supplied paper. Cap generators at 2r by retaining the largest
  generators and boxing the others using their absolute row sums. Both
  operations preserve inclusion. This uses the paper's merge construction;
  it does not implement its all-pairs greedy selection algorithm.

For two zonotopes the merged center and generators are

    c12=(c1+c2)/2,
    G12=[G1+G2, G1-G2, c1-c2]/2,

after zero padding unequal generator counts. For example, any point
c1+G1 u is represented using coefficients (u,u,1); the second component uses
(u,-u,-1). This explicitly verifies containment. Boxing discarded generators
preserves their Minkowski sum because each coordinate is bounded by its
absolute row sum.

These are bounds on deterministic approximation uncertainty. We introduce no
mode probabilities or posterior weights. The source contributions coexist
and are already combined with a_i. A single enclosing zonotope is used for
the feature geometry; this prototype is not the full mixture-history filter.

## 4. Certifying a finite internal step

The exact Hessian comparison under exponential tilting gives

    d^T Hess f(s+t d) d
      <= exp(R(s)+t R(d)) d^T A d.

Combine that with (1) or (3) and integrate the second-order remainder:

    f(s+d)-f(s)
      <= (g0+A s).d
         + lambda(s) width(d)
         + exp(R(s)) phi2(R(d)) d^T A d,                  (4)

where width(d) is osc(Vd) or 2||G^T d||_1. A negative right-hand side
certifies a decrease of the original objective in exact arithmetic.

The reduced iteration chooses a negative central gradient with a scalar
secant scale; it uses no linear-system solve. It first requires the uncertain
directional derivative to retain 5% of the nominal descent margin, then
backtracks using (4), without kernel queries. It stops to refresh when the
uncertainty blocks the direction, the central gradient is small, the next
step is numerically negligible, or a ceiling is reached. A numerical radius
ceiling of 20 and an internal ceiling of 64 prevent runaway computation.
Neither ceiling relaxes the descent inequality or final stopping condition.

We require at least two admitted inner steps to use a proposed block. An
unproductive acquisition falls back to the established scalar controller and
waits four ordinary outer steps before trying another chart. There are eight
initial scalar steps. All fallback work, rejected acquisitions, compression,
inner steps, final independent log-domain verification, and history storage
are charged to the complete solve.

## 5. What is tested and what is scored independently

The primary battery is the unchanged n=128 source generator with seeds 0,1
and epsilon=.01,.003,.001, initialized at y=0. Every method is stopped using
max_j |c_j/b_j-1| <= 1e-9, the same criterion as the preceding context-descent
experiment. Final results are checked independently in log arithmetic.

The final four-coordinate experiment has five complete timing repeats on the
M4 Mini, with one BLAS thread. The two-coordinate screen is retained separately.
Controls are lean ordinary BLAS Sinkhorn and the evolving scalar controller.

After the solver returns, a separate scorer evaluates every admitted internal
point using the full log-domain conditional distributions. It records actual
objective changes and violations of both directional and finite-step bounds.
These expensive scoring operations are outside the solve timer and their
values never inform acquisition, proposal, acceptance, or stopping. They are
validation costs, not hidden algorithmic work. The score is retained rather
than summarized as a categorical floating-point 'certificate'.

The focused tests check chart gradients and curvature, finite nonlinear
support inequalities in random directions, merge and generator-reduction
containment through support comparisons, full convergence, and independent
internal-step descent. The exact-arithmetic proofs above establish their
scope; finite tests are not offered as proofs for untested inputs.

## 6. Development screens

The initial four-coordinate screen acquired a Krylov chart using two kernel
products per column and imposed radius 2. It spent too much time in tiny
steps near that radius. That failed-cost screen is retained, but it is not
the final algorithm or timing evidence. The current implementation acquires
its chart from observed motion, needs one product per column, and stops
negligible steps before refreshing. The unchanged inequality (4) still decides
whether every admitted move is allowed. The two-coordinate snapshot screen
and final four-coordinate snapshot battery separate chart richness from
compression effects.

## 7. Measured result

All 30 retained method/case results and all stored repeat residuals meet the 1e-9 marginal tolerance. Five complete repeats determine the following medians. The exact evaluations include initialization and the final independent log-domain check.

| Seed | epsilon | Scalar exact evaluations | Box exact evaluations | Box internal steps | Scalar ms | Exact support ms | Box ms | Merged ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01 | 50 | 37 | 172 | 2.203 | 9.766 | 8.868 | 24.496 |
| 0 | 0.003 | 112 | 76 | 516 | 4.857 | 31.199 | 26.597 | 62.042 |
| 0 | 0.001 | 269 | 160 | 1140 | 11.288 | 68.936 | 58.083 | 154.351 |
| 1 | 0.01 | 43 | 35 | 151 | 1.857 | 8.494 | 8.046 | 19.984 |
| 1 | 0.003 | 104 | 69 | 378 | 4.442 | 26.446 | 20.624 | 51.207 |
| 1 | 0.001 | 232 | 141 | 862 | 9.803 | 53.942 | 48.208 | 124.428 |

The separate scorer checked 9,983 internal steps across the three enclosure variants. It found no objective increase above 1e-24. Strict floating-point bound inequalities are not exact: the largest positive finite-increment discrepancy is 3.68e-23, and the largest directional-support discrepancy is 4.72e-16. Near stopping, even a tiny absolute discrepancy can be a sizable fraction of the predicted margin; the raw records retain those relative discrepancies. Therefore these results support the exact-arithmetic derivation numerically but do not provide an outward-rounded machine certificate.

The box variant uses fewer exact state evaluations in all six cases, but its complete cost is higher than the evolving scalar control in all six. Those evaluation counts omit neither construction nor its kernel work from the timing: chart acquisition adds up to four kernel-vector products per refresh, and internal support tests also cost time. Reduced state-evaluation count alone is not a speedup.

Removing feature compression does not yield a decisive improvement. The exact-feature control still uses the same analytic nonlinear remainder bound; it is not an exact-future-gradient oracle. The merged enclosure is more expensive and does not consistently need fewer refreshes. The new conservative representation succeeds as a finite descent mechanism; it is not promoted as a competitive accelerator.

The rank-two snapshot screen uses fewer small-matrix operations and cheaper acquisition, but usually refreshes more frequently. It also loses to the scalar controller on all six cases. Its one-repeat times are a screen, not the final timing comparison.

### One larger cost check

A separate n=1024, seed=0, epsilon=.001, rank-two run with three timing repeats tests whether a more expensive kernel changes the cost picture. This is an additional source problem, not one of the six originals. All methods converge. Median complete times are ordinary Sinkhorn 717.30 ms, scalar 118.51 ms, exact feature support 411.01 ms, box 267.70 ms, and merged 1167.65 ms. The box beats ordinary iteration but still loses to the scalar control. Thus the negative comparison cannot be attributed only to the tiny n=128 kernel.

### Scope of the outcome

The experiment supplies a proved finite nonlinear response enclosure, sound geometric compression, and real blocks of unevaluated descent on the original problems. It does not establish an efficient long-horizon replacement, a probabilistic coverage guarantee, or a general impossibility result. The present response model is first order with a second-order uncertainty envelope. Acquisition amortization, chart adequacy, and tightness of the analytic remainder all remain possible improvement targets; geometric compression alone did not resolve the cost problem.

Code: `enclosed_continuation.py`; reproduction: `probe_enclosed_continuation.py`; focused tests: `test_enclosed_continuation.py`. Raw final and screen JSONs are in `results/theory/enclosed_continuation_*.json.gz`.

The full internal coordinate and feature traces are retained losslessly in gzip
JSON files. Decompress before reading with ordinary JSON tooling; the probe
writes gzip directly when its output path ends in `.gz`. All 21 focused and
neighboring regression tests passed on the Mini, recorded in
`results/theory/enclosed_continuation_verification.txt`.

In the table order, box acquisition plus evaluation uses respectively
188, 422, 910, 176, 372, and 804 kernel-vector products, versus
153, 345, 829, 128, 319, and 716 for the scalar control. These counts exclude
the common final log-domain check and all offline scoring. Batched products
can have different wall cost, but the extra work shows why counting only
exact state evaluations would give a misleading acceleration claim.
