# Compatible continuous current: follow-up findings

2026-09-04. This follows `CONV_FUNCTIONAL_CURRENT_AUDIT.md` and implements a
stronger current proposal, continuous admission, and a continuation constrained
by the existing CONV excursion envelope. It includes unsuccessful ablations.

The subsequent `CONV_JOINT_POTENTIAL_FINDINGS.md` implements the shared 2-D
potential proposed below, with structural derivative traces, continuous
subdivision certificates, and a separate 12-field validation battery.

**The resulting 1-D candidate improves 64 of 68 expanded validation cases.**
Mean interior value error falls by 49.62% on sinusoids, 27.80% on chirps,
36.30% on mixed edge/texture signals, and 9.24% on sigmoids versus CONV.
It preserves the observed derivative-sign order, and its one-factor global
range is contained in that of the compact functional CONV baseline, up to
numerical tolerance. The latter has the same global extrema as original
CONV in exact arithmetic: only already monotone cells differ.

The main remaining representation issue is **joint multidimensional
compatibility**. Sequentially composing improved 1-D operators does not
inherit the excursion comparison to the old 2-D output. The 2-D probe
demonstrates this failure directly.

## 1. Why the next move was global compatibility

The first audit removed an unnecessarily restrictive positivity certificate.
It did not repair the local FIR proposal's high-frequency bias. Previous
repository work already found that tuning the fifth-difference nullspace
coefficients had conflicting optima on different populations.

The new proposal instead reconstructs one globally compatible quintic spline
S through all observed samples, with data-derived first and second endpoint
derivatives. The endpoint derivatives come from the degree-five polynomial
through the nearest six samples. This is deterministic and reproduces
polynomials through degree five; there is no frequency estimate, family
classifier, sharpening coefficient, or ground-truth-dependent selection.

With those endpoint jets fixed, the spline solves the variational problem

\[
\min_S\frac12\int |S'''(x)|^2\,dx,
\qquad S(i)=y_i.
\]

Equivalently, for p = S', it minimizes the squared curvature of the current,
`integral |p''|^2`, while preserving every cell's integrated current. The
Euler equation is degree-five within cells; compatible third and fourth
traces give the classical C4 quintic spline. The quadratic nullspace is
fixed by the nodal observations, giving uniqueness. This is established
spline theory, not a new variational theorem.

The original raw two-jet already shares first and second traces: it is a
continuous C2 piecewise-quintic field before admission. The new condition is
stronger: global variational stationarity and compatible third and fourth
traces give a C4 proposal. This changes the effective passband while retaining
exactly the same five current coordinates per cell. Classical cardinal spline
interpolation already obtains high-quality reconstruction through inverse
filtering rather than a short interpolatory FIR alone; see
[Unser, *Splines: A Perfect Fit for Signal and Image Processing*](https://bigwww.epfl.ch/publications/unser9902.pdf).
The new research combination is this compatible proposal with CONV's ordered
current law, its continuous positivity extension, and the excursion guard.

The endpoint closure matters. Unrestricted six-sample boundary jets can
produce large excursions on oscillatory data. They are not a satisfactory
complete interpolation rule without admission.

## 2. One current state throughout the construction

Let `a` be the compatible spline's current bank and `b` the compact CONV
current bank. Both use

\[
U_i(u)=y_i+\sum_{k=0}^4c_{ik}\tau_k(u),\qquad
\tau_k(u)=\sum_{j=k+1}^{5}B_j^5(u),
\]

with `sum c_i = y_(i+1)-y_i`. The procedure is:

1. Form the sign ledger from the original compact proposal. Keeping this
   ledger fixes the same observed sign-transition allowance.
2. On uniform-sign cells, admit `a` against the actual continuous quartic
   derivative and minimize integrated squared slope error. On mixed-sign
   cells, retain the original signed-coefficient fibre. This preserves the
   original factorwise sign-variation bound.
3. Continue from the compact functional profile toward the admitted compatible
   profile subject to a continuous excursion envelope.

For source cell i, let the source support be samples i-2 through i+3, clipped
to the domain. Define

\[
L_i=\min\{\min_{j\in\mathcal S_i}y_j,\inf_u U_i^b(u)\},\qquad
H_i=\max\{\max_{j\in\mathcal S_i}y_j,\sup_u U_i^b(u)\}.
\]

The final current is

\[
c_i=b_i+t_i(a_i^{\rm admitted}-b_i),
\quad t_i=\max\{t\in[0,1]:L_i\le U_i^{b+t(a-b)}(u)\le H_i
\ \forall u\in[0,1]\}.
\]

The set of feasible t is an interval containing zero. The implementation
finds its endpoint with 48 bisections; every range check evaluates the
polynomial at the endpoints and all real quartic derivative roots. This is
a constraint on the continuous profile before output sampling, not a clamp
on the output raster. Numerical root finding is not a formal interval proof.

### Guarantees in exact arithmetic

* Cardinality and integrated cell current are unchanged.
* On a uniform-sign cell, both endpoints of the current segment have the same
  derivative sign; every convex combination retains it.
* On a mixed-sign cell, both endpoints belong to the original convex signed
  fibre; the interpolation remains in it.
* Consequently no new derivative-sign pair is introduced relative to the
  original ledger.
* Every source value lies in the compact baseline's global range, and each
  baseline cell range does too. Therefore every `[L_i,H_i]` lies inside that
  global range, so the final line cannot increase global excursion beyond
  the compact baseline.

This envelope is a deliberately conservative continuation of CONV. It does
not claim that CONV's existing extremum amplitude is the unknown true one.
It allows improvement without enlarging the existing excursion allowance.
It is also not the full constrained minimum of the fidelity objective:
the last stage is a maximal feasible ray, and that restriction leaves
recoverable fidelity on the table.

## 3. The ablations that mattered

The discovery population contains 56 cases at 25 source nodes and 32-fold
refinement. The expanded validation contains 68 cases at 33 source nodes,
new phases, new sigmoid widths, and additional frequencies including .46
cycles/sample. Ten methods are recorded. Operator choices were not fitted
to analytic truth. Validation inputs did expose a numerical solver tolerance
problem that was repaired before completing the run; this is not a sealed,
preregistered holdout experiment.

**The unbounded compatible proposal has much more detail, but overshoots.**
On the discovery population, admitting that proposal without an amplitude
envelope reduces interior sine NMSE from 4.595e-3 to 6.034e-4 and chirp NMSE
from 5.788e-4 to 3.142e-5. However, whole-line excursions reach 0.867 on a
sine and 0.691 on a chirp, versus 0.033 and 0.049 for original CONV. These
figures rule out adopting the unbounded variant.

**Strict sample-range restriction is too restrictive for oscillatory detail.**
Constraining the continuous curve to the source-support sample range removes
excursion, but a ray from the monotone baseline increases discovery chirp
NMSE to 2.221e-3. Legitimate intersample extrema and movement along the ray
are both restricted. A zero-excursion method can still have poor fidelity.

**The compact excursion envelope recovers a useful compromise.**
It reduces discovery chirp NMSE to 4.424e-4 while retaining the baseline
excursion. The same rule then improves the expanded population below.

**The larger continuous cone is not responsible for all the spectral gain.**
The compatible proposal with the original coefficient admission has very
similar sine/chirp interior errors to the unbounded functional variant.
Global proposal compatibility supplies most of that gain. Continuous
positivity remains essential to exact shoulder reproduction and to avoiding
the basis-induced defect from the first audit.

Changing the mixed-cell ledger to use the compatible proposal did not provide
a compelling improvement and weakened the direct connection to the original
allowance. The selected candidate reuses the compact ledger.

## 4. Expanded validation results

Value errors are compared to analytic truth. The primary region excludes four
source intervals at each boundary; shoulders, sigmoids and steps use a focused
seven-interval transition region. The mean NMSE is normalized by each case's
squared true range on that region. Derivative errors use exact polynomial
derivatives of reconstructed cells and centered differences of the analytic
truth at an offset of 1e-5 source units; discontinuities have no derivative MSE.

Negative percentages mean lower error for the **compatible envelope** variant.

| Family | Cases | Value NMSE vs CONV | Value NMSE vs Lanczos-3 | Slope NMSE vs CONV |
|---|---:|---:|---:|---:|
| Sigmoid | 20 | -9.24% | -15.35% | -7.23% |
| Sine | 20 | -49.62% | -35.24% | -50.43% |
| Chirp | 4 | -27.80% | +206.90% | -36.67% |
| Edge + texture | 4 | -36.30% | +0.25% | -40.64% |
| Step | 4 | -1.04% | -7.15% | — |
| Box | 4 | -1.05% | -6.11% | — |
| Cubic shoulder | 12 | floating-point error | floating-point error | numerical truth-differencing floor |

There are **64 paired value wins and 4 losses**, with an absolute NMSE tie
threshold of 1e-14. The worst relative regression is 4.56%, for a narrow
sigmoid of width .22 and phase .63. Step and box regressions at that phase
are approximately 2.8%. All 20 sine cases improve versus CONV. The displayed
discovery sigmoid at width .7/phase .5 is deliberately retained in the figure:
it is a counterexample to assuming that every sigmoid improves.

Across the complete line, including boundary cells, mean sine NMSE is
0.00992083 for CONV, 0.00685095 for the candidate, and 0.00751348 for Lanczos.
Thus the sine improvement is 30.94% versus CONV and about 8.82% versus Lanczos
when boundary costs are included. Full-line chirp error is **3.20% worse**
than CONV despite its interior improvement. That residual boundary weakness
must not be hidden by the interior metric.

No extra sampled derivative-sign changes occur for the candidate in either
population. The numerical count uses exactly the nonzero source secants used
by the ledger; silently deleting small source secants would change the
theorem being tested. Range checks and the monotonicity argument are more
informative than a sampled sign count alone.

## 5. Shared derivative constraints

A separate 17-node, five-field probe jointly minimizes W1 distance to the
compatible proposal subject to cell masses, the same sign law, and shared
first traces (C1) or first and second traces (C2). Continuous sign constraints
are again separated using polynomial stationary points. This probe has no
excursion envelope, so its results are not guarantees for the selected
envelope candidate.

All 15 combinations completed. C1 eliminates slope jumps on the tested
fields; C2 also eliminates curvature jumps. On the sine probe, interior
MSE changes from 5.303e-5 without shared traces to 3.775e-5 with C1 and
4.914e-5 with C2. On the box it changes from 0.031561 to 0.030396 with C1
and 0.026990 with C2. The sigmoid changes little.

C1 alone can increase the curvature discontinuity: on the step, the curvature
jump rises from 1.766 to 11.751 even though the first derivative becomes
continuous. Smoothness requirements should therefore live in one joint
representation, not be applied successively as independent repairs.
Feasibility of shared traces for every possible original mixed-cell ledger
has not been proved; the implementation checks a linear feasibility problem
and reports failures rather than silently changing the ledger.

## 6. Two-dimensional check: improvement and a failed inheritance

Eight analytic fields were sampled at 17x17 and reconstructed at 49x49.
All methods use the same horizontal-then-vertical Cartesian order; this is
a matched factor comparison, not the formal Eikonal blend or direct joint
warp operator. No natural-image superiority claim follows from eight fields.

| Field | Interior MSE change vs CONV | Full-image change |
|---|---:|---:|
| Diagonal wave | -17.64% | -37.38% |
| Crossing waves | -55.97% | -41.83% |
| Diagonal sigmoid | -44.30% | -34.32% |
| Curved sigmoid | -20.35% | -20.21% |
| Diagonal step | +0.17% | +0.43% |
| Disk | +0.36% | +0.36% |
| Chirped field | -42.37% | +6.56% |
| Edge + texture | -32.95% | -34.89% |

Cardinality errors are zero for all current-based methods. However, the
one-factor excursion comparison does **not** compose into a comparison with
the original final image. The second factor sees different intermediate
data and hence a different compact reference. For the diagonal wave,
excursion rises from zero to .01637; for the chirped field, from .001489 to
.04452. The curved sigmoid improves its excursion from .008392 to .000253.
The outcome is not a uniform 2-D dominance result.

This identifies the next foundational requirement quite precisely:

\[
\min_U\int\|dU-dU_{\rm proposal}\|^2,
\qquad L_a[U]=y_a,
\]

with a **single source-derived admissible potential set**, shared derivative
traces, current-cone constraints, and a common continuous excursion envelope.
The admissible set must be defined before selecting factor order or query
grid. Derivatives remain exact differentials of U; aperture averages and
warp samples remain observation functionals of U. Independently improving
two raster passes cannot substitute for that shared state.

The existing direct-warp appendix already supplies shared control entities
and directional constraints. A natural next implementation would replace its
raw proposal with the compatible potential and admit it jointly, testing
functional constraints rather than only coefficient signs. Neither that full
joint construction nor a universal feasibility/efficiency theorem is claimed
by this experiment.

## 7. Numerical and execution status

The implementation is a research oracle. Its compatible spline solve uses
SciPy's B-spline construction; the nonlinear admission and root checking are
Python/SciPy code. The tested 49x49 envelope reconstructions take roughly
0.1–3.2 seconds on the Mini, versus roughly 1–2 milliseconds for the existing
vectorized CONV factor path in the same script. These are diagnostic timings,
not a native-backend performance comparison. This implementation is not ready
to replace the production renderer.

The continuous-cone oracle was hardened for a large proposal over tiny cell
mass: optimization uses unit-mass coefficients, the metric follows any
coordinate reversal, and contiguous arrays make the numerical orientation
consistent. The cone exchange targets 2e-8 negativity in unit-mass coefficient
coordinates, then removes it with a mass-preserving convex mix toward the
positive constant. This is an explicit numerical tolerance, not an exact
semidefinite solver. Tests of data rescaling use 5e-6 coefficient tolerance
near double contacts; invariants and integrated values are more accurate.

Twelve test methods pass on the M4 system Python. They include the previous
40 random positive-cone projections, quintic polynomial reproduction,
proposal trace compatibility, reflection, conservation, cardinality,
affine covariance, support range, the one-factor envelope excursion bound,
and tiny conserved mass in the presence of a much larger proposal. Both
complete 1-D populations, all 15 shared-trace cases, and all 40 2-D
field/method combinations completed and were copied back.

The code is in `experiments/conv_compatible_current.py`; the 2-D probe is
`experiments/probe_conv_compatible_2d.py`; the tests are in
`experiments/test_conv_compatible_current.py`. Raw results and figures are
under `output/support_geometry/conv_compatible_current/`.

The practical conclusion is a reproducible stronger 1-D CONV candidate:
**compatible global current inference, continuous slope admission, and
continuation within the compact excursion envelope**. The strong gains are
not purchased with additional one-dimensional ringing allowance. The
boundary and multidimensional results show exactly where more work is needed.
