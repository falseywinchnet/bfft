# CONV joint potential: shared slope geometry with continuous certificates

2026-09-04. Experimental scalar 2-D reference implementation. The production
CONV operator and manuscript are unchanged.

**Design correction:** the user has clarified that CONV does not use solvers.
The QP construction in this document is therefore **not an accepted CONV
candidate**. Its measured gains are diagnostic evidence about available shape
freedom; they do not demonstrate the required direct construction. The finite
shared-jet follow-up is documented in `CONV_JOINT_FINITE_FINDINGS.md`.

**Result:** one jointly admitted C1 potential recovers much of the lost slope
detail while controlling excursion on the continuous source domain. With the
natural-boundary proposal, all eight matched fields improve on CONV in both
interior and whole-image value MSE. All eight also improve on Lanczos-3 in
interior value MSE; six improve over the whole image. These are measurements
on the stated analytic battery, not a general ordering of interpolation methods.

## What is new relative to the existing joint reference

The manuscript's warp appendix and `conv_warp/joint_reference.py` already
represent a joint Bernstein potential with shared C0 edges and support-gradient
cones. Integrability itself is not a discovery of this follow-up. The tested
advance is the combination of:

1. A source-only, commuting tensor-quintic proposal, instead of constructing
   the proposal from a sequentially admitted destination raster.
2. Shared C1 derivative traces represented as common coordinates. C2 is also
   implemented and tested, but the reported primary sweep uses C1.
3. The exact integrated squared **gradient** correction as the joint objective.
4. Bernstein subdivision as an inner certificate of continuous range and
   directional positivity, admitting valid functions rejected by the original
   whole-cell coefficient cone.
5. One joint range restriction, applied before any destination queries.

The gradient Gram metric and polynomial subdivision are established
mathematical tools. The result here is their concrete consequence for the
existing CONV reconstruction problem, not a claim of inventing those tools.

## One field defines values, currents, and traces

Each source cell carries a bidegree-(5,5) potential

\[
F_C(u,v)=\sum_{i,j=0}^{5}P^C_{ij}B_i^5(v)B_j^5(u).
\]

Adjacent patches refer to the same edge controls. At every interior vertical
source-grid boundary, the additional identity is

\[
P_{b+1}=2P_b-P_{b-1},
\]

and the horizontal identity is identical with coordinates exchanged. The
implementation eliminates the right-hand controls using these identities;
they are not approximate equality constraints left for the QP solver. For C2,
the additional eliminated coordinate is
\(P_{b+2}=P_{b-2}-4P_{b-1}+4P_b\).

The source vertices are fixed exactly. A 17 by 17 source has 81 by 81 global
controls; fixing its source vertices and sharing C1 traces leaves **4,067**
independent correction coordinates, versus 6,272 with only shared values.

Define \(J_x=\partial_xF\) and \(J_y=\partial_yF\). These currents are never
admitted as independent images. On every source-cell row,

\[
\int_0^1 J_x(u,v)\,du=F(1,v)-F(0,v),
\]

with the corresponding vertical identity. Closed-loop integrals vanish;
mixed derivatives agree inside each patch, and the C1 field has no gradient
jump across patch boundaries. Evaluation at any new collection of coordinates
uses the same retained field. There is no horizontal-versus-vertical admission
order or destination-grid-dependent reconstruction.

## The variational admission

Let \(P_Y\) denote the compatible proposal for the source samples. The solve is

\[
 F_Y=\operatorname*{argmin}_{F\in\mathcal A_Y}
 \frac12\sum_C\int_C\|\nabla F-\nabla P_Y\|_2^2\,dx\,dy.
\]

The feasible set includes the fixed source vertices, shared traces, continuous
support ranges, and directional constraints from the existing support cones.
Equivalently, at its exact optimum,

\[
\int\nabla(F_Y-P_Y)\cdot\nabla(G-F_Y)\ge0
\quad\text{for every }G\in\mathcal A_Y.
\]

This is the useful unified representation: correct the potential by the least
necessary change to its gradient, while all of its values and derivatives
continue to refer to that same potential. The gradient seminorm becomes
positive definite on the correction space: zero gradient energy implies a
global constant, and the fixed source vertices force that constant to zero.
Consequently the optimum is unique **when the feasible set is nonempty**.
The C0 Q1 construction proves feasibility for the original C0 constraints;
that proof does not establish feasibility after adding C1 or C2. The measured
cases are checked numerically. There is no silent fallback to C0.

For one cell, the exact gradient matrix is
\(H=M_5\otimes K+K\otimes M_5\), where
\(K=D^TM_4D\), \(D\) is five times the first-difference matrix, and

\[
(M_n)_{ij}=\frac{\binom ni\binom nj}
 {(2n+1)\binom{2n}{i+j}}.
\]

The implementation assembles this sparse quadratic form, eliminates fixed
values and shared traces, and solves the remaining linearly constrained QP
with scaled proximal ADMM. The proximal term does not change the requested
objective at convergence. This is a research oracle, not a production speed
implementation.

## Why subdivision preserves the near-flat slope

For a scalar directional current \(q=\ell^T\nabla F\), whole-cell Bernstein
nonnegativity is sufficient but unnecessarily restrictive. Subdivide the
same polynomial and require nonnegative Bernstein coefficients on every
subcell. No new fit, sample, degree of freedom, or target resolution enters
that operation. Each subdivision relaxes the sufficient certificate: a
coarsely feasible polynomial remains feasible at every finer depth.

The motivating example survives this construction exactly:

\[
F'(u)=(u-1/2)^2+0.001.
\]

Its degree-elevated whole-cell derivative coefficients include a value below
-0.04. One subdivision certifies the same polynomial's derivative with all
subcell coefficients at least 0.001. There is no need to flatten or broaden
the potential to repair the sign of a coordinate that was not a negative
physical derivative. A regression test verifies this identity.

The range certificate applies the same subdivision matrices to the potential
itself and bounds each subcell's controls by the original 6 by 6 support's
minimum and maximum. Convex-hull containment then bounds the complete
continuous field, not just a displayed raster. These remain sufficient inner
certificates, not an exact finite characterization of every nonnegative
bivariate polynomial. In particular, zeros can remain difficult at finite depth.

## Proposal and endpoint experiment

Both proposals are linear tensor products of 1-D C4 quintic interpolation
maps, so their axis operations commute before admission.

* `polynomial`: endpoint first/second derivatives come from the first/last six
  samples. It reproduces polynomials through degree five before admission.
* `natural`: endpoint third/fourth derivatives vanish. Each line is the
  variational quintic minimizing integrated squared third derivative with its
  nodal values fixed and endpoint first/second derivatives free. It reproduces
  quadratics, but does not retain degree-five polynomial reproduction.

This characterization is one-dimensional; the tensor proposal is not claimed
to minimize an isotropic two-dimensional third-derivative energy. Also, these
are proposal boundary conditions, not extra constraints imposed on the final
admitted potential.

The polynomial proposal delivered the first strong interior gains but retained
large boundary errors. The natural proposal improved every matched field's
whole-image MSE relative to the polynomial proposal. It was selected on those
eight exploratory fields, then frozen before running the separate 12-field
validation battery. It is an actual boundary assumption, not a cost-free
improvement with identical reproduction guarantees.

## Matched eight-field results

Exactly the prior 17 by 17 sources and 49 by 49 analytic truth arrays are used.
Interior means both normalized coordinates lie strictly between 0.2 and 0.8.
Primary settings: natural proposal, depth-one certificates, C1 traces, support
cones enabled. Negative percentages mean less value MSE.

| Field | Interior vs CONV | Full vs CONV | Interior vs Lanczos-3 | Full vs Lanczos-3 |
|---|---:|---:|---:|---:|
| Diagonal wave | -96.01% | -75.90% | -25.44% | -5.52% |
| Crossing waves | -93.81% | -65.90% | -7.48% | +37.81% |
| Diagonal sigmoid | -47.42% | -47.61% | -35.59% | -51.72% |
| Curved sigmoid | -16.66% | -15.99% | -6.94% | -19.40% |
| Diagonal step | -8.60% | -7.45% | -13.14% | -12.99% |
| Disk | -5.46% | -5.33% | -12.33% | -13.89% |
| Chirped field | -84.48% | -25.33% | -50.14% | +41.31% |
| Edge and texture | -41.60% | -48.58% | -11.56% | -10.33% |

All eight primary solves converged at the requested scaled residual tolerance
2e-7. The largest unscaled constraint violation in normalized source units was
8.62e-7; the largest measured raster excursion was 3.72e-9. Source samples were
exact on the matched raster. These are floating-point checks, not outward-rounded
interval certificates. A caller must inspect `converged` and
`maximum_violation`; an unsuccessful research solve is returned with its flag,
not silently labeled an admissible field.

Compared with the previous sequential compatible-envelope prototype, seven of
eight interior errors improve. The curved sigmoid is 4.63% worse than that
prototype, while its range behavior is substantially better. The original disk
excursion of about 0.21847 is reduced to numerical tolerance. Source-range
containment is not a complete no-ringing theorem: an oscillation can stay
inside that range. Support cones are uninformative on several oscillatory
fields because their witnessed directions span the whole plane.

## Separate fixed-setting validation

The second battery changes the source to 13 by 13 and evaluates on 37 by 37.
It includes six plane waves (three frequency/orientation pairs, two phases),
three sigmoid edges with different slopes and widths, and three chirps with
different phase curvature. The formulas are fixed in
`validate_conv_joint_potential.py`; no per-case parameter selection is used.

**All 12 cases improve on both CONV and Lanczos-3 in interior value MSE and
interior sampled-gradient MSE.** All 12 improve on CONV in whole-image MSE;
10 of 12 improve on Lanczos-3 there. Gradient error uses the same centered
finite-difference operator on every 37 by 37 reconstruction and on the sampled
analytic truth. It measures sampled slope fidelity, not a continuous derivative
error theorem. The implementation's own analytical derivatives are checked
separately by the invariant tests.

The following percentages compare sums of errors within each family, rather
than averaging per-case ratios. Higher-error fields consequently carry more
weight; the per-case win counts above prevent that aggregation from concealing
a losing case.

| Family | Interior value vs CONV | Interior gradient vs CONV | Interior value vs Lanczos | Interior gradient vs Lanczos |
|---|---:|---:|---:|---:|
| Six waves | -80.26% | -80.25% | -62.74% | -63.63% |
| Three sigmoid edges | -15.22% | -16.21% | -15.06% | -23.41% |
| Three chirps | -67.35% | -67.60% | -33.72% | -34.61% |

Whole-image family error falls by 54.89%, 13.14%, and 26.98% against CONV.
Against Lanczos it falls by 17.43% and 11.98% for waves and sigmoids; chirps
increase by 1.41%. The two individual whole-image losses to Lanczos are
`wave 3.8 2.1 0.19` (+0.34%) and `chirp 2.8` (+9.62%).

Every validation solve converged. The largest normalized constraint violation
was 1.25e-6, the largest measured excursion was 2.55e-9, and source-site
evaluation differed only by floating-point query arithmetic (at most 8.89e-16).
The retained source vertex controls themselves are assigned exactly.

The primary 17 by 17 solves took approximately 1.1 to 115 seconds on the M4
Mini; validation solves took 0.48 to 72 seconds. Nearly degenerate edge cones
dominate this cost. These timings establish that this is a usable small-field
research oracle, not yet a competitive image-resizing implementation. Some
exploratory processes overlapped, so these are run timings rather than an
isolated performance benchmark.

## Structural ablations

These retain the polynomial proposal to isolate the certificate and trace
effects. The archived first ablation used explicit C1 equalities; the current
implementation represents the same C1 space by elimination. The natural
primary sweep was rerun with elimination and is stored separately.

| Certificate / trace | Diagonal wave interior MSE | Chirped field interior MSE |
|---|---:|---:|
| Depth 0, C1 | 1.56301e-4 | 1.85758e-4 |
| Depth 1, C1 | 6.29445e-5 | 1.46126e-4 |
| Depth 2, C1 | 3.42048e-5 | 1.34459e-4 |
| Depth 1, C0 | 6.53307e-5 | 1.58231e-4 |

At depth one, subdivision lowers the diagonal-wave error by 59.73% relative
to the coarse certificate. A second level lowers it another 45.66%. The
chirp improves by 21.34% and then 7.98%. The exact gradient-correction objective
also decreases with depth, as expected for nested feasible sets. Whole-image
truth error need not decrease with depth: the chirp gets slightly worse there.
The metric minimizes correction to the proposal, not error to unknown truth.

At fixed depth one, C1 improves both of these interior errors relative to C0,
despite its smaller feasible set. This is evidence that sharing derivative
traces contributes useful structure; it is not just a visual seam repair.

## Reproduction and files

The invariant suite has **19 passing tests** across the joint potential and
the two preceding current prototypes. The joint tests cover tensor polynomial
and derivative reproduction, exact gradient integration, source interpolation,
continuous certificate margins, structural C1/C2 traces, transpose symmetry,
affine value covariance, query independence, path independence, and the
positive-current/negative-coefficient counterexample.

Run the commands in the root `AGENTS.md` on the M4 Mini. No extra remote solver
package was installed. `conv_joint_potential.py` is the numerical reference;
`validate_conv_joint_potential.py` defines the separate battery;
`plot_conv_joint_potential.py` renders saved measurements locally.

Artifacts live under `output/support_geometry/conv_joint_potential/`:

* `initial_depth1_c1.json`: first polynomial-proposal sweep, including the
  curved-edge solve that had not converged at 30,000 iterations.
* `study/curved_retry.json`: that solve converged at 38,450 iterations.
* `study/`: endpoint and certificate/continuity ablations before trace elimination.
* `final/natural.json`: all eight primary cases, rerun after trace elimination.
* `final/validation.json`: the fixed-setting, different-grid 12-field battery.
* `summary.json` and the PNG figures: derived comparisons.

The remaining research bottlenecks are robust fast admission at nearly
degenerate cone contacts, boundaries on rapidly changing phase, and useful
topology constraints when the full support cone spans the plane. The shared
potential has removed the factor-order inconsistency; it has not removed the
need to infer those geometries from finite observations.
