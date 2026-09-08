# Jointly admissible derivative modes: characteristic potentials

Follow-up: `CONV_BUDGET_REPLACEMENT.md` retains the shared-potential construction,
adds admitted scalar curvature, and replaces all-pairs recognition and Python
certificate loops with finite witnesses and support filters. It reports the
measured budgeted replacement and its unchanged CONV fallback.

2026-09-04. Direct construction; no optimization, linear-system solve,
iterative admission, trained model, or supplied analytic edge geometry.

**Finding:** the useful unit is a scalar profile composed with a scalar
geometric coordinate, together with its complete derivative identities:

\[
 F=\phi(G),\qquad
 \nabla F=\phi'(G)\nabla G,\qquad
 \nabla^2 F=\phi''(G)\nabla G\nabla G^T+\phi'(G)\nabla^2G.
\]

This is an application of the chain rule, not a new calculus identity. Its
consequence for CONV is substantive: direction, magnitude, and curvature can
be admitted as the derivatives of one potential, without subsequently
repairing independent Cartesian jets. The tested finite catalog contains
straight and radial coordinates and explicitly rejects unsupported geometry.
It is not yet a universal CONV replacement.

## Why independently admissible compact correction modes cannot work

Let a differentiable correction vanish on the entire boundary of a bounded
convex patch, and require its gradient to belong everywhere to one fixed
proper closed convex cone K. Select any nonzero v in the dual cone. Along
each line parallel to v through the patch, the correction is nondecreasing.
Both ends of that line have boundary value zero, so the correction is zero
throughout the segment. These segments cover the patch.

Thus a nonzero, independently cone-admissible, zero-boundary correction does
not exist under these assumptions. This is not an impossibility theorem for
all joint modes: it does not apply to unrestricted whole-plane cones, modes
carrying nonzero traces, varying cones in general, or corrections that are
admissible only together with their baseline. It explains why seeking a
collection of independently safe local bumps is the wrong goal near a clean
edge. The observation differences and their shared geometry must participate
in the mode itself.

## Straight modes and integrability

For G(x,y)=n_x x+n_y y,

\[
 D^kF=\phi^{(k)}(G)n^{\otimes k}.
\]

All derivative orders share the same direction and scalar phase. With
phi' nonnegative, the current stays on the forward ray through n. No
independent choices of x slope, y slope, and mixed curvature are necessary.

More generally, simply declaring a vector field a(x,y)n(x,y) does not make
it a gradient. It must satisfy

\[
 n_y\partial_x a-n_x\partial_y a
 +a(\partial_x n_y-\partial_y n_x)=0.
\]

If n is already grad G, this requires grad a to be parallel to grad G;
locally, away from critical points, a is a function of G. Composition gives
that compatibility by identity.

The compiler obtains possible straight directions from the samples. Every
ordered pair Y_j>Y_i requires n dot (p_j-p_i)>0 for a monotone profile.
The planar homogeneous inequalities are intersected by sorting their angles
and finding the largest circular gap. There is no angle-grid sweep or
numerical feasibility solver. Repeated equal-value lattice pairs supply
finite candidate directions perpendicular to their displacement. The most
repeated admissible direction is chosen; absent such a witness, the angular
midpoint is used. Affine fields are recognized directly from their finite
differences and retain their exact gradient direction.

This is a geometric preference among compatible explanations, not proof that
the true direction is uniquely identifiable. For the matched diagonal edge,
the order interval remains 0.8425 degrees wide. The repeated-level selector
chooses (1,0.7)/sqrt(1+0.7^2), which matches that test's analytic direction.
No case label, known slope, or truth error enters the choice.

## Curved modes require a curvature companion

The radial family uses G=+/-|p-c|^2, so

\[
 \nabla G=+/-2(p-c),\qquad \nabla^2G=+/-2I.
\]

Consequently its Hessian contains the full-rank term phi'(G)(+/-2I) as well
as the rank-one phi'' term. Retaining only directional powers would lose
this necessary curvature. A regression test reconstructs G itself and
checks that its Hessian is 2I, even where the scalar profile's second
derivative is zero.

Candidate centers are the barycenters of observed minimum and maximum sets;
both radial polarities are checked. These are a small, explicit family of
source witnesses, not a general circle-fitting method. Every candidate must
respect all observation ordering and equal-phase values. Off-center circles
whose centers are not recovered by these witnesses are rejected.

## A direct shared C2 scalar profile

Once a coordinate G is admitted as a candidate, all source observations are
ordered by t_i=G(p_i). Equal phases must have equal observed values. The
profile is interpolated through these phase samples by quintic Hermite
pieces with common first derivatives m_i and zero second derivatives at
the phase knots.

For secants d_i=(f_{i+1}-f_i)/(t_{i+1}-t_i), use

\[
 m_i=\min\left(
 \frac{2d_{i-1}d_i}{d_{i-1}+d_i},
 \frac54\min(d_{i-1},d_i)\right),
\]

with m_i=0 when either adjacent secant is zero. Ordinary endpoint slopes are
the endpoint secants; radial profiles use zero endpoint slopes so constant
extensions into unsampled inner phase ranges remain C2. No iteration is used.

The physical derivative Bernstein coefficients on a phase interval are

\[
 [m_i,m_i,5d_i-2m_i-2m_{i+1},m_{i+1},m_{i+1}].
\]

They are nonnegative by construction. The factor 5/4 follows from the
quintic coefficient identity; it is not a tuned sharpness parameter. Every
piece is monotone and bounded by its observed endpoint values, with first
and second derivatives shared across phase knots. Composing this C2 profile
with the smooth coordinate preserves C2 spatial derivatives automatically.

In exact arithmetic this is cardinal. The implementation merges phase
coincidences and permits only small floating-point observation discrepancies;
its compatibility threshold is 2e-11 in normalized source units. Actual
source reconstruction error across the measured accepted cases is below
6.9e-15. Results report that residual rather than claiming bitwise equality.

## The important correction: certify current, not direction alone

The original CONV law is

\[
 \ell_C^T\nabla F=\phi'(G)\,\ell_C^T\nabla G\ge0.
\]

Requiring grad G to lie in every support cone, independently of phi', is
stronger than that law. Where phi' is zero, the mode contributes zero current
and its otherwise irrelevant direction must not create a restriction.

This distinction was measured. In the 13 by 13 edge with slope sqrt(1/2)
and width 32, sample ordering admits an angular interval 0.5457 degrees wide,
near 35.265 degrees. The intersection of all direction-only support cones
collapses to a different ray near 44.47 degrees. The first compiler rejected
the field. The completed profile has zero derivative on the conflicting
flat phase ranges. Checking the **product current** against every original
cone admits the field with a minimum current certificate margin of zero.
No cone was enlarged, and no observation was discarded to achieve this.

The straight-mode certificate bounds phi' on every phase piece intersecting
each source cell using its derivative Bernstein coefficients. Multiplying
those bounds by ell dot n certifies the complete continuous current. The
range follows from monotonicity and the phase extrema at cell corners.

For radial candidates the present certificate is more conservative: it checks
ell dot grad G at all four cell corners. This is sufficient throughout the
cell because that expression is affine. Exact minimum/maximum radial phase
over the rectangle, followed by the monotone scalar profile, certifies the
source-support range. These are finite algebraic checks, not raster tests.
Acceptance tolerance is 2e-11; the reported successful nonconstant current
and range lower bounds in the main runs are nonnegative.

## Matched results: four modes admitted out of eight fields

The unchanged 17 by 17 sources and 49 by 49 analytic truth arrays are used.
Interior means both normalized coordinates lie strictly between 0.2 and 0.8.
The prototype admits four cases and rejects four. It supplies no fallback
output for a rejection.

| Admitted field | Interior MSE | Value-error reduction vs CONV | Sampled-gradient error reduction vs CONV |
|---|---:|---:|---:|
| Diagonal sigmoid | 2.18519e-9 | 99.9972% | 99.9982% |
| Curved sigmoid | 5.01527e-5 | 97.0422% | 96.6157% |
| Diagonal step | 2.61880e-4 | 98.3342% | 98.2217% |
| Disk | 1.47977e-2 | 66.8898% | 62.8343% |

All four improve over both CONV and Lanczos-3 in interior and whole-image
value MSE and in sampled-gradient MSE. The diagonal wave, crossing waves,
chirp, and mixed edge/texture field are rejected. No eight-field win rate is
claimed. The point is that the two smooth-edge cases rejected by the prior
finite zero-jet baseline now have direct, shared, admissible constructions.

The straight sigmoid has 219 distinct witnessed phase samples across its
289 source pixels, versus only 17 samples along one row. Reusing these
cross-row observations under the admitted geometry explains much of the
gain. This is a structural model advantage, not a demonstration that a
fixed local interpolation kernel has acquired the same accuracy.

The early straight-only cubic probe is archived separately under
`output/support_geometry/conv_joint_ridge_modes/cubic_first_probe.json`.
It had lower diagonal-sigmoid value error than the C2 quintic profile, but
only C1 traces. The reported main construction uses the C2 profile throughout.

## Separate 16-field check

With the geometric selectors and scalar construction fixed, the second
battery changes to 13 by 13 sources and 37 by 37 evaluation grids:

* Eight straight edges: slopes 0.35, sqrt(1/2), -0.4, and 1.1; widths 18 and 32.
* Six radial edges: centers (0.5,0.5), (6.5/12,5.5/12), and (0.43,0.57);
  widths 25 and 45.
* An ellipse and a crossing-wave field as rejection probes.

**Twelve cases are admitted; four are rejected.** The admitted cases are all
eight straight edges and the four circles whose integer or half-integer
centers are recovered. The two circles centered at (0.43,0.57), the ellipse,
and the crossing-wave field are rejected.

Each of the 12 accepted cases improves over both baselines in all three
measured quantities: interior value MSE, whole-image value MSE, and interior
sampled-gradient MSE. Relative to CONV, the weakest accepted improvement is
92.38% in interior values and 90.55% in sampled gradients. Relative to
Lanczos-3, the corresponding weakest improvements are 91.44% and 90.11%.
These conditional results must be reported together with the acceptance
coverage; they do not describe an operator that handles all 16 fields.

The sampled-gradient metric applies the same centered finite-difference
measurement to the reconstructed raster and to analytic truth sampled on that
raster. It is not an exact continuous derivative-error metric. Continuous
derivative admissibility and path independence are checked separately.

Across both batteries, measured excursion is at most 4.45e-16. Source-site
error is below 6.9e-15. The invariant suite comprises seven characteristic
tests and four earlier finite-capacity tests, all passing on the M4 Mini.
It includes C2 phase traces, affine reproduction, transpose/reflection/value
covariance, continuous range checks, the radial Hessian companion, exact
piecewise-polynomial gradient path integration, unsupported-mode rejection,
and the zero-amplitude direction case. Solver entry points are replaced with
raising functions in the direct-construction test.

## What is and is not established

The demonstrated modes are source-defined characteristic potentials. Their
derivatives satisfy the shared identities before admission; the admissibility
checks concern their complete current. The scalar profile is constructed in
a fixed finite pass. There is no hidden optimization replacing the rejected
joint solver.

The catalog is deliberately small and global. The ordering check uses all
source pairs (quadratic in the number of pixels), and phase knots are not
restricted to Cartesian cell boundaries. These are research implementations
of a representation, not a production complexity result or a compiled
replacement for the existing tensor atlas and pixel-integration path.

Remaining work is acquiring more general scalar coordinates directly and
making them compatible across regions. Spatially blending modes is not
automatically safe: grad(sum w_i F_i) contains sum F_i grad w_i as well as
sum w_i grad F_i. Those terms must be represented and admitted, not ignored.
Likewise, independently selecting a local normal and a local curvature does
not supply a compatible scalar coordinate. The next construction should
retain the entire characteristic derivative identity, including its curvature
and zero-amplitude behavior, as the unit of transport.

## Reproduction

Run the commands recorded in root `AGENTS.md` on the M4 Mini. Implementation:
`conv_joint_ridge_modes.py` and `conv_joint_characteristic_modes.py`.
Tests: `test_conv_joint_characteristic_modes.py`.
Plotting: `plot_conv_joint_characteristic_modes.py`.

Main artifacts: `output/support_geometry/conv_joint_characteristic_modes/`:
`matched.json`, `validation.json`, `summary.json`, `profiles.png`, and
`slope_errors.png`. Every rejected case and its reason is retained in the JSON.
