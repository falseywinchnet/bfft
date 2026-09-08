# Direct joint potential: correction after the solver detour

2026-09-04. User constraint: **we do not do solvers**.

Follow-up: `CONV_JOINT_CHARACTERISTIC_MODES.md` constructs direct characteristic
potentials, retaining shared direction, curvature, and amplitude identities.
Its certified straight/radial cases include the two smooth-edge fields that
the zero-jet baseline below rejects; unsupported cases remain explicit.

The preceding QP results are not an accepted CONV construction. They are
diagnostics about available shape freedom. The construction must obtain its
shared geometry through direct algebra and finite admission, without an
optimization loop or a linear-system solve.

## Direct construction tested

`conv_joint_finite.py` constructs a tensor-quintic Hermite potential from
shared nodal derivatives. Each node supplies value, first and second
derivatives in both coordinates, including mixed derivatives: nine entries,
of which the value is fixed. Adjacent cells reuse these entries. C2 continuity,
interpolation, and an integrable gradient are consequently structural.

The proposal derivatives come from the existing compact CONV FIR stencils.
No spline-system solve is used. The baseline retains the observed values and
sets every derivative to zero. Its patches are bilinear interpolation in the
quintic smoothstep coordinates, hence range-feasible and C2.

Write the proposal as a baseline plus shared-jet entities,

\[
 P=B+\sum_g d_g.
\]

For every affine certificate \(a_k^TF\ge b_k\), form

\[
 m_k=a_k^TB-b_k,\quad r_{kg}=a_k^Td_g,
 \quad H_k=\sum_g\max(0,-r_{kg}).
\]

Assign each harmful entity the smallest incident capacity

\[
 \alpha_g=\min\left(1,\min_{k:r_{kg}<0}\frac{m_k}{H_k}\right).
\]

Provided every baseline margin is nonnegative,
\(B+\sum_g\alpha_gd_g\) satisfies every certificate. A single common
remaining ray has a closed first-contact ratio. These are fixed finite
reductions, not a convergent iteration, active-set search, or optimization.
The extension from the older finite reference is that the shared-jet entities
may overlap in control space; their complete responses are combined before
measuring harmful contributions. Shared derivatives survive every allocation.

Depth-one Bernstein certificates enforce the same support ranges and
directional cones as the joint experiment. Candidate feasibility is checked
first so an already feasible field passes unchanged. If admission is required
but the baseline fails, the construction explicitly rejects that case.

## The baseline obstruction is real

Range-feasible and smooth does not imply direction-feasible. Even an affine
field \(x+a y\) illustrates the distinction. Its witnessed gradient cone is
the ray through \((1,a)\). The zero-jet baseline instead has gradient

\[
 (S'(x),aS'(y)),\qquad S(t)=10t^3-15t^4+6t^5,
\]

which generally leaves that ray. The affine proposal passes unchanged, so the
implementation still reproduces affine fields, but the example disproves a
general cone-feasibility argument for the baseline.

On the eight matched fields, the baseline fails the original directional
certificates on the diagonal sigmoid, curved sigmoid, and chirp. Their
normalized minimum margins are -0.14633, -0.04117, and -0.09063, respectively.
These are structural failures, not floating-point tolerances. All three are
reported as rejected; the directional constraints are not relaxed.

## Measured finite-admission result

With each node's complete derivative data grouped together, the remaining
five fields admit successfully with continuous certificate residuals near
roundoff. However, the accuracy does not establish a better operator:

| Field | Interior MSE change vs CONV | Whole-image change vs CONV |
|---|---:|---:|
| Diagonal wave | +19.52% | -6.98% |
| Crossing waves | +28.23% | +12.33% |
| Diagonal step | +10.87% | +8.12% |
| Disk | +7.63% | +7.63% |
| Edge and texture | -3.79% | -9.39% |

Splitting the entities by derivative order or by individual components did
not solve this. All three rejected cases remained rejected; admitted wave
accuracy became worse. For example, diagonal-wave interior MSE was 0.001376
with complete nodal bundles, 0.003395 by derivative order, and 0.005899 by
individual component. Beneficial cancellation inside a shared derivative
bundle matters; finer coordinatewise limiting can destroy it.

The common completion ray was zero in these cases. At least one active
certificate prevented motion along the entire remaining proposal direction.
This diagnoses the conservatism of this particular finite allocation, not an
impossibility of all direct joint-potential constructions.

The direct runs took approximately 0.2–0.5 seconds per 17 by 17 source on the
M4 Mini. Four tests pass: overlapping-entity capacity validity, explicit
infeasible-baseline rejection, affine reproduction, and interpolation/C2/
transpose/reflection invariants. The latter test replaces the NumPy linear
solve, SciPy optimizer, and earlier QP entry points with functions that raise;
the direct construction still completes.

## What this leaves unresolved

The finite capacity identity and shared Hermite representation work. The
zero-jet baseline and these entity allocations do not deliver the requested
improvement. A valid next construction must supply shared derivative geometry
that already respects the witnessed directions and retain useful cancellation
within its modes. Replacing that missing construction with an optimizer, or
broadening the support cone until the baseline passes, would evade the task.

Saved results: `output/support_geometry/conv_joint_finite/node_bundle.json`,
`conv_joint_finite_order.json`, and `conv_joint_finite_component.json`.
Reproduction commands are in the root `AGENTS.md`.
