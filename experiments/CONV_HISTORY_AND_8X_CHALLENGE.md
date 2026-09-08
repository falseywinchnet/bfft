# Recovered CONV history and the 8x round-trip challenge

September 5, 2026. Conversation recovery and mathematical deductions, not
a newly measured 8x improvement. The production operator and manuscript
have not been changed by this recovery.

## Conversations recovered

The main deeper investigation is **Find Eikonal Laplace evidence**,
`01a03c28-1c85-70c3-89b8-1557434192d6`, created August 25. The original
local transcript contains the following milestones:

- At transcript line 18132, synthesis-only companion charts were rejected:
  the coarse basin raster had already lost the phase they attempted to move.
- At lines 19090–20418, the cameraman tripod exposed phase/cadence changes
  across independent owner frames. Smoothing chart weights did not repair
  that defect. Compatible characteristic transport, rather than independent
  local direction estimates, was the unresolved mathematical object.
- At lines 28483–28938, the high-frequency investigation distinguished nodal
  interpolation from cell-average measurement and identified an extra aperture
  in reconstruct-then-average use on already averaged data.
- At lines 28944–30099, native four-child fusion, the broader census, and the
  scale-depth cancellation identity established the scope of the moment atlas.

The related **Find Eikonal Laplace evidence (2)**,
`01a0482d-88fa-7512-9ab0-a70d873bf464`, contains the passband, powered
partition, and zeta/alpha investigation. Its final correction retained the
Euclidean CONV* projection because the tiny profile-metric benefit did not
justify restoring dense face enumeration. The standalone result note had
not carried that final correction; it now does.

**Compare conv and Remiz-McLellan**,
`01a0578b-759b-7760-b12c-08aac24d2e76`, instead developed general FIR
optimization with solvers. It is not the requested solver-free 2-D route.

Transcript source for the main task:
`/Users/ultimussecundai/.codex/sessions/2026/08/25/rollout-2026-08-25T22-40-56-01a03c28-1c85-70c3-89b8-1557434192d6.jsonl`.

## What the atlas evidence establishes

See `CONV_FOUR_CHILD_ATLAS_CENSUS_RESULT.md`. The broader comparison is
against positive overlap of piecewise-constant source cells, using analytic
cell-average truth. Its 6,000 reduction cases recorded 5,876 wins, 36 losses,
and 88 ties, with 13.03x geometric-mean MSE advantage. This is not an 8x
round-trip comparison against the native two-order CONV demo. Six reduction
cases still had final source-range excursion; the maximum was 0.004793.
Enlargement exposed much larger child excursions. These results cannot be
promoted to a zero-ringing universal resizer claim.

## Exact consequence for aligned 8x reduction

Suppose source entries are equal-area cell means and the coarse pixels
each contain exactly 8 by 8 complete source cells. Let A_J be any refined
atlas preserving each original source-cell mean. Then

\[
 (R_8 A_J y)_{p,q}
 =\frac1{64}\sum_{a,b=0}^{7} y_{8p+a,8q+b}
 =(R_8 y)_{p,q}.
\]

This follows by summing the preserved mass of each complete source cell.
It is a special case of `CONV_ATLAS_SCALE_DEPTH_DERIVATION.md`: generated
detail contributes only through cut source-cell boundaries. There are none
in this aligned footprint. More conservative atlas depth cannot improve
this reduction; all such constructions give the same coarse raster.

The native demo uses endpoint-aligned nodal coordinates and basin integration,
so this identity must not be silently applied to its different footprints.
Both contracts can be tested, with their coordinate and aperture conventions
stated. A changed contract must not be reported as a synthesis improvement.

## The surviving deeper question

For a cell-mean input, the reconstructed potential F should satisfy the
measurement equations themselves:

\[
 \frac1{|C_{ij}|}\int_{C_{ij}} F(x,y)\,dx\,dy=c_{ij}.
\]

The directional currents must belong to this same potential, so compatible
mixed derivatives and shared traces are structural requirements. Admission
should control the actual current, while retaining the necessary topology
law; merely increasing its coefficients or retaining more energy does not
prove phase fidelity or absence of oscillation. The current conversation's
stationary-inflection counterexample identifies excessive coefficient-level
restriction, but its successful 1-D repair is not a complete 2-D construction.

This points to an aperture-consistent, jointly admitted potential whose
directional behavior comes from shared source data, without independent
owner-frame resets. It is a research target, not an available inexpensive
operator. A global Eikonal or potential solve violates the current design
constraint. The missing step is a direct finite construction and its cost
proof/measurement, not another catalogue recognizing straight or radial cases.

For aligned means, improved 8x reconstruction has to come from better use of
the coarse measurements during synthesis. For fractional or warped footprints,
the reduction representation is also a valid place to improve. Neither case
permits hidden retained fine-scale residuals in a claimed scalar-raster round trip.

## Backport disposition

- Correct the archived zeta/alpha recommendation and mark the rejected budget
  catalog clearly: done in this recovery.
- Preserve the explicit distinction between ordered-current topology and
  containment in the range of sample averages. Neither implies universal
  arbitrary-direction absence of halos.
- Carry the coefficient-cone counterexample, admission-induced derivative
  mismatch, and the actual derivative of the spatial factor blend into an
  evidence-grounded manuscript revision. See `CONV_FUNCTIONAL_CURRENT_AUDIT.md`;
  some of these distinctions already appear in the manuscript.
- Keep the atlas fusion and census backport tied to its existing precise
  edit notes, `CONV_FOUR_CHILD_ATLAS_PAPER_TWEAKS.md`. Do not call fusion a new
  mathematical operator or its reduction gains a solved round-trip challenge.
- Require the actual native two-order demo baseline, unclipped values,
  phase/edge diagnostics as well as MSE, and total down/up cost. Target at
  most 1.2x, ceiling 1.5x. The 8x case is 512 to 64 to 512 for cameraman.

No new 8x image-quality or runtime result is claimed in this note.
