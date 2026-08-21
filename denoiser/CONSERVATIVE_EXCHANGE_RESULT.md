# Conservative posterior--residual exchange

## Status

This is a promising research dynamics, not a promoted denoiser and not an
FMMT repair.  It implements a two-reservoir interpretation of smoothing while
preserving the observation exactly.  The operator now improves both average
error and edge retention on the first broad screen, but it does not yet have a
truth-consistent terminal law.  It is therefore not connected to the Dear
PyGui workbench.

## State and smoothing operator

The state is an exact signed decomposition

\[
    y=p+r.
\]

Neither reservoir is ontologically named truth or noise.  For a field `f`,
the smoothing act is the positive screened resolvent

\[
 S_f=\left(I+{L(M(p,r^2),1-\chi_f)\over d_{\max}}\right)^{-1}.
\]

`M` is the evolving structure/uncertainty metric.  Its inverse is decomposed
into nonnegative local lattice fluxes by Selling reduction, and `L` is the
resulting symmetric conservative Laplacian.  The operator-normalized time
`1/d_max` is fixed by the generator and is not a smoothing-strength setting.
The field `chi_f` is reciprocal phase susceptibility conditioned on the
action of the complete continuous-scale decomposition.  Phase is therefore a
null authority inside the operator rather than a post-hoc choice of a texture
band.

## One cycle

### 1. Posterior erosion

\[
 \delta_p=p-S_p p,\qquad
 p\leftarrow S_p p,\qquad
 r\leftarrow r+\delta_p.
\]

Anything the posterior smoothing rejects is donated to the residual exactly.

### 2. Residual return

\[
 q_r=S_r r,\qquad
 \delta_r=\chi_r q_r,\qquad
 p\leftarrow p+\delta_r,\qquad
 r\leftarrow r-\delta_r.
\]

The positive smoother supplies a possible explanation, but smoothing cannot
certify its own amplitude.  Only the reciprocal-phase-supported part returns;
the rest remains explicit residual ignorance.

### 3. Witnessed joint closure

The literal interpretation `p <- S_y(p+r)` was tested first and rejected.  It
substituted the smoothed observation wholesale and imported too much
replacement amplitude.  The corrected joint act forms only a proposal

\[
 c=S_y(p+r)-p.
\]

A target-excluded Selling-neighbour cavity regresses `c` on the posterior's
normalized curvature.  Let `z_cavity` be its bounded Schur explained-action
coordinate.  Residual phase and cavity relation are independent necessary
witnesses, so their continuous intersection is the parameter-free Hellinger
product

\[
 a=\sqrt{\chi_r z_{\rm cavity}}.
\]

The closure is

\[
 p\leftarrow p+ac,\qquad r\leftarrow y-p.
\]

Passing either witness alone cannot certify a correction.  Every substep
retains `p+r=y` pointwise to floating-point roundoff.

The transferred object is the complete signed field, not a scalar energy.
Separate squared reservoir energies are not additive under a signed transfer
because of their cross term.  The implementation records shed, donation,
refusal, joint-proposal, admitted-joint, and displacement actions rather than
claiming a false scalar conservation law.

## Falsified literal cycle

The first implementation donated all of `S_r r` and replaced the posterior by
`S_y y`.  On clean Cameraman, tapered hair, and woven chirps after six cycles,
mean MSE improved from `.000504` to `.000166` and edge retention from `.825`
to `.911`.  On mixed replacement-plus-uniform corruption, however, MSE worsened
from `.010691` to `.019311` even as edge retention rose from `.461` to `.616`.

Substep ablation located the failure in wholesale joint assignment.  Before
that closure, the phase-supported exchange slightly improved mixed Cameraman
and woven error while raising their edge retention.  The cycle was recovering
support and simultaneously importing unsupported amplitude.

## Witnessed-cycle measurements

At 32 pixels, three witnessed cycles give:

| group (3 scenes) | initial MSE | cycle-3 MSE | initial edge | cycle-3 edge |
|---|---:|---:|---:|---:|
| clean | .000504 | .000356 | .825 | .863 |
| mixed replacement + uniform .25 | .010691 | .011191 | .461 | .531 |

The mixed result is now near the initial error while recovering appreciable
structure, rather than doubling the error.  Cameraman mixed error improves
through cycle two (`.014808 -> .014168`) while edge retention rises
`.452 -> .482`.  Tapered hair and woven chirps gain more edge action than error
at this stage.  None of the six orbits reached a numerical equilibrium.

The more important screen is the first witnessed cycle across three scenes
and all nine unnamed corruption conditions in the repository battery:

| corruption | mean initial MSE | mean cycle-1 MSE | change | edge retention |
|---|---:|---:|---:|---:|
| uniform .10 | .002169 | .002081 | -4.05% | .727 -> .743 |
| Gaussian .10 | .004016 | .003840 | -4.40% | .722 -> .750 |
| Laplace .08 | .004703 | .004438 | -5.63% | .664 -> .683 |
| multiplicative .12 | .002459 | .002289 | -6.88% | .755 -> .778 |
| replacement .10 | .003544 | .003545 | +0.00% | .688 -> .724 |
| replacement .25 | .012663 | .012356 | -2.42% | .504 -> .532 |
| salt--pepper .10 | .006622 | .006655 | +0.50% | .626 -> .673 |
| mixed .10 | .005081 | .005017 | -1.26% | .598 -> .624 |
| mixed .25 | .010691 | .010691 | +0.00% | .461 -> .498 |

Across all 27 corrupted scene/condition pairs, mean MSE changes
`.005772 -> .005657`, mean edge retention changes `.638 -> .667`, MSE improves
in 20 cases, and edge retention improves in all 27.  Truth was used only for
retrospective measurement, never by the operator.

## What remains unresolved

1. **Terminal law.** Later exchange can continue gaining edges after it begins
   importing corruption.  Observation fidelity, fixed-point contraction, and
   displacement decay cannot identify that transition because the observation
   contains the corruption.  A numerical cycle ceiling is only a guard.
2. **Amplitude coverage.** Reciprocal phase can justify structural transport,
   but it is not a calibrated amplitude posterior.  The next state should
   carry bounded residual enclosures on antisymmetric Selling-edge fluxes.
   Conclusive set violation may release amplitude; set membership must never
   create probability.
3. **Oscillatory clean texture.** The Hellinger conjunction is slightly too
   restrictive for clean woven chirps: edge retention improves, but MSE rises
   modestly.  The curvature cavity is weak on reciprocal oscillatory support.
   Phase and curvature must remain separate feasible components rather than
   being permanently collapsed into one scalar authority.
4. **Continuous-scale law.** The heat decomposition is exact, but the current
   per-generation phase recurrence still depends on the stored semigroup
   trace.  A scale-density limit is required before the posterior is
   representation invariant.
5. **Performance.** Every cycle presently recomputes several Python/SciPy
   continuous-scale transports and sparse solves.  Native fusion would be
   premature until the state, enclosure transport, and terminal law stabilize.

The next experiment should retain the two explanation modes as a small
zonotopic mixture carried by conservative edge flux.  Posterior erosion and
residual return then change enclosure generators and component coverage, not
only a scalar image.  A cycle ends when no feasible component can exchange
additional signed action—not when an iteration count, MSE surrogate, or
smoothing depth says to stop.

The exact posterior-shed flux must also remain a separately labelled lineage
component. Once shed ancestry is marginalized into an anonymous residual,
later phase and curvature witnesses cannot distinguish recovered structure
from newly invented amplitude.

## Executable artifacts

- `conservative_exchange_transport_2d.py`: exact exchange orbit and ledgers.
- `test_conservative_exchange_transport_2d.py`: conservation, fixed-point,
  trajectory, and input invariants.
- `probe_conservative_exchange_transport_2d.py`: clean/mixed or full
  corruption trajectory battery.
- `probe_exchange_transfer_laws_2d.py`: return/closure falsification study.
