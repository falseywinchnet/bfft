# Transport Chambolle: cross-project research ledger

## Transfer audit

The five source projects agree on a stricter operator principle than ordinary
anisotropic TV.

1. **V3 segmenter.** A precision measure becomes a positive Selling lattice,
   then a causal first-arrival graph. The valuable state is incidence plus
   parent/reverse transport, not the final segment labels or a local edge mask.
2. **Representation-residual codec.** Geometry wins when it supplies a better
   coordinate for the exact correction. A lower-error support raster can be a
   worse representation, and boundary error requires a two-sided state.
3. **CONV conservative interpolation.** Coarse prediction and stored detail
   are algebraically distinct coordinates. Analysis/synthesis is exact, while
   zero detail is merely one admissible section.
4. **Meyer cartoon decomposition.** Discontinuity is an oriented bond measure,
   not a scalar halo. Hodge lift and projection are constructive flux proofs;
   acceleration is safe on a frozen flux problem.
5. **High Vision / Night Vision.** Independent evidence lanes remove
   self-products, and connection or circle-valued state must be transported
   without first collapsing its gauge to a scalar mean.

For denoising this means that target-excluded observation, positive transport
incidence, exact correction, and uncertainty about the transport coordinate
must coexist. None can be reconstructed after premature scalar smoothing.

## Hypothesis ledger

### H0 — Cartesian transport Chambolle

The first form used two outgoing Cartesian fluxes with an oriented local box.
It improved structural retention over classical TV but under-represented
signed replacement residuals.

### H1 — positive Selling-edge network flow

For Selling incidence \(B\), removable resistance \(r_N\), and endpoint-union
edge capacities \(h_e\), solve

\[
  \min_{|p_e|\le h_e}\frac12\|r_N+B^Tp\|_2^2.
\]

Each point allocates its removable amplitude over incident edges in proportion
to positive Selling conductance, so its allocation sums exactly to that
amplitude. Every correction \(B^Tp\) has zero total mass.

This improves clean, Gaussian, and uniform cases but is too conservative for
sparse replacement. At 32 pixels its aggregate is MSE 0.007602, SSIM 0.6188,
edge retention 0.7512. The topology is useful; local capacity allocation is
not a complete residual coordinate.

### H2 — literal conservative CONV detail

The unsupported residual was analyzed into a parent mean and three exact 2x2
CONV moment details. Removing only the detail retains edge action (0.793 after
one pass) but gives MSE 0.0100. Perfect reconstruction is necessary state
bookkeeping, not evidence that all fine detail is noise.

### H3 — independent-lane cross action

Pairwise products of the three target-excluded residual lanes remove self
energy and improve corruption MSE, but common clean fine structure appears in
every lane too. Two passes of the median statistic reach MSE 0.00660 while
clean MSE rises to 0.00360 and edge retention collapses to 0.401. The Night
Vision cross-product lesson requires transported identity before common action
can be called noise.

### H4 — Hodge-lifted transport-uncertainty zonotope

One Selling Laplacian factorization lifts the observation departure and every
CONV generator into minimum-action edge flux. Sum, maximum, RMS, and mean
generator bodies were tested. The literal zonotope is far too permissive:
global Hodge lifts turn local witness uncertainty into long correlated flux.
Its 32-pixel MSE is 0.01611. Narrower bodies denoise only by recreating clean
TV loss. The operator is rejected, though its required componentwise Hodge
factorization is retained.

### H5 — population phase of Selling flow and exact detail

The two surviving coordinates are not collapsed. Their projective phase is

\[
 \alpha=min\left(1,
   \frac{\sqrt{\mathbb E r_N^2}}
        {\sqrt{\mathbb E s_{predictive}^2}}
 \right).
\]

Let \(u_G\) be the conservative Selling-flow result and
\(u_D=y-r_N\) the exact-detail result. The readout is

\[
 u=(1-\alpha)u_G+\alpha u_D.
\]

This is not a truth-fitted blend: \(\alpha\) is the amplitude coordinate of
two measured actions. It approaches Selling flow while removable action lies
inside predictive transport uncertainty and approaches exact correction when
that action dominates.

### H6 — paired one-sided jump trace

The codec's two-sided boundary state and Meyer's BV bond measure suggest the
affine-annihilating trace

\[
 [f]_{i+1/2}=\tfrac32(f_{i+1}-f_i)-
               \tfrac12(f_{i+2}-f_{i-1}).
\]

Its between-chart covariance was subtracted from its mean outer product and
the positive tensor part supplied the Selling metric. This is a valid
structure diagnostic, but not an amplitude authority. At 32 pixels it raises
edge retention only from 0.74595 to 0.74831 while worsening MSE to 0.006847
and SSIM to 0.63218. Its large geometric-mixed edge gain coincides with a
large distortion loss. The metric-only placement is rejected.

### H7 — pointwise population phase

Replacing the global RMS quotient with
\(\alpha(x)=\min(1,|r_N(x)|/s_{predictive}(x))\) loses the ancestry shared by
adjacent sites. Aggregate MSE rises to 0.006863, SSIM falls to 0.62771, and
edge retention falls to 0.71644. Locality is not causal identity; this form is
rejected.

### H8 — transported phase-action authority

The continual-noise system contributes a state the first Chambolle forms did
not retain: paired phase-covector numerator and denominator. These sufficient
statistics, rather than their quotient or angle, are moved through the
positive Selling Markov operator. Their departure from the integrable 2-D
covector manifold gives phase-noise authority \(A_p\). The local projective
amplitude authority is

\[
 A_a=\min\left(1,\frac{|r_N|}{s_{predictive}}\right).
\]

They are joined as independent reasons for removal,

\[
 A_N=1-(1-A_p)(1-A_a), \qquad \widetilde r_N=A_Nr_N.
\]

The population-phase Selling/CONV contraction then acts on
\(\widetilde r_N\). There is no noise label, threshold, fitted mixture weight,
or selected band. Either transported statistic can justify removal; both
must refuse it to protect structure. A phase-only veto was also measured: it
preserved clean structure very well but under-denoised replacement damage,
which is why the amplitude witness is a necessary independent coordinate.

### H9 — distinct-chart phase action

High Vision's independent-gatherer result transfers because the four CONV*
parity charts are synthesized from disjoint anchor sublattices. For direction
\(d\), chart \(k\) supplies target-excluded one-sided jets
\(m_k=f_k(x)-f_k(x-d)\) and \(p_k=f_k(x+d)-f_k(x)\). Their ordered
distinct-chart moments are evaluated without enumerating pairs:

\[
 C_{mm}=\frac{(\sum_k m_k)^2-\sum_km_k^2}{K(K-1)},\quad
 C_{pp}=\frac{(\sum_k p_k)^2-\sum_kp_k^2}{K(K-1)},
\]

\[
 C_{mp}=\frac{(\sum_km_k)(\sum_kp_k)-\sum_km_kp_k}{K(K-1)}.
\]

The symmetric jet tensor is projected onto the PSD cone. Its cross entry and
half trace become phase numerator and denominator, which are then transported
separately. Thus same-chart noise energy never enters the phase readout.
Boundary directions with fewer than two target-excluded charts contribute
zero evidence and are reported rather than filled from the observation.
These cross sufficient statistics are measured exactly once from the original
observation. Later observer cycles transport that fixed evidence through their
evolving metric; they never rebuild supposedly independent charts from an
estimate that has already mixed all four lanes.

Ordering matters. Transporting each chart lane before taking cross moments
was tested and rejected: spatial transport makes formerly disjoint corruption
partly common and harms geometric Gaussian edges. Likewise, scalar union and
intersection of residual-phase and chart-phase authorities both lose. The
surviving construction contracts distinct charts locally, then transports the
sufficient statistic without retaining a scalar branch posterior.

### H10 — exact chart-lane flux synthesis

The admitted residual was partitioned exactly over its three valid chart
identities, each part was contracted through a chart-specific Selling metric,
and the four conservative divergence readouts were summed. This preserves
\(r=\sum_kr_k\) and exact observation recomposition, but reaches only MSE
0.006721 / SSIM 0.63245 / edge 0.75489 at 32 pixels while requiring four flux
solves. Exact ancestry did not materially change the endpoint and is rejected
as a production representation.

### H11 — Meyer outer-map defect

A second ROF evaluation measured the state-to-state defect of the coherent
survivor. Adding that defect to removable action lowers MSE to 0.006675 and
substantially improves replacement corruption, but reduces aggregate edge
retention to 0.74655 and damages clean geometric structure. Keeping the
ordinary residual and defect as separately authorized lanes recovers only
part of the loss. The defect is genuine information about remaining descent,
but it is not by itself noise authority.

### H12 — continuous and conserved stopping attempts

Three stopping repairs were falsified.

- Intersecting the terminal transport step with exact noise/structure action
  equality nearly restores clean identity and raises edge retention to 0.775,
  but under-denoises mixed and replacement corruption.
- Freezing initial admitted action as a Euclidean displacement budget confuses
  transport-coordinate evidence with pixel norm and worsens MSE to 0.00793.
- Freezing a pointwise correction zonotope preserves edges (0.812) but prevents
  correction authority from moving through reconstruction neighbourhoods;
  every case hits the cycle ceiling and MSE worsens to 0.00827.

The current whole-cycle phase balance remains the balanced empirical stop.
Its endpoint is numerically jagged, but the tested continuous replacements
select a different clean/denoising frontier rather than dominating it.

### H13 — target-excluded complete-action stopping

Stopping can be measured without a corruption label by retaining the four
disjoint CONV* parity observers through time.  Each observer lane is generated
from one parity sublattice and scored only on the other three, so no target is
allowed to validate an estimate that sampled that target.  The first tested
complete action is the sum of the three-observer barycentre error and original
source displacement not authorized by the fixed phase/amplitude law.  The
source-protection term is divided by three because it is paired with a
three-observer barycentre.  Between numerical generations this action is an
exact scalar quadratic, giving a continuous rather than integer-cycle
endpoint.

Unconstrained held-out action improves noisy MSE but smooths clean geometric
interfaces.  The causal repair inherits the lifted endpoint contractor: let
``tau`` be the exact continuous noise/structure phase crossing and ``T`` the
held-out action endpoint.  A later endpoint receives temporal authority
``tau / T`` and is contracted affinely toward the phase-balanced estimate.
Zero witnessed phase time therefore gives exact identity; sustained positive
noise action opens the late branch without a threshold.

The barycentre action is the first tested stop to improve both distortion and
edge retention over distinct-chart phase at both measured sizes.  At 32 pixels
it gives 0.006650/0.6335/0.7761 MSE/SSIM/edge; at 64 pixels it gives
0.006217/0.5879/0.8056.  It nevertheless leaves some diffuse uniform noise at
identity because collapsing the observer family discards disagreement.

Retaining the complete three-observer population through the same quadratic
endpoint repairs that failure.  This is now the primary research form.  It
gives 0.006562/0.6361/0.7525 at 32 pixels and
0.006096/0.5936/0.7877 at 64 pixels.  The 64-pixel form beats DCNT on all three
measures and beats distinct-chart phase on MSE and SSIM while giving up 0.0017
edge retention.  Clean cameraman remains exact identity, and clean woven MSE
is 0.0000024.

The four research observer lanes still cost more than the primary solve. They
should be fused as one four-component lifted transport after the stopping law
is stable.  The remaining theoretical refinement is to retain the
barycentre's exceptional edge action and the population's additive-noise
action in one transported two-coordinate endpoint instead of selecting either
readout.

## Matched M4 results

### 32 pixels, 15 source/corruption cases

| method | MSE | SSIM | edge retention |
|---|---:|---:|---:|
| classical Chambolle 0.10 | 0.007045 | 0.5591 | 0.4203 |
| Cartesian transport Chambolle | 0.007111 | 0.6255 | 0.7299 |
| raw Selling-edge Chambolle | 0.007602 | 0.6188 | 0.7512 |
| DCNT uncertainty | 0.006696 | 0.6311 | 0.7261 |
| population-phase Chambolle | 0.006753 | **0.6334** | 0.7459 |
| phase-action Chambolle | 0.006721 | 0.6332 | **0.7542** |
| distinct-chart phase Chambolle | **0.006695** | 0.6323 | 0.7531 |
| barycentric causal stop | 0.006650 | 0.6335 | **0.7761** |
| complete-population causal stop | **0.006562** | **0.6361** | 0.7525 |
| integrated FMMT control | 0.004088 | 0.6941 | 0.5390 |

### 64 pixels

| method | MSE | SSIM | edge retention |
|---|---:|---:|---:|
| classical Chambolle 0.10 | 0.006068 | 0.5495 | 0.4196 |
| Cartesian transport Chambolle | 0.006478 | 0.5904 | 0.7677 |
| raw Selling-edge Chambolle | 0.006956 | 0.5744 | **0.7898** |
| DCNT uncertainty | **0.006319** | **0.5907** | 0.7589 |
| population-phase Chambolle | 0.006399 | 0.5890 | 0.7829 |
| phase-action Chambolle | 0.006368 | 0.5885 | 0.7875 |
| distinct-chart phase Chambolle | 0.006332 | 0.5882 | 0.7894 |
| barycentric causal stop | 0.006217 | 0.5879 | **0.8056** |
| complete-population causal stop | **0.006096** | **0.5936** | 0.7877 |
| integrated FMMT control | 0.002892 | 0.7209 | 0.5950 |

Residual phase-action is the balanced current transport-Chambolle compromise:
it improves population phase's MSE and edge retention at both resolutions
with a very small SSIM trade. Distinct-chart phase is the lower-distortion
edge of the same frontier. Neither is promoted as a final denoiser: FMMT
remains much cleaner but loses far more edge action, and mixed corruption
remains unresolved. At 64 pixels residual phase-action costs 0.919 s/case in
research Python on the M4 Mini versus 0.827 s for population phase.

Distinct-chart phase now defines the lower-distortion edge of that frontier.
At 32 pixels its MSE is 0.000001 below DCNT while its SSIM and edge retention
are higher; at 64 pixels it is only 0.000013 above DCNT and retains 0.7894
rather than 0.7589 edge action. Residual phase-action remains the
preferred balanced display because it has slightly better SSIM. Reusing the
chart family already carried by the flux body reduces the distinct-chart M4
cost to 0.199 s per 32-square case, only about 7% over residual phase-action.
At 64 pixels, measuring chart evidence once makes it slightly faster than
residual phase-action (0.915 versus 0.919 s/case in the matched run).

## Current boundary

The next missing state is not another capacity function. Phase-action shows
that transported sufficient statistics outperform scalar local phase, but
the exact CONV correction is still mixed by one global population quotient.
V3 and FlowCells say that a local graph flow without causal parent identity
averages incompatible branches. The next experiment should derive roots from
the observer law itself and retain the V3 one-parent/barycentric-parent stream,
so exact detail is returned along an achieving branch rather than assigned by
a pixelwise or image-wide heuristic. It must preserve:

- target exclusion;
- positive Selling incidence;
- exact observation recomposition;
- branch-specific correction ancestry;
- refinement-invariant phase stopping.

H13 supplies a continuous causal endpoint and materially changes this
boundary. Complete population action makes diffuse additive corruption
visible without a MAD estimate, noise class, fitted multiplier, or cycle
threshold. The next state must retain population spread and barycentric edge
support together as causal coordinates. Only after that law is stable should
the four lanes be fused into a shared lifted representation and native
implementation.
