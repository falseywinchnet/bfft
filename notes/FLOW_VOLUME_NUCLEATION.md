# Flow-volume support

## One-shot method

The complete successful path is:

1. Run the BFFT pass sequence once.
2. At each pass, form the normalized event precision tensor \(Q_p\).
3. Convert it to a local support measure

   \[
   \rho_p^*={\sqrt{\det(Q_p+\ell_{\max}^{-2}I)}\over\pi}.
   \]

4. Reduce the stack by the transported upper envelope

   \[
   \rho_p=\max(\rho_p^*,\mathcal T_p\rho_{p-1}).
   \]

5. Retain the disjoint scale decomposition of that envelope:

   \[
   d\rho_1=\rho_1^*,\qquad
   d\rho_p=[\rho_p^*-\mathcal T_p\rho_{p-1}]_+.
   \]

   These are not births.  They are mutually exclusive constituents of one
   completed measure, analogous to decomposing a measure by the scale at
   which its constraint becomes binding.

6. Quantize every \(d\rho_p\) simultaneously with a fixed interleaved-gradient
   phase.  A pixel contributes a site exactly when its local measure crosses
   that phase.  No candidates, ordering, ranking, ownership, or iteration is
   involved.  The realized count is stochastic rounding of
   \(\int\rho_P\,dx\).
7. Give each site the eigenvectors and eigenvalue ratio of its own \(Q_p\).
   With aperture coverage \(k\), its ellipse is

   \[
   A_i={k\over\rho_p^*(x_i)},\quad
   {a_i\over b_i}=\sqrt{\lambda_{\max}/\lambda_{\min}},\quad
   a_i=\sqrt{{A_i(a_i/b_i)\over\pi}},\quad
   b_i=\sqrt{{A_i\over\pi(a_i/b_i)}}.
   \]

8. Fit one owner-free affine partition of unity.  There is no subsequent
   geometry optimization.

The broad precision floor is separated as its own isotropic component before
quantization.  It contributes only about ten supports on a square 128 px
image, but those broad supports close boundary holes and retain low-frequency
affine structure.  They are already part of the determinant measure and do
not increase its budget.

Compact aperture coverage \(k=5\) is the robust default.  For independently
located compact supports the uncovered fraction scales approximately as
\(\exp(-k)\); \(k=5\) is the 99.3% coverage point before the broad component
is included.  On the tested images the combined representation has complete
or greater than 99.98% discrete coverage.

### Measured results

At 128 px with no requested cell count:

| image | realized supports | one-shot ellipses | same centers circular | uniform circular |
|---|---:|---:|---:|---:|
| Pikachu | 1,234 | **30.20 dB** | 24.55 dB | 23.44 dB |
| camera | 1,612 | **31.92 dB** | 29.97 dB | 25.89 dB |
| astronaut | 1,747 | **29.53 dB** | 27.75 dB | 25.51 dB |
| grass | 2,398 | **27.58 dB** | 27.17 dB | 24.87 dB |

The same-center control isolates shape from placement.  The 5.65 dB gap on
Pikachu is the direct contribution of transport-derived scale and
anisotropy; the additional gap to uniform circles is density placement.

### Linear-time native path

`bfft_meyer_split_trace` returns every intermediate state from one native
pass sequence.  It is bit-identical to independently requesting pass counts
1 through \(P\), and was 9.3x faster at \(P=12\) in the validation probe.

`bfft_meyer_split_visit` streams those states without retaining a pass-deep
volume.  The production builder uses two linear visits: one for directional
persistence and one that immediately reduces and quantizes the measure.
Live geometry memory is \(O(HW+N)\), not \(O(PHW)\).

The HD coefficient finish uses fixed preconditioned CG.  Below two million
compact samples it uses a sparse design operator; above that point it calls
native compact-support forward/transpose/normal kernels and never constructs
the sparse design or normal matrix.  At 475x475:

| stage | measurement |
|---|---:|
| inferred supports | 8,522 |
| streaming geometry | 3.67 s |
| 40-pass native CG | 3.08 s |
| reconstruction | 30.90 dB |
| composite objective | 1.204e-3 |

The canonical viewer is `viewer/segmenting_veroni_viewer.py`.

## Correction: there is no population process

The earlier carrier/offspring interpretation was wrong, and the subsequent
bifurcation interpretation still smuggled in a population process.  The BFFT
pass axis is not time.  It is a stack of simultaneous support constraints.
Nothing is born, persists, dies, or divides while traversing it.

For normalized event tensor \(Q_p\), impose only a finite broad-support
horizon \(\ell_{\max}\).  The locally admissible support area is

\[
A_*(x,p)={\pi\over
\sqrt{\det(Q_p(x)+\ell_{\max}^{-2}I)}},
\qquad
\rho_*(x,p)=A_*^{-1}.
\]

The eigenvectors and eigenvalue ratio of the same tensor determine
orientation and anisotropy.  Its determinant determines density.  These are
not independent decisions.

If \(\mathcal T_p\) pushes a support measure from pass \(p-1\) into pass
\(p\), the correct combination is the transported upper envelope

\[
\rho_p=\max(\rho_p^*,\mathcal T_p\rho_{p-1}).
\]

The maximum is a union of obligations, not accumulation of cells.  The final
count is the integral \(\int\rho_P(x)\,dx\).  Sites are merely a one-shot
quantization of this final measure.  There is no initial count, ranking,
target count, deletion, residual-selected birth, separate fine population,
or bifurcation.

The rejected bifurcation audit used 180 initial cells at 128 px and produced
the following encouraging but contaminated counts:

| image | inferred cells |
|---|---:|
| Pikachu | 1,250 |
| camera | 1,534 |
| astronaut | 1,751 |
| grass | 2,051 |

The ordering remains evidence that the flow contains a usable density law,
but those numbers must not be treated as results of the corrected model.
`flow_support_measure.py` removes the initial-count assumption and computes
the transported support union directly.

The carrier-plus-splats results below remain useful representation controls.
They must not be read as the intended ontology.

## Result

The BFFT pass sequence contains useful support geometry that the final
cartoon/texture pair discards.  A cell can be emitted directly from this
scale-time volume with no image-space residual loop, ownership, candidate
search, ranking, top-k operation, deletion, or geometric refinement.

The first useful composition control was:

1. retain the known-good cartoon-effective support;
2. measure every BFFT pass transition;
3. locally quantize transition amplitude into fine support samples;
4. derive every fine ellipse from the normalized transition tensor;
5. optionally advect a sample through later, directionally persistent
   transitions before rasterizing it.

On the 128 px Pikachu control:

| model | cells | PSNR | cartoon MSE | texture MSE | objective |
|---|---:|---:|---:|---:|---:|
| known-good resource carrier | 458 | 28.65 | 1.913e-4 | 4.147e-4 | 1.970e-3 |
| carrier + 332 uniform residual circles | 790 | 29.53 | 5.911e-5 | 3.545e-4 | 1.529e-3 |
| carrier + same flow sites, circular | 790 | 29.77 | 7.714e-5 | 3.517e-4 | 1.482e-3 |
| carrier + flow ellipses | 790 | **30.16** | 6.729e-5 | **3.286e-4** | **1.359e-3** |

The gated scale-time advection is a small positive on this clean control
(30.149 -> 30.165 dB).  Most of the gain comes from flow placement and the
closed-form anisotropic support itself.

## Geometry

Let the outputs after BFFT pass \(p\) be cartoon \(c_p\), texture \(t_p\),
and cartoon-to-TV defect \(g_p\).  The event is

\[
s_p = (\Delta c_p,\Delta t_p,2^{-1/2}\Delta g_p).
\]

At each pixel:

\[
C_p=G_\sigma\sum_k s_{p,k}^2,\qquad
J_p=G_\sigma\sum_k\nabla s_{p,k}\nabla s_{p,k}^{T},\qquad
Q_p={J_p\over C_p+\epsilon}.
\]

\(Q_p\) has units pixel\(^{-2}\).  If its eigenvalues are
\(\lambda_1\geq\lambda_2\), the inferred support lengths are

\[
r_\perp=(\lambda_1+\ell_{\max}^{-2})^{-1/2},\qquad
r_\parallel=(\lambda_2+\ell_{\max}^{-2})^{-1/2}.
\]

The minor axis therefore becomes small exactly when the event becomes
spatially specific.  The major axis remains long only when that specificity
is consistent along its tangent.  On Pikachu, late passes have median minor
radius about 1.44 px and median major radius about 10.96 px (5.71:1).

### Curvature horizon

A straight ellipse cannot buy an arbitrarily long curved boundary.  If the
local tangent curvature is \(\kappa\), its departure from the curve after
length \(L\) is approximately \(\kappa L^2/2\).  Requiring this to remain
inside the minor radius gives

\[
L\leq\sqrt{2r_\perp/\kappa}.
\]

This cap is a small positive on Pikachu and neutral on cameraman.  It is a
valid geometric safeguard, not the main source of the gain.

## Local emission

The current research control uses

\[
\rho_p(x)\propto\sqrt{C_p(x)}
\]

and deterministic local stochastic rounding against one mass quantum.  The
target count only calibrates the quantum for matched-complexity experiments;
no cells are selected globally.

Pass 1 is best treated as part of the broad support transition.  Starting
fine emission at pass 2 is marginally better than allowing the initial
source-to-cartoon jump to dominate it.

## True scale-time transport

The glass gradient and event tangent are not the transport between states.
For consecutive cartoons, the experiment solves the local two-dimensional
normal-flow system

\[
\underset{v_p}{\operatorname{argmin}}\;
G_\sigma(\nabla c\,v_p+\Delta c_p)^2+\epsilon\|v_p\|^2
\]

in closed form per pixel.  A sample at pass \(p\) may be carried through
\(v_{p+1},\ldots,v_P\) before it splatters.

Transport persistence is

\[
\chi(x)=
{\left\|\sum_p q_p(x)v_p(x)\right\|
 \over
 \sum_p q_p(x)\|v_p(x)\|+\epsilon},
\]

where \(q_p\) is the local flow confidence.  Gating advection by \(\chi\)
improves clean, directionally consistent geometry and suppresses some
natural-image mis-transport.

## Complexity curve

The flow population by itself is not a good low-count substrate:

| total emitted population | Pikachu PSNR |
|---:|---:|
| 482 | 23.51 dB |
| 824 | 27.07 dB |
| 1,046 | 29.56 dB |
| 1,459 | 32.45 dB |

This was the central architectural clue.  Flow ellipses are an efficient
high-density texture basis, while low-frequency flow supports broad regions.
The corrected model regards both as scales of one bifurcating population.

## Rejected or qualified ideas

### Raw squared event mass

Pass 1 outweighs all later transitions and recreates an edge-heavy first
coat.  Amplitude, \(\sqrt{C}\), distributes mass more usefully over time.

### Capacity-only density

Replacing amplitude by
\(\sqrt{C}/(\pi r_\parallel r_\perp)^\gamma\) over-focuses fine events.  It
helps cameraman for \(\gamma=0.5\), but degrades Pikachu and astronaut.
Broad and specific flow must coexist; specificity cannot replace amplitude.

### Fixed-pixel causal accumulation

Accumulating mass vertically through pass number is worse on all controls.
A fixed image coordinate is not a scale-time trajectory.  Causality needs
advection through the measured flow.

### Transporting substrate density

A one-shot Poisson transport from uniform germs toward blurred event density
moves broad cells back onto edges and recreates the earlier crowding failure.
The substrate must remain space-filling.

### Tilting or advecting the substrate

Weak event-tensor, glass-vector, and true normal-flow hints are
image-dependent and negative overall when stapled onto a fixed broad
substrate.  They do not refute transport-conditioned bifurcation of a single
population.

### Separate cartoon/texture fits

Circular cartoon supports plus anisotropic texture supports help some natural
images but are not consistently better than fitting the full field with the
flow ellipses.  The support law is more important than imposing a semantic
two-layer fit.

## Files

- `experiments/bfft_flow_stage_geometry.py` builds and visualizes the full
  pass volume.
- `experiments/flow_volume_cells.py` emits and fits a frozen one-shot
  population with matched controls.
- `experiments/coarse_support_flow_splats.py` composes the known-good carrier
  with flow-born residual splats.
- `experiments/flow_bifurcation_density.py` records the rejected population
  interpretation.
- `experiments/flow_support_measure.py` infers the final support density and
  count without population dynamics or a requested cell count.
- `experiments/out/flow_stage_geometry.png` shows selected pass states.
- `experiments/out/flow_stage_summary.png` summarizes emergence time and
  anisotropy.
- `experiments/out/resource_carrier_flow_splats_advected.png` is the strongest
  composition from this round.

## Next experiment

Quantize the final transported measure directly.  A local phase field or
weighted blue-noise threshold can turn its mass into sites in one parallel
operation.  Each site samples its support tensor from the same final measure.
There must be no ancestry or iterative creation story hidden inside the
quantizer.

## Partition diagnostics and aperture conditioning

The canonical viewer now exposes three different meanings of "cell":

1. **Cell outlines** are the literal compact ellipses sampled from the
   transported precision tensor.
2. **Soft site IDs** replace the fitted affine coefficients with deterministic
   hashed colours and render them with the exact normalized compact-support
   weights.  This is the direct analogue of SAD's Site IDs view.
3. **Dominant site IDs** round the soft partition to its maximum contributor
   for diagnosis only.  The reconstruction itself still has no pixel owners.

The soft-ID raster agrees with an independently assembled sparse
partition-of-unity render to \(2.9\times10^{-8}\) absolute error.

The user's observation that large smooth-aperture coverage removes defects is
support-conditioning evidence.  On Pikachu at 128 px with frozen centers and
anisotropy:

| aperture coverage | covered pixels | effective contributors p10 / median | median dominance | pixels with effective count < 1.25 |
|---:|---:|---:|---:|---:|
| 2 | 99.896% | 1.84 / 3.34 | 0.402 | 2.98% |
| 3 | 100.000% | 2.82 / 4.82 | 0.293 | 0.31% |
| 5 | 100.000% | 4.67 / 7.69 | 0.192 | 0.00% |
| 8 | 100.000% | 7.55 / 11.72 | 0.129 | 0.00% |
| 10 | 100.000% | 9.38 / 14.35 | 0.107 | 0.00% |

Thus many visible seams at low aperture are not failed nucleation.  They occur
where the normalized basis is supported by too few functions or crosses a
compact rim with poor redundancy.  The next geometry refinement should infer
the smallest locally sufficient aperture from this condition field while
checking that soft IDs do not wash across actual image boundaries.

## Fixed-population canopy descent

The Site IDs diagnostic falsified the static method as a segmentation: its
high PSNR was produced by an overcomplete spline field with roughly 8--12
effective contributors per pixel.  A new control fixes the population once
from the final measure and carries those same sites through every BFFT pass.
At each pass, one simultaneous sparse canopy step transports the centroids,
balances one quantum of support mass per site, moves sites toward their
support-measure centroid, updates covariance and reach, and anneals the
partition.  There are no births, deletions, candidates, or target-RGB geometry
gradients.

At the calibrated Pikachu setting (128 px, overlap 8, sharpness 1.5 to 10),
1,290 fixed sites reach 30.46 dB with 1.94 median effective contributors.
Cross-image results versus the static overlap-8 control are:

| image | static PSNR / objective | canopy PSNR / objective |
|---|---:|---:|
| Pikachu | 29.67 / 0.001496 | 30.46 / 0.001241 |
| camera | 31.29 / 0.002189 | 31.44 / 0.001824 |
| astronaut | 28.89 / 0.001720 | 28.68 / 0.001801 |
| chelsea | 34.98 / 0.000724 | 34.19 / 0.000752 |
| coins | 27.93 / 0.002643 | 29.12 / 0.001692 |
| grass | 26.85 / 0.003344 | 26.58 / 0.003081 |

This is a geometry/quality Pareto advance rather than a universal score win.
The narrow calibration is informative: covariance response that is too weak
cannot form detail, too strong creates unstable needles; sharpness above 10
hardens faster than the sites can reach their support.  The current prototype
is `experiments/transport_canopy_cells.py` and is exposed as the default
research mode in the canonical segmentation viewer.
