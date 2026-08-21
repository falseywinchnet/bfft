# Causal scale transport: first exact filtration and its falsification

## Status

This experiment establishes the correct decomposition order but does **not**
yet establish the final denoising posterior.  The implementation is
`causal_scale_transport_2d.py`; the 32-square oracle record is
`causal_scale_transport_32.json`.

The decisive advance is procedural and mathematical: the observation is
completely decomposed before any fine component can be removed.  The decisive
negative result is that the first confidence recurrence is not invariant to
refinement of the numerical scale trace.  It must therefore remain a probe,
not enter the GUI or replace the active image method.

## A continuous filtration rather than chosen bands

Let `L0` be the positive conservative Selling Laplacian emitted by the identity
metric.  On the rectangular reflected lattice it is diagonalized exactly by
the DCT-II.  Its heat orbit is

\[
u(t)=\exp(-tL_0)y.
\]

The continuous detail measure is

\[
g(t)=L_0\exp(-tL_0)y=-\partial_tu(t),
\]

and the scene has the exact coarse-to-fine representation

\[
y=\bar y+\int_0^\infty g(t)\,dt.
\]

The code samples this orbit at nested times only to inspect it.  Every finite
trace telescopes exactly:

\[
y=u(t_N)+\sum_{k=0}^{N-1}\{u(t_k)-u(t_{k+1})\}.
\]

Thus no fine noise is smoothed away while a coarse scene is being guessed.
All components exist before the provisional confidence readout is formed.

## Directional action, isotropic action, and phase

At a generation's own transport time, its first-jet second moment is a PSD
tensor `J`.  If its eigenvalues are `lambda1 >= lambda2 >= 0`, the experiment
carries two additive actions,

\[
A=\lambda_1-\lambda_2,\qquad I=2\lambda_2,
\]

so `tr(J)=A+I`.  `A` is directional excess and `I` is unresolved isotropic
action.  Isotropy is not called noise.  In the first recurrence it can enter
only when reciprocal-chart phase persists from the preceding coarser state.

The complete three nonzero mod-two covectors provide reciprocal disjoint
charts.  Their signed sufficient statistics are transported through the
Selling metric inherited from the already accepted coarse scene.  A chart
phase is

\[
h_q=\frac{\mathcal T(2a_qb_q)}
{\mathcal T(a_q^2+b_q^2)},\qquad |h_q|\leq1,
\]

and `h_q^2` is phase certainty.  This retains antisymmetric texture rather
than equating agreement with equal sample values.

## Two confirmation errors now removed

The first implementation allowed tensor ancestry to admit a component by
itself.  The oracle trace showed why that is false: adjacent heat generations
of the same noisy observation are automatically aligned.  On the finest
mixed-Cameraman generation it retained `0.616` of component action while the
oracle bounded signal gain was only `0.046`.  Same-orbit ancestry is geometry,
not independent evidence.

The retained probe therefore requires an independently witnessed phase birth.
Lineage can route and geometrically couple that birth, but cannot manufacture
authority.  It also constructs each later metric from the confidence-bearing
scene, not from the exact raw recomposition.  Otherwise rejected noise writes
the geometry used to judge later texture.

## What works at the base numerical trace

At 32 x 32 under mixed replacement plus uniform corruption:

| source | observed MSE | scale readout MSE | SSIM | edge retention |
|---|---:|---:|---:|---:|
| Cameraman | .047895 | .015865 | .4382 | .4679 |
| tapered hair | .039122 | .010126 | .3998 | .3837 |
| woven chirps | .029809 | .007628 | .3985 | .2983 |

The scale law removes substantial corruption, including null fields: Gaussian
null MSE falls from `1.0115` to `0.1333`, and uniform null MSE from `.08208` to
`.01289`.  Clean scenes are not fixed points, however.  Clean Cameraman MSE is
`.000716` with `.829` edge retention.  This is visibly too soft and therefore
not competitive with the desired posterior.

The local oracle explains the remaining blur.  On mixed Cameraman, confidence
retains more true action than noise action at every measured generation.  Its
local correlation with truth-versus-noise dominance is about `.51` in the
middle of the orbit but decays to `.066` at the finest endpoint.  Global scale
ordering is working; spatial ownership at fine scale is not yet sharp enough.

At 96 x 96 the provisional Cameraman readout changes MSE from `.04244` to
`.01085` and SSIM from `.1753` to `.4208`, but edge retention is only `.5433`.
The tripod and hair remain visibly attenuated.  Runtime is about 2.7 seconds in
the current Python/SciPy research form; performance is irrelevant until the
state equation is corrected.

## The refinement falsification

The semigroup decomposition remains exact under nested refinement, but the
confidence-weighted output does not yet converge:

| trace generations | Cameraman mixed MSE | SSIM | edge retention |
|---:|---:|---:|---:|
| 16 | .015865 | .4382 | .4679 |
| 32 | .016749 | .4135 | .5126 |
| 64 | .017368 | .4022 | .5660 |

The RMS output changes are `.01928` and `.01774`, not a contracting sequence.
Therefore a per-generation confidence recurrence still makes the numerical
trace act like hidden bands, even though the underlying decomposition is
continuous and exact.

## Hadamard pull and the transport spectrum

The next experiment makes “transport refused to pull” measurable.  For each
reciprocal disjoint chart pair `(a,b)`, positive Selling transport acts on the
complete outer-product moment,

\[
C_q=\mathcal T
\begin{pmatrix}a_q^2&a_qb_q\\a_qb_q&b_q^2\end{pmatrix}\succeq0.
\]

Hadamard's determinant inequality gives

\[
0\leq \frac{\det C_q}{C_{11}C_{22}}\leq1.
\]

Its complementary pull coordinate is

\[
P_H=1-\frac{\det C_q}{C_{11}C_{22}}
=\frac{C_{12}^2}{C_{11}C_{22}}.
\]

The normalized correlation eigenvalues are
`1 +/- sqrt(P_H)`.  Supported observations collapse toward one eigen-direction
and have `P_H -> 1`; unpulled independent variation retains two-dimensional
volume and has `P_H -> 0`.  Equivalently, its effective rank is

\[
r_{eff}=\frac{2}{1+P_H}\in[1,2].
\]

This does say something real.  It is insensitive to unequal amplitudes in two
coherent charts, unlike the signed phase coordinate.  It also rejects null
fields more strongly.  But it is conservative: on the 32-square mixed
Cameraman control its MSE/SSIM/edge retention is
`.01905/.3780/.3626`, while the signed spectral phase gives
`.01809/.3983/.5038`.

The crucial correction is to evaluate this moment at the operator's normalized
time `1/max_degree(L)`, not at the width of a stored scale interval.  The
Hadamard readout's nested-trace RMS changes become about `.00198`, `.00146`,
and `.00147` over 16, 32, 64, and 128 generations.  The normalized signed-phase
readout gives `.00167`, `.00154`, and `.00147`.  These are an order of magnitude
smaller than the rejected recurrence and now describe a converging continuous
scale observable rather than a hidden band setting.

There is a second spectral coordinate within the spatial transport itself.  If
`g` is one scale-density component,

\[
P_T=\frac{(\mathcal T g)^2}{\mathcal T(g^2)}\leq1
\]

by Jensen's inequality.  Its global companion is the normalized Rayleigh
eigenvalue `g^T L g/(max_degree(L) g^T g)`.  Low Rayleigh action and high `P_T`
mean the component survives as a coherent transport direction; noise that the
operator cannot pull has low local concentration.

This coordinate preserves clean detail exceptionally well when smoothly
united with signed phase: clean Cameraman, tapered hair, and woven-chirp edge
retention becomes `.934`, `.930`, and `.945`.  It is not independent evidence,
however, and the same union retains too much null and mixed corruption.  A
Hellinger coupling with Hadamard pull is safer but again too conservative.

The result is therefore a useful separation, not a final combination:

1. Hadamard deficit measures collapse of independent observation volume;
2. signed spectral phase measures coherent antisymmetric structure;
3. Jensen/Rayleigh pull measures survival under spatial transport;
4. none may manufacture authority merely because another coordinate moved.

At 96 x 96 mixed Cameraman, Hadamard pull gives MSE/edge
`.01342/.460`, spectral phase `.01376/.596`, and the permissive spatial-pull
union `.02527/.689`.  The latter exposes substantially more tripod and hair but
also visibly retains noise.  The missing posterior must transport these as a
joint law rather than collapse them by a product, union, or fitted blend.

### Rejected direct eigenprojection

The Hadamard moment also suggests a matrix-valued extractor rather than a
scalar confidence.  The continuous spectral-excess operator

\[
K=\frac{C-\lambda_{min}(C)I}{\operatorname{tr}C}
\]

has eigenvalues `(lambda_max-lambda_min)/tr(C)` and zero.  It becomes the exact
principal projection for rank-one support and vanishes continuously when the
moment is isotropic, avoiding an unstable eigenvector choice at repeated
eigenvalues.

Applied directly to reciprocal chart values, it preserves clean structure
well: clean Cameraman and tapered-hair edge retention are about `.869` and
`.866`.  It does not denoise mixed observations sufficiently; mixed Cameraman
MSE remains `.03868` and Gaussian null MSE `.5866`.  The eigenvector explains a
transported relation but does not prove that its amplitude is true.  Therefore
eigenvectors may define support coordinates, while eigenvalue volume carries
uncertainty; direct principal projection is rejected as a terminal estimator.

## Phase susceptibility and the retention audit

The normalized Rayleigh coordinate has two ordered endpoints, not one.  With

\[
r(g)=\frac{g^TLg}{d_{max}g^Tg}\in[0,2],\qquad
d(r)=\min(r,2-r),
\]

`r=0` is smooth order, `r=2` is alternating order, and `r=1` is maximum
refusal by the conservative Markov step.  The parameter-free susceptibility

\[
\chi(r)=4d(r)(1-d(r))
\]

lets either ordered endpoint stand on its own and consults transported signed
phase only through the transition.  This phase-susceptibility readout is a new
Pareto point rather than a final posterior.  Against the old discrete
confidence on the 32-square mixed battery it changes lost/retained first-jet
action as follows:

| scene | old truth lost | phase truth lost | old noise kept | phase noise kept |
|---|---:|---:|---:|---:|
| Cameraman | .234 | .200 | .106 | .136 |
| tapered hair | .298 | .234 | .100 | .124 |
| woven chirps | .497 | .364 | .108 | .124 |

Thus it retains substantially more hair and woven structure for a modest rise
in retained noise action.  Curvature remains the failure: phase susceptibility
still loses `.472`, `.454`, and `.481` of true curvature action respectively.

The audit now decomposes every scalar readout exactly into retained truth,
lost truth, retained corruption, and their MSE cross term.  It also partitions
phase susceptibility and the Selling-jet witness pointwise into common
support, phase-only support, jet-only support, and common refusal.  On all
three mixed scenes, phase-only first-jet evidence has positive oracle
truth-minus-noise separation (`.090`, `.099`, `.063`).  Selling-jet-only
evidence has negative separation.  Therefore the conservative jet law must
not veto signed phase.

Most importantly, common refusal contains about `68--71%` of true curvature
action and `73--76%` of corruption curvature action.  The present observer
cannot distinguish those two.  Calling common refusal noise and smoothing it
would erase real scene structure.  It must remain an exact unresolved
residual until another observation supplies genuinely new evidence.

### Differential and Krylov lifts: useful rejections

A positive normalized Selling-jet Gram law was tested:

\[
J(a,b)=ab+\Gamma_{L/d}(a,b)+(L a/d)(L b/d).
\]

It combines value, Selling-flux, and curvature agreement before positive
transport, so its Hadamard pull is rigorously bounded without thresholds.  It
rejects noise strongly, but it is not positive support: on mixed woven chirps
it lowers retained noise first-jet action from `.108` to `.068` while raising
lost true first-jet action from `.497` to `.605`.  It is retained as a negative
uncertainty witness, not an estimator.

A transported Krylov Gram law of `(g,Pg,P^2g)` was also tested.  Applying it to
one heat increment is invalid because the numerical trace has already made
that increment spectrally localized; the witness certifies its own hidden
band.  Applying it to the complete unresolved field is still too permissive:
deterministic local transport makes mixed noise look low-rank and retains
about `.79--.81` of its action.  Local Krylov collinearity is therefore
rejected as evidence of truth.

The sound spectral diagnostic is instead the full-field generator measure.
For `Q=L/d`, its mean and normalized Bhatia--Davis dispersion are

\[
\mu=\frac{\langle g,Qg\rangle}{\langle g,g\rangle},\qquad
\delta=\frac{\langle Qg,Qg\rangle/\langle g,g\rangle-\mu^2}
{\mu(2-\mu)}.
\]

Mixed corruption raises unresolved `mu` dramatically (roughly `.55--.88`,
versus `.07--.64` on the clean controls), directly measuring how strongly the
observation fights transport.  Turning `mu` into a shrinkage weight damages
detail, so it belongs to the evolving noise/transport-uncertainty law rather
than to structural support.

## Required next equation

Confidence must live on the continuous scale orbit itself.  The candidate
scene should have the form

\[
x(\tau)=\bar y+\int_\tau^\infty c(t,x)g(t)\,dt,
\]

where `c` is a transported scale-density state, not a weight updated once per
stored increment.  Its law must:

1. use reciprocal phase as independent source measure;
2. use accepted-scene anisotropy only as transport geometry;
3. retain isotropic action as unresolved until it has scale persistence;
4. conserve confidence under refinement of `dt` or `d(log t)`;
5. make the final integral invariant to the quadrature trace;
6. leave all fine action untouched until the full decomposition and its
   confidence measure exist.

Only after this scale-continuity gate passes should the posterior be compared
against the phase-union observer, FMMT after-pass, or a native implementation.
