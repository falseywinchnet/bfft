# V3 deep dive and the band-free denoising replacement

## Result of the audit

The useful V3 inheritance is not a collection of edge rules. It is this chain:

\[
\text{transported information}
\longrightarrow Q
\longrightarrow \rho=\frac{\sqrt{\det Q}}{\pi}
\longrightarrow M
\longrightarrow \text{causal first arrival}
\longrightarrow \text{support lineage}.
\]

V3 does not begin with a requested segment count and then search for good
boundaries. A measured precision tensor supplies a Riemannian volume form. That
volume emits the population. A shared eikonal action then determines which
source reaches each point first. Connected support is an output of transport.

That principle should replace FMMT's provisional chart, fixed anchor lattice,
support certification bands, travel radius, packet windows, and posterior
inertia. The literal V3 implementation cannot simply be imported: several V3
stages are historical reconstruction controls, and its local support tensor
mistakes untransported noise for fine support.

## What V3 actually does

### 1. A frozen observation becomes a precision field

The canonical substrate freezes one Meyer cartoon/texture decomposition of the
unchanged image. Event energy `E` and a structure tensor `J` produce an
amplitude-normalized precision,

\[
Q_0(x)=r(x)\frac{J(x)}{E(x)+\epsilon E_*}.
\]

This matters more than the particular Meyer construction. `Q` is interpreted
geometrically: its eigenvectors are support directions and its eigenvalues are
reciprocal squared support lengths. Contrast is not itself population.

The existing implementation still contains selected scales, percentiles,
weights, reliability gates, and a finite isotropic horizon. Those numbers are
not part of the transferable theory.

### 2. The volume form commands population

The support ellipse `xi' Q xi <= 1` has area

\[
A_Q=\frac{\pi}{\sqrt{\det Q}}.
\]

Therefore the reciprocal area

\[
\rho_Q(x)=\frac{\sqrt{\det Q(x)}}{\pi}
\]

is a local population measure. V3 quantizes this complete measure
simultaneously. There is no candidate ranking, top-k selection, birth score,
or target cell count. A safety ceiling is memory policy, not image theory.

The curvature correction is also geometric rather than classificatory. A
rank-one tangent with predicted semi-spans `(a,b)` remains valid only while its
sagitta `kappa a^2/2` stays within `b`. It changes population because the local
straight chart ceased to cover its promised area.

### 3. Eikonal action extracts connected ownership

For a positive metric `M`, V3 solves

\[
d_M(s,x)=\inf_\gamma\int_0^1
\sqrt{\dot\gamma^\top M(\gamma)\dot\gamma}\,dt,
\]

and assigns `x` to the first source reaching it. Equivalently, away from
sources the arrival field obeys

\[
\nabla T^\top M^{-1}\nabla T=1.
\]

The continuous solver reduces the local metric lattice, constructs causal
simplices, and uses an analytic Hopf--Lax minimizer over each opposing simplex
edge. The winning direction is selected by local action rather than an
eight-direction crystalline menu. Every accepted point retains its arrival
covector, causal parent or parent pair, barycentric fraction, and acceptance
order.

This causal forest is more important to denoising than the final integer cell
labels. It is a lossless statement of how support arrived and therefore a
natural graph on which signal/residual measure can move in both directions.

### 4. Support mass can be transported backward

V3's characteristic correction reverses population mass through the accepted
Hopf--Lax DAG. Barycentric parent fractions conserve that mass. A proposed site
movement survives only if every germ remains represented and exact total
arrival action decreases.

The intrinsic stopping statement is therefore not “perform N passes.” It is
“no topology-preserving displacement lowers the measured causal action.” The
current trust fraction, core radius, and one-pass cap are implementation
controls around that statement.

## Which parts are theory and which are retained settings

| V3 mechanism | Status for fused denoising |
|---|---|
| precision as inverse support geometry | retain |
| determinant volume as population | retain |
| simultaneous measure quantization | retain as numerical representation |
| continuous Hopf--Lax first arrival | retain |
| causal parents, covectors, and reverse mass transport | retain |
| exact action descent and germ survival | retain |
| Meyer lambda/mu/sweep count | do not inherit as denoising law |
| half-scale cartoon scaffold | reject; selected physical scale |
| owner boundary band, radius, sweep count, strength | reject |
| cartoon refit blend `0.5` | reject |
| texture support weight and phase shift | reject as physical decisions |
| split error ratios and minimum pixel count | reject |
| merge penalty and merge rounds | reject |
| interface confidence/error bands | reject |
| straight/eikonal coordinate choice and eikonal sweep count | reconstruction controls, not support theory |
| safety cells, heap workspace, quadrature counts | numerical policy only |

The `canonical_v2` V3 branch is closer to the transport principle than the
default half-cartoon branch, but it still carries support weights, metric gains,
characteristic radii, and reconstruction model tests. We take the geometric
lineage, not the defaults.

## Why direct V3 support fails on a noisy observation

V3 is a segmenter/representation. Isotropic texture is allowed to command
population. Independent replacement noise also creates isotropic local tensor
energy. A local determinant cannot know whether that variation will predict
another observation.

On the 128-square tapered-hair control, before any denoising:

| local V3 measure | clean cells | mixed replacement + uniform | ratio |
|---|---:|---:|---:|
| half-cartoon structural | 42.2 | 682.0 | 16.2x |
| full texture-bearing | 66.4 | 830.0 | 12.5x |

Uniform additive corruption likewise raises the structural measure to about
196 cells and the texture-bearing measure to about 748 cells. The precise M4
record is `v3_support_under_corruption.json`.

This is the decisive rework: denoising support must be the information volume
of a *transported predictive law*. It cannot be the structure tensor of raw
observed atoms, even when that tensor came from an elegant segmenter.

## Diagnosis of current FMMT

FMMT has one genuinely correct instinct: signal proposals and residual laws
travel on the same fronts and meet through the observation equation. But the
state is not actually fused:

1. a separable empirical smoother creates `x0`;
2. support rules edit `x0` before it becomes geometry;
3. a residual quantile creates a scale field;
4. fixed anchors and a tuned metric create graph fronts;
5. fixed packet windows create separate signal and residual histograms;
6. a travel temperature and cutoff decide reach;
7. a likelihood floor repairs empty bins;
8. an entropy ramp blends the posterior back with `x0`.

The transport is downstream of nearly every physical decision. Consequently
the method cannot restore structure erased by the provisional chart, and every
setting can compensate for a missing state variable.

## The fused state

Let the observation be `y(x)`. The denoiser state is one measure

\[
\mu_x(dz\,dr\,dj)
\]

over latent value `z`, residual `r`, and transported jet `j`. It is supported
on the exact observation graph

\[
y(x)=z+r.
\]

There is no independent signal histogram and noise histogram to reconcile
later. Their marginals are projections of this same state:

\[
p_x(dz\,dj)=\int\mu_x(dz\,dr\,dj),\qquad
\nu_x(dr)=\int\mu_x(dz\,dr\,dj).
\]

A named noise family never appears. Residual behavior is whatever marginal is
required by the transported joint measure.

## Conservative jet transport

Along a spatial characteristic with velocity `v`, a first jet predicts latent
value by `dot z = j dot v`. Higher jet state is parallel transported by the
connection induced by the current information geometry. Residual atoms are
carried as residual atoms; they are not spread into `z`. In schematic
Liouville form,

\[
\partial_t\mu
+\operatorname{div}_x(v\mu)
+\partial_z((j\cdot v)\mu)
+\operatorname{div}_j((\nabla_vj)\mu)=0,
\qquad \dot r=0.
\]

This is the mathematical version of the image observation: replacement noise
does not diffuse the structure underneath it. Signal hypotheses move along
predictive jets; residual mass remains a coordinate of the same conserved
particle.

Observation coupling is an exact measure disintegration onto `y=z+r`, not a
positive likelihood floor. Cross-prediction is obtained by excluding the local
observation atom from the incoming causal measure, not by constructing parity
support bands.

## Noise and structure are coordinates, not classes

Gaussian, uniform, impulse, replacement, and mixed corruption are useful names
for producing and diagnosing controls. They are not states of nature and do
not appear in the solver. Neither does a binary structure mask. Every atom has
a continuous latent coordinate `z`, residual coordinate `r`, jet `j`, mass,
and causal action, with the exact constraint `y=z+r`.

The optimal disintegration of that mass is the distinction. Mass whose value
and jet can be carried coherently has low signal transport action. Mass whose
residual coordinate becomes common under parallel transport has low residual
rearrangement action. Between those limits lies a continuum; there is no
threshold at which noise becomes structure. Correlated or patterned
corruption is not forbidden from acquiring geometry, and fine structure is
not required to look smooth. The joint action decides which coordinate can
carry each part most economically.

This is the defensible remnant of FABADA's idealism. Its iteration-depth
mixture was a Cesaro average of reused diffusions, not Bayesian evidence. The
new system does not select or average smoothing times. Continuation is
accumulated causal action of the joint measure, and the entire path evolves
until no admissible mass-conserving transport lowers that action.

## Support is a horizontal transport pullback

Let `p_x` be the signal/jet marginal after causal transport and observation
projection. An ordinary derivative of `p_x` is not geometric enough. For an
affine signal, the value distribution translates from point to point, but its
jet predicts that translation exactly. Counting the untranslated change would
turn a single affine relation into artificial population.

Let `P_{x->m}` denote parallel characteristic transport from base point `x` to
a common midpoint `m`, including `dot z = j dot v` and jet parallel transport.
For a tangent direction `v`, define the horizontal quadratic-Wasserstein
pullback by

\[
W_x(v,v)=\lim_{h\to0}\frac{1}{h^2}
W_{2,S}^2\!\left(
 (P_{x\to m})_\#p_x,
 (P_{x+hv\to m})_\#p_{x+hv}
\right),
\qquad m=x+\tfrac h2v,
\]

where the ground cost is the Sasaki metric on the value/jet bundle. Polarizing
this quadratic form gives `W_ab`. This is the transport-corrected information
tensor:

- a spatially invariant law has `W=0`;
- an affine law with its correct jet also has `W=0` after transport;
- changing curvature, competing continuations, and genuine interfaces create
  horizontal transport cost; and
- an exchangeable residual amplitude does not create support solely because
  one finite sample happened to be large.

For a smooth positive density, a horizontal Fisher--Rao pullback remains a
valid continuum alternative,

\[
F_{ab}=4\int D_a^\nabla\sqrt{p_x}\,D_b^\nabla\sqrt{p_x},
\]

but the empirical state is atomic. Ordinary histogram Fisher geometry becomes
singular under bin refinement. In one scalar value coordinate, quadratic
Wasserstein distance has the exact quantile representation

\[
W_{ab}(x)=\int_0^1
D_a^\nabla q_x(u)\,D_b^\nabla q_x(u)\,du,
\]

which needs neither bins nor a kernel bandwidth. The joint `(z,j)` solver uses
the corresponding optimal particle coupling under the Sasaki cost.

Let `G_Omega` be the constant Euclidean domain metric normalized so that its
volume over the whole image is exactly one support unit. Define

\[
Q[\mu]=G_\Omega+W[p].
\]

The base term is topological: a completely uninformative connected image still
has one support component. It replaces FMMT's scale clips and V3's selected
maximum support length.

Population and travel now separate canonically:

\[
\rho[\mu](x)=\frac{\sqrt{\det Q[\mu](x)}}{\pi},
\qquad
M[\mu](x)=\frac{Q[\mu](x)}{\sqrt{\det Q[\mu](x)}}.
\]

In two dimensions `det M = 1`. Absolute information volume decides how many
support sources exist; normalized anisotropy decides how they travel. There is
no metric-strength control and no opportunity to count the same evidence twice.

The eikonal system is self-consistent:

\[
\nabla T_i^\top M[\mu]^{-1}\nabla T_i=1,
\qquad
C_i=\{x:T_i(x)=\min_jT_j(x)\},
\]

while the source population is the quantization of `rho[mu]`. The resulting
causal forest transports the next joint measure. Support is therefore a fixed
point of predictive measure transport, information volume, population, and
arrival—not a preliminary mask.

## One variational object, not alternating tuned modules

The intended continuum solution is the minimum kinetic action of the joint
measure on the jet bundle, subject to:

1. mass conservation;
2. exact observation coupling `y=z+r`;
3. causal eikonal transport under `M[mu]`;
4. population volume `rho[mu]`;
5. parallel residual exchangeability in transported coordinates.

The jet bundle receives the Sasaki metric induced by `Q`; this gives value,
direction, and curvature transport common units rather than a tuned blend of
smoothness penalties. The numerical fixed-point iteration is accepted only
when this single causal action decreases and all represented mass survives.
Equilibrium—not a sweep count, denoising duration, or support threshold—is the
physical stopping condition.

Existence, uniqueness, and a practical action-decreasing discretization remain
open work. They should be proven or falsified before this replaces FMMT.

## Positive lineage is the missing coordinate

The latest one-dimensional gate makes target exclusion precise.  Deleting the
target and every validation index whose predictor reads it produces an exactly
J-invariant interior score, yet performs worse than the leaky control.  The
deletion destroyed effective population without replacing it by causal source
identity.  Conversely, carrying covariance on all lag/path branches preserves
clean oscillatory structure extremely well but amplifies corruption because
those branches have no transported ancestry.

The next state is therefore not merely `mu_x(dz dr dj)`.  It is a positive
measure on the characteristic bundle,

\[
\eta_t(dx\,db\,dz\,dr\,dj\,da),
\]

where `b` is continuous ray/scale identity and `a` is source-lineage mass.
Lineage is not a confidence scalar.  It is pushed through each accepted
Hopf--Lax parent simplex by its barycentric parent fractions and is conserved
exactly.  Its lifted continuity equation has the schematic form

\[
\partial_t\eta
+\operatorname{div}_x(v\eta)
+\operatorname{div}_b(\dot b\,\eta)
+\partial_z((j\!\cdot\!v)\eta)
+\operatorname{div}_j((\nabla_vj)\eta)=0,
\qquad \dot r=0,
\]

on the exact graph `y=z+r`.  The connection term `dot b` carries a branch
along its own characteristic; it is the piece missing from spatial branch
averaging and from global per-index covariance.

At a target, let `pi_x` be the arrival marginal in predicted value after
parallel transport to a common base point.  Its parameter-free interaction
potential is

\[
U_x(q)=\int |q-q'|\,d\pi_x(q').
\]

This is the empirical W1 population action, not a kernel density: it has no
bandwidth.  The line gate shows that adding `U_x` removes isolated zero-action
affine monopolies.  It does not by itself choose the denoised value; that
requires first arrival in the lifted Hamilton--Jacobi system

\[
H\!\left(x,b,\nabla_xT,\partial_bT;\eta\right)=1.
\]

First arrival selects a continuous section `b_*(x)` together with its positive
ancestry law.  Only then is the amplitude marginal reduced.  Thus population,
support, and residual are still projections of one transport state, while
neither a maximum branch nor a premature mean/median becomes physical law.

The remaining analytical task is to derive the Sasaki ground metric that puts
base motion, branch motion, jet defect, and residual displacement in common
units from `Q[eta]` itself.  Introducing coefficients for those terms would
only recreate the settings being removed.  Until that metric and an
action-decreasing discretization survive refinement, this is a research
equation rather than a promoted denoiser.

The first line-bundle construction now supplies that metric empirically rather
than by coefficients.  At each adjacent midpoint, the two transported branch
laws define a covariance `Sigma_x` on `(z,j,r)`.  The precision

\[
G_x=\frac{\Sigma_x^{-1}}
{\det(\Sigma_x^{-1})^{1/3}}
\]

has determinant one and therefore changes only relative travel geometry, not
transport strength.  Reciprocal Sasaki distance under `G_x` carries positive
branch mass forward and backward before a W1 readout.  On the complete
three-seed line gate this improves value and both derivative errors over the
untransported W1 law while slightly increasing clean total-variation
retention.  Signal-only and unwhitened direct-sum metrics form the two rejected
extremes.

This is evidence that the fused covariance metric is the right coordinate
system.  It is not yet the final Hamilton--Jacobi solver: bidirectional message
normalization forces a dense branch coupling, and the current branch atoms do
not retain distinct Hopf--Lax root identities.  Those are the precise tasks of
the 2-D causal lift.

## Executable kernels and the first rejected seed

`fused_transport_geometry.py` implements the band-free part that is already
well-defined:

- exact joint observation lifting on `y=z+r`;
- Fisher pullback for smooth predictive densities;
- bin-free scalar Wasserstein pullback for equal-mass predictive particles;
- the one-support-unit domain base metric;
- determinant population volume; and
- determinant-one eikonal anisotropy.

Its invariants establish:

- every spatially invariant predictive distribution has exactly one support
  unit, regardless of its amplitude distribution;
- the observation lift conserves probability and satisfies `y=z+r` atomwise;
- particle ordering and uniform particle replication do not change the
  Wasserstein geometry;
- an untransported translating distribution creates oriented cost, exposing
  exactly why the final derivative must be horizontal; and
- raw random one-hot atoms generate enormous false information, while their
  transported spatially invariant law returns to one support unit.

The first leave-one-out image seed used twelve immediate affine predictions per
pixel and the exact scalar quantile pullback. It removed histogram resolution
entirely, but failed the hair control: clean support was `3.62`, mixed
replacement-plus-uniform support was `50.55` (`13.96x`), and uniform-additive
support was `12.01` (`3.32x`). The exact record is
`predictive_seed_under_corruption.json`.

That seed is rejected. Local cross-prediction alone does not create a
predictive law; finite-sample disagreement is still being differentiated. The
joint measure must first undergo conservative, local-observation-excluded jet
transport, and only then may its horizontal Wasserstein volume command V3
population.

## Replacement map for FMMT

| Current FMMT object | Fused replacement |
|---|---|
| provisional histogram smoother `x0` | no scalar pilot; initialize joint observation measure |
| support-density/certification bands | horizontal predictive transport volume `rho[mu]` |
| residual quantile scale | residual marginal of `mu` |
| regular anchor stride | quantized information-volume population |
| eight-neighbor graph | V3 continuous reduced-simplex eikonal metric |
| metric strength/cap/exponent | determinant-one `M[mu]` |
| travel temperature and cutoff | causal support/collision action |
| separate signal/noise packets | one joint `(z,r,j)` measure |
| likelihood floor | exact observation-graph projection |
| entropy inertia blend | requested barycentric projection of final `mu` |
| fixed continuation rounds | causal-action equilibrium |

The existing accelerated FMMT remains a matched engineering control. It should
not receive more support heuristics.

## No-settings covenant

Permitted numerical controls must converge to the same continuum object:

- spatial grid resolution;
- value/jet quadrature resolution;
- population quantization phase;
- heap and memory ceilings;
- floating-point convergence tolerance.

The following are disallowed as promoted physical controls:

- support thresholds or bands;
- chosen image scales or windows;
- anchor spacing;
- metric strength;
- packet radius;
- propagation temperature or cutoff;
- smoothing time;
- fixed iteration count;
- posterior blend or confidence clip;
- truth-selected corruption regimes.

A numerical knob is legitimate only after refinement demonstrates that changing
it does not change the inferred continuum state.

## Next executable experiment

The minimal four-tangent lift in `2D_CHARACTERISTIC_LIFT_RESULT.md` is the
latest falsification boundary. Its success on distributed oscillatory
relations and failure on sparse interfaces show that horizontal Wasserstein
geometry alone is not enough when effective population has no causal identity.
The eikonal parent fractions are therefore part of the statistical state, not
merely solver bookkeeping.

`causal_ancestry.py` is the first executable piece of this state. For an
accepted simplex with stored barycentric fraction \(t\), it pushes the complete
source law by \(A_x=(1-t)A_p+tA_q\) and derives collision population from
\(1/\lVert A_x\rVert_2^2\). Retaining the complete law is essential: two parent
paths may already share roots, so scalar “effective counts” cannot be composed.
The existing V3 equal-owner restriction makes this law one-hot by construction.
The denoising march must therefore occur before hard ownership, with a shared
transport label and distinct observation-excluded root identities.

`CAUSAL_PREDICTIVE_SIMMER.md` executes that shared-label march. It rejects the
checkerboard population and its value barycenter, but retains two theoretical
advances: the ancestry-weighted predictive law converges under quantile
refinement, and quotienting its common scalar translation reduces false
corruption population materially. Determinant-normalized self-consistency
descent then exposes a nonzero residual shelf caused by discrete parent switches.
The next state must therefore emit a continuous source measure before any
raster population quantization; adding remarches or damping the parity lattice
is not a valid repair.

1. Represent `mu` by particles carrying `(z,r,j)` and mass.
2. Use the existing continuous eikonal solver and its causal parent fractions.
3. Reverse-transport predictive particles with local-observation exclusion.
4. Couple neighboring laws only after parallel jet transport to their common
   midpoint; compute the horizontal Wasserstein pullback.
5. Recompute `Q`, `rho`, and `M` from that transported predictive marginal.
6. Re-emit/remarch only when total causal action decreases and all mass survives.
7. Test constant, affine, jump, chirp, damped ripple, and mixed replacement +
   uniform controls simultaneously.
8. Require clean near-identity, corruption reduction, jet survival, population
   refinement, and action monotonicity before adding any GUI path.

No C++ specialization should begin until this fixed-point state stops changing
under spatial and value-measure refinement.

## First root-resolved causal lift

The earlier proposed lift now exists in `causal_information_lineage_2d.py`.
It does not classify noise, choose support bands, or run for a selected time.
The strictly predictive continuous-tangent measure supplies joint
`(z,j_y,j_x,r)` atoms; the horizontal Wasserstein tensor supplies continuous
population and determinant-one travel; the V3 Hopf--Lax march supplies the
accepted parent DAG; and exact positive root/branch mass is pushed through
that DAG.

For a parent-child pair, both jets predict their values at the same midpoint.
The covariance of the pooled transported atoms is inverted and normalized by
the fourth root of its determinant. This produces a unit-determinant Sasaki
precision on all four joint coordinates without scale weights. A reciprocal
distance kernel moves branch density, while the recorded parent fraction mixes
only causal incoming laws. The state is

\[
\eta_x(a,k)\ge0,\qquad \sum_{a,k}\eta_x(a,k)=1,
\]

where `a` is continuous source identity and `k` is numerical branch
quadrature. Marginalizing `a` recovers the branch law exactly; source collision
population is computed from the transported root marginal rather than a
conductance proxy.

The first refinement evidence is favorable. Doubling projective directions
from four to eight reduces mean population-phase RMS from roughly `0.00959` to
`0.00545`, while the causal collision section still improves MSE, SSIM, and
edge retention together. On the six-structure, ten-condition external gate it
improves local MSE in 55 of 60 cases and local SSIM and edge retention in 56 of
60. Aggregate MSE/SSIM/edge move from `0.006990/0.6780/0.4855` to
`0.006203/0.7000/0.5276`.

This does not close the theory. Integrated FMMT remains substantially ahead in
aggregate quality, the clean geometric-interface case regresses, and density
phase is converging rather than invariant. The root-resolved tensor is also a
dense research representation. The next mathematical task is not a tuned
repair: it is a representation-convergent continuous root measure and a
causal action whose scalar projection is derived uniquely, especially at
sparse interfaces. GUI promotion, repeated denoising time, and C++
specialization remain deferred.

The clean-interface diagnostic further identifies why phase is still visible.
Its continuous information volume is only `1.0836` units before causal
transport and `1.0664` afterward, but local Bernoulli quantization realizes
three hard germs. This is precisely the kind of discrete support decision the
fused theory is meant to remove. The source term of the Hamilton--Jacobi law
must remain a weighted continuous measure—or integrate the complete
quantization phase measure—until first arrival. Integer germ realization is a
numerical quadrature of that source measure, never the physical support state.

The phase-fibre experiments now sharpen this statement. Merely averaging the
root/branch measures over population realization produces a convergent scalar
mean but loses branch variance. Selecting the strongest two-particle collision
section independently in each realization and only then integrating phase
retains both sparse interfaces and oscillatory structure. Thus section
selection and representation marginalization do not commute.

This gives the next continuum order of operations:

\[
\rho[\eta]\longrightarrow
\eta^\theta\longrightarrow
b_*^\theta(x)\text{ by lifted first arrival}\longrightarrow
\int z_{b_*^\theta(x)}\,d\theta.
\]

The current finite probe uses `argmax eta^2/h` in place of lifted first
arrival. It improves aggregate edge and variance fidelity on all twelve
three-structure controls and its dyadic phase differences decrease, but hard
branch switches remain visible. Therefore the result validates ordering, not
the discrete maximum. The final Hamiltonian must include the population fibre
as a representation coordinate and produce a continuous section in the joint
position/branch bundle before that coordinate is integrated out.

### The first lifted Hamilton--Jacobi action

That Hamiltonian now has an executable forward form. Write `h_x` for branch
Haar measure, `m_x` for transported branch mass, and `S=log(m/h)` for its
density action. The determinant-one joint metric supplies a parent-child
Markov kernel `K`; its representation-invariant density is
`kappa=K/h_child`. The stored Hopf--Lax parent fraction then combines max-plus
messages in exactly the same barycentric proportion used by the spatial
characteristic:

\[
S_x(l)=\log L_x(l)+(1-t)\sup_k(S_p(k)+\log\kappa_{px}(k,l))
 +t\sup_k(S_q(k)+\log\kappa_{qx}(k,l)).
\]

Raw quadrature mass is never maximized. The scalar endpoint also avoids a
branch argmax. Two coherent path witnesses induce the collision density
`exp(2S)`, which is normalized against `h`; latent value is its Haar
barycenter. This is the parameter-free path-coherent counterpart of the
ordinary sum-product collision mean. Population phase is integrated after
that endpoint measure is formed.

The first gate is positive but not final. Across twelve clean and corrupted
structure controls the lifted barycenter improves local MSE and edge retention
in every case and SSIM in eleven; it also beats the ordinary causal collision
mean on all three of those metrics in all twelve cases. It is substantially
more stable under phase refinement than the hard HJ section. It still loses
too much variance, so the hard section and smooth barycenter now bracket a
precise missing object: continuous terminal interface concentration in the
joint signal/residual/jet measure. No edge label, noise class, temperature, or
tuned continuation is licensed by this result.

The first six-structure, ten-condition screen agrees: 57 of 60 local MSE and
SSIM comparisons improve, as do 50 edge comparisons. The law remains below
FMMT in aggregate and the residual failures concentrate on tapered hair and a
replacement-corrupted geometric interface. The screen's corruption names are
generator labels outside the state equation. Its four population phases make
it a broad falsification pass, not a resolved continuum result.

### Collision multiplicity is causal dimension

The fixed two-path collision has now been replaced by a quantity already
present in the eikonal solve. If a child has parent fraction `t`, the inverse
Simpson/Hill count `1/((1-t)^2+t^2)` is the continuous number of represented
parent witnesses. Adding the local likelihood witness gives collision order
`alpha=1+1/((1-t)^2+t^2)`, with the root and one-parent cases reducing
exactly to orders one and two. The endpoint law is
`exp(alpha S) h`, followed by its Haar barycenter.

This is the desired relation between support and denoising: sharper scalar
concentration is earned by the causal dimensionality of transport, not by an
edge detector. Across the full 60-case screen it improves the fixed-order HJ
endpoint on MSE/SSIM/edge/variance in `51/50/58/56` cases. It also repairs the
clean tapered-hair regression, leaving only its replacement-corrupted form and
the heavily replaced geometric interface below their local laws. Spatial,
branch, and population refinement remain necessary before promotion.

## Branch transport replaces cell transplantation

The literal transfer of V3 germ cells has now been tested and rejected for the
denoising state.  A nominal `1.29` clean support units can realize three raster
germs under one low-discrepancy phase; one cell then transports while the
others veto it.  Local pixelwise authority has the opposite failure: it obeys
its convex energy theorem at every target but moves `72.5%` of the clean
geometric field immediately and is still unresolved after 128 steps.  A
Selling graph-Wasserstein flow also descends the stated joint energy exactly
while manufacturing enormous jet volume.  These results are
`causal_support_joint_geometric_mixed.json`,
`local_joint_geometric_mixed.json`, and
`joint_graph_gradient_geometric_mixed_32.json`.

V3's deeper inheritance survives: causal identity must be established before
amplitude is reduced.  In the denoising law the relevant identity is not an
integer spatial cell but a characteristic branch of the joint predictive
measure.  A pointwise maximum-posterior branch already preserves substantially
more Cameraman/hair and woven structure than a W1 barycenter.  Spatially
averaging its probability through the Selling operator is rejected; branch
identity has a characteristic direction and cannot be transported sideways as
a generic label.

The next eikonal object therefore lives on a bundle.  If `b` denotes a
continuous characteristic coordinate (projective direction, physical radius,
and sided affine chart), the desired arrival action has the schematic form

\[
H\!\left(x,b,\nabla_x T,\partial_b T;\mu\right)=1,
\]

where `mu` is the exact joint signal/residual predictive measure.  The base
metric comes from its horizontal Wasserstein geometry; motion in `b` follows
the characteristic connection.  First arrival selects a continuous branch
section `b(x)`, after which amplitude `z_{b(x)}(x)` is read once.  This is the
denoising analogue of V3 ownership without importing a germ count, owner band,
or hard segment boundary.

## Continuous-tangent update

The common-physical-scale off-grid tangent law now converges under nested
projective-circle quadrature and excludes the target observation exactly. Its
horizontal Wasserstein translation quotient supplies the first predictive
geometry whose corruption response is remotely compatible with V3 population:
on the three-structure 20-square gate, 32-quantile support changes by roughly
`1.00x`, `1.13x`, and `1.16x` for uniform, replacement, and mixed corruption
when the strictly held-out signal prior is used. The 16-to-32 quantile support
change averages `1.23%`.

The apparent next step—adding local directional-jet variation as the vertical
Sasaki term—is false. It produces `3.09x`, `9.29x`, and `15.30x` support
inflation. Quotienting repeated particles by exact characteristic source
lineage reduces those factors to `1.64x`, `2.20x`, and `3.84x`, but does not
make the law predictive. Target exclusion is necessary; causal identity is
also necessary.

Therefore the horizontal scalar metric and vertical jet population must not be
formed in one local pass. The scalar transported law can provide the initial
determinant-one travel metric. The Hopf--Lax march must then attach parent
simplexes and reverse lineage to joint particles. Only jets compared after that
parallel transport may contribute vertical information volume. This staged
construction is still one fixed-point state equation; it is not a support mask
followed by a denoiser.

The existing V3 density emitter is not yet acceptable as this bridge. Its
fixed hash phase and `0.8` pixel jitter are harmless engineering choices for a
segmenter, but here population phase must be explicitly refined and shown not
to change the continuum state. On the current small controls the implied
horizontal support is only about one to two atlas units, so a single raster
realization would dominate the experiment.

## The 1-D root is a causal branch, not a fidelity term

The disjoint-shell experiment now gives a one-dimensional version of the same
lesson. A contextual branch can be built from the midpoint and secant of the
opposite samples at `x+-s`, with the shell at `2s` providing an affine-exact
value--jet action. This branch excludes the target observation and uses the
complete `ds/s` scale measure. One-sided extrapolated arrivals are unstable;
context alone is stable but erases curvature.

The observation must therefore survive as a distinct causal root. It cannot
be inserted as a quadratic fidelity coefficient. A root branch inherits the
full contextual path action and then pays the terminal collision edge
`|y-z_s|`. Comparing only that last edge gives root and context unequal causal
depth and causes corrupt observations to dominate the lineage. Restoring the
inherited action corrects that mass defect without a strength parameter.

At the endpoint, the contextual value has two opposite-side parents and the
observed value has one, giving the provisional simplex reference measure

\[
h_x^{(1D)}=\frac23 h_x^{ctx}+\frac13 h_x^{root}.
\]

This improves the heavy replacement and mixed screens but is not the final
law: it deflates clean structure and admits a false-support mode under dense
impulses. Consequently causal source count is necessary but insufficient.
The terminal concentration order must depend continuously on *effective
independent ancestry* after lineage overlap, just as the 2-D order depends on
the Hopf--Lax parent simplex. This is now the common theoretical problem in
one and two dimensions.

### Root membership is an energy-distance invariant

The root/context experiments now separate two questions that were previously
confounded. Whether the observed root belongs to the contextual measure can be
measured without a noise model. For contextual latent law `mu_x`, the ratio

\[
c_x=\frac{\mathbb E|Z-Z'|}{2\mathbb E|y(x)-Z|}
\]

is bounded by one and is exactly the complement of normalized scalar energy
distance. The transported density contributes a second witness through its
order-two collision concentration `k_x=1-1/K_x`, where
`K_x=integral (dm_x/dh_x)^2 dh_x`. Adding their log odds produces a continuous
root authority with no threshold or fidelity coefficient.

This invariant rejects dense impulses and improves derivative transport, but
it does not solve 1-D denoising by itself. The stable target-free midpoint law
does not contain all clean curvature, so neither amplitude barycenters nor a
Monge rank section can recover it without deflation. Conversely, inserting
Richardson curvature locally introduces negative source weights and amplifies
corruption. Therefore curvature must become a transported vertical coordinate
of the causal bundle. It may influence terminal support only after parallel
transport and ancestry collision; it may not be reconstructed independently
at each target. This is the exact 1-D analogue of delaying jet comparison until
after the 2-D Hopf--Lax lineage is known.

### Two phases, only one of them physical signal state

Population phase in the 2-D atlas is the origin offset of a numerical
quadrature. It is a gauge: it must be integrated and shown convergent. Signal
phase is instead the local orbit coordinate carried by the transported
value--jet pair. It remains physical after the population quadrature is
removed. A mixed signal can have several locally coherent phase trajectories
at once; no Fourier band or named component is allowed to own them.

The first exact 1-D phase section confirms both the value and the limitation of
this coordinate. A determinant-one Sasaki metric on midpoint `(value, jet)`
states sharply lowers derivative corruption and rejects dense impulses. Its
clean particle clouds are much more anisotropic than corrupted clouds. But
projecting that phase law by a quadratic barycenter contracts amplitude, and
using pointwise phase membership to restore the observation admits diffuse
noise. Therefore the fused state equation must transport phase coherence in
the Hamilton--Jacobi arrival law and let it determine continuous collision
order before amplitude marginalization. Phase is an ownership coordinate,
not a post-denoising weight.

Nor can coherence be replaced by balanced forward/backward marginals. The
exact inverse-participation count of those two arrival densities gives maximum
collision order to two equally diffuse, contradictory histories. Phase
agreement is a property of their transported coupling. The 1-D parent object
must therefore remain a measure on paired left/right ancestries until the
terminal simplex is formed, directly paralleling the retained parent identity
in the V3 Hopf--Lax construction.
