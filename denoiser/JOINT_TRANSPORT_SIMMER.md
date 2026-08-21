# Joint signal/residual transport simmer

## The advance

The current image experiment now has an executable joint measure rather than
a smoother followed by a residual repair. At every target `x`, a dense set of
first-jet signal proposals `z_i(x)` is constructed without reading `y(x)`. A
residual coordinate is then imposed exactly,

\[
r_i(x)=y(x)-z_i(x), \qquad y(x)=z_i(x)+r_i(x).
\]

Signal and residual are therefore coordinates of one particle, not two models
stapled together. Uniform, replacement, and mixed corruption exist only in
the benchmark generator. They do not select a solver branch.

The implementation is `witnessed_characteristic_transport_2d.py`. It remains
a research representation and is not exposed in the GUI.

## Strict witness

For each primitive tangent `d`, the witness uses odd-lag source points and
nonzero even-lag validators. A four-colour lattice coordinate separates every
odd source from its target. No reflected value is invented at a boundary; an
invalid characteristic carries zero mass.

This gives an exact executable invariant: changing a target observation does
not change its own predictive particles or their validation mass. Constants
and affine fields are reproduced to floating-point roundoff.

The strict witness is not itself a good reconstructor. On the 18-case,
32-square gate its median gives MSE `0.01039`, SSIM `0.5195`, and edge retention
`0.3764`, below both the earlier four-direction seed and FMMT. Splitting the
whole reconstruction into parity lanes discards too much geometric evidence.
The witness is retained as independent evidence, not as the image estimate.

## Proper transport action

Let `Q_x` be the strictly cross-fitted witness law. A dense jet proposal `z`
is judged by the continuous ranked probability score

\[
\operatorname{CRPS}(Q_x,z)
=\mathbb E_{Q_x}|Z-z|
-\frac12\mathbb E_{Q_x}|Z-Z'|.
\]

This proper one-dimensional transport score needs no histogram, kernel width,
likelihood floor, acceptance band, or named noise law. Proposal conductance is

\[
g_i^{\mathrm{signal}}(x)=
\frac{1}{s_i\,[V_i(x)+\operatorname{CRPS}(Q_x,z_i)]},
\]

where `V_i` is transported first-jet variation and `ds/s` supplies the complete
scale measure. The first attempted product of variation and CRPS had the wrong
physical units and is preserved in
`crossfit_characteristic_transport_product_action.json` as a negative control.

## Exact residual disintegration

The residual prior is another empirical transport law, not a fitted density.
Each source contributes its valid odd-characteristic residual population.
Sources are pooled only inside the same four-colour lane, and the target's
complete source law is removed analytically. For candidate residual `r`, the
leave-one-source-out residual CRPS is evaluated exactly from sorted measures.

The final joint action is

\[
A_i(x)=V_i(x)
+\operatorname{CRPS}(Q_x,z_i)
+\operatorname{CRPS}(N_{-x},y(x)-z_i),
\qquad
w_i(x)\propto\frac{1}{s_i A_i(x)}.
\]

Only the last term couples the posterior to `y(x)`. The signal proposal and
residual prior exclude that identity. The code tests this distinction.

## First gate

`crossfit_characteristic_transport.json` contains six structures under
uniform, replacement, and mixed corruption, one seed each, at 32 square. The
joint W1 barycenter is the strongest new readout.

| Method | MSE | SSIM | Variance | Range | Edge |
|---|---:|---:|---:|---:|---:|
| Joint median | **0.007304** | 0.6656 | 0.6993 | 0.8310 | 0.4358 |
| Earlier four-direction seed | 0.008132 | 0.6161 | 0.6931 | 0.8469 | 0.4409 |
| Integrated FMMT | 0.007527 | **0.6981** | **0.7484** | **0.8666** | **0.5320** |

The aggregate MSE improvement is small but real. More important is its source:

| Corruption | Joint median MSE / SSIM | FMMT MSE / SSIM |
|---|---:|---:|
| Uniform additive | 0.005576 / 0.6732 | **0.003980 / 0.7160** |
| Random replacement | **0.007484** / 0.7440 | 0.009048 / **0.7545** |
| Mixed replacement + uniform | **0.008853** / 0.5797 | 0.009554 / **0.6239** |

The joint law is strongest when observations are deceptive rather than merely
diffuse. That supports the central claim: replacement noise did not erase the
structure; barycentric smoothing erased it. Clean fields still pay MSE
`0.004466`, with variance `0.7693` and edge retention `0.5675`, so this is not
yet the desired continuous reconstruction.

## What continuation taught us

Applying the same joint law recursively proves missing detail remains
transportable. Global covariance raises edge retention from `0.4358` to
`0.5709` and clean edge retention to `0.8298`. It also restores replacement
residual: aggregate MSE becomes `0.008147` and SSIM falls to `0.5260`. The full
record is `joint_characteristic_continuation.json`.

Two local uncertainty laws bracket the missing equation:

1. Full posterior spread `sum w_i (z_i-q)^2` accepts no continuation.
2. Particle collision variance `sum w_i^2 (z_i-q)^2` treats correlated
   direction/scale views as independent. It accepts 4.83 steps on average,
   raises edge retention to `0.5747`, but lowers SSIM to `0.5406`. Replacement
   cases accept 8--11 steps. The record is
   `joint_authority_collision_variance.json`.

Source ancestry now aggregates all particle coefficients by actual observation
identity before uncertainty is squared. Its rows sum to one and have exactly
zero target diagonal. This is the right conserved object, but independent
scalar source errors still leave continuation too persistent. The broad run
was deliberately terminated rather than dignifying failure to equilibrate as
a quality setting.

## Next equation

The remaining uncertainty is covariance on the causal source measure, not a
scalar variance attached to particles or pixels. The next experiment should:

1. transport source error covariance through the determinant-one Selling/
   eikonal operator;
2. quotient repeated affine views by common causal ancestry;
3. evaluate positive predictive energy on that transported covariance;
4. retain the exact joint observation disintegration; and
5. stop only when no covariance-aware mass transport lowers action.

This is where V3 re-enters properly. Its valuable object is the causal parent
simplex and reverse conservative lineage, not its old support settings. The
current dense Python matrices and global sorts are theorem probes. C++ and
representation optimization remain deferred until the source covariance law
survives the image gate.

## Lineage-local covariance result

The next experiment replaced pointwise signal variance with the transported
version of the successful 1-D covariance test. For residual `r`, the signal
prior supplies a strictly held-out prediction `q`. Exact positive source
lineage `B` carries the innovation product to each target:

\[
c_s=r_s q_s^{(-s)},\qquad
\mu_x=\sum_s B_{xs}c_s,\qquad
v_x=\sum_s B_{xs}^2(c_s-\mu_x)^2.
\]

The continuation authority is

\[
a_x=\mathbf 1_{\mu_x>0}
\frac{(\mu_x^2-v_x)_+}{\mu_x^2}.
\]

`B_xx=0` exactly, each row of `B` sums to one, and repeated direction/scale
particles contributing the same observation source are combined before the
finite-population variance is evaluated. The posterior update remains the
joint signal coordinate; only its authority comes from this prior-only law.

`lineage_covariance_full.json` is the complete 24-run, 32-square record. Every
clean and corrupted case reaches intrinsic covariance equilibrium below the
32-step numerical ceiling.

| Method | MSE | SSIM | Variance | Range | Edge |
|---|---:|---:|---:|---:|---:|
| Joint median | 0.007304 | 0.6656 | 0.6993 | 0.8310 | 0.4358 |
| Lineage covariance | **0.006520** | 0.6695 | 0.7368 | 0.8525 | 0.4759 |
| Integrated FMMT | 0.007527 | **0.6981** | **0.7484** | **0.8666** | **0.5320** |

The continuation improves every aggregate measure and enlarges the MSE lead
over FMMT. By corruption:

| Corruption | Lineage covariance MSE / SSIM | FMMT MSE / SSIM |
|---|---:|---:|
| Uniform additive | 0.004353 / 0.6889 | **0.003980 / 0.7160** |
| Random replacement | **0.006945** / 0.7337 | 0.009048 / **0.7545** |
| Mixed replacement + uniform | **0.008263** / 0.5860 | 0.009554 / **0.6239** |

The result is not uniformly dominant. Line drawing gains are large and drive
part of the aggregate MSE advantage; Cameraman, tapered hair, and multiscale
blobs also improve. Geometric interfaces and woven chirps expose the remaining
crystalline proposal defect. FMMT still leads aggregate SSIM and edge
retention, so GUI promotion remains deferred.

## Tangent and residual-transport negatives

The four-direction proposal was generalized to every primitive unoriented
lattice direction through nested Farey orders 1, 2, and 3. Exact periodic
Voronoi widths supply angular quadrature mass summing to `pi`. Target exclusion
and affine reproduction survive every order.

The fields do not converge. Even after angular weighting, mean successive RMS
changes are `0.0263` and `0.0256`, while MSE worsens from `0.00506` to `0.00571`
and `0.00672`. A primitive vector of length greater than one first appears at a
coarser radial scale; adding angles therefore changes the radial measure. This
is not a valid continuous tangent-sphere discretization. The matched records
are `angular_convergence_unweighted.json` and
`angular_convergence_weighted.json`.

Replacing the global four-colour residual pool with direct target source
lineage slightly improves woven SSIM but narrows the residual population and
deflates variance. Transporting that lineage with the determinant-one Selling
Markov operator until held-out CRPS stops decreasing recovers only part of the
loss. Both reach intrinsic transport equilibrium and are retained as negative
geometry probes in `lineage_residual_smoke.json` and
`transported_lineage_residual_smoke.json`.

The next geometry must provide common physical radial scales for every tangent,
most likely through continuous off-grid jet transport or the V3 analytic
Hopf--Lax simplex. It must also derive its metric from the strictly predictive
law rather than the current crystalline relation precursor. The accepted
lineage covariance equation should remain fixed while that geometric
representation changes.

## Common-scale continuous tangents

`continuous_tangent_transport_2d.py` now replaces primitive lattice directions
with nested projective-circle quadrature at common integer physical radii.
Off-grid samples use affine cell coordinates; whenever a bilinear cell would
read the target observation, that coefficient is eliminated and the remaining
triangle is solved analytically. Every proposal therefore has exactly zero
target self-coefficient and reproduces affine fields to roundoff.

The broad-secant version converges as angular count doubles. At 4, 8, 16, and
32 directions its mean field changes fall from RMS `0.01346` to `0.00711` to
`0.00428`. This is the first defensible continuous tangent discretization, but
its jet was not actually transported: it re-estimated a secant over twice the
radius at every scale.

The corrected characteristic carries a unit local jet and judges only its
change at the next radius,

\[
j_s^-=y(x-su)-y(x-(s+1)u),\qquad
z_s^-=y(x-su)+s j_s^-,
\]

with the symmetric right characteristic. It remains affine exact and angularly
convergent. The correction removes a favorable smoothing shortcut: on the
four-structure gate the 16-direction raw median changes from MSE `0.00544`,
SSIM `0.6975` to MSE `0.00582`, SSIM `0.6792`. Lineage covariance recovers
some structure but remains below the crystalline control on the geometric and
woven smoke gate. The records are
`parallel_jet_tangent_convergence.json` and
`parallel_jet_continuation_smoke.json`.

An integrable Hodge readout of the posterior jet is decisively rejected. Its
variance ratio is `0.2962` and edge retention `0.0300` on the geometric/woven
gate. The posterior directional derivatives are evidence coordinates, not a
scalar image gradient to integrate directly.

## Predictive information geometry

The common-scale law permits the horizontal Wasserstein experiment that V3
requires. Weighted signal particles are resolved on a common quantile
coordinate, with no bins or bandwidth. The scalar translation quotient

\[
q_x^H(u)=q_x(u)-\int_0^1q_x(v)\,dv
\]

is refinement-stable and radically less corruption-sensitive than raw V3
support. Across tapered hair, geometric interfaces, and woven chirps at 32
quantiles, posterior translation-quotient support changes by `0.994x` under
uniform corruption, `1.140x` under replacement, and `1.139x` under mixed
corruption. The strictly held-out prior gives `1.002x`, `1.129x`, and `1.156x`.
Mean support change from 16 to 32 quantiles is about one percent. Compare raw
V3 texture-bearing support on the larger hair control, which inflated by
`12.5x` under mixed corruption.

This is not yet the full support law. Quotienting arbitrary scalar translation
also removes nonlinear mean motion. Curvature must return through the vertical
jet coordinate of the Sasaki bundle. Two experiments show why that coordinate
cannot be differentiated before causal transport:

| Vertical law at 32 quantiles | Uniform / clean | Replacement / clean | Mixed / clean |
|---|---:|---:|---:|
| raw held-out directional jets | 3.09x | 9.29x | 15.30x |
| jets after characteristic source lineage | 1.64x | 2.20x | 3.84x |

Source lineage removes repeated-view inflation by a large factor, but local
jet noise still manufactures support. These are retained negative controls in
`continuous_tangent_sasaki_geometry.json` and
`continuous_tangent_lineage_jet_geometry.json`.

The conclusion is now sharper than “use an eikonal metric.” The scalar
horizontal law may command a continuous determinant-one travel geometry, but
vertical jet variation may create population only after the joint particles
have acquired Hopf--Lax causal parent identity and have been parallel
transported to a common base point. Raster germ phase, hash jitter, and an
integer cell count cannot be promoted as physical state. The next experiment
must retain the continuous support measure through the causal march and test
population phase only as a convergent numerical representation.

## Differentiate after identity transport: the surviving result

The next round tested the ordering literally.  The old vertical law assigns a
jet to each source first and then transports a distribution of those jets.
The dual construction first forms the transported scalar section

\[
z(x)=\sum_s B_{xs}z_s
\]

and only then prolongs it.  Its vertical Sasaki pullback is the Gram tensor of
the physical Hessian,

\[
V_{ab}(x)=\sum_c
\partial_a\partial_c z(x)\,\partial_b\partial_c z(x).
\]

This annihilates every affine section exactly and retains curvature which
survives source transport.  The implementation is
`predictive_lineage_prolongation_geometry`; its matched record is
`post_lineage_prolongation_full_gate.json`.

On tapered hair, geometric interfaces, and woven chirps, the pre-differentiated
jet law calls uniform, replacement, and mixed corruption `1.67x`, `2.85x`, and
`3.29x` clean vertical support.  Post-lineage prolongation instead reads
`0.82x`, `1.41x`, and `0.93x`.  After the existing strict source smoothing it
falls further to `0.71x`, `0.55x`, and `0.42x`.  This is the executable form of
the visual observation: much of the structure remains inferable under the
corruption, while the smoothing flow removes it.

The transported section itself is not the denoiser.  Its one-pass readout has
variance ratio `0.38` and edge retention `0.18`; a literal lineage-transported
residual improves those to about `0.67` and `0.43` but returns no residual on
the corrupted controls.  Those negatives are recorded in
`post_lineage_readout_gate.json` and `post_lineage_residual_forms.json`.

## Branch identity before amplitude

The posterior median collapses incompatible characteristic branches before
support has selected one.  Reading the maximum-posterior branch, with every
other equation unchanged, raises corrupted edge retention from `0.389` to
`0.544` on the 20-square gate.  At 40 square on Cameraman and tapered hair it
raises edge retention from `0.497` to `0.654` and variance ratio from `0.761`
to `0.886`.  FMMT remains ahead at `0.720` edge retention, and the branch
readout trades some MSE/SSIM for structure.  The matched records are
`post_lineage_branch_forms.json` and `branch_forms_cameraman_hair_40.json`.

Two apparently principled smoothers were then falsified:

- proper-score transport of branch probability through the spatial Selling
  operator worsens aggregate MSE, SSIM, and edge retention;
- strict joint-energy graph-gradient flow descends its action pointwise but
  makes vertical information explode, proving that the pre-differentiated jet
  witness is the wrong cotangent target.

The surviving next state is therefore a characteristic branch section on the
joint space of position and ray/scale identity.  Its Hamilton--Jacobi action
must propagate a branch along its own characteristic, not average branch
indices sideways and not average their amplitudes.  Causal branch identity is
selected before scalar readout; the exact graph `y=z+r` and target-excluded
witness remain unchanged.  This is still a foundational experiment and is not
exposed in the GUI.

## The one-dimensional lineage obstruction

The corresponding line experiment now isolates the same obstruction without
angular quadrature.  The W1 median of the complete `ds/s` characteristic law
substantially improves random replacement, but repeating a scalar median loses
oscillatory phase.  Keeping residual covariance on every lag/path particle has
the dual behavior: clean MSE falls to `2.12e-5` with `0.969` total-variation
retention, while corrupted variation grows and 78 cases fail to reach
equilibrium.

A target-excluded affine collision action was then made coercive with source
path variation.  Sparse accidental collisions still created reciprocal-action
singularities.  The exact Haar-weighted W1 potential of the full arrival
population removes those singularities—aggregate MSE falls from `0.0250` to
`0.00161`—but the independent affine germs still oversmooth.  Exact deletion
of every validation index whose ancestry contains the target is pointwise
J-invariant and also loses (`0.00171` versus `0.00146`).

Thus target exclusion is not an operation on an index window.  It is a
property of transported positive source identity.  The unified 1-D/2-D next
state is a measure on `(x,b,z,r,j,a)`, where `b` is continuous characteristic
identity and `a` is causal ancestry mass.  W1 population regularizes the
arrival law, while a Hamilton--Jacobi flow carries `a`; scalar amplitude is
read only after that flow.  The matched records are
`1d_particle_continuation_gate.json`, `1d_joint_collision_gate.json`, and
`1d_causal_crossfit_gate.json`.

The positive-lineage version of that state is now executable in one dimension.
At every adjacent midpoint it pools the transported `(z,j,r)` arrival law,
inverts its covariance, and determinant-normalizes the resulting precision.
This supplies the previously missing Sasaki metric without a signal/residual
blend. Positive forward and backward messages then transport branch density
relative to the full Haar `ds/s` reference measure.

The full 234-case result is positive: one W1 readout after lineage has MSE
`0.001595`, first-difference error `0.000948`, and second-difference error
`0.002050`, versus `0.001683`, `0.001092`, and `0.002628` for the same local
law before lineage. Clean TV rises from `0.812` to `0.833` without variance
deflation. Reapplying the operator to the residual is rejected because it
restores mixed corruption. So is a collision-conditioned scalar readout until
distinct root identity replaces same-quadrature-index collision.

This establishes the 1-D branch-bundle transport but not yet the desired
Hamilton--Jacobi first-arrival discretization. The next lift must replace the
bidirectional dense message product with continuous causal parent transport in
two spatial dimensions while retaining the determinant-one joint metric.

## Root-resolved causal information transport in two dimensions

That lift is now executable in `causal_information_lineage_2d.py`.  The
continuous-tangent joint law supplies branch atoms

\[
\xi_{xk}=(z_{xk},j^y_{xk},j^x_{xk},r_{xk}),\qquad y_x=z_{xk}+r_{xk},
\]

with the target excluded from every signal proposal.  Neighboring parent and
child atoms are first parallel-transported to their common spatial midpoint.
Their pooled covariance `Sigma_pc` determines the four-dimensional precision

\[
G_{pc}=\frac{\Sigma_{pc}^{-1}}
{\det(\Sigma_{pc}^{-1})^{1/4}},\qquad \det G_{pc}=1.
\]

Thus signal, both jet coordinates, and residual receive common units from the
arrival law itself.  No fitted blend coefficient appears.  Relative to the
child quadrature reference `h`, the reciprocal Sasaki transition is

\[
K_{pc}(k,l)=\frac{h_{cl}/d_{G_{pc}}(\xi_{pk},\xi_{cl})}
{\sum_m h_{cm}/d_{G_{pc}}(\xi_{pk},\xi_{cm})}.
\]

Let `a` denote continuous eikonal root identity.  Each root injects its local
branch law into `eta_x(a,k)`.  For a causal child with the exact stored parent
fraction `t`, positive propagation is

\[
\widetilde\eta_x(a,l)=L_x(l)\left[
(1-t)\sum_k\eta_p(a,k)K_{px}(k,l)
+t\sum_k\eta_q(a,k)K_{qx}(k,l)\right],
\qquad
\eta_x=\frac{\widetilde\eta_x}{\lVert\widetilde\eta_x\rVert_1}.
\]

The observation likelihood `L_x=1/A_x` is the same joint action for every
root. Root identity is therefore transported, not inferred afterward from a
confidence score. The branch marginal is the exact projection
`m_x(k)=sum_a eta_x(a,k)`.

Two collision measures are now distinguished. The quadrature-invariant branch
collision density is `m_x(k)^2/h_x(k)`. The independently arrived component is

\[
c_x^{\ne}(k)=\frac{m_x(k)^2-\sum_a\eta_x(a,k)^2}{h_x(k)}.
\]

It vanishes continuously when only one root is represented. The corresponding
readout returns unresolved same-root probability to the local law rather than
inventing cross-lineage evidence. On the present two-to-three-germ population
this distinct-root section is nearly neutral; the broader branch-collision
section is the positive reconstruction diagnostic.

Angular refinement supports the continuum interpretation. Doubling from four
to eight projective directions changes mean population-phase RMS from about
`0.00959` to `0.00545`. At eight directions the aggregate local-to-causal
change is MSE `0.005079 -> 0.004766`, SSIM `0.6650 -> 0.6850`, and edge
retention `0.4328 -> 0.4665`.

The full external gate contains six structures and clean plus nine generated
corruption conditions, but no condition identity reaches the solver. Across
its 60 cases the causal collision section improves the local law on MSE in 55,
SSIM in 56, edge retention in 56, and variance fidelity in 40; all four improve
together in 36. Aggregate MSE is `0.006203` versus local `0.006990`, while
SSIM is `0.7000` versus `0.6780` and edge retention is `0.5276` versus
`0.4855`. It beats integrated FMMT MSE in four cases, but FMMT remains ahead
overall (`0.004828`, `0.7863`, `0.6449`). The clean geometric-interface case
still regresses in MSE, SSIM, and edge retention. This is a promising causal
kernel, not a promoted denoiser.

The exact records are `causal_information_lineage_2d_gate_20.json`,
`causal_information_lineage_2d_angular8_gate20.json`,
`causal_information_lineage_2d_root_resolved_gate20.json`, and
`causal_information_lineage_2d_six_source_full_gate20.json`.

The new post-transport volume diagnostic localizes the clean-interface debt.
For the clean geometric field, the initial continuous command is `1.0836`
support units and the causally transported law commands `1.0664`; their local
measure fields differ by `7.89%` relative RMS. The legacy density emitter
nevertheless realizes three hard germs at the tested phase. The continuum law
is not asking for a three-region smoothing operation. A fractional support
measure has been replaced by a jagged integer sample before the march. The
record is `causal_information_lineage_2d_interface_diagnostic20.json`.

Therefore the next object is the phase-integrated causal source measure, or an
equivalent weighted-source Hamilton--Jacobi formulation. It must converge to
the same result without selecting a hash phase and without treating a support
unit of `1.08` as three equally authoritative roots. Repeating or tuning the
present germ realization would attack the representation artifact rather than
the equation.

## Section order on the population fibre

Population phase is now represented as a nested base-two numerical fibre
`theta`. For every phase the same continuous information law emits a causal
root/branch measure `eta^theta`; phase never enters the image likelihood. Two
noncommuting projections can therefore be tested:

\[
\mathcal S\!\left(\int\eta^\theta\,d\theta\right)
\quad\hbox{and}\quad
\int\mathcal S(\eta^\theta)\,d\theta.
\]

Integrating branch mass first makes the collision mean stable but deflates
variance. Its 8-to-16 phase RMS is `0.00127` on the three-structure gate, yet
aggregate variance is only `0.717`. The scalar collision median retains more
variance but jumps whenever cumulative mass crosses one half. A Hodge
projection of the collision-weighted mean jet is even more stable and
catastrophically false: interface variance falls to about `0.05`. A joint
determinant-one geometric median converges but still loses the clean sparse
interface. These are retained negative sections.

The opposite ordering is the first positive structural section. At each phase,
select the maximum density of the exact two-particle collision law,

\[
k_\theta(x)\in\arg\max_k
\frac{\eta_x^\theta(k)^2}{h_x(k)},\qquad
q_P(x)=\frac1P\sum_{\theta\in\Theta_P}z_{xk_\theta(x)}.
\]

The hard maximum is not claimed as continuum physics; it is a quadrature of
the missing first-arrival section. Averaging after selection is essential.
On the geometric interface, the section's phase RMS falls from `0.00993` for
4-to-8 phases to `0.00596` for 8-to-16 and `0.00509` for 16-to-32. At sixteen
phases the clean interface improves local MSE `0.006143 -> 0.005343`, SSIM
`0.8287 -> 0.8572`, variance `0.8125 -> 0.8416`, and edge retention
`0.4784 -> 0.5408`. Mixed corruption improves all four simultaneously.

On the three-structure, four-condition gate, the sixteen-phase section changes
aggregate MSE/SSIM/variance/edge from
`0.005268/0.6583/0.7445/0.4462` locally to
`0.004695/0.6780/0.8580/0.5764`. All twelve cases improve edge and variance
fidelity; nine improve MSE, eight improve SSIM, and six improve all four.
The 8-to-16 section RMS is `0.00449`. It beats integrated FMMT MSE on tapered
hair under replacement and on geometric interfaces under uniform corruption,
but FMMT remains stronger in aggregate.

This ordering result is the important object, not sixteen as a setting. The
next Hamilton--Jacobi law must select a continuous branch section before
marginalizing numerical population representation, and must make the hard
`argmax` disappear under joint branch-space refinement. The main records are
`population_phase_collision_mean_refinement_gate20.json`,
`population_phase_section_order_refinement_gate20_8_16.json`, and
`population_phase_section_order_interface20_16_32.json`.

## Haar-density Hamilton--Jacobi lift

The ordering probe has now been replaced by a causal branch-space action. A
branch mass is not a continuum density: if `h_x(k)` is the branch Haar
quadrature, the invariant root datum is

\[
f_x(k)=\frac{m_x(k)}{h_x(k)},\qquad S_x(k)=\log f_x(k).
\]

This correction matters. Max-product propagation of raw bin mass changes when
the same branch measure is repartitioned. For a parent-child Markov kernel
`K`, the corresponding density kernel is
`kappa_px(k,l)=K_px(k,l)/h_x(l)`. On the stored Hopf--Lax parent simplex the
forward action is

\[
S_x(l)=\log L_x(l)
 +(1-t)\max_k\{S_p(k)+\log\kappa_{px}(k,l)\}
 +t\max_k\{S_q(k)+\log\kappa_{qx}(k,l)\}.
\]

This is a max-plus dynamic program on the same causal DAG as the positive
sum-product lineage law. It has no diffusion time, temperature, support band,
or corruption-dependent case. Subtracting the targetwise maximum of `S` is
only the additive gauge of Hamilton--Jacobi action.

The hard endpoint `argmax S` validates global path coherence but is not the
desired scalar map. The continuous endpoint instead collides two independent
coherent histories and integrates their density against Haar measure,

\[
\pi_x^{HJ}(dk)=
\frac{e^{2S_x(k)}h_x(dk)}{\int e^{2S_x(b)}h_x(db)},\qquad
q_x^{HJ}=\int z_x(k)\,\pi_x^{HJ}(dk).
\]

The population fibre is marginalized only after this per-realization path
integral. Thus real image interfaces may remain sharp, but no finite branch
index is selected in the scalar readout. Hamilton--Jacobi caustics are part of
the transported geometry; quadrature jumps are not.

On the three-structure, four-condition gate at eight phases, this HJ collision
barycenter changes aggregate MSE/SSIM/edge from
`0.005268/0.6583/0.4462` locally to `0.004271/0.7121/0.4962`. It improves MSE
and edge retention in all twelve cases and SSIM in eleven. More decisively, it
beats the ordinary causal collision mean in MSE, SSIM, and edge retention in
every case. Its remaining weakness is not hidden: aggregate variance is
`0.7257`, only slightly above the ordinary collision mean's `0.7166`, and
variance fidelity improves over the local law in only three cases.

A wider phase-four falsification screen uses six structures and clean plus all
nine external corruption generators. Across those 60 cases the same HJ
barycenter improves the local joint law on MSE in 57, SSIM in 57, and edge
retention in 50. It beats the ordinary causal collision mean on MSE and SSIM
in 53 and on edge retention in 56. Aggregate MSE/SSIM/edge are
`0.005967/0.7398/0.5536`, versus local
`0.007443/0.6911/0.5253`. Integrated FMMT remains stronger overall at
`0.005250/0.7864/0.6475`, although the HJ barycenter has lower condition-mean
MSE at both tested mixed-corruption densities and wins per-case MSE in 12 of
60. Because this screen uses only four population phases, it is falsification
evidence rather than a convergence claim.

On the clean/mixed geometric-interface pair, the barycenter's 8-to-16 phase
RMS is `0.00337`, compared with `0.00787` for the hard HJ section. Under mixed
corruption it improves the hard section's MSE/SSIM from `0.008589/0.5252` to
`0.007892/0.5893`, while the hard section retains more edge amplitude. The
next theoretical debt is therefore inside the terminal joint measure: retain
transported interface concentration without returning to a discontinuous
mode. The records are `population_phase_hj_barycenter_gate20_phase8.json` and
`population_phase_hj_barycenter_interface20_8_16.json`; the wider record is
`population_phase_hj_barycenter_six_source_full_gate16_phase4.json`.

## Causal-simplex collision order

The fixed exponent two still treats every accepted point as though exactly two
independent histories support it. The Hopf--Lax forest already contains the
correct continuous witness count. A root has only its local law. A one-parent
arrival has the local law and one causal parent. For a two-parent simplex with
barycentric fraction `t`, the effective number of parent witnesses is the
order-two Hill number

\[
N_{p}(t)=\frac{1}{(1-t)^2+t^2}.
\]

The terminal collision order is therefore

\[
\alpha_x=
\begin{cases}
1,&x\text{ is a root},\\
2,&x\text{ has one causal parent},\\
1+N_p(t_x),&x\text{ has a parent pair},
\end{cases}
\qquad
\pi_x^{\mathrm{simplex}}(dk)
\propto e^{\alpha_xS_x(k)}h_x(dk).
\]

Thus `alpha` moves continuously from two to three as a genuine parent pair
becomes balanced. It is not fitted from an edge score and never reads a noise
label. The scalar endpoint remains the Haar barycenter, so no branch switch is
introduced.

This improves the fixed two-history endpoint on the twelve-case phase-eight
gate in MSE and SSIM on nine cases and in edge and variance fidelity on all
twelve; seven improve all four simultaneously. Aggregate
MSE/SSIM/variance/edge move from
`0.004271/0.7121/0.7257/0.4962` to
`0.004173/0.7158/0.7385/0.5136`. On the clean and mixed geometric-interface
pair, all four metrics improve in both cases; 8-to-16 phase RMS is `0.00406`,
between the fixed-order barycenter's `0.00337` and hard section's `0.00787`.

The full phase-four 60-case screen is also positive. Relative to the fixed
two-history HJ endpoint, simplex order improves MSE in 51 cases, SSIM in 50,
edge in 58, variance fidelity in 56, and all four together in 46. Relative to
the local law it wins MSE and SSIM in 58 cases and edge in 53. Aggregate
MSE/SSIM/variance/edge become
`0.005764/0.7432/0.7289/0.5669`; FMMT remains ahead overall at
`0.005250/0.7864/0.8112/0.6475`. Tapered hair under 10% replacement and the
geometric interface under 25% replacement are the two remaining local
MSE/SSIM regressions. The records are
`population_phase_hj_simplex_gate20_phase8.json`,
`population_phase_hj_simplex_interface20_8_16.json`, and
`population_phase_hj_simplex_six_source_full_gate16_phase4.json`.

## Optimizing the transport posterior, not its fixed point

The first 1-D experiment above the fixed-map level now retains the posterior
over predecessor--current--successor transport histories. If `K^-` and `K^+`
are the two adjacent joint information-metric kernels, the current branch `j`
couples its two causal sides by

\[
\Gamma_x(i,k\mid j)=P_x^-(i\mid j)P_x^+(k\mid j).
\]

The outer `(value, jet)` states are parallel transported to `x`. Their pooled
transported covariance supplies a determinant-one Sasaki precision `P_x`, and
the branch action is the complete conditional phase disagreement

\[
a_x(j)=\sum_{i,k}\Gamma_x(i,k\mid j)
\left\|\tau_+q_{x-1,i}-\tau_-q_{x+1,k}\right\|_{P_x}.
\]

This is not the previously rejected inverse-participation count of two
marginal arrival densities. Two balanced but contradictory histories have
large action. The complete path-density posterior is changed by reciprocal
action,

\[
\pi'_x(j)=
\frac{\pi_x(j)/a_x(j)}{\int \pi_x(b)/a_x(b)\,dh_x(b)}.
\]

Consequently its expected phase action is the harmonic mean and cannot exceed
the original arithmetic mean:

\[
\mathbb E_{\pi'_x}a_x
=\frac{1}{\mathbb E_{\pi_x}(1/a_x)}
\leq\mathbb E_{\pi_x}a_x.
\]

The implementation checks that inequality at every vertex. Probability is
conserved, constants are exact, and affine fields are exact away from the
explicit reflected boundary chart. There is no duration, phase threshold,
bandwidth, or noise label. This is the first executable state in the denoiser
that changes the probability of the transport histories themselves rather
than accelerating a fixed restoration map.

The endpoint is nevertheless rejected. On the focused 96-sample,
six-structure, thirteen-condition, one-seed gate, applying the coupled law to
the target-free nested midpoint fibre contracts the desired phase action but
contracts the signal orbit as well. Its coupled mean has clean TV/variance
`0.352/0.594` and noisy MSE/variance `0.003081/0.491`; the corresponding
collision mean is only slightly less deflated. The exact record is
`1d_coupled_transport_posterior_focused96.json`.

The same transport posterior on the richer three-characteristic information
fibre proves that the state can retain amplitude. Its coupled collision median
reaches clean TV/variance `0.748/0.856`, compared with `0.700/0.811` for the
earlier information-lineage collision mean. But it admits corrupted histories:
noisy MSE rises from `0.002308` to `0.003830` and noisy TV from `1.031` to
`1.629`. The smoother coupled collision mean also loses, with noisy MSE
`0.003021`. The exact record is
`1d_coupled_transport_full_fibre_focused96.json`.

The failure is mathematical, not a relaxation defect. A flat latent history
has nearly zero two-sided phase disagreement even when it has displaced
coherent structure into the residual coordinate. Reciprocal phase action
therefore possesses a false null direction. Reweighting it more strongly,
iterating it, or tuning its metric would deepen the wrong contraction.

The retained advance is the coupled transport posterior `Gamma`. The next
action must constrain its descent by two additional transported facts:

1. legitimate signal spread may not disappear merely because a flatter
   history has smaller phase action; the update needs a Kantorovich/coverage
   constraint on the transported signal law; and
2. structure may move from residual to signal only when residual evidence is
   carried from distinct observation ancestries. Endpoint residual
   `r_x=y_x-z_x` cannot certify itself.

This is the 1-D analogue of the zonotopic-mixture lesson: retain competing
transport histories and falsify them by independent observation consistency;
do not merge them into their mean. No 2-D port is justified until this
constrained joint action improves the broad 1-D battery without clean
deflation.

### Pointwise coverage is not structural coverage

The first constraint probe applied a parameter-free information projection to
the contracted law. Let `pi` be the pre-contraction path law, `pi'` the
reciprocal-phase-action law, `z` the branch value, and
`m=E_pi z`. If contraction reduces

\[
C_x=\mathbb E_\pi (z-m)^2,
\]

the corrected law is the unique exponential tilt

\[
d\widehat\pi_\lambda(z)
\propto e^{\lambda(z-m)^2}\,d\pi'(z),\qquad \lambda\geq0,
\]

whose moment equals `C_x`. If `pi'` already has at least that coverage,
`lambda=0`. Thus the update is the minimum-information projection onto a
transported second-moment half-space; no preservation strength is selected.
Every fibre conserves mass and the implementation verifies zero moment deficit
to numerical precision.

This repair is also rejected. On the richer information fibre its mean has
clean MSE/TV/variance `0.000517/0.564/0.759` and noisy
`0.002550/1.080/0.691`, slightly worse than the unprojected coupled mean. On
the nested fibre the changes are negligible and clean TV remains `0.351`.
The exact record is `1d_coupled_transport_coverage_focused96.json`.

The negative is decisive: scalar-fibre variance can be preserved exactly
while the section's spatial orbit, total variation, and dynamical phase still
collapse. Structural spread is a property of the coupled path law, not the
pointwise value marginal. The required Kantorovich/coverage constraint must
therefore act on transported increments or complete phase paths, for example
on the pushforward of `Gamma_x(i,k|j)` through its value--jet displacement.
Another pointwise variance, range, or amplitude correction is ruled out.

A stricter control replaced scalar distance by the determinant-one Mahalanobis
radius of each local `(value, jet)` particle and restored that complete phase
second moment by the same exact information projection. It too is neutral or
harmful: on the information fibre its clean mean remains at TV `0.565` and its
noisy MSE is `0.002534`; on the nested fibre clean TV remains `0.351`. The
record is `1d_coupled_transport_phase_coverage_focused96.json`. Thus even a
pointwise phase-bundle moment is not path coverage. The constraint must retain
the joint law across distinct base points, not merely richer coordinates in
each local fibre.

## Complete histories and uncertainty about transport

Pointwise collision was next lifted to the complete branch path.  With child
reference measure `h` and Markov kernel `K`, the order-two path collision uses

\[
K^{(2)}(i,dj)=h(dj)\left(\frac{K(i,dj)}{h(dj)}\right)^2
=\frac{K(i,dj)^2}{h(dj)},
\]

squares the observation likelihood density, and only then runs a second
sum-product pass.  This retains every coherent history; unlike a Viterbi
traceback it does not let one corrupted branch seize the path.  Constants and
affine interiors remain exact and every marginal conserves mass.

On the broad 128-sample, six-structure, thirteen-condition, two-seed gate,
complete-path collision improves the clean information-lineage mean from MSE
`0.000304` to `0.000281` and variance ratio from `0.822` to `0.832`, but noisy
MSE rises from `0.002204` to `0.002563`.  It helps sparse replacement and some
impulse cases while overcommitting under diffuse noise.  A single Viterbi
history is much worse (`0.01371` noisy MSE); posterior remarching is still
`0.00574`.  The exact records are
`1d_global_path_collision_focused96.json` and
`1d_path_transport_broad128_2seed.json`.

The Hellinger fidelity between the ordinary and collided state marginals,

\[
F_x=\left(\sum_j\sqrt{\pi_x(j)\pi_x^{(2)}(j)}\right)^2,
\]

supplies a parameter-free Fisher--Rao geodesic coordinate.  Its mean readout
improves replacement at densities `0.10`, `0.25`, and `0.40`, and improves
first- and second-difference error in aggregate, but noisy MSE remains 3.1%
above the local collision baseline.  Fidelity is therefore an interpolation
coordinate, not a cleanliness certificate: mean fidelity is actually higher
for noisy (`0.894`) than clean (`0.863`) inputs.

State-marginal fidelity still omits uncertainty about the map.  For each edge,
the ordinary posterior transport plan is

\[
\Xi_x(i,j)\propto
\alpha_x(i)K_x(i,j)L_{x+1}(j)\beta_{x+1}(j).
\]

Comparing `Xi` with its order-two counterpart by Hellinger fidelity measures
agreement about predecessor--successor coupling itself.  The product of the
two incident edge fidelities is the vertex survival probability.  This
transport-plan geodesic lowers broad noisy MSE from `0.002204` to `0.002186`
and lowers both derivative errors, but clean MSE rises to `0.000331` and clean
TV falls to `0.645`.  Correcting the geodesic endpoints to the local-collision
and complete-path-collision laws restores clean behavior, but the best such
history mixture is only near-neutral on noisy MSE.  Records:
`1d_transport_plan_fidelity_broad128_2seed.json`,
`1d_self_consistent_transport_broad128_2seed.json`, and
`1d_two_history_action_transport_broad128_2seed.json`.

This separates two necessary state variables:

1. fibre uncertainty describes which scalar/jet section is supported; and
2. edge-plan uncertainty describes which transport map carried that support.

Neither may stand in for the other.  Direct reciprocal-action selection
between their scalar readouts contracts expected estimator action by the
arithmetic--harmonic inequality, but the histories have nearly identical
pointwise action and the resulting midpoint is worse.  Exact contraction of a
readout selector is not yet contraction of the joint transport estimator.

## The connection is a random variable

The v3 support study supplied one useful prohibition.  Its complete positive
algebra is appropriate for independent semantic supports, but value, jet, and
conserved residual are coordinates of one phase bundle, not three independent
supports.  Applying

\[
(1+K_v)(1+K_j)(1+K_r)-1
\]

to them lowers scalar error in a small control while increasing derivative
error and jaggedness.  This control is rejected; a unified phase metric must
not be split into pseudo-support rules.

The productive lift instead makes the connection itself uncertain.  On edge
`e`, the two endpoint particle laws induce

\[
\mu_e=\mathbb E q_e^- - \mathbb E q_e^+,
\qquad
\Sigma_e=\operatorname{Cov}(q_e^-)+\operatorname{Cov}(q_e^+).
\]

Using `Sigma_e` while retaining the exact zero connection improves all clean
quick-gate value and derivative metrics but loses on corruption.  Using the
full inferred drift `mu_e` improves noisy quick-gate MSE by about 7% but
deflates clean structure.  Thus drift and metric uncertainty are genuinely
complementary phases, not interchangeable implementation choices.

Connection authority is measured without a scale setting by its Hotelling
action and survival probability,

\[
s_e=\mu_e^\top\Sigma_e^{-1}\mu_e,
\qquad \rho_e=\frac{s_e}{1+s_e}.
\]

The strongest control requires two independent connection histories at both
edge endpoints, transporting `rho_e^4 mu_e`.  On a quick 64-sample gate it
improves clean and noisy MSE plus both derivative errors.  The broad gate
falsifies it as the unified estimator: clean MSE rises 3.4%, while aggregate
noisy MSE improves 6.8% and both noisy derivative errors improve about 6.5%.
The gain is concentrated in replacement and impulse phases: replacement 0.40
MSE falls 18.5%, and salt--pepper 0.25 falls 36.9%, whereas Gaussian and
uniform cases regress about 3%.  The exact record is
`1d_bidirectional_connection_broad128_2seed.json`.

This is the clearest phase result so far.  Diffuse corruption creates high
connection-drift authority without granting that drift structural ownership;
sparse replacement creates lower authority but coherent path ownership.  A
scalar power or threshold on `rho` cannot distinguish them.  The next retained
experiment must therefore lift connection ownership into the joint path state
and let its ancestry evolve under sum-product transport.  No 2-D port is yet
justified.

## Connection ownership is a transported reference measure

The connection lift was implemented as a genuine joint path law on
`(ownership mode, signal branch)`.  Mode zero carries the fused zero-defect
information connection and mode one carries the bidirectionally certified
drift connection.  The signal likelihood is common to both modes.  Thus mode
ownership is not an extra observation likelihood and cannot win merely by
explaining the same target twice.

The first ownership reference used the exact target-free energy-distance
authority

\[
a_x=\frac{\mathbb E|Z_x-Z'_x|}{2\,\mathbb E|y_x-Z_x|}
\]

and its value `c_x` transported from the two exact source identities of every
particle.  Its sparse bypass mass was

\[
s_x=(1-a_x)\frac{(c_x-a_x)^2}{a_x^2+c_x^2},
\qquad m_x=(1-s_x,s_x).
\]

Adjacent Bernoulli references are connected by their exact minimum-switching
optimal-transport plan.  Sum-product then evolves the complete joint state,
and order-two collision is applied against the joint reference
`m_x \otimes h_x` before either coordinate is marginalized.  This matters:
using a uniform mode prior or treating `s_x` as a likelihood silently changes
the estimator.

The broad 128-sample, six-structure, thirteen-condition, two-seed gate gives
the joint collision a 1.01% noisy-MSE improvement and approximately 1% gains
in both derivative errors, but clean MSE rises 0.71%.  The gain is again
concentrated in replacement 0.40, while the clean loss is largest on smooth
geometry.  The exact record is
`1d_ot_connection_ownership_broad128_2seed.json`.  The joint state and its
optimal-transport reference are retained; energy-root ownership is rejected.

## Phase is disagreement about the connection law

For adjacent edge connection posteriors
`N(g_{x-1}, C_{x-1})` and `N(g_x,C_x)`, the coordinate-free Bhattacharyya
affinity is

\[
B_x=
\frac{\det(C_{x-1})^{1/4}\det(C_x)^{1/4}}
     {\det((C_{x-1}+C_x)/2)^{1/2}}
\exp\!\left[-\frac18\Delta g_x^\top
\left(\frac{C_{x-1}+C_x}{2}\right)^{-1}\Delta g_x\right].
\]

The squared Hellinger defect `1-B_x` is continuous, dimensionless, and
invariant under any common nonsingular change of value--jet--residual
coordinates.  It directly represents uncertainty about transport itself.
The implementation verifies that invariance numerically as well as constant
exactness and mass conservation.

Using the absolute Hellinger defect as the mode reference strongly separates
the aggregate phases: mean drift ownership is `0.207` on clean signals and
`0.643` on noisy signals.  It improves broad noisy MSE by 2.49%, first
difference error by 1.30%, second difference error by 0.86%, and moves noisy
TV toward one.  It nevertheless raises clean MSE by 1.65% and both clean
derivative errors by roughly 1.5%.  It still mistakes legitimate evolution of
a clean connection for permission to change estimators.  Record:
`1d_connection_hellinger_broad128_2seed.json`.

A continuous-marginal control integrated reciprocal Mahalanobis conductance
over the Gaussian connection law.  In the three-dimensional phase bundle the
integral is the exact Gaussian Newton potential

\[
\mathbb E\,\|d-G\|_{\Sigma^{-1}}^{-1}
=\frac{\operatorname{erf}(r/\sqrt2)}{r},
\qquad
r^2=(d-\mu)^\top\Sigma^{-1}(d-\mu),
\]

with analytic origin value `sqrt(2/pi)`.  This is a beautiful parameter-free
continuous kernel, but it is the wrong operation at this stage: marginalizing
connection uncertainty before paths compete flattens transition selectivity.
On the quick gate its collision readout worsens clean and noisy MSE and
deflates noisy variance from `0.645` to `0.603`.  Record:
`1d_gaussian_connection_potential_quick64.json`.  Continuous marginalization
before ancestry comparison is rejected.

The productive phase coordinate compares local connection uncertainty `u_x`
with the uncertainty `\bar u_x` transported from the exact sources of the
current branch posterior.  Their Bernoulli Fisher--Rao affinity gives

\[
F_x=\sqrt{u_x\bar u_x}
   +\sqrt{(1-u_x)(1-\bar u_x)},
\qquad s_x=1-F_x.
\]

This vanishes when a noisy phase is spatially self-consistent and activates
when the local connection law disagrees with what its own ancestry carries.
It does not name Gaussian, uniform, replacement, impulse, or any mixture.
On the broad gate its joint collision improves noisy MSE by 1.27%, first
difference error by 1.36%, second difference error by 1.21%, and moves noisy
TV substantially toward one.  The aggregate clean tax falls to 0.14% MSE,
0.07% first difference, and 0.12% second difference.  The remaining clean
loss is localized in the chirp and mixed-transport stress signals; smooth
geometry is essentially neutral.  Record:
`1d_transported_hellinger_contrast_broad128_2seed.json`.

Separating covariance from mean provides the final control.  Transported
covariance-only contrast makes drift ownership almost vanish (`0.0045` clean,
`0.0108` noisy) and removes the clean effect, but improves quick noisy MSE by
only 0.28%.  Therefore expected connection change is necessary evidence and
connection covariance is necessary authority.  They cannot be reduced to
independent scalar gates without losing the useful phase.

The retained candidate is transported full Hellinger contrast, but it is not
yet the unified estimator.  Its next lift must preserve the joint dependence
of connection mean and covariance through ancestry and contract action in
that joint law before scalar ownership is formed.  This is not another
fixed-point denoiser and not another exponent on `rho`: it is a transport
estimator over transport estimators.  The 2-D gate remains closed until the
chirp/mixed clean tax is removed rather than averaged away.

## Transport the law before scalarizing it

The next implementation keeps the Gaussian connection state intact through
ancestry.  The two incident edge laws at a vertex meet at the
affine-invariant SPD geodesic midpoint of their covariances.  For every signal
branch, the two exact source-vertex Gaussian laws meet by the same operation.
Only then is their Hellinger defect from the local vertex law evaluated.  The
branch posterior marginalizes those defects after transport, not before it.
This complete-law construction is invariant under a common nonsingular
change of bundle coordinates, constant exact, and probability conserving.

Used as another scalar ownership variable, the complete law increases broad
noisy gains to 2.51% MSE, 2.41% first-difference error, and 1.92%
second-difference error, but clean MSE rises 1.80%.  Preserving the law helps;
asking it to choose between two modes is still the wrong endpoint.  Record:
`1d_transported_gaussian_law_contrast_broad128_2seed.json`.

The useful endpoint acts directly on transition conductance.  If `K_0` is
the zero-connection conductance, `K_1` the certified drift conductance, and
`s_{ij}` the topological midpoint of the source/target branch connection
contrasts, define

\[
K_{ij}^{\star}=(1-s_{ij})K_{0,ij}+s_{ij}K_{1,ij}.
\]

Equivalently, with actions `A_k=1/K_k`,

\[
A_{ij}^{\star}
=\left(\frac{1-s_{ij}}{A_{0,ij}}
       +\frac{s_{ij}}{A_{1,ij}}\right)^{-1}
\le (1-s_{ij})A_{0,ij}+s_{ij}A_{1,ij}.
\]

Thus the estimator contracts action by the arithmetic--harmonic inequality;
it does not iterate a restoration map until a chosen stopping time.  The first
broad form improves noisy MSE by 3.20%, both noisy derivative errors by about
3.6%, and noisy TV from `1.259` to `1.196`.  Clean derivative errors improve
about 0.14%, while clean MSE rises 0.42%.  Exact removal of self-referential or
duplicate source pairs raises the valid independent ancestry fraction to
`0.9818` and reduces the clean MSE tax to about 0.37% in the matched clean
control.  Record: `1d_action_contracting_connection_broad128_2seed.json`.

Two controls clarify the geometry.  Pulling source defect vectors through the
first-jet shear `(v,j,r) -> (v+(x-s)j,j,r)` improves a 64-sample screen but
worsens the 128-sample clean gate; edge defects were already expressed in a
common comparison frame and the shear double-transports them.  It is rejected.
Requiring the complete-law defect and population phase defect to survive by
simple multiplication is clean-safe but reduces quick noisy MSE gain to only
0.61%; strict conjunction is also rejected.

The balanced law treats the branch Gaussian comparison `s_b` and transported
population phase comparison `s_p` as independent witnesses of one latent
connection state.  Their normalized agreement is odds multiplication,

\[
s=\frac{s_b s_p}
        {s_b s_p+(1-s_b)(1-s_p)}.
\]

This is not an exponent, threshold, or fitted mixing coefficient.  It accepts
drift when both witnesses agree, accepts the zero connection when both reject
drift, and removes contradictory evidence before harmonic action contraction.
On the broad 128-sample, six-structure, thirteen-condition, two-seed gate its
collision readout gives:

- noisy MSE `0.002084 -> 0.002038` (`-2.22%`);
- noisy first-difference error `-2.47%`;
- noisy second-difference error `-2.41%`;
- noisy TV ratio `1.259 -> 1.209`, toward one;
- clean MSE `+0.14%`, first difference `+0.02%`, and second difference
  `+0.06%`.

The noisy gain is not confined to one named corruption: MSE improves on mixed
0.25/0.40, all replacement densities, both salt--pepper densities, and is
nearly neutral under diffuse noise.  Gaussian and uniform MSE regress only
about 0.3--0.6%, while Gaussian derivative errors improve.  Across individual
rows, MSE improves in `93/156`, first-difference error in `113/156`, and
second-difference error in `111/156`.  The record is
`1d_action_contracting_phase_odds_broad128_2seed.json`.

This is the strongest balanced candidate, not the end of the 1-D problem.
The clean chirp still rises 4.1% MSE and mixed transport stress 1.8%; those
cannot be hidden by the aggregate.  The candidate is exposed in the Dear
PyGui 1-D method selector for direct composited-series inspection, with no new
controls.  The next theoretical target is a connection law whose phase
evolution follows legitimate chirp acceleration without granting diffuse
noise the same action.  The 2-D gate remains closed.

## Coherent acceleration and Newton-supported transport

Two fixed-order controls fail to solve the remaining chirp error.  Comparing
each Gaussian connection law with the affine-invariant midpoint of its two
neighboring laws measures a covariant acceleration defect.  It improves the
64-sample aggregate, but on the 128-sample clean gate it raises chirp MSE 6.8%
and mixed-stress MSE 9.5%: a chirp legitimately has nonzero connection
acceleration.  Differentiating that scalar phase once more with a Bernoulli
Hellinger midpoint lowers the damage but still raises chirp MSE 4.9%.
Acceleration magnitude and scalar jerk are therefore rejected as drift
authority.  Records:
`1d_action_contracting_acceleration_odds_quick64.json` and
`1d_action_contracting_jerk_odds_quick64.json`.

The existing symmetric second-jet fibre is not a shortcut.  Although a
quadratic packet jet closes algebraically through a linear chirp walk, its
current reference law worsens 128-sample clean chirp MSE more than sevenfold
and damages four of the other five structures.  Missing representation cannot
be repaired by stapling an unvalidated curvature fibre onto the estimator.

The useful correction comes from optimizing the accepted transport direction
itself.  Let `Xi_e` be the posterior edge plan under the zero connection,
`delta` the branch-pair connection defect, and `(mu_e,Sigma_e)` the inferred
connection law.  Along the certified drift direction, the exact quadratic
line minimizer is

\[
\alpha_e=
\Pi_{[0,1]}
\frac{\mu_e^\top\Sigma_e^{-1}
      \mathbb E_{\Xi_e}\delta}
     {\mu_e^\top\Sigma_e^{-1}\mu_e}.
\]

The convex projection is the physical segment between zero and inferred
connection, not a fitted clipping threshold.  The drift kernel is recomputed
at `alpha_e rho_e^4 mu_e`; no scalar output interpolation is performed.  On
clean 128-sample signals, mean `alpha` is `0.70` for chirp, `0.36` for smooth
geometry, `0.29` for pulses, and `0.94` for the oscillatory composite, while a
matched mixed corruption drives it to roughly `0.92` across structures.

Combining this Newton-supported kernel with the retained phase-odds harmonic
action gives the strongest balanced broad result:

- clean MSE improves `0.21%`;
- clean first/second-difference error improves `0.10%/0.02%`;
- noisy MSE improves `2.06%`;
- noisy first/second-difference error improves `2.30%/2.26%`;
- noisy TV ratio moves `1.259 -> 1.213`, toward one.

It improves noisy MSE on `94/156` rows and the two derivative errors on
`110/156` and `108/156` rows.  The gain remains distributed across mixed,
replacement, salt--pepper, and Gaussian derivative cases.  The clean aggregate
no longer pays a tax, but chirp still rises 3.8% MSE and mixed stress 1.6%; the
1-D goal is therefore not complete.  Exact record:
`1d_action_contracting_newton_odds_broad128_2seed.json`.

The Dear PyGui research method now runs this Newton-supported law at a default
128 samples.  It has no physical controls or duration.  Its current Python
oracle repeats three full path laws to keep the comparison transparent; this
is representation debt, not theoretical necessity, and must be removed by
sharing particle geometry before higher-resolution or 2-D work.

## Transport uncertainty above the Newton point

A full tangent control was constructed before changing the estimator.  Each
edge Gaussian was lifted to the nine-dimensional tangent bundle consisting of
three mean directions and the six-dimensional affine-invariant SPD tangent.
Mean and covariance accelerations were congruence-transported from each exact,
target-free ancestry pair into the target Gaussian frame.  In that whitened
frame the intrinsic Fisher metric plus the transported ancestry-population
covariance measures uncertainty of the tangent itself, and the exact Gaussian
Hellinger defect maps action to phase.

This is geometrically sound but not yet the estimator.  It improves the
64-sample clean chirp, mixed, and oscillatory controls, and its broad noisy
MSE, but the 128-sample chirp reverses to a roughly 6% tax.  Raw covariant
acceleration still depends on path parameterization; calling it a phase law
does not cure that.  The tangent lift remains a diagnostic and is not promoted.
Records: `1d_action_contracting_tangent_rational_quick64.json` (rejected
non-Hellinger scalarization) and
`1d_action_contracting_tangent_hellinger_quick64.json`.

The more fundamental defect is the Newton collapse itself.  Write the
connection on edge `e` as the geodesic segment

\[
\Gamma_e(\alpha)=\rho_e^4\alpha\mu_e,\qquad 0\le\alpha\le1,
\]

and let `P_e=Sigma_e^{-1}` and
`d_e=E_{Xi_e}[delta]`.  Apart from an additive constant, its action is

\[
S_e(\alpha)
=\frac12\left(
q_e\alpha^2-2r_e\alpha
\right),\qquad
q_e=\mu_e^\top P_e\mu_e,\quad
r_e=\mu_e^\top P_e d_e.
\]

Newton retains only the projected mode `r_e/q_e`.  The lifted estimator keeps
the complete action measure instead,

\[
d\pi_e(\alpha)=
\frac{
  \exp[-S_e(\alpha)]\,d\alpha
}{
  \int_0^1\exp[-S_e(a)]\,da
},
\]

which is a truncated Gaussian on the physical zero--drift segment.  Its mean
and variance are analytic.  The mean transports the connection.  Crucially,
the variance has two different appearances that must not be confused:

\[
\operatorname{Cov}(\Gamma_e)
=\rho_e^8\operatorname{Var}_{\pi_e}(\alpha)\,
  \mu_e\mu_e^\top
\]

broadens the predictive Gaussian connection law, while marginalizing the
path action gives

\[
\mathbb E_{\pi_e}
\|\delta-\rho_e^4\alpha\mu_e\|_P^2
=
\|\delta-\rho_e^4\mathbb E\alpha\,\mu_e\|_P^2
+\rho_e^8\operatorname{Var}(\alpha)\|\mu_e\|_P^2.
\]

Thus epistemic uncertainty about transport adds irreducible path action; it
does not merely inflate covariance and excuse uncertain histories.  The flat
action limit is the uniform segment law with mean `1/2` and variance `1/12`.
There is no iteration horizon, fitted phase threshold, noise class, or
physical scale setting.  This is the first candidate in the simmer that is
strictly higher than the Newton transport point estimate.

Moment transport alone is not sufficient.  If the `alpha` variance is merely
folded into the edge covariance before path transport, dense replacement
histories survive too easily: a 64-sample broad control improves clean MSE but
is essentially neutral on noisy collision MSE.  Epistemic broadening after
branch identity has been forgotten cannot reconstruct the correlation between
connection and lineage.

That correlation can be retained analytically.  For every source/target
branch pair `(i,j)`, let `z_ij` be its connection defect and let
`v=rho^4 mu` be the full certified drift direction.  In the Gaussian
information metric,

\[
A_{ij}(\alpha)
=\|z_{ij}-\alpha v\|_P^2
=c_{ij}-2r_{ij}\alpha+q\alpha^2,
\]

where `c_ij=z_ij^T P z_ij`, `r_ij=z_ij^T P v`, and `q=v^T P v`.
The continuous connection coordinate can therefore be removed *after* it has
been coupled to the branch, but before path messages, with the exact Gibbs
conductance

\[
\begin{aligned}
K_{ij}
&\propto \nu_j\int_0^1
\exp[-A_{ij}(\alpha)/2],d\alpha\\
&=\nu_j
\exp\!\left[-\frac12
\left(c_{ij}-\frac{r_{ij}^2}{q}\right)\right]
\sqrt{\frac{2\pi}{q}}
\left[
\Phi\!\left(\frac{q-r_{ij}}{\sqrt q}\right)
-\Phi\!\left(\frac{-r_{ij}}{\sqrt q}\right)
\right].
\end{aligned}
\]

The `q -> 0` limit is `nu_j exp(-c_ij/2)`.  This is a continuous branchwise
action marginal, not an alpha grid or iterative contractor.  It uses the raw
Gaussian information metric: determinant-normalizing that metric, harmless
for reciprocal distance up to row scale, changes a Gibbs posterior and is a
rejected control.

At 64 samples the raw-information form's complete-path collision improves
clean MSE about 4.4%, noisy MSE 1.5%, and noisy first/second-difference error
about 5.0%/7.9%.  Ordinary collision is nearly neutral on aggregate noisy MSE.
At 128, however, complete-path collision over-concentrates coherent chirp and
smooth histories, while ordinary collision retains the clean chirp gain but
still regresses some smooth/step mixed cases.  The continuous connection law
is therefore ahead of the Newton point estimate in representation, but its
phase/path readout is not yet scale-consistent.  The 1-D gate remains open and
the 2-D gate remains closed.

A full-vector Gaussian marginal is the next necessary falsification control,
but not the answer.  Treating the transported connection as
`G ~ N(rho^4 mu, rho^8 Sigma)` makes the branch defect law the exact Gaussian
convolution with covariance `Sigma + rho^8 Sigma`.  This carries uncertainty
in all three connection directions rather than only along `alpha`.  On the
targeted 128-sample gate it nevertheless worsens Gaussian-corruption MSE by
roughly 1--3%, raises clean chirp and mixed error about 1.5--1.8%, and more than
doubles the already small clean smooth-geometry error.  A diffuse Gaussian
explanation of diffuse fluctuations is still rewarded by its own likelihood.
Covariance transport is not phase transport, and this control is rejected.

The surviving research question is therefore joint but more specific: phase
must be represented as coherent evolution of the connection law along lineage,
while the connection and lineage remain coupled.  It cannot be a fixed scalar
derivative order, a covariance inflation, a post-hoc collision order, or
another iteration of a point denoiser.

## Spherical phase order from exact ancestry

Phase is not another derivative magnitude.  At vertex `x`, whiten the Gaussian
connection mean in its own covariance frame,

\[
w_x=C_x^{-1/2}m_x,\qquad u_x=\frac{w_x}{\|w_x\|}.
\]

Under an arbitrary nonsingular change of connection coordinates, these
directions change only by an orthogonal gauge in the whitened frame.  Angular
relations are therefore coordinate-invariant.  For each target branch `b`,
the two exact source laws are congruence-transported into the target frame.
Their unit directions `u_a,u_c` define the great-circle phase path.  With the
actual source and target base coordinates, spherical interpolation or
extrapolation gives

\[
\widehat u_{x,b}
=\frac{
 \sin[(1-t)\theta]u_a+\sin(t\theta)u_c
}{\sin\theta},\qquad
t=\frac{x-a}{c-a},\quad
\theta=\arccos(u_a^\top u_c).
\]

No lag or frequency band is chosen: every valid, distinct, target-free
ancestry pair contributes with its posterior branch mass.  Their spherical
order parameter is

\[
R_x=\left\|\sum_b\pi_{x,b}\widehat u_{x,b}\right\|,
\qquad 0\le R_x\le1.
\]

`R_x` is high only when the complete ancestry population predicts a coherent
phase at the target.  In the 128-sample diagnostic it is about `0.96` on clean
smooth geometry and `0.68` on clean chirp, versus roughly `0.17--0.21` under
Gaussian or mixed corruption and `0.27--0.32` under salt--pepper.

Several combination controls determine its meaning.  Treating spherical phase
as a second positive drift witness through odds protects chirp and Gaussian
cases but suppresses the strong connection needed for impulse replacement.
Taking a union restores impulses but over-transports clean structure.  A raw
phase veto is closer, but discards the older population phase-defect evidence
that distinguishes salt impulses from diffuse Gaussian fluctuations.

The coherent factorization retains both objects.  If `s_p(x)` is the existing
Bernoulli/Hellinger population phase-defect survival, define

\[
s_{\mathrm{coherent}}(x,b)
=R_x[1-s_p(x)][1-s_{\mathrm{defect}}(x,b)],\qquad
s_{\mathrm{transport\ phase}}(x,b)
=s_p(x)\,[1-s_{\mathrm{coherent}}(x)].
\]

Thus spherical order counts as protective coherence only when neither the
population phase-defect law nor the branch Gaussian connection defect already
rejects it.  A decisive impulse defect therefore retains transport authority;
an ambiguous clean branch may be protected by coherent evolution.  The
phase-defect witness
then combines with the Gaussian connection defect by normalized odds, exactly
as before.  The construction introduces no threshold, physical constant,
band, named corruption, or run duration.

The more consequential use of phase is to evolve uncertainty about the
connection estimator itself.  Newton point transport contracts noisy paths
strongly, while the continuous action posterior retains clean structure more
faithfully.  They are not competing output filters: they are two measures on
the same zero--drift connection segment.  Let `K_N` be the Newton conductance
kernel and `K_pi` the branchwise action-marginal kernel.  Posterior ownership
is extracted from the same phase and defect laws,

\[
\omega_{x,b}
=R_x^2[1-s_p(x)][1-s_{\mathrm{defect}}(x,b)],
\qquad
\omega_x=\sum_b\pi_{x,b}\omega_{x,b}.
\]

The square is not a fitted exponent.  It is the order-two collision
probability that two independent ancestry-phase histories belong to the same
ordered population.  Accidental alignment under diffuse or replacement noise
therefore vanishes quadratically, while coherent clean phase remains.

A cavity-surprise control is informative but rejected as ownership.  Dividing
the smoothed branch posterior by its local likelihood and renormalizing gives
the exact target-free cavity law; its Hellinger displacement identifies many
unsupported impulses.  Multiplying `omega` by one minus this displacement,
however, worsens low-density salt by as much as 2% on the targeted smooth
control.  Harmonic path transport is nonlinear, so moving a local kernel
toward the Newton endpoint need not move its final collision readout
monotonically.  Cavity surprise remains a diagnostic, not a gate.

At an edge, endpoint phase ownerships are conductances in series.  Their
harmonic phase conductance
`omega_e=2 omega_x omega_{x+1}/(omega_x+omega_{x+1})` evolves the connection
kernel,

\[
K_e=(1-\omega_e)K_{N,e}+\omega_e K_{\pi,e}.
\]

The Gaussian connection state follows the identical mixture law.  Its mean is
the weighted Newton/posterior mean; its covariance includes the posterior
alpha variance and the exact between-estimator outer product
`omega_e(1-omega_e)(m_N-m_pi)(m_N-m_pi)^T`.  Thus phase changes not only a
scalar gate but the uncertainty of transport itself.  Strong branch or
population defects force `omega` toward Newton; coherent, non-defective phase
admits the continuous action posterior.  No fixed-point iteration is used.

Two edge controls are rejected.  Independent endpoint collision
`omega_x omega_{x+1}` blocks posterior leakage through isolated impulses but
collapses most clean gains back toward the Newton endpoint.  The normalized
Fisher--Rao/Hellinger midpoint is smoother and retains clean gains, but does
not materially improve the low-density salt tail across structures.  The
arithmetic topological midpoint remains a broad reference control.  Harmonic
phase conductance halves much of the sparse-tail cost while preserving
coherent clean gains, and is the promoted continuous series law.

On the matched 128-sample, six-structure, thirteen-condition, two-seed gate,
the local collision readout gives:

- clean MSE `0.000303732 -> 0.000302818` (`-0.30%` versus the zero
  connection), improving `0.088%` over the retained Newton/phase candidate;
- clean mixed stress `-0.20%`, smooth geometry `-1.69%`, chirp `-0.10%`,
  pulses `-0.26%`, with step and oscillatory structure essentially neutral
  relative to the retained candidate;
- noisy MSE `0.002083774 -> 0.002040730` (`-2.07%`), marginally better
  than the retained candidate;
- noisy first/second-difference error `-2.31%/-2.27%`, both also better
  than the retained candidate;
- noisy TV ratio `1.259 -> 1.213`, toward one;
- noisy row wins against the zero connection increase from `94/156` to
  `96/156` for MSE, while derivative wins are `111/156` and `109/156`.

The remaining differences are small but visible: replacement at densities
`0.10--0.40` gives back about `0.02--0.06%` MSE relative to the retained law,
and salt--pepper `0.10` gives back `0.09%`, while salt--pepper `0.25`, uniform,
Laplace, and most mixed cases improve or remain essentially tied.  This is a
research promotion, not a claim that the 1-D problem is complete.  It is the
first candidate here whose evolving uncertainty about transport produces a
measurable structural lift without a material aggregate denoising loss.
Exact record: `1d_phase_collision_posterior_harmonic_broad128_2seed.json`.

Matched resolution screens preserve the same aggregate pattern.  At 96
samples, clean MSE improves `0.04%` over retained while noisy MSE gives back
`0.08%`; the only material structural ratio is a roughly 5% derivative rise
on the under-resolved smooth step, localized to two edges with absolute error
around `5e-6`.  At 160 samples, clean MSE improves `0.10%`, noisy MSE gives
back `0.04%`, and smooth geometry improves `5.1%` MSE and `2.9%` first-
difference error; pulse derivatives rise about `0.8--1.0%`.  The law is not
resolution-closed, but its structure/noise balance is not a 128-only effect.
Records: `1d_phase_posterior_resolution96.json` and
`1d_phase_posterior_resolution160.json`.

A three-seed replication at the harder 96-sample resolution shows that the
one-seed noisy deficit was mostly realization variance.  Relative to the
retained Newton/phase law, clean MSE improves `0.026%`; pooled noisy MSE is
only `0.0094%` higher, first-difference error `0.0007%` higher, and second-
difference error `0.0021%` lower.  Uniform and mixed corruption improve;
the repeatable tail is concentrated in replacement (`+0.049%` MSE) and
salt--pepper 0.10 (`+0.052%`).  This sharpens the open problem: sparse foreign
samples remain a small transport-uncertainty error, not evidence for a broad
resolution failure.  Exact record:
`1d_phase_posterior_resolution96_3seed.json`.

The matched 160-sample, three-seed replication gives the same verdict.  Clean
MSE improves `0.082%` over retained; pooled noisy MSE is `0.0097%` higher and
first/second-difference errors are only `0.0088%/0.0081%` higher.  Replacement
again supplies the clearest tail (`+0.053%` MSE).  Salt--pepper gives back
`0.034%` MSE while improving both derivative errors.  The phase-posterior law
is therefore scale-stable at this gate; the unresolved defect is sparse-source
topology, not a missing run length or a resolution-specific smoothing rate.
Exact record: `1d_phase_posterior_resolution160_3seed.json`.

The 160-sample pulse derivative loss is spatially sparse rather than a global
softening: `99.3%` of its positive first-difference cost is concentrated on
six edges, chiefly the flanks of the three genuine pulses.  This suggested
conditioning the ancestry collision on target phase fidelity `F`, the
spherical agreement between transported and local whitened connection
directions.  Both natural joint laws are rejected.  Treating target agreement
as two independent events, `R^2 F^2`, reduces the 160-sample pulse derivative
errors by `0.18--0.29%` but raises clean pulse MSE `0.08%`, damages clean
chirp/mixed structure, and worsens sparse-salt derivatives.  The correct
single-observation conditioning `R^2 F` shows the same trade: pulse derivative
errors fall `0.13--0.21%`, while clean pulse MSE rises `0.05%`, 128-sample
chirp/mixed errors rise `0.02--0.04%`, and salt derivatives worsen.  Local
phase fidelity therefore diagnoses genuine turns as well as unsupported
transport.  It cannot be an ownership gate without erasing the distinction
the estimator is meant to preserve, and no fidelity switch is retained.

The V3 support construction gives a more discriminating diagnostic.  Push
each complete ancestry branch's posterior mass equally onto its two exact
sources, and repeat the pushforward with the branch reference measure.  This
produces a posterior source-incidence degree `d_x` and a purely topological
opportunity degree `h_x`.  Their symmetric commensurability

\[
F_{\mathrm{support}}(x)=\frac{2d_xh_x}{d_x^2+h_x^2}
\]

selects no source, scale, or support.  Across 96, 128, and 160 samples, the
raw ratio `d_x/h_x` has median `0.08--0.13` on salt samples and `0.21--0.33`
on replacement samples, versus roughly `0.6--1.2` on ordinary samples;
absolute corruption and source support correlate between `-0.51` and
`-0.71`.  Thus foreign samples really are low-degree vertices in transport
itself, not merely large residuals.

But scalarizing this graph back into an ownership multiplier is rejected.
Direct support conditioning improves most sparse-noise and pulse-derivative
cases, yet worsens clean smooth geometry about `0.72%`.  The probabilistic
union of support commensurability with spherical phase collision cuts that
penalty to `0.23%` and retains the sparse improvements, but still spends clean
structure.  This precisely matches the segmenter lesson: support is the
transported proposal topology, not a confidence score.  The source-incidence
fidelity is retained as a diagnostic; the next estimator must evolve the
support connection itself rather than multiply by it.

A cavity-collision readout provides the opposite control.  Removing the local
likelihood from the smoothed ancestry law and colliding only the target-free
transport reduces some salt derivative error, but raises clean pulse MSE by
`32--114%` and clean chirp/mixed errors by `325--609%`.  The observation cannot
be discarded merely because the surrounding transport doubts it: that is the
mechanism by which smoothing obliterates structure that noise had not actually
concealed.
