# Complete-moment transport simmer

## Decision

The paper-derived idea survives, but in a narrower and more interesting form
than a variance-stabilizing pretransform. The useful invariant is the
zero-referenced complete residual moment

\[
\mathbb E[R^2]=\operatorname{Var}(R)+(\mathbb E R)^2.
\]

The old residual posterior transported both mean and variance, then allowed
only the central variance to shape its Selling metric. A residual component
shared by every directional witness therefore vanished from metric
uncertainty precisely because it was a common mean. That is disastrous for
photon noise and other signal-dependent nuisance: a shared target fluctuation
looks geometrically certain.

Letting the complete second moment shape the metric improves the old posterior
on the focused Cameraman gate, the six-source corruption gate, and the new
photon/correlated-noise gate. The same correction also improves the newer
causal Hamilton--Jacobi lineage law when inserted into its joint branch metric.
This is the active result. Coordinate transforms, a locally inferred
anisotropic tensor, a diagonal backward smoother, and removal of the phase
constraint were all informative failures.

Nothing here is promoted to the GUI or C++.

## Experiment A: observation coordinates alone

A target-excluded directional law was fitted as

\[
V(s)=a+bs+cs^2,\qquad a,b,c\geq0,
\]

with coherent phase contributing vanishing fitting measure. The unchanged
residual posterior was then evaluated either in Fisher arclength

\[
z(x)=\int^x {ds\over\sqrt{V(s)}}
\]

or canonical coordinate

\[
\eta(x)=\int^x {ds\over V(s)}.
\]

The coordinate maps are monotone, invertible on the observed support, and use
the observed sample values as their only quadrature. Fisher coordinates give
a small SSIM improvement at Poisson exposure 32 (`.4955 -> .5128`) but do not
improve low-exposure Poisson MSE and do not unfreeze correlated noise.
Canonical coordinates add measurable negative bias. The pretransform is
therefore rejected as a denoiser. Observation geometry must enter the
posterior state and metric, not wrap them.

## Experiment B: complete moment in the continual Selling metric

The accepted change is exact and small. With directional residual centre
`mu_t` and central dispersion `v_t`, replace

\[
M_t=M(u_t,v_t)
\]

by

\[
M_t=M(u_t,v_t+\mu_t^2).
\]

The positive Selling decomposition, conservative Markov step, residual
mixture, physical-time fusion, contractor, and numerical stopping law remain
unchanged. There is no new parameter or noise label.

### Focused 128-pixel Cameraman

MSE / SSIM / strong-edge retention are:

| corruption | central-moment posterior | complete-moment posterior |
|---|---:|---:|
| clean | 0 / 1 / 1 | 0 / 1 / 1 |
| uniform .12 | .002265 / .5711 / .8693 | .002114 / .5922 / .8493 |
| Gaussian .12 | .005696 / .4010 / .8274 | .005125 / .4199 / .8105 |
| Laplace .10 | .007018 / .3760 / .8550 | .006247 / .3949 / .8373 |
| replacement .15 | .011664 / .3426 / .7605 | .010271 / .3610 / .7422 |
| salt-pepper .15 | .022559 / .2438 / .7893 | .019478 / .2593 / .7715 |
| mixed .12/.15 | .013122 / .3016 / .7357 | .011359 / .3225 / .7169 |

It improves MSE and SSIM in every corrupted focused case while giving up a
small amount of edge magnitude. The strong edges remain substantially less
attenuated than FMMT in the additive cases.

### Six-source, 54-case gate

| form | MSE | SSIM | variance ratio | range ratio | edge retention |
|---|---:|---:|---:|---:|---:|
| central moment | .009575 | .5394 | 1.0148 | 1.0118 | .7147 |
| complete moment | **.009047** | **.5456** | **.9843** | **.9975** | .7019 |
| FMMT control | .006487 | .7245 | .7632 | .8801 | .5607 |

The complete moment also improves the six clean-source aggregate from
`.000577/.9628/.8349` to `.000533/.9642/.8499` for MSE/SSIM/edge. It is not a
universal pointwise win: uniform `.10` and low-density replacement regress,
while Gaussian, Laplace, high-density replacement, salt-pepper, and both mixed
conditions improve in aggregate.

### Photon and correlated nuisance

| condition | central MSE / SSIM / edge | complete MSE / SSIM / edge |
|---|---:|---:|
| Poisson exposure 8 | .01989 / .2927 / .680 | **.01717 / .3124** / .647 |
| Poisson exposure 32 | .00653 / .4955 / .742 | **.00584 / .5159** / .710 |
| row-correlated signal-dependent .08 | .00288 / .6800 / .946 | **.00277** / .6734 / .883 |
| row-correlated signal-dependent .15 | .01009 / .4787 / .992 | unchanged observation |

At exposure 8 the truth-variance ratio moves from `1.63` to `1.47`; at
exposure 32 it moves from `1.12` to `1.07`. This is still under-denoised, but
the direction is correct and arises without knowing that the generator was
Poisson.

## Experiment C: failures that located the next state

### Locally inferred anisotropic nuisance

The gradient outer product of the transported residual centre was added to an
anisotropic nuisance tensor, gated by phase-incoherence, before structure was
projected back onto the PSD cone. This loses part of the complete-moment gain
and does not unfreeze strong row-correlated noise. Local residual geometry is
still generated and judged by the same observation; it is not independent
evidence. Rejected.

### Diagonal backward smoother

A one-transition RTS-style marginal used

\[
K_t=\operatorname{diag}(P_tA_t^T)
     /\operatorname{diag}(A_tP_tA_t^T)
\]

to reconcile the filtered endpoint back to depth zero. It preserves more edge
energy but is far too timid on Poisson noise: aggregate MSE becomes `.01519`
versus `.00897` for the forward complete-moment posterior. A diagonal
covariance cannot carry causal branch identity. Rejected; backward smoothing
must occur on the full transported lineage law.

### Removing phase coherence from the contractor

Strong row-correlated noise is rejected because coherent-residual penalty,
not bounded distribution mismatch, dominates the action. Removing that term
unfreezes the estimator and produces low Poisson MSE, but collapses edge
retention to `.247` in aggregate and `.219` under strong row-correlated noise.
This proves both sides:

- phase coherence cannot universally mean structure;
- bounded residual feasibility cannot protect structure by itself.

The next state must retain coherent-structure and coherent-nuisance
hypotheses, then distinguish them through causal ancestry rather than local
phase shape.

## Experiment D: complete residual moments in the causal HJ metric

The newer causal lineage law transports branches with joint coordinate

\[
q=(z,j_y,j_x,r).
\]

Its determinant-one precision used centered covariance, making the residual
coordinate translation invariant. This silently erased a residual mean
shared by a parent/child branch family. The experiment changes only

\[
\Sigma_{rr}\leftarrow\Sigma_{rr}+\mu_r^2
\]

inside each parallel-transported parent/child branch kernel. Signal and jet
coordinates remain gauge-centered; residual zero is physical and is therefore
not a gauge. The Hopf--Lax forest, branch Haar measure, simplex collision
order, and population-phase integration remain unchanged.

On the first 18-case, three-source, six-condition screen at 16 pixels and two
population phases, complete residual moments improve MSE in 13 cases, SSIM in
13, and edge retention in 9. Aggregate HJ-simplex metrics change from
`.006819/.6371/.5324` to `.006729/.6401/.5296`. The improvement is strongest
in the intended regimes:

- Poisson exposure 16: MSE `.00955 -> .00919`, SSIM `.5154 -> .5245`;
- row-correlated signal-dependent `.15`: MSE `.01109 -> .01098`, with edge
  retention `.5955 -> .5992`;
- replacement `.25`: MSE `.005144 -> .005103`, SSIM `.7083 -> .7146`.

Clean MSE regresses slightly (`.002901 -> .002965`). Thus the zero-referenced
moment belongs inside the modern transport, but its authority still needs the
zero-noise atom and causal source identity explicitly represented rather than
implicitly applied to every branch family.

The six-source confirmation contains 36 cases at the same resolution and two
population phases. Complete residual moments improve MSE in 23 cases, SSIM in
25, and edge retention in 17. Aggregate central/complete/FMMT metrics are:

| form | MSE | SSIM | variance ratio | edge retention |
|---|---:|---:|---:|---:|
| centered-residual HJ | .008030 | .6633 | .7876 | .5349 |
| complete-residual HJ | **.007937** | **.6663** | .7804 | .5328 |
| FMMT control | .007806 | .6965 | .7812 | .5595 |

The complete HJ law is now close to FMMT in aggregate MSE and beats it on MSE
under Poisson (`.00992` versus `.01119`), replacement (`.00809` versus
`.00874`), and mixed corruption (`.01011` versus `.01054`). Under Poisson it
also retains `.528` edge magnitude versus FMMT's `.425`. FMMT still leads
aggregate SSIM and edges, and strong correlated nuisance remains unresolved.
Across six clean sources complete moments slightly improve MSE but lower SSIM
and edge retention; clean structural authority is the specific next debt.

The displacement rerun confirms that this is a modest reweighting, not a new
reconstruction regime. Across the same 36 cases, centered and complete HJ move
from the observation by RMS `.12387` and `.12397`; they move 82.20% and 82.12%
of pixels by more than one 8-bit level. Aggregate motion is virtually
unchanged even where the local reweighting is useful. See
`complete_moment_lineage_36_phase2_displacement.json`.

## Next experiment

Do not choose between phase-constrained and bounded smoothing, and do not
average them. Lift the residual law in the causal state to two explicit
components:

\[
(b,z,j,r,a,\nu),\qquad \nu\in\{r=0,\ r\neq0\},
\]

where `nu` is a numerical mixture component, not a named noise class. Transport
its complete moments and root ancestry through the same Hopf--Lax parent
simplexes. Coherent residual may be removed only when distinct causal roots
agree on its nuisance component; phase may veto a proposed collapse but may
not declare coherent residual structural by itself. A backward pass should
then smooth this full branch/component law, not a diagonal pixel covariance.

## Artifacts

- `canonical_variance_transport_2d.py`
- `backward_moment_smoother_2d.py`
- complete/anisotropic/bounded variants in
  `continual_fabada_eikonal_2d.py`
- complete-residual branch option in `causal_information_lineage_2d.py`
- `probe_nuisance_geometry_2d.py`
- `probe_complete_moment_lineage_2d.py`
- `complete_moment_128/results.json`
- `complete_moment_six_source_32.json`
- `complete_moment_lineage_16_phase2.json`
- `complete_moment_lineage_36_phase2.json`
- `complete_moment_lineage_36_phase2_displacement.json`
