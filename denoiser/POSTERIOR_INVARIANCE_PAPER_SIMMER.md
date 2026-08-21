# Posterior invariance simmer: statistics without neural architecture

## Result

The papers occupy the same mathematical space as this denoiser, but their
neural architectures do not need to come with them. Their shared useful object
is an observation-law-aware posterior trajectory: signal and nuisance are
represented together, nuisance geometry evolves, and candidate transport must
agree across independent views, scale, and transport depth.

Two immediate experiments were run. Using residual reflection as positive
posterior authority was decisively rejected. A second probe exposed a more
important missing state: the current residual posterior does not carry the
variance and correlation geometry of signal-dependent nuisance. The next
implementation should be a cross-fitted posterior over conservative
Selling-edge fluxes in a continually estimated canonical noise coordinate.

This is a research synthesis, not a claim that the cited model architectures
or their application-specific assumptions have been reproduced.

## What transfers, and what does not

| source | architecture not imported | transferable statistical principle | transport translation |
|---|---|---|---|
| Poisson2Poisson-Sparse | unfolded sparse network, patch dictionary, named Poisson runtime branch | likelihood geometry must follow signal-dependent variance | infer a continuous local variance law and transport in its canonical coordinate |
| Moments Matter | log-network | an intensity-space posterior mean omits covariance and higher moments; for Poisson data these are derivatives of the posterior log-mean | transport posterior jets in canonical coordinate, not only scalar radiance |
| Poisson2Sparse | convolutional dictionary and ISTA unrolling | Poisson fidelity is `1^T x-y^T log x`; positivity and the observation law precede regularization | make fidelity observation-law aware while keeping the geometric operator independent of a noise label |
| COSDD | blind-spot neural encoder/AR decoder | separation comes from making the nuisance model structurally incapable of explaining the full 2-D signal | restrict each nuisance chart to held-out directional ancestry; require another chart to authorize removal |
| Positive2Negative | learned denoiser and symmetric-noise training scheme | hold signal fixed, alter the nuisance realization, and demand consistency | residual reflection is a falsifier, not posterior mass |
| Next-Scale Prediction | multiscale neural predictor | noise decorrelation and detail preservation require distinct source and target supports | compare disjoint transported scales; never generate evidence and target from one smoothed chart |
| Gaussian-integral Bayesian smoother | polynomial Gaussian quadrature implementation | filtering forward is not smoothing; later evidence revises earlier states through cross-covariance | run a backward posterior pass over transport depth |
| PPFM/PFCM | learned Poisson flow/consistency model and augmented-space hyperparameters | a posterior is a trajectory; points on one valid trajectory have one endpoint | endpoint/depth consistency vetoes operator mismatch and belongs in uncertainty about transport itself |

The unification is not “use Poisson when Poisson is detected.” Noise names are
experimental coordinates. Runtime state instead carries a smooth local
variance function `V_t(x)`, directional correlation, and uncertainty in those
quantities. Gaussian and Poisson become limiting charts of one observation
geometry.

## Canonical noise geometry

For a mean-parameterized exponential-dispersion observation law with variance
function `V(x)`, the canonical and Fisher-arclength coordinates satisfy

\[
 {d\eta\over dx}={1\over V(x)},\qquad
 {dz\over dx}={1\over\sqrt{V(x)}}.
\]

For constant variance these are affine coordinates. For Poisson variance
`V(x)=x`, they become `eta=log x` and `z=2 sqrt(x)` up to constants. The
Poisson papers therefore do not merely recommend a different loss: they show
why an intensity-space residual mean cannot be the universal posterior state.

The denoiser should estimate `V_t` from target-excluded residual witnesses,
smoothly over radiance and transport depth. This is an inference suggested by
the papers, not a theorem they state. It removes the named-noise branch while
retaining the correct continuum:

\[
 V_t(x)=\operatorname{CrossFitMoment}_2(r\mid x,t),
 \qquad
 \eta_t(x)=\int^x {ds\over V_t(s)}.
\]

Directional residual covariance supplies the nuisance component of the
inverse metric. Structure supplies an independently witnessed component.
Their uncertainty, including uncertainty in `V_t` and the metric itself, is
part of the state rather than a fixed tuning constant.

## The operator that follows

Let `e=(i,j)` be a Selling edge of the positive decomposition of the inverse
evolving structure/noise metric. The primitive act remains a conservative,
antisymmetric nearest-neighbor flux,

\[
 J_{e,t}=c_{e,t}\,[\eta_t(u_j)-\eta_t(u_i)],\qquad
 u_i^{t+1}=u_i^t+\Delta t_t\sum_{e\sim i}J_{e,t},
 \qquad J_{ji,t}=-J_{ij,t}.
\]

`c_e,t` is not a tuned edge-stopping function. It is posterior capacity for
the hypothesis that the discrepancy can be transported as nuisance. It may be
reduced by four independent vetoes:

1. target-excluded directional ancestry does not reproduce the discrepancy;
2. a disjoint scale view does not preserve the proposed endpoint;
3. a changed nuisance realization maps to a different endpoint;
4. forward and backward depth posteriors disagree beyond their transported
   cross-covariance.

None is positive evidence alone. In particular, an estimator cannot create a
counterfactual and then use its own agreement as proof. Positive capacity
requires agreement between genuinely disjoint source lineages. The stable time
step remains derived from the Selling row sum; it is not a quality control.

After the forward trajectory `(u_t, Sigma_t, M_t)` is recorded, a backward
smoothing pass revises it schematically as

\[
 \bar u_t=u_t+C_{t,t+1}C_{t+1,t+1}^{-1}
 (\bar u_{t+1}-u_{t+1}^{\mathrm{pred}}),
\]

with the same operation applied to observation-law and metric uncertainty.
This is the non-neural lesson from Bayesian smoothing: later transport evidence
must be able to revise earlier geometric commitments.

## Experiment 1: reflection is a veto, not authority

The base estimate `x=D(y)` defines `r=y-x`; the counterfactual `y-=x-r`
reverses the residual while holding the candidate signal fixed. The tested
authority was

\[
 a={r^2\over r^2+(D(y^-)-x)^2+\epsilon},\qquad
 \hat x=y-ar.
\]

It preserves clean identity and edges, but it preserves noise for the same
reason. On the focused 128-pixel Cameraman gate:

| corruption | residual posterior MSE / SSIM / edge | reflected authority MSE / SSIM / edge |
|---|---:|---:|
| uniform .12 | .00227 / .571 / .869 | .00317 / .500 / .938 |
| Gaussian .12 | .00570 / .401 / .827 | .00890 / .337 / .914 |
| replacement .15 | .01166 / .343 / .761 | .01825 / .288 / .827 |
| salt-pepper .15 | .02256 / .244 / .789 | .03686 / .198 / .856 |
| mixed .12/.15 | .01312 / .302 / .736 | .02071 / .250 / .805 |

Across the 54-case six-source gate, reflection raises aggregate MSE from
`.00957` to `.01366`, lowers SSIM from `.5394` to `.4938`, and wins no SSIM
case. Edge retention rises from `.7147` to `.8231`, showing exactly what went
wrong: residual persistence was mistaken for structural truth. The probe is
retained as a falsified control and diagnostic.

## Experiment 2: observation geometry is missing

One fixed estimator was tested on three sources under photon counting and
signal-dependent row-correlated nuisance. Generator labels and levels are
known only to the benchmark.

| condition | observation MSE / SSIM / edge | screened transport | residual posterior |
|---|---:|---:|---:|
| Poisson exposure 8 | .04626 / .207 / .922 | .00859 / .407 / .283 | .01989 / .293 / .680 |
| Poisson exposure 32 | .01382 / .393 / .979 | .00452 / .576 / .446 | .00653 / .496 / .742 |
| row-correlated signal-dependent .08 | .00295 / .689 / 1.000 | .00330 / .665 / .626 | .00288 / .680 / .946 |
| row-correlated signal-dependent .15 | .01009 / .479 / .992 | .00668 / .495 / .551 | .01009 / .479 / .992 |

At strong row-correlated nuisance the residual posterior admits no packet and
returns the observation exactly. At low photon exposure it reduces noise but
retains excess variance (`1.63` times truth). Screened FMMT lowers MSE by
obliterating fine structure, retaining only `.283` of strong-edge magnitude at
exposure 8. Neither is the desired solution.

Reflection authority averages about `.47` in both Poisson cases despite its
large truth error. This independently confirms that same-operator reflection
agreement is not calibrated posterior probability.

## Next executable posterior

The next implementation should be one coherent state, not another wrapper:

1. construct mutually target-excluded directional/scale witness charts;
2. estimate a continuous variance function and directional residual covariance
   in each chart;
3. form the Selling decomposition of the joint structure/noise inverse metric;
4. transport conservative canonical-coordinate edge flux and its posterior
   moments;
5. record uncertainty in flux, variance law, and metric during the forward
   trajectory;
6. reconcile that trajectory backward through transported cross-covariance;
7. stop at action contraction plus endpoint consistency, with consistency used
   only to reject unsupported flux.

This addresses the present failures without patches, bands, noise
classification, denoising-strength controls, or a neural model. It also gives
a clean later route to C++: fixed small Selling stencils, fused edge-flux and
moment kernels, structure-of-arrays posterior state, and one reverse traversal
of the stored depth trajectory. Native optimization remains premature until
this posterior earns promotion.

## Reproducible artifacts

- `reflection_consistent_posterior_2d.py` and
  `test_reflection_consistent_posterior_2d.py`
- `reflection_posterior_128/results.json` and its full/ROI image panels
- `paper_consistency_six_source_32.json`
- `probe_nuisance_geometry_2d.py`,
  `test_probe_nuisance_geometry_2d.py`, and `nuisance_geometry_40.json`
