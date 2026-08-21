# Transported residual posterior on the positive eikonal flow

## Result

The failed smoothing-depth barycenter has been replaced by a posterior over
the *residual law*.  This is a real advance, but not yet a promotion candidate.
It preserves the pure positive averaging hierarchy, restores exact clean
identity on the focused 128-pixel Cameraman, and materially improves the
additive-noise trade over the first FABADA-order path posterior.  It still
underperforms the screened checkpoint in aggregate and remains poor on sparse
replacement corruption.

## Coupled law

Let `L_t` be the symmetric positive Selling reduction of the inverse evolving
structure/noise metric.  The only geometric smoothing act is

\[
 A_t=I-\Delta t_tL_t,\qquad
 \Delta t_t={1\over2d_{\max,t}},\qquad
 u_{t+1}=A_tu_t.
\]

The step is positive, symmetric, conservative, range preserving, and lowers
the frozen Dirichlet action.  The target-excluded directional witnesses yield
a transported bounded residual law `(mu_t,v_t,h_t)`.  Its nonzero-hypothesis
mass is the parameter-free mixture agreement

\[
 \pi_t={\mu_t^2\over\mu_t^2+v_t+\epsilon_{\rm mach}}.
\]

The exact zero-noise atom and the residual branch form one Bernoulli mixture.
Its complete first two moments are

\[
 m_t=\pi_t\mu_t,
 \qquad
 s_t^2=\pi_tv_t+\pi_t(1-\pi_t)\mu_t^2.
\]

The second term is essential: uncertainty between “no residual” and “noise
residual” must not be mistaken for repeated-data precision.  The outer radius
is the smallest centred radius containing zero and both endpoints of the
bounded residual branch.

The existing posterior residual law is first transported by `A_t`.  The new
branch is then fused over the exact physical-time fraction

\[
 q_t=1-e^{-\Delta t_t}
\]

using the law of total variance.  The operator-split radiance proposal is

\[
 x_{t+1}=A_tu_t-\bar m_{t+1}.
\]

Thus averaging supplies geometric flow and the posterior residual mean
supplies a contemporaneous source term.  There is no screened solve and no
user-selected denoising duration.  The full packet is admitted only when the
joint bounded-residual/phase-Sasaki contractor decreases.  Phase is retained
as an independent structural falsifier; multiplying it into `pi_t` was tested
and rejected because it counted the same uncertainty twice and nearly froze
the estimator.

## Focused 128-pixel gate

Metrics are MSE / SSIM / strong-edge retention / tripod retention.

| corruption | residual posterior | screened phase-eikonal | first path posterior |
|---|---:|---:|---:|
| clean | 0 / 1 / 1 / 1 | 0 / 1 / 1 / 1 | .000046 / .9970 / .950 / .937 |
| uniform .12 | .002265 / .5711 / .869 / .806 | .002104 / .5990 / .840 / .767 | .003159 / .4987 / .917 / .894 |
| Gaussian .12 | .005696 / .4010 / .827 / .773 | .005232 / .4148 / .752 / .669 | .007644 / .3535 / .840 / .812 |
| Laplace .10 | .007018 / .3760 / .855 / .793 | .005799 / .4044 / .778 / .691 | .008429 / .3467 / .851 / .812 |
| replacement .15 | .011664 / .3426 / .761 / .702 | .008458 / .3956 / .747 / .672 | .009315 / .3841 / .710 / .658 |
| salt-pepper .15 | .022559 / .2438 / .789 / .725 | .015445 / .2922 / .779 / .699 | .016614 / .2859 / .744 / .679 |
| mixed .12/.15 | .013122 / .3016 / .736 / .694 | .010703 / .3286 / .729 / .674 | .012381 / .3062 / .698 / .662 |

The posterior now occupies the intended middle ground.  On uniform and
Gaussian noise it gives back much of the first path's excess noise while
retaining more Cameraman and tripod geometry than the screened form.  Sparse
corruption exposes the present defect: a globally admitted linear averaging
packet spreads replacement damage before the residual posterior can localize
its support.

## Six-source gate

Across six structures and nine unnamed corruption states at 32 pixels:

| method | MSE | SSIM | variance ratio | range ratio | edge retention |
|---|---:|---:|---:|---:|---:|
| residual posterior | .00957 | .5394 | 1.0148 | 1.0118 | .7147 |
| screened phase-eikonal | .00814 | .5576 | .9535 | .9870 | .6904 |
| first path posterior | .00999 | .5362 | 1.0446 | 1.0259 | .7349 |
| FMMT control | .00649 | .7245 | .7632 | .8801 | .5607 |

The aggregate is modest, but the condition split is informative:

- uniform `.10`: posterior `.00215/.7139/.7948` versus screened
  `.00301/.6425/.8792` for MSE/SSIM/edge retention;
- Gaussian `.10`: posterior `.00478/.5763/.7919` versus screened
  `.00471/.5800/.7426`;
- multiplicative `.12`: posterior `.00284/.7228/.8351` versus screened
  `.00279/.7376/.7430`.

The posterior therefore improves truth recovery under uniform corruption and
nearly matches screened error under Gaussian/multiplicative corruption while
retaining more structure.  Its advantage does not extend to replacement,
salt-pepper, or high-density mixed damage.

At 32 pixels its clean-source aggregate is
`.000577/.9628/.8349`, slightly better than screened
`.000708/.9554/.8137`, but neither is an exact universal clean fixed point.
The focused 128-pixel clean identity is exact.  This remaining resolution
dependence prevents promotion.

## Falsified posterior constructions

Three nearby constructions were run and rejected rather than hidden:

1. Multiplying agreement and phase authority inside the residual mean, then
   integrating from a zero law, was too timid and retained almost all diffuse
   noise.
2. Initializing the full residual branch and reading `y-E[N]` restored exact
   clean identity but rejected uniform and Gaussian transport entirely.  A
   residual mean alone is not a geometric displacement.
3. Continuous scalar minimization of the contractor along each transport
   chord lowered the declared action but worsened additive truth metrics and
   moved the clean image.  The contractor is a valid rejection gate, not a
   calibrated posterior likelihood.

The third result is especially important: optimizing the current action more
aggressively is not the next theory.

## Next posterior

The remaining failure is spatial allocation.  One global packet decision
forces diffuse-noise smoothing and sparse-damage localization to share a
single continuation event.  The next posterior should be a conservative
*local flux law*:

- posterior mass lives on Selling edges, not pixels or global iterations;
- each flux carries the transported probability that its exchanged radiance
  is residual noise rather than coherent structure;
- antisymmetric edge flux preserves global mass even when local continuation
  probabilities differ;
- phase-Sasaki evidence may veto a flux but cannot positively certify itself;
- the residual-law source term remains separate from geometric averaging.

That is the route to sparse replacement recovery without reintroducing bands,
patches, named noise classes, or a denoising-strength control.

Artifacts are `residual_posterior_128/results.json`,
`residual_posterior_six_source_32.json`, and the posterior invariants in
`test_continual_fabada_eikonal_2d.py`.
