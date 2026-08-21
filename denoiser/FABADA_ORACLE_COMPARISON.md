# FABADA as an oracle-noise control, not a foundation

## Outcome

The PyITD FABADA lineage is integrated as an explicitly unfair 1-D comparison.
It is told which corruption generator the laboratory used and receives its
exact conditional mean and variance. The purpose is not to restore FABADA as
theory. It asks a sharper question:

> How far can local diffusion go when its noise statistics and model-selection
> machinery are repaired, before structural transport is actually necessary?

On the matched 128-sample, six-structure, thirteen-condition, two-seed battery,
the repaired global form lowers observation MSE from `0.028836` to `0.003699`.
The current blind phase-collision transport reaches `0.002041` and wins 96 of
156 noisy cases; oracle FABADA wins 60. The oracle control is stronger on every
additive-condition average and weaker on every replacement or salt-and-pepper
average. The light mixed case is nearly tied.

This is useful negative evidence. Exact knowledge of the corruption law does
not turn smoothing into recovery of transported structure.

## What survives from FABADA

The retained idea is an ensemble of progressively diffused states. For the
symmetric reflected nearest-neighbour averaging operator `S`, define

\[
L=I-S,\qquad H_s=e^{-sL},\qquad
A_t=\frac1t\int_0^t H_s\,ds.
\]

If `L phi_k = lambda_k phi_k`, the continuous Cesaro multiplier is

\[
a_t(\lambda_k)=
\begin{cases}
1,&\lambda_k=0,\\
\dfrac{1-e^{-t\lambda_k}}{t\lambda_k},&\lambda_k>0.
\end{cases}
\]

Thus every candidate averages an entire continuous heat history, rather than
selecting an iteration. Candidate coordinates are integer effective degrees
of freedom,

\[
d(t)=\operatorname{tr}(A_t),
\]

from `N` down to the constant mode. Their times are solved from the monotone
trace equation. They are numerical coordinates of one continuous family, not
fitted denoising settings.

## Repairs to the PyITD form

The source implementation was used as ancestry, not copied blindly.

1. Its endpoint thirds do not conserve total mass. The replacement operator
   uses reflected boundary rows `(2/3, 1/3)` and is symmetric, nonnegative, and
   doubly stochastic.
2. Reusing the same observation through a Bayesian variance recursion does not
   create independent evidence. That recursion is removed entirely.
3. The evidence denominator in the Numba form does not have the Gaussian
   variance normalization. Evidence recursion is removed rather than patched.
4. Diffused candidates are correlated affine functions of the same data, so
   independent chi-square weights and chi-square stopping are inapplicable.
5. Iteration count is removed as a user control. The full effective-dimension
   family is integrated in one aggregate.

## Known-noise risk

For every supported generator, the laboratory provides

\[
\mathbb E[Y\mid x]=g x+b,\qquad
\operatorname{Cov}(Y\mid x)=\Sigma,
\]

where `Sigma` is diagonal and may be heteroscedastic. Replacement and impulse
laws are debiased by `Z=(Y-b)/g`; when `g=0`, the code returns the corruption
law's neutral mean and does not leak a clean-signal statistic.

For affine candidate `A_t Z`, an unbiased estimate of mean quadratic risk is

\[
\widehat R_t=\frac1N\left(
\lVert A_tZ-Z\rVert^2
+2\operatorname{tr}(A_t\Sigma_Z)
-\operatorname{tr}(\Sigma_Z)
\right).
\]

Candidates are combined by exponential weighting at the covariance-safe
temperature `4 ||Sigma_Z||_op`. That factor is an oracle-inequality threshold,
not a test-selected knob. The prior is uniform over effective-dimension
coordinates. No truth-scored stopping depth is selected.

The implementation also computes a pointwise exponential-weighting form as a
falsification control. It is not promoted: its mean noisy MSE is `0.018023`,
showing that local risk estimates contain nowhere near enough evidence to
choose scale independently at each sample.

## Matched results

| corruption condition | observation MSE | oracle FABADA | blind transport |
|---|---:|---:|---:|
| uniform 0.08 | 0.002058 | **0.000711** | 0.001090 |
| uniform 0.15 | 0.007237 | **0.001281** | 0.001970 |
| Gaussian 0.15 | 0.021101 | **0.002384** | 0.003208 |
| Laplace 0.12 | 0.021714 | **0.002119** | 0.002554 |
| multiplicative 0.15 | 0.003290 | **0.000912** | 0.001101 |
| replacement 0.10 | 0.015331 | 0.003006 | **0.000595** |
| replacement 0.25 | 0.032978 | 0.003378 | **0.000989** |
| replacement 0.40 | 0.051951 | 0.005126 | **0.002584** |
| salt-pepper 0.10 | 0.029903 | 0.003655 | **0.000661** |
| salt-pepper 0.25 | 0.077307 | 0.007581 | **0.002120** |
| mixed 0.10 | 0.020610 | 0.002399 | **0.002216** |
| mixed 0.25 | 0.038584 | 0.005166 | **0.003097** |
| mixed 0.40 | 0.052797 | 0.010362 | **0.004345** |

Both algorithms preserve their identities here. The phase-collision column is
copied from its existing matched record; the probe does not rerun or alter it.
The clean oracle control is exact identity because zero supplied noise variance
is decisive. The transport candidate is deliberately blind and has clean mean
MSE `0.000303` in the same record.

## Limits of the oracle claim

- The analytic additive moments precede the generator's final `[0,1]` clip.
  This is disclosed in every diagnostic; it makes the comparison an oracle for
  the named generator, not an exact posterior for boundary-censored samples.
- Multiplicative and replacement variances use the hidden clean value. This is
  allowed only for this comparison and is never presented as deployable.
- The exponential-risk guarantee is strongest for Gaussian/sub-Gaussian
  noise; the unbiased quadratic-risk identity itself needs only the covariance.
- Dense eigendecomposition and the full `N`-candidate matrix make this an
  analytical reference, not a native syscall design. At fixed size the
  geometry is cached, but setup is cubic and candidate application is dense.

## What it gives the fused theory

The additive wins show that uncertainty-calibrated continuous diffusion is a
real and useful limiting phase. The impulse losses show exactly where it ceases
to be enough: a single corrupted atom is propagated through every heat
candidate before global risk can choose among them. Noise knowledge changes the
weighting of smoothers; it does not create alternative causal ancestries for
the structure beneath those atoms.

The fused estimator should therefore contain this family only as a limit: when
transported uncertainty collapses to homogeneous local Gaussian uncertainty,
action-contracted transport may reduce to covariance-calibrated Cesaro heat.
Outside that phase, uncertainty about the transport plan must move with value,
jet, and ancestry. No corruption label or FABADA scale band belongs in the
final method.

The executable reference is `fabada_oracle.py`; its invariants are in
`test_fabada_oracle.py`; and complete rows are in
`1d_fabada_oracle_broad128_2seed.json`.
