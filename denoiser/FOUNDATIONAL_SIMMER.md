# Foundational simmer: relation transport before smoothing

## The correction

FABADA was not principled. Its recursive variance treats repeatedly reused data
as new information, and its chi-square interpretation ignores the correlations
created by diffusion. Nothing here rehabilitates that mathematics.

Its idealistic residue is narrower and useful: do not decide a smoothing depth
before the observation has expressed which structures survive across scale.
The present 1-D support experiment violates even that ideal. It first commits
to a Gaussian provisional chart and then transports that chart. On mixed
inputs, the second operation cannot recover relations erased by the first.

## The observation that changes the model

In replacement-plus-uniform corruption, substantial latent structure remains
visually and relationally present. The corruption changes or perturbs sample
atoms; it does not physically diffuse the latent curve. A smoother nevertheless
spreads each corrupted amplitude into its neighbors. It can therefore destroy
more structural information than the corruption concealed.

The foundational object should be a measure of *relations that predict other
observations*, not a field of intensities waiting to be averaged.

## First executable lift

For samples `(x_i,y_i)`, every local chord transports an affine jet:

\[
T_{a,b}(x)=y_a+\frac{x-a}{b-a}(y_b-y_a).
\]

At each target `x`, the push-forwards of bracketing and adjacent chords form an
empirical measure `mu_x`. Coherent latent structure places repeated mass near a
common value; uniform replacement produces dispersed proposals. The current
readout uses the continuous self-interaction density

\[
\rho_x(v)=\int\!\int
  \exp\!\left[-\frac{(v-u)^2}{2s^2}\right]
  \,d\mu_x(u)\,d\Lambda_x(s),
\qquad
\hat f(x)=\frac{\int v\rho_x(v)\,d\mu_x(v)}
                 {\int \rho_x(v)\,d\mu_x(v)}.
\]

`Lambda_x` is not a selected bandwidth. It is the empirical pair-distance
measure, integrated with the scale-invariant `ds/s` weight. The implementation
in `affine_relation_transport.py` is pointwise J-invariant: `y_i` never enters
the proposal measure used to reconstruct index `i`.

This is already qualitatively different from smoothing: values travel along
relations, and relation multiplicity supplies credibility. It is still only a
first simmer.

### First M4 result

Across four composites, replacement densities `0.10/0.25/0.40`, and three
seeds, the fixed-horizon affine pushforward reduced mean MSE to `0.00372`, from
`0.00686` for Gaussian sigma 2, `0.00727` for median width 5, and `0.00552` for
the current Gaussian-plus-support flow. That apparent win is not sufficient.
Its first-difference MSE (`0.00145`) is worse than the Gaussian control
(`0.000923`) and current flow (`0.000635`).

The clean oscillatory control is the decisive rejection. At lag 2 it retains
only `0.647` of truth total variation while noisy MSE remains `0.0139`; at lag
16 it lowers noisy MSE to `0.00510` but retains only `0.111` of clean total
variation. Thus relation consensus without transported jet order and scale can
be another disguised smoother. The complete matched record is
`1d_foundational_simmer.json`.

## Why it is not yet the unified law

A fixed long chord horizon helps heavy replacement corruption but can deform a
clean oscillatory composite. A short horizon preserves that composite but has
too little redundancy under heavy replacement. Choosing the horizon from a
truth-scored test would merely install another bad constant.

The missing state is relation scale itself. The intended unified state is a
measure on a jet-scale bundle,

\[
  \mathcal M_t(dx\,dv\,dj\,ds\,dc),
\]

where `x` is position, `v` value, `j` the transported jet, `s` relation scale,
and `c` continuous credibility. Signal, residual uncertainty, scale, and
support are marginals of this one state; they are not sequential modules.

For each disjoint observation lane, relation particles move by their jet
characteristics. Their scale credibility is paid for by prediction on the
other lane. A natural dimensionless action is the cross-predictive log score
under the transported empirical measure:

\[
 A_x(s)=-\log p_{-x,s}(y_x),
 \qquad
 d\Pi_x(s)\propto e^{-A_x(s)}\,\frac{ds}{s}.
\]

The reconstructed distribution is the scale marginal of `M`; a mean, median,
tail probability, or uncertainty interval is only a projection requested at
the end. There is no noise-family switch and no smoothing-time stop bolted on
afterward.

This equation is a research target, not an achievement claim. A valid next
prototype must demonstrate grid/scale refinement, clean-input near-identity,
mixed-input improvement, jump survival, and chirp/ripple survival before it can
replace the current 1-D laboratory path.

## Two separate paths

### 1-D foundation

1. Transport affine jets as the minimal relation state.
2. Add curvature jets only when affine predictive action fails under
   refinement—not as a named signal branch.
3. Make scale a transported marginal with cross-lane predictive likelihood.
4. Require clean near-identity and mixed-corruption recovery simultaneously.
5. Retain failed horizons and scale laws as negative evidence.

### 2-D representation speed

The image estimator is presently the stronger object and should not be changed
to make it fast. Profiling showed that the earlier scalar implementation of the
separable histogram bootstrap owned almost all runtime at 128 square. The same
recurrence is now advanced as vector packets. Random-state comparison is
bit-for-bit equal to the scalar reference. Ordered-front batches are enlarged
under an explicit memory cap; batch changes agree within floating-point summing
order.

On the recovered 256-square Cameraman observation, the continuous-support path
now ran in `1.34 s` on the M4 Mini with an adaptive batch of 256. The same
vectorized estimator with batch 16 took `1.96 s`; their maximum output
difference was `3.33e-16`. Earlier scalar-bootstrap measurements on this input
were about `25–26 s`. Runtime varies with host load, so the durable claims are
the stage profile, recurrence equivalence, and batch-invariance test—not a
single stopwatch ratio.

These are representation changes only. They do not introduce the provisional
blur into the final image, alter the FMMT posterior, or merge this denoiser with
the repository's other systems.

## Constants still on trial

The full FMMT audit remains in `CONSTANT_AUDIT.md`. For this new 1-D simmer:

| Item | Present status | Required resolution |
|---|---|---|
| maximum chord lag | numerical experimental horizon, currently about `sqrt(N)` | become a scale marginal with predictive survival |
| affine chord catalogue | bracketing plus adjacent extrapolating chords | derive as a quadrature of the local jet bundle |
| scale quadrature count | numerical integration resolution | demonstrate convergence |
| affine-only jet | deliberately minimal model state | admit curvature continuously through predictive action |
| boundary chord deficit | exposed in diagnostics | continuum-normalized boundary measure |

The M4 sweep is produced by `run_1d_foundational_simmer.py`. Its scores are
falsification evidence, not runtime parameters.

## The next simmer passed its first broad gate

`cross_predictive_transport.py` removes the fixed relation horizon. Every lag
to half the interval remains a state coordinate under `ds/s`, and three
minimal first-jet characteristics compete by reciprocal cross-predictive path
action. Predictable residual mass continues only while its covariance energy
exceeds its analytically estimated finite-sample variance.

Across six structures, thirteen corruption conditions, and three seeds, this
form achieved mean noisy MSE `0.002398`, versus `0.002684` for the fixed-horizon
affine simmer and `0.004898` for the legacy support flow. Clean variance and
total-variation retention were `0.967` and `0.933`. All 234 noisy runs reached
intrinsic equilibrium in at most two admitted continuations. The equations,
failure audit, and full table are in `1D_CROSS_PREDICTIVE_RESULT.md` and
`1d_cross_predictive_battery.json`.

This does not erase the earlier negative result. It explains it: scale needed
to remain a transported marginal, path multiplicity needed to remain present,
and continuation needed a debiased action equilibrium. Strictly ancestry-held-
out conductance and an explicit jet marginal remain unresolved.

## The first image lift locates the missing state

The four-tangent 2-D lift retains the complete directionwise scale marginal
and the same intrinsic continuation law. On the 108-case M4 screen it reaches
equilibrium and avoids variance, range, and mean collapse. It outperforms FMMT
on woven chirps and line drawings, yet loses sharply on hair, isolated
interfaces, and heavy replacement. The equations and table are in
`2D_CHARACTERISTIC_LIFT_RESULT.md`.

This split rejects the idea that the next repair is simply more smoothing or a
named impulsive-noise branch. Repeated relations create enough image-global
covariance to survive; sparse but causally coherent boundaries do not. The
missing coordinate is therefore distinct causal ancestry. V3 eikonal parent
fractions must become part of the transported measure before population and
continuation are evaluated.
