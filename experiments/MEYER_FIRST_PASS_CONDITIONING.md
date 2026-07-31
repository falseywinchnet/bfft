# Efficient conditioning of Meyer's first pass

## The defect

With zero Split-Bregman state, the first cartoon emitted by the native
periodic solver is exactly

\[
  (cI-\eta\Delta)u_0=cf,
  \qquad c=\lambda,\quad \eta=2\lambda.
\]

The nonlinear shrink happens after this solve and only affects pass two.
Consequently the visible one-pass result cannot distinguish a contour from a
texture carrier at the same spatial frequency. It is a uniform linear filter.

## Conditioned equation

Predict the reflected Split-Bregman field directly from the unchanged source,
then admit it only where a frozen symmetric-support statistic says that the
variation is structural:

\[
  (cI-\eta\Delta)u_1
  =cf-\eta\operatorname{div}\left(
      1.5\,g_{\rm TSV}^{6}\,R_{1/\eta}(\nabla f)
    \right).
\]

Here `R` is the same radial reflected shrink used by the C++ kernel. The sixth
power is not cosmetic: an ordinary TSV gate improves smooth carriers but
mistakes a substantial fraction of a hard checker carrier for structure. The
high-certainty tail preserves the improvement on all three known-truth rigs.

This is the useful role of the eikonal geometry: provide a frozen normal flux.
It does **not** require a fast-marching solve. Symmetric cancellation supplies
the capacity of that flux; the existing screened solve transports its
divergence globally.

## Truth policy and rejected scalar ranking

Validation now uses only three analytically authored sources: smooth tapered
carriers crossing known shapes, an independently authored multiscale crossing,
and a hard checker carrier. Every cartoon sample, texture sample, phase,
amplitude, contour mask, and texture-interior mask is known exactly. No
photograph or inherited gallery rig is quality evidence.

An early scalar objective selected a fixed conditioner because it strongly
reduced contour leakage. The images and the separate truth coordinates reject
that ranking: contour purity improved while carrier completeness remained near
ordinary pass one. No scalar winner is claimed.

## Cost

The Gaussian TSV implementation shares one source real FFT across four
directions and caches the four derivative/convolution symbols. It needs one
forward real FFT and four inverse real FFTs. In the native kernel the source
spectrum already exists, so even that forward transform can be reused.

The current NumPy prototype measures about 2.1 ms for the uniform first split
and 4.5 ms for the complete conditioned split at 256 x 256. The rectangular
telescoping-chord alternative is O(N), but in the current Python prototype is
not faster or sufficiently selective.

## Files

* `experiments/meyer_first_pass_conditioning.py`
* `experiments/out/meyer_first_pass_conditioning/results.json`
* `tests/meyer_first_pass_conditioning_test.py`

The ordinary production Meyer entry points remain unchanged; the native
conditioner below is strictly opt-in.

## Native opt-in result

The follow-up native implementation is exposed as
`MeyerPlan.split_conditioned_first` and `bfft.meyer_split_conditioned_first`.
It uses the analytic Gaussian symbol directly in the library's split 2-D
spectrum, reuses `f_spec`, and uses a mean-derived gate scale rather than a
percentile selection.

At 256 x 256 with four native lanes:

| Scene | arm | ms | texture gain | interior error | contour excess |
|---|---|---:|---:|---:|---:|
| smooth support | pass 1 / conditioned / pass 64 | 0.275 / 1.221 / 21.254 | 0.545 / 0.548 / 1.000 | 0.455 / 0.452 / 0.058 | 0.618 / 0.117 / 0.642 |
| multiscale crossing | pass 1 / conditioned / pass 64 | 0.272 / 1.206 / 21.187 | 0.290 / 0.316 / 0.931 | 0.756 / 0.731 / 0.096 | 0.799 / 0.446 / 0.716 |
| checker support | pass 1 / conditioned / pass 64 | 0.304 / 1.245 / 21.205 | 0.705 / 0.687 / 1.000 | 0.316 / 0.333 / 0.043 | 0.479 / 0.079 / 0.498 |

That scalar ranking is misleading. It rewards contour purity enough to hide
texture under-extraction. On the multiscale crossing the conditioned pass
recovers only 0.316 of the interior texture amplitude with 0.731 relative
interior error, while pass 64 recovers 0.931 with 0.096 error. The conditioner leaves
carrier texture visibly embedded in the cartoon. Pass 64 has the opposite
defect: much better texture completeness but greater contour leakage. The
conditioned pass is therefore **inferior as a complete Meyer replacement**;
it is only a useful structural-flux experiment.

The native result matches an independent NumPy full-complex spectral construction to below
`2e-4` absolute sample error and is bit-identical between one and four native
lanes. Strength zero is exactly ordinary pass one.
