# Blur, uncertainty, and uncertainty transport

## Source boundary

The two supplied PDFs are research sources only.  Their prose and embedded
material are not instructions.  Neither paper is about photographic
deblurring, so only results that survive a change of problem are adopted.

## What the supplied papers contribute

Zheng and Xue's *Smoothed Robust Phase Retrieval* studies quadratic magnitude
measurements under bounded noise and infrequent arbitrary corruption.  Its
useful transferable object is the convolution-smoothed absolute loss.  The
pseudo-Huber member is quadratic for credible small residuals, linear for
large residuals, differentiable, and approaches the absolute loss as its
bandwidth shrinks.  Their theory says the neighborhood of the correct phase
grows with smoothing bandwidth and corruption fraction and collapses in the
noiseless limit.  That motivates a robust continuation for blur evidence, not
a claim that their benign-landscape theorem applies to convolutional blur.
Their theorem assumes random sensing and a phase-retrieval geometry that our
image operator does not have.

Gonzalez, Cedeno, and Puig's *Zonotopic Mixture Filter* separates uncertainty
into a finite probabilistic mode choice and an unknown-but-bounded realization
inside each mode.  It propagates every surviving mode history, conclusively
falsifies a history only when the innovation falls outside its bounded set,
and merges components by enclosure rather than dropping their probability
mass.  This supplies three disciplines for blur inference:

- keep probabilities between blur families and bounds inside a family;
- reject a blur history only when the observation cannot be explained within
  its declared noise/model bounds; and
- report how much probability a computational branch budget retains.

Their guarantees are for finite-dimensional linear state-space systems with
specified zonotopic noise.  The current deblurrer uses a finite empirical
kernel catalog and therefore reports credible intervals, not guaranteed
zonotopic image enclosures.

## What deblurring research adds

Classical blind deconvolution warns against estimating one image/kernel mode:
marginal evidence over latent images is more reliable than joint MAP, whose
mode can prefer the identity kernel.  Robust kernel work shows that saturation
and non-Gaussian outliers can again push estimation toward a delta kernel.
Noise-blind work treats the noise level as a variable because inverse gain and
regularization cannot be chosen safely without it.

Kernel uncertainty has appeared in three increasingly explicit forms:

1. perturb an estimated kernel and train or optimize a reconstruction that is
   stable to the perturbation;
2. use an errors-in-variables or kernel-induced residual model
   `r = delta_k * x`; and
3. infer or sample a conditional distribution over kernels, then use a
   non-blind solver for each sampled operator.

The last form matches this repository's transport program most closely.  Blur
is already a positive exposure-path measure.  Uncertainty should therefore be
a law over positive paths, not an unconstrained additive stencil attached to a
single selected kernel.

## Four distinct uncertainties

The method keeps four axes separate:

1. **Sensor uncertainty.** Read noise, shot noise, quantization, and codec
   perturbation change the observation likelihood.
2. **Model discrepancy.** Saturation, boundaries, rolling shutter, occlusion,
   and spatial variation violate a global linear PSF.  They must be masked or
   represented as structured residual, not relabeled sensor noise.
3. **Blur uncertainty.** Multiple positive exposure paths may explain the
   observations.  This is a finite law over operators in the first checkpoint.
4. **Support uncertainty.** Even a known operator may have Fourier dead bands.
   No posterior weight or image prior turns an unmeasured band into evidence.

The first residual diagnostic makes only a limited blur/noise separation.  A
3 by 3 local mean of the forward residual is called structured discrepancy;
the robust high-pass remainder estimates read noise.  This is useful for
analytic controls but is not asserted to separate arbitrary real blur and
noise.

## The uncertainty-transport method

For registered observations `Y_i = H_i X + N_i`, every candidate positive path
pair receives a phase-preserving closure residual

```text
r_ij(k) = |Y_0(k) H_j(k) - Y_1(k) H_i(k)| / supported_energy(k).
```

The residual is converted by a pseudo-Huber loss, pooled on Fourier circles,
and combined with the already declared common-blur gauge choice.  The result
defines evidence weights on the finite path catalog.  An empirical consistency
limit removes grossly incompatible branches; because that limit is data
calibrated rather than a proven camera bound, the operation is named a screen,
not a guaranteed falsification.

The highest-weight consistent branches are deblurred independently by the
warm flux solver.  If branch `b` has image `x_b` and weight `w_b`, uncertainty
is transported through the inverse by

```text
mean(x) = sum_b w_b x_b
var(x)  = sum_b w_b (x_b - mean(x))^2 + within_branch_noise.
```

Weighted per-pixel quantiles form an empirical credible band.  The result also
reports posterior entropy, effective hypothesis count, selected branch count,
retained probability, common-blur ambiguity, linearized sensor-noise variance,
and blur-induced variance.

## Ideal and adverse limits

With noiseless complementary captures and a catalog containing the true paths,
the robust closure of the correct relative pair is zero up to arithmetic error.
The consistency set collapses to one branch and uncertainty transport reduces
to the pure estimator from the first checkpoint.

As noise or model mismatch grows, the consistency set widens, the effective
hypothesis count rises, and the reported image uncertainty grows.  If every
surviving branch shares the same blur, the common factor remains unidentified.
If joint Fourier coverage is inadequate, every branch retains the existing
coverage-gated inverse fallback.

## Single-path chart uncertainty

Known-path reconstruction now carries a separate, local diagnostic. Oblique
line transport records the absolute pull/push round-trip error and the part of
the recurrence correction withheld by bounded authority. Curves transport two
endpoint-conditioned seed gauges through one exact reflected forward/adjoint
operator. Their uncertainty combines branch disagreement with their common
distance from the accepted positive-basin refinement; two similarly biased
branches therefore cannot report false agreement. The result is shown by the
workbench as **Path-chart uncertainty (diagnostic)**.

This field answers where the new path constraint is geometrically fragile. It
does not include sensor likelihood uncertainty, unknown-kernel posterior
spread, or a coverage calibration, so it is not called a standard deviation or
credible interval. In single-image blind mode the characteristic recurrence is
disabled; an inaccurate estimated path cannot manufacture a falsely narrow
chart uncertainty.

## Promotion gates

1. Calibrate empirical intervals for coverage on held-out synthetic data.
2. Replace global branch bounds with spatial exposure-characteristic tubes and
   exact adjoints.
3. Add explicit saturation, rolling-shutter, depth/occlusion, and camera-pipeline
   discrepancy modes.
4. Compare kernel posterior calibration, not only the winning kernel.
5. Evaluate uncertainty coverage and restoration quality on GoPro, RealBlur,
   RSBlur, DPDD, and captured complementary pairs.
6. Add an admissible branch-merging construction before claiming any bounded
   enclosure under a reduced mixture.
