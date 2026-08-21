# Positive fourth-cumulant shape transport

## Why covariance is not the exposure law

The accepted local atlas transports the second cumulant of each centered
positive exposure. Different paths can share that covariance. Their relative
Fourier magnitude continues as

```text
log |Y_j/Y_i|
  = -2 pi^2 f^T (C_j-C_i) f
    + (2 pi)^4 / 24 * (K_j-K_i)[f,f,f,f]
    + O(|f|^6).
```

The fourth-order term distinguishes Gaussian-like mixing, uniform path
occupation, endpoint-heavy exposure, and other non-Gaussian shapes without
assigning a blur-family label.

## Positive axis realization

For covariance eigenvalue `lambda`, a symmetric three-point axis measure has
side mass `w`, center mass `1-2w`, and extent

```text
a = sqrt(lambda / (2w)),        0 < w < 1/2.
```

Its normalized fourth cumulant is

```text
kappa4 / lambda^2 = 1/(2w) - 3.
```

`w=1/6` is the Gaussian-matched sigma rule. A continuous uniform line has
normalized fourth cumulant `-1.2`, corresponding to `w=5/18 ~= 0.278`.
Changing `w` therefore transports shape while preserving positive mass, zero
centroid, and covariance exactly. The Cartesian product of both axes is the
same nine-point measure used by the covariance atlas.

## Exchange-symmetric estimation

Every supported Fourier coefficient contributes to positive-energy radial and
antipodal-direction cells. Antipodal frequencies are canonicalized onto one
half-plane before barycentric pooling; otherwise their Cartesian coordinates
would cancel to DC.

At fixed covariance, the `O(f^4)` correction is linear in two normalized axis
cumulants per capture. Two checkerboard Fourier-cell folds solve the complete
capture system by ridge least squares. Each fold is evaluated only on cells it
did not fit. The capture-mean cumulant is removed exactly on each axis because
common shape is a gauge. Fold disagreement continuously reduces parameter
authority. The final cumulant maps analytically back to `w`, so no signed or
non-positive reconstruction operator is introduced.

For spatial mixing, every overlapping chart estimates shape in the same local
covariance eigenframe. Its deviation is additionally tempered by the chart's
covariance-graph authority and blended with the same positive spatial window.
No chart, capture, or shape family wins a branch.

## Evidence

On the four-capture shifted line control, axis-quartic transport improves the
global covariance reconstruction by about 0.05 dB at both 64×64 and 96×96.
All generated measures retain unit mass and reproduce their supplied
covariances to `2e-10` in the invariant.

The global real fit is rejected: it reaches 27.883 dB/0.8341 versus the
27.893/0.8349 global covariance baseline, despite reducing the outer Fourier
ratio to 0.348. Its artifact is
`real_capture_results/personal_deblurrer_koehler_multicapture_quartic_global/`.

The spatial cross-fitted form correctly abstains on the accepted full burst.
All 49 chart authorities are exactly zero; every side weight remains `1/6`.
It reaches 28.254825 dB/0.8412656 versus 28.254837/0.8412661 for covariance,
a numerical difference of only `1.2e-5` dB and `4.7e-7` SSIM. The abstention
artifact is
`real_capture_results/personal_deblurrer_koehler_multicapture_spatial_quartic/`.

```text
spatial quartic results.json
0be7f4e793f24d665bea52b5876012f1d59098d44e74a620d45492e31db205ed

spatial quartic deblurred.png
6b1f01cb6a55673c455bfc567466aa8a31f93b7e320a87b3e560b16536d0855c
```

## Native execution and next boundary

ABI v5 accepts constant or spatial side-mass fields. It derives axis extents
and all nine positive weights inside the generated covariance operator. The
M4 profile retains `4.40e-13` forward and `2.45e-13` adjoint parity and projects
5,713,920,000 materialized bytes to 245,760,192 bytes. Profile SHA-256:

```text
5c27a448c8ad402f54108d9d5b19ae667631f3cc61fd78fca2048c57f3637a95
```

Axis-separable cumulants cannot represent `K_xxxy`, `K_xxyy`, or `K_xyyy`
independently. Curved and asymmetric exposure can live precisely in those
cross-axis terms. The next representation must transport the full symmetric
fourth-order tensor into a positive directional measure while retaining the
same crossfit, gauge, and abstention discipline.
