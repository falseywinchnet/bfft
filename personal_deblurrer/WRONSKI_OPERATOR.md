# Wronski's filter as one analytic transport measure

## Constraint

The reconstruction does not choose among Gaussian, disk, line, curve, or
Wronski blur classes. Those names remain useful only for generating controlled
observations. The inverse receives one positive displacement measure

```text
mu = sum_a w_a delta(z-z_a),  w_a >= 0,  sum_a w_a = 1,
```

and derives its transport coordinates directly.

For a single unknown image, absolute blur and latent texture are not uniquely
factorable without additional evidence. An analytic method must report this
gauge rather than replace a classifier with an unreported image prior. Known
synthetic operators, calibrated optics, or relative multi-capture spectra do
provide the missing operator evidence.

## The finite binomial measure

The filter in Wronski's article is

```text
[1/4, 1/2, 1/4].
```

It is the probability law of two independent signed half-step transports. On
an axis-aligned unit displacement fiber `d`, its centered characteristic
function is

```text
phi(f) = cos(pi f dot d)^2.
```

Applying it twice is not a second blur family. Convolution adds independent
displacements, producing

```text
[1, 4, 6, 4, 1] / 16,
phi_2(f) = cos(pi f dot d)^4.
```

The separable 2-D construction is likewise the product measure

```text
phi_xy(f) = cos(pi f_x)^2 cos(pi f_y)^2.
```

These cases are exposed in the workbench as synthetic operators. They enter
the same exact positive gather/scatter inverse used by every other finite
exposure measure.

## Direction without classification

For any positive measure, the analytic support begins with its cumulant jet:

```text
m_i       = E[z_i]
C_ij      = E[(z_i-m_i)(z_j-m_j)]
T_ijk     = E[(z_i-m_i)(z_j-m_j)(z_k-m_k)]
K_ijkl    = fourth cumulant.
```

The covariance eigenframe supplies the low-frequency transport coordinates.
This is a coordinate calculation, not a line/disk decision. A rank-one
measure has one resolved coordinate. An isotropic measure has no privileged
principal direction. Rasterized oblique paths honestly acquire a small second
coordinate from bilinear footprint width. The mixed third moment
`E[tangent^2 normal]` supplies signed bend evidence for asymmetric curved
support, while the fourth cumulant distinguishes measures sharing the same
covariance.

The full directional object is not the covariance eigenvector. It is the
Fourier-eikonal field of the exact characteristic function:

```text
A(f) = -log |phi(f)|^2,
v(f) = grad_f A(f).
```

`analytic_support.fourier_eikonal_field` evaluates both `phi` and its gradient
from the positive atoms in closed form. No finite-difference direction bank,
optimizer, blur label, or fitted filter is involved. At Wronski's Nyquist null
`f dot d = 1/2`, `A` is infinite and `v` is withheld: the observation did not
transport that coefficient.

## Consequence for the double-warped web example

If a displayed example has also been spatially transformed, resized,
quantized, or compressed, the observed raster is a composition. The unknown
inverse must not begin by deciding which of those stages exists. Positive
spatial stages instead close as one conditional row measure

```text
K_21(p,dq) = integral K_1(r,dq) K_2(p,dr).
```

`composed_transport.py` now performs this composition exactly for the discrete
reflected/bilinear operator. The inverse receives the consolidated measure,
not its factorization. A double radial exposure is particularly direct: radial
scale atoms multiply under composition, so their log-scale coordinates add.
Quantization and nonlinear radiometry cannot generally be absorbed into a
linear positive row measure; absent independent metadata they remain explicit
uncertainty/support loss, not a stage inferred from appearance. See
`COMPOSED_OBSERVATION_TRANSPORT.md`.

## First measured checkpoint

`run_wronski_operator_benchmark.py` evaluated the same inverse on the six
denoiser scaffold sources at 96x96, with read-noise sigma `0.002` and 32
maximum positive-transport passes. Values are source-mean PSNR and SSIM:

| Positive measure | Observation | Transport inverse |
| --- | ---: | ---: |
| single-axis binomial | 32.819 dB / 0.9653 | 39.577 dB / 0.9900 |
| repeated-axis binomial | 30.117 dB / 0.9342 | 34.522 dB / 0.9731 |
| repeated oblique binomial | 30.314 dB / 0.9255 | 41.069 dB / 0.9861 |
| separable binomial product | 29.732 dB / 0.9173 | 34.857 dB / 0.9704 |
| shift then repeated oblique binomial | 19.038 dB / 0.4411 | 37.731 dB / 0.9764 |

No row activates a blur-family decision. The shifted control factors its
centroid transport first; every centered control then follows the same
positive-measure inverse and absolute characteristic-support gate. The raw
44-KB measurement ledger is retained locally as the ignored artifact
`wronski_operator_results.json`.
