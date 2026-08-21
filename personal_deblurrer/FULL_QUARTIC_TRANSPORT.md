# Full symmetric fourth-cumulant transport

## Representation

Covariance cannot distinguish positive exposure measures that have the same
center and second moment. The next Fourier log-magnitude term is the complete
symmetric fourth-cumulant tensor

```text
K4 = (Kxxxx, Kxxxy, Kxxyy, Kxyyy, Kyyyy).
```

The cross-axis components distinguish rotated, curved, and mixed-direction
occupation that an eigenaxis-only model cannot represent. The implementation
does not classify those paths. It uses one Gaussian-matched baseline plus
positive bounded-axis and tailed-axis measures at eight fixed unoriented
angles. Varying only one standardized axis is essential: equal axis kurtosis
has fourfold symmetry and spans only three tensor coordinates. The corrected
dictionary has numerical rank five. Every convex mixture has positive mass,
unit mass, zero centroid, and exact supplied covariance.

For capture covariance `C = A A^T`, dictionary points `z` become `A z`. The
canonical factor `A` uses covariance principal axes. This detail is necessary:
at zero tensor authority, the baseline dictionary measure is then exactly the
existing covariance exposure measure, including unmodelled sixth and higher
moments. A symmetric matrix square root matches covariance but violates this
exact gauge identity.

## Estimation and uncertainty transport

`full_quartic_transport.py` pools antipodally canonicalized Fourier-circle
cells and subtracts the exact covariance-measure log transfer. Two checkerboard
folds fit all five tensor coordinates for every capture. Every fold is scored
only on cells it did not fit. Capture-mean tensor is removed exactly, because
common fourth-order blur is unidentifiable.

Authority is transported continuously through three measured quantities:

1. held-out predictive improvement over the covariance transfer;
2. componentwise agreement between folds; and
3. a shared dimensionless fourth-power signal taper calibrated against the
   reflection-boundary and sixth-order null floor.

All capture tensors enter one joint positive program. It fits their relative
tensors while solving a single common fourth-order gauge with an exact analytic
gradient. Production estimation evaluates the baseline common gauge; the full
17-member gauge catalog is retained only as an explicit research audit. This
removes repeated solutions that cannot create new relative evidence.

`quartic_gauge_posterior.py` reconstructs both the covariance and relative-K4
positive measures. It transports their forward-closure ratio and absolute
outer Fourier-circle redistribution into continuous posterior mass, then
returns the posterior image and within/between-gauge uncertainty. No capture,
direction, blur family, or gauge wins a selection branch. When evidence
vanishes, the two reconstructions coincide continuously.

## Controlled evidence

The M4 96×96, four-capture battery supplies covariance to isolate shape
estimation. With 32 optimal positive-line passes:

| Regime | Authority | Covariance | Relative K4 | Posterior | Posterior delta |
|---|---:|---:|---:|---:|---:|
| Exact covariance null | 0.000020 | 71.477 | 71.476 | 71.476 | -0.0004 dB |
| Moderate directional + sensor noise | 0.225720 | 40.595 | 42.446 | 42.724 | +2.129 dB |
| Strong directional | 0.345283 | 35.249 | 38.701 | 37.736 | +2.487 dB |
| Opposed directions, common K4 unanchored | 0.324277 | 39.637 | 38.967 | 39.863 | +0.227 dB |

All realized covariance errors are below `9.8e-15`; all atom weights are
non-negative. The opposed case preserves the important diagnostic failure of
the relative-K4 member (`-0.670 dB`): minimum-distance common gauge is not
evidence for true common shape. The image-domain posterior does not select it;
it retains `0.0311` K4 mass and improves the covariance reconstruction by
`0.227 dB`. The reproducible artifact is
`full_quartic_positive_directional_battery.json`, SHA-256
`2d465505b91f352da6d4505bd1a3031dc21568254c39dcfa09693314e7bd849c`.

## Generated-operator optimization

`CompactGlobalExposureField` stores only the centered positive atoms and
weights. `CompactGlobalReflectedExposureOperator` applies the identical
bilinear reflected transport as one circular FFT on a `2H x 2W` even
extension, and implements its exact matched adjoint by crop embedding and
reflection folding. Grayscale/RGB forward and adjoint parity are below
`1e-12`, the adjoint inner-product gate passes, and unit mass is exact.

On the M4 CPU, an 800x800 153-atom operator stores `20,534,872` bytes versus
`5,483,520,000` bytes for the materialized field/plan representation: a
`267.03x` reduction. Construction takes `0.0140 s`, forward `0.0380 s`, and
adjoint `0.0332 s`; adjoint inner-product error is `3.30e-12`. This removes the
former global quartic materialization barrier without changing the inverse or
posterior values. Native specialization is deferred until this representation
has passed the next real reconstruction gate.

## Real-capture gate

On the twelve Köhler scene-1 web captures, full-rank estimation reduces
held-out relative log-magnitude RMS from `0.59193` to `0.57233`, but earns only
`0.02802` mean shape authority. Its largest transported coordinate is
`0.07474`. The image-domain gate now completes. Relative K4 improves the global
covariance member from 27.89344 dB / 0.83493 SSIM to 27.89569 dB / 0.83547.
The intrinsic action assigns it 0.14017 posterior mass; the posterior reaches
27.89381 dB / 0.83501. Precision-weighted closure changes from 0.967236 to
0.967131 of center disagreement in the K4 member and 0.967218 in the posterior.
Its outer Fourier ratio rises from 0.3660 to 0.4008, which correctly prevents
the small closure gain from receiving excessive authority.

This global posterior does not displace the selected center-first spatial
atlas at 28.31305 dB / 0.84244. A follow-up fits the full five-coordinate K4
model independently in all 49 overlapping 192-pixel charts. Every chart has
exactly zero held-out predictive authority. Spatial K4 therefore abstains;
building a 153-atom spatial field would add unsupported capacity rather than
measured transport.

The estimation-only M4 artifact preserves every source hash; SHA-256
`34cb8e2d32f2bc062b1ea35da25aefaf8e9b88a3e0dee4ce2794621c2a5a1ba5`.
The current posterior artifact is
`real_capture_results/personal_deblurrer_koehler_quartic_image_posterior_v2/`
with results SHA-256
`72ad98da965159682ddcf812b1ababf4508c43c58f41e8a76b157460e267e6a6`.
Four-way compact FFT batching reduces its wall time from 134.096 to 70.997
seconds. The spatial probe SHA-256 is
`46f7b9aed709a20cb7b642f4c0900b25beedb6c31ad1cd6a2a67b0092bda05a8`.
The earlier rank-3 artifact is archived with a
`rank3_rejected` suffix and is not evidence for the corrected method.

## Next boundary

Moment-only sparse compression remains rejected because matching covariance
and K4 does not preserve sixth-order transfer. The global and spatial real
gates are now resolved: weak global K4 remains uncertainty mass, while local
K4 abstains. The next boundary is generalization of the center-first adaptive
atlas across other recorded bursts and blur/noise regimes; fourth-order shape
should reactivate only where held-out local evidence becomes nonzero.
