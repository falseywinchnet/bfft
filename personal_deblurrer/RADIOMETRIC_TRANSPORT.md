# Radiometric uncertainty transport

## Why geometry alone fails

A pair can carry the correct motion phase while still disagreeing in measured
intensity. Exposure gain is a radiometric coordinate change; clipping is a
censored measurement. Treating either as additive noise corrupts dense
photometric flow, cross-predictive ownership, and the positive normal equation.

The pre-transport control used gains `0.70` and `1/0.70`. Its global circle
still recovered the correct six-pixel displacement, but final reconstructions
ranged from roughly 18--23 dB on most sources because clipped samples retained
full authority.

## Symmetric exposure gauge

`radiometric_transport.py` measures relative gain from 38 full-image luminance
quantiles. Only the continuously bounded midrange contributes to the robust
median log ratio. If

```text
y_2 approximately equals g y_1,
```

both images move into the geometric-mean exposure gauge:

```text
z_1 = sqrt(g) y_1
z_2 = y_2 / sqrt(g).
```

Swapping frames maps `g` to `1/g` and swaps `z_1,z_2` exactly. Absolute scene
radiance remains a pair gauge unless exposure metadata or a calibrated anchor
is supplied; the implementation does not pretend otherwise.

Authority is continuous in `|log g|`:

```text
a_rad = 1 - exp(-(|log g| / 0.12)^4).
```

The working observation is `(1-a_rad)y + a_rad z`, so equal-exposure pairs are
preserved to the measured tolerance without a radiometric mode switch.

## Positive censoring precision

Each sensor sample receives a positive precision derived from smooth headroom
at the upper and lower sensor bounds. The floor is `0.01`; a clipped frame is
never selected or discarded. Precision is transported through every relevant
operator:

- dense common-gauge reconstruction uses spatial frame precision;
- forward/reverse atlas residuals use the geometric mean of reference and
  transported-moving precision;
- latent ownership backprojection uses the matched precision-weighted adjoint;
- multi-appearance descent uses the same sensor precision in its normal
  equation.

This is censoring uncertainty, not a claim that saturated detail can be
recovered when every observation is clipped there.

## Exposure/clipping checkpoint

| Case | Raw average | Censored symmetric average | Unified flow atlas | SSIM |
|---|---:|---:|---:|---:|
| Complementary exposure | 21.608 dB | 22.437 dB | 31.645 dB | 0.9663 |
| Severe complementary exposure | 19.968 dB | 21.829 dB | 26.719 dB | 0.9235 |

All 12 trials improve over the raw average. Mean/minimum gains are
8.394/2.847 dB. Mean/maximum absolute log-gain errors are 0.0191/0.0937. The M4
battery takes 5.77 seconds. Selected artifact SHA-256:
`0cc049ea46cdd7a0d0a9c523b289a976ee7c457e439938edfd80f736824156e8`.

The equal-exposure straight, curved, and smooth batteries were rerun after
integration. Their aggregate conclusions are unchanged; maximum observed PSNR
drift from the pre-radiometric ledgers is 0.0055 dB.

## Rolling-shutter checkpoint

Constant-velocity row timing shared by both frames is a common warp gauge and
cannot be recovered from a pair alone. The executable control instead measures
observable row-dependent relative acceleration plus positive exposure
integration. It enters the existing local circle atlas as a spatial path—not a
rolling-shutter class.

| Case | Average | Dense gauge | Unified flow atlas | SSIM |
|---|---:|---:|---:|---:|
| Moderate row acceleration | 23.270 dB | 24.862 dB | 30.912 dB | 0.9636 |
| Strong row acceleration | 23.326 dB | 24.660 dB | 29.669 dB | 0.9479 |

All 12 trials improve over averaging and dense flow. Mean/minimum gains over
averaging are 6.992/0.842 dB; mean/minimum gains over dense are 5.530/1.056 dB.
The M4 battery takes 5.87 seconds. Selected artifact SHA-256:
`fff5df883a41b265046fcb8bfd34df0c0005926dc97fda7dd72b9a7f88820862`.

The Dear PyGui generator exposes both **Exposure gain** and **Rolling shutter
exposure** while retaining the two source roles: **Blur image** and **Use as-is
for deblurring**. Pair reconstruction reports relative exposure, censoring
precision, atlas chart variation, and authority.
