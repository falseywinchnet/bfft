# V3 support-conditioned 2x super-resolution experiment

## Observation model

The original Golden Gate image is cropped to an even lattice and retained as
ground truth `x`.  The algorithm receives only

```
y = D_Lanczos x
```

where `D_Lanczos` is a fixed two-times Lanczos reduction.  No full-resolution
pixels participate in reconstruction or parameter estimation.

The comparison includes nearest, bilinear, bicubic, Lanczos, and two fixed
classical back-projections.  This is single-image restoration, not a
multi-frame or neural comparison.

## Continuous v3 lift

The diagnostic form of v3 retains its final low-resolution design matrix.  At
twice the sampling density:

1. affine cell coordinates are evaluated exactly in normalized physical
   coordinates;
2. graph-phase and paired one-sided ridge columns are Lanczos-lifted;
3. the fixed per-cell coefficients are evaluated on the dense lattice.

Let `M_2` be that continuous evaluation and let `U M_1` be Lanczos
interpolation of the same model's low-resolution raster.  The only candidate
super-resolution information supplied by v3 is therefore

```
delta_v3 = M_2 - U M_1.
```

This subtraction is essential.  Replacing the observation with `M_2` also
replaces reliable low-frequency image content with the deliberately lossy v3
model.

## Gaussian support posterior

Per-cell fit error defines a confidence

```
c_i = exp(-e_i / quantile_0.8(e)).
```

Local fitted texture density supplies a second factor.  Their product retains
between a conservative structural floor and full support.  The 95th percentile
of `|delta_v3|` clips isolated speculative terms.  Two fixed
owner-respecting back-projections then reduce `|D x_hat - y|`; this is a fixed
projection schedule, not a convergence descent.

## Golden Gate result

| Method | PSNR | Edge PSNR |
|---|---:|---:|
| Bicubic | 26.069 dB | 20.630 dB |
| Lanczos | 26.345 dB | 20.968 dB |
| Lanczos + two fixed back-projections | **26.742 dB** | **21.391 dB** |
| Raw continuous v3 basis lift | 24.011 dB | 18.810 dB |
| V3 Gaussian support posterior | 26.583 dB | 21.228 dB |
| V3 owner-conditioned projection, no new octave | 26.700 dB | 21.341 dB |

The v3 posterior beats plain Lanczos but not classical back-projection.  More
importantly, the measured gain ladder is monotone in the wrong direction:

| V3 innovation gain | PSNR |
|---:|---:|
| 0.000 | **26.700 dB** |
| 0.125 | 26.697 dB |
| 0.250 | 26.691 dB |
| 0.500 | 26.669 dB |
| 1.000 | 26.583 dB |

This distinguishes two claims:

- V3 demonstrably recovers continuous subpixel geometry and phase alignment
  on the observed lattice.
- A single subsampled image does not determine a new Nyquist octave merely
  because the support geometry is continuous.

The useful v3 contribution at this stage is an owner-aware restoration prior.
Extrapolating its current ridge/phase basis into genuinely unobserved
frequencies is speculative and should remain disabled until another invariant
constrains that octave.

## Eikonal Lanczos

The closest known construction is AMD FSR1 EASU, described by AMD as a
directionally and anisotropically adaptive radial Lanczos whose dimensions are
rotated to match features, with ringing reduction and color clamping. Other
neighboring families include contour-stencil interpolation, geodesic-distance
weighted autoregression, and reversible anisotropic diffusion-projection.
None of these is a Lanczos reconstruction kernel evaluated in an eikonal
metric induced by an objective segmentation.

The first v3 prototype uses a fixed local chart. The structure tensor defines
normal and tangent coordinates; Lanczos-3 is evaluated as a tensor product in
that chart; taps belonging to another structural owner are inadmissible; the
weights are normalized to reproduce DC exactly; and the result is clamped to
the range of its same-owner samples. It uses a fixed stencil and no search or
convergence solve.

On an analytically supersampled oblique `0.2 -> 0.8` step:

| Method | PSNR | Halo mean | Transition fraction | Output range |
|---|---:|---:|---:|---:|
| Lanczos | **40.695 dB** | 3.99e-5 | 0.00775 | 0.135–0.860 |
| Local-eikonal Lanczos | 40.154 dB | **0** | **0.00719** | 0.161–0.840 |

Thus PSNR prefers the reconstruction with false light and dark halos. The
eikonal-support result has zero out-of-support ringing and a 7.2% narrower
transition despite its lower PSNR.

The local-chart form is not yet the mathematically exact curved-space
generalization. Replacing Euclidean radius by geodesic distance is insufficient
because source samples cease to be uniform in that coordinate. The exact
generalization is a windowed spectral projector of the metric's
Laplace–Beltrami operator,

```
K_G(x,y) = sum_k w(sqrt(lambda_k) / Omega)
                 phi_k(x) phi_k(y),
```

with the observation constraint retained separately. A practical v3 version
can approximate this projector by a fixed polynomial of the owner-masked
metric graph operator. This supplies the missing density compensation and
curvature consistency without computing an eigendecomposition.
