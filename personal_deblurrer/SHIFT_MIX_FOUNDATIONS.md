# Deterministic transport before centered mixing

## Correction to the first checkpoint

Blur-family classification is not the reconstruction. Gaussian, disk, line,
curve, and random path remain useful synthetic generators. A multi-capture
camera can also use a family or finite hypothesis law as an estimation aid.
Neither fact makes a catalogue the inverse operator.

The unified image-formation object is an exposure integral of transported
radiance:

```text
y(u) = integral rho(t,u) x(W_t^-1(u)) dt + sensor error.
```

`W_t` is the coordinate transport at exposure time `t`; `rho` is positive
exposure mass. A spatially invariant displacement law is the special case

```text
y(u) = integral x(u-z) d mu(z).
```

Every positive displacement measure has a centroid `m` and centered residual
measure `nu`. Therefore

```text
mu = translate(m) # nu
H_mu(f) = exp(-2 pi i f dot m) H_nu(f).
```

The phase ramp is deterministic translation. The centered characteristic
function is mixing around the transported center. The analytical order is:

1. recover any observable deterministic coordinate transport;
2. pull the observation back to transported-center coordinates;
3. invert the residual positive mixing measure;
4. push the reconstruction through the original forward operator and audit
   its residual against the immutable observation.

For spatially varying motion the factors need not commute. `W` must therefore
be recovered first; only then is a local centered mixing law meaningful.

## Two physical regimes, one operator

Camera or object motion integrates a trajectory of warped images. Projective
camera-shake models represent the result as a weighted sum of homographically
transformed sharp images. Motion-density and filter-flow work show why one
global convolution is only a restricted case.

Defocus and many optical aberrations instead mix radiance around an ideal ray
center. Their support may be a disk, polygonal aperture, field-dependent
aberration, or depth-dependent circle of confusion. There may be no coordinate
shift to recover, but the same second-stage centered positive inverse applies.

Thus the distinction is not “motion algorithm versus defocus algorithm.” It is
whether the forward operator contains an identifiable deterministic coordinate
transport before its residual centered mixing.

## Identifiability boundary

An absolute global translation cannot be recovered from one unknown image.
For any candidate sharp image `x`, translating `x` and changing the supposed
camera offset produces the same pixels. The implementation reports
`absolute_translation_is_single_image_gauge` rather than inventing a shift.

A relative shift is observable when two captures share scene coordinates.
`estimate_relative_shift` uses phase correlation for this relative case.
Known synthetic operators expose their centroid directly. Future spatial warp
estimation will need local geometric, multi-frame, inertial, rolling-shutter,
or scene-line evidence.

## Center-mixing transport

The current inverse applies the positive forward/adjoint correction

```text
x_(n+1) = x_n * M_nu^*(y / M_nu x_n).
```

Reflection padding prevents the opposite edge of the image from becoming
false evidence. Positivity prevents the signed overshoots of a raw inverse
filter. Every run records the observation fingerprint before and after, the
forward residual, and, for a genuine synthetic run only, the before/after
PSNR with clean truth used solely for evaluation.

Uniform line exposure still has exact Fourier null lines. No single-image
method can infer those coefficients from the data term. Positive iterations
can indirectly place energy there and create faint halos, so the final state is
projected by absolute Fourier coverage: well-measured bands retain the inverse;
near-null bands return toward the observation. The removed unsupported energy
and dead fraction are recorded. Complementary observations remain the honest
way to measure those bands.

## Provisional single-image estimation

For a file marked **Use as-is**, absolute shift is declared unobservable. The
current provisional estimator uses the autocorrelation of the absolute
phase-only image to measure centered displacement covariance. The complete
covariance becomes one tensor Lobatto positive cubature: every numerically
resolved eigenaxis contributes continuously through the same three-node
moment rule. There is no anisotropy threshold, line/cloud branch, or Gaussian
surrogate. The resulting measure passes through the same positive
center-mixing inverse as a known operator.

This estimator can sharpen analytic controls but does not yet recover curved
path handedness, projective motion, depth-varying defocus, occlusions, or real
camera ISP effects. Those are active promotion gates.

## Transported characteristic charts

For a uniform line exposure of length `L`, differentiation along its
characteristic gives

```text
L d y(s)/d s = x(s + L/2) - x(s - L/2).
```

This is a path recurrence, not an FFT division. It exposes the unmeasured
residue-class seeds explicitly. Each finite seed vector is selected by minimum
correction plus longitudinal flux action.

An axis-aligned path uses the exact raster chart. An oblique path is pulled
into an expanded rotated coordinate domain, recurred horizontally, pushed
back, and center-cropped. `reshape=True` is essential: a same-size rotated
chart silently clips source evidence. Bilinear interpolation still has a
measurable conditioning cost, so the recurrence authority is

```text
a_lattice(theta) = 0.025 + 0.225 cos^2(2 theta).
```

This is not an angle-family classifier. The angle is the principal coordinate
of the known or consensus exposure measure, and the cosine term prices its
alignment with the sampling lattice.

Rasterization makes even a perfectly straight oblique PSF have nonzero minor
covariance. Straightness is therefore decided by a quadratic displacement
skeleton: a line has constant fitted tangent, whereas a curved path has
measured tangent turn. The intervening checkpoint gave mild curvature three
local straight charts and refused the bend-8 control.

That tangent-atlas construction is now retired. A curved path is not a
translation group, so it cannot inherit the exact line recurrence by
coordinate rotation. `CURVILINEAR_EIKONAL.md` instead lifts every positive PSF
atom into one ordered exposure tube and retains the original atom in an exact
reflected gather/scatter operator.

Differentiating the exposure integral also differentiates noise, and a late
recurrence can duplicate sharpening already performed by the positive solve.
The recurrence is therefore one auxiliary constraint with continuous
authority rather than a line-selected solver:

```text
a = a_trust a_descent a_residual a_anisotropy a_tangent a_extent
    (0.025 + 0.225 cos^2(2 theta)).
```

Here `a_descent = min(1,sqrt(12 / maximum_positive_passes))`; the remaining
factors measure operator trust, residual demand above the common noise
discrepancy, covariance anisotropy, fitted-tangent coherence, and path extent.
Center clouds and curved paths therefore suppress this recurrence continuously
while still using the same exact positive-exposure gather/scatter descent.

Line-constraint uncertainty combines coordinate round-trip error and the
recurrence correction not granted authority. Every positive measure uses the
exact lifted operator; its uncertainty transports both endpoint seed gauges
and includes their common displacement from the accepted basin. These are
internal diagnostics, not calibrated credible intervals.
Single-image blind estimates are not allowed to activate the recurrence:
estimated geometry is not trusted geometry. A known synthetic operator or
future multi-observation path consensus is required.

The next solver must lift this exact gather/scatter contract from global
translation paths to spatial projective or optical-flow transport. There the
image-warp determinant, visibility, and depth-layer ownership enter the
operator; they cannot be hidden in the one-dimensional path Jacobian.

## Primary sources

- Gupta et al., *Single Image Deblurring Using Motion Density Functions*,
  ECCV 2010.
- Hirsch et al., *Fast Removal of Non-uniform Camera Shake*, ICCV 2011.
- Whyte et al., *Non-uniform Deblurring for Shaken Images*, IJCV 2012.
- Welk et al., *Fast and Robust Linear Motion Deblurring*, Signal, Image and
  Video Processing 2015 (arXiv 2012).
- Zheng et al., *Forward Motion Deblurring*, ICCV 2013.
- Pan et al., *Phase-only Image Based Kernel Estimation for Single-image Blind
  Deblurring*, CVPR 2019.
- Zhang et al., *Exposure Trajectory Recovery from Motion Blur*, ECCV 2020.
- Dansereau et al., *Richardson-Lucy Deblurring for Moving Light Field
  Cameras*, CVPR 2016.
- Son et al., *Single Image Defocus Deblurring Using Kernel-Sharing Parallel
  Atrous Convolutions*, ICCV 2021.
- Liu et al., *Motion-adaptive Separable Collaborative Filters for Blind Motion
  Deblurring*, CVPR 2024.
