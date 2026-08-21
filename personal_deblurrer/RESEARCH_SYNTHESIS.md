# Repository synthesis for exposure transport

## Chronology and trust

The two direct blur precursors are older than the repository's requested V3
time anchor. Their code, constants, and conclusions are not inherited. The
new experiment was derived from the named research lines and checked against
their surviving measurements.

V3 has no methodological relationship to this task. It is not imported and
supplies no segmentation, representation, labels, or image state; its first
appearance is used only as the date boundary for deciding which work could be
trusted as later research.

## Optimal descent

Transport Geometry Fusion Descent established that expensive nested solves can
be replaced by single sweeps whose auxiliary transport state persists across
passes. The exact FFT linear subproblem should be solved once per pass; warm
flux/Bregman state carries the active set. Primal momentum is unsafe when the
objective changes, while acceleration is meaningful inside a fixed dual-flux
projection.

For deblurring this gives one shared-latent linear solve followed by one flux
projection per pass. It rules out a nested latent solve inside a nested kernel
solve as the first implementation.

The finite Meyer preconditioner contributes a second point: a transport state
can sometimes be approximated by finite spectral evolution plus one Hodge
lift and one positive capacity projection. The state that moves is a flux; the
source structure may gate where rerouting is allowed but should not dictate
the rerouting vector.

## Superresolution

The two-lattice program says that multiple observations are constraint
families on one latent object. Their useful information is their intersection,
not an image-space average. Recovery gain tracks spectral complementarity;
identical blur adds no coverage, and no stronger projection operator repairs a
joint dead band. Relative reliability and absolute support are separate:

- relative reliability decides which observation is more informative;
- absolute support decides whether any observation measured that degree of
  freedom.

The deblurring consequence is strict. A single blurred capture may support a
regularized inverse, but a complementary capture can change what is
identifiable. The benchmark therefore includes identical, two-angle, and
three-angle controls rather than reporting only a favorable deconvolution.

## Eikonal basis and characteristics

Continuous support transport falsified finite global direction catalogues for
anisotropic geometry. A small locally reduced basis should bound continuous
cones, with a Hopf--Lax coordinate selecting the actual characteristic.
Separately, the Eikonal ray study found that differentiating the assembled
transport ray was more useful than differentiating every primitive basis ray.

The current global checkpoint applies this lesson at the exposure level. A
motion PSF is sampled from one integrated continuous path, not assembled by
selecting a winning raster direction. The next spatially varying version will
carry one local exposure characteristic field and its exact adjoint; it should
not install a per-pixel menu of blur kernels.

## Fourier circles

High Vision showed that independent temporal halves can estimate supported
scene power and noise without clean truth, and that circle pooling can provide
a direction-neutral support law. Fourier-shell Eikonal work added the crucial
qualification: correctly located connection confidence helps inside observed
coherent regions, while uncertain connection cannot justify long transport
through a blind wedge.

Here the OTF is the characteristic function of exposure flow. Circle pooling
therefore audits the transport itself. The estimator retains complex phase in
the closure relation and only pools the scalar closure defect afterward. A
circle is an evidence aggregation domain, not an image prior.

## Transport theory

The positive-cone Fourier theory requires routed mass, an exact projection,
and an explicit accounting of dissipation. Image optimization likewise found
that positive and negative constituents must be transported separately and
that recorded flow must reconstruct coefficient displacement.

Blur formation is simpler: exposure weights are already non-negative. Every
kernel is normalized to one, so DC mass is conserved. Forward and adjoint
operators are tested as a pair. Boundary loss, clipping, saturation, and
sensor nonlinearity cannot be hidden inside a normalized PSF; they will enter
later as separately reported observation operators.

## The new object

The shared object is a positive exposure-path measure `p_i` for each capture
and one latent radiance image `x`. Its Fourier transform `H_i` is not merely a
deconvolution kernel: it is the measured characteristic function of the
transport path. For two registered observations,

```text
Y_i = H_i X,
Y_0 H_1 - Y_1 H_0 = 0.
```

The second relation removes the unknown scene and preserves phase. It also
exposes an exact gauge: multiplying both `H_i` by one common blur factor leaves
the relation unchanged. The implementation chooses the maximally covered
representative of that gauge for computation and declares the common factor
unidentifiable. It never relabels relative evidence as absolute blur.

Given estimated or known paths, the latent solve is

```text
min_x  1/2 sum_i precision_i ||H_i x - y_i||^2 + lambda TV(x).
```

The data Hessian and periodic gradient Hessian are diagonal in Fourier space,
so every latent step is exact. The TV state is a persistent bounded flux. This
is the first fast checkpoint; later nonuniform exposure transport will replace
the FFT diagonalization with exact forward/adjoint characteristic sweeps while
retaining the same evidence and coverage contract.

## Promotion gates

1. Preserve the identity image exactly and close the forward/adjoint pairing.
2. Recover known relative paths from synthetic complementary pairs without
   clean truth entering estimation.
3. Beat each constituent capture on complementary blur and refuse to claim a
   gain on identical blur.
4. Report joint dead bands and forward residuals for every result.
5. Survive non-periodic crops, exposure nonlinearity, Poisson/read noise,
   saturation, and JPEG/ISP perturbations before a real-camera claim.
6. Add depth-layer and rolling-shutter generators before calling the method
   spatially varying.
7. Compare against strong classical and current learned baselines on RealBlur,
   GoPro, RSBlur, DPDD, and a held-out real capture set before making a speed or
   quality ranking.
