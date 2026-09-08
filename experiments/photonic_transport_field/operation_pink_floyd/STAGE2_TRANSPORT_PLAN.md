# Stage 2 transport optimization plan

Stage 2 begins only from the accepted physical image and its byte-identical
analytic oracle. Lane A means transport fusing, splitting, and collapsing.
Lane B means retained deformation inside child volumes. An optimization is
accepted only when it preserves boundary ownership, energy, spectral order,
physical support, and the camera image within a declared terminal error.

## Present cost anatomy

The overhead baseline performs 2,578,181 packet-density evaluations and
reconstructs 76,311,236 Fourier terms. The geometry is not responsible for this cost.
At each camera quadrature point it scans many spectral packets, and each scan
repeats the same `y` and `z` trigonometric reconstruction. The analytic oracle
finishes in 78ms because its Gaussian evaluation is constant work; the
retained Fourier backend takes 123ms because those repeated mode sums have not
yet been fused.

## Lane A: fuse, split, collapse

1. **Projected-support bins.** Project every packet's certified volume support
   into camera tiles. A quadrature point visits only packets registered in its
   tile. This removes the present all-packets scan without changing transport.
2. **Shared incident geometry.** All wavelengths have one carrier and one
   transverse field before the prism. Retain one geometric packet with a
   spectral coefficient block. The left volume gathers its camera RGB moments
   once; the prism consumes the uncollapsed spectrum at its boundary.
3. **Boundary-only spectral split.** Split the common packet only where the
   Sellmeier boundary map makes wavelength lanes occupy distinguishable
   recipient regions. Wavelengths whose projected supports are subpixel-close
   remain one certified lane.
4. **Continuous fan coordinate.** Fit the smooth wavelength-to-exit-position
   map with a low-order Chebyshev or rational basis and integrate camera color
   responses over intervals. Sixty-five discrete packet visits then become a
   small number of fan segments, refined only at spectral/display boundaries.
5. **Terminal collapse.** Once no downstream object distinguishes wavelength,
   collapse the spectral block directly into the camera or floor response.
   Never carry 65 lanes beyond their last wavelength-selective boundary.
6. **Recipient interval ownership.** The floor and camera receive exact
   projected intervals from each packet/fan segment. Pixels outside those
   intervals perform no field evaluation.
7. **Backward observable fusion.** Pull the camera RGB and floor RGB response
   functions backward through the volume and prism operators. Retain only the
   local spectral span that can alter either terminal above its error bound.

## Lane B: retained volume evaluation

1. **Mode-major structure of arrays.** Store coefficients, phase increments,
   extinction, and support bounds in aligned mode-major blocks. Evaluate
   several packets with SIMD rather than completing one scalar packet at a
   time.
2. **Separable reconstruction cache.** The current field is separable in `y`
   and `z`. Cache each unique one-dimensional reconstruction per camera sample
   and share it across wavelength lanes with the same source footprint.
3. **Phase recurrence.** Replace repeated `cos(kx)` calls with one sine/cosine
   seed and a stable recurrence across modes. Precompute attenuation and
   diffusion multipliers per retained block.
4. **Analytic camera-ray integration.** Along one ray through a homogeneous
   body, carrier displacement, Fourier phase, extinction, and camera
   transmittance are exponential functions of the same ray parameter. Integrate
   each retained mode over the segment analytically, replacing ten volume
   quadrature samples with one mode response.
5. **Homogeneous block collapse.** Consecutive equal extinction/diffusion
   steps are one diagonal multiplier. Preserve marching as a construction and
   update operation; do not reapply identical steps during every camera query.
6. **Observable-bounded mode truncation.** Bound every mode by its largest
   possible contribution through phase, scattering, exposure, and the camera
   response. Discard it only when that terminal contribution is subpixel.
7. **Sparse medium coupling.** When a later medium introduces anisotropic or
   spatially varying scattering, represent it as a banded or low-rank coupling
   between retained modes. Homogeneous convolution remains diagonal.
8. **Revision-keyed responses.** Cache a child-volume response by boundary,
   medium, source-basis, and terminal-observable revisions. Intensity changes
   update coefficients; topology changes rebuild the affected response.

## Required measurements after every stage

- Fourier/reference maximum and mean channel error at 256x256;
- energy-ledger closure at every registered boundary;
- blue/green/red ordering and exit-angle residuals;
- absence of field outside each registered support;
- packet visits, density evaluations, Fourier terms, and wall time;
- identical geometry and boundary ledgers for both backends, with all camera
  visibility arising from the registered sheet face and refracted glass path.
