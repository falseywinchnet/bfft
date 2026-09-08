# Operation Pink Floyd

This is the minimal three-dimensional child-volume demonstrator for retained
photonic transport. A near-overhead pinhole camera looks down on a vertically
extruded glass prism sitting directly on a thin glittery black sheet. The
optical cross-section is the prism's triangular footprint in the sheet plane;
it is not a screen-space triangle and light is never drawn as a line.

The source aperture sits at the sheet boundary. The exact first moments of its
vertical Gaussian divide it into two equal registrations: the lower half enters
the sheet and the upper half enters the prism. Both retain the same transverse
support and spectral state. The lower packet is absorbed and weakly diffused by
the black material. The upper packet crosses the prism's actual side faces,
where 65 wavelength lanes from 420 nm through 680 nm undergo Sellmeier N-F2
dispersion, Snell refraction, Fresnel loss, glass absorption, and affine
footprint deformation.

The entry boundary also hands those lanes to a prism-owned internal retained
field. The clear bulk has a small scattering coefficient. A camera
characteristic integrates a lane only when it crosses that narrow field, so
the illuminated entry edge and faint internal response appear without making
the complete glass volume emissive.

The sheet is a transparent parent volume containing an invisible Fourier
deformation membrane. Neither the membrane nor the beam registers directly
with the camera. The membrane receives the incident and dispersed fields; the
sheet's complete upper face turns that response into outgoing radiance using
its local deformed normal, volume extinction, scattering phase, and glitter
response. Consequently the beam becomes visible only where the material
scatters it toward the camera.

The glass camera path intersects the real triangular mesh. Camera wavelengths
refract through the entry and exit faces independently. A ray leaving through
the bottom face crosses the glass-to-sheet boundary and queries the parent
face response; this makes the clear prism an optical participant rather than a
black mask or a painted overlay.

Each packet retains a compact separable Fourier field with a finite physical
support seal. `fourier` reconstructs that field; `reference` evaluates the
corresponding unbounded Gaussian diffusion law analytically. Both use the same
geometry, boundaries, sheet response, glass response, and pinhole camera.

Run on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  make -C experiments/photonic_transport_field/operation_pink_floyd \
  clean test render-both
```

The accepted 256×256 overhead image is
[`pink_floyd_volume_256.png`](pink_floyd_volume_256.png). Measurements are in
[`RESULTS.md`](RESULTS.md), and the transport optimization plan is in
[`STAGE2_TRANSPORT_PLAN.md`](STAGE2_TRANSPORT_PLAN.md).
