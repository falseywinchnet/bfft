# Retain the planar reflected view's surface response

The follow-up replaces a specific repeated source integration with a retained
field gather. It uses the unconditional one-per-band renderer as the baseline.
The largest measured benefit is in mirror-relay: about **6.2% less warm frame
time**, with essentially unchanged first-frame time.

## What was still being recomputed

`TransportField` retains diffuse illumination, coupling, and source irradiance
atlases. `CameraSpecularField` additionally retains the directional surface
response for a particular eye position. `compiled_specular_area` previously
used this field only when the shading direction pointed toward that eye.
A reflected view often failed that check and ran `specular_area` again,
integrating the same area sources at many nearby surface positions.

For a planar mirror with point p and unit normal n, reflect the actual eye e:

```
e_virtual = e - 2 n dot(e - p, n)
```

The reflected ray leaving a point on that plane lies on a line from this
virtual eye. This is a shared geometric relation, independent of the sensor
pixel, color channel, camera rotation, field of view, and resolution. A surface
response field indexed by this virtual eye can therefore serve the whole
reflected view.

The comparison registers one virtual origin per rectangular mirror: four in
mirror-relay and one in each other scene. At a matching shading direction,
`compiled_specular_area` gathers that field. The field owns fixed surface grids;
its existing demand evaluator fills an ordinate at most once. It is not a
per-ray response cache and does not add registrations for camera samples.

The ray-to-virtual-eye agreement uses the existing camera-field angular
criterion, dot product > 1 - 1e-11. The reflection identity is exact algebraically;
the finite surface atlas and its interpolation remain approximations. The
implementation does not claim exact image equivalence from the identity alone.

## Complete-frame comparison

M4 Mini; 800×600; one unconditional wavelength per band; retained child laws;
BVH disabled; the same spatial coverage and existing source integration laws.
Both modes retain their ordinary camera-origin field across frames, so the
comparison does not give only the new mode an already available reuse benefit.
Each mode runs three frames in alternating order. Warm times are means of the
two subsequent frames; they are not confidence intervals.

| Scene | First frame, baseline / new | Warm frame, baseline / new | Warm source quadrature saved |
|---|---:|---:|---:|
| Mirror relay | 411.2 / 407.9 ms | **337.1 / 316.4 ms** | **700,902** |
| Aperture canyon | 324.5 / 321.0 ms | 244.3 / 244.1 ms | 17,850 |
| Standard | 155.1 / 152.6 ms | 132.9 / 132.7 ms | 0 |
| Occlusion garden | 325.6 / 327.4 ms | 276.1 / 272.8 ms | 122,859 |

Only the mirror result shows a substantial time reduction in this small screen.
The standard scene performs zero reflected-field gathers, so its tiny timing
change is noise. The other small timing changes should not be overinterpreted.

Virtual-field allocation/setup takes an additional 0.2–0.7 ms before the first
frame. The first-frame timings include the demand field's actual source
integrations. Static scene/beam/transport construction is excluded equally.
Source integration inside field construction is counted separately in the raw
JSON; it must not be omitted when assessing first-frame work.

In mirror-relay:

- 3,575 per-frame specular source integrations become field gathers.
- Direct source quadrature falls from 5,420,109 to 4,719,207 per warm frame.
- The virtual fields evaluate 2,344 surface ordinates during the first frame,
  using 529,218 source quadrature evaluations.
- Subsequent identical frames perform **zero new virtual-field source
  evaluations**. They reuse the ordinates and interpolation decisions.

The warm aperture baseline is lower than the previous N=1 cold-frame table
because its ordinary camera field is now retained, an existing capability of
the renderer. That difference is not attributed to the new reflected fields.

## Image and structural checks

Across the four scenes, 24 comparison frames were measured. Repeated baseline
frames match exactly. A further mirror run asserts that both modes produce
identical output across first and warm frames, and checks stale-generation
rejection. Its output is `reflected_gather_verified.json`.

| Scene | Pixels with any change / 480,000 | Pixels differing by >1 level | Maximum channel difference |
|---|---:|---:|---:|
| Mirror relay | 540 | 86 | 8 |
| Aperture canyon | 0 | 0 | 0 |
| Standard | 0 | 0 | 0 |
| Occlusion garden | 37 | 7 | 14 |

Differences are measured in final 8-bit channel values against the N=1 baseline.
The two images look very similar at full-frame viewing size. The interpolation
is not a strict one-byte-error certificate.

The probe verifies the virtual-eye reflection identity at an interior planar
point to squared direction error < 1e-24. The point is projected onto the plane
used by the intersection law: some scene rectangles have rounded supplied
normals that are not exactly orthogonal to their spans. A changed illumination
field generation rejects the old field and takes the existing evaluation path.

## Scope

This is a runnable retained-gather prototype; production defaults are unchanged.
It covers the directional source response for supported planar reflected views.
It does not eliminate primary visibility, spatial ownership classification,
curved reflection, refractive path propagation, or every remaining specular
source integral. Those quantities are not all present in the existing retained
field. The concrete cleanup here is to retain another shared response instead
of recomputing it per camera path.

Reproduce:

```sh
experiments/photonic_transport_field/run_reflected_gather.sh
.venv-jpeg/bin/python experiments/photonic_transport_field/plot_reflected_gather.py
```

The wrapper generates its comparison source from the authoritative renderer,
checks all substitution anchors, builds on the Mini via `m4build`, and copies
each run immediately. Raw frames, counters, `summary.json`, and `comparison.png`
are in `reflected_gather_m4/`.
