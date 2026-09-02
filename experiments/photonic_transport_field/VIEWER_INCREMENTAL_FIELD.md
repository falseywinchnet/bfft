# Incremental viewer field

## Fixed-origin theorem

Let the eye occupy world point `O`, let `R` be the camera orientation, and let
`d(p)` be the pinhole direction of sensor coordinate `p` in the camera's local
chart. The world ray is

```text
r(p; O, R) = O + t R d(p).
```

For a static scene, define `F_O(w)` to contain the first hit, visible optical
chain, terminal signature, linear radiance, and certified interpolation state
for world direction `w` on the unit sphere. Then

```text
I(O, R, p) = F_O(R d(p)).
```

Consequently, changing yaw, pitch, roll, FOV, aspect ratio, or sensor
resolution while holding `O` fixed does not change `F_O`. It changes only the
chart used to sample it. No world-space visibility, illumination, BRDF
view-vector, reflection chain, or refraction chain needs to be rediscovered.

This is stronger than temporal reprojection. Reprojection guesses that the
previous image remains useful. The spherical viewer field states that the
underlying relation is exactly the same object under a coordinate change.

## Existing invalidation boundaries

| Representation | Rotation in place | Viewer translation | Object/light change |
| --- | --- | --- | --- |
| Scene primitives and acceleration | reuse | reuse | update affected bounds |
| Collimated beam field | reuse | reuse | update affected source paths |
| Diffuse transport graph/fixed point | reuse | reuse | residual march from changed nodes |
| Direct surface atlases | reuse | reuse | update affected receiver/source regions |
| Viewer-origin specular atlas | reuse exactly | update glossy/metal surfaces | update affected surfaces |
| World-direction terminal field | sample through new chart | reproject and repair | repair affected angular cells |
| Sensor edge coverage | reproject cached spherical arcs | reproject and repair | reproject affected arcs |
| Output pixels | resample/write | resample/write | resample/write |

The implementation now makes the first camera-specific boundary explicit:
the former `CameraSpecularField` compiler accepts only `viewer_origin`. A
cached field is valid whenever its three origin coordinates match, regardless
of camera orientation or projection parameters.

## Spherical sparse representation

The remaining camera solve should be stored independently of any rectangular
viewport:

```text
ViewerDirectionalField(O)
  angular cells: hierarchical cube-map or octahedral chart
  per cell:
    angular bounds
    first-surface owner
    visible optical terminal signature
    radiance controls and CONV currents
    error certificate
    intersecting visible-boundary arcs
```

A cube-map is especially natural here: it is the camera's proposed
"sphere-as-cube window," but used as six charts of direction rather than as a
geometric approximation to a sphere. Each face can use the current 1-D CONV
fibres or a certified 2-D profile. Cells bifurcate only when owner/topology
changes or the radiance residual violates the display budget.

World-space primitive boundaries also have reusable angular forms about `O`:

- rectangle and triangle edges become great-circle arc segments because the
  edge and `O` define a plane through the origin of direction space;
- a sphere silhouette becomes a small circle defined by its tangent cone;
- reflected/refracted terminal changes become arcs carried by the relevant
  optical surface chart;
- caustic support boundaries are projected from their compiled receiver
  regions.

Visibility tests certify these arcs once in world direction. Rotation then
clips the already-visible arcs to the new frustum and maps them to sensor
coordinates. It does not reshoot contour witnesses.

## Rotation update

For old and new view footprints `Q0` and `Q1` on the direction sphere:

1. Retain every certified angular cell intersecting `Q0 ∩ Q1`.
2. Construct cells only for the newly exposed strip `Q1 \ Q0`.
3. Clip cached visible-boundary arcs to `Q1` and transform them through the
   new pinhole chart.
4. Gather/interpolate `F_O(R1 d(p))` for output pixels.
5. Refine only cells whose new sensor footprint makes the inherited angular
   certificate too coarse for the pixel error limit.

For a small horizontal rotation `delta` and viewport angular width `W`, the
new expensive domain is approximately `delta / W` of the image, rather than
the whole image. At the current roughly 72-degree horizontal field of view, a
one-degree yaw exposes about 1.4 percent new angular area. Pixel writes remain
proportional to output size, but geometry and optical evaluation need not.

## Translation update

Translation changes parallax and the surface-to-eye direction, so it cannot be
handled by a coordinate swap alone. It can still be subsected:

1. Reproject each cached first-hit world point `x` to direction
   `normalize(x - O1)`.
2. Retain its owner when conservative depth intervals prove that no primitive
   bound can enter before `x` along the new ray.
3. Mark disoccluded cells, depth-order ambiguities, and silhouette bands as
   dirty; cast only there.
4. Reuse diffuse radiance at retained world points exactly.
5. Re-evaluate only directional glossy/metal response and optical chains whose
   incident/view direction changed beyond their certificate.
6. March changes into neighboring angular cells as a residual frontier until
   the display-energy bound is exhausted.

Thus translation invalidates a sparse parallax/BRDF frontier, while scene
illumination remains intact. Large translation can eventually dirty most of
the directional field; the method degrades to a complete camera solve without
corrupting it.

## Current measured opportunity

The retained 1920x1280 render spends about 59 ms compiling the viewer-origin
specular atlas and about 957 ms in camera raster/topology work. Rotation can
already reuse the 59 ms atlas exactly through the new cache interface. Moving
the terminal ownership/radiance representation from sensor rows to the
spherical field is what can subsect the remaining work. It also attacks the
current 2,339,064 full-resolution ownership classifications and 2,975,262
boundary-label queries at their cause: they are attached to a disposable
viewport instead of persistent world directions.

The output image still requires an output-sized gather unless a display API
performs the final warp. That gather is cheap and parallel. The objective is
not fewer pixel writes; it is to prevent those writes from causing geometry,
optics, or source integration to be solved again.
