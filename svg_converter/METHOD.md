# Method study: transport-locked residual contours

This converter is a fresh implementation. It imports no code from the BFFT
segmenter or the Portsmouth archive. The source projects were studied as
method references, then reduced to the following independent design rules.

## Inputs studied

- `../segmenting_v3.py` and `../../viewer/SEGMENTING_V3.md`
- `../../../PortsmouthProject.zip`, inspected from an isolated temporary
  extraction, especially `boundary_trace.py`, `makima_quadratic.py`,
  `medial_transport.py`, and `shock_graph.py`

## Retained principles

From segmenting v3:

1. Decide low-frequency ownership on a reduced image.
2. Lift those owner IDs without inventing or deleting owners.
3. Reconsider only a narrow band around lifted interfaces at full resolution.
4. Explain residual error with children that are locked to one structural
   parent during construction.
5. Make the representation deterministic and inspect its cell hierarchy.

From Portsmouth:

1. Boundary points form ordered loops, not unordered edge clouds.
2. Topology is fixed before curve fitting.
3. Arc/event structure matters: corners are hard events and smooth spans may
   use a curve.
4. A curve earns its place only when a geometric error test accepts it.

## New logic basis

The converter represents an image as a quotient tree rather than a flat color
cluster map:

`coarse perceptual owner -> full-resolution interface correction -> residual child`

Structural colors are selected by deterministic farthest coverage in
premultiplied Oklab, then relaxed with an edge cost. The full-resolution pass
is active only in a dilation of the lifted owner interfaces. Residual children
are proposed from high-error pixels, but a child may claim pixels only inside
its immutable structural parent and only when it improves perceptual error by
a configured ratio.

Each final color mask is converted to the exact oriented boundary of its union
of raster squares. Closed loops preserve holes with SVG's even-odd fill rule.
Collinear events are removed and the remaining contour is simplified under a
distance budget. A vertex becomes a quadratic event only when its turn is
below the hard-corner threshold and its measured rounding displacement is
within the curve budget; otherwise it remains linear.

Before simplification, a periodic subpixel relaxation removes raster staircase
energy while pinning corners whose turn persists across a multi-pixel window.
Because SVG renderers can expose the page through antialiased cracks between
separately fitted adjacent paths, every filled contour also receives a narrow
same-color under-stroke. This seam closure is configurable and does not change
the region hierarchy.

Transparency is modeled separately from color ownership. In `cutout` mode,
source alpha is treated as raster coverage and thresholded before tracing; the
SVG renderer then recreates subpixel edge coverage around the vector path.
`preserve` keeps alpha as a region feature for genuinely translucent artwork.
The default `auto` mode selects cutout when partial-alpha pixels are confined
to the support boundary and preserve when translucency continues into broad
interiors. Near-zero palette regions are never emitted as SVG paths.

This differs from ordinary palette tracing in two important ways: full-scale
pixels cannot globally reshuffle the coarse topology, and detail colors are
conditional residual explanations rather than peer clusters competing across
the whole image.

## Limits

- It intentionally produces flat filled shapes; gradients and strokes are not
  inferred.
- Photographs become posterized illustrations rather than semantically
  editable scenes.
- Exact preservation of diagonal one-point contacts depends on SVG even-odd
  rasterization and can differ between renderers at extreme zoom.
- Premultiplied color prevents invisible RGB from affecting topology, but SVG
  shape opacity is one mean value per final region.
