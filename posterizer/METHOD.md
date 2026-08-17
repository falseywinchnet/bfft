# Perceptual bifurcation method

## Objective

Posterizer seeks a small set of display colors whose Voronoi cells divide the
occupied perceptual volume of an image. It does not place colors uniformly in
the theoretical OKLCH gamut, because most of that gamut may be absent from the
image.

For OKLCH colors `(L1, C1, h1)` and `(L2, C2, h2)`, the implemented squared
color distance is

```text
wL^2 (L1 - L2)^2
+ wC^2 (C1 - C2)^2
+ wH^2 4 C1 C2 sin^2((h1 - h2) / 2)
+ wA^2 (alpha1 - alpha2)^2.
```

The sine term makes hue circular: angles immediately to either side of zero
remain close. When `wL = wC = wH = 1`, the first three terms are exactly
squared Euclidean distance in OKLab. The controls therefore adjust a known
perceptual baseline rather than inventing an unrelated metric.

## Binary palette tree

The root contains a deterministic sample of all visible image colors. Every
leaf maintains its assigned population. A split proposal:

1. builds a tangent coordinate system from local lightness, chroma, circular
   hue displacement, and alpha;
2. tests the principal covariance direction and each coordinate direction;
3. initializes cuts at the 35th, 50th, and 65th percentiles;
4. refines every valid cut with deterministic two-means iterations;
5. retains the cut with the largest within-leaf distortion reduction.

All leaves compete in one priority queue. The proposal with the greatest
absolute perceptual gain is split next. A difficult skin-tone or illumination
interval may therefore receive several descendants while a nearly constant
background receives none.

This is a locally optimized binary quantizer, not a claim of globally solving
unconstrained k-means. The global k-color problem is non-convex. The useful
properties are deterministic proposals, an explicit distortion gain, and
allocation across the entire active tree rather than a fixed number of
children per initial color.

## Shifted display nodes

An accepted child is first placed at the centroid of its occupied population.
The final display node is then

```text
parent + separation * (child centroid - parent).
```

At separation `1.0`, nodes are measured centroids. A value above one
exaggerates the final bifurcations without changing their learned directions.
After shifting, lightness and alpha are clipped to their physical intervals.
Out-of-gamut RGB colors retain OKLCH lightness and hue while chroma is reduced
by binary search until the node lies inside sRGB.

## Spatial and vector stages

Full-resolution pixels are assigned to the shifted palette under the same
cylindrical distance. Small connected islands may then be absorbed into a
larger adjacent component, choosing a perceptually nearby neighbor normalized
by shared boundary length. Palette colors remain fixed during cleanup so the
intentional bifurcation shifts are not averaged away.

Finally, the inherited converter-v2 compiler groups every disconnected region
of a color into one compound even-odd path, serializes exact lattice edges
with compact `H`/`V` commands, and produces deterministic SVGZ.
