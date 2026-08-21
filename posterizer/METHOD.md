# Perceptual bifurcation method

## Objective

Posterizer seeks a small set of display colors whose Voronoi cells divide the
important occupied perceptual volume of an image. It does not place colors
uniformly in the theoretical OKLCH gamut, because most of that gamut may be
absent from the image. It also does not equate importance with pixel count:
large flat backgrounds receive less weight than rare, structured foreground
colors.

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

Before constructing the tree, the image is converted into a structural color
field. A compact bilateral neighborhood averages nearby OKLab colors with a
weight that falls with both spatial distance and perceptual color distance.
Visible and transparent support never mix. The result reduces camera noise,
JPEG ringing, and antialiasing jitter inside a region without averaging across
a strong object boundary.

Every original pixel then receives an importance weight with two factors:

1. **Spatial information** combines OKLab Sobel gradient magnitude with the
   difference from a Gaussian local mean. The `detail_priority` control scales
   this factor.
2. **Sublinear population mass** bins occupied lightness, chroma, and circular
   hue. A bin containing `n` pixels contributes total mass proportional to
   `n^p`, where `p` is `population_exponent`. At `p=1`, area wins normally. At
   `p<1`, common background colors are tempered and rare colors gain relative
   influence.

Weights are normalized and robustly capped. The root then contains a
deterministic weighted sample of all visible image colors. Every leaf maintains
its assigned population and importance. A split proposal:

1. builds a tangent coordinate system from local lightness, chroma, circular
   hue displacement, and alpha;
2. tests the principal covariance direction and each coordinate direction;
3. uses weighted prefix sufficient statistics to find the exact lowest-SSE
   valid cut along each tested direction;
4. deduplicates those seeds and refines each with importance-weighted,
   deterministic two-means;
5. retains the cut with the largest weighted within-leaf distortion reduction.

All leaves compete in one priority queue. The proposal with the greatest
absolute perceptual gain is split next. A difficult skin-tone or illumination
interval may therefore receive several descendants while a nearly constant
background receives none.

When at least four colors are requested, Posterizer also constructs a proposal
tree whose root temporarily reduces lightness and emphasizes chroma and
circular hue. It measures which proposal node is both well occupied and least
represented by the ordinary tonal tree. That node becomes a reserved family
anchor. Posterizer tests each ordinary node's removal and replaces the one
whose loss adds the least weighted distortion. The final palette consequently
retains all but one tonal allocation while preventing total hue collapse.
`family_priority=0` disables this reservation.

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

## Spatial and raster stages

Palette learning uses the edge-aware structural field, but final assignment
returns to the original full-resolution pixels. Before assignment, Posterizer
measures the lightness residual from a compact Gaussian local field and adds a
controlled multiple back to the original lightness. This texture transport
pushes real high-frequency contrast across categorical palette boundaries;
unlike dithering, it does not manufacture texture in a flat field. The
`texture_priority` control sets its strength and `0` disables it.

The transported pixels are assigned to the shifted palette under the same
cylindrical distance. Small connected islands may then be absorbed into a
larger adjacent component, choosing a perceptually nearby neighbor normalized
by shared boundary length. Palette colors remain fixed during cleanup so the
intentional bifurcation shifts are not averaged away.

An optional spatial-mixing pass follows cleanup. For each flat-assigned pixel,
it projects the source color onto segments from that base node to its nearest
palette neighbors. Mixing is admitted only in a stable 3x3 label interior,
suppressed by source gradient, and scaled by the continuous segment's error
reduction. A deterministic serpentine error diffusion converts the desired
minority-color density into exact palette labels. Diffusion residuals travel
only between pixels with the same base/partner pair, so they cannot leak across
object contours, transparency, or color-family boundaries. This reconstructs
intermediate optical tones while the file still contains exactly the original
palette; `mixing_strength=0` disables it.

The cylindrical metric is evaluated algebraically as a small feature matrix
product, without allocating separate lightness, chroma, and trigonometric
pixel-by-palette tensors. Spatial cleanup labels all equal-color neighbor
connections in one sparse graph pass. Its cost therefore scales with pixels
and local edges rather than rescanning the image once for every palette color.

Shifted nodes are gamut-mapped and rounded to 8-bit display RGBA before final
assignment. Display-identical nodes are collapsed in stable palette order.
Assignment, cleanup, diagnostics, and rasterization consequently use the same
realizable colors rather than comparing pixels to an ideal node that cannot be
displayed. PNG retains alpha; JPEG is written as optimized 4:4:4 quality-95
RGB. No path tracing or SVG construction occurs in Posterizer.

## Relational chart diagnostics

Pixel error is not the primary perceptual question. Let `X(i)` and `Y(i)` be
the source and result at corresponding image sites in Cartesian OKLab, the
nonsingular metric form of OKLCH. For sampled site pairs, Posterizer reports

```text
sqrt(E[(|Y(i)-Y(j)| - |X(i)-X(j)|)^2] / E[|X(i)-X(j)|^2]).
```

Pairs include global relations and local spatial offsets from 1 through 32
pixels after a 1.5-pixel viewing filter. The diagnostics also report distance
correlation and the fraction of source relation energy collapsed below one
quarter of its original length. Finally, materially occupied source-hue
sectors are measured separately; the worst sector's alignment and relative
chromatic error prevent a low average from concealing a lost color-family
branch. This chart objective is the intended basis for further palette
allocation work; conventional RMSE remains only a secondary diagnostic.
