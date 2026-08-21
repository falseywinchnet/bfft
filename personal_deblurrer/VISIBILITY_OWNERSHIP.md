# Positive visibility ownership and joint fold transport

## Ownership is induced in the latent domain

Each observation carries a non-negative sensor precision `w_i(p)` and a
positive exposure operator `A_i`. The latent ownership measure is not a hard
frame/layer assignment:

```text
pi_i(q) = A_i^* w_i(q) / sum_j A_j^* w_j(q),
pi_i >= 0,   sum_i pi_i = 1.
```

This is the amount of measured support each observation transports to latent
position `q`. A frame may own nearly all mass where another frame is occluded,
while both remain continuously active elsewhere. No class, layer label, or
winner is selected. The normalized shared-latent update is

```text
x <- x * [sum_i A_i^*(w_i y_i / max(A_i x, eps))]
         / [sum_i A_i^* w_i].
```

Scalar frame weights are the constant limiting case of the `N x H x W`
precision measure. Jointly unsupported latent points are reported and assigned
maximum visibility sensitivity rather than filled silently.

## A fold changes the chart, not the reconstruction law

For one observation, a nonpositive determinant of `I-grad(m)` invalidates the
single-valued barycentric pullback, so the single-image inverse still abstains.
For several observations, an individual fold is not necessarily a joint
information failure. The exact original `A_i` and `A_i^*` remain valid linear
operators even when the coordinate map folds.

`solve_spatial_field_consensus` therefore uses barycentric pullback only as an
invertible preconditioner. If any field folds, it applies the same positive
normal equation directly in the symmetric latent gauge and audits

```text
C(q) = sum_i A_i^* w_i(q).
```

The solve proceeds when joint coverage is positive. This is not a
warp-versus-mixing branch: the objective, exposure fields, adjoints, update,
and discrepancy law are unchanged; only an invalid coordinate preconditioner
is omitted.

## Cycle failure is uncertainty, not rejection

An attempted sharper forward/reverse visibility mask was falsified: it reduced
the moderate-disocclusion control by 0.96 dB. A failed correspondence often
means that a frame contains uniquely visible evidence, not evidence that
should be erased. Cycle and photometric closure therefore remain flow
uncertainty. Visibility ownership comes from transported measurement coverage.

A second tempting shortcut was also falsified. Seventeen positive local flow
particles around the dense solution were scored continuously and inserted as
extra atoms of the exposure measure. On the 64-pixel moving-layer control this
reduced the moderate result from 20.04 to 19.51 dB and the folded result from
12.30 to 11.90 dB. Motion-sheet ambiguity is not exposure mixing of one latent
appearance. The particle prototype was removed from active code; the next
multi-sheet representation must carry distinct latent appearance states while
retaining soft ownership.

## Measured moving-layer control

`visibility_results/results.json` moves a textured foreground over a stationary
background in symmetric captures, with read noise sigma 0.002. This is a
composite observation control, not a supplied segmentation; estimation sees
only the two rasters.

| Case | Best capture | Average | Positive ownership | SSIM |
|---|---:|---:|---:|---:|
| Moderate disocclusion | 21.873 dB | 24.059 dB | 28.994 dB | 0.9393 |
| Folded disocclusion | 20.685 dB | 22.843 dB | 25.617 dB | 0.8886 |

Five of six large-disocclusion trials produce an individual fold. All five
continue through the direct joint operator, have zero jointly unsupported
pixels, and improve over averaging by at least 2.454 dB. The M4 battery takes
4.68 seconds.

The remaining woven-chirp trial does not fold but loses 0.465 dB. Its failure
is upstream: the single smooth flow under-resolves a motion discontinuity. The
current ownership measure can route complementary coverage but cannot invent a
second motion sheet. A positive multi-sheet flow measure is therefore the next
representation gate; it must remain soft and permutation-symmetric rather than
classifying pixels into foreground and background.
