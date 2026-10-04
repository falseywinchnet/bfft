# M4 follow-up: throughput, visible color-area, and thin structures

This is the second-stage investigation, following FINDINGS.md. The useful result
is a faster simple coverage shader and a bounded, accurate visibility extension.
The latter still misses the FXAA-like speed target. Neither is a universally
optimal Metal shader or a general replacement for the existing CONV operator.

## Metal throughput

`geometry_gpu` accepts an optional `fast` or `half` argument. The latter uses
fast math plus a half-precision edge CDF; positions, clipping and integration
remain float. Retained runs are `fastmath/` and `half/` under
`output/support_geometry/conv_fast_aa/`. Earlier strict results remain in
`geometry_final/`. Median GPU milliseconds for the additive boundary path:

| Scene | Strict | Fast float | Fast + half CDF | Point + FXAA in fast-float run | MSAA4 in original strict run |
|---|---:|---:|---:|---:|---:|
| 1080p sparse | .0716 | .0643 | .0642 | .2683 | .0682 |
| 1080p dense | .3887 | .1880 | .1891 | .3958 | .0818 |
| 4K sparse | .3535 | .3270 | .3198 | 1.1403 | .3439 |
| 4K dense | 1.5218 | .7312 | .7348 | 1.6231 | .3712 |

These are separate benchmark runs, not a claim of exact cross-run ratios.
Fast math gives a substantial dense-scene improvement; half CDF has no convincing
additional benefit. Both pass the existing geometry gates. Fast math changes
shared-edge output by up to one RGBA8 code relative to the independent reference
(previous strict shared-edge maximum was .000786). This is not bit equivalence.
The existing tests retain overlap failures for this simple additive path; enabling
fast math does not repair its visibility model. Hardware MSAA remains faster on
the dense simple scenes. Input upload and application scene preparation are not timed.

Apple's [Metal optimization guidance](https://developer.apple.com/videos/play/wwdc2020/10632/)
motivated the precision experiment. We measured command-buffer GPU duration;
we did not collect hardware occupancy/counter evidence. Thus register pressure
is a hypothesis, not a diagnosed hardware bottleneck, and these are the best
measured variants in this experiment, not proof of global shader optimality.

## Visible color-area, including crossings within a pixel

`visibility.metal` integrates each primitive's color over its actually visible
subset of the pixel. Input depth and RGB are affine functions. The input contract
is a complete list of at most four positive-winding triangles per 32×32 tile.
CPU packing rejects overflow; no primitive is silently discarded. Generic
binning, arbitrary depth complexity, perspective-correct texture sampling,
transparency, and an overflow fallback remain unimplemented.

For receiver i, let B_i be its triangle intersected with the pixel. For every
other primitive j, its occluding set is its triangle intersected with the depth
halfplane z_j ≤ z_i. Exact coplanar ties use the lower primitive index. The
visible moments follow finite inclusion-exclusion over those occluding sets.
Every intersection is convex, so direct plane clipping plus Green's polygon
formulas gives area A and first moments Mx, My. For affine RGB c0+cx*x+cy*y,
output contribution is c0*A+cx*Mx+cy*My. There is no solver, iterative fit,
image-feature classification, or extra source analysis.

The independent CPU oracle uses successive convex subtraction into disjoint
pieces instead of inclusion-exclusion. Final oracle arithmetic promotes the
supplied float32 geometry to float64 before forming planes or moments. Earlier
receipts used float32 plane formation; retain those as historical, and use
`visibility_validated/` for the final quality gate.

All 16 invariant tests pass on the M4. All 72 quality records (24 scenes ×
three GPU modes) pass the 1e-4 gate; the largest RGB absolute error against
the final float64 oracle is 1.8075542e-06. This is a measured bound on these
fixtures, not a universal floating-point error guarantee.

Four implementations are retained:

1. `visibility_v1`: direct subsets over all supplied primitives.
2. `visibility_v2`: direct whole-pixel support/depth certificates and subset masks.
3. `visibility_v3`: lightweight certificate pass, atomic queue, GPU-generated
   indirect dispatch, exact integration only for unresolved pixels.
4. `visibility_v4`: additionally remove full-pixel occluders from subset expansion.
   Such an occluder simply clips the receiver by the opposite depth halfplane.

The queue has capacity equal to the pixel count, with at most one append per
pixel. Timing includes counter clearing, classification, appends, indirect
argument construction, exact integration, and final RGBA32F writes. Separate
encoders and tracked resources establish dispatch dependencies. No CPU readback
is needed to launch the boundary work. This is a geometry-driven coverage pass;
it does not invent lines missing from a resolved image.

Median 1080p GPU milliseconds, repeating each complete 32×32 geometry tile:

| Scene | First exact kernel, fast | Final dense, fast | Final queued |
|---|---:|---:|---:|
| Single | 1.505 | 1.494 | .724 |
| Duplicate opaque surface | 6.702 | 4.341 | 1.317 |
| Overlap | 5.631 | 3.073 | 1.918 |
| Crossing depth planes | 6.052 | 3.687 | 2.398 |
| Shared edges | 5.064 | 2.989 | 1.621 |
| Thin crossing | 1.943 | .812 | .646 |
| Affine color | 6.006 | 3.690 | 2.395 |
| Four layers | 60.043 | 10.116 | 7.944 |

Seven alternating repeats per mode; timing JSON retains the maximum (nearest-rank
95th percentile at n=7). Between 6% and 24% of pixels reach the queue in these
scenes. This is a 7.6× four-layer improvement over the first implementation,
but the target is not met. The scenes/output format differ from the RGBA8 simple
coverage benchmark above: do not treat their timings as a matched FXAA comparison.
Tile-list preparation and upload are excluded, so this is not total frame time.
No temporal/motion quality claim follows from these static visibility tests.

## What the thin-line augmentation test establishes

`augmentation.py` reuses the existing raw current R and admitted current J:

    reconstructed(lambda) = synthesize(source, J) + lambda*synthesize(0, R-J)
                         = synthesize(source, J + lambda*(R-J)).

The equality measured within 3.33e-16; the residual is exactly zero at source
nodes. This augmentation can be fused into coefficient formation and requires
no new source analysis or separate image pass. The experiment evaluates fixed
lambda values 0, .25, .5, .75, 1 on 19 sine frequencies × 16 phases, steps and
three strip widths. It is a 1-D reconstruction diagnosis, not GPU AA timing.

It fails as a general improvement. Mean squared error at .4, .45 and .5 cycles
per source pixel is respectively .0135882, .0321609 and .0625236 for every tested
weight. Restoring the admission residual recovers essentially nothing in that
band. Step overshoot grows to 2.52%, 5.04%, 7.56%, 10.08% for weights .25 through 1.
No nonzero weight is promoted.

Ringing-free does constrain reconstruction, but is not synonymous with inability
to represent all Nyquist-frequency data: the alternating 0,1 sequence retains
full peak-to-peak amplitude with zero excursion in admitted CONV. Arbitrary
continuous phase at Nyquist is a different requirement and is unidentifiable
from samples. A thin line entirely between source points likewise creates no
source current; its residual is zero. These observations separate the compact
proposal's limitations, admission losses, and missing observations.

A useful augmentation therefore needs information already retained beyond the
resolved samples. The geometry-driven boundary queue demonstrates such a path:
the tested ~.16-pixel thin triangles retain their visible area and color without
introducing ringing. It is a prototype with a strict supplied-geometry contract,
not a completed general CONV thin-line recovery feature. Future speed work should
reduce the remaining exact boundary integration cost; another image-analysis
pass or unconstrained residual boost does not address this evidence.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
 'sh experiments/conv_fast_aa/build.sh && \
  python3 -m experiments.conv_fast_aa.visibility generate --out /tmp/conv_visibility && \
  /tmp/conv_fast_aa_build/visibility_gpu experiments/conv_fast_aa /tmp/conv_visibility && \
  python3 -m experiments.conv_fast_aa.visibility analyze --out /tmp/conv_visibility'

/Users/ultimussecundai/.local/bin/m4build -- sh -c \
 'python3 -m experiments.conv_fast_aa.visibility generate --random-cases 16 --out /tmp/conv_visibility_validation && \
  /tmp/conv_fast_aa_build/visibility_gpu experiments/conv_fast_aa /tmp/conv_visibility_validation quality && \
  python3 -m experiments.conv_fast_aa.visibility analyze --out /tmp/conv_visibility_validation'

/Users/ultimussecundai/.local/bin/m4build -- python3 -m experiments.conv_fast_aa.augmentation \
 --out /tmp/conv_augmentation

/Users/ultimussecundai/.local/bin/m4build -- python3 -m unittest \
 experiments.conv_fast_aa.test_core experiments.conv_fast_aa.test_geometry \
 experiments.conv_fast_aa.test_visibility experiments.conv_fast_aa.test_augmentation -v
```

Copy each result immediately using the `m4host` selector. Rendering retained
plots is local: `.venv-jpeg/bin/python -m experiments.conv_fast_aa.plot_followup`.
