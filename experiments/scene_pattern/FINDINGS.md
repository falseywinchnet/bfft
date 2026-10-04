# First real-photo prototype — 2026-09-13

The actual forest segmenter now participates in alignment of six real Oxford
Graffiti photographs. The photos share physical content. The failed generated
forest composite is not used as validation or as a source of scene geometry.

The five adjacent pairs were estimated without reference homographies, SIFT,
a learned model, or an optical-flow fallback. The reference maps were opened
only after each estimate was finalized. Metrics below are transfer errors in
the original 800 × 640 moving image, on a uniform 40 × 32 reference grid
restricted to the reference-map overlap. The 1-based Oxford coordinate
convention is explicitly converted to normalized coordinates in `run.py`.

| Pair | Median error (px) | 95th percentile (px) | Within 3 px | Registration (s) |
|---|---:|---:|---:|---:|
| 1 → 2 | 0.682 | 1.326 | 100% | 2.679 |
| 2 → 3 | 1.116 | 1.553 | 100% | 1.681 |
| 3 → 4 | 1.044 | 1.687 | 100% | 2.651 |
| 4 → 5 | 1.339 | 2.161 | 100% | 1.686 |
| 5 → 6 | 1.674 | 5.564 | 73.0% | 1.861 |

The hardest pair retains a substantial error tail. These are planar-region
registration results, not global scene reconstruction or an empirical depth
model. Composition of adjacent maps is available, but loop consistency has not
been enforced and chained accuracy has not been certified.

## Contributions and costs

Analysis width was 256 (205 pixels high). Each image used five independent
causal-density segmentations: four texture coordinates and cartoon context.
Native Oklab Meyer decomposition took about 46 ms per image; complete analysis
was 0.97–0.98 s. Six-image analysis totaled 5.83 s and pair registration 10.56 s.
The recorded complete run took 157.86 s including full-resolution native CONV
image diagnostics and I/O. The roughly 141 s difference is predominantly the
current full-frame arbitrary-coordinate rendering path; it must not be omitted
from an end-to-end performance claim. No matched hardware-browser benchmark
has been run. These are single-run measurements, not latency guarantees.

The regional seed supplied 112, 57, 18, 61 and 45 inliers. Its unrefined median
errors were 2.99, 6.80, 18.58, 5.58 and 7.20 px. The winning final coarse
hypothesis came from the rotation/scale bank for the first two pairs and from
the region seed for the last three. This experiment therefore demonstrates the
combined regional/direct pipeline, not that segmentation alone solves matching.

## Ringing and order ablation

The standard and ringing delay arms used identical regional patches and the
same initial map. Both use subpixel correlation; the ringing arm first applies
a sharp 25-tap FIR. At the late stage, the ringing correction was accepted on
one pair. On pair 3 → 4 its median error fell from 1.064 to 1.044 px, while the
95th percentile rose from 1.285 to 1.687 px. The appearance gate can accept a
tradeoff that does not improve every independent geometric metric.

`study_early.py` separately tests the requested ordering: coarse regional fit,
then ordinary phase / ringing / radial correction, then identical finer
geometric annealing. The final branch errors are nearly the same. There is no
clear evidence here that the ringing variant beats ordinary phase registration.
Excluding the shared warped-texture construction, median early-probe times were
17.75 ms for phase, 24.00 ms for ringing and 16.27 ms for radial traces. The
unwound trace is useful as a geometric diagnostic, but its additional cost has
not yet been justified by a substantial real-photo improvement.

## Radial trace result

The independent analytic image test imposed scale 1.16 and rotation +23°.
The trace recovered 1.160227 and +22.998280°, with correlation 0.99935. A center
error of (+4, −2.5) pixels biased the recovered scale to 1.08228 and rotation to
+26.184°, reducing correlation to 0.753. The largest tested center error caused
an incorrect scale match. The trace therefore needs an accurate center or a
joint center-correction procedure; it is not automatically translation-invariant.

The angular direction is circular; logarithmic radius is finite-overlap.
Angular phase is preserved. The synthetic result does not establish invariance
to shear, perspective, noise, or occlusion.

## Native repair and review

A real-size query exposed a nested-dispatch deadlock in CONV's native thread
pool. Thread-local nesting now keeps inner work on its existing outer worker.
A timed subprocess test finishes at 1 and 4 threads with bit-identical output.
The focused suite plus existing native backend suite has 31 passing tests.

The local WebGL review was inspected at the halfway view and with the hardest
pair's checkerboard, and playback/controls were exercised. The reviewer samples
with bilinear GPU textures over the Python-estimated homographies. Its smooth
playback is not a browser implementation or timing of the Python matcher.

`results-summary.json` retains compact measurements, input checksums and the
exact source hashes of the measured main run. Full receipts, photos and visual
diagnostics are in the ignored `out/v2/` directory. Source URLs and archive hashes
are retained by the dataset fetcher.

## Six-photo fusion (2026-09-13)

The first actual fused atlas uses photo 3's coordinate frame and the union of
all six photographs. It is 960 × 431 pixels; 231,768 pixels have coverage and
63.8401% of those have at least two views. Average contribution shares for photos
1–6 are 3.03%, 5.48%, 9.90%, 14.90%, 24.20%, and 42.49%; these include the outer
single-view regions and are not independent quality scores.

Two direct anchor refinements were accepted using the image-only score (photos
2 and 5); the other three proposals were rejected. After geometry and fusion
froze, evaluation against Oxford's reference maps gave:

| Source photo sampled from anchor frame | Median error (source pixels) | 95th percentile |
|---|---:|---:|
| 1 | 1.321 | 2.636 |
| 2 | 1.453 | 2.371 |
| 3 (fixed anchor) | 0 | 0 |
| 4 | 1.044 | 1.687 |
| 5 | 1.044 | 2.076 |
| 6 | 1.525 | 2.920 |

This run took 53.406 s on the Mini, including 15.645 s of analysis and anchor
refinement and 36.848 s of native source sampling. It reuses the previous
registration receipt; initial collection registration time is not included.
Timing does not establish real-time operation. All reference errors apply to
samples inside the anchor frame's true overlap, not the entire extrapolated
union. Outer-footprint accuracy is not certified by these numbers.

The wall is visually coherent in the fused view. The disagreement map still
outlines many edges, and foliage/ground do not obey the single-plane model.
No objective ground-truth fused color image is available here, so there is no
claim that robust fusion has improved image fidelity over the weighted average.
The reviewer exposes both for comparison. Next work should reduce geometric
edge disagreement and retain source detail before extending to depth or claiming
that weighted blending solves visibility.

## Residual-field experiment (2026-09-13)

The final regional residual run accepted 2,795 local proposals on segmentation
sites (412, 504, 0, 619, 617, 643 across photos 1–6). It uses an explicit absolute
central-difference Jacobian for fractional displacement, avoiding coordinate-
relative steps that can disappear at the zero initialization in float32 samples.
The earlier exploratory run with relative steps remains in `out/residual-v1/`;
its numbers are not the final implementation's result.

With fixed exposure correction, native final sampling, common before/after
coverage and identical fusion, mean overlap Oklab disagreement decreased from
0.052718 to 0.048636 (7.744%). Per-source local contrast error decreased by
17.9%, 7.8%, 9.5%, 14.1%, and 14.2% for sources 1, 2, 4, 5, and 6. Mean fused
Oklab-L gradient magnitude increased 1.61%. This is a sharpness diagnostic, not
proof of improved detail fidelity. Sampled displacement-field Jacobians ranged
from 0.9329 to 1.1030; no fold was observed on the working lattice. That check
is not a continuum guarantee for the final interpolated field.

Crucially, evaluation-only geometric errors did not support accepting this as
a general geometry improvement:

| Source | Median before → after (px) | p95 before → after (px) |
|---|---:|---:|
| 1 | 1.321 → 1.560 | 2.636 → 6.200 |
| 2 | 1.453 → 1.592 | 2.371 → 3.243 |
| 4 | 1.044 → 1.216 | 1.687 → 2.027 |
| 5 | 1.044 → 0.944 | 2.076 → 2.650 |
| 6 | 1.525 → 1.459 | 2.920 → 3.238 |

Every geometric tail worsened. The experiment therefore remains a candidate
comparison, and the original fused view is preserved. Do not select a local
field using these ground-truth errors; use them to identify inadequacy of the
image-only objective. Neighboring held-out pixels do not prevent coherent
appearance-driven geometric bias. Stronger independent multi-view consistency
and sampling-footprint control are needed before promoting such corrections.

The final comparison run took 89.057 seconds on the Mini, including rebuilding
both baseline and candidate from original photos. Nine focused tests passed,
including independent known-shift recovery. Full receipts and per-region
proposals are in `residual-results.json`; `/fusion/residual/` exposes both
images and disagreement maps with a common color scale.

## Freely captured eleven-photo interior collection (2026-09-13)

The new collection has eleven overlapping portrait photographs and no supplied
reference geometry. Regional matching found 26, 33, 19, 68, 133, 79, 90, 123,
147, and 127 inliers across the ten consecutive pairs. Initial analysis and
registration took 38.635 s on the Mini.

The previous unrestricted projective chain severely stretched the first views,
and its >45% overlap screen dropped the valid regional seed for one join.
Fixed-overlap contrast refinement recovered that narrow join. A second preview
that merely projected the homographies to rotations was also poor. The retained
layout instead fits proper rotations directly to the regional inlier ray
correspondences, with robust reweighting and approximate EXIF-derived intrinsics.
This avoids treating a parallax-distorted projective matrix as a camera rotation.
The final rotation layout does not consume the contrast-refined homographies.

The retained native-CONV output is 1400 × 817. There are 810,040 covered pixels,
65.9644% of them supported by multiple sources. All eleven photos contribute
pixels to the graph-cut seam merge, with individual selected areas ranging
from 24,792 to 115,043 pixels. The final render, exposure adjustment and fusion
comparison took 169.246 s; initial registration and exploratory previews are
excluded. The earlier 1200-pixel linear preview took 5.170 s and must not be
reported as native-CONV performance.

The seam merge substantially reduces the visible duplicate laptop and other
near-object ghosting seen in broad averaging. It does not make the scene
geometrically consistent: wall and couch edges still have discontinuities,
near objects have parallax, brightness differs across some joins, and input
blur remains. Spherical projection curves straight scene lines. Neither a
calibrated 3-D reconstruction nor novel-view interpolation has been established.
No reference geometric error can be reported for these photos.

Five focused panorama/seam tests, plus nine adjacent fusion/residual tests,
passed. They cover known-rotation recovery from synthetic region rays, transform
direction, proper rotation matrices, valid-only seam selection, constant overlap
and unsupported pixels. They do not certify the captured scene's alignment.
Collection-specific images, registration, original hashes and render receipts
remain in ignored local data/output directories; the viewer is `/room/`.

## Ordinary-average deblurrer graph diagnostic (2026-09-13)

Recovered the relevant existing deblurrer implementation and its documented
center-first order: Fourier-circle relative centers are reconciled through a
robust multi-capture graph; deterministic transport precedes residual mixing.
The average-only phase estimator supplies centered mixing evidence but cannot
attribute its response uniquely to registration or recover absolute translation.
The scene integration imports these actual routines rather than replacing them
with the earlier pairwise residual fitter.

A 700 × 408 working re-render of the room yielded 69 valid local charts and
103 accepted pair measurements, including 24 nonconsecutive edges. Seventeen
charts have cycle evidence; the rest cannot independently test closure beyond
two-view consistency. The median equal-view center spread is 0.902 pixels,
the 90th percentile 2.008, and the largest 3.306. Units are working-panorama
pixels, not original-camera pixels. Unmeasured regions remain unknown.

Chart 43 illustrates why average blur cannot simply be renamed alignment error:
the two measured centers are about (-3.29,-0.36) and (+3.29,+0.36), but the
average-only phase statistic is 1.97 px RMS and the individual-view median is
2.12 px RMS. The method sees texture and intrinsic blur as well as displacement.
That chart has no cycle; its 0.00 graph residual is not confirmation of the
estimated displacement. Chart 44 has three views and a 0.545 px graph-closure
RMS, exposing a conflicting set of local relative constraints.

The conservative, smoothed zero-mean field was frozen before rerendering source
samples through native CONV. Both arms use common support, frozen exposure gains,
ordinary linear-light averaging, and no seam selection or sharpening inverse.
Mean overlap Oklab disagreement decreased from 0.066109 to 0.064668 (2.18%).
Mean average-image gradient magnitude rose from 0.034968 to 0.035357 (1.11%).
Local contrast agreement improved on 20 of 31 evaluable source pairs; the best
change was -14.63%, and the worst was +3.14%. These are image consistency
measurements, not ground-truth geometric accuracy.

The largest applied correction was 1.138 working pixels after shrinkage.
The sampled field Jacobian minimum was 0.897; no fold was observed on that
lattice. This is not a continuum guarantee. Local translations and their smooth
interpolation do not resolve independent depth layers or visibility ownership.
The correction remains an experimental candidate alongside the untouched
original room fusion.

The estimation/diagnostic stage took 1.399 s on the Mini. Full baseline and
candidate native resampling plus checks took 88.056 s total. Nineteen focused
and adjacent tests passed, including known fractional displacement direction,
graph permutation, disconnected gauges and deliberately inconsistent cycles.
The `/room/alignment/` viewer overlays region measurements on the ordinary
average and exposes per-photo measured versus applied offsets, blind phase
statistics, graph closure and a magnified local crop. All private observations,
fields, measurements and source hashes remain in ignored output directories.

## Targeted affine floor alignment

The user-identified doubled tile line beside the chair motivated a bounded
two-view experiment on IMG_1441 and IMG_1443. The floor polygon was selected
manually; correspondence parameters were estimated from the observed images.
The previous translation chart measured about 4.61 working pixels of relative
separation but applied only about 1.44 pixels after confidence shrinkage.
Neighboring charts also implied different displacements across the floor.

The new solver surveys competing shifts, then fits six local affine parameters
per hypothesis. Separate spatial blocks are used for fitting, selecting a
hypothesis, and reporting its error. The selected local translation is
(-5.674, 2.336) pixels with linear map [[1.04031, -0.04876],
[0.03187, 1.04079]]. Symmetric matrix half-transforms distribute the relative
correction between the two views; the field fades outside the selected floor.
The two changed views are resampled through native CONV, retaining the previous
corrections for the other nine views and ordinary linear-light averaging.

On common floor support, line-contrast disagreement was 0.062179 originally,
0.060427 after the conservative correction, and 0.022897 after the affine
correction: 62.1% below the previous correction. Reporting-block feature error
fell from 0.517477 to 0.302219 (41.6%). The doubled diagonal line visibly
converges in the comparison. These are photometric consistency measurements,
not independently measured geometric errors; spatial blocks from the same
captures are not an independent dataset. This does not establish automatic
surface discovery, correct occlusion handling, or whole-scene registration.

The two-source native experiment took 10.179 seconds on the Mini. Sampled
Jacobian minima were 0.820 and 0.712, with maximum source corrections of 4.145
and 4.244 working pixels. No sampled fold was observed. All 22 focused tests
passed, including synthetic affine recovery and the symmetric transform split.
The private review, alternatives, fields, and receipt are retained at
`out/v2/room/alignment/floor/`, served by `/room/alignment/floor/`.

## Shared references for the bronze lamp, laptop, and doorway lamp

Three additional manually localized regions now fit contributing observations
against a common local reference (source indices 2, 5, and 7 respectively).
Two bronze-lamp, six laptop, and three doorway-lamp corrections were accepted.
These are star-shaped local constraints, not a complete all-pairs global solve.
Reference choices define coordinates and do not designate ground-truth images.
One partial bronze-region observation contains no visible shaft and is excluded
from fitting; it remains in the ordinary average with its previous coordinates.

The bronze lamp's repeated ornaments exposed a wrong minimum in texture-only
matching. Broader Oklab shape/color evidence and a wider affine search recover
its changed proportions. Adaptive transition widths retain full regional
corrections while avoiding sampled folds outside the objects. Incremental
maps compose by evaluating the prior field at the new coordinates, rather than
simply adding displacement arrays. The earlier floor polygon's coordinates
in sources 8 and 10 remain exactly unchanged (maximum difference 0.0).

Frozen preview fits were rendered through native CONV for both baseline and
candidate. Under common source support, pixel-count-weighted source/reference
local contrast error changed as follows:

| Region | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Bronze lamp | 0.134235 | 0.093298 | 30.5% |
| Laptop | 0.119782 | 0.082263 | 31.3% |
| Doorway lamp | 0.156868 | 0.082830 | 47.2% |

The final images retain ordinary linear-light averaging and frozen gains.
The bronze shaft, main laptop frame, and doorway lamp visibly converge;
shade contours, laptop interior detail/lower edge, and other depth boundaries
still have residual disagreement. This is an improvement in photometric
consistency, not independent geometric validation or automatic object discovery.

Fitting with linear preview sampling took 14.506 seconds on the Mini; the
complete native baseline/candidate render and evaluation took 85.564 seconds.
Minimum sampled field Jacobian was 0.316622. All 26 focused tests passed,
including broad-color displacement recovery, map composition, fold detection,
and JSON polygon round-trip compatibility. The saved-fit polygon handoff bug
was fixed and the complete native run repeated successfully. Private results
and the browser review are at `out/v2/room/alignment/structures/`; use
`structure_report` to build its comparison, source hashes, and viewer.

## Shade coverage and cross-region audit

The structure corrections are image-estimated transforms in manually selected
regions, with scene-specific reference choices, a bronze-view exclusion, and
floor protection. They are not yet an automatic region/visibility solution.
The shade audit deliberately keeps the existing fused view unchanged.

The earlier bronze region measured the shaft. For source index 1 (photo 2),
8118 of 8675 inspected common shade pixels (93.6%) lie outside that measured
support. The incremental structure field reaches 16.491 pixels there, with
an extrapolated median of 6.575 pixels. Source index 0 contributes 979 common
shade pixels but received no shaft correction, because it has no visible shaft.
Source index 3 also has predominantly extrapolated shade coverage. A positive
sampled Jacobian establishes neither supported correspondence nor visibility.

The shaft correction did not create all the shade disagreement: on common
support its shade contrast error improved from 0.041496 to 0.037755 in photo 2
and from 0.035256 to 0.026177 in photo 4. It nevertheless left the shade outside
its own fitting and acceptance domain.

Controlled shade-only affine fits against photo 3, using the same reusable
structural matcher, yield:

| Moving photo | Shade contrast before | After | Reporting-block feature reduction |
| --- | ---: | ---: | ---: |
| 2 | 0.037883 | 0.029517 | 60.4% |
| 4 | 0.027760 | 0.017779 | 50.3% |

Neither selected fit reaches its parameter bounds. Photo 1 has no admissible
affine hypothesis under the current overlap, search, and geometric checks;
that is not proof that the observation is intrinsically unalignable.

Extending photo 4's shade-only affine map over the shaft raises shaft contrast
error from 0.056789 to 0.073644 (29.7%), while shade error falls 36.0%. Photo 2's
corresponding shaft error instead falls slightly, 0.110688 to 0.107276. Thus the
measured conflict is source-specific. Missing coverage and a cross-region
tradeoff are established; a necessary projective/3D model is not established.

`registration_audit.py` provides object-independent evidence-domain and regional
contrast diagnostics. Tests distinguish a smooth, nonfolding but unsupported
translation from measured support, detect a correction helping one region while
hurting another, and report empty support as unknown. The 29 focused tests pass.
These audits do not yet implement automatic region discovery or replace the
solver's acceptance rule. All shade model images are linear-sampled diagnostic
pairs, not newly accepted native-CONV fused output. Review them at
`out/v2/room/alignment/shade/`, served by `/room/alignment/shade/`.

## Joint distortion-basis and continuation experiment

The scene-independent `distortion_basis` fitter now searches affine,
projective, and quadratic coordinate maps. It balances the observed regions
instead of allowing shade area to dominate shaft area, retains multiple
initial shifts, and uses normalized convolution near missing-image boundaries.
The room experiment supplies the same manually selected shade/shaft regions and
reference photo 3. It does not supply correspondence points or fitted parameters.

The first boundary-nearest extension introduced folds outside the observed fit
support despite positive Jacobians inside it. A screened harmonic continuation
now holds observed displacement values fixed and solves the exterior field with
zero canvas-boundary displacement. Accepted solves have relative residuals below
1e-6. Sampling the original captures directly removes the extra interpolation
from candidate acceptance. A final regional holdout gate retains the previous
map if either region regresses by more than 2%; it does not search further
against a failed holdout. That holdout is acceptance evidence, not an independent
post-acceptance test.

Native CONV results on fixed common support:

| Photo | Accepted basis | Shade contrast reduction | Shaft-area contrast reduction |
| --- | --- | ---: | ---: |
| 1 | affine | 32.2% | 1.9% |
| 2 | prior map retained | 0.0% | 0.0% |
| 4 | quadratic | 38.7% | 43.5% |

Photo 4's quadratic result improves both regions, unlike the earlier isolated
shade affine experiment that hurt the shaft. The wider search also finds an
admissible candidate for photo 1; its earlier rejection was a model/search
outcome, not intrinsic unalignability. Photo 2's candidate fails the final
shaft holdout and remains unresolved with this experiment. The lower region's
available image pixels do not guarantee that every capture sees the same shaft
surface, especially photo 1. Residual scores are photometric consistency, not
verified geometric correspondences or recovered depth.

The full shade is visibly still doubled, with photo 2 a major unresolved
contributor. This candidate must not be described as a completed shade alignment
or a global arbitrary-pose reconstruction. Ordinary linear-light averaging and
frozen gains are retained; no seam or sharpening change masks the residual.
The native baseline/candidate render and evaluation took 85.992 seconds, and
the minimum sampled composed Jacobian was 0.246421. The 35 focused tests pass,
including independent known-projective recovery, harmonic convergence with fixed
evidence, and region nonregression. Results are in the ignored
`out/v2/room/alignment/joint/` folder, served at `/room/alignment/joint/`.

## Laptop diagnostic and automation boundary

The owner identifies a real drape behind the shade. The preceding visual claim
that the full shade remained doubled was too strong: those contours alone do
not establish registration error. Retain the drape as scene structure.

The laptop diagnostic starts from the joint fields, with a manually chosen
region and reference photo 6. Four capture corrections pass the regional gates:
photos 3 and 9 use affine maps, photos 4 and 5 quadratic maps. Photos 7 and 8
retain their prior maps. Fixed-support, overlap-weighted local contrast error
falls from 0.0859535152 to 0.0815702066 (5.10%). The native CONV baseline and
candidate render takes 85.53 seconds. Minimum sampled composed Jacobian is
0.203618; displacement changes and rendered pixel differences outside the
bounded laptop domain are exactly zero. All 37 focused tests pass. Residual
registration remains visible; this is not a completed laptop reconstruction.

Partial-visibility and original-atlas fitting experiments for photo 7 accept no
additional correction. Its original capture includes camera blur and crops out
the physical left laptop edge. A capture boundary must not be assumed to be an
object edge; rejection does not establish that no better alignment exists.

These object-directed experiments are supervised diagnostics. Their reusable
fitting and continuation components do not remove manual region, reference,
protection, and experiment-selection dependencies. AUTOMATION_CONTRACT.md now
requires an end-to-end, fixed-policy collection method without these inputs.
The room is development data; even an automatic replay will not establish
unseen-collection generalization. The laptop reviewer and receipt explicitly
record this diagnostic status.
