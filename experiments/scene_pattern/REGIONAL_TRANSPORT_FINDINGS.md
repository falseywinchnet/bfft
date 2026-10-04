# Continuous regional transport on the walking collection

This development run uses the private 28 panoramas and 92 ordinary photographs.
The acquisition and rejected single-origin interpretation remain specified in
`CAPTURE_GEOMETRY.md`. No object annotations, hand-picked repair regions,
station labels, or old flat-atlas coordinates enter this path. The collection
has guided development; it is not an unseen generalization benchmark.

## What changed

Independently segmented region centers need not identify exactly the same
physical point. `region_transport` fits the image pattern within each proposed
region, allowing a full local affine map and sub-region displacement. It uses
Oklab color, the existing native Meyer/forest-segmentation proposals, normalized
patch correlation, separate training/checking pixels, an independently fitted
reverse map, and a translation-information test for aperture ambiguity.

`dense_transport` grows those maps on an eight-pixel lattice. It checks continuous
composition A→B→C against A→C, including the local Jacobian, rather than requiring
the three independently segmented centers to coincide. `dense_scene` chooses a
mutually compatible source clique at each region. It never combines two source
branches merely because both agree separately with the reference.

`attach_fields` expands through capture triangles, starting with panorama charts.
A photograph can contribute to several panoramas; it has no exclusive assigned
station. Capture-level graph edges require at least 12 cycle-supported grid
sites in both directions. Three fixed propagation rounds were run.

`scene_pose` now competes a planar-point camera initializer with ordinary 3-D DLT.
Wall/ground points can make the latter rank deficient. The planar initializer
does not flatten the scene: every candidate is fitted and checked against the
actual proposed 3-D points. Duplicate physical target sites cannot multiply pose
votes, and target-site holdouts are excluded from initialization and refinement.

## Retained results

All results below are copied back from the selected M4 Mini. `out/` and `data/`
are private, ignored artifacts; source hashes and policy receipts are retained.

| Stage | Retained output | Result |
|---|---|---|
| Regional candidates, all captures | `out/capture-transport-all-v2/` | 1,446 candidate pairs; 563.1 s recorded pass |
| Dense panorama fields | `out/dense-transport-v1/` | 90 pair fields; 28,532 cycle-supported directed grid sites; 162.7 s |
| Panorama fusion charts | `out/dense-scene-v1/` | 10,245 anchored regions; all 28 panorama references rendered |
| Photo attachment | `out/attached-fields-v1/` | 544 pair fields; 81,645 cycle-supported directed sites; 303.4 s |
| Attached fusion charts | `out/attached-scene-v1/` | 30,048 regions across 118 capture charts; 28 panorama references rendered in 24.7 s |
| Panorama-only physical hypothesis | `out/dense-points-v1/` | 7 camera poses; 1,216 depth proposals, of which 893 pass the all-available-view consistency audit |
| Physical hypothesis after attachment | `out/attached-points-v1/` | 8 camera poses; 1,282 depth proposals, of which 888 pass that audit; 43.9 s including evidence construction |

These are separate stage timings, not an end-to-end or real-time benchmark.
They reuse prepared image features and candidate evidence.

The capture graph has one component of **112 captures: 28 panoramas and 84
ordinary photographs**. Eight captures do not pass the graph-support gate:
IMG_1477, IMG_1478, IMG_1489, IMG_1491, IMG_1505, IMG_1519, IMG_1537, IMG_1538.
Small local matches exist for some of these, which explains why 118 capture
charts contain regional observations. Neither count means 118 recovered poses.

**60 ordinary photographs actually contribute substantial pixels to the 28
rendered panorama views.** The remaining attached photographs may participate
through intermediate photo charts. The current renderer does not compose an
arbitrarily long graph path into permission to fuse those photographs.

## Rendered comparison

The control and corrected output use identical photographs, weights and valid
support. The control translates prior region-center predictions; the corrected
output adds image-fitted displacement and affine shape. Both use a linear-RGB
ordinary weighted average, with no seam selection, sharpening, or robust
outlier-suppressed averaging. Output sampling is a linear preview, not CONV.

Across all 28 panorama views, the area with at least two substantial observations
ranges from **8.55% to 40.87%**, median **21.52%**. Substantial means an additional
observation has more than one-quarter of the reference weight. Median color
dispersion reduction is **36.09%**; per-view reductions range **21.72–49.38%**.
These are development photometric measurements, not independent geometric truth.

The first reference, IMG_1444, has 40.87% two-view support, 40.67% three-view
support, and 27.97% reduced dispersion. IMG_1458 has 33.16% two-view support and
41.71% reduced dispersion. Every panorama remains in the viewer, including low
coverage cases. Unsupported pixels remain the original reference photograph.

## Geometry and remaining work

The latest physical hypothesis contains IMG_1462, IMG_1466–1472. All eight are
panoramas; none of the ordinary-photo poses passes the current growth process.
The seed is IMG_1468 / IMG_1470, with an arbitrary unit baseline. Additional-camera
holdout median angular errors span about 0.28–0.87 degrees. The 888 promoted
points have positive depth and at most 1.2 degrees reprojection error in every
available fitted view, with at least three such observations. The other 394
points remain tentative. Counts from the two runs describe different discovered
track sets and are not an exact before/after point-survival comparison.

Camera calibration is still nominal. Discovery and seed selection use the
collection, so the pose holdout is conditional, not independent validation.
Local cycle closure can coexist with a globally incorrect physical model.
Panorama stitching distortion, visibility, depth boundaries and camera intrinsics
remain unresolved. Regional source-view fusion and sparse physical hypotheses
are separate outputs; the former is not rendered from the latter's 3-D points.

The full objective remains one shared, visibility-aware scene with all captures
and movable viewpoints. This run establishes denser automatic pattern transport,
nonexclusive photo attachment, useful local ordinary-average fusion, and expanded
partial geometry. It does not complete the 120-image quasi-3-D scene.

## Checks and demonstration

`test_region_transport` and `test_multiview` cover affine recovery, different
segmentation centers, flat/black support, aperture ambiguity, unrelated-image
rejection, continuous cycles, incompatible branches, capture attachment support,
equal-support fusion, independent origins, planar camera initialization, duplicate
votes, held-out pose failure, and poisoned third-camera observations.

The local reviewer is `/capture/transport/` on port 8876. It exposes the ordinary
before/after average, reference-only control, support overlay, every panorama,
source attribution, unresolved inputs and the current conditional geometry.
Its automatically chosen inspection crop is an evaluation display only and does
not feed back into alignment. Commands are in `README.md`.


## Observation completion and movable-surface diagnostic

The next pass identified a representation bottleneck: the eight-camera cloud
contained only 12 observations in its best-supported ordinary photograph. The
pose fitter therefore never attempted ordinary-photo recovery. The dense fields
contained more evidence than those short tracklets exposed.

`scene_completion` extends each track through original measured observations.
At least two distinct source cameras must predict the same new image location,
within a 1.5-pixel pairwise diameter and a strict majority of available paths.
Each path also passes independent reverse-position and Jacobian closure checks.
New observations cannot vote recursively in the same pass. Conflicting paths
are retained in the completion diagnostics, not averaged into a new location.

Retained outputs:

- `out/completed-scene-v1.json`: 17,397 added observations, from 19,197 multi-path
  candidates; 892,077 attempted paths; 2.58 seconds on the Mini.
- `out/completed-points-v1/`: 30,048 tracks, 493 supported pair hypotheses,
  **17 recovered cameras**, **1,620 point proposals**, **1,436 points consistent
  in every available fitted view**. Evidence and pose computation took 90.09 s.
- The selected seed changed to IMG_1444 / IMG_1462. This is a different
  conditional reconstruction, not merely nine poses appended to the previous
  eight-camera winner.
- Four ordinary-photo poses now pass: IMG_1490, IMG_1522, IMG_1523, IMG_1542.
  The remaining 13 fitted cameras are panoramas. All 120 physical poses remain
  the objective; graph attachment must not be substituted for pose recovery.

`scene_surface` makes local Delaunay triangles among those consistent points
in each supporting image. It rejects image edges over 24 feature pixels and
vertex camera-range ratios over 1.5. Exact shared triangles receive one source
texture, selected by image sampling area. It does not fill gaps or claim
measured interior depth. The first output has **5,325 local triangles**, with
17 texture sources. Overlapping fragment interiors are not globally reconciled.

The local `/capture/scene/` page is a WebGL diagnostic with orbit, capture-camera
viewpoints, source textures, point display, camera positions and depth testing.
Browser inspection from both orbit and an ordinary-photo viewpoint shows
**substantial surface folding and stretching**. The pointwise image-space gate
is too weak to establish a correct surface. This is useful failure evidence,
not a finished quasi-3-D reconstruction or a replacement for the ordinary-average
comparison. The displayed geometry remains the pre-bundle 17-camera result.

### Joint-refinement audits

`scene_bundle` jointly varies camera poses and multi-view points, fixing the
seed camera frame and original seed-baseline length. Duplicate image-site votes
are capped. A physical-site holdout is excluded from optimization. The calibrated
branch additionally varies cylindrical horizontal/vertical focal scales and
vertical principal point under fixed weak priors and bounds. It minimizes pixel
reprojection errors, so focal changes cannot win merely by shrinking angles.

Both branches initially reached their 100-evaluation cap. The fixed follow-up
allowed 2,000 evaluations and retained the candidate arrays, without selecting
iterations against holdout scores. They stopped by cost tolerance (status 2),
not the evaluation cap; large reported optimality values mean this is not a
certificate of a tightly stationary or global optimum.

| Candidate | Evaluations | Training median before → after | Holdout median before → after | Holdout p90 before → after | Applied |
|---|---:|---|---|---|---|
| Fixed nominal camera model | 326 | 0.288° → 0.150° | 0.286° → 0.365° | 0.937° → 1.429° | No |
| Cropped/anisotropic cylindrical model | 689 | 0.876 → 0.413 px | 0.831 → 0.886 px | 2.690 → 2.756 px | No |

The audits use 1,407 points, 4,785 training observations and 643 held-out
observations. Results and rejected candidates are in
`out/fixed_bundle_converged_v1{,.candidate}.json` and
`out/calibrated_bundle_converged_v1{,.candidate}.json`; the early capped outputs
are retained separately. No gate was weakened to install either fit.

The focused suite now has **29 passing checks** on both the Mini and local
runtime. New checks cover nonrecursive path consensus, duplicate-source votes,
conflicting hypotheses, crop/anisotropic-resize covariance, joint-fit gauge and
holdout exclusion, a synthetic cropped panorama, and gap/depth-edge rejection.

### Next representation work

Local affine transport provides more than point locations: its Jacobian measures
how a scene neighborhood projects into another camera. The current geometry fit
throws most of that differential information away. The next candidate should
infer local surface position and tangents jointly from several transported image
neighborhoods, with uncertainty and reprojection checks, then reconcile nearby
surface hypotheses. Merely relaxing point gates or hiding folded triangles would
not solve the problem. Separate camera submaps and their relative scales also
remain to be reconciled; the current six-seed winner is not a complete global
hypothesis search.

## Differential surfaces and automatic camera growth

The next implemented representation retains the measured affine neighborhood,
not just its center. `surface_jets` fits a local 3-D center and two tangent vectors
from a 3×3 sample grid of radius three feature pixels. Training ray projectors
must determine all nine parameters with condition at most 1e7. A withheld
capture checks the entire predicted patch; positive rays and a maximum 1.5-pixel
error are required in both splits. These are conditional differential checks:
correspondence discovery and the starting cameras already used this collection.
They are not an independent reconstruction benchmark.

The first holdout rule always selected the last capture. Its joint-fit receipt
exposed cameras with no or very little training support (one had zero training
patches and 24 held-out patches). The retained v1 experiment is therefore a
failed diagnostic. The v2 rule hashes the physical reference site to select a
non-reference held-out capture, without consulting fit scores. All 17 cameras
then have training and withheld patches. This is a documented development-policy
change, not an untouched final holdout or a generalization result.

`jet_bundle` jointly fits camera poses and these surface patches while preserving
the seed frame and baseline length. Intrinsics remain fixed. All nine samples
of the withheld patch observation are excluded; sample weighting prevents those
nine locations from counting as nine independent regional measurements. Selection
uses only the training patch fit. A candidate is applied only when both withheld
median and p90 improve and the positive-ray fraction does not fall.

| Quantity | Before joint fit | After joint fit |
|---|---:|---:|
| Training median | 0.41746 px | 0.22356 px |
| Withheld median | 1.23729 px | 0.82324 px |
| Withheld p90 | 4.43341 px | 3.36250 px |
| Positive-ray fraction | 1 | 1 |
| Patches passing full local check | 382 | 504 |

The v2 optimization uses 810 training-supported patches, 20,403 training sample
positions and 7,290 withheld positions. It took 22.95 seconds on the Mini and
74 evaluations, terminating by cost tolerance (status 2; optimality 0.4033).
This is not a global optimum certificate. The v1 joint candidate worsened
withheld median from 1.496 to 3.216 pixels and was rejected.

`scene_growth` refreshes the structure using all tracks with at least three
recovered views, minimum 0.7° parallax, positive rays and at most 1.2° error in
every available recovered view. It caps duplicate physical image sites, retains
accepted refined patch centers where available, and fits new cameras under the
existing initialization and withheld-site gates. Established poses remain fixed.
The control run from the old cameras reaches **18 poses**. The same growth rule
from the differentially refined cameras reaches **25 poses**, including eight
ordinary photographs, with **2,024 distinct multi-view-consistent points**.
Thus, refreshing the point pool alone does not explain the additional coverage.

Retained local receipts are `out/surface-jets-v1.json`, `surface-jets-v2.json`,
`jet_bundle_v1{,.candidate}.json`, `jet_bundle_v2{,.candidate}.json`,
`baseline-growth-v1.json`, and `jet-growth-v1.json` (all under `out/`).
The eight recovered ordinary photographs are IMG_1490, IMG_1521, IMG_1522,
IMG_1523, IMG_1541, IMG_1542, IMG_1558, and IMG_1559.

The `/capture/differential/` viewer displays **492 textured tangent patches**
(984 triangles), after 12 of the 504 local passes fail the center check against
all currently recovered cameras. It uses 14 texture sources. Patch absence from
the deduplicated point pool is not itself a rejection: centers are independently
rechecked against their observations. There are no fabricated connections between
these patches. Browser inspection still shows sparse, distorted structure. The
older crumpled Delaunay result remains at `/capture/scene/` as failure evidence.
Neither view is a completed fused scene or resolved visibility model.

## Complete panorama-seed survey and rejected map joins

`scene_submaps` evaluates all 70 translation-supported panorama seeds instead of
only choosing among the original six. Four worker processes on the Mini completed
in 202.73 seconds. The 68 successful local reconstructions collectively contain
**53 distinct captures**, each in its own conditional coordinate frame. That is
not 53 captures in one registered scene. The shared result remains 25 captures.
The complete policy, index, and per-seed receipts are in `out/scene-submaps-v1/`.

`submap_alignment` estimates a positive-scale similarity using training shared
points and refines it against their bearings. It reserves physical-site holdouts
and checks the orientations of shared cameras. `merge_submaps` additionally
checks newly transferred cameras against existing structure, never overwrites
established poses, and requires retention of the current point support.

The real merge attempt accepted **zero joins**. Of 56 candidates, 27 failed the
withheld/orientation checks, 20 lacked shared points, five lacked a supported
similarity, and four lacked split support. For example, seed [17,24] has 565
shared points and 56/87 withheld points agreeing within 1.2°, with median maximum
held-out error 0.84745°. Yet the shared cameras disagree in orientation by a
median **8.1448°** after the fitted frame transform, exceeding the fixed 1.2°
gate. Point agreement alone would have admitted an incompatible camera model.
The full rejected attempt is `out/merged-submaps-v1.json`.

The next unresolved issue is whether the camera projection model, ambiguous
correspondences, or weakly constrained scene geometry explains this disagreement.
Smooth panorama ray distortion and motion within a stitched sweep are candidate
models, not established diagnoses. A shared-camera joint fit must predict withheld
neighborhoods and reconcile maps before they can be fused. Relaxing the join gate
or labeling the union of local hypotheses a global scene would not solve this.
All 120 captures, surface agreement and visibility-aware fusion remain the goal.

The focused suite now has **38 passing tests on the Mini**. New tests cover
surface tangents and rank failure, spatial holdout coverage, poisoned withheld
patches without optimizer leakage, growth and withheld-camera rejection, proper
similarity camera transfer, and accepted/rejected synthetic map joins. Synthetic
mechanism checks do not certify the physical outdoor reconstruction.

## Panorama calibration with differential constraints

`jet_bundle --calibrate-panoramas` now jointly varies horizontal focal scale,
vertical focal scale and vertical principal point for every cylindrical camera
with training patches. It retains the same 810-patch input, whole-observation
holdout assignment and fixed seed frame/baseline as the fixed-intrinsic fit.
Parameters have fixed weak priors (0.2, 0.2 log scale; 0.15 image height for crop)
and bounds (0.6–1.8 focal multipliers and ±0.4 image height crop). Ordinary-photo
intrinsics remain fixed. Existing input camera-model overrides are composed,
not discarded. This is a central cylindrical calibration candidate, not a model
of within-sweep camera translation.

The Mini run `out/calibrated_jet_bundle_v1{,.candidate}.json` reaches its declared
1,000-evaluation cap in 213.01 seconds (status 0; optimality 2.2298). Convergence
is unproven. It passes the original aggregate prediction gate:

| Quantity | Fixed-intrinsic differential result | Calibrated differential candidate |
|---|---:|---:|
| Training sample median | 0.22356 px | 0.18992 px |
| Withheld sample median | 0.82324 px | 0.64352 px |
| Withheld sample p90 | 3.36250 px | 2.71860 px |
| Positive-ray fraction | 1 | 1 |
| Full local patch passes | 504 | 563 |

This aggregate pass is insufficient for promotion as the shared scene.
`jet_residual_audit` measures center error separately from the variation remaining
after removing that center offset, with one vote per patch observation and no
accepted-patch filtering. The fixed result has held-out center median 0.77469 px
and differential RMS median 0.22583 px. Calibration reduces these aggregates to
0.60856 and 0.20119 px. However, only nine of 17 cameras improve their held-out
center medians. Cameras 13 and 44 regress from 1.210 to 7.495 px and 2.313 to
8.413 px respectively (seven held-out patches each). Other regressions are
retained in the per-camera receipts. No object-specific repair follows from this
audit; the next joint model must account for camera-level prediction and support.

The same growth procedure reaches **23 cameras and 2,085 consistent points**,
compared with 25 and 2,024 from the fixed candidate. It adds six cameras to the
17-camera calibrated seed result and does not recover cameras 75 or 95.
The renderer retains 548 of the 563 passing patches after all-current-camera
center checks. This alternative is exposed at `/capture/calibrated/`, alongside
`/capture/differential/`; the 25-camera fixed result remains available unchanged.
The browser labels the capped fit explicitly. Inspection still shows sparse,
distorted structure, not a completed fused scene.

Joining the calibrated result against the original panorama submaps again
accepts zero maps. Seed [17,24] now has a 9.249° median shared-camera orientation
disagreement and only 52/88 held-out points agree. Calibration has not resolved
the map conflict. Receipts: `out/calibrated_jet_growth_v1.json`,
`out/calibrated-map-joins-v1.json`, `out/jet-residual-fixed-v1.json`, and
`out/jet-residual-calibrated-v1.json`.

Synthetic checks verify recovery of deliberately cropped/scaled cylindrical
neighborhoods, exact exclusion of poisoned withheld observations from fitted
intrinsics, center-versus-differential residual separation, and preservation of
calibrated camera models through map transfer and structure refresh. These
mechanism checks do not overcome the real per-camera regressions above.

## Stage-two search across ordinary-photo seeds

`scene_submaps --seed-scope all` completes all **198** translation-supported
pairs: 70 panorama/panorama, 86 panorama/photo and 42 photo/photo. The default
panorama-first path is unchanged; stage two explicitly expands the seed scope.
The same cycle support, separate-origin triangulation and held-out third-view
pose rules apply. The Mini survey took **464.04 seconds** with four workers.
All 198 per-seed outputs and their policy are retained in
`out/all-scene-submaps-v1/`.

**191 local maps pass**, collectively covering **95 captures: all 28 panoramas
and 67 ordinary photos**. This adds 42 capture IDs to the panorama-only union.
The largest individual map has 35 cameras. Twenty-five captures remain absent
from all accepted maps: 30, 31, 32, 34, 38, 43, 45, 49, 51, 59, 60, 73, 74,
91, 92, 93, 94, 101, 102, 107, 108, 109, 110, 118, and 119. IDs index the retained
records, not inferred physical station numbers.

The expanded map pool still cannot be directly joined to the retained 25-camera
hypothesis. `out/all_submap_joins_v1.json` records 159 attempts: 80 lack enough
shared points, 50 fail withheld prediction/orientation, 19 lack a supported
similarity and ten lack a usable train/holdout split. Zero joins pass. The
25-camera shared candidate and its 2,024 points remain unchanged.

This separates two remaining tasks: (1) reconcile overlapping local maps and
their shared camera variables without requiring every distant map to overlap
one starting map directly; (2) recover remaining captures and integrate their
visibility-supported information. The larger local union must not be described
as a 95-camera globally registered scene. Future joint reconciliation must expose
per-camera regressions as well as aggregate fit quality. No manual object regions,
station assignment or source-specific coordinate repairs were added.

The focused suite has **40 passing tests locally and on the Mini**. The browser retains fixed and calibrated alternatives with explicit
limitations. Original panoramas were also inspected for XMP projection/crop
metadata: Pillow exposes no XMP block for any of the 28 JPEGs. This does not
establish absence of all vendor metadata or validate the nominal projection.

## Bidirectional map graph and accepted neighbor joins

`submap_graph` compares all 191 accepted local hypotheses, retaining smaller
maps that could connect larger groups. Of 6,515 pairs sharing at least 20
consistent track IDs, **46 pass** independently fitted forward/reverse frame
relations and round-trip checks. The round trip requires at most 1.2° rotation,
0.05 absolute log scale, and 0.05 relative point p90 error, normalized by the
centered source cloud extent. It does not use arbitrary chart-origin distance
as its normalization. The Mini run took **358.56 seconds** with four workers.
All pair receipts, including rejections, are in `out/submap-graph-v1/`.

There are 157 compatibility components. These are overlapping alternative
hypotheses, not 157 physically disjoint pieces of the scene. Graph connectivity
alone does not compose their coordinates or establish global consistency.
`reconcile_graph` chooses the maximum-camera-coverage seed in each component,
breaking ties by consistent-point support and seed ID, refreshes/deduplicates
its structure, and checks each candidate against accumulated geometry using the
existing merge rules. It can discover points needed for a later neighbor that
has no direct overlap with the first map. A synthetic chain tests that case.

The real run accepts **three joins** in 35.43 seconds on the Mini:

| Component | Initial → final cameras | Accepted source seed |
|---|---:|---|
| 2 | 13 → 18 | [0,3] |
| 75 | 31 → 32 | [14,17] |
| 78 | 35 → 37 | [17,25] |

`out/reconciled-components-v1/078.json` is the largest result, seeded by [17,24].
It has **37 cameras (20 panoramas, 17 ordinary photos)** and **4,152 consistent
points**. This is a different hypothesis from the retained 25-camera result,
not twelve cameras appended to it. The complete component outputs and summary
are retained beside it; their coordinate frames remain independent.

The stronger `surface_jets` check tests all 4,152 points and accepts 584 local
patches; the median withheld patch error over all tested points is 2.42338 feature
pixels. Two more fail the all-current-camera center check. The private
`/capture/neighbors/` viewer therefore textures **582 patches** (1,164 triangles)
from 28 sources, with the other points visible as context. Browser inspection
shows substantial folding and inconsistent patch orientation. Extra camera
coverage does not certify the surface or establish correct global fusion.
The jet receipt is `out/neighbor_surface_jets_v1.json`.

## Joint map candidates and protected prediction support

`joint_submap` uses a failed frame relation only as an initializer, retains one
variable per shared camera, and jointly fits cameras and point positions. It
requires the existing aggregate gate plus per-camera checks against the original
point support. Existing-camera withheld median must stay below 1.1 times baseline
plus 0.1 bearing-equivalent feature pixel; p90 below 1.2 times baseline plus 0.2
pixel. Every new camera needs four held-out sites and at least 60% within 1.2°.
No old validation sites are dropped to make a candidate pass.

The v1 ranking counted only new-camera observations of points already retained
in the base map. It selected [6,7] with only two such training links. The 28-camera
candidate predicts its three new cameras, but fails the protected old-camera
checks. Its 300-evaluation run took 27.34 seconds and reached the cap. Aggregate
withheld median improved from 0.33572° to 0.25756° and p90 from 2.40212° to
1.06334°, illustrating again why an aggregate score is insufficient.

The v2 ranking also counts source tracks with two training observations in
existing cameras and training observations in new cameras. This corrects the
omission of new scene points that connect old and new views. It selects [17,24]
with **1,223 training links and 21 proposed new cameras**, giving a 46-camera,
6,793-point initialization. The 300-evaluation fit takes 14.91 seconds and is
not converged (reported optimality 4505.66). Its pre-refresh withheld median
changes from 0.30695° to 0.29422°, and p90 from 8.58443° to 1.71355°, but seven
fitted observations lose positive depth. It is rejected by the unchanged
positive-depth gate as well as the per-camera checks.

A further audit addresses stale point coordinates. Joint fitting caps duplicate
image-site votes and can omit whole points, which otherwise retain coordinates
from before camera motion. `refresh_training_points` refits point coordinates
from the final cameras using only non-held-out rays. It records points with too
few training rays or condition over 1e5 rather than silently treating them as
refreshed. Unsupported points retain their prior coordinates and remain in the
prediction gate. On v2 it refreshes 6,051 points and leaves 742 unsupported.
After that audit, 20/21 new-camera checks pass but 0/25 old-camera checks pass.
The v1 refresh-only control still fails (3/25 old-camera checks pass), so stale
coordinates alone do not explain the failed reconciliation.

Retained receipts: `out/joint_submap_v1{,.candidate}.json`,
`out/joint_submap_v2{,.candidate}.json`, and
`out/joint-submap-v1-refresh-audit.json`. The bundle's numerical comparison now
allows 1e-8 degree/pixel roundoff: a synthetic exact old-camera fit had a median
increase of only 2e-10 degrees while correcting a new camera by about a degree.
This roundoff fix does not affect the substantive real-data rejections above.

The fixed longer v2 run allows 2,000 evaluations and stops at 1,048 by cost
tolerance (status 2), taking 51.34 seconds. Reported optimality remains 44.97,
so it is not a tightly stationary or global optimum certificate. Its aggregate
withheld median is 0.27846° and p90 1.53464°, with the same seven non-positive
fitted observations. Training-only refresh updates 6,075 points and leaves 718
unsupported. **All 21 new-camera checks pass, but all 25 old-camera checks fail.**
The candidate remains rejected; `out/joint_submap_v2_long{,.candidate}.json`
preserves it. Running beyond the first cap did not remove the conflict.

The next representation work must retain neighborhood shape and its spatial
coherence in the cross-map fit, with positive-depth and visibility accounting.
The point-only union can improve aggregate image error while damaging previously
supported cameras; it is not an acceptable replacement for the surface method.
Neither the 37-camera component nor the rejected 46-camera candidate completes
the requested 120-capture quasi-3-D scene.

The focused Mini suite passes **47 tests** after these changes. The additions
cover bidirectional frame round trips, rejected-edge isolation, a real synthetic
neighbor chain, shared-camera identity in joint initialization, rejection of
old-view regressions despite new coverage, joint recovery of a perturbed camera,
and exclusion of withheld rays from point refresh. Numerical compilation and
repository whitespace checks also pass. These checks validate mechanisms, not
the visibly folded outdoor reconstruction.

## Training-derived surface continuity

`surface_coherence` links patches only through at least two common training
cameras. A candidate separation is 2–12 feature pixels, both local affine maps
must predict its transport within 1.5 pixels, and their transported Jacobians
must differ by at most 0.15 relative Frobenius norm. Deterministic ranking caps
node degree at six. No object labels or manually chosen image regions enter.
The residual combines trapezoidal position integrability and tangent agreement
in a shared image coordinate system. Initial training-camera range divided by
focal length supplies a frozen world-to-feature normalization. Nine robust
continuity residuals receive the same 1/3 sample normalization as a patch.

The 37-camera component supplies 2,075 training-supported patches and 4,305
links connecting 1,934 patches. There are 216 components, including 141 isolated
patches; the largest contains 1,381 patches. Disconnected evidence remains
separate. A neighborhood connection is conditional measured compatibility,
not independent proof of a common physical surface.

`coherence_study` runs a matched control and continuity alternative with fixed
intrinsics, identical initialization, and the existing seed frame/baseline
constraint. Both now require every camera's withheld median to remain below
1.1 times its baseline plus 0.1 pixel, and p90 below 1.2 times baseline plus
0.2 pixel. All 37 cameras have some patch holdouts; counts vary and are retained.
These are within-collection development checks, not unseen-scene validation.

The first study retains only the differential subset in fitting, as the prior
jet solver did. The control stops at 356 evaluations in 268.62 seconds; the
continuity alternative stops at 123 evaluations in 99.09 seconds. Both stop by
cost tolerance; optimalities 4.38 and 7.11 do not certify global convergence.

| Metric at feature resolution | Initial | Control | Continuity |
|---|---:|---:|---:|
| Withheld sample median (pixels) | 2.18023 | 0.76681 | 0.89920 |
| Withheld sample p90 (pixels) | 5.84897 | 4.47642 | 4.36930 |
| Positive projected fraction | 1 | 0.999889 | 0.999986 |
| Locally accepted patches | 584 | 1,276 | 1,098 |
| Failing per-camera checks | — | 7 | 7 |

The continuity position residual falls from 7.99398 to 0.17798 feature-equivalent
units; tangent residual from 1.59854 to 0.35414. This confirms that the term
couples surfaces, but neither candidate is accepted. Both regress cameras
4, 12, 16, 62, 63 and 96; control additionally fails 47, continuity fails 77.
Camera 96's withheld median grows from 4.99 to 49.05 pixels under continuity,
despite a training-center median of 0.43 pixel. One continuity sample and eight
control samples lose positive depth. Smoother geometry is not a sufficient gate.
All candidates and retained inputs are preserved in `out/coherence-study-v1/`.

The next matched study restores point constraints outside the surface subset.
`point_support` retains original point hypotheses with at least two training
rays and condition at most 1e5. It excludes physical sites divisible by seven
and all surface-heldout camera/site pairs from fitting. It does not synthesize
an affine map for an ordinary point. Each point contributes its center pixel
residual, with the same effective one-pixel robust transition and observation
weight as nine repeated patch samples. Its position is jointly optimized with
camera poses and surfaces. Separate per-camera point holdouts and positive-depth
checks join the patch gate. The real input adds 1,999 points, 7,311 training
observations and 1,036 withheld observations. Seventy-eight unsupported points
retain prior coordinates and are explicitly listed, not certified as refreshed.

### Retaining the remaining point support

The v2 point-supported comparison retains both branches in
`out/coherence-study-v2/`. The control stops at 222 evaluations in 149.71 seconds;
continuity stops at 328 in 201.10 seconds. Both stop by cost tolerance, with
optimality 28.90 and 10.69 respectively. The continuity branch keeps every
projected surface and support-point sample positive.

| Metric | Point-supported control | Point-supported continuity |
|---|---:|---:|
| Withheld patch sample median (pixels) | 0.66002 | 0.68812 |
| Withheld patch sample p90 (pixels) | 2.66371 | 2.32353 |
| Surface camera checks passed | 32/37 | 35/37 |
| Locally accepted surface patches | 1,405 | 1,318 |
| Additional-point camera stability checks failed | 10 | 14 |

Continuity's two remaining surface failures are cameras 4 and 16: their medians
improve, but their p90 tails worsen. The additional point checks preserve the
stored point positions as a stability reference. Both candidates remain rejected.
The point-supported continuity preview at `/capture/coherence-supported/` is
explicitly labeled rejected trial geometry: 3,955 audited points and 1,317
rendered patches after one patch fails the all-view center audit. It remains
sparse and visibly distorted; 37 cameras are not the full 120-capture result.
The older subset-only trial is at `/capture/coherence/`.

`point_support_audit` distinguishes point stability from prediction. The stored
point positions were fitted before this run reserved its point holdouts. A
separate control therefore holds the original cameras fixed, initializes each
point from training rays alone, and refines it using only those rays. It scores
exactly the same 1,036 withheld observations; nothing is removed to improve the
comparison. Median/p90 feature errors are 1.38627/4.03993 for stored coordinates,
2.35843/6.64335 for the training-only control, and 0.57804/2.17538 for the joint
continuity candidate. Four camera comparisons against the training-only control
still fail (4, 19, 26, 47). Five of the 1,999 independent control fits reach their
100-evaluation cap, so the control is not fully converged. This diagnostic does
not override the candidate's rejection or establish unseen-scene accuracy.
Receipt: `out/coherence-point-control-v2.json`.

The next trial combines the existing bounded panorama scale/crop calibration
with continuity and retained point support, from the same initial component.
The earlier calibration alternative lacked these constraints; its failed result
is preserved separately. All 120 original JPEGs were also checked for embedded
GPS position and camera-direction fields: neither is present, so no position
or heading metadata is available to independently place the captures.

### Calibration and missing depth information

Combining bounded panorama scale/crop calibration with point-supported continuity
stops at 605 evaluations in 488.75 seconds (cost tolerance; optimality 169.28).
Its patch median/p90 are 0.76840/3.61168 pixels, worse than the fixed-intrinsics
point-supported candidate. Surface camera checks fail for 4 and 116; ten point
stability checks fail. All projected samples remain positive, and 1,224 local
patches pass. It is rejected and does not replace either retained component.
Both outputs are preserved in `out/coherence_calibrated_v1{,.candidate}.json`.

The worst seven-patch neighborhood under the fixed-intrinsics candidate has only
cameras 14 and 15 in its entire training-view union; its withheld camera is 16.
The recovered training-camera separation is 0.01839 initially and 0.01607 after
refinement, in units where the seed baseline is one. Thus graph connectivity
within this neighborhood supplies no additional displaced view. In the broader
audit, 526/1,934 linked patches and 94/141 isolated patches exceed the 1.5-pixel
withheld maximum. Failures are not confined to isolated points.

`surface_observability` explicitly projects the reference patch's nine directions
at infinite range using camera rotation alone. Passing the original 1.5-pixel
training tolerance demonstrates that this patch's local training observations
also admit infinite depth; failing this test does not certify finite depth.
Among the same 2,075 patches, 40 pass this infinity test in the initial cameras
and 104 pass in the fixed-intrinsics refined cameras. Five of the worst seven
patches pass after refinement. Their training evidence cannot by itself justify
a unique finite range. The finite-world continuity regularizer is not additional
measured parallax. Receipts are `out/coherence-study-v2/initial-observability.json`
and `candidate-observability.json`.

This identifies the next systemic requirement: propagate depth from neighborhoods
with an informative displaced-view baseline, and retain uncertainty in groups
without that support. The current practice of withholding one view from every
patch can remove the sole useful baseline from a whole small neighborhood.
A future spatially blocked prediction design must retain actual depth anchors in
training while withholding other regions, rotate the blocks so every observation
is tested, and prevent those heldout camera/image sites from re-entering through
other point or patch constraints. It must be compared with a control using the
same blocks; changing the split must not be presented as improvement on the old
score. Neither candidate acceptance nor the all-120-capture goal is changed by
these diagnostics.

An all-observation availability audit confirms that the withheld partition
changes this neighborhood's connectivity. With all measured view maps available,
the same 2,075 patches produce 4,905 links and 127 components (largest 1,468).
The worst seven-patch training component joins a 19-patch neighborhood spanning
captures 13, 14, 15 and 16. This audit uses former holdout observations and is
explicitly not a predictive score or permission to insert them into the old
training objective. It demonstrates available additional support to use under
a correctly separated future evaluation. Receipt:
`out/coherence-study-v2/all-observation-connectivity.json`.

The expanded focused suite passes **56 tests on the selected Mini** (6.365 seconds).
The new checks cover training-only neighborhood selection, surface coordinate
invariance, coupled optimization without reading patch holdouts, retained point
constraints and physical-site exclusions, training-only point controls, and the
difference between colocated and displaced training views in the infinity test.
Python compilation and repository whitespace checks also pass. These remain
mechanism checks; no new candidate has passed every real-scene acceptance gate.

## Spatially blocked surface prediction

`spatial_surface_study` uses all 4,152 patch records with measured observation
maps, without filtering by the previous one-camera holdout score. Each capture's
image is partitioned into deterministic 32-pixel tiles, assigned by a fixed hash
to four folds. An observation center is scored in exactly one fold. Training
excludes an entire affine patch footprint plus a two-pixel guard whenever its
bounding box touches a reserved tile. Guarded observations are neither fitted
nor scored in that fold; their centers are scored in their own fold. Initialization
uses training rays only, including a uniform first ray-fit weighting independent
of stored point coordinates. Rank-deficient or negative-depth initializations
receive a starting range from the median of other training-supported fits. This
is explicitly an initialization, not measured finite-depth support.

Patches with fewer than two training observations remain unresolved. Every
original observation remains in the final coverage accounting, and an assertion
checks that all 16,480 observation centers appear exactly once across the four
folds. Control and continuity use identical resolved support. The experiment
continues to condition on existing camera poses and previously discovered image
maps; it does not establish end-to-end unseen-image generalization.

The fixed-camera study is preserved in `out/spatial-surface-v1/`, including all
eight candidate/retained outputs, receipts, predictions and the combined summary.
The four executed source snapshots are archived under its `sources/` directory
and match the SHA-256 values in the receipts.

| Fixed-camera blocked prediction | Control | Continuity |
|---|---:|---:|
| All observation centers | 16,480 | 16,480 |
| Resolved predictions | 10,692 | 10,692 |
| Explicitly unresolved | 5,788 | 5,788 |
| Median error on resolved samples (pixels) | 1.66333 | 1.52245 |
| P90 on resolved samples (pixels) | 6.08752 | 5.19326 |
| Observations passing all nine samples and positivity | 4,000 | 4,070 |
| Observations with a nonpositive sample | 1 | 1 |

Control fold evaluation counts are 57, 70, 55 and 42. Continuity uses 135, 97,
169 and 300; the last hits the cap. The first three continuity folds improve
p90, while fold three changes p90 from 5.88767 to 5.89695 pixels and decreases
its passing-observation count. This is a modest aggregate improvement with an
unconverged fold and substantial unresolved coverage, not a completed scene.
The scores cannot be compared directly with the earlier 2,075-patch subset.

A synthetic slanted-plane case separately tests the intended information flow.
Four central patches have only two exactly colocated training cameras; their
displaced views are withheld. Neighboring patches retain the displaced views.
The control's withheld median/p90 are 0.85644/1.29531 pixels; continuity gives
0.00895/0.01915. All eight withheld observations pass with continuity. The test
also checks that poisoning reserved coordinates and affines after the split
leaves initialization and optimized geometry unchanged. Real unresolved coverage
remains explicit despite this successful mechanism check.

The next matched run, `out/spatial-joint-v1/`, permits camera poses to move under
exactly the same four folds and support. Explicit spatial roles are rejected if
passed to the old point-support path, which does not yet have matching tile
exclusions. This prevents accidental reintroduction of reserved sites.

The joint-camera study has now completed and is preserved in
`out/spatial-joint-v1/`, with executed source snapshots matching every receipt.
It retains the same 16,480 centers, 10,692 resolved predictions and 5,788 unresolved
centers. Control median/p90 are 0.96710/4.11857 pixels; continuity gives
0.88715/3.05376. Passing observations increase from 6,073 to 6,406 and observations
with nonpositive samples decrease from three to zero. Three continuity folds
reach the 300-evaluation cap. These are four conditional geometries, not one
accepted global scene, and the correspondences remain inherited.

The subsequent all-observation fit is preserved as
`out/full-surface-field-v1.json`. It retains 37 cameras and 4,152 point records,
with 2,837 training-supported patches and only 378 passing every attached blocked
prediction. It stops by cost tolerance after 257 evaluations, taking 369.41 s.
It has no new independent prediction split and remains unpromoted. On review,
the user identified the more fundamental limitation: optimizing geometry inside
mostly fixed regional identities cannot reconcile duplicated scene hypotheses.
The all-observation field is therefore a diagnostic, not the next accepted scene.

## Mutable regional identity search

`identity_register` and `identity_study` reopen identity search directly from
cached native-segmentation sites and Oklab/Meyer image planes. Neither old tracks,
old image-pair inventories, camera poses nor fused pixels enter retrieval. The
new path retains rotation-invariant angular-magnitude signatures of finite
annular traces at two radii. A 20-dimensional PCA accelerates nearest-neighbor
retrieval; full signature distances re-rank those proposals. Signed hue chord
coordinates avoid a hue-angle discontinuity. Magnitude is used for retrieval
only: the original angular phase is retained to generate several competing
rotation/scale hypotheses, with nonperiodic radial overlap.

Each proposed similarity is a starting point for a six-parameter affine image
fit. A deliberately sidelobed, finite 25-tap bandpass also proposes multiple
subpixel translations after local rectification. The zero-delay alternative is
retained for a matched ringing ablation. Direct Lab fit, reverse fit, Jacobian
closure and aperture information test every alternative. The fit's reserved
pixels are a local optimizer diagnostic, not an independent holdout: retrieval
and phase proposals have already inspected the image neighborhood. Ringing
reshapes existing signal evidence; it does not create identity information or
guarantee an exact delay.

Alternatives survive as edges with recomputed relative scores and an unresolved
competitor. Third-view affine cycles can revise their ordering; each third
capture has at most one vote per edge. No greedy union freezes identity, and
removing support can reverse the choice. These are heuristic weights, not
calibrated probabilities. Repeated scenery can itself form consistent cycles;
cycle closure alone does not certify identity or camera position. A photo's
regional overlap with a panorama must not be interpreted as co-location.

The first bounded real run indexes every capture and verifies at most 12
candidate identities per source capture. This is a mechanism probe, not an
exhaustive matching run or a reconstructed scene. All retrieved but untested
candidates stay in the output ledger. The initial run failed before verification
on NumPy integer JSON serialization; the source was corrected and a regression
check added before the fresh v2 run. No result from that failed run was promoted.

The v2 probe completed in 41.13 seconds: 9,815 source sites / 19,630 annular
traces across all 120 captures produced 58,890 directed candidate identities.
Of 1,440 verified candidates, 240 passed local checks (391 distinct fitted
transforms), involving 103 captures. Phase-only starts passed 231 candidates;
ringing starts passed 212, adding nine candidates missed by phase-only starts.
Thirty-seven transforms had one third-capture supporter; none had two.

The expanded v3 run verifies all six retrieved identities at each selected
source site, with a budget of 192 candidates per capture. It also fixes the
relative-score calculation so multiple fitted distortions of the same target
site cannot inflate that identity's weight. The complete run, source snapshots,
feature hashes, verification ledger and association audit are preserved under
`out/identity-register-v3/`.

| Expanded image-only identity probe | Result |
|---|---:|
| Captures indexed | 120 |
| Retrieved candidate identities | 58,890 |
| Verified candidate identities | 23,040 |
| Candidates passing local checks | 927 |
| Distinct retained transforms | 1,591 |
| Captures involved in a passing pair | 119 |
| Transforms with at least one third capture | 480 |
| Transforms with at least two third captures | 186 |
| Phase-only candidates passing | 842 |
| Ringing candidates passing | 717 |
| Extra candidates recovered by ringing | 85 |
| Total runtime on the Mini | 268.78 s |

Forty-seven source-region/target-capture groups retain multiple passing target
sites. Third-view scoring changes the leading regional choice in four groups.
Those changes are not verified corrections of physical identity: nearby region
sites can themselves represent overlapping content, and repeated scene objects
can form consistent loops. The sole capture without a passing pair in this
bounded probe is index 17 (`IMG_1462.jpg`). Unverified proposals remain explicit.
No station assignment or global scene geometry is promoted.

`identity_view` builds `/capture/identity/`, showing the original capture context,
rectified regional crops, ordinary averages, competing identities and directed
pair-evidence matrix. Visual checks sampled automatically ranked matches; no
object-specific repair or ROI was added. The expanded focused suite passes
70 tests on the selected Mini in 7.428 seconds, including eight identity tests.
The next integration must let competing regional identities propose and revise
camera/place hypotheses; pair overlap must remain distinct from co-location.

## Differential camera hypotheses and reversible geometry feedback

The next iteration confirmed that the 927 passing regional identities in the
bounded probe were too sparse for ordinary point-only camera fitting: 405
undirected image pairs had local matches, but only one pair had eight distinct
source/target regions. `identity_fields` therefore extends the independently
verified affine maps onto a 12-pixel lattice over three rounds. It retains up to
four spatially distinct target/distortion hypotheses per source site. Competing
maps are never averaged, and every new proposal must pass a direct image fit,
a reverse fit, closure and aperture checks. Algebraically inverted seed maps are
proposals only. Their grown reverse maps must be fitted to the original images.

The bounded-parent field result (`out/identity-fields-v1/`) contains 20,852
directed source regions / 32,187 hypotheses across 810 directed pairs, involving
119 captures, in 95.75 seconds. Descendants and neighboring patches are correlated;
these counts are not independent physical landmarks.

`affine_camera_geometry` uses the center epipolar constraint and its two source
image derivatives. The implementation differentiates the calibrated unit-ray
map for perspective and cylindrical captures, including the measured local
image-map Jacobian. Three nondegenerate affine regions can initialize a relative
pose in the synthetic check. Planar configurations instead propose several poses
through a ray homography. Its SVD decomposition retains both tilt and sign
branches; it never fixes the reconstructed scene to a plane. This use of local
affine information is consistent with the central-camera formulation in
[Eichhardt and Chetverikov, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ivan_Eichhardt_Affine_Correspondences_between_ECCV_2018_paper.pdf).
Our code uses a direct differential linear fit and nonlinear refinement, with
synthetic checks for the equations actually implemented.

`identity_geometry` retains multiple rotation and essential-matrix hypotheses.
Each alternates training-region selection and camera fitting, then predicts
reserved 48-pixel source tiles. A 14-pixel guard excludes training patches whose
support reaches those tiles. Alternative samples cannot multiply source/target
votes. Geometry then recomputes region-choice scores while retaining every
original alternative. The reserved geometry checks remain conditional on image
matching that already inspected the photographs; they are not unseen-image
validation. Nominal intrinsics and the central panorama model remain assumptions.

On the bounded-parent fields (`out/identity-geometry-v1/`), 106 directed pairs
retain at least one model after the guarded checks, with 373 proposed local
choice changes, in 10.84 seconds. The retained models include 376 essential and
21 rotation hypotheses. Translation-compatible alternatives span 67 undirected
pairs / 82 captures. These are competing pairwise poses, not 82 cameras in one
registered coordinate frame.

`identity_pose_graph` tests alternative rotations around three-view loops, then
checks positive baseline-direction compatibility and fits relative baseline
scale from shared regional identities. Exactly collinear camera directions
remain possible, explicitly with unresolved direction-only scale. Binning scale
proposals uses medians of actual observations; using bin centers lost narrow
valid peaks in a synthetic test and was corrected before v2. Both v1 and v2
receipts and executed sources are retained. The bounded-parent v2 has 18
orientation-consistent loops, 13 with compatible baseline directions, and 23
scale hypotheses. None passes every regional/third-baseline check.

`identity_joint` subsequently allows both non-reference cameras and each local
depth value/gradient to move. It alternates those fits with region-identity
selection; all candidate image maps remain available. A reference camera and a
unit first baseline fix gauge only. Training positivity is audited. For reserved
source regions, camera-C measurements cannot initialize depth: A/B measurements
predict C after the cameras are fitted. `identity_joint_study` also checks the
previously unfitted B/C field against the resulting relative camera motion.

The first real joint study (`out/identity-joint-v1/`) contains 12 distinct
initializations automatically derived from the loop packet, all on captures
79/80/81. Every candidate passes the separate B/C check (23 of 27 regions across
three tiles). The best predicts four of seven reserved A/B-to-C regions; it
still fails the 60% full check, and all 12 remain rejected. The synthetic joint
fit recovers camera positions while retaining wrong higher-appearance alternatives,
and poisoning reserved C measurements leaves camera estimates unchanged.

The complete retrieved-identity run (`out/identity-register-complete-v1/`) then
verified all 58,890 candidates, without changing matching thresholds. It finds
1,838 passing candidate identities / 3,275 retained transforms involving all
120 captures. Phase-only starts pass 1,632; ringing adds 206 candidates. There
are 798 transforms with at least one third capture and 309 with at least two.
Runtime is 662.00 seconds on the Mini. All retrieved candidates were checked;
the receipt's generic final sentence about unverified candidates does not apply
to this run. This is complete verification of this bounded retrieval inventory,
not an exhaustive search of all possible image regions or a solved scene.

## Complete-inventory camera iteration and hypothesis preservation

The full field pass (`out/identity-fields-complete-v1/`) contains 32,757 directed
regions / 52,481 hypotheses across 1,414 directed pairs, involving all 120
captures. It took 183.63 seconds. The initial complete camera iteration
(`out/identity-scene-complete-v1/`) retained models on 158 directed pairs, with
654 revised regional rankings. Translation-compatible hypotheses span 90
undirected pairs / 99 captures, but the 16 joint candidates all failed: each
79/80/81 candidate predicted zero of seven reserved regions. Local model and
match coverage increased while global predictions regressed.

The regression audit compared earlier measurements and camera branches rather
than selecting a visual repair. The full fields retain near-equivalent versions
of 358/375 earlier 79→80 maps, all 103 earlier 79→81 maps, and 149/160 earlier
80→81 maps. The earlier joint pose still passes 23/33 expanded B/C check regions
across three tiles. Every earlier essential matrix for those three directed
pairs also passes the expanded pairwise checks, and its selected rotation and
translation sign remain unchanged. Thus the failed new candidates cannot be
explained by loss of all useful regional evidence or a sign flip alone.

`identity_geometry --prior-geometry` now replays ALL previous training-selected
matrices, including previously rejected ones. The capture identities, camera
calibration, tile split and guard must match. Past check outcomes do not select
which guesses are admitted. Imported matrices are refitted and checked against
the new fields. Both the starting proposal and its pair-refined descendant can
reach the joint optimizer, because a lower pairwise training error need not
improve a multi-camera solution. Neither is held fixed in the joint fit.

The replay receipts are preserved under `out/identity-scene-replay-v1/` and
`out/identity-scene-replay-v2/`, alongside executed sources. The second replay
retains models on 165 directed pairs and produces 811 revised local rankings.
Its relative-pose alternatives cover 100 undirected pairs / 102 captures. It has
339 orientation-consistent loop combinations, 272 direction-compatible ones,
and 429 scale hypotheses. These combinations are alternative hypotheses, not
independent votes or distinct reconstructed scenes.

All 88 final joint candidates remain rejected. Two have no reserved-region
coverage; the remaining 86 predict zero of seven reserved A/B-to-C regions.
None passes its separate third-pair check. The branch-preservation changes are
therefore mechanism improvements, not a demonstrated recovery of real-scene
fusion. The earlier bounded-parent result (best four of seven) remains a
separate rejected diagnostic, not a validation reference to fit toward.

The expanded focused suite passes **85 tests on the selected Mini in 7.650 s**.
It includes direct synthetic camera recovery from regional derivatives,
cylindrical projection, planar ambiguity, reversible identity choices,
collinear scale underdetermination, joint camera/local-depth recovery, and
reserved-data poisoning checks. These do not certify the real collection.

Next integration: the earlier image-local dense measurements in
`out/attached-fields-v1/` remain available independently of their old greedy
tracks and camera poses. Their NPZ records include source grids, target points,
affine maps, validity, image correlations, independently fitted reverse maps,
and third-view support. Re-verify and retain these as competing image
observations alongside the new ringing-derived maps; do not throw away their
broader panorama coverage or import their committed scene identities. Begin the
shared reconstruction with the panorama subgraph, then attach ordinary captures
with independent poses. The complete-inventory reviewer is at
`/capture/identity-complete/`; it displays regional registration, not a new fused
scene. No real scene promotion or completion is claimed.

## Reverified panorama identities and joint initialization

`identity_pool` now re-fits the older dense image-local proposals, the complete
ringing fields, and original irregular ringing seeds against the current Lab
images in both directions. It does not import old tracks, camera poses, cycle
votes or correlation bounds. It transports proposals onto a common two-pixel
source lattice before re-fitting, uses stable physical source keys, and retains
up to four distinct targets/Jacobians per site without averaging them. Every
proposal's fit outcome and provenance are saved, including rejected proposals.

`out/identity-pool-panoramas-v1/` contains the panorama-only run, with executed
sources. It verifies 74,667 proposals over 274 directed pairs in 52.60 seconds
on the selected Mini: 64,359 proposals pass; deduplication retains 64,210
hypotheses at 62,915 directed source regions, including 921 sites with competing
hypotheses. All 28 panoramas participate. Retained provenance counts are 58,528
dense-only, 5,281 ringing-field-only, 384 original-ringing-seed-only, seven
shared dense/seed and ten shared ringing-field/seed. These are correlated local
image measurements, not independent matches or reconstructed camera counts.

The closed-loop initialization control (`out/identity-scene-panoramas-v1/`)
retains models on 40 directed pairs and revises 37 local rankings. Translation
hypotheses cover 28 undirected pairs / 25 panoramas. There are six complete
translated pair triangles, but none meets the existing 1.2-degree rotation
closure gate; the smallest discrepancy is about 4.002 degrees. No joint fit
was attempted. No accepted rotation-only pair was present. This failure cannot
be explained merely by omission of rotation-only edges in this particular run.

`identity_wedges` adds shared-image initialization: any two relative-camera
hypotheses with a common image source and an available third image map can
propose a joint fit. Every source anchor is considered, without an imposed
capture order or station group. Initial pair rotations need not already close,
and the third pair need not have an accepted independent camera estimate.
Scale modes are selected from training data only. Final joint prediction,
positive-depth and third-pair gates remain unchanged. This removes a premature
initialization veto; it does not accept an inconsistent pose loop.

The resulting `out/identity-scene-panoramas-wedges-v1/` retains 40 proposal
wedges, 377 scale hypotheses and 322 distinct joint initializations. Total
runtime is 43.13 seconds, including 32.66 seconds for joint fits. Fifty-two
candidates change regional assignments; 281 retain positive training depths.
Six pass shared-source reserved prediction; 83 pass the separate third-pair
check, but **none passes both**. The best shared-source candidates predict
13/14 reserved regions yet only 3/25 third-pair regions. All 322 remain rejected;
no fused scene or camera positions are promoted. Executed sources are saved
alongside the outputs. The ordinary photographs have not been attached in
this panorama-first branch.

Next: put the additional third-pair image regions into the joint training
objective, rather than use the entire third pair only as an evaluation edge.
Retain separate spatial checks, and propagate their exclusion footprints
through the image maps in both directions: otherwise a region withheld in one
source could leak back into training through another camera's regional factor.
Keep competing correspondences and independent source-local depth hypotheses.
The one-source star fit's narrow overlap can predict well while the remaining
scene rejects its camera geometry; this is evidence for adding mutual image
constraints, not proof that calibration, panorama stitching distortion or
regional identity ambiguity has been solved. Do not tune to the displayed
13/14 candidate or promote it as a reference.

Validation for this iteration: 53 identity, pooling, affine, radial, multiview,
pose-graph and joint-fit tests pass on the selected Mini in 0.756 seconds.
The new checks cover input-lattice-independent physical keys, no averaging of
competing targets, original-image rejection of false proposals, all automatic
source anchors, and reversed relative-pose composition. These are mechanism
checks; all real panorama joint candidates remain rejected.
