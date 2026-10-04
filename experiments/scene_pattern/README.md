# Shared-scene pattern registration

Python prototype using real photographs of one physical scene. It reuses the
causal-density transport segmenter from the retired redwood experiment. The
redwood result failed visually because independently generated scenes diverged
in content; its image-fusion scores were not evidence of successful scene
reconstruction.

The prototype has no SIFT, learned descriptors, optical-flow fallback, or
reference-map initialization. Oxford's homographies enter only the evaluation
code after a pair has been estimated. Six real Graffiti views constitute a
planar projective benchmark, not a general 3D Photosynth replacement.

## Representation

1. Antialiased analysis restriction to width 256.
2. Native Oklab L,a,b Meyer decomposition using the current `fast` three-jump
   finite-flow method. Keep cartoon, signed texture and unresolved residual.
3. Texture coordinates: L texture; actual radial chroma difference; and two
   chroma-weighted circular hue chord components. Hue vanishes at achromatic
   endpoints and has no artificial discontinuity at ±pi.
4. Independently segment those four texture coordinates and the cartoon with
   `port_needed.pipeline.build_segmenting_representation`, the actual forest
   causal-density pipeline. Scalar coordinates are neutral-Oklab encoded into
   its unchanged RGB API. Their 99.5-percentile range normalization clips only
   the segmentation carrier; signed evidence is retained in the scene.
5. Regions supply correspondence sites and broad, two-scale cartoon-color
   descriptors with texture-energy evidence. Mutual matches produce a robust
   projective seed. These are hand-constructed descriptors, not invariant
   keypoints or semantic identities.
6. A finite rotation/scale bank with correlated translation supplies additional
   coarse hypotheses. Image-only robust fitting narrows the best hypotheses at
   80, 160 and 256 pixels. Region sites participate in the fitting objective.
7. Compare ordinary fractional-delay phase registration with a deliberately
   sharp 25-tap boxcar-windowed band-pass FIR on the same regional patches.
   Both arms use local upsampled correlation. Filtered phase alone does not
   create information or make the recovered delay mathematically exact.
8. The final shared geometric polish uses the existing native two-order CONV
   arbitrary-coordinate evaluator. It is distinct from the joint admitted
   Bernstein atlas. Diagnostic image sampling uses the same native CONV path.

No ground truth affects candidate selection, delay acceptance or refinement.
The two delay arms start from the same geometric fit. A rejected proposal is
recorded rather than silently presented as an improvement. Search-time pyramid
sampling is bilinear; final CONV refinement is separately identified.

## Reproduction

Download locally so the authoritative collection is mirrored to the Mini:

```sh
python3 -m experiments.scene_pattern.fetch
/Users/ultimussecundai/.local/bin/m4build -- make -j4 build/libbfft.so
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m unittest experiments.scene_pattern.test_core -v
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.run --out /tmp/scene_pattern
```

Copy `/tmp/scene_pattern/` back using `m4host` before further synchronization.
The fetcher retains source URLs and SHA-256 receipts. Raw dataset and generated
outputs are locally ignored. `--count 2` limits an exploratory run to one pair.

The HTML viewer is a review surface over Python estimates. Copy `viewer.html`
as `index.html` into the result directory and serve it with `python3 -m
http.server`. Its WebGL homography interpolation uses bilinear texture lookup;
it does not claim to run the matcher or CONV in JavaScript.

## Boundaries

Each adjacent pair is estimated independently. This initial prototype does not
yet solve collection-wide cycle consistency, infer occlusion/depth, select
nonadjacent overlap pairs, or recover a physical camera path. Large changes in
perspective and repeated regions can defeat the coarse descriptor. Benchmark
those failures explicitly. A good scalar appearance score is insufficient;
inspect reference-map errors and the checkerboards. Numerical absence of a
homography pole at sampled points is not a proof of global admissibility.

## Radial unwound trace

`radial.py` samples a finite annulus as `(log radius, angle)`. Rotation translates
angle and centered scale translates log radius. The angular seam is periodic;
radial overlap is finite and is never wrapped. Complex angular phase is retained
through the correlation calculation. Tests use independently evaluated analytic
images with known scale/rotation and show loss of coherence for a misplaced
center. On photographs, several existing region centers provide residual probes
after coarse alignment. Their estimates must agree and improve the same
image-only score before a proposed correction is accepted. A successful centered
similarity test does not make the trace invariant to arbitrary homographies.

## Native issue discovered by this prototype

At realistic trace sizes, the arbitrary-site 2D CONV evaluator entered its
parallel pool and called current construction, which recursively entered the
same pool while its dispatch mutex was held. A sampled blocked stack confirmed
the deadlock. Native pool callbacks now mark thread-local nesting; nested work
runs serially on its current outer worker. A subprocess regression with a timeout
compares 1-thread and 4-thread profiles exactly. This preserves the mathematics
and leaves the independent outer query work parallel.

## Fused six-photo view

`fusion.py` uses the saved image-only registration receipt as initialization,
reanchors all transforms to photo 3, and accepts direct anchor refinements only
when the existing image-only appearance score improves. This is an anchor
refinement, not a jointly optimized multi-view bundle. It renders the union of
all six footprints with native CONV at valid source locations, estimates bounded
linear-RGB exposure gains against the anchor, and blends in linear light with
border, local sampling-density, and robust Oklab agreement weights. All six
sources contribute. It does not use reference homographies until evaluation.

Run from the repository root on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.fusion --out /tmp/scene_pattern_fusion
mkdir -p experiments/scene_pattern/out/v2/fusion
rsync -a "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/scene_pattern_fusion/" experiments/scene_pattern/out/v2/fusion/
cp experiments/scene_pattern/fusion-viewer.html experiments/scene_pattern/out/v2/fusion/index.html
```

With the existing reviewer served on port 8876, open `/fusion/`. Its controls
show the robust fusion, ordinary weighted average, coverage count, unsuppressed
color disagreement, dominant contributor, and each individual warped source.
The individual source JPEGs are before exposure correction; both composites use
the same corrected observations. `atlas.npz` contains transforms, bounds,
support, contribution fractions and disagreement for subsequent experiments.
`fusion-results.json` preserves the first run's full measurement receipt.

The 960 × 431 union currently reduces many source regions. CONV here evaluates
point samples; this prototype does not perform footprint-integrated antialiasing.
Robust blending reduces color outliers but does not solve residual geometric
misalignment or separate surfaces at different depths. The nonplanar foliage
and ground remain useful failure cases. A single-view region has no independent
agreement evidence; a dark disagreement map there is not proof of correctness.
The Oklab rejection scale of 0.035 and exposure bounds [0.8, 1.25] are prototype
choices, not calibrated uncertainties. The browser only reviews Python output.

Checks: add `experiments.scene_pattern.test_fusion` to the focused unittest
command above. The combined suite passed 36 tests on the Mini, including affine
frame direction, horizon rejection, constant preservation, zero-support masks,
linear-light round trips, outlier suppression with retained disagreement, and
the native dispatch regression.

## Regional residual correction

`residual.py` refines the frozen fusion with local translations centered on
photo 3's causal-density segmentation sites. It estimates each translation from
alternating pixels in a local-contrast patch, and accepts a proposal only when
it improves the complementary pixels by at least 3%. Corrections are bounded,
spatially averaged, shrunk toward zero where support is weak, and fade at the
anchor boundary. Outside the anchor frame, the original homography remains.
The final image samples the original photos using native CONV, not a second
resampling of the previously fused image.

The before/after comparison freezes exposure gains, source-support masks,
sampling-density weights and fusion parameters. It retains the unsuppressed
Oklab disagreement and reports local contrast error, mean fused gradient,
field Jacobian extrema, and evaluation-only reference errors. Alternating
pixels are spatially correlated; passing this holdout is not independent
geometric validation. Lower photometric error can coexist with worse reference
homography error. This is a local appearance correction experiment, not proof
of a recovered nonplanar scene.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.residual
mkdir -p experiments/scene_pattern/out/v2/fusion/residual
rsync -a "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/scene_pattern_residual/" experiments/scene_pattern/out/v2/fusion/residual/
cp experiments/scene_pattern/residual-viewer.html experiments/scene_pattern/out/v2/fusion/residual/index.html
```

Open `/fusion/residual/` on the existing port 8876 reviewer. Focused checks are
`experiments.scene_pattern.test_residual` and `experiments.scene_pattern.test_fusion`.
The tests include independent synthetic fractional displacement recovery,
coordinate direction, invalid-region rejection, and zero-field behavior.

## Eleven freely captured Desktop photos

The room collection uses EXIF-oriented, 900 × 1200 working copies of eleven
3024 × 4032 JPEGs. `prepare_collection.py` preserves original files, records
content hashes, strips metadata from the working copies, and makes a contact
sheet. Images and all collection-specific receipts stay under ignored `data/`
and `out/`; they are local research inputs, not public assets.

The implemented sequence is:

1. `collection.py`: native Meyer Oklab decomposition, causal-density segmentation,
   regional descriptors, mutual matches and robust pairwise projective seeds;
   retain the original direct/phase/ringing/radial registration receipts.
2. `collection_refine.py`: a diagnostic alternative that starts with regional
   seeds and fits local contrast on fixed overlap. The older 45% overlap screen
   rejected a valid narrow-overlap join in this collection. Keep the original
   receipt rather than silently rewriting that experiment.
3. `panorama.py --rotation`: estimate proper camera rotations directly from the
   inlier regional correspondences using robust ray alignment, compose around
   image 6, and render onto a sphere. This mode uses the **seed correspondences**,
   not the projective-refinement matrices. The spherical view avoids the
   unrestricted projective chain's extreme extrapolation. It is an approximate
   rotation model, not calibrated pose or recovered room geometry.
4. Render source samples with native CONV, solve bounded exposure gains across
   overlaps, and compare Oklab robust fusion, ordinary averaging, and graph-cut
   seams. `seams.py` uses sequential binary cuts on a quarter-resolution lattice,
   then a 0.65-output-pixel boundary feather. Seams choose observed pixels; they
   cannot recover hidden surfaces or fix actual parallax.

`panorama.py` currently uses this collection's EXIF 28 mm equivalent focal length
and 3:4 portrait working dimensions to construct approximate intrinsics. It is
not yet a general unknown-camera panorama package. The `--rotation` fallback
without regional points projects a homography to a proper rotation, but the
final eleven-photo result uses regional points for every pair. A linear sampler
is available for fast layout previews; final receipts explicitly name the
sampler so previews are not represented as CONV output.

Preparation example (local):

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.prepare_collection --source '/Users/ultimussecundai/Desktop/untitled folder' --out experiments/scene_pattern/data/desktop-scene
```

Each compute command below uses `m4build -- env OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3` as its prefix.
Copy each `/tmp` output back before the next synchronization:

```text
-m experiments.scene_pattern.collection --data experiments/scene_pattern/data/desktop-scene --out /tmp/desktop_scene
-m experiments.scene_pattern.collection_refine --base experiments/scene_pattern/out/desktop-scene --out /tmp/desktop_refined
-m experiments.scene_pattern.panorama --data experiments/scene_pattern/data/desktop-scene --registration experiments/scene_pattern/out/desktop-refined/registration.json --out /tmp/desktop_final --width 1400 --rotation --sampler conv
```

Copy the final directory to `out/v2/room/`, `collection-viewer.html` to its
`index.html`, the preparation contact sheet to `contact.jpg`, and the refined
registration receipt to `registration.json`. The existing server then exposes
`/room/`. The viewer defaults to the seam-based result and retains all sources,
robust blend, ordinary average, disagreement, coverage, and original inputs.
Focused checks: `test_panorama`, `test_fusion`, and `test_residual` (14 tests).
No reference homographies exist for this collection; lower appearance error is
not an independently measured geometric accuracy result.

## Anti-camera-blur alignment evidence from the ordinary average

`average_alignment.py` reuses the actual personal-deblurrer machinery:
`circles.prepare_phase_circle_spectrum`,
`circles.phase_circle_translation_from_spectra`,
`multicapture_transport._graph_coordinates`, and
`decomposition.estimate_centered_mixing_phase`. The integration does not invoke
the deblurring inverse or treat the average as a sharp reference.

The existing room geometry supplies a common spherical coordinate raster.
At 700-pixel width, 64×64 charts at stride 24 admit only source patches with
full interior coverage and measurable contrast. All usable source pairs in a
chart are considered, including nonconsecutive frames. Fourier-circle centers
receive bounded fractional refinement and a spatially interleaved pixel check.
Disconnected overlap groups are solved independently, preventing an arbitrary
constraint between unconnected observations or a singular graph solve.

For an edge, d_j - d_i = measured_delta_ij. Each connected component uses the
existing robust zero-mean graph gauge. A global or component-common translation
is not observable from these relations. Local graph closure checks consistency;
a two-view edge has no cycle and can always be fit exactly. Pair-only chart
corrections therefore receive lower authority. Smooth overlapping chart weights
and shrinkage construct per-view correction fields, with a local zero-mean gauge
and bounded magnitude. These fields are approximations to spatial transport,
not a solved globally coupled affine/depth/visibility model.

The ordinary average is reconstructed as weighted linear-light averaging of
sources with frozen exposure gains. It is a 700-pixel native re-render of the
same geometry, not the previously displayed 1400-pixel raster. Blind phase
covariance is computed on its estimation charts and individual source charts;
it is retained as a mixing/texture diagnostic, not labeled true registration
variance. Graph-center covariance instead attributes displacement spread to
observed source relations. `alignment_report.py` also records the actual applied
field, its derivatives, and weighted within-component covariance. Unobservable
between-component offsets are not included in that covariance.

The source-sampling correction is q + d_i(q). Thus a positive measured center
shift is removed by sampling farther in that direction; displayed content moves
in the opposite direction. The before/after comparison freezes exposure gains
and uses common native source support. It does not use seam selection or a
sharpening/deconvolution filter. The result remains an experimental candidate;
appearance improvements cannot certify geometry without independent evidence.

Run on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.average_alignment
```

Copy `/tmp/room_alignment/` to `out/v2/room/alignment/` before another sync, then:

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.alignment_report experiments/scene_pattern/out/v2/room/alignment
cp experiments/scene_pattern/alignment-viewer.html experiments/scene_pattern/out/v2/room/alignment/index.html
```

Open `/room/alignment/`. Select chart markers to see per-photo measured and
applied offsets, graph closure, phase-mixing evidence, and a magnified crop.
`fields.npz` keeps floating-point source observations, fixed masks and fields.
`test_average_alignment` verifies graph direction and gauge, disconnected
components, inconsistent cycles, permutation of graph coordinates, and known
fractional displacement recovery.

## Targeted floor-line affine experiment

The user identified doubled tile lines next to the chair. At chart 66 the
previous code measured about 4.61 working pixels of relative separation but
applied about 1.44 pixels after pair-only authority and smoothing. A neighboring
chart estimated a different displacement. Increasing one translation everywhere
would not account for that spatial variation.

`floor_affine.py` selects the indicated floor polygon as a diagnostic ROI, using
IMG_1441 and IMG_1443 where they have observed overlap. This explicit ROI is
not an automatically discovered surface or supplied correspondence. A ±12-pixel
translation survey retains eight separated starts plus the zero start. Each is
optimized as a six-parameter local affine map. Spatial four-pixel blocks are
split into fitting, candidate selection, and reporting-test sets. Hypothesis
choice does not use the reporting-test blocks.

The selected map is split between both sources with exp(±log(H)/2), preserving
a symmetric local coordinate gauge. Full correction is applied inside the
floor polygon and fades smoothly outside it. These two views are rerendered
through native CONV. The other nine cached native views retain their previous
corrections. Before/previous/after comparisons use one common support and the
ordinary weighted average, with no seam changes or sharpening inverse.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.floor_affine --out /tmp/floor_affine_final
```

Copy `/tmp/floor_affine_final/` back to `out/v2/room/alignment/floor/` before
another sync, and copy `floor-viewer.html` there as `index.html`. Open
`/room/alignment/floor/`. `--preview` uses linear resampling for the two changed
views and must not be reported as native-render evidence. Run
`experiments.scene_pattern.test_floor_affine` for independent synthetic affine
recovery, exact interior symmetric splitting, and the identity case.

This remains a two-view plane-local experiment. It demonstrates that multimodal
initialization and a deformable local chart can resolve the chosen line better.
It does not establish automatic surface discovery, selection of the correct
physical plane everywhere, or full-scene correspondence consistency.

### Shared local references for room structures

`structure_alignment.py` extends the floor candidate to the user-identified
bronze lamp, laptop, and doorway lamp. Each manually localized region uses a
chosen observed reference to define a shared coordinate chart. The other views
are fitted against that chart with competing affine hypotheses. The bronze and
laptop fits include broad Oklab shape/color evidence to distinguish repetitive
fine structure. The bronze source with no visible shaft is excluded explicitly;
this is a diagnostic visibility choice, not automatic occlusion inference.

Incremental maps compose with the previous displacement fields by evaluating
the old field at the newly mapped coordinates. Transition widths expand when
necessary to avoid sampled folds; the fitted transform keeps full strength
inside each region. The existing floor polygon is protected in its two corrected
sources. Rejected or insufficient-overlap observations remain in the average
with their previous coordinates. Reference choices are gauges, not truth.

Run fitting and preview on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.structure_alignment --preview --out /tmp/structure_preview4
```

Copy that output to `out/structure-preview4/` before the next sync, then render
both baseline and candidate through native CONV with frozen fitted fields:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.structure_alignment --fitted experiments/scene_pattern/out/structure-preview4 --out /tmp/structure_final
```

Copy `/tmp/structure_final/` back to `out/v2/room/alignment/structures/`, and copy
`structure-viewer.html` there as `index.html`. All photographs, fitted fields,
and capture-specific receipts stay in ignored local output. Test composition
and fold detection with `test_structure_alignment`; affine recovery and gauge
splitting remain covered by `test_floor_affine`.

### Shade evidence audit

The shade work is a controlled diagnosis, not an applied room correction.
Run `experiments.scene_pattern.shade_diagnostic` on the Mini with the same
single-threaded BLAS environment as above. Copy `/tmp/shade_diagnostic/` back
to `out/shade-diagnostic/` before syncing again. Then run
`experiments.scene_pattern.shade_model_audit` on the Mini and copy
`/tmp/shade_model_audit/` back to `out/shade-model-audit/`. Both stages use
linear sampling and preserve the current native room result.

Run `python -m experiments.scene_pattern.shade_report` locally for the small
regional cross-checks and evidence receipt; copy `shade-viewer.html` to the
result directory as `index.html`. The report's pairwise shade fits must not
be described as a new full-scene correction. `registration_audit.py` is reusable
and has no object names, scene coordinates, or manually supplied transforms.
Test it with `experiments.scene_pattern.test_registration_audit`.

### Joint distortion-basis search

`distortion_basis.py` is a scene-independent fitter over caller-supplied regions.
It compares affine, projective, and quadratic source-coordinate maps. Each
observed region receives equal training mass. Missing-image boundaries use
normalized convolution. Candidate generation retains distinct translation
starts; affine fits initialize the projective and quadratic alternatives.
Geometry and per-region selection scores constrain model admission.

`joint_distortion.py` applies this fitter to the existing shade and shaft audit
regions. The manually selected regions and reference remain experiment inputs,
not supplied correspondences. It fits all three other observed captures,
including the capture excluded by the earlier shaft-specific experiment.
Extension is a converged screened harmonic solve with the observed displacement
values held fixed. Every field is composed through the previous map and checked
for sampled folds. Selection and a one-shot final holdout gate evaluate original
captures directly. A final-gate failure retains the prior map and does not tune
or search further against the holdout. Holdout scores are acceptance evidence,
not an independent post-acceptance test or known geometric truth.

Run fitting and preview with the usual Mini environment:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.joint_distortion --out /tmp/joint_validated
```

Copy output back before another sync. `--hypotheses <previous-output>` reuses
fitted model hypotheses while reevaluating continuation and gates; it does not
supply manually changed transform parameters. Render frozen accepted fields:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.joint_distortion --native --fitted experiments/scene_pattern/out/joint-validated-preview --out /tmp/joint_native
```

Copy `/tmp/joint_native/` to `out/v2/room/alignment/joint/` and copy
`joint-viewer.html` there as `index.html`. Test the distortion hierarchy and
continuation with `test_distortion_basis`, and the regional nonregression gate
with `test_registration_audit`. The full native baseline/candidate comparison
uses common source support and ordinary averaging.

### Laptop-only continuation of the accepted scene

`laptop_alignment.py` starts from the accepted joint scene. It uses the existing
laptop region, split into upper/lower validation regions, and a bounded 32-pixel
continuation neighborhood excluding the previously corrected floor. The shared
`fit_joint` routine now accepts arbitrary regions, reference view, contributing
views, and continuation domains. Other scene coordinates remain fixed.

Run the usual Mini environment with `-m experiments.scene_pattern.laptop_alignment`
for preview fitting, then copy `/tmp/laptop_alignment/` to `out/laptop-preview/`
before another sync. `--partial-view 6` tests visibility changes with at least
75% retained soft overlap per region. `--absolute` additionally fits against the
original spherical source atlas, rather than the previous local warp. These
optional experiments do not bypass direct-source or final holdout checks.

For checked native rendering:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.run_laptop_validation --native --fitted experiments/scene_pattern/out/laptop-preview --out /tmp/laptop_native
```

Copy `/tmp/laptop_native/` to `out/v2/room/alignment/laptop/` before the next sync.
Run `python -m experiments.scene_pattern.laptop_report <that-directory>` locally
for the viewer, comparison, source fingerprints, and exact pixel-preservation
check outside the laptop domain. The original drape is scene structure and is
not a target of this refinement.

## End-to-end automation requirement

Read [AUTOMATION_CONTRACT.md](AUTOMATION_CONTRACT.md) before further pipeline
work. The object-directed experiments above are diagnostic evidence only.
The final method must discover and schedule its own correspondences and updates
from photographs under a fixed policy. Reusable fitting routines plus manually
chosen repairs are not an automatic solution. This room is development data,
not an unseen validation collection.

## Automatic panorama-first walking collection

Read `SWEEP_METHOD.md` for the fixed policy and limits. This independent path
uses no room annotations or inherited repair fields. Private inputs and outputs
remain inside ignored `data/` and `out/` directories.

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.sweep_collection \
  --source /Users/ultimussecundai/Desktop/big_capture \
  --data experiments/scene_pattern/data/big-capture
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m experiments.scene_pattern.sweep_features \
  --data experiments/scene_pattern/data/big-capture --out /tmp/big_capture_features
# Copy /tmp/big_capture_features/ back to out/big-capture-features/ before syncing.
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m experiments.scene_pattern.run_sweeps \
  --data experiments/scene_pattern/data/big-capture \
  --features /tmp/big_capture_features --out /tmp/big_capture_run
# Copy /tmp/big_capture_run/ back to out/big-capture-run/ before syncing.
.venv-jpeg/bin/python -m experiments.scene_pattern.sweep_report \
  --run experiments/scene_pattern/out/big-capture-run \
  --data experiments/scene_pattern/data/big-capture
```

`run_sweeps` runs synthetic tests, the whole candidate graph, then both mesh
stages and the linear-sampling preview. The receipt records unresolved inputs,
source hashes, graph edges, mesh residuals, fold prevention and per-source
rendered coverage. Keep the ordinary average alongside information weighting.
The current mesh representation does not recover metric 3-D or prove visibility.

## Multiple-origin evidence and reconstruction

The walking collection's earlier flat and periodic atlases were rejected as
geometrically wrong. Read `CAPTURE_GEOMETRY.md`; those rendered atlases are not
initial geometry for this path. `scene_evidence.py` only reuses their candidate
pair inventory, alongside the original Oklab/Meyer regional descriptors.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m experiments.scene_pattern.scene_evidence \
  --data experiments/scene_pattern/data/big-capture \
  --features /tmp/big_capture_features \
  --candidate-graph /tmp/big_capture_run/graph/graph.json \
  --out /tmp/scene_evidence_v2
# Copy that directory back to out/scene-evidence-v2 before another sync.
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m experiments.scene_pattern.scene_pose \
  --evidence /tmp/scene_evidence_v2/evidence.json --out /tmp/scene_pose.json
# Copy that file back immediately before another sync.
```

`multiview_geometry.py` compares proper rotation and essential-matrix hypotheses
on spherical bearings. Candidate alternatives and duplicate descriptor scales
cannot multiply physical-site votes. A fixed-site holdout and a 63-permutation
null comparison screen pair support. These tests are conditional on descriptor
selection, not independent scene truth. The current multiple-comparison screen
is exploratory and does not control a collection-wide false-discovery rate.

Tracks enforce one site per capture and retain incompatible joins. Bridge
observations do not inherit the independent cycles elsewhere in their track.
Photographs can overlap multiple panoramas; these affinities are not exclusive
station assignments. Greedy track construction is still a limitation: recording
an alternative does not mean the full alternative graph was jointly solved.

`scene_pose.py` proposes independent camera origins and positive-depth structure
from panorama seeds, then checks predictions in further captures. Rotation and
baseline direction can initialize a new camera, but their scale is free. Every
candidate touching a held-out target site is removed before fitting even these
initializations. The pose holdout uses physical target-site IDs, not mutable
track numbers. Track discovery and seed selection still see collection evidence;
passing this conditional gate is not independent reconstruction validation.

A fixed eight-family panorama calibration screen runs with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m experiments.scene_pattern.scene_calibration \
  --evidence /tmp/scene_evidence_v2/evidence.json --features /tmp/big_capture_features \
  --out /tmp/scene_calibration_v2
# Copy the whole directory back before another synchronization.
```

It compares cylindrical and equirectangular projections with common focal
multipliers 0.8, 1, 1.25 and 1.6. This stage only reconstructs panorama tracks;
it does not inherit the all-capture greedy graph or force the ordinary photos
onto a panorama camera. Selecting a family with these checks requires a further
independent audit. Local panorama stitching and region correspondence uncertainty
are not modeled by this finite global family.

Run focused mechanism checks with `python -m unittest
experiments.scene_pattern.test_multiview -v`. The synthetic three-camera check
must recover separate origins and reject poisoned held-out observations while
keeping the seed and training observations exact.

Generate the private source reviewer:

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.scene_evidence_report \
  --evidence experiments/scene_pattern/out/scene-evidence-v2/evidence.json \
  --pose experiments/scene_pattern/out/scene-evidence-v2/pose.json \
  --data experiments/scene_pattern/data/big-capture \
  --out experiments/scene_pattern/out/v2/capture
```

Source images, rejected reconstruction attempts, uncertainty and candidate
observations remain inspectable. A sparse graph or cloud is not a fused scene.

Current evidence and failed audits are recorded in `MULTIVIEW_FINDINGS.md`.
`scene_promotion --evidence <evidence.json> --pose <pose.json> --out
<pose-audited.json>` checks each point against every fitted view. Pass the audited
pose and optional `--calibration <screen.json>` to `scene_evidence_report` to show
third-view consistency and the complete projection-family comparison.


## Continuous regional transport and photo attachment

Read `REGIONAL_TRANSPORT_FINDINGS.md` for the current results and limits. This
path uses regional image transport and separate capture charts; it does not
inherit the rejected common-origin atlas. Use the existing prepared feature
cache and evidence inventory from the commands above.

Run each numerical command on the selected Mini with this prefix:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m <module> <arguments>
```

The modules and arguments, in order, are:

```sh
experiments.scene_pattern.capture_transport \
  --evidence /tmp/scene_evidence_v2/evidence.json \
  --features /tmp/big_capture_features --out /tmp/capture_panorama --workers 4

experiments.scene_pattern.dense_transport \
  --transport /tmp/capture_panorama --features /tmp/big_capture_features \
  --out /tmp/dense_panorama --workers 4

experiments.scene_pattern.dense_scene \
  --dense /tmp/dense_panorama --data experiments/scene_pattern/data/big-capture \
  --out /tmp/panorama_charts --width 1536

experiments.scene_pattern.capture_transport \
  --evidence /tmp/scene_evidence_v2/evidence.json \
  --features /tmp/big_capture_features --out /tmp/capture_all \
  --all-captures --workers 4

experiments.scene_pattern.attach_fields \
  --base /tmp/dense_panorama --transport /tmp/capture_all \
  --features /tmp/big_capture_features --out /tmp/attached_fields --workers 4

experiments.scene_pattern.dense_scene \
  --dense /tmp/attached_fields --data experiments/scene_pattern/data/big-capture \
  --out /tmp/attached_charts --width 1536

experiments.scene_pattern.dense_points \
  --scene /tmp/attached_charts/regional-scene.json --out /tmp/attached_points
```

Copy **each complete output directory back immediately before another mirror
sync**, using the selector, for example:

```sh
rsync -az "$('/Users/ultimussecundai/.local/bin/m4host'):/tmp/attached_points/" \
  experiments/scene_pattern/out/attached-points/
```

Use separate panorama-only and all-capture transport receipts. A scoped
`capture_transport` run can resume compatible pair files, but its `transport.json`
reflects the last requested scope. Do not silently substitute an all-capture
receipt for the panorama-stage input. Existing output policies bind source and
feature hashes; changed algorithms require a new output directory.

The retained development directories use the versioned names in the findings.
Build their current private reviewer locally:

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.transport_report \
  --fusion experiments/scene_pattern/out/attached-scene-v1 \
  --attachment experiments/scene_pattern/out/attached-fields-v1 \
  --out experiments/scene_pattern/out/v2/capture/transport

.venv-jpeg/bin/python -m experiments.scene_pattern.scene_evidence_report \
  --evidence experiments/scene_pattern/out/attached-points-v1/evidence.json \
  --pose experiments/scene_pattern/out/attached-points-v1/pose.json \
  --data experiments/scene_pattern/data/big-capture \
  --out experiments/scene_pattern/out/v2/capture/transport/geometry
```

The reviewer serves rendered Python results. It is not a browser registration
engine or a complete 120-camera reconstruction. Reference-view support, graph
attachment, and accepted physical camera poses are distinct quantities.

Run focused mechanism checks with:

```sh
.venv-jpeg/bin/python -m unittest \
  experiments.scene_pattern.test_region_transport \
  experiments.scene_pattern.test_multiview -v
```

Use the same modules through `m4build` for the Mini runtime check. Keep the
ordinary mean and fixed support comparison intact; a visually sharper seam or
reduced coverage is not a replacement for improved registration.


## Completed observations and the movable surface diagnostic

Using the same Mini environment prefix above:

```sh
experiments.scene_pattern.scene_completion \
  --scene /tmp/attached_scene_v1/regional-scene.json \
  --fields /tmp/attached_fields_v1 --out /tmp/completed_scene_v1.json

experiments.scene_pattern.dense_points \
  --scene /tmp/completed_scene_v1.json --out /tmp/completed_points_v1

experiments.scene_pattern.scene_bundle \
  --evidence /tmp/completed_points_v1/evidence.json \
  --pose /tmp/completed_points_v1/pose.json \
  --out /tmp/calibrated_bundle_converged_v1.json \
  --calibrate-panoramas --max-evaluations 2000
```

Run the fixed-intrinsic audit by omitting `--calibrate-panoramas` and choosing
another output name. Copy every output, including `.candidate.json` files, before
another Mini sync. A rejected bundle output retains the input geometry and a
receipt; its candidate is saved separately. Cost-tolerance termination is not a
proof of a global or tightly stationary solution.

The current bundle candidates failed their holdout gates. Build the diagnostic
from the retained pre-bundle geometry:

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.scene_surface \
  --evidence experiments/scene_pattern/out/completed-points-v1/evidence.json \
  --pose experiments/scene_pattern/out/completed-points-v1/pose.json \
  --data experiments/scene_pattern/data/big-capture \
  --out experiments/scene_pattern/out/v2/capture/scene
```

The WebGL page exposes surface failures, not a completed shared scene. Texture
selection for its triangles is separate from the equal-support ordinary-average
fusion comparison. Add `test_scene_completion`, `test_scene_bundle`, and
`test_scene_surface` to the focused tests. Run `scene_evidence_report` with the
completed-points evidence and pose to refresh the linked camera reviewer.

## Differential surfaces, camera growth, and local map joins

The current retained hypothesis has 25 poses (17 panorama and eight ordinary
photo), 2,024 consistent points and 492 checked textured tangent patches. The
full panorama-seed survey covers 53 distinct cameras only as separate local
hypotheses; none of the attempted joins passed. See the findings for the control
run, failed holdout design, accepted refinement and unresolved projection issue.

Run each module below with the Mini prefix:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 \
  /usr/bin/python3 -m <module and arguments below>
```

```text
experiments.scene_pattern.surface_jets
  --evidence /tmp/completed_points_v1/evidence.json
  --pose /tmp/completed_points_v1/pose.json
  --fields /tmp/attached_fields_v1 --out /tmp/surface_jets_v2.json

experiments.scene_pattern.jet_bundle
  --evidence /tmp/completed_points_v1/evidence.json
  --pose /tmp/completed_points_v1/pose.json
  --jets /tmp/surface_jets_v2.json --out /tmp/jet_bundle_v2.json

experiments.scene_pattern.scene_growth
  --evidence /tmp/completed_points_v1/evidence.json
  --pose /tmp/jet_bundle_v2.json --out /tmp/jet_growth_v1.json

experiments.scene_pattern.scene_submaps
  --evidence /tmp/completed_points_v1/evidence.json
  --out /tmp/scene_submaps_v1 --workers 4

experiments.scene_pattern.merge_submaps
  --evidence /tmp/completed_points_v1/evidence.json
  --pose /tmp/jet_growth_v1.json --submaps /tmp/scene_submaps_v1
  --out /tmp/merged_submaps_v1.json
```

Copy each JSON and `.candidate.json`, and the complete submap directory, back
before another Mini synchronization. The baseline growth control uses the
original `completed_points_v1/pose.json` and a distinct output. Policy-changing
runs must use new output paths. Retain failed candidates and audit receipts.

Build the current private viewer locally:

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.scene_surface \
  --evidence experiments/scene_pattern/out/completed-points-v1/evidence.json \
  --pose experiments/scene_pattern/out/jet-growth-v1.json \
  --jets experiments/scene_pattern/out/jet_bundle_v2.json \
  --data experiments/scene_pattern/data/big-capture \
  --out experiments/scene_pattern/out/v2/capture/differential
```

Add `test_surface_jets`, `test_jet_bundle`, `test_scene_growth`,
`test_submap_alignment`, and `test_merge_submaps` to the existing focused suite.
The viewer is a sparse, conditional surface diagnostic; reconstruction still
runs in Python on the Mini. It is not the eventual browser reconstruction engine.

The differential calibration alternative uses the same Mini prefix:

```text
experiments.scene_pattern.jet_bundle
  --evidence /tmp/completed_points_v1/evidence.json
  --pose /tmp/completed_points_v1/pose.json
  --jets /tmp/surface_jets_v2.json --calibrate-panoramas
  --max-evaluations 1000 --out /tmp/calibrated_jet_bundle_v1.json
```

Copy both output JSONs before another sync. The retained candidate reached its
cap and improves aggregate residuals while seriously worsening some cameras;
it is an experimental alternative, not the promoted shared scene. Growth from
it writes `/tmp/calibrated_jet_growth_v1.json`. Render with that pose and its
calibrated bundle jets into `out/v2/capture/calibrated/`. Preserve the fixed-camera
view for comparison. The diagnostic `jet_residual_audit --pose ... --evidence ...
--out ...` reports every patch's center and offset-removed differential error,
including per-camera holdouts. Add `test_jet_calibration` and
`test_jet_residual_audit` to the focused suite.

For stage-two coverage, `scene_submaps --seed-scope all` evaluates every supported
translation pair, including panorama/photo and photo/photo pairs. Default scope
remains `panorama`. Write a new output directory such as
`/tmp/all_scene_submaps_v1`; its source-bound policy prevents silently reusing a
result from another scope. These additional seeds do not bypass the third-view,
physical-site holdout, parallax, or separate-origin requirements. Keep their
independent frames until an actual map join passes.

## Neighbor-map reconciliation and joint candidates

Run these modules with the same selected-Mini prefix and copy every complete
output directory back before another synchronization:

```text
experiments.scene_pattern.submap_graph
  --evidence /tmp/completed_points_v1/evidence.json
  --submaps /tmp/all_scene_submaps_v1
  --out /tmp/submap_graph_v1 --workers 4

experiments.scene_pattern.reconcile_graph
  --evidence /tmp/completed_points_v1/evidence.json
  --submaps /tmp/all_scene_submaps_v1
  --graph /tmp/submap_graph_v1/graph.json
  --out /tmp/reconciled_components_v1 --workers 4

experiments.scene_pattern.joint_submap
  --evidence /tmp/completed_points_v1/evidence.json
  --pose /tmp/jet_growth_v1.json
  --submaps /tmp/all_scene_submaps_v1
  --joins /tmp/all_submap_joins_v1.json
  --out /tmp/joint_submap_v1.json --max-evaluations 300
```

The graph checks both directions independently and checks the frame round trip.
A component is only a proposal for reconciliation, not permission to concatenate
its points. Reconciliation refreshes and deduplicates structure, then rechecks
each merge against accumulated evidence. Components retain separate frames.

The joint candidate ranks failed frame initializers by training links into new
cameras, retains one variable per shared camera, and jointly refines the union.
It must pass both aggregate and per-camera prediction gates before replacing its
parent. Its `.candidate.json` is retained even when rejected. Add
`test_submap_graph`, `test_reconcile_graph` and `test_joint_submap` to the focused
suite. The synthetic neighbor chain includes a final map with no direct point
overlap with the first map.

### Matched surface-continuity study

From the local authoritative bfft tree:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.coherence_study --evidence /tmp/completed_points_v1/evidence.json --pose /tmp/reconciled_components_v1/078.json --jets /tmp/neighbor_surface_jets_v1.json --out /tmp/coherence_study_v2 --max-evaluations 1000 --retain-point-support
```

The two CPU worker processes compare identical input hypotheses with and without
training-derived neighborhood continuity. Omit `--retain-point-support` only
to reproduce the earlier surface-subset control, using a different output path.
Both branches protect per-camera patch predictions; the point-supported branch
also checks its separate point predictions. Every rejected candidate is saved.
Copy the complete output directory from the selected Mini into
`experiments/scene_pattern/out/coherence-study-v2/` before another synchronization.
See `REGIONAL_TRANSPORT_FINDINGS.md` for the failed v1 comparison and caveats.
Focused additions: `test_surface_coherence`, `test_point_support`, and the expanded
`test_jet_bundle` integration tests. Default `jet_bundle` behavior remains the
legacy surface fit; the new single-run flags are `--coherent-surfaces`,
`--protect-cameras`, and `--retain-point-support`.

For the fixed-camera point prediction audit, run on the selected Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /usr/bin/python3 -m experiments.scene_pattern.point_support_audit --evidence /tmp/completed_points_v1/evidence.json --pose /tmp/reconciled_components_v1/078.json --jets /tmp/neighbor_surface_jets_v1.json --candidate /tmp/coherence_study_v2/coherent.candidate.json --out /tmp/coherence_point_control_v2.json
```

Copy that receipt before another sync. `test_point_support_audit` checks that
withheld rays cannot affect this control. Stored points are a stability reference;
the training-only control is a separate conditional prediction comparison.
`scene_surface --trial` explicitly labels rejected candidate geometry, avoiding
confusion with the retained output.

### Guarded spatial prediction and all-view surface fields

Run the matched four-fold fixed-camera study on the selected Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.spatial_surface_study --evidence /tmp/completed_points_v1/evidence.json --pose /tmp/reconciled_components_v1/078.json --jets /tmp/neighbor_surface_jets_v1.json --out /tmp/spatial_surface_v1 --max-evaluations 300 --workers 2
```

Add `--joint-cameras` and use `/tmp/spatial_joint_v1` for the matched camera-motion
study. Copy the complete output directory back before another synchronization.
Every fold saves candidate and retained geometry, predictions and a receipt.
The combined summary asserts that every original observation center is accounted
for once and that the two branches have identical prediction support. Unresolved
observations remain in counts; their absence is not a registration improvement.

`full_surface_field` subsequently fits every observation and attaches the already
computed blocked predictions after optimization. It requires the same camera
policy and source inputs as the study. An all-view fit has no new prediction
holdouts and cannot claim a new independent pass. Render its unverified fitting
support only with `scene_surface --training-estimates --trial`; that geometry is
kept in a separate switchable buffer from patches passing all blocked predictions.
Focused additions: `test_spatial_surface_study` and `test_full_surface_field`.

### Revisable regional identities and ringing

From the authoritative local bfft tree, run the bounded all-capture probe:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.identity_study --data experiments/scene_pattern/data/big-capture --features /tmp/big_capture_features --out /tmp/identity_register_v2 --verify-per-capture 12 --workers 3
```

Copy the entire output directory to `out/identity-register-v2/` before another
sync. A fresh output path is required for another policy or budget. The candidate
ledger retains untested identities; verification compares phase-initialized
local affine fits with and without ringing delay proposals. `identities.json`
retains competing transforms, third-capture support and relative weights with an
unresolved alternative. These are uncalibrated scores, not station assignments.
No old track identities or scene poses enter this run. No result automatically
replaces retained reconstruction geometry. Run `test_identity_register` with the
existing regional and radial tests; its repeated-pattern case verifies that a
third image can reverse a pairwise preference without destroying either identity.

The expanded run uses `--out /tmp/identity_register_v3 --verify-per-capture 192
--workers 6`; keep these on the same command line as the other options. It
checks six identities at each selected region. Copy it to
`out/identity-register-v3/`. Build the local reviewer with:

```sh
.venv-jpeg/bin/python -m experiments.scene_pattern.identity_view --study experiments/scene_pattern/out/identity-register-v3 --data experiments/scene_pattern/data/big-capture --features experiments/scene_pattern/out/big-capture-features --out experiments/scene_pattern/out/v2/capture/identity
```

The reviewer includes original image context, registered crops, an ordinary
average and alternative scores. It does not update the scene viewer's geometry.

### Geometry and identity feedback

The full retrieval-inventory run uses `identity_study --verify-per-capture 576`
with `/tmp/identity_register_complete_v1`. Copy the entire output before a sync.
Grow competing image fields from it on the selected Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.identity_fields --identities /tmp/identity_register_complete_v1/identities.json --data experiments/scene_pattern/data/big-capture --features /tmp/big_capture_features --out /tmp/identity_fields_complete_v1 --workers 6
```

Copy `/tmp/identity_fields_complete_v1/` into `out/identity-fields-complete-v1/`
before another synchronization. Run one geometric association iteration:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.identity_scene_step --fields /tmp/identity_fields_complete_v1 --data experiments/scene_pattern/data/big-capture --features /tmp/big_capture_features --out /tmp/identity_scene_complete_v1 --workers 6
```

The step runs relative-pose alternatives, three-view loops/scale hypotheses, and
joint camera/local-depth fitting in one unchanged remote source snapshot. Copy
its whole output tree to `out/identity-scene-complete-v1/` before another sync.
It preserves rejected hypotheses and never claims global scene promotion from
a local pass. A photo/panorama pair remains an overlap hypothesis, not a station
assignment. Retain the panorama-first final reconstruction requirement.

Focused additions: `test_identity_fields`, `test_affine_camera_geometry`,
`test_identity_geometry`, `test_identity_pose_graph`, and `test_identity_joint`.
Tests cover cylindrical ray derivatives, planar ambiguity, competing identities,
straight-line direction degeneracy, joint prediction and reserved-data poisoning.

To replay earlier training-selected camera branches under the complete fields,
add `--prior-geometry /tmp/identity_geometry_v1` to `identity_scene_step` and use
a fresh output such as `/tmp/identity_scene_replay_v2`. All previous matrices,
including previously rejected ones, are eligible; capture/calibration and
training/check-role compatibility are checked before replay. Original proposals
and pair-refined descendants remain alternatives for the joint fit. Copy the
whole output tree before another sync. These real replay runs remain rejected;
see the findings for exact prediction failures and the next image-evidence
integration step.

### Reverified panorama identity pool

Combine the older image-local proposals with the new ringing fields, retaining
physical source-site keys and distinct target alternatives. No old camera pose,
track union, cycle vote or nominal correlation bound is inherited. All inputs
are refitted against the current Lab images in both directions. The default
scope is the panoramic captures; `--scope all` is available for the second stage.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=4 /usr/bin/python3 -m experiments.scene_pattern.identity_pool --dense /tmp/attached_fields_v1 --fields /tmp/identity_fields_complete_v1 --identities /tmp/identity_register_complete_v1/identities.json --data experiments/scene_pattern/data/big-capture --features /tmp/big_capture_features --out /tmp/identity_pool_panoramas_v1 --scope panoramas --workers 6
```

Copy `/tmp/identity_pool_panoramas_v1/` into `out/identity-pool-panoramas-v1/`
before another sync. The pool saves rejected fits, proposal provenance, input
hashes and executed source snapshots as well as retained alternatives. Source
coordinates are transported to a common two-pixel lattice before re-fitting;
source IDs do not depend on the parent grid or its array order. Distinct target
maps survive separately, up to four per source site under the fixed training
score. Neighboring sites remain correlated measurements, not independent votes.

Pass `--scope panoramas` to `identity_geometry` or `identity_scene_step` for the
first geometry stage. Capture IDs remain the original 120-image indices; the
scope filter never renumbers them or imposes shared camera centers. Add
`test_identity_pool` to the identity/geometry tests.

`identity_scene_step --initialization image-wedges --scope panoramas` allows
shared image evidence to initialize a joint three-view fit before independently
estimated pair rotations have closed. It considers every available source
anchor, estimates relative scale from training regions, and lets the joint
optimizer revise both camera poses and image alternatives. The third pair needs
image maps but need not already have an accepted relative-camera estimate.
Final joint prediction and third-pair gates are unchanged. Use a fresh output
such as `/tmp/identity_scene_panoramas_wedges_v1` and copy it back immediately.
The original `closed-loops` initialization remains available as a control.
Add `test_identity_wedges` to the focused tests. In an image-wedge receipt,
`orientation_loops: 0` means that closed loops were not the initializer; it must
not be interpreted as a post-fit closure count.
