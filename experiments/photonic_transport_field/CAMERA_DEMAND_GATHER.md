# Camera information audit: demand gathering and compact crossings

The subsequent [boundary discovery audit](BOUNDARY_INFORMATION_REUSE.md) reduces
the standard-view boundary stage further and records acceleration explicitly.
The timing tables below retain the earlier camera baseline.

The September 5 camera follow-up changes the actual retained renderer in
`native/regime_scene_native.cpp`. The target remains a complete CPU-rendered
waterfall/grass/free-flight frame at **at least 800×600, 30 fps (33.33 ms)**.
This pass improves camera evaluation; it does not claim that target is complete.

## What the camera was doing unnecessarily

Before any image lookup, `compile_viewer_origin_specular_field` integrated
adaptive glossy atlases across every supported glossy/metal rectangle and
sphere in the scene. It did this even for surfaces and portions of surfaces
that no camera query subsequently used. In the paired standard-view run, this
serial setup took about 294 ms and evaluated 16,543 source samples. An overhead
view constructed 15,560 samples despite making no gathers from that field.

The camera also represented every horizontal and vertical sensor adjacency with
a full `GridCrossing`, including the overwhelmingly common absence of an optical
ownership crossing. At 800×600 the two crossing arrays occupied 53,681,600 bytes.
This was temporary crossing storage, not the total renderer memory footprint.

Source shading, primary ownership, topology signatures, spectral path discovery,
and boundary reconstruction still contain other overlapping work. This pass
retains their existing decisions and output while removing the two demonstrated
representation costs.

## Evaluate the same finite function only where requested

The old glossy atlas defines a piecewise bilinear function. A coarse cell uses
its four corner values unless the tone-mapped centre differs from the predicted
centre by more than one byte. A refined cell uses a 5×5 set of samples. Those
rules, coordinates, quadrature, sphere seam, BRDF, and interpolation arithmetic
are unchanged.

The demand representation gives each fine-grid coordinate one shared sample and
each coarse cell one shared refinement decision. A gather proceeds as follows:

1. Map the actual hit to the same surface coordinates as the eager atlas.
2. Request the cell's centre and four corners to make the original refinement
   decision, if it has not already been made.
3. Request the four interpolation ordinates for the selected coarse or refined
   cell. A refined gather does not need all 25 fine ordinates immediately.
4. Interpolate with the original arithmetic. Retain each requested ordinate and
   decision for subsequent gathers.

No occlusion guess or screen-space approximation is needed to decide which
samples can be omitted: an unused ordinate cannot affect a gather. An omitted
ordinate is constructed if a later gather requests it. Shared cell boundaries
and the periodic sphere seam refer to the same sample. The existing one-byte
centre test remains a heuristic finite refinement test, not a global physical
error certificate; this work preserves it rather than strengthening its claim.

Each sample and cell decision is published through `std::call_once`. The samples
can therefore be constructed within the camera's existing parallel work, and
simultaneous requests integrate a sample only once. The evaluator uses immutable
scene/lighting state and local counters. Atomic totals record new work after
worker completion.

The field belongs to an exact eye origin and lighting generation. Repeated
queries retain their values. Rotation at the same origin extends the field with
newly requested samples. Translation or a changed lighting generation rejects
the field through the existing validity check. This is **not** a translation
transport certificate or a persistent cache of complete rendered frames.

The current implementation reserves a dense addressable fine-sample array and
cell flags per supported primitive, although it evaluates only demanded values.
A rectangle reserves `129×129` sample slots and `32×32` cell decisions; a sphere
reserves `129×65` slots and `32×16` decisions. The synchronization state and
retained samples trade persistent memory for reduced integration. Sparse block
allocation is a possible later improvement; this is not yet a fully continuous,
source-defined region representation.

## Store crossings where crossings exist

The new crossing representation keeps a signed integer index per sensor
adjacency, with `-1` for no crossing. The crossing's geometry and ownership pair
live in a packed vector only when present. Refinement, traversal order, boundary
pairing, line fragments, and reconstruction are unchanged. The dense form stays
available as an oracle through `--camera-crossings dense`.

For the measured standard views, allocated crossing storage fell from
**53,681,600 to 4,293,152 bytes**, a **92.0% reduction**. Default-view edge discovery
fell from a median 68.18 ms to 63.99 ms with the eager shading path held fixed.
This is a modest timing gain and a substantial reduction in temporary storage;
it does not mean total renderer memory fell by 92%.

The audit also repaired camera-frustum contour discovery for buffered geometry
insertions. It now considers primitives in the insertion delta outside the old
BVH, matching the retained ray-intersection path. A focused test covers an
inserted primitive in front of the observer.

## Paired CPU measurements

The machine reports Apple M4. The main benchmark compiles the standard scene
once and measures camera work at 800×600, terminal error 1, over four poses.
Each pose has three repetitions of each representation, with the mode order
rotated between repetitions. Every resulting image is compared byte for byte.
The three modes are:

- `eager-dense`: original eager glossy field and dense crossing storage.
- `eager`: eager glossy field with compact crossings; isolates crossing storage.
- `demand`: demand glossy field and compact crossings; the new default.

| Pose | Original camera median (ms) | Eager + compact crossings (ms) | Demand + compact crossings (ms) | Original / demand |
|---|---:|---:|---:|---:|
| Default | 567.69 | 561.27 | 295.33 | 1.92× |
| Rotated observer | 567.98 | 563.65 | 294.76 | 1.93× |
| Translated observer | 561.33 | 556.86 | 285.44 | 1.97× |
| Prism overhead | 573.06 | 571.58 | 300.06 | 1.91× |

These are camera-only times, excluding fresh scene/lighting compilation, output,
and display presentation. The benchmark helper uses the Scene default of BVH
traversal. Initial command-line measurements (383 ms versus 220 ms) instead used
the CLI's automatic linear traversal for this 33-primitive scene. The later
boundary audit identified this configuration difference; it should not be
explained merely as timing variation on the shared host. Each paired comparison
used matching traversal, so its speedup remains valid. The new boundary records
name acceleration explicitly, including both policies for the standard view.

The default demand view evaluates **7,856 versus 16,543** glossy source samples,
and **771,563 versus 1,304,907** source quadrature samples. The overhead view
requests **zero** glossy source samples and avoids all of its unused eager work.
All four poses remain byte-identical across all three representations.

An explicitly retained field was also exercised across frames:

| Request | New glossy samples | New quadrature samples | Camera time (ms) |
|---|---:|---:|---:|
| First default view | 7,856 | 771,563 | 299.11 |
| Identical second view | 0 | 0 | 266.96 |
| Rotate the same observer | 1,955 | 106,365 | 270.97 |

This retains source responses; the second frame still recomputes ownership,
optical paths, boundary geometry, and radiance gathers. It is not a static-frame
blit. All three images match fresh eager evaluation.

Three additional scenes were checked with BVH acceleration at 800×600. These
are single pairs for broader correctness and cost coverage, not median gates:

| Scene | Eager + dense camera (ms) | Demand + compact camera (ms) |
|---|---:|---:|
| Aperture canyon | 2,026.96 | 1,409.79 |
| Mirror relay | 2,636.57 | 1,163.87 |
| Occlusion garden | 1,275.03 | 499.41 |

Each pair is byte-identical. The larger residual camera costs in the first two
scenes are important: boundary reconstruction alone costs about 986 ms and
765 ms, respectively. The standard-room improvement is not evidence that all
optical scenes now share its frame time.

The existing retained geometry-update benchmark was rerun at 800×600 with
its default BVH traversal:

| Edit | Lighting repair (ms) | Camera (ms) | Engine frame including geometry (ms) |
|---|---:|---:|---:|
| First move | 77.15 | 284.97 | 362.12 |
| Second move + observer translation | 77.96 | 286.21 | 364.17 |
| Insertion + observer translation | 138.25 | 285.90 | 424.15 |

These edited frames are byte-identical to fresh compilation of the same updated
scene/camera. Their 362–424 ms cost remains about **11–13× the 33.33 ms budget**.
The earlier update baseline was 636–703 ms, recorded in
[RETAINED_SCENE_UPDATES.md](RETAINED_SCENE_UPDATES.md). Fresh lighting compilation
has not been optimized in this camera pass.

## The next mathematical target

For a fixed geometry state, the incident light at a surface can be treated as a
measure over directions:

\[
d\mu_x(\ell)=L_i(x,\ell)\,(n_x\cdot\ell)_+\,d\omega,
\qquad L_o(x,v)=\int f_r(x,\ell,v)\,d\mu_x(\ell).
\]

Moving the observer changes `v`. It need not reconstruct the geometry-dependent
measure `mu_x`. The current code still folds source visibility integration and
the view-dependent BRDF into each glossy sample. Demand gathering removes unused
samples, but translating the camera still rebuilds the needed samples' source
partitions. Retaining an incident directional response, then applying the
observer-dependent weighting, is the next useful separation. A finite basis or
regional representation would need explicit error control for the existing
sharp lobes; an arbitrary low-order angular approximation would change the task.

The remaining default-view stages are approximately 104 ms for adaptive camera
evaluation, 64 ms for edge discovery, and 127 ms for boundary reconstruction.
The latter requests 218,135 labels and 29,889 radiance evaluations in the
reference default view. The code separately walks paths for ownership signatures,
spectral-region discovery, and radiance. Mixed prism spectra additionally seed
eight intervals in each band and refine topology changes before evaluating
radiance. Those operations overlap geometrically, but different wavelengths can
have genuinely different refracted paths. They cannot simply share an RGB hit
without checking that equivalence.

A further camera reworking should retain geometric optical continuations and
their validity regions, so topology and radiance consume the same discovered
geometry. It should also move boundary ownership toward scene-defined curves
and regions. The current implementation continues to rediscover and reconstruct
sensor boundaries every frame. Those are the measured remaining targets, rather
than the already-microsecond diffuse residual march.

No illumination effect, shimmer, stochastic term, reduced resolution, enlarged
error tolerance, or changed BRDF was introduced by this pass. Geometry and scene
signals remain the source of visibility and optical behavior.

## Validation and reproduction

Run from the authoritative checkout:

```sh
sh experiments/photonic_transport_field/run_retained_camera.sh
```

The script uses `m4build` and `m4host`, builds outside the shared mirror's binary
paths, and copies records to `camera_gather_m4/`. It runs the native scene and
retained-update invariants, the camera-specific suite, the four-pose benchmark,
the edited-scene benchmark, and all three additional scene comparisons. ASan
and UBSan instrument both the C++ engine and C CONV dependency. ThreadSanitizer
runs the concurrent camera suite with both dependencies instrumented. All passed.
The sanitizer build alone reports three existing unfulfilled vectorization
requests; the optimized build is warning-free.

The camera-specific suite compares eager and demand gathers with exact double
equality at surface edges, corners, sphere seams/poles, and interior samples.
Eight workers request overlapping samples in different orders. A repeated query
pass must add no work. The image benchmark independently checks complete camera
output, retained-field reuse, and rotation. These establish representation and
concurrency correctness on the covered cases, not continuous radiometric accuracy
or worst-case frame latency.

Primary records:

- [Paired camera benchmark](camera_gather_m4/retained_camera_benchmark.json)
- [Integrated geometry updates](camera_gather_m4/retained_camera_updates.json)
- [Validation log](camera_gather_m4/validation.log)
- [ThreadSanitizer log](camera_gather_m4/thread_sanitizer.log)
- [Source hashes](camera_gather_m4/source_hashes.json)
