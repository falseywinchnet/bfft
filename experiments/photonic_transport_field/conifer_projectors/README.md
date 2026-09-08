# A conifer, one light, and a registered smoke volume

This standalone experiment implements and tests the user's distinction: **a diffuse hit terminates the identity-bearing geometric path and registers the received energy as that surface's own outgoing source**. A later receiver consumes that source distribution. It does not reconstruct the source's incoming reflection ancestry.

The scene has one conifer (28,640 triangles), one white point light, a white partially transparent Fourier smoke field inside an invisible rectangular support, and a matte ground receiver for the shadow and scattered volume illumination. The dark backdrop contributes no illumination. There is no bloom filter or added ambient lamp. The smoke is a participating field, not a translucent mesh painted over the tree.

Open [the saved image gallery](output/review.html), [the complete scene](output/conifer_smoke.png), or [the diagnostic contact sheet](output/contact_sheet.png). The replayable response is `output/conifer.sheet`. The native executable has separate `bake` and `play` entry points. Playback can run without constructing a tree, light, volume, or BVH.

## What was found in the existing engine

In `native/regime_scene_native.cpp`, `trace_primary` immediately returns registered/base radiance for `MaterialKind::Diffuse`. The recursive and sealed secondary paths also end after adding that radiance. The existing compiled `TransportField` retains diffuse energy separately, and `base_radiance` reads it. This is already the desired separation of identity-bearing geometry from registered outgoing energy.

The new native regression supplies a nonzero registered diffuse energy value, checks that it appears in the output, and verifies zero secondary geometry continuations for a primary diffuse hit. It checks both existing secondary implementations as well.

The termination contract is the material kind, rather than a measured entropy or microstructure-noise threshold. `Glossy` and `Metal` retain their directional continuation term, including at high configured roughness. This experiment does not silently reclassify those materials. Here “identity” means distinct image-bearing geometric continuation, not a numerical estimate of electromagnetic phase coherence or thermodynamic entropy. The tree keeps its own spatial identity as an emitting/scattering source.

## Registered operations

1. **The physical source owns a pinhole projector.** A 256 × 256 angular chart captures first-hit object depth. Ninety-six depth intervals carry cumulative smoke optical depth along those same characteristics. Every receiver reuses this source registration. Mixed source tiles fall back to exact BVH visibility during capture; no fallback is allowed during replay.
2. **Diffuse hits become their own source records.** A finite set of anchor points captures only the conifer, using six pinhole faces per anchor. Misses contribute zero; nothing behind the object is captured. A ray stops at the first diffuse hit. Its registered outgoing radiance contributes to finite angular moments. These moments carry energy and an outgoing angular distribution, not an image of the illumination's ancestry.
3. **The cloud also illuminates surfaces.** An 8³ volume partition retains 402 illuminated source cells in this scene. Fourier cell masses are integrated exactly; incident illumination, phase, visibility and attenuation remain sampled. These are registrations of energy from the one physical lamp. Each tree triangle receives front/back outgoing diffuse radiance at its centroid; an 81² ground chart receives the same volume contribution. A finite equal-volume spherical-cell kernel avoids singular point-source hotspots. The cell-shape replacement and within-cell illumination approximation still require refinement. See [only the volume light landing on surfaces](output/volume_light_on_surfaces.png).
4. **The cloud registers an output response for each observer characteristic.** It carries transmission `T` and scattered radiance `S`, truncated at the actual first opaque surface. The source chart is shared by these characteristics. The registered diffuse field supplies the tree-to-medium contribution. This first optical model includes direct illumination and one registered tree contribution followed by one medium-scattering event, plus point-light → medium → diffuse surface → camera; it does not recursively simulate all medium/surface feedback.
5. **The saved receiver sheet composes its own endpoint values.** It evaluates `L = T × B + S`, where `B` is registered terminal radiance. Geometry and density are absent from the independent playback process. A replacement terminal radiance can use the same `T,S` without recapturing, provided the geometry, light, medium and angular support are unchanged. This is not permission to move an opaque object or change smoke density while retaining an old sheet.

The saved diffuse grid is a first finite sampling choice, not a claim that a regular grid is the minimal representation. It uses 24³ anchor points, 32² samples per pinhole face and ten RGB angular moments (one trace is redundant). It occupies about 3.48 MiB, including the mean-radiance working array. The coarser 8³/8² capture was too inaccurate. `output/grid_study.json` records the initial resolution screen. The final independent mean-radiance probe uses a 64²-per-face direct capture.

The source registration occupies about 24.75 MiB. The 1280 × 720 receiver sheet uses 28.125 MiB: eight floats per pixel for `T`, `S`, terminal radiance and depth. These allocations are explicit. This implementation has not yet minimized empty source columns, sheet payloads or repeated angular coefficients.

## Why addition alone is insufficient

For a bounded segment, retain the affine operator

```
R(B) = T B + S.
```

For a front segment A and a back segment B, the retained composition is

```
T_AB = T_A T_B
S_AB = S_A + T_A S_B.
```

The composition is associative, so a fixed chain can be compiled and shared without replaying its internal integrations. It is order-sensitive. Positive added scattering cannot by itself darken a background. Extinction acts through `T`; object blockage also reduces source illumination before `S` is registered. The `additive_only_wrong` image deliberately drops the camera-path `T` term to make that distinction inspectable.

This follows the transmittance and source terms of the volume transfer equation; transmittance over consecutive segments multiplies. See [PBRT: Transmittance](https://pbr-book.org/4ed/Volume_Scattering/Transmittance) and [The Equation of Transfer](https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer).

## Fourier and angular representation

The rectangular support is `[-4, 4] × [0.05, 7.8] × [-2.5, 5]` in world x/y/z. With normalized coordinates x/y/z within that support, extinction density is

```
0.62 sin²(πx) sin²(πy) sin²(πz)
     × [0.68 + 0.32 cos(2π(2x+y+z) + 0.7)]
     × [0.65 + 0.35 cos(2π(x−2y+2z) − 0.5)].
```

Every factor is nonnegative. The field deforms into a cloud while tapering to zero on the invisible support boundary. The scattering albedo is 0.96. The compact product expands to 207 nonzero complex Fourier modes; the conjugate pairs represent a real field.

During a characteristic march, five complex oscillators advance these factors. Trigonometric initialization occurs once per characteristic, followed by phase multiplication. An independent expanded-mode oracle integrates each density mode over a straight chord with

```
length × sinc((k·direction) length / 2)
       × Re[coefficient × exp(i k·midpoint)].
```

Tests compare the factorized density, oscillator march, expanded Fourier field and analytic chord integral. The source/scattering integral still contains visibility and attenuation; the analytic density oracle alone does not make that entire integral analytic.

The angular phase is a normalized, positive degree-two law:

```
p(c) = [1 + 3 g c + 5 g² (3 c² − 1)/2] / (4π),   g = 0.5.
```

The minimum is positive for this configured value. Both primary illumination and the registered tree contribution use this same law. The diffuse angular moments contract with the observer direction; their incoming geometric paths are not continued. This is a controlled finite angular model, not a measured Mie phase function. The Fourier frequencies above are spatial frequencies, not optical wavelengths; this white-light demo does not claim a wavelength-resolved smoke simulation.

A signed Fourier coefficient or signed surface slope is not used as an energy switch between reflection and transmission. The unit test checks that opposite slope signs can have the same Fresnel weight and can both transmit. Signed boundary orientation can label incoming and outgoing channels, but its sign alone does not determine their energy fractions. The invisible smoke support is not introduced as a reflective dielectric interface.

## Deduplication and invalidation

`CaptureRegistry` owns immutable sheets indexed by fingerprints of geometry, medium, physical light, material and camera/integration settings. Repeating a key returns the same object without invoking the capture function. The final bake deliberately repeats its request and asserts one build and one reuse. A changed medium key creates a distinct entry. The complete key and pixel checksum are serialized.

The independent `play` command verifies integrity, composes/tone-maps 120 frames and verifies that the sheet is unchanged. Its two terminal-radiance choices exercise actual composition rather than timing a static file copy. It is a static-response replay benchmark, not an animated-density or moving-camera benchmark. These fingerprints cover this demo's implemented inputs; they are not an automatic dependency graph for arbitrary future engine features.

## Measurements and acceptance scope

Measured on the host selected by `m4host`, using six M4 CPU threads:

| Measurement | Result |
|---|---:|
| Source projector bake | 0.237 s |
| Diffuse source registration | 17.18 s |
| Volume light registration on surfaces | 7.91 s |
| 1280 × 720 camera capture, 192 samples per chord | 48.78 s |
| First still through camera capture, excluding diagnostics | 74.13 s |
| Complete bake including reference diagnostics | 132.33 s |
| Independent composition per image, average over 120 repetitions | 7.88 ms |
| Geometry/density queries during replay | 0 / 0 |
| Projector versus direct-source diagnostic RMS | 0.0386 of an 8-bit level |
| Projector/direct 99th percentile / maximum | 0 / 3 levels |
| Camera quadrature 192 versus 384 RMS | 0.319 levels |
| Camera quadrature 99th percentile / maximum | 2 / 10 levels |
| Isolated diffuse mean-radiance probe relative L2 error | 9.09% |
| Volume-to-ground 8³ versus 12³ registration difference, 20 positions | 10.90% |
| Fourier expanded/factorized density maximum error | 3.33 × 10⁻¹⁶ |
| Analytic chord versus fine quadrature maximum error | 6.60 × 10⁻¹⁴ |

The main projector/direct comparison shares the same registered diffuse field and the same camera quadrature. It isolates source-visibility/transmittance reuse; **it is not a full-scene physical-accuracy claim**. The camera-resolution comparison and the diffuse capture probe expose separate remaining approximations. The diffuse probe measures mean radiance at 16 positions, not a complete angular-field error bound. The source chart's unmixed-tile shortcut is sampled, not a conservative proof that no subpixel needle was missed.

The original source/replay gates pass: source RMS below 0.5 levels with 99th percentile at most 1 and maximum at most 8, diffuse mean probe within 10%, byte-identical saved/replayed output, and zero replay transport work. The new volume-to-surface path is provisional: its 10.90% ground refinement difference does not pass a 10% convergence target, and does not bound tree-triangle or ground-interpolation error. These empirical gates are not universal certificates. Native tests also cover diffuse termination with retained energy, affine-response identity/associativity/order, extinction darkening, angular positivity/contraction, real registration reuse, changed-key invalidation and serialization.

## Still-rendering cost and the intended engine wrapper

The intended use is still-image ambience, including scattered light reaching actual objects. Replaying the same fixed camera quickly is only a storage/composition check. It does not establish faster first-still rendering. The present first still costs about 74.13 seconds including its registrations. Ground hits outside the ±24 chart recompute volume illumination directly; this deliberately preserves lighting but makes the camera pass expensive. The source-reference comparison shares the surface and diffuse registrations, so it is not an independent end-to-end baseline. No massive whole-scene speedup has been established.

An engine-owned wrapper can expose supported transmission/scattered-light characteristics and registered outgoing surface energy while retaining dependency keys for geometry, illumination and medium changes. The implemented affine composition and immutable registration are small mechanisms. Efficient coverage, adaptive capture and invalidation remain the substantial work. For a first still, success requires the complete capture-plus-render cost to beat direct evaluation at matched error. This prototype establishes the representation and bounded light paths; the surface pass is not promoted as a production accelerator.

## Translation to larger scenes

The source projector and receiver sheets amortize work over the scene they already capture. Adding trees increases geometry and visibility discovery at capture time; it does not require rerunning those queries for each replay pixel. The registered diffuse response can be shared after its incident distribution is fixed. Repeated tree geometry alone does not make its lighting and visibility identical.

For arbitrary new viewpoints, a single 2-D pinhole image does not contain the missing position/direction information. In the included test, planar reprojection covers 85.25% of the new view, but still differs by 6.69 levels RMS on that covered region. Some geometry and volume chords are different. Missing support is shown in purple. The full two-plane description is a four-dimensional ray field; sparse/adaptive charts can sample it, but one image is only a slice. See [Levoy and Hanrahan, Light Field Rendering](https://www.graphics.stanford.edu/papers/light/).

The scalable next step is therefore finite supported projector relations, shared outgoing energy records, and dependency-aware recapture. This demo establishes those operations in one scene. It does not claim that thousands of mutually occluding trees, arbitrary camera translation or all scattering feedback have already been validated. Cyclic scene feedback still needs a retained closure or a documented truncation; merely registering an edge does not numerically account for every possible repeated interaction. The present standalone model explicitly limits its path family.

## Reproduce

From the authoritative MacBook repository:

```sh
experiments/photonic_transport_field/conifer_projectors/run_m4.sh
.venv-jpeg/bin/python experiments/photonic_transport_field/conifer_projectors/summarize.py
```

The wrapper compiles and tests through `m4build`, bakes `/tmp/conifer_projectors/`, runs independent replay, and copies results back immediately using `m4host`. It installs no software. `conifer_demo.cpp` includes the native BFFT scene/intersection engine and links its existing native dependencies. The new projector/volume implementation is in `projector_volume.hpp`; the home scene and its renderer are unchanged.

The separate initial diffuse resolution screen is reproducible with the compiled executable's `grid-study OUTPUT.json` mode. It compares 8/12/16/24 point grids and different angular capture resolutions against a higher-resolution direct mean-radiance capture. The initial screen predates the angular-moment extension; its mean component remains the same measured quantity.
