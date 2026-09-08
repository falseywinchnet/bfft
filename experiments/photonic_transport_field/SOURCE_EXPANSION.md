# Why 794 pixels requested ten million source samples

The 800×600 aperture-canyon audit found a multiplicative camera cost, concentrated
at the upper edge of the prism. These are source-integration ordinates, not ten
million independent primary camera rays. The previous boundary audit counted
10,288,940 source ordinates, including diffuse irradiance work. The new per-pixel
audit isolates the glossy source kernel and the optical work that invokes it.

| Work in the 794 fallback pixels | Count |
|---|---:|
| Coverage-bucket radiance evaluations | 4,087 |
| Primary evaluations with mixed prism spectra | 1,386 |
| Spectral topology probes | 565,897 |
| Merged wavelength regions | 44,264 |
| Spectral radiance evaluations, including unmixed spectra | 46,502 |
| Optical packets processed after the existing cutoff | 581,657 |
| Packets shading a non-dielectric, non-mirror surface | 278,524 |
| Glossy/metal area-light integrations | 149,330 |
| Glossy source quadrature ordinates | 9,686,729 |
| Glossy ordinates whose incoming RGB is exactly zero | 3,686,325 (38.06%) |

The 100-call difference from the previous 149,230 specular-call receipt comes
from demand-atlas construction: the new callback audit also sees construction
performed on behalf of the triggering boundary worker. Attribution of shared
atlas construction to a particular pixel can depend on worker scheduling.
Aggregate optical work and the catastrophic prism pixels identify the same
cost concentration.

Pixel **(275, 324)** alone asks for 1,517,827 glossy source ordinates:

- Its 16×16 coverage lattice finds 73 distinct optical labels.
- Each label triggers three channel-band integrations: 219 bands total.
- Those bands become 4,539 spectral regions.
- Their radiance evaluations process 60,921 optical packets and request 20,538
  glossy source integrals.
- 541,595 of the final source ordinates carry exactly zero incoming light.

The two worst pixels account for 29.93% of all fallback glossy ordinates. The
worst eight account for 56.89%; the worst twenty account for 79.18%. This is a
small geometric region causing enormous nested work, rather than an evenly
expensive 800×600 image.

## What the geometric words reveal

`native/source_expansion_probe.cpp` reconstructs the exact 73 coverage buckets
of that pixel, including the renderer's centroid validation and first-sample
fallback. It then reads the eight-event spectral path classifier on a lattice
of 8,192 steps per channel-band width. Its output uses ordinary primitive IDs
for the continued path and `1000 + primitive` for a dielectric's first reflected
hit. The initial diagnostic receipt caps displayed transitions at 110 per
bucket; its transition count includes every change on the probe lattice. The
current runner emits all changes.

One common word begins:

```
prism_side_2 -> prism_side_0 -> aperture_fin_4 -> rough_metal_sphere
```

It then continues through another fin, the glossy sphere, the glass sheet,
frames, mirrors, and other objects. Many wavelength transitions occur in these
later events. The classifier is a finite eight-event geometric word; it is not
a certificate of constant radiance inside a region.

`integrate_spectral_band` subdivides each of eight seed intervals until its
endpoint/midpoint signatures agree, or until depth ten. It merges equal adjacent
leaf labels and evaluates the **entire radiance expression** at each region's
midpoint. A late change in the word therefore repeats the expensive lighting
of every earlier surface along the path. The 40,162 depth-limited leaves are
mostly transition localization, not evidence by themselves of numerical noise.
The smallest admitted region is 1/8,192 of a channel-band width.

The packet cutoff also measures a local continuation weight. Primary Fresnel
factors, wavelength-region width, and pixel-coverage weight are multiplied
outside that continuation. An extremely small contribution to the final pixel
can therefore still request a full local lighting integral. Raising the cutoff
without bounding the **sum** of omitted pixel contributions would not be a
justified accuracy-preserving change.

A separate existing correctness issue surfaced during the path audit: when
`refract` reports total internal reflection, the camera dielectric branches
retain the Schlick reflection fraction instead of transferring the full
untransmitted energy into reflection. The beam compiler likewise retains its
transmission weight when redirecting a TIR ray into reflection. This is not
changed or used as an optimization here. It needs a dedicated energy-conservation
test and correction before treating the finite optical operator as a physical
reference for a new spectral integrator.

## Two accepted changes

Both are enabled by default in the integrated native engine. They leave geometry,
coverage labels, spectral partitions, optical cutoff, quadrature nodes, and
nonzero illumination calculations intact.

**Annihilate a proven zero source suffix before evaluating it.** A finite source
program lists crossed surfaces in receiver-to-source order. Discovery now records
its source-most opaque operation. At a quadrature ordinate, the evaluator checks
the same finite intersection predicates and limits that the ordinary evaluator
would encounter from that operation to the source. If all are valid, the ordinary
evaluator must return zero; the preceding Fresnel and absorption arithmetic cannot
change that result. If membership or an ignore changes, the ordinary evaluator,
including its existing fallback, runs unchanged. Discovery is not truncated at an
opaque surface, and no unverified silhouette classification is substituted.

The glossy kernel separately returns immediately when incoming RGB is exactly
zero. This avoids half-vector normalization, Fresnel evaluation, and the powered
reflection lobe. It does not use a brightness threshold. In the audited aperture
frame, the ordinary render counters record 5,741,622 such lobe elisions across
camera work. These counters do not include the separate demand-atlas construction
statistics. Quadrature counters continue to count the original ordinates, so a
lower expensive-lobe count is not misreported as lower integration resolution.

This is equivalence to the current finite segment-radiance operator, including
its existing opaque return and fallback behavior. It is not a new physical model
of scattering around occluders.

**Compare source cones without reconstructing angles.** The angular-disk broad
phase previously calculated two `asin` values and one `acos` for each candidate.
For cone half-angles α and β and axis separation θ, the same real-valued decision is

\[
\theta\leq\alpha+\beta
\quad\Longleftrightarrow\quad
\cos\theta\geq
\sqrt{1-s_\alpha^2}\sqrt{1-s_\beta^2}-s_\alpha s_\beta,
\qquad s_\alpha=R_\alpha/d_\alpha.
\]

The implementation evaluates the square-root radicand as `(1-s)*(1+s)` to
avoid cancellation near a cone horizon. The source direction, radius, sine, and
cosine are prepared once per receiver–emitter pair. The candidate test uses dot products and square roots.
The established angle implementation decides numerical ties within 128 machine
epsilons of the cosine comparison. Existing horizon and finite-distance gates
are preserved. The old angle path remains available as an executable oracle.

## Paired M4 measurements

`source_expansion_m4/source_expansion_benchmark.json` contains 108 frames:
four scenes, both linear/BVH intersection policies, three standard-scene camera
poses, three modes, and three repeats with rotated mode order. Every complete
800×600 RGB frame is byte-identical across modes. The field is compiled once per
scene and shared immutably for camera timing; each render constructs its own
viewer field so one mode cannot inherit another mode's demand samples.

Mode 0 uses the original source evaluator and angles. Disabled modes also skip
the optimization’s preparation work; they do not pay for unused certificates or
a source cone. Mode 1 enables only zero elision. Mode 2 also enables algebraic cones. These are medians, not a frame-time
guarantee or a comparison against older, differently loaded runs.

| Scene/view | Intersections | Full frame, original → accepted | Boundary, original → accepted |
|---|---|---:|---:|
| Standard, default | Linear | 189.38 → 188.07 ms | 68.21 → 68.34 ms |
| Standard, default | BVH | 259.90 → 256.68 ms | 92.09 → 90.58 ms |
| Aperture canyon | Linear | 1,105.58 → 1,078.84 ms | 768.01 → 747.07 ms |
| Aperture canyon | BVH | 1,347.81 → 1,318.43 ms | 931.15 → 912.65 ms |
| Mirror relay | Linear | 788.99 → 758.95 ms | 496.73 → 475.58 ms |
| Occlusion garden | BVH | 477.56 → 467.04 ms | 213.23 → 206.55 ms |

The other standard-scene camera poses improve by 0.7–1.2% in these paired runs.
Across tested cases the full-frame time reduction is 0.7–3.8%. Removing millions of
powered lobes is therefore useful but not the main solution: geometry queries,
source partition construction, and repeated nonzero lighting still dominate.
The 33.33 ms budget for 30 fps at 800×600 remains unmet.

Fresh field compilation was timed separately with three alternating repeats per
mode. This benchmark uses the scene builder's BVH setting for compilation,
regardless of the camera's independently tested policy.

| Scene | Original field compilation | Accepted |
|---|---:|---:|
| Standard | 503.69 ms | 505.21 ms |
| Aperture canyon | 2,313.31 ms | 2,293.03 ms |
| Mirror relay | 2,370.49 ms | 2,355.39 ms |
| Occlusion garden | 1,904.88 ms | 1,861.97 ms |

Compilation remains essentially unchanged in the standard scene and improves
by about 0.6–2.3% in the other three scenes in this run.

Validation includes 185,700 source-radiance/affine-response comparisons, 45,266
successful zero certificates, and 67,479 scene/grazing cone comparisons. The
source suite also compares freshly compiled direct, bounce, radiance, coupling,
and coarse/refined direct-atlas values exactly for all four scenes. It passes
under both the optimized build and AddressSanitizer/UndefinedBehaviorSanitizer.
Existing scene, retained-update, camera-gather, and boundary-reuse native suites
also pass. Sanitizer compilation retains the pre-existing packet-loop
vectorization warnings; it reports no sanitizer failures.

## Rejected spectral shortcut and the next representation

A prototype made the spectral classifier follow actual rough-surface coverage
and continuation weights, while retaining the same packet cutoff. It also used
three-point Gaussian integration inside each resulting wavelength region.
This did not solve the duplication. Introducing cutoff crossings actually
increased fallback region count from 44,264 to 56,072. Combined with the extra
quadrature, glossy ordinates rose from 9.69 million to 32.99 million and the
single diagnostic frame slowed substantially. No quality claim was made from
that run; finer spectral-reference validation was not pursued after it failed
the work-reduction screen. The prototype is archived as a patch and is absent
from the production path.

The more useful reordering follows the additive transport expression. For
coverage bucket b and contributing optical event j, write its contribution as

\[
I_p=\sum_b w_b\sum_j
\int_{\Lambda_j}T_{bj}(\lambda)
L_j(x_{bj}(\lambda),\omega_{bj}(\lambda))\,d\lambda.
\]

The current implementation effectively forms a combined path partition, then
re-evaluates the sum of all contributions in each combined region. A boundary
in a distant descendant can multiply evaluations of an earlier expensive term.
Linearity allows each term to be integrated over the geometry and medium events
that affect **that term**, then accumulated into the pixel. It does not allow a
changing earlier response to be replaced with a constant or an unvalidated
cache entry.

The next implementation should retain, for each contributing event, its
wavelength support, geometric position/view mapping, transfer weight, and
source-response representation. Splitting a descendant's support would then
split its contribution rather than duplicate its ancestors' source integration.
Shared terms must remain functions of wavelength, position, and direction; only
proven invariants may be reused as constants. A finite error test against a finer
independent spectral reference must account for smooth variation as well as
visibility transitions. Any later importance pruning must charge the total
omitted contribution against a pixel-level bound, including coverage, wavelength
measure, and Fresnel factors.

This reordering is a proposed engine change, not an implemented speedup or a
30 fps claim. The present work establishes its target with per-pixel receipts
and removes two exact sources of wasted arithmetic without changing the image.

## Reproduce

From the repository root:

```sh
experiments/photonic_transport_field/run_source_expansion.sh
```

Use `--source-zero-elision off --source-cones angles` for the original evaluator.
`--expansion-audit PATH` records the nested work for every reconstructed boundary
pixel. It adds instrumentation and output cost and should not be enabled for
frame-time comparisons. The geometry-only probe accepts optional pixel x/y
arguments and assumes the established 800×600 aperture-canyon view.
