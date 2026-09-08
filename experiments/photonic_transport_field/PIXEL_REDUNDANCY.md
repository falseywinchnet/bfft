# One pixel: many evaluations of few source relationships

## Architectural correction from the user

Parent transport must terminate at a surface; interiors belong to private
subsystems. Reflected output is generated at the surface, and a child exposes
only its boundary response. The existing `CHILD_VOLUME_RESPONSE.md` specifies
this contract explicitly. The native source query audited below instead walks
across dielectric surfaces in the parent's scene, and the native optical
packet tracer continues refracted packets through that same scene. Its
"sealed" label refers to residual feedback termination, not child-domain
isolation. The jelly chord shortcut is also a local special case, not general
parent/child response integration.

Therefore these counts diagnose a native path that does not enforce the
intended subsystem boundary. They must not be presented as work demanded by
the intended architecture. Wiring camera and source queries to retained child
boundary responses takes precedence over further optimizing this flattened
traversal. The native retained block kernel is already used for a transport
field; that does not by itself make these optical queries obey child sealing.

The million-count work is a multiplication of numerical domains, not a count
of independent scene objects or physical light exchanges. A fresh census of
the two worst aperture-canyon pixels identifies both the shared relationships
and why an exact-value cache cannot remove this multiplication.

The stress scene contains 46 geometric primitives and three area emitters.
Frames, caps, and fins contribute separate primitives. This count alone is not
a bound on the complexity of visibility, reflection, or a continuous integral.

## Pixel (275,324)

The standalone probe reconstructs the established 16x16 coverage lattice,
groups identical optical labels, validates each group's centroid, uses the
same first-sample fallback, and accumulates the same coverage weights.

| Evaluation layer | Count |
|---|---:|
| Coverage groups | 73 |
| Wavelength regions across their RGB bands | 4,539 |
| Processed optical packets | 60,921 |
| Surface-lighting requests | 30,430 |
| Glossy/metal lighting calls | 20,538 |
| Glossy source intervals visited | 826,666 |
| Glossy source quadrature ordinates | 1,517,827 |
| Glossy ordinates with zero incoming RGB | 541,595 |

All source work, including diffuse lighting, is 850,284 source intervals and
1,562,888 ordinates. Intervals and ordinates are different counters: some source
ordinates are rejected by geometric cosine tests before invoking the kernel.

The **826,666 glossy intervals contain only 180 distinct ordered source words
across 31 receiver-emitter pairs**. A word is the receiver ID, emitter ID, and
ordered primitive IDs from the existing finite segment-program discovery.
It deliberately omits coordinates, angles, and numerical weights. Equal words
identify the same sequence of material operations, not equal radiance values.
This is the renderer's finite visibility description, not a new certificate of
every continuous visibility event.

Three repeatedly encountered relationships account for 58.7% of glossy source
interval visits:

| Relationship | Interval visits |
|---|---:|
| Rough copper sphere, large soft light, glass sheet between them | 215,786 |
| Rough copper sphere, large soft light, no intervening surface | 169,898 |
| Glossy sphere, large soft light, no intervening surface | 99,474 |

There are also **47,748 adjacent glossy interval boundaries (5.78% of interval
visits) across which the ordered source word does not change**, within the same
source row and receiver-emitter integration. Those cuts do not represent new
ordered visibility relationships. Some subdivision may still be useful for
integrating a varying smooth kernel: deleting cuts and retaining the same
fixed-order quadrature is not automatically an accuracy-preserving change.

## What is and is not redundant

All 30,430 surface requests have distinct exact numerical keys, including
position and, for directional materials, view direction. Even omitting view
leaves 30,430 distinct position/normal keys. Each of the 20,538 specular calls
also has distinct receiver geometry.

The probe tested an unlimited exact-key cache scoped to a coverage group and
an unlimited cache scoped to the whole pixel. Both find zero reuse. The existing
32-entry cache also has zero hits for this pixel. Larger capacity does not
address this case; quantizing coordinates without validating response error
would change the question.

The repeated work is evaluation and reconstruction of the same parameterized
relationships at different inputs. Each coverage group subdivides wavelength,
each wavelength ordinate evaluates the optical expression, and each shaded
surface initiates area-light integration. In particular, later changes in an
optical path cause earlier lighting terms to be evaluated again at new
wavelength ordinates. The current scalar-query interface does not retain a
source-response function over that incoming parameter domain.

This census does not establish that all evaluations of one source word can be
replaced with one constant, or that all but 180 evaluations can be deleted.
Fresnel factors, source distances, normals, and the reflection lobe still vary.
It establishes a specific target for functional reuse, and rules out cache
eviction or exact repeated return states as the main explanation here.

## Second pixel and verification

Pixel (268,323) independently exhibits the same pattern: 65 coverage groups,
4,102 wavelength regions, 25,499 distinct surface requests, 692,387 glossy
intervals, and 1,381,283 glossy ordinates. Its source intervals use 166 distinct
ordered glossy words across 29 receiver-emitter pairs; 35,195 adjacent glossy
interval boundaries leave the word unchanged.

Twenty-four optimized runs cover both pixels, four modes, and three rotated
repeats. All produce exactly identical linear RGB values across modes, not just
identical display bytes. The hooks also check that a repeated full key never
produces inconsistent RGB. Controlled checks exercise initialized entries,
cache scope, and rejection of nearby but unequal coordinates. Eight additional
ASan/UBSan runs pass. The initial exploratory receipt predates a diagnostic
context-pointer lifetime fix; the final receipt and sanitizer run include it.

Without active instrumentation, the two pixels take median 328.55 ms and
276.75 ms respectively on the M4 CPU in this isolated serial probe. These are
not complete-frame timings. The map-based census adds overhead; its timings
must not be presented as the baseline or an accelerator.

## Artifacts and reproduction

`native/pixel_redundancy_probe.cpp` and `native/pixel_redundancy_audit.hpp`
implement the census and exact-cache control. Hooks in the native scene file
are excluded unless `PHOTONIC_PIXEL_REUSE_AUDIT` is defined. The standard
renderer receives no new runtime audit or caching behavior.

The final raw receipt, summary, and sanitizer artifacts are in
`pixel_redundancy_m4/`. Run the selected-M4 compile, census, and immediate copy:

```sh
sh experiments/photonic_transport_field/run_pixel_redundancy.sh
```

No performance optimization or source-interval coalescing is promoted by this
census. The next substantive test is retaining the dominant source responses
as functions over their actual incoming supports, with error checked against
independent finer integration. Merely reordering the existing nested samples
has already failed the separate own-domain prototype.
