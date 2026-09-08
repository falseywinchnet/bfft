# Why classification expands

Measured on the M4 Mini, aperture-canyon, 800×600, terminal error 1,
shared child laws, child boundaries enabled, BVH disabled, default adaptive
topology and shared boundary filter. This is an attribution and exact-input
census, not a speed benchmark or a changed renderer.

## Call attribution

| Caller | Child classification calls | Share |
|---|---:|---:|
| Spectral probes while shading boundary regions | 1,078,562 | 54.7% |
| Boundary coverage and centroid validation | 400,880 | 20.3% |
| Spectral probes during the initial adaptive render | 301,392 | 15.3% |
| Initial adaptive ownership classification | 82,233 | 4.2% |
| Edge discovery and refinement | 77,139 | 3.9% |
| Shared boundary filter construction | 32,793 | 1.7% |
| **Total** | **1,972,999** | **100%** |

Spectral probing accounts for 1,379,954 child calls (69.94%). The child call
count is higher than the number of spectral probes because a classifier may
continue through multiple dielectric children along its path.

The independent raw census exactly reproduces the prior shared-law frame's
1,972,999 classification calls. There are 1,894,858 distinct full input keys:
78,141 repeated requests, or 3.96%. Keys compare the actual IEEE bit patterns
of incident origin/direction, hit position/normal/front, primitive, channel,
and spectral coordinate. Scene and numerical budgets are fixed in this run.
Hash collisions are resolved by full key equality. No coordinate rounding,
approximate matching, or cross-wavelength equivalence is assumed.

Within-stage duplicate counts sum to 49,805; the remaining 28,336 repeats
are overlaps between stages. A perfect cache of these full inputs could avoid
3.96% of classification calls, not 70% of frame time. The census deliberately
allocates a large transient map and uses a mutex, so its timings must not be
used for performance conclusions. The production renderer has no such map.

## The larger repetition

`trace_primary` first probes the prism at spectral coordinates 0, 1, and 2.
If any of those path signatures disagree, it integrates all three bands.
For each band, `integrate_spectral_band`:

1. Starts with eight wavelength intervals and nine endpoint classifications.
2. Classifies each midpoint and recursively divides intervals with differing
   signatures, to a depth limit of ten.
3. Merges adjacent leaves whose topology labels agree.
4. Evaluates radiance once per merged region.

This frame shades 11,234 prism camera samples. Of these, 4,279 trigger mixed
spectral integration: 1,319 in the initial render and 2,960 in boundary shading.
They generate 12,837 bands, 522,353 subdivision leaves, and only 78,186 final
regions. The tree performs 1,057,543 spectral probes; the initial three tests
per prism sample add 33,702, giving **1,091,245 total spectral probes**.

The count follows directly from a full binary subdivision forest: for each
band, probes = 2 × leaves + 1. Across the frame:

```
2 × 522353 + 12837 = 1057543
1057543 + 3 × 11234 = 1091245
```

There are about 247 subdivision probes and 18.3 final spectral regions per
mixed camera sample, across its three bands. Approximately 85% of subdivision
leaves disappear into adjacent regions before shading. That does not prove
those discovery probes were unnecessary: the current algorithm uses them to
locate discontinuities. It identifies where the implementation spends work
rediscovering boundaries instead of retaining their geometric relations.

Small literal duplication is visible in the code: the three initial center
queries are repeated among the integration seed endpoints, and the two shared
RGB-band endpoints are evaluated twice. Reusing these would save five spectral
probes per mixed sample (21,395 here), only 1.96% of total spectral probes.
Additional exact repeats occur during boundary centroid validation and across
passes. They are secondary to the subdivision workload.

## Spatial coverage still matters

The frame has 16,692 boundary pixels and 170,198 boundary topology sample
requests. Shared filtering already reduces 62,336 overlapping filter requests
to 22,080 evaluated samples. Only 683 boundary pixels use the 16×16 fallback.
The initial ownership pass and boundary discovery are not the million-call
source. Labels can contain up to three channel paths, with up to eight path
steps, so a label request and a child classification are different units.

## Engineering implication

The strongest next target is the wavelength event structure inside
`integrate_spectral_band`: retain where a reflected or transmitted path changes
terminal ownership as a relation of camera position/direction and wavelength,
then integrate the resulting intervals. Today that structure is discovered
independently for every sample, including nearby samples looking through the
same prism. The shared child response law does not yet share those path-event
boundaries or terminal visibility relations.

Different directions and wavelengths can reach different objects. Thus one
constant label per object, or accepting a whole band merely because a few
samples agree, would change the problem. A useful replacement needs explicit
event validity domains and matched output checks, including small occluders,
reflection/transmission branch changes, and grazing incidence. The census
measures the opportunity; it does not establish that neighboring event curves
can be reused safely or promise a particular speedup.

No production behavior was changed in this investigation.

## Reproduction and artifacts

Run `experiments/photonic_transport_field/run_classification_audit.sh`.
It generates a diagnostic translation unit from the authoritative source,
compiles and executes on the selected Mini, and immediately copies the JSON
into `classification_audit_m4/`. Replacement anchors are checked for uniqueness
so a changed source cannot silently instrument the wrong location.

Stage indices 0–3 are adaptive render, edge discovery, shared filter, and
boundary shading/coverage. Stages 4–7 are spectral child calls within those
same phases. The `spectral` records count probe invocations, integrated bands,
merged regions, leaves, and mixed rays separately.
