# One child law, many evaluations

The camera no longer adds a persistent record for each distinct child query.
The shared backend retains three child-law instances (prism, sheet, jelly), with
seven geometric boundary faces, independent of pixel count, direction samples,
or spectral samples. Position, direction, wavelength, and channel are arguments
of the law. All its geometric branches remain available without registering
individual evaluations.

This fixes the registration explosion. It does not imply that the renderer
performs one radiance calculation per object: evaluation work is reported
separately below.

## Implementation

`ConvexOpticalChild::response` defaults to shared-law evaluation. Its exact-state
hash cache is absent from shared instances, including the empty shard allocation.
Only the explicitly selected `--child-response exact` comparison backend can
insert query records. Existing geometry updates retain unchanged law instances
and replace an affected instance rather than append observer-dependent copies.

Full radiance evaluation preserves the previous unit cutoff `1e-12`, numerical
ceiling 128, all emitted ports, and source coefficients. Thin-sheet behavior is
also preserved. There is no quantization or averaging of query values.

A separate `classification_response` stops after the first two ports, because
those are the only ports used by the existing spectral and visible classifiers.
The later tail cannot change those earlier ports. Full radiance still evaluates
the complete response. The exact-state comparison backend continues to execute
its old full-response classification path.

`--child-boundaries on` now uses the shared backend by default. The broader child
optical route remains opt-in because its images differ from the legacy non-child
route, as documented in `NATIVE_CHILD_BOUNDARY.md`. To reproduce the old cache,
select `--child-response exact` explicitly.

## Matched M4 screen

Four scenes, 800 × 600, retained transport, linear intersection traversal,
terminal error 1, three repeats per backend with alternating order: 24 renders.
Scene/beam/field construction is outside the frame timer and stored in the JSON.
The first frame is listed separately; warm time is the median of two repeats.
The comparison is against the previous **child-boundary exact-state backend**.

| Scene | Exact-state first / warm | Shared-law first / warm |
|---|---:|---:|
| standard | 254.4 / 169.4 ms | 169.3 / 170.1 ms |
| aperture-canyon | 1460.4 / 1251.6 ms | 1223.4 / 1233.7 ms |
| mirror-relay | 1000.7 / 712.0 ms | 690.5 / 689.6 ms |
| occlusion-garden | 440.9 / 364.4 ms | 362.9 / 364.1 ms |

All 24 frame checks passed byte equality within each scene across both backends
and repeats. The numerical response probe also checks 448 complete responses and
448 classification prefixes with exact port positions, directions, and weights.
This is a matched-output comparison, unlike the earlier legacy-versus-child
optical experiment. The small warm-time differences are not confidence intervals
or strong speedup claims.

Aperture-canyon:

| Quantity | Previous exact-state backend | Shared-law backend |
|---|---:|---:|
| Child-law instances | 3 | 3 |
| Retained query answers | 1,381,556 | **0** |
| Query-answer payload, excluding hash overhead | 1,082,370,368 bytes | **0 bytes** |
| Child evaluations per frame | 2,200,875 | 2,200,875 |
| Classification evaluations per frame | 1,972,999 | 1,972,999 |
| Source quadrature evaluations | 19,572,874 | 19,572,874 |

The shared backend still stores the fixed child definitions and creates transient
response values. Zero query-cache bytes must not be described as zero renderer
memory usage.

The initial shared version computed the full response even for classification.
It performed 19,794,816 internal boundary steps per aperture frame. The final
prefix-only classification reduces this to 3,213,931 (83.8% fewer) with unchanged
images. The preliminary screen is saved as `full_response_classification.json`.
The warm exact-state backend avoids internal marches through its large cache;
the shared backend instead avoids retaining those individual answers.

## Verification and scope

- 448 complete response and 448 prefix comparisons against the exact backend.
- No query registrations and no cache allocation under varying spatial/spectral
  queries or concurrent observers.
- Unchanged geometry retains the same three law instances.
- Editing the sheet replaces precisely its law; the prism and jelly remain shared.
- Exact full-frame byte comparisons across 24 renders.
- Prior child-boundary invariants and the existing scene/update/camera/boundary/
  source regressions pass.
- AddressSanitizer and UndefinedBehaviorSanitizer pass the focused probe.

The remaining bottleneck is explicit: 89.65% of the child calls are classification
queries, and the unchanged source integration still costs 19.57 million quadrature
evaluations. This implementation does not yet replace those repeated evaluations
with a single object-wide radiance field construction and camera gather. It does
establish that observing the scene cannot grow the persistent registration set.

## Reproduce

```sh
experiments/photonic_transport_field/run_observer_registration.sh
```

This uses `m4build` and copies the screen and test log immediately into
`observer_registration_m4/`. The focused Make target is
`make -C experiments/photonic_transport_field/native observer-registration-test`.
Raw measurements and the compact table data are `observer_registration_screen.json`
and `summary.json` in that output directory. The old child-boundary diagnostic
still selects the exact backend explicitly so its archived experiment remains
reproducible.
