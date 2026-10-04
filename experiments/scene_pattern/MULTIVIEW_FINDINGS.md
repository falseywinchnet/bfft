# Walking-collection evidence and conditional geometry

This path follows `CAPTURE_GEOMETRY.md`: independent capture origins, uncertain
revisit positions, no fixed station count, and no inherited flat/periodic atlas.
All results below are development results on the owner's existing collection.
They do not establish a correct fused scene or a successful panorama-first stage.

## Current run

The private `out/scene-evidence-v2/evidence.json` receipt contains all 120 inputs,
1,446 tested candidate pairs, 6,573 proposed tracks spanning at least three
captures, and 787 tracks with observation cycles. It retains 5,776 incompatible
track joins instead of merging distinct sites in the same capture. Pair results
under nominal cylindrical panorama calibration are:

| Pair hypothesis status | Count |
| --- | ---: |
| Translation supported by the conditional angular screen | 225 |
| Rotation compatible | 181 |
| Ambiguous geometry | 124 |
| Unresolved | 916 |

The evidence run took 183.956 seconds on the Mini, reusing prepared Oklab/Meyer
segmentation-derived descriptors. Feature preparation is not included in that
number. Pair support uses held-out physical reference sites and a 63-permutation
null comparison. It is conditional on descriptor candidate selection, and the
exploratory per-pair threshold does not control collection-wide false discovery.

## Conditional three-panorama hypothesis

The current pose receipt is `out/scene-evidence-v2/pose.json`; its point audit is
`pose-audited.json`. It proposes separate origins for IMG_1458, IMG_1459 and
IMG_1460. IMG_1458/IMG_1460 define the arbitrary unit baseline. The third camera
fits 31 training observations and 6 of 10 held-out target sites inside 1.2 degrees;
its median held-out angular error is 1.052 degrees. This is a conditional gate
pass, not a validated reconstruction or a calibrated probability of correctness.

Of its 106 depth proposals, 34 agree with all three fitted views. The other 72
remain tentative. Their absolute distances are unknown. Agreement at regional
centers does not certify exact physical surface identity, occlusion, a dense
surface, or permission to fuse all contributing pixels.

This hypothesis uses tracks discovered in the **all-capture graph**; ordinary
photographs can influence those track joins. It must not be described as a
successful panorama-only first stage. The eight panorama-only runs below do
not pass the third-camera check.

## Panorama-only projection screen

The fixed family screen uses all 378 panorama pairs and the same 2-D candidate
inventory. It rebuilds the tracks independently for each family and does not
inherit the all-capture graph's track joins. Results are in
`out/scene-calibration-v2/screen.json`.

| Projection | Nominal focal multiplier | Supported pairs | Tracks with cycles | Third-camera gate |
| --- | ---: | ---: | ---: | --- |
| Cylindrical | 0.8 | 100 | 67 | No accepted group |
| Cylindrical | 1.0 | 121 | 142 | No accepted group |
| Cylindrical | 1.25 | 90 | 82 | No accepted group |
| Cylindrical | 1.6 | 110 | 96 | No accepted group |
| Equirectangular | 0.8 | 106 | 68 | No accepted group |
| Equirectangular | 1.0 | 123 | 105 | No accepted group |
| Equirectangular | 1.25 | 112 | 92 | No accepted group |
| Equirectangular | 1.6 | 100 | 107 | No accepted group |

This does not show that the captures are unalignable or select a correct camera
family. It shows that this finite global family and current regional tracks do
not yet produce the requested reliable panorama-first reconstruction. Candidate
selection, spatially varying panorama stitching, and uncertainty in where a
region corresponds to a physical surface remain unresolved.

## Audits that changed the result

- A rotation/baseline initialization derived from saved pair models had seen
  some target holdout sites. Its apparent three-camera pass was a diagnostic
  failure. Initialization now refits after excluding every candidate touching
  a held-out target site.
- Holdout membership formerly depended on mutable global track IDs. The current
  pose split uses physical target-site IDs, so rebuilding a graph cannot silently
  change which target observations are used to fit versus check a camera.
- These corrections changed which experiments pass. Keep the earlier receipts
  as diagnostics; do not quote their passes as current validation.
- The current result is still selected using collection evidence and repeated
  development checks. A further independent audit is required before promotion.

## Implemented mechanisms and remaining work

Implemented: alternative regional matches, scale-deduplicated physical votes,
rotation/essential competition, independent origins, positive-depth proposals,
bridge-aware cycle support, conflicting-track retention, multi-panorama photo
affinities, unused-site pose checks, fixed projection-family comparison, and
point-level third-view auditing. Ten focused tests include exact synthetic
three-camera recovery and rejection when only held-out observations are poisoned.

The next representation must carry regional correspondence uncertainty rather
than silently identifying two independently segmented centroids as an exact
shared 3-D point. It also needs spatially varying panorama distortion, coherent
alternative-track selection, further camera growth, and visibility-aware
surface fusion. A larger warp alone would not establish those facts.

The private `/capture/` reviewer shows the actual source regions, pair evidence,
projection failures, and conditional point geometry. Green points have the
three-view agreement above; gray points remain tentative. No new fused outdoor
image is claimed by this run.


The later continuous-image transport, dense fusion and photo-attachment results
are recorded in `REGIONAL_TRANSPORT_FINDINGS.md`. The figures above retain the
earlier sparse-centroid experiment; they are not the latest reconstruction.
