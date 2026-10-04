# Two-stage walking-capture scene prototype

The input is a folder of photographs. Panorama classification uses oriented
aspect ratio greater than 2.4. No room-specific region, reference, mask or
repair script enters this path. Full originals remain untouched; metadata-free
2048-pixel panorama and 1000-pixel ordinary-image copies are private working data.

Every capture gets Oklab fast Meyer decomposition and the same five native
causal-density segmentations. Multiscale DAISY gradient-distribution descriptors
and cartoon color compare the resulting region centers. DAISY supplies the
patch descriptor only; it does not replace the segmentation with a keypoint
detector. This descriptor addition is an explicit new experimental component.
There is no trained recognition model, SIFT detector or supplied correspondence.

Stage one compares every panorama pair. Stage two compares each ordinary image
with six panorama and six photo retrieval candidates, plus two neighbors in
each direction in capture order. Retrieval ranks the best fifth of sparse
regional descriptor distances; capture order only proposes candidates.
Reciprocal, distinct-site ratio matching and spatially distributed projective
consensus decide whether an edge is admitted. A homography certifies the matched
region, not the entire panorama or a shared camera center.

A graph-selected panorama fixes the arbitrary scene coordinate gauge. Pairwise
similarities initialize a mesh graph, and all admitted region ties jointly
constrain the vertices. The mesh uses cells about 32 feature pixels across,
second-difference weight 0.7, edge-change weight 0.06 and position weight 0.025.
Six robust least-squares passes use a 5-pixel residual scale. Positive triangle
area checks shorten each proposed update; folds are never a valid alignment.
The panorama meshes are fixed exactly when individual-image meshes attach.
This is a 2-D chart with source layers, not calibrated geometry or metric depth.

The ordinary average includes all connected layers on their full valid support.
A separate information-weighted average uses distance to measured ties, graph
residual, color agreement and at most a factor of two local-detail preference.
Its sharper appearance is not registration validation, independent visibility
proof or super-resolution. Original layer contributions, disagreement, graph
residuals, disconnected inputs and both averages remain available. Linear
sampling is the initial registration preview; native CONV is an explicit
renderer option once geometry merits that expense.

This collection starts a new automatic development implementation. Its results
must not be called independent generalization after policy changes informed by
this run. No local visual failure may trigger a manual patch in the reported
collection result. Synthetic mechanism checks accompany the entire run.

## Rejected global-chart result

The owner rejected the flat and periodic chart outputs as geometrically wrong.
Read `CAPTURE_GEOMETRY.md` before further work. These runs fail to establish the
multi-position capture geometry and must not be promoted as scene reconstruction.
The dense-flow continuation was stopped. Reusable matching experiments remain
diagnostics; subsequent work must retain distinct capture origins, station
associations, translation, depth-dependent parallax and visibility.
