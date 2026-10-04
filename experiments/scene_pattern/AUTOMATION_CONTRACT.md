# Final-method contract

The user requires an automatic shared-scene fusion method. The final algorithm
must not be assembled or operated by visually hunting for an object, repairing
its region, and repeating that process elsewhere. A reusable optimizer inside a
manually directed experiment does not satisfy this requirement.

## Current evidence and its limits

The floor, structure, shade, joint, and laptop experiments are supervised
local diagnostics. Their transforms were estimated from image data, but their
regions, reference choices, protected areas, and some experiment/view choices
were manually supplied. They demonstrate component behavior and failure modes.
They do not establish an automatic scene solver or generalization to unseen
collections. Their output must be labeled accordingly.

The reusable numerical components include Oklab/Meyer/segmentation processing,
distortion-basis fitting, coordinate composition, missing-support handling,
regional acceptance checks, and continuation. General numerical tests do not
substitute for an annotation-free collection-level test.

## Required interface

A final run accepts photographs and one documented global configuration. It
must not require object names, hand-drawn regions, chosen reference photographs,
protected floor/lamp/laptop areas, manually selected repair views, supplied
correspondences, or parameters adjusted after inspecting that collection.

From the photographs it must derive candidate regions and their overlap,
visibility/support evidence, local and collection-wide correspondences, coordinate
gauges, applicable distortion models, and updates. The existing Oklab/Meyer
segmenting representation is the intended starting point for region discovery.
A raster support mask alone is not proof of shared visible scene content.

## Automatic iteration

The update schedule must use computed disagreement, geometric consistency,
support, and uncertainty. Candidate basis expansion, region subdivision or
merging, and stopping must follow the same documented policy across the entire
collection. There must be no object-specific branches or fallback repairs.
Rejected candidates and unresolved regions remain explicit. A score reduction
obtained only by dropping overlap or changing the averaging rule is not evidence
of a registration improvement.

## Validation

Freeze the processing policy before evaluating held-out photographs or scenes.
Use known synthetic projection/warp/visibility cases to verify mechanism, then
run complete real collections without annotations or visual steering. Report
whole-collection coverage, regional regressions, convergence, unresolved support,
runtime, and rendered output, including failures. Post-run visual inspection
may evaluate results but must not direct repairs in the reported evaluation.

The current room has already guided development and cannot serve as an unseen
collection. A successful automatic replay on it would establish removal of
manual inputs, not independent generalization. A new collection is needed for
that latter claim; the existing inputs remain sufficient to implement the
annotation-free pipeline before obtaining one.

The walking collection's acquisition is specified in `CAPTURE_GEOMETRY.md`.
Automatic processing must honor its multiple capture positions and revisits.
The failed flat/periodic chart runs are not a validated initial scene. A shared
output view must not silently impose a shared capture origin.
