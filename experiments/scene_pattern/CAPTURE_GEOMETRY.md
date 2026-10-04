# Authoritative walking-collection capture geometry

The owner clarified the acquisition after rejecting the first global-chart
results as geometrically wrong. Preserve this account in every continuation.

- There are multiple capture positions along the walkway next to the house,
  roughly along a north-south line, from the road end toward the back yard.
- At a position the owner made a panoramic capture, rotated, and made another.
- The owner moved approximately four or five paces and repeated at multiple
  positions. Neither exact spacing nor an exact common camera center is known.
- The owner subsequently revisited the positions, rotating while taking the
  individual photographs there. Revisited positions need not be identical poses.
- The 28 panoramas and 92 individual photographs are not observations from one
  position. Do not infer exactly fourteen stations solely from the counts.

## Rejected interpretation

The flat and periodic common-chart runs are failed diagnostic experiments.
They do not establish scene geometry. Large loop residuals are not sufficient
proof of a shared angular period: changing camera centers, parallax, ambiguous
matches and panorama stitching must be distinguished. Do not continue refining
these atlases as though their geometry were validated. Lower sparse residuals,
positive mesh triangles, connected input graphs, or improved local flow do not
certify the capture arrangement. The dense refinement was interrupted after
this clarification.

## Required reconstruction model

Retain a separate pose/origin for each capture and infer groups of nearby
capture positions. Capture sequence and the approximately linear route provide
candidate priors, not fixed station labels or exact collinearity. Panoramas
made near one position can establish local angular coverage. Relate different
positions through shared scene structure with translation and depth-dependent
parallax (or an equivalent visibility-aware layered representation). Keep
panorama stitching distortion distinct from inter-position geometry.

The requested two stages remain: establish the scene with the panoramic
sweeps, then attach and promote information from individual photographs. The
first stage is a multi-position scene representation, not a single-center
panorama. Returning photographs should first be associated with candidate
places, then have their actual pose refined. Do not force an entire revisit
onto one camera origin.

Alignment evidence still starts in the 2-D captures. Fusion must account for
which observations refer to the same visible surface before combining them.
One displayed view is rendered from the shared representation; it does not
imply that every observation shares one viewpoint. No manual object repair
regions or source-specific coordinate adjustments are authorized by this
clarification.
