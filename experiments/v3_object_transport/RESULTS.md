# V3 object transport: initial control audit

> **Fixture correction (2026-08-20).** The raster originally recorded in this
> audit as “hard Pikachu” was byte-for-byte the user-supplied **easy** image.
> Its measurements remain reproducible but all Pikachu entries below through
> the resolution study refer to that easy raster.  The corrected hard control
> changes only the exterior margin to black and adds an eight-pixel white wall
> outside the untouched original panel; its separate 256-side rerun is in
> `results/v3_object_transport_hard_256/` and is summarized in the addendum at
> the end.  No earlier number has been silently replaced.

## Question and contract

The old object-decomposition work predates the combination that now matters:
the final V3 segmenter and the fused cartoon/texture decomposition.  This audit
therefore starts after V3.  It treats V3 output as a measured planar complex,
not as a tentative object partition.

## Repository archaeology

The history answers the chronology question directly:

- commit `bc1deaf`, July 28, introduced the original
  `transport_object_support.py`, its viewer, the 454-line experiment ledger,
  and `transport_relation_forensics.py`;
- commit `92e6ebc`, July 30, introduced `segmenting_v3.py` and made only a
  small follow-up edit to the already-existing object-support code;
- V3 then changed repeatedly through August 2;
- commit `4fd2d57`, August 2, introduced `compound_segment_quotient.py` and
  `region_family_fusion.py` as optional stages inside V3.

So the remembered distinction is exact: the **object decomposer predates V3**.
The compound quotient and family fusion are later V3-native extensions, but
they are not a first-principles rebuild of the decomposer.  The old ledger
itself says its starting point is the earlier finished transport-cell
representation and records the black-tip, tail/body, flag/suit, and
cup/plate failures.

There is a second seam.  The fused warm-interleaved Gilles--Osher/Bregman
engine in `experiments/meyer_bregman.py` is not what V3 currently invokes.
V3 calls `build_frozen_geometry`, whose decomposition is the older frozen
Meyer path.  The present audit therefore preserves V3 labels and computes the
fused split independently on the identical OKLab luminance raster.  This
separation is intentional: it lets us measure the newer decomposition without
quietly changing the segmentation substrate.

The experiment deliberately contains no object IDs, learned classifier,
pairwise affinity threshold, merge schedule, or image-specific rule.  The
historical region-family result is retained only as a control.  Sparse semantic
landmarks are held out and evaluated after inference.

All measurements below use the exact final V3 defaults (`half_cartoon`) at a
maximum side of 256 pixels.  The aligned fused evidence uses the established
`lambda=0.05`, `mu=40` engine with 400 deterministic warm-interleaved passes.
Its exact retained residual makes cartoon + texture + residual reproduce the
input to floating-point precision (maximum error `2.22e-16` over all five
controls); it adds roughly 0.6--1.0 seconds per control at this size.

## What V3 actually gives us

| control | leaves | compounds | old families | connected compound arcs | junctions | region pairs with multiple disconnected arcs | maximum arcs for one pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard Pikachu | 7,004 | 662 | 608 | 1,670 | 1,014 | 71 | 5 |
| coffee | 14,034 | 807 | 748 | 1,553 | 759 | 71 | 7 |
| astronaut | 15,775 | 1,467 | 1,285 | 3,506 | 2,050 | 161 | 6 |
| checkerboard | 1,476 | 582 | 565 | 1,535 | 955 | 31 | 3 |
| coins | 18,575 | 1,263 | 1,205 | 2,621 | 1,394 | 87 | 18 |

V3 has already done the difficult local measurement.  Its reconstruction is
excellent, its compounds are useful large supports, and the fine leaves retain
the lawful interfaces that those compounds summarize.  But a region-pair
graph is already too destructive: every control contains region pairs joined
by several spatially disconnected arcs.  Coins reaches eighteen.  The
connected arc, its side, its outside state, and its junction endpoints must be
preserved.

That observation changes the state space.  The natural atom for recognition
is not merely a region `r`, but a one-sided incidence

```
(r, incoming connected arc, outside region).
```

The current incidence bundle materializes both orientations of every arc and
the exact continuations available at junctions.  At leaf level this is 7,650
to 86,882 directed incidences across the five controls; this remains modest
enough for a sparse analytical operator.

## The controls reject a scalar notion of sameness

The sparse landmark audit gives the following diagnostic numbers.  Recall is
the fraction of labeled same-object pairs already united; false join is the
fraction of labeled different-object pairs united.  The distance AUC is the
probability that a same-object pair is closer than a different-object pair.

| control | compound recall | old-family recall | old-family false join | target-distance AUC | cartoon-distance AUC | texture-distance AUC |
|---|---:|---:|---:|---:|---:|---:|
| hard Pikachu | 0.000 | 0.333 | 0.000 | 0.704 | 0.704 | 1.000 |
| coffee | 0.250 | 0.250 | 0.000 | 0.364 | 0.409 | 0.409 |
| astronaut | 0.250 | 0.250 | 0.118 | 0.882 | 0.882 | 0.294 |
| checkerboard | 0.000 | 0.000 | n/a | n/a | n/a | n/a |
| coins | n/a | n/a | 0.000 | n/a | n/a | n/a |

These are intentionally sparse diagnostics, not benchmark scores.  Their job
is to expose contradictions:

- **Hard Pikachu.**  The old family control unites the two black ear tips and,
  separately, the yellow body and tail.  It cannot cross from either black tip
  into the yellow body.  The black surround is much closer in target space to
  the tips (0.078 and 0.124) than the body is (0.657 and 0.703).  This is the
  original fiasco in numerical form: local content produces three coherent
  manifolds, not one Pikachu.
- **Coffee.**  Content similarity is actively misleading.  Cup wall to table
  is 0.095 and cup wall to spoon handle is 0.114, while cup wall to plate is
  0.274.  The spoon's sampled bowl and handle happen to share one compound,
  but the visible specular spoon remains internally fractured.  Meanwhile the
  desired participatory unit is cup plus plate, not merely equal material.
- **Astronaut.**  Red and white flag samples unite locally, while blue does
  not.  Worse, the historical family attaches the red/white flag family to the
  suit, yielding an 11.8% sparse false-join rate.  Palette similarity helps
  within the flag and simultaneously corrupts depth ownership.
- **Checkerboard.**  Segmentation is nearly the ideal input, yet every sampled
  square remains a different compound and family.  A board is the coherent
  alternation and closure of unlike regions; no same-color merge can discover
  it.
- **Coins.**  The easy case works because each object offers a separate closed
  enclosure against a broad ground.  Similar coin contents do not need to be
  joined, and the ground relation prevents transport between their cycles.

Content is therefore useful evidence but not an object scalar.  Structure is
also necessary but not sufficient: closure alone cannot decide cup-with-plate,
flag-with-suit, or foreground ownership.  The missing quantity is consistent
transport of *relations* around and through the embedded complex.

The new fused split strengthens that conclusion rather than replacing it with
a better scalar.  On the sparse controls, fused-cartoon distance AUC is 0.704
for Pikachu, 0.409 for coffee, and 0.912 for astronaut.  Fused-texture-mean
distance gives 0.537, 0.386, and 0.750 respectively.  In particular, it makes
blue/red/white flag texture relations compact while separating their means
from the suit, but it still cannot declare object identity by itself.  On
Pikachu the tip/body fused-texture distances improve relative to the enormous
cartoon jump, yet the black field remains competitive; on coffee no marginal
channel reverses the contradiction.  The fused coordinates belong in the
connection, where signed transitions and their arrangement survive—not in a
new nearest-neighbor quotient.

The interface statistics also show that this is genuinely additional
evidence, not a renamed V3 channel.  Across the five controls, old-versus-fused
cartoon-distance correlation ranges from 0.525 to 0.788, while
old-versus-fused texture-distance correlation is only 0.342 to 0.571.  The
connection should therefore retain both decompositions initially and let the
ablation gate decide whether either is redundant.

## First analytical candidate: connection bloom

The next experiment should build a sparse connection on the incidence bundle,
not another merger.  Each directed incidence carries the V3 measurements of
its inside, outside, cartoon/texture transition, boundary support, tangent,
and structural ancestry.  Junctions provide the lawful continuation paths.

The working hypothesis is a connection heat/eikonal bloom:

1. Derive local transport maps from the empirical relation geometry itself,
   using covariance whitening and an orthogonal polar factor rather than
   hand-weighted cues or semantic labels.
2. Assemble those maps into a sparse connection Laplacian on directed
   incidences.  All incidences emit participation simultaneously; no chosen
   foreground seed and no greedy merge order is required.
3. Evaluate a single heat kernel, resolvent, or first-arrival solve.  Agreement
   under parallel transport reinforces rapidly.  Incompatible arrivals remain
   a smooth barycentric competition rather than becoming a brittle union-find
   decision.
4. Read object candidates from persistent low-curvature participation
   manifolds only downstream.  Hard labels are a report of the field, not the
   mechanism that creates it.

This construction gives each hard control a plausible route without adding a
rule for it.  Alternating black/white checker transitions can have coherent
cycle holonomy; distinct coins have distinct closed cycles; Pikachu's tips can
participate through consistent boundary/shape transport despite the color
jump; the flag can preserve its repeated stripe relation and depth ownership;
and cup/plate nesting can become a participatory manifold while the spoon's
own contour closes across specular changes.

This is a hypothesis, not a claimed solution.  In particular, raw raster
junction tangents are strongly axis-aliased and many junctions are merely
fine-scale V3 bookkeeping.  The operator must consume their measure without
mistaking every recorded junction for semantic evidence.

## Falsification plan

The first bloom implementation should be judged before any visual tuning:

1. Keep all semantic landmarks and masks out of construction.  They are
   evaluation only.
2. Require permutation invariance of region and arc identifiers and exact
   reversal consistency of the two incidence orientations.
3. Shuffle outside states while preserving the ordinary region graph.  If the
   result survives, relation lifting contributed nothing.
4. Ablate tangent/closure, structural ancestry, cartoon transition, and texture
   transition one at a time.  Each channel must earn its place on contradictory
   controls, not just improve an aggregate score.
5. Check 128/256/384 resolution stability and perturb V3's lawful
   over-segmentation without changing the source image.
6. Admit numerical resolution parameters, but no per-image thresholds,
   semantic seeds, expected object counts, or control-specific branches.
7. Test the five controls jointly.  A change that repairs Pikachu while
   attaching coins, or repairs the flag while swallowing the suit, is a failed
   transport law.

The immediate implementation target is therefore the connection operator and
its null/shuffle controls.  Full object masks and quantitative participation
scores come after the analytical field exists; adding them sooner would risk
turning the controls into rules.

## First bloom result: one scalar state is still the wrong object

The first operator was implemented exactly as proposed: all directed
incidences were whitened under one covariance measured jointly across the five
controls, placed on the exact region/arc/junction complex, and evaluated with
one normalized matrix exponential.  It has 30 raw channels and 29 non-null
empirical modes.  No semantic source enters the operator; landmarks query the
finished field only.

As a **relation-role embedding**, this is useful.  Sparse cosine-distance AUC
is 0.944 on hard Pikachu, 0.682 on coffee, and 0.794 on astronaut.  Removing
the fused split lowers coffee to 0.568 and astronaut to 0.676.  Shuffling the
outside state lowers Pikachu to 0.852 and astronaut to 0.559.  Thus fused
evidence and the directed outside relation both contribute.

But the operator also supplied its own falsification: repeated coins become
similar because they have the same role.  Role recognition is not instance
recognition.  A unit-time localized signed heat kernel then stayed so close to
its source that every nonidentical landmark similarity was approximately
zero; increasing diffusion time would merely add a tuning control.  The
parameter-free Green inverse integrated all path scales, but signed phase
helped Pikachu while harming coffee/astronaut, and unsigned transport did the
opposite.  Neither is promoted.

## The topological coordinates that were missing

Two exact planar constructions changed the picture.

### One-sided contour participation

A connected contour component follows a fixed region side through exact
junction continuations.  Its opposite regions participate in one witnessed
boundary relation.  This is why a black Pikachu tip can continue into yellow
body while the black surround remains the same side of the contour.  It is
also why separate coins retain separate cycles despite sharing one ground.

The compound controls contain 583 to 1,632 one-sided contour components.  The
flag result is the clearest positive: blue-to-red and blue-to-white acquire
participation 0.142, while every sampled flag-to-suit participation is exactly
zero.  Sparse astronaut AUC reaches 0.875 without a depth label.  The spoon's
sampled bowl and handle also remain one contour participant.  Hard Pikachu's
right tip reaches the body, but the left tip and tail remain separated at the
particular sampled compounds; contour transport alone is therefore
insufficient.

### Bounded relative complements

For every possible owner region, deleting it from the literal planar RAG and
taking all components with no route to the image frame yields bounded relative
complements.  Every owner emits them simultaneously; there is no selected
foreground or object count.

This exposes the hierarchy in the hard control.  The four sampled Pikachu
parts—both tips, body, and tail—co-occur perfectly in bounded manifolds while
the white wall does not.  They also co-occur with the black field in a larger
wall-bounded “picture panel” manifold.  That is not an error in the evidence:
it proves that enclosures are nested participatory objects and must not be
flattened into one equivalence relation.  Coins behave as hoped: sampled coins
share no bounded manifold with one another or with ground.

Coffee and astronaut reveal the limit.  A cup-wall enclosure recovers the cup
interior but not cup plus plate; no current bounded manifold unites the sampled
cup and plate.  At projected contacts, ordinary complement connectivity also
cannot decide occlusion order.  This is precisely where T-junction ownership,
nesting, and depth-contour continuation must enter.

## Participation is an algebra, not a weighted score

The controls now separate three independently meaningful coordinates:

- **relation role** unites the checkerboard and recognizes all coins as coins;
- **one-sided contour** localizes one coin and keeps flag separate from suit;
- **bounded enclosure** supplies object/scene nesting and the hard Pikachu
  panel hierarchy.

Choosing a different weighted blend for each image would be the prohibited
rule mistake.  The current candidate instead retains the complete non-empty
tensor algebra

```
K = (1 + K_role) (1 + K_contour) (1 + K_enclosure) - 1.
```

Every individual coordinate and every conjunction has unit algebraic
multiplicity.  Schur products preserve positive kernels, so this remains an
analytical participation geometry.  Sparse AUC is 0.926 on hard Pikachu,
0.659 on coffee, and 0.824 on astronaut.  More importantly, the field atlas
shows the intended distinction: checker unity remains global; all coins share
a role but the queried coin remains the only bright instance; the astronaut
flag remains a localized contour; and the Pikachu body query blooms across
most of the character while retaining the larger panel as a lower-confidence
nested manifold.

This is the first result here that looks like a smoothly confident
participatory manifold rather than a hard quotient.  It is not complete.
The black left-ear apex remains slightly more role-compatible with the black
field than with the body, and coffee still refuses the desired cup/plate
composition.  Those failures now have a narrow common target: an oriented
depth/nesting connection on contour components, not another content feature or
merge threshold.

## Revised next gate

The next operator should act on contour components and their junction ports:

1. retain tangent direction and signed inside/outside transition through each
   T-junction rather than reducing the junction to an undirected pair;
2. measure which contour continues and which terminates as an oriented depth
   relation;
3. add closed-contour nesting and shared curvature/center transport without a
   “cup” or “plate” branch;
4. place that depth/nesting kernel into the same complete participation
   algebra and demand simultaneous improvement on cup/plate, spoon integrity,
   flag/suit separation, both Pikachu tips, checker unity, and coin isolation;
5. retain shuffled-port, contrast-inversion, and resolution controls.

The failed heat and Green arms remain in the artifact bundle.  They are useful
nulls and should not be silently tuned into successes.

## Oriented depth census: two more useful nulls

The exact 2-by-2 sector census finds 756 classical three-sector cap records
on hard Pikachu, 518 on coffee, 1,581 on astronaut, 848 on checkerboard, and
959 on coins.  At this raster representation every accepted record has the
same tangent anisotropy (`1/3`) and perfect cap-side tangent continuation
(`1`).  Those quantities are consequences of the lattice definition, not
depth evidence.  Treating them as confidence would simply reward ubiquitous
V3 bookkeeping.

The nontrivial observables were therefore kept separately and lifted onto an
entire one-sided contour component:

- cap-junction count and persistence along the component;
- agreement of the two signed native and fused content transitions;
- contrast-normalized interface-focus match margin and reliability.

Focus is measurable on nearly every compound contour component.  Between 179
and 566 components per natural-image control carry both a T cap and focus
evidence.  But the coffee cup and plate do not share a direct contour port,
and the old focus forensics had already shown that plate and table are nearly
coplanar.  Consequently these measurements are retained as an oriented depth
chart, not promoted into an object kernel.

## Exact winding and centered nesting

Every even-degree one-sided contour component was next converted to its exact
mod-2 winding field.  Duplicate physical fields were removed, and each field
emitted two region coordinates: literal area overlap and overlap weighted by
the field's own covariance-normalized Cauchy centrality.  This construction
has no selected foreground side, radius, or semantic center.

At compound level it finds 666 distinct fields on hard Pikachu, 770 on
coffee, 1,387 on astronaut, 485 on checkerboard, and 1,285 on coins.  Most are
single-region fields; only 9, 26, 42, 2, and 60 respectively contain multiple
regions.  Centering improves the standalone hard-Pikachu sparse AUC from
0.778 for literal winding to 0.870, while keeping distinct sampled coins
exactly separate.

Placed into the complete participation algebra as a fourth coordinate,
centered winding finally reverses the original left-tip inequality:
left-tip/body becomes `0.2975`, just above black-surround/left-tip at
`0.2888`, and hard-Pikachu AUC rises from `0.926` to `0.963`.  It fails the
joint gate, however: coffee drops from `0.659` to `0.636` and astronaut from
`0.824` to `0.765`.  The coordinate is therefore a valuable Pikachu-specific
explanation but is **not promoted** into the canonical three-coordinate
algebra.

The same construction at the 14,034-leaf coffee level is an even stronger
falsification.  It produces 13,558 tiny winding fields, 13,470 of them
single-region, and still gives exactly zero sampled cup-wall/plate
participation.  Finer segmentation does not manufacture the missing
relation.

## Generic composition is not the missing transport

Two analytical, seed-free composition controls tested whether cup-to-cup and
cup-to-plate paths merely needed transitive closure:

1. a spectral exponential, `exp(K / lambda_max) - I`, which sums every path
   order at one measured spectral unit;
2. a direct-sum feature kernel containing every ordered coordinate word of
   length one and two with unit multiplicity.

The base spectral bloom moves coffee AUC from `0.659` to `0.682`, but it does
not create a discriminating assembly relation: cup-wall/plate is `0.2254`,
cup-wall/table is `0.2211`, and the false plate/spoon attraction rises to
`0.3735`.  The typed order-two algebra saturates almost every natural-image
pair near one and lowers coffee to `0.591`.  Adding centered winding improves
hard Pikachu as high as `0.981` in the spectral arm but lowers coffee and
astronaut again.

This rules out generic transitivity as the sought rapid bloom.  The missing
state is now sharply identified: **oriented amodal continuation across an
intervening contour**.  A lawful next operator must transport a continuing
surface behind an occluding boundary while keeping the occluder, terminated
surface, and focus reliability distinct.  It must not allow arbitrary paths
through a dense participation graph.

## Next research gate: amodal contour connection

The next experiment should operate on contour ports rather than region pairs:

1. At every exact T port, preserve the continuing cap contour and the two
   terminating stem sides as different states.  Pixel-scale tangent identity
   is only topology; signed transition agreement and focus ownership remain
   independent measures.
2. Propagate each port along its complete one-sided contour component, then
   compare only geometrically compatible *terminating* ports across the
   occluder.  Retain all covariance-normalized displacement, tangent, and
   curvature coordinates rather than selecting a completion threshold.
3. Emit an amodal incidence hyperedge `(surface-before, occluder,
   surface-after)`.  This ternary state is essential: flattening it to a
   pairwise join would recreate flag/suit and plate/spoon failures.
4. Evaluate the resulting field first on hard Pikachu, coffee, and astronaut;
   checkerboard and coins remain mandatory nulls.  Promotion requires both
   Pikachu tips to prefer body over black surround, cup/plate to separate from
   cup/table and plate/spoon, flag colors to remain above every flag/suit
   relation, and distinct coins to stay isolated.
5. Repeat contrast inversion and 128/256/384 resolution controls before the
   amodal coordinate enters the canonical participation algebra.

## Ternary amodal ports: the representation is right, the evidence is sparse

The first amodal implementation preserves the distinction the pairwise graph
could not: a T port contains a continuing cap contour, a terminating stem,
and two ordered background sides.  Port pairs propose the two background
correspondences while the cap remains explicit context and is never inserted
into the background participation kernel.

Exact same-contour pairing yields 3,919 proposals on hard Pikachu, 2,060 on
coffee, 4,291 on astronaut, 856 on checkerboard, and 2,904 on coins.  This is
too literal because a physical occluder such as the suit crosses many V3
regions.  Adding the parameter-free Delaunay neighborhood of all T ports
raises those counts to 5,643, 3,274, 8,018, 2,959, and 5,229 while retaining
every exact-contour pair.

The expanded chart creates a small flag-blue/flag-red proposal and leaves all
sampled flag/suit entries exactly zero.  That is the desired ternary
invariant.  But most high-scoring coffee proposals are one-pixel
tessellation events.  Two additional literal observables expose them:

- the port tangents must face the hidden gap, represented continuously by
  tangent opposition and two facing cosines;
- the raster segment between the ports must cross the proposed cap support.

Their conjunction is the first amodal coffee coordinate to reverse the local
liquid/plate versus plate/spoon error: `0.01245` versus `0.00977`, with sparse
amodal AUC `0.773`.  It does not repair the flag samples and it lowers hard
Pikachu when flattened into the complete part algebra, so it is retained as a
directed transport chart rather than promoted as another object kernel.

The Hodge projection explains why a local T gate is unsafe.  Only `0.484`,
`0.494`, `0.450`, `0.363`, and `0.497` of the unit T-arrow energy on Pikachu,
coffee, astronaut, checkerboard, and coins respectively lies in one globally
consistent scalar depth potential.  The rest is cyclic depth evidence.  A
positive-potential gate suppresses the true liquid/plate proposal while
retaining several plate/spoon proposals.  Global depth is meaningful
evidence, but not an object rule.

## Support/manifold assembly: a genuine second algebraic level

Cup plus plate is not merely one surface continued behind an occluder.  It is
a supported assembly.  The next construction therefore changes level:

- every exact bounded relative-complement manifold remains a soft part
  candidate;
- every V3 region simultaneously acts as a possible spatial support;
- their displacement is measured in the support region's full second moment,
  including the exact `1/12` unit-pixel aperture variance;
- their scale agreement is the symmetric dimensionless quantity
  `2 A_m A_s / (A_m^2 + A_s^2)`;
- centeredness is the parameter-free Cauchy coordinate
  `1 / sqrt(1 + d^T Sigma_s^-1 d)`.

Every manifold/support proposal is emitted.  There is no chosen support,
center radius, gravity direction, object count, or threshold.  The closed-form
Gram avoids explicitly materializing the large proposal feature matrix.

At the established 256-pixel audit scale this is the strongest coffee result
so far.  Standalone assembly AUC is `0.977`; the hierarchical algebra

```
(1 + K_complete_parts) (1 + K_assembly) - 1
```

reaches `0.909` on coffee, `0.981` on hard Pikachu, and leaves astronaut at
`0.824`.  Cup-wall/plate is `0.0741`, cup-wall/spoon `0.0149`, and
cup-wall/table `0.0727`; liquid/plate is `0.1447`.  Assembly itself gives
plate/spoon exactly zero and the hierarchy cuts the old plate/spoon attraction
from `0.2893` to `0.0964`.

The hierarchy matters.  Flattening assembly beside role, contour, and
enclosure reaches only `0.727` on coffee.  Shuffling bounded-manifold member
identities while preserving their cardinalities collapses standalone coffee
from `0.977` to `0.409` and the hierarchical result back to the canonical
`0.659`.  Thus the gain comes from the measured topology, not a generic
center/size prior.  Scale alone reaches standalone `0.977`, but centeredness
is required for the stronger hierarchical separation and Pikachu result.

The field atlas also preserves the intended coordinate distinction:
support/manifold assembly localizes the queried coin, remains almost inert on
the astronaut, and does not replace checkerboard's board-wide role field.

## Resolution falsification: not yet promotable

The complete V3 pipeline was rerun at maximum sides 128 and 384, not merely
resampled after inference.  Compound counts change substantially—for coffee,
247, 807, and 1,808 regions at sides 128, 256, and 384.  The exact-point
canonical AUCs are:

| control | 128 | 256 | 384 |
|---|---:|---:|---:|
| hard Pikachu | 0.648 | 0.926 | 0.759 |
| coffee | 0.591 | 0.659 | 0.500 |
| astronaut | 0.838 | 0.824 | 0.824 |

Assembly remains independently informative on coffee (`0.750`, `0.977`,
`0.727`), but the hierarchical improvement appears only at 256
(`0.591`, `0.909`, `0.500`).  It is therefore **not promoted** into the
canonical object stack.

This is partly, but not merely, probe instability.  At 256 the coffee plate
landmark lies in a 3,741-pixel plate compound; at 384 the same normalized
point lies in a 30-pixel highlight compound that fails to inherit the plate
field.  An evaluation-only Gaussian aperture scale-space reports every fixed
aperture from zero through 4% of the image side.  At a 1% aperture, coffee's
hierarchical AUC is `0.886`, `0.909`, and `0.818`, showing that the underlying
field is more stable than the atom query.  Hard Pikachu does not share that
aperture stability because its tips abut the black surround; smoothing across
the true boundary is correctly destructive.

The exact-point scale curvature contains a promising discriminator.  At the
middle log scale, cup-wall/plate assembly has positive excess `0.0723`, versus
`0.0243` for cup-wall/table, `-0.0046` for cup-wall/spoon, and `-0.0050` for
plate/spoon.  The scale-curvature atlas shows these as localized ridges.
However, an elementwise curvature field is not automatically a positive
participation kernel, and a naive equal direct sum of overlap-lifted kernels
from the three resolutions dilutes the useful middle-scale relation.  Choosing
256 because it wins these controls would be precisely the prohibited tuning
mistake.

## Revised research gate

The next construction must be scale-covariant rather than scale-selected:

1. align the RKHS feature frames of the 128/256/384 region kernels through
   their exact common-pixel overlap maps;
2. retain value, first log-scale derivative, and second log-scale derivative
   as separate feature coordinates, so a participation ridge remains PSD
   instead of becoming an elementwise score;
3. transport a fine highlight through its coarse plate support without
   averaging unrelated coarse spoon relations into the same field;
4. repeat the outside-state shuffle, bounded-manifold shuffle, contrast
   inversion, and all five controls before any scale ridge is promoted;
5. keep amodal background continuation and supported assembly as different
   ternary operations.  Their failures show they cannot be flattened into one
   notion of object sameness.

## Corrected hard-frame addendum

The deterministic fixture builder freezes the supplied image as
`pikachu_easy.png`, replaces only its exterior white margin by black, and puts
an eight-pixel white wall immediately outside the unchanged 431-by-405 panel.
The wall's inner top is row 35 and the first dark ear-tip pixel is row 38.  The
corrected raster has SHA-256
`f1a94868ed5ce13347af73ada8123956fc9d49241395b4dbe70501d3008d34a4`.

A fresh V3 run produces 7,007 leaves, 695 compounds, 643 historical families,
1,762 connected compound arcs, and 1,074 junctions.  The canonical sparse AUC
is `0.944`, assembly alone is `0.667`, and hierarchical canonical-plus-assembly
is `0.944`.  The decisive left-tip inequality remains unresolved:

| coordinate | tip/body | tip/black surround | desired ordering |
|---|---:|---:|---|
| canonical | 0.335733 | 0.347291 | fails narrowly |
| assembly | 0.007931 | 0.000081 | succeeds |
| hierarchical | 0.115442 | 0.115800 | fails narrowly |

The right tip already prefers the body (`0.386291`) to the black surround
(`0.328066`) canonically.  The wall changes global sparse performance but does
not solve transport through the black-on-black left-tip ambiguity.  This is a
stronger and cleaner confirmation of the original failure, now on the intended
fixture.

An exact RGB-complement control was also run on the easy raster.  V3 merges
both tip landmarks with the complemented surround into one compound, causing
similarity `1.0` before object transport.  This is an upstream contrast-
equivariance failure, not a quotient failure; the result is retained at
`results/v3_object_transport_easy_contrast_invert/`.

Finally, blind cross-resolution Procrustes alignment is rejected for both the
full hierarchy and assembly alone.  Both kernels are full rank because every
proposal contains a support-identity diagonal.  Aligning those RKHS frames
therefore transports segmentation identity rather than persistent assembly.
The next scale-covariant construction must align explicit manifold/support
proposal topology, not eigencoordinates of the already-collapsed region Gram.

## Explicit proposal topology: the identity diagonal was the wrong collapse

The support/manifold proposal feature `sqrt(w_ms)(e_s + p_m)` necessarily
contains a full-rank support-identity diagonal. Its useful relation is the
off-diagonal cross incidence `C = P.T W`. The symmetric connection
`C + C.T`, with self loops removed and exact degree normalization, retains
every support/manifold proposal while separating transport from identity. The
unit combinatorial heat exponential is one closed-form bloom containing all
path orders with factorial measure.

At 256, proposal heat has coffee AUC `0.977`. Transporting the existing
canonical part kernel through its half heat reaches `0.955` on coffee and
`0.971` on astronaut. On the corrected hard Pikachu it reaches `0.963`; the
left tip now prefers body (`0.4793`) over black surround (`0.3851`). Shuffling
manifold membership drops corrected-hard Pikachu to `0.704`, coffee to `0.636`,
and astronaut to `0.853`.

This is typed assembly transport, not a universal sameness kernel. Its field
localizes the queried coin and does not supply checkerboard's global role.
Flattening it back into the complete algebra removes much of the gain.

Resolution remains a falsification. Transported-part AUC on the original easy
raster is `0.667/0.852/0.833`; coffee is `0.636/0.955/0.523`; astronaut is
`0.632/0.971/0.765` at 128/256/384. On the corrected hard fixture it is
`0.444/0.963/0.981`: the native three-pixel ear/wall gap becomes subpixel at
128 and the necessary proposal is genuinely absent. A single multiplex graph
using exact common-pixel overlap between all resolution layers does not fix
this. Its coffee heat AUC is `0.886` versus `0.864` under shuffled scale
alignment, and its transported coffee result is worse than the shuffle. Raw
overlap is therefore location alignment, not structural inheritance.

## Wavelet leaders: scale-causal content without label inference

The wavelet-leader construction in León-López et al.,
[arXiv:2501.08694](https://arxiv.org/abs/2501.08694), supplies a missing
observable. A leader is the supremum of normalized wavelet detail over the
`3 lambda` neighborhood and every finer scale. Unlike a post-hoc average of
segment kernels, it carries a local irregularity into coarser cells before any
segmentation label is inferred.

The adaptation uses orthonormal Haar details at every dyadic scale on target,
fused cartoon, fused texture, and exact residual. It retains centered
log-leader mean and RMS at every scale. A second chart retains the closed-form
affine scale law of regional mean and variance: slope, intercept, and residual
over every available scale. There is no selected scale band, class count,
training label, Potts coefficient, or iteration.

At side 256 the raw leader kernel reaches `1.000` on both coffee and astronaut.
Proposal transport preserves `1.000`; shuffled region-to-leader correspondence
falls to `0.773` and `0.882`. The named relations have the right ordering:

| relation | raw leader | proposal-transported leader |
|---|---:|---:|
| cup wall / plate | 0.4127 | 0.6427 |
| cup wall / table | 0.2050 | 0.4253 |
| cup wall / spoon | 0.3884 | 0.4675 |
| liquid / plate | 0.8590 | 0.9304 |
| plate / spoon | 0.3493 | 0.4028 |
| flag blue / red | 0.3980 | 0.4314 |
| flag blue / suit | 0.0108 | 0.0259 |
| flag red / suit | -0.1769 | -0.1542 |

The full field is more important than the sparse score. Coffee blooms across
cup and plate while keeping the table weaker. The flag colors bloom together
and the suit stays dark, though the face and rocket expose other regions with
similar regularity. Coins show shared material regularity across instances;
checkerboard remains nearly inert. Thus the leader chart measures **content
regularity**, not object identity, exactly as required.

Raw leader charts are not resolution invariant: their proposal-transported
AUCs are `0.750/1.000/0.909` on coffee and `0.926/1.000/0.750` on astronaut.
The scale-law chart improves corrected-hard Pikachu's standalone result to
`0.852/0.778/0.741`, compared with its shuffled null
`0.444/0.648/0.722`, and keeps astronaut at `1.000/0.882/0.912`; however it
loses the coffee distinction at high scale. The raw chart and scale law are
therefore retained as separate content coordinates. Their complete algebra is
not promoted: flattening complementary meanings again reduces the strongest
controls.

The paper's Potts/Gibbs segmenter is deliberately rejected. It assumes a
chosen class count, four-neighbor homogeneity, learned granularity parameters,
k-means initialization, 300 Gibbs iterations, and MAP labels. The useful
contribution here is the measured leader chart, not its rule-bearing label
inference. Detailed correspondence is recorded in
`PAPER_NOTES_2501_08694.md`.

## Directed leader incidence: endpoint identity matters selectively

Raw-plus-scale-law leader evidence was lifted into the directed V3 incidence
fibre in two distinct forms. `transition_only` appends the outside-minus-inside
leader transition. `ordered_endpoints` retains the complete ordered pair of
inside and outside leader coordinates; the transition is already in its linear
span. Both are jointly whitened with the original incidence measurements and
bloomed through the same exact directed topology. Region correspondence is
then shuffled before incidence construction as the matched null.

At side 256, ordered endpoints improve astronaut role AUC from `0.794` to
`1.000`; after contour/enclosure completion and proposal transport the result
is `1.000`, versus `0.824` under the ordered shuffle. The final ordered
astronaut advantage survives at all three resolutions:

| side | ordered endpoints | shuffled ordered endpoints |
|---:|---:|---:|
| 128 | 0.691 | 0.574 |
| 256 | 1.000 | 0.824 |
| 384 | 0.765 | 0.662 |

Coffee does not validate the same claim. Its 256 final score is `0.977` in
both the real and shuffled ordered arms, and at 128/384 both arms again tie.
The proposal connection, not leader alignment, carries that apparent success.

On the corrected hard Pikachu, transition-only incidence is already strong at
the role level (`0.963/1.000/1.000` at sides 128/256/384). Final proposal
transport is `0.556/0.981/1.000`, but the shuffled null is
`0.370/0.926/1.000`. Thus the 256 ear-tip inequality is genuine—left
tip/body `0.4537` versus tip/surround `0.3443`, and right tip/body `0.5262`
versus tip/surround `0.3332`—but neither low-resolution transport nor the
high-resolution null permits promotion as a stable object operator.

## Dense content diffusion: rejected twice

A symmetric Strang split tested leader correspondence and proposal topology as
separate unit-spectral heat generators:

```
H_half = exp(P / 4) exp(C / 2) exp(P / 4)
K_out  = normalize(H_half K_parts H_half)
```

This has no bandwidth, merge threshold, seed, class count, or iterative stop.
It is nevertheless the wrong algebra. At side 256 it reaches `0.963` on the
historical easy Pikachu versus `0.833` under shuffled content alignment, but
coffee is `0.932` versus a stronger `0.955` null, and astronaut is `0.853`
versus `0.971`. On the corrected hard Pikachu, both real and shuffled arms are
`1.000`. The field spreads material equivalence too freely across instances.

An exact boundary-role gate was then applied before diffusion. The leader and
independent incidence-role kernels are conjoined by their Schur product, which
is the parameter-free positive-kernel conjunction. The isolated gated
coordinate retains some real signal—Pikachu `0.815` versus `0.704` and
astronaut `0.853` versus `0.794`—but diffusion again reverses the astronaut
comparison (`0.853` versus `0.971`) and fails coffee (`0.932` versus `0.955`).

These failures settle the first type distinction. Content regularity may be
carried as a payload coordinate; it must not itself become a generic dense
transport graph. The next control asks whether typed payload coordinates can
be scalarized before a non-iterative object decomposition, while sparse
proposal/incidence operators remain the lawful transport between regions.

## First all-region object packets: scalar completion is also too early

Every region was next given a soft packet column in a scalar complete algebra
of four independently measured coordinates: structural parts, proposal-
transported raw leader payload, separately transported scale-law payload, and
ordered endpoint incidence role. The matched null jointly permutes both
leader charts and uses the shuffled ordered endpoint role. All resulting
kernels are positive; the minimum eigenvalue is positive on every control.

The scalar packet reaches astronaut `1.000` versus `0.882` under shuffle and
the historical easy Pikachu `0.926` versus `0.759`. Both tip/body relations
exceed their tip/surround controls. Coffee is decisive, however: the packet
falls to `0.795`, from `1.000` for its raw leader payload alone, and incorrectly
places cup-wall/spoon (`0.2184`) above cup-wall/plate (`0.1524`). Removing the
ordered coordinate improves only to `0.841`.

Therefore even the unit-multiplicity complete positive algebra is not a valid
universal scalarization of typed object evidence. The next packet ABI must
remain operator- or vector-valued through recomposition: role, boundary,
assembly, raw content, and scale law can agree or disagree without one being
silently converted into generic sameness. Discretization and idempotent
extraction remain downstream questions.
