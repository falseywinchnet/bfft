# Transport-object support experiment

## Scope

`viewer/object_transport_segmentation_viewer.py` is an isolated laboratory
that begins with the finished canonical transport-cell representation. It does
not modify cell allocation or the canonical viewer.

The question is narrower and falsifiable:

> Does the sparse interface graph of the BFFT transport cells contain enough
> support to make persistent object regions emerge without a learned model,
> a requested object count, or a per-object optimization?

## Literal graph

Let `C(x)` be the final hard site ID. A reconstruction site is not assumed to
be an object atom: its hard ownership may have disconnected islands or may
cross an observed discontinuity through a narrow pixel bridge. Each site is
therefore split into four-connected support fragments after removing
within-site pixel adjacencies whose unchanged-target OKLab jump exceeds the
fragment waterline. This does not create, move, refit, or delete a site. It
restores local topology that the site-ID quotient had hidden.

Two support fragments `i` and `j` are adjacent only when their labels meet
across a horizontal or vertical pixel edge. This gives a planar
region-adjacency graph `G=(V,E)`. It is the correct economy for the 50k-cell
case: the graph remains linear in image pixels and literal interfaces, not
`|V|^2` possible pairs.

Every edge reduces measurements only over its literal shared interface:

- unchanged-target OKLab jump;
- neighbouring-region OKLab difference, retained as an explicit failure
  control rather than trusted globally;
- cartoon and glass jumps;
- finite boundary confidence;
- BFFT metric action across the interface normal;
- cross-scale null reliability;
- a transport-support signature composed from local population density,
  energy, metric spectrum/director, and texture magnitude.

The last signature is important. On the matched-texture curved control, direct
target, cartoon, boundary, glass, and normal-action interfaces all have weak
or inverted discrimination. Differences in the BFFT population/metric support
are the first tested local quantities with useful separation:

| Interface cue | Cross-boundary AUC |
| --- | ---: |
| population-density log jump | 0.695 |
| energy log jump | 0.679 |
| texture-magnitude jump | 0.623 |
| metric trace jump | 0.586 |
| direct target jump | 0.453 |
| boundary confidence | 0.439 |
| normal transport action | 0.427 |

This does not yet solve the control: the signal is statistical and has gaps.
It does establish that the support geometry contains information absent from
ordinary color/cartoon comparison.

## Barrier and permeability

For every interface, robust dimensionless cue values are combined as positive
evidence. Unsupported finest-scale activity attenuates the soft cues, while a
decisive unchanged-target discontinuity cannot be cancelled:

```
e_soft = null_reliability * sum_k weight_k * cue_k
barrier = 1 - exp(-(boundary_weight * decisive_jump + e_soft))
affinity = exp(-barrier_scale * barrier)
```

The isotropic frequency floor in the transport metric is removed before
normalizing normal action. Otherwise every perfectly uniform interface would
incorrectly receive a barrier.

Region color is never allowed to connect nonadjacent cells. Thus a face and a
bridge with similar color/composition cannot merge without crossing every
literal interface between them.

## Simultaneous highpoints and waterline

Cell support density plus boundary enclosure defines a provisional altitude.
All graph-local maxima are born simultaneously. Equal plateaus use a
deterministic irrational/hash phase only to select a representative; this is
the limited and honest blue-noise role.

Edges are processed from most permeable to least permeable by a
maximum-support forest. When two peak-bearing components meet, the weaker peak
dies at that saddle. Its persistence is:

```
peak_score * (1 - saddle_affinity)
```

Low-persistence peaks are therefore gathered by surrounding support. A peak
behind a strong intervening barrier retains a distinct object ID.

The failure audit found an important qualification: `peak_score` and
`saddle_affinity` belong to different filtrations, so their product is not
standard topological persistence. It can create a sharp seed-selection cliff.
The maximum-support forest remains a useful complete edge hierarchy, but the
next object construction must guarantee that every nontrivial component at a
chosen barrier waterline has a representative. Objects may merge only when
their components meet at a saddle; an independently thresholded detail germ
may not erase a basin.

The current Python specification uses stable comparison sorting for graph
assembly/filtration and a heap for the two-label readout. This is acceptable
for the laboratory but is not the final form. A fixed-bin counting/radix
filtration and native union-find make the intended production complexity
`O(pixels + interfaces)`.

## Hard IDs, soft support, and two different confidences

The maximum-support forest preserves the widest path between every pair of
cells. A two-label seeded widest-path pass gives each cell:

- its winning persistent highpoint;
- its best competing highpoint;
- the bottleneck affinity of each path;
- geometric path distance from its winning core.

These yield two distinct diagnostics:

1. **Core altitude** is one at the persistent highpoint and decreases along
   the object tree. This is the requested highlight-to-waterline topology.
2. **Saddle margin** is best minus second-best bottleneck support. This says
   how decisively two proposed objects are separated.

They must not be conflated. The first implementation did so and consequently
lit confident contours rather than object cores.

Hard object IDs use the winning highpoint. Soft membership must not use
distance from that highpoint: doing so forced the edge of every large object
towards a 50/50 blend even when the runner-up was topologically implausible.
For best and second widest-path affinities `A1 >= A2`, the corrected winner
weight is

```
delta = log(A1 / A2) / barrier_scale
w1 = sigmoid(delta / temperature)
```

and is allowed to diffuse only in a narrow geodesic band inward from an actual
hard-object interface. Thus 50/50 means a genuine tie. Core altitude remains a
separate topographic diagnostic. No dense `pixels × objects` membership tensor
is constructed.

### Topology audit: sites are basis functions, not object atoms

The full-resolution Pikachu failure supplied a precise counterexample to the
original graph. Of 1,030 reconstruction sites, 463 had disconnected hard
ownership. The path assigning the lower black surround to Pikachu crossed
from black to yellow *inside site 831*. Its offending connected island was 24
pixels, 29% black and 58% yellow. Since no graph edge represented that
silhouette, no barrier weighting could repair the assignment.

The connected discontinuity-cut fragment graph reduces exact-black pixels in
the body part from roughly 18,500 to 36–92 over fragment waterlines from 0.03
through 0.20. The broad stable interval is evidence for a restored invariant,
not threshold luck. At the current 0.12 default the reconstruction remains
unchanged while the lower black surround has its own certain black basin.

Hard readout now uses a rooted first-arrival maximum-support watershed.
Persistent germs flood simultaneously, and a settled front cannot carry a
losing germ through already claimed territory. The unconstrained two-path
solve is retained only as an uncertainty diagnostic. Every hard part is a
connected raster region.

The astronaut control then falsified a second suspected mechanism: the flag
and suit ambiguity remains connected even under rooted first arrival. It is
not graph teleportation. It is a projected contact where appearance supports
continuity while contour termination supports an occlusion ordering.

This exposes the next missing variable. Every current interface cue is reduced
to an absolute magnitude and every graph edge is undirected. That quotient
discards front/back information. Single photographs still provide weak but
nonzero depth evidence through:

- T-junction termination and continuing occluding contours;
- signed cartoon/glass change across an oriented interface normal;
- relative transport scale, focus, and texture attenuation;
- enclosure polarity and contour ownership.

These cues must define a directed part-to-depth relation above the connected
hard basins. They must not be folded back into the scalar material barrier:
material continuity and occlusion order answer different questions.

## Failure-led hierarchy round

The first full-resolution Pikachu audit separated three failures that looked
similar in the original soft display:

- The connected white surround was already **99.23% one hard object**. Its
  apparent fragmentation was chiefly the invalid distance-based soft blend.
- Tail and body are different hard IDs because black genuinely disconnects
  them. Their widest-path bottleneck is no better than the path into black, so
  the unsigned literal adjacency graph has no lawful way to group them first.
- Astronaut overmerge is a seed-selection cliff. On a fixed graph, changing
  peak prominence from `0.30` to `0.40` changed 21 objects to one. Once only
  one seed remains, the propagation must give it every cell; barriers cannot
  leave a basin unclaimed.

These observations require a hierarchy, not another flat weighting sweep:

1. **connected material atoms** from witnessed scene discontinuities;
2. **parts** joined by internal seams, containment, and junction topology;
3. **parent objects** joined only by a local topological proposal such as a
   common surround, T-junction, or tangent-continuous amodal completion.

Appearance similarity alone is never a parent proposal. This is what prevents
a face and a remote bridge with similar colour/support from merging.

### Ontic, epistemic, and allocation evidence

The original additive barrier allowed `support_jump` to manufacture a wall.
That field measures how the allocator changed its density, energy, metric, or
texture support. It is useful evidence about the representation, not direct
evidence that the scene changed.

The viewer now includes an anchored research control:

```
D = direct target/cartoon/glass witness
M = 1 + boundary + region + transport + support modulation
B = 1 - exp(-D * reliability * M)
```

The invariant is `D == 0 => B == 0`. Additive and anchored barriers, the
direct visual witness, and the latent support frontier are all separately
visible. The anchored control intentionally remains a control until the direct
contour witness is calibrated; at current default weights it undersegments.

### Embedded interface topology

`experiments/embedded_interface_topology.py` now retains the planar information
that a cell-pair RAG average discarded:

- separately connected arcs even when the same two cells meet more than once;
- every dual-grid interface edgel;
- arc endpoints and endpoint tangents;
- closed arcs;
- junction vertices and their incident arcs.

Extraction is linear in interface pixels and the union kernel is compiled.
For full-resolution `25.png` it exposes 4,112 connected arcs and 2,712
junction vertices in roughly 0.2 seconds together with all other object
diagnostics. This is the missing support for closure, common surround,
good continuation, and T-junction experiments.

### Short-contact control

The previous widest-path affinity ignored shared-interface length. A one-pixel
permissive contact could therefore chain components exactly like a long
interface. The viewer now exposes bounded empirical shrinkage:

```
n_eff = interface_length / sqrt(min(cell_area_i, cell_area_j))
r = n_eff / (n_eff + short_contact_scale)
B_eff = r * B + (1 - r) * B_prior
```

It is off by default while being falsified. Unlike an unbounded length reward,
long interfaces recover their measured barrier and short contacts can only
move toward an explicit conservative prior.

### New diagnostic quotient

`experiments/object_hierarchy_diagnostics.py` contracts literal interfaces
below a chosen diagnostic waterline independently of object IDs. It reports
both directions:

- one connected material basin split among several object IDs;
- one object ID swallowing several disconnected material basins.

The viewer exposes these as four maps and provides click-through inspection of
the selected pixel, cell, object, runner-up, material basin, and neighboring
interface statistics.

## First part-to-parent experiment

`experiments/transport_object_hierarchy.py` keeps the hard object IDs as
visible parts and adds a conservative parent hypothesis above them. It does
not recolor or erase the part representation.

### T-junction depth order, and a rejected merge rule

At an embedded three-region junction `(A, B, S)`, the two arcs `A-S` and
`B-S` can form a continuing contour while `A-B` terminates against it. Under
the ordinary occlusion interpretation this supports a directed statement:

```
S is in front of A
S is in front of B
```

It does **not** prove that interpretation, and it does **not** by itself
support `A == B`. A material seam ending at an object's silhouette produces
the same local T geometry with the opposite physical interpretation. The
first parent implementation treated the
terminating pair as an internal seam and attracted `A` and `B` when a second
endpoint agreed. On the astronaut image this joined the face to the distant
background at otherwise convincing junctions near `(170, 116)` and
`(180, 149)`. The topology was real; the inferred identity was not.

Direct seam strength does not repair the rule. Plausible internal seams and
false occlusion contacts have overlapping strength distributions. Junction
attraction and its associated mutexes are therefore disabled by default and
retained only as an explicit experimental control. The useful information is
kept as a sparse directed depth proposal. Two-ended agreement can raise the
confidence of that proposal, but never changes it into an identity relation.

### Border ownership as a sparse partial order

`experiments/transport_border_ownership.py` lifts each T-junction occlusion
hypothesis onto the exact embedded interface arcs. Each continuing arc
receives a proposed signed owner: the incident pixels on the hypothesized
front region's side are marked front and the opposite side is marked back.
This produces a sparse candidate partial order without inventing a dense
monocular depth map. Conflicting hypotheses remain unresolved rather than
being averaged into a false certainty.

The competing material-attachment hypothesis now has its own global test. For
each repeated terminating pair `(A, B)` with common third region `S`, remove
the candidate seam and measure the remaining boundary of `A ∪ B`. When the
union is bounded and nearly all of that exterior faces `S`, the local T is
also consistent with one compound silhouette whose internal material seam
terminates at its exterior. This is exposed as **Enclosed seam attachment
proposals** and is not yet allowed to merge automatically.

The same module asks a separate scientific question: can the transport
support predict ownership away from witnessed junctions? Each hard part is
described by its population measure, transport energy, metric trace and
coherence, texture magnitude, cartoon and glass states, and null reliability.
A single robustly standardized direction is fitted analytically from the
directed relations in that image. Exact leave-one-relation-out subtraction
checks whether the result merely memorized each witness.

The answer so far is useful but negative in an important way. There is no
universal scalar rule such as “nearer regions are denser”:

| Image | directed relations | fit agreement | leave-one-out agreement |
| --- | ---: | ---: | ---: |
| Pikachu | 16 | 0.875 | 0.875 |
| Astronaut | 39 | 0.615 | 0.615 |
| Cameraman | 8 | 0.750 | 0.625 |
| Chelsea | 105 | 0.762 | 0.762 |

The selected support direction changes substantially by scene. Therefore
**Transport-depth extrapolation** is diagnostic only and cannot authorize a
merge. **T-implied contour sides** is the explicit occlusion hypothesis;
**T-implied frontness** is its part-level aggregation. The next open object is a
contour-network continuation law that can carry ownership beyond isolated
junctions while preserving an explicit unknown state.

### Common-surround first-arrival completion

Pikachu's tail and body remain genuinely disconnected in the hard part graph,
so no local seam can join them. The bounded nonlocal proposal is itself a
transport:

1. Every part boundary emits a labeled front into its surrounding parent
   region.
2. Fronts move only inside that common surround on the existing sparse cell
   graph.
3. First-arrival collisions create only the planar Voronoi neighbours of those
   boundary sources; no all-pairs comparison exists.
4. Gap travel, signed contrast polarity relative to the surround, boundary
   confidence, transport-scale agreement, and repeated collision support form
   the proposal score.
5. An accepted proposal attracts the two exterior parts and installs mutexes
   against the surround.

This is the ray direction we wanted: detail boundaries emit into the currently
unexplained surround, and the earliest lawful collisions reveal possible
amodal completion.

On full-resolution `25.png` with the current defaults:

- local seam/containment produces four preliminary parent groups;
- common-surround transport accepts the body-tail collision;
- yellow body and yellow tail share one parent;
- both black regions and the white surround remain outside that parent.

The parent pass is roughly 20–30 ms for the current Pikachu/Astronaut cell
graphs. It remains experimental. Astronaut exposes the next limit: a parent
layer cannot recover a distinction that the hard part readout never emitted.

### Lexicographic widest-path correction

The hard readout computed path distance but previously ignored it when two
different seeds had equal bottleneck affinity. The winning seed was then an
arrival-order artifact. The corrected semiring is lexicographic:

1. maximize bottleneck affinity;
2. only on an exact affinity tie, minimize supported path length.

This changes no non-tied transport decision and removes scan/heap order from
flat permeable plateaus.

## Honest present results

At a 256-pixel longest side:

| Image | Cells | Persistent objects | Object analysis |
| --- | ---: | ---: | ---: |
| matched-texture curved truth | 2,546 | 7 | 37 ms |
| cameraman | 1,731 | 17 | 29 ms |
| Pikachu (`25.png`) | 511 | 18 | 18 ms |

At 384 pixels, cameraman used 4,171 cells and 30 persistent objects in 74 ms;
Pikachu used 804 cells and 28 objects in 37 ms.

Visually, cameraman already gathers the broad field/background, isolates the
person, and preserves the thin tripod structure. Pikachu separates the broad
background, body, tail, eyes, mouth, cheek, paw, and several nested details.

The matched-texture truth is the important rejection: seven regions where the
truth has two means the local barrier still has contour gaps and texture
false positives. A one-edge bottleneck hierarchy can leak through the weakest
gap. The next supported experiments are:

1. tangent-coherent interface continuation using the transport director;
2. a core-altitude component tree rather than one-hop detail maxima;
3. bucketed hierarchy cuts scored by exact VI / adapted Rand / boundary F1;
4. internal-versus-external merge regret under the existing single-stage
   decomposition objective.

## Figure/ground limitation

The current support is deliberately unsigned. The metric tensor, jump
confidence, cartoon/texture/glass magnitude differences, and symmetric
first-arrival action remain unchanged if the two sides of an interface are
swapped. They can establish separation but cannot establish which side is
foreground.

Foreground ownership needs an asymmetric cue such as T-junctions,
surroundedness/closure, motion, parallax, or a signed occlusion model.
First-arrival timing or seed-population bias is not accepted as depth evidence.

## Run

From the repository root:

```sh
python viewer/object_transport_segmentation_viewer.py
```

Build the canonical cells once. The **Recompute objects only** button reuses
the sparse graph, so evidence, persistence, waterline, and soft-confidence
controls can be explored without rerunning the image decomposition or cell
fit.
