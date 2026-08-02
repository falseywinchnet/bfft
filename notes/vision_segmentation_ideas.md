# Decomposition ideas from how human vision segments scenes (2026-07-26)

Candidate directions for the cell/transport decomposition. Each is chosen to be
(a) absent from the current stack, (b) implementable with primitives we already
own (Meyer ladder, ROF/shade solves, structure tensor, geodesic propagation,
blue noise, fast FFT), and (c) cheap enough to run in the viewer loop.

## The three axis shifts

Everything currently in `viewer/*_decomposition.py` is the same algorithm with
different bookkeeping: **allocate a budget bottom-up in proportion to
reconstruction error, under a symmetric spatial metric, and fit pixels.**
Human vision differs on three axes, and each axis is a separate project:

1. **Ownership.** Boundaries are not symmetric. A contour belongs to one side;
   the other side continues behind it. Our cells treat a boundary as a wall
   that costs the same from both directions.
2. **Objective.** The visual system does not minimize pixel error. It stops
   when the encoding is *indistinguishable*, which for texture means matching
   summary statistics, not samples.
3. **Priority.** Salience is not error magnitude. It is persistence across
   scale and rarity in feature space. Our allocator gives its budget to exactly
   the high-contrast, high-frequency, perceptually inert content that a person
   ignores — which is why the README has to special-case "persistent one-pixel
   edge mismatch must not consume the entire cell budget."

Ideas below are ordered by (novelty here x expected payoff x cheapness).

---

## 1. Border ownership: make the geodesic cost asymmetric

**Vision.** V2 border-ownership cells assign each contour to one side within
~25ms, using context far outside the classical receptive field (convexity,
surroundedness, closure). Figure owns the edge; ground continues behind it.

**Mechanism.** Compute a signed ownership field on the cartoon boundary. Cheap
version: for each boundary pixel with normal `n`, cast ring votes at several
radii into both sides; the side that is more *enclosed* (votes land on other
boundary pixels whose normals point back) wins. This is a medial-axis /
generalized-Hough vote, two passes of shifted accumulation per radius, no
optimization. Output `beta(x) in [-1,1]` per boundary pixel.

**Use.** Replace the symmetric `cartoon crossing barrier` with a **directed**
graph cost: crossing *out of* the figure side is expensive; crossing *under*
from the ground side is cheap. Our geodesic propagation already runs on a pixel
graph — this is a per-edge cost that differs by direction, which multi-label
Dijkstra/fast-marching handles without structural change.

**Payoff.** Ground cells stop fragmenting around every occluder and complete
behind figures — amodal completion falls out of the metric instead of being
bolted on. Directly attacks §9.3 and the Pikachu background shattering.

**Test.** Count background cells on Pikachu before/after; the ears and limbs
should stop cutting the background into separate territories. Stronger test: an
occluding bar over a textured field — one ground cell, not two.

**Cost.** O(radii x pixels) accumulation, ~5 shifted passes. Negligible.

---

## 2. Fill in from owned borders instead of fitting patches

**Vision.** Surface appearance is *constructed by propagation from boundaries*.
Craik–O'Brien–Cornsweet: two regions with identical interiors look different
because of the edge profile between them. Neon color spreading. Interiors are
not measured, they are filled.

**Mechanism.** A cell stores boundary values on the border it *owns* (idea 1)
plus a single interior anchor, and the interior is reconstructed by anisotropic
diffusion under the cartoon metric — Laplace/Poisson with Dirichlet data on the
owned border. We already run Jacobi-style sweeps in `shade()` and the ROF
solver; this is the same machinery with different boundary data.

**Payoff.** (i) Kills §9.5 outright: a diffusion solution obeys a maximum
principle, so no cell can extrapolate a bright/dark shard. (ii) Storage goes
from O(area) to O(perimeter) — large flat regions get nearly free, which is the
efficiency story the model currently lacks. (iii) The failure modes become
*human* failure modes.

**Test.** Feed the decomposition a Cornsweet stimulus. A patch-fitting model
reproduces the flat interiors correctly; a filling-in model reproduces the
*illusion*. If we reproduce it, that is a headline result, not a bug.

**Cost.** One multigrid/Jacobi solve per update over the cell support; cheaper
than the current least-squares plane fits at large cell counts.

---

## 3. Stop at a metamer, not at a small residual

**Vision.** Peripheral vision encodes texture as summary statistics over pooling
regions (Portilla–Simoncelli statistics; Freeman–Simoncelli metamers;
Rosenholtz's texture tiling model). Two images with matching statistics in
matching pooling regions are indistinguishable, however different their pixels.

**Mechanism.** Per cell, build a statistic vector from the Meyer ladder we
already compute: per-band energy, cross-band correlations, band cross-products
at a few relative offsets, and the marginal skew/kurtosis of the lowpass.
Define cell error as **distance in statistic space**, not pixel MSE. When a
cell is statistically converged, it is *done* regardless of pixel residual, and
its interior is synthesized rather than fitted: take the cell's own texture-layer
content and randomize phase in-place (one forward and one inverse transform —
we own the fast path), which preserves the second-order statistics exactly.

**Payoff.** The budget stops flowing to content nobody can localize. This is the
principled replacement for allocation driven directly by raw residual error.

**Test.** At fixed cell count, compare pixel-PSNR (will drop) against a
statistics-distance score and against `meyer_segregation.py`-style scoring
(should rise). The interesting number is cells spent on the bricks/grass region.

**Cost.** Statistics are sums over bands we already have; synthesis is two
transforms per cell refresh.

---

## 4. Orientation as a line-field defect problem, not a feature vector

**Vision.** Texture regions differing only in orientation segregate
pre-attentively. `experiments/meyer_segregation.py` measures our failure here:
0.542, chance, because band energy is isotropic. The standard fix is an oriented
filter bank, which is expensive and adds a whole new front end.

**Mechanism.** We already compute the structure tensor. Its principal direction
is a **line field** (director, defined mod pi), and the right object is the
Q-tensor `Q = [[cos2t, sin2t],[sin2t, -cos2t]] * coherence`. Orientation
boundaries are precisely where *parallel transport of the director fails*:
compute the holonomy around small loops, i.e. the loop integral of the
doubled-angle gradient, and the winding/defect density of the line field. Two
regions with identical band energy but different orientation are separated by a
curve of large transport discrepancy, and their interiors have near-zero
discrepancy — which is exactly the interior-labelling that band energy cannot do.

**Payoff.** Attacks a *measured* null with a field we already build, and reuses
the holonomy/gauge language from the phase work — same mathematics, spatial
instead of spectral. Doubles as the honest driver for metric anisotropy: use
coherence-weighted `Q` rather than a per-site frozen ellipse (§9.2).

**Test.** Re-run `meyer_segregation.py` on the orientation field with holonomy
features appended. The bar is 0.542; anything above ~0.85 closes the known gap.

**Cost.** Two convolutions plus a 2x2 loop sum. Cheapest idea in this note.

---

## 5. Priority by scale-space lifetime, not gradient magnitude

**Vision.** Structures that survive across many scales are objects; structures
that die immediately are texture or noise. Salience tracks persistence.

**Mechanism.** We already produce a ladder. For each local feature, record the
*range of rungs over which it survives* — a one-dimensional persistence
computation along the scale axis (track extrema/zero-crossings up the ladder;
pair birth with death). Allocate cells by persistence, and **fix each cell's
scale to the rung at which its feature was born**, so a cell inherits a native
size instead of getting one from a tuning knob.

**Payoff.** A high-contrast single-pixel edge has near-zero lifetime and gets
near-zero priority *automatically* — the README's budget problem stops being a
special case. Cell scale becomes a measured property, not a slider.

**Test.** Priority map on an image with both a sharp film-grain region and a
soft object boundary. Grain should go dark, boundary bright. Then compare cell
counts spent on each region against the error-driven allocator.

**Cost.** One pass up the ladder with a small tracking table.

---

## 6. Rarity, not contrast — and inhibition of return in *feature* space

**Vision.** Pop-out is feature-space isolation, not intensity. One vertical bar
among horizontals is instantly visible although it has the same contrast,
energy, and error as its neighbours. Conversely, after you have seen one brick
you do not re-encode the other four hundred.

**Mechanism.** Build a small codebook (k-means or a hash, 64–256 entries) over
local Meyer band-signatures, augmented with the idea-4 orientation feature.
Priority = inverse density of a location's signature in that codebook. Then,
when a cell is placed, apply inhibition of return **in feature space** as well
as spatially: suppress all locations whose signature is near the one just
encoded.

**Payoff.** (i) A second, independent route at the orientation failure: even
with a weak feature, *rarity* in the joint distribution segments where energy
does not. (ii) Repeating texture is encoded once and reused — the largest
available compression win, and the honest reason human vision does not store
every brick. (iii) Naturally produces the "one cell per surprising thing"
allocation that our error-driven germination cannot express.

**Test.** Pop-out fixture (one oddly-oriented element in a field). The oddball
must receive a cell before any of the distractors do. Second test: cells spent
on a regular brick wall should be roughly constant as the wall grows.

**Cost.** One k-means over subsampled features at init; a nearest-centroid
lookup per pixel afterwards.

---

## 7. Completion fields as the barrier, so cells stop leaking through weak edges

**Vision.** Illusory contours (Kanizsa) and good continuation: the visual system
completes boundaries across gaps when the fragments are collinear/cocircular,
and treats the completed contour as real for the purposes of region formation.

**Mechanism.** Stochastic completion fields (Williams–Jacobs) approximated by
diffusion in the orientation-lifted space `(x, y, theta)` with ~12–16
orientation channels: advection along `theta`, small diffusion in `theta`,
implemented as shifted separable convolutions. Use the resulting completion
field — not the raw cartoon edge map — as the crossing barrier.

**Payoff.** Cells stop bleeding through the weak spot in an otherwise strong
boundary, which is the standard visible failure of every Voronoi-style
segmentation. Boundary strength becomes *contextual* rather than local.

**Test.** Kanizsa figure: the decomposition should form a triangle cell with no
luminance edge to support it. If we hallucinate the triangle, the barrier is
doing what V2 does.

**Cost.** 16 channels x a few sweeps of separable convolution; the most
expensive idea here, still bounded and FFT-friendly.

---

## 8. Cell parents from nodal domains of the anisotropic Laplacian

**Vision.** Scene structure is apprehended coarse-to-fine and globally: gist
before detail, whole before parts. Our hierarchy is the opposite — parents are
whatever geometry happened to nucleate first, and children are spatially nested
inside them.

**Mechanism.** Take the Laplacian of the *anisotropic geodesic metric we already
built* (idea 1's directed cost, idea 4's Q-tensor), and compute its first ~8
eigenvectors by Lanczos with a fast apply (our multigrid/FFT machinery is the
apply). The nodal domains of those eigenvectors are the coarse perceptual
regions, and their sign patterns give a **binary hierarchy of the scene**. Seed
the coarse underlayer from nodal domains instead of blue noise; blue noise then
only fills within a region.

**Payoff.** The hierarchy encodes objects rather than nesting order, so
"children may never leave the parent footprint" stops being a limitation and
starts being correct. Also gives a principled cell *count* per region (eigenvalue
gap) instead of a global budget.

**Test.** Compare the coarse underlayer against the flow-basin nucleation on
Pikachu: nodal domains should place a single seed per body part.

**Cost.** ~8 Lanczos iterations x a fast Laplacian apply. Once, at init.

---

## 9. Common fate from band phase (video)

**Vision.** Motion dominates every static grouping cue. Two dots moving together
are one object regardless of colour, shape, or proximity. MT solves the aperture
problem by pooling V1 normal-flow constraints.

**Mechanism.** No optical flow solver. Make each Meyer band analytic and take
the temporal phase derivative — phase-based motion (Fleet–Jepson) gives each
band, at each point, a *constraint line* in velocity space (normal flow only:
the aperture problem, stated honestly). Group by **intersection of constraint
lines** across bands and neighbours: a set of measurements whose lines meet near
a common point is one moving thing. Cells inherit that grouping.

**Payoff.** This is our own aperture-ladder argument in image clothes — each
band is an aperture, and complementarity across apertures decides what is
knowable. It reuses the transform we are fast at, and avoids the whole
variational-flow apparatus.

**Test.** Two overlapping transparent gratings (plaid): the grouping should
report a single coherent velocity when the constraint lines intersect, and two
when they do not — matching the human coherent/transparent switch.

**Cost.** One transform per band per frame; phase differences are pointwise.

---

## 10. T-junctions give an occlusion partial order

**Vision.** T-junctions are the classic non-accidental depth cue: the stem goes
behind the bar. From junctions alone you can recover a partial order of surfaces.

**Mechanism.** Detect junctions on the cartoon boundary graph (angular histogram
of incident boundary directions at each skeleton node), classify T vs Y vs X,
and emit a partial order. Then permit cells to **merge across an occluder** when
they are the two collinear arms of the same stem.

**Payoff.** The completion in idea 1 is metric and implicit; this is explicit
and can merge regions that are far apart geodesically. Together they give the
model a depth-ordered scene rather than a flat partition.

**Test.** A bar over a striped field: stripe cells on either side of the bar
should merge into single cells with amodal interiors.

**Cost.** Junction detection on a thinned edge map; small.

---

## Suggested order

Ideas 4 and 5 are the cheapest and each attacks a *measured* failure — do them
first, in a day. Idea 1 changes the metric and unlocks 8 and 10. Idea 2 changes
what a cell stores and idea 3 changes what "done" means; those two together are
the actual thesis of the model and should be A/B'd against the current fitter on
the same cell budget. Idea 6 is the compression story. 7 and 9 are the two
larger builds.

The honest framing for all of it: the current allocator answers "where is my
reconstruction wrong?" Human vision answers "where would a different scene have
looked different to me?" Those are different questions, and only the second one
has a stopping rule.
