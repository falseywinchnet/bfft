# Adaptive photonic transport field

The active proof renderer now includes BVH visibility, conservative
source-pyramid and camera-frustum queries, output-sensitive blocker sweeps,
sparse incoming/outgoing transport edges, and a source-forward residual march.
The measured million-object visibility benchmark and the bounded dual-tree
design needed to remove quadratic relation discovery are documented in
[SCALING_ARCHITECTURE.md](SCALING_ARCHITECTURE.md).

## Scene demonstrator suite

The native proof renderer also accepts `--scene standard`,
`--scene aperture-canyon`, `--scene mirror-relay`, and
`--scene occlusion-garden`.  These are not cosmetic environment maps.  Each
variant preserves the standard room, prism, cavity, beam, and material stack
while adding a different transport burden:

- `aperture-canyon` adds a second, violet rectangular emitter, seven staggered
  fins, and five diffuse receivers.  It demonstrates exact source-window
  subdivision when two broad emitters overlap through planar occluders.
- `mirror-relay` adds a cyan side emitter, three differently oriented mirrors,
  two colored diffuse receiver panels, and a rough-metal relay node.  It makes
  camera-terminal directional transport and sealed specular returns visible.
- `occlusion-garden` suspends 28 colored spheres and five flags in the room.
  Its many curved tangencies exercise the visible-edge field and show that
  primitive count alone does not determine evaluation cost.

All three variants remain below the default 96-primitive safety ceiling.  Their
closed camera paths reuse the collision-audited standard position spline but
retarget its gaze toward the relevant transport structure.  `--journey-position
P`, for `P` in `[0,1)`, exposes any point on that path as a still camera.  It is
mutually exclusive with `--animation-frames`.

The retained M4 Mini animation measurements are recorded in
[`demonstrator_scenes_m4.json`](demonstrator_scenes_m4.json).  At 640x360,
240 frames, and 30 fps, aperture canyon averaged 1.904 s/frame, mirror relay
2.117 s/frame, and the larger occlusion garden 1.263 s/frame.  The comparison
is instructive: the garden has 47 diffuse nodes and 1,491 sparse couplings, but
remains cheaper than the two-emitter scenes because source-window integration
and camera-terminal boundary overlap dominate its additional object count.

## Multi-regime proof scene

`native/regime_scene_native.cpp` is the larger hybrid proof renderer. It keeps
diffuse global transport in a persistent field while assigning directional
state only to material boundaries that require it. The scene deliberately
contains the following simultaneous regimes:

- a 19x19 collimated aperture represented as one `BeamBundle`, compiled into
  879 RGB spectral fibers and then clustered into 22 terminal caustic regions;
- a continuous rectangular soft emitter whose projected sphere conics and
  polygon silhouettes bifurcate its own two-dimensional source domain;
- diffuse, clear-coated glossy, ideal mirror, and rough-metal terminal
  responses;
- a thin smoke-glass sheet in the soft-light path;
- a closed triangular-prism volume with three rectangular sides and two
  triangular caps, channel-dependent IOR, absorption, entry/exit refraction,
  and internal reflection;
- a roofed narrow cavity whose teal target has zero measured direct energy but
  nonzero diffuse-field energy;
- a nearly edge-on thin baffle;
- a prism-beam caustic on the floor, with a mirror oriented from the reflection
  law so that the caustic reaches the camera through a
  specular--diffuse--specular chain.

The collimated source is not evaluated as hundreds of independent lights at
every surface. Its dense fibers are marched once through dielectric/mirror
boundaries. Their diffuse endpoints are clustered by receiver and position,
and those caustic regions become sources for the diffuse fixed point and camera
field. The prism is consequently a small internal transport bottleneck rather
than an RGB surface tint.

The rectangular emitter is no longer a 3x3 proxy. For each receiver point,
opaque and dielectric sphere silhouettes contribute exact cone cuts; rectangle
and triangle silhouettes are perspective-projected onto the emitter. Projected
polygon vertices and sphere-cone tangent extrema partition the vertical source
coordinate, and each resulting row is split again at its exact horizontal
silhouette roots. Gaussian quadrature measures the smooth kernel inside those
source-owned intervals but never determines where a shadow begins. This removes
the repeated hard silhouettes formerly exposed by the nine point samples.

The active evaluator first applies a conservative angular-disk rejection to
each blocker. Disjoint enclosing disks cannot remove a real source interval,
but avoid projecting geometry that cannot overlap the emitter. Silhouette roots
alone do not make an overlap region invariant: two planar blockers can exchange
front-to-back order while both remain present. Their ray depths are
rational-linear along an emitter row, so every pair contributes its analytic
depth-equality root to the partition. The ordered opaque/dielectric program is
then discovered once at the interval centroid. Each Gaussian ordinate
re-evaluates angle-dependent Fresnel against only those known faces, without a
whole-scene visibility walk. This removes the quadrature comb beside the thin
baffle without changing the transport energy.

The camera keeps the accepted scale-free scanline CONV terminal for smooth
radiance. Geometry is a separate sparse field. Rectangle and triangle edges are
projected through the pinhole as exact sensor lines; every sphere contributes
its exact camera-tangent contour. Those candidate contours are sampled below
one eighth of a pixel and retained only where first-hit tests show that the
owning primitive is actually visible. Primary and central mirror/refraction
endpoint changes found by the terminal pass need only one midpoint
classification: the result is consumed by a pixel-basin marker, so the former
twelve-step, 1/4096-pixel localization computed precision that was immediately
rounded away. Area-light blocker signatures are explicitly excluded: they are
illumination topology already represented by the compiled direct atlas, not
visible geometry.

Only pixel basins crossed by this visible-edge field are revisited. Four binary
coverage subdivisions give a 16x16 ownership lattice, whose quantum 1/256 is
below one 8-bit output level. The 256 tests are geometric only. Equal owners are
gathered and full radiance is evaluated once at each owner's subpixel centroid,
so edge accuracy does not multiply the rectangular-emitter solve by 256.

The retained visible-edge 1920x1280 M4 Mini proof compiles 29 analytic
primitives, 13 materials, 14 diffuse nodes, and 140 supported diffuse
couplings. It refines 20,392 edge-crossed pixels, 0.83% of the sensor, using
5,220,352 cheap ownership tests. With source-interval visibility fused and the
false five-copy metal lobe removed, the measured field compilation is 3.829 ms,
the guarded raster is 2,894.460 ms, and the complete run is 2,905.418 ms. The
faster 1,698 ms midpoint-transmittance candidate was rejected because it changed
the cavity's indirect energy from 0.023 to 0.018.

### Sparse evaluator checkpoint

The September 2026 evaluator pass moves repeated emitter work into reusable
surface and camera-direction fields. A diffuse surface atlas now caches one
value per unique fine-lattice coordinate: parent corners, cell centres, and
shared refined edges are never reintegrated. Spherical base charts use 33x17
samples and retain 5x5 refinement only in cells whose centre violates the
one-display-level interpolation certificate. This reduced direct-atlas source
evaluations from 70,353 to 48,354. Against the preceding 49x25 chart, only 111
pixels changed by more than one level, the maximum channel change was five,
and no pixel changed by more than ten.

The camera owns a separate directional specular field. Convex-sphere back
hemispheres are absent rather than treated as two-sided surfaces. Exact
collinearity also folds terminal rays transmitted by the thin planar glass
sheet back into that same field. The camera field requires 15,296 unique
relations and replaces 88,619 camera-aligned gathers; remaining exact
secondary specular integrations fell to 48,005. This is the backward fusion
case: paths with the same position-to-camera directional map share one
terminal operator even when one path includes a transparent sheet.

The current material-adaptive quadrature candidate uses order two for diffuse
source intervals, order four for clearcoat roughness at most 0.18, and order
three for broader metal/glossy lobes. On the 1920x1280 M4 Mini run it measured
2,296,016 raster quadrature ordinates plus 392,826 one-time camera-field
ordinates, down from 138,159,470 in the older repeated-source baseline. Cold
end-to-end time was 1,780.839 ms. Relative to the retained all-order-four
image, 1,366 pixels changed by more than ten channel levels, 43 by more than
thirty, and the maximum change was 34; consequently the adaptive-order rule is
an explicit speed/precision candidate, while lattice deduplication and
collinear backward fusion are structural reductions.

Further exact reuse now caches complete surface radiance within an optical
fibre, not merely its specular emitter term. A fixed four-probe table records
571,556 exact hits in the retained run. Visible contour sampling also stops
after the first successful witness for a primitive/pixel pair, and adjacent
pixel transitions use the single midpoint classification described above.
Together these changes reduced topology queries from 2,906,940 to 2,339,064,
visible contour witness writes from 147,812 to 20,275, and edge refinements
from 341,016 to 28,418 without changing a byte of the image.

CONV certification and synthesis now share one prepared current tensor.
Synthesis runs serially inside each already-parallel renderer worker, avoiding
a nested thread-pool dispatch. The prepared-profile result is byte-identical
and reduced the measured full run from 1,576.141 ms to 1,490.899 ms. Camera
topology no longer recasts a 3x3 emitter stencil at every visible sample: that
visibility was already compiled into `field.direct_atlas`, while the radiance
certificate remains responsible for rejecting a bad interpolant. This reduced
raster time to 964.263 ms and total time to 1,108.684 ms. Relative to the
pre-removal image, the maximum channel delta is two levels, only two pixels
exceed one level, and no pixel exceeds two.

Rough terminal hits are now retained and passed directly into spectral
propagation; metal/glossy terminal geometry and mirror-caustic diagnostics are
computed once per optical fibre rather than once per RGB channel. This stage is
byte-identical to its input and measured 956.858 ms raster / 1,090.266 ms total.
The next oversized representation is camera boundary classification, not
light propagation: 2,975,262 label-only subpixel ownership tests remain for
31,096 crossed pixels. Most belong to the generic 16x16 fallback. A trial that
splatted local sphere chords into neighboring pixels increased that work and
was rejected. The correct follow-up is analytic conic/polygon basin integration
or certified adaptive label subdivision only in already-crossed pixels.

The following SIMD pass separates immutable primitive invariants from the
authoring representation: rectangle Gram matrices, triangle edges, and squared
sphere radii are compiled once. Dense camera rays are transposed into scanline
SoA packets, and shape-specialized rectangle, sphere, and triangle kernels march
one primitive across independent sensor fibres. Clang's M4 vectorization report
confirms all three kernels at width four. The same packet is reused for 16x16
boundary ownership lattices and 8x8 topology filters; `restrict` applies only to
the disjoint numeric packet arrays, not the scene objects. Packet winners now
assemble their `Hit` directly from the retained distance instead of repeating
the scalar intersection. A mirror's first reflected hit likewise supplies both
the diagnostic and all three spectral marches. The full 1920x1280 result remains
byte-identical (`SHA-256 79efa0c5277ac353cb5e4f4afc4e5e3940cb1d6bfa354cb19ba2b7081e7d9d37`).
Across three warm M4 Mini runs, median raster time is 907.695 ms and median total
time is 1,041.505 ms; 5,397,248 camera-origin rays use the packet kernel.

Phase counters expose the next structural bottleneck: adaptive radiance takes
about 198 ms, visible-edge discovery 43 ms, and boundary reconstruction 605 ms.
Only 67,843 boundary radiance centroids are evaluated, but 2,310,146 boundary
samples recompute a full specular/dielectric topology signature. A fixed optical
packet stack was about 5 ms slower than the allocator-backed frontier. Packing
incoherent one-bounce rays for SIMD was also slower, because dielectric lanes
still require divergent continuation. A 5x5-certified quadtree cut topology
queries to 1,753,457 and boundary time to 512 ms, but changed 17 pixels by as
much as three channel levels, so it is not retained. The next exact reduction
must compile the dielectric topology-region boundaries and integrate their
coverage, rather than sampling or vectorizing millions of redundant signatures.

The prism caps now have outward windings: the bottom normal points down and the
top normal points up. Its former 3 cm air gap above the floor is reduced to a
1 mm numerical contact epsilon, keeping the bottom dielectric sheet hittable
before the coincident opaque floor. The collimated beam crosses side faces only;
the cap correction affects camera/mirror paths, which measured 56,657 bottom-cap
and 15,763 top-cap spectral events in the corrected full render.

Build, test, and run it on the M4 Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  make -C experiments/photonic_transport_field/native proof-test run-proof
```

The proof intentionally omits godrays and all other participating media. Its
diffuse field is currently facet/object-level rather than the source-exact
implicit relation field used by `source_region_native.cpp`; merging those two
representations is the next structural step. Rough metal now retains one exact
terminal reflection relation plus the continuously integrated rectangular
emitter lobe. This removes the former five coherent copies of reflected objects
and four recursive paths, but a compiled rough directional lobe remains future
work. Prism caustics use three spectral channels rather than a continuous
dispersion spectrum. The remaining red/white stair at the prism is not a first
surface ownership error or a beam-caustic edge. A nadir camera at
`(-2.193,5.70,-3.960)` exposes the feature directly on the top face. Fresnel
branch decomposition puts the arrow/stair wholly in transmission, while the
ordinary top-face reflection contains only the clean rectangular ceiling
emitter. Adjacent-pixel path enumeration identifies the dominant ghost as
`T(prism_cap_top) -> R(prism_cap_bottom) -> T(prism_cap_top) -> large_soft_light`.
At the red step the red channel still intersects the emitter after dispersion,
while green and blue have crossed its aperture edge; one row later all three
miss. The 1 mm contact epsilon currently makes the lower event a glass-to-air
Fresnel reflection even though the prism nominally rests on the floor. The next
physical repair is therefore a glass--floor contact boundary condition (or a
deliberately specified intervening medium), not camera filtering or additional
beam subdivision. Earlier front-view probes that found the floor--mirror path
were sampling a different bright part of the cap. These are explicit
approximations, not hidden material substitutions.

### Sealed optical feedback and prism camera orbit

Specular transport no longer has to rediscover a dielectric cavity through a
recursive call tree. The default `--optical-mode sealed` evaluator carries a
small residual-energy work list. A return to the same face after one
intervening face (`A -> B -> A`) with the same propagation direction identifies
an optical phase-state feedback relation. At
each return, transmitted branches surrender their energy exactly once to their
terminal, while only the reflected remainder stays in the loop. Its observed
contraction supplies a geometric upper bound on all energy still imprisoned in
the cavity. The relation is sealed once that complete bound is below the
default `1e-5` transport budget. `--optical-mode recursive` remains available
as a comparison oracle, and `--optical-cutoff` exposes the budget.

On the 640x426 nadir prism view, the sealed work frontier contained at most two
packets. Against the previous depth-seven recursive oracle, secondary optical
evaluations fell from 1,176,653 to 967,723 and rectangular-emitter quadrature
samples from 106,174,690 to 85,719,391. Total M4 Mini time fell from 2,486.043
ms to 2,006.185 ms. Only 61 of 817,920 output channel bytes changed and every
change was one level. The summed sealed residual bound was 0.166 over 189,607
evaluated camera samples, rather than an unreported recursive truncation.

Four retained oblique cameras, `prism-front`, `prism-right`, `prism-back`, and
`prism-left`, orbit the same prism centre at equal radius and elevation. The
branch diagnostic also emits `_bottom_emitter_ghost.ppm`, containing only
camera paths that reflect from `prism_cap_bottom` before terminating at the
ceiling emitter. This invariant relation becomes a clipped bright wedge from
the front, a broad quadrilateral from the right, split trapezoids from the
back, and narrow fragments from the left. Thus the former overhead stair is
not intrinsically stair-shaped: it is the rectangular source aperture, clipped
and spectrally displaced by different prism-face projections. The four-view
montage is `prism_orbit_sealed_montage.png`; exact measurements are recorded in
`regime_sealed_feedback_orbit.json`. The right and back cameras record no exact
phase-state return even though their isolated ghost remains strong: those rays
reflect from the bottom once and then escape through a different face. This
separates the two issues. Sealing feedback removes redundant higher returns,
but cannot remove the first physically permitted bottom reflection. Its
persistence across open and closed paths confirms that the glass--air contact
model, rather than recursion itself, creates the visible consequence.

### Oriented phase-space bundle A/B

The optional `--beam-mode oriented` compiler tests whether retaining bundle
orientation changes the visible defect. Fibres are grouped only
when wavelength, terminal receiver, and complete dielectric/mirror path agree.
Each group becomes one sheet carrying its terminal position and direction,
surface tangent frame, total spectral power, and a 4x4 covariance over the two
spatial and two angular tangent coordinates. The spatial marginal is an
oriented Gaussian whose covariance includes the same 0.14 transport blur as
the clustered baseline; it therefore preserves energy and matches the first
two spatial moments without retaining all centroid splats.

On the matched 1920x1280 M4 Mini run, this compressed 22 isotropic deposits to
3 wavelength sheets. Maximum spatial anisotropy was 1.062 and maximum
normalized position--direction coupling was 0.320. Cavity indirect energy
remained 0.023. The oriented render took 2,805.708 ms versus 2,734.859 ms for
the matched clustered run; the difference is small enough to regard as a
runtime tie, while beam compilation itself fell from 1.253 ms to 0.725 ms.
Only 12,761 of 7,372,800 output channel bytes changed, with mean absolute
difference 0.009831. The prism stair and tiny gold speck are visually
unchanged. Orientation is therefore a useful transport compression and a
necessary state for a future camera operator, but it does not solve this
artifact because the stair is an emitter-aperture ghost from the prism's lower
contact interface rather than a beam-field footprint.
The clustered renderer remains the default current-best image. Exact A/B
measurements are in `regime_oriented_bundle_comparison.json`.

## Current implementation: source-exact implicit regions

`native/source_region_native.cpp` is the active implementation. It does not
tile receiver surfaces. A geometric record is one directed source/receiver
relation together with the implicit silhouettes that can clip its source
aperture. The standard scene therefore stores 70 directed source regions and
162 occluder boundaries, not a spatial mesh.

For a planar source, each spherical blocker projects a quadratic cone onto
the source plane. At every integration ordinate the solver computes the exact
roots of those conics, unions the resulting source intervals, and tests only
the intervals themselves. The transport weight is evaluated conservatively as

```text
exact whole-source solid angle - integral over blocked source intervals.
```

Thus a visibility transition begins with exactly zero correction and grows
continuously. There is no receiver quadtree, terminal coverage cell, hanging
edge, or chart seam. Sphere receivers are evaluated directly from their world
position and normal; spheres are not unfolded into cube faces. Spherical
sources and blockers use their analytic angular disks.

The Gaussian rule integrates weights *inside* source-defined intervals. It
does not decide their topology: conic roots decide where a region begins and
ends. Increasing quadrature order therefore changes numerical accuracy, not
the geometry of the light field.

The transport state is one RGB residual per directed source region, not one
RGB total per object. For an incoming relation `r: a -> i` and an outgoing
relation `s: i -> b`, the reciprocal relation `reverse(s): b -> i` supplies the
outgoing region's sensitivity on surface `i`. Their overlap integral is

```text
C[s,r] = area(i) factor(s)
         integral_i density(r,x) density(reverse(s),x) dx
         / (factor(r) factor(reverse(s))).
```

Zero overlap creates no coupling. Each incoming column is capped at 0.9, so
the resulting 1-D gather/scatter operator is nonnegative and
column-substochastic. This is the missing locality condition: diffuse
reflection broadens direction, but it does not make an unsupported part of an
object glow. Illumination is the residual series

```text
q = e + R T e + R T R T e + ...,
```

marched over the saved region indices until its active energy threshold is
reached. The camera performs analytic primary intersection followed by
evaluation of the small set of incoming implicit regions for the hit object.

The 3200x2400 M4 Mini checkpoint stores zero spatial nodes: 70 source regions
plus 162 implicit occluder boundaries, an 890x reduction from the rejected
206,497-node receiver quadtree. It compiled 546 supported transport couplings.
Twenty residual generations reduced active energy from 977.533 to 0.002.
After compiling receiver-specific evaluator programs, storing blocker lists in
each relation, replacing hot-path allocations with fixed arrays, and adding a
conservative angular no-overlap test, timings were 9.075 ms to construct and
integrate the regions, 75.886 ms to compile the overlap operator, 0.012 ms to
march it, 1,561.743 ms to raster 7.68 million pixels, and 1,663.528 ms cold end
to end. The optimized image is byte-identical to the accepted checkpoint. Exact
production measurements are in `source_region_production_16x.json`.

`--profile-evaluator` enables thread-local counters without atomic hot-path
traffic. The measured scene performs 40.6 million relation evaluations: 25.3
million planar and 15.3 million spherical. Of 25.3 million plane evaluations,
21.0 million contain genuine source clipping and produce 79.3 million blocked
intervals. This makes interval integration, rather than relation dispatch, the
next optimization target.

The production interval rule remains eight-point Gaussian. An optional
`--interval-order 4` mode reduces the measured total to 1,513.403 ms, but alters
9,880 of 23,040,000 output channel values by at most two integer levels. It is
therefore an explicit speed/precision mode rather than the default.

## Terminal camera transport, backward fusion, and CONV

The production default makes the camera a terminal geometric receiver rather
than asking the transport evaluator to reconstruct every sensor sample from
scratch. There is no terminal spatial scale. `--terminal-error 1` selects the
adaptive field evaluator; `--terminal-error 0` selects the exact reference.
The residual march is still performed exactly once. Its complete depth history
is then folded backward into one coefficient per incoming source region:

```text
coefficient(r) = total_delivered(r) scale(r) / factor(r)

L_i(x) = emission(i)
       + albedo(i) / pi
         sum over r -> i [coefficient(r) raw_density(r,x)].
```

The 20-level propagation history therefore becomes a 70-term terminal
program; paths and duplicated depth states are absent from camera evaluation.
The camera applies CONV only after that RGB gather. Applying its nonlinear,
data-dependent interpolation independently to source paths would not commute
with their sum.

The active proof-scene camera begins each exact primary-surface scanline span as
one interval. Five exact controls provide the candidate native CONV field and
twelve independent residual witnesses determine whether that interval is valid.
A topology change or residual beyond the output-error budget bifurcates only
that interval. This remains the accepted smooth-field evaluator.

The rejected 2-D rectangular ownership experiment is retained only as a control.
It repartitioned the image rather than representing visible geometry and blurred
too much of the sensor. The active edge field instead projects analytic contours,
clips them by actual visibility, refines endpoint changes below a pixel, and
performs 1/256-quantized coverage only in basins crossed by those contours.

At 3200x2400, the scale-free evaluator starts from 7,849 primary-surface runs
and recursively rejects 213,435 candidate intervals. It performs 2,018,249
unique exact surface evaluations and synthesizes 4,222,743 visible output
pixels inside accepted leaves. The image remains within one 8-bit level of the
exact renderer across all 23,040,000 channel values; no pixel differs by two
levels.

Profiled relation evaluations fall from 40,614,482 to 16,023,594 (60.55%), and
integrated blocked intervals fall from 79,297,248 to 35,022,700 (55.83%). In
three alternating M4 Mini runs, median raster time falls from 1,697.682 ms to
1,439.447 ms (15.21%) and median end-to-end time from 1,783.038 ms to 1,522.951
ms (14.59%). Measurements and the final image are stored in
`terminal_transport_benchmark.json` and `terminal_adaptive_16x.png`.

Build and test the active operator on the M4 Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  make -C experiments/photonic_transport_field/native clean all test
```

`window_transport_native.cpp` is retained as the rejected receiver-quadtree
control. The older `photonic_native.cpp` and `lightfield_native.cpp` are also
rejected controls: they respectively pre-mesh surfaces and replace optical
windows with point-emitter splats. All three are excluded from the default
build.

This experiment begins a renderer whose persistent object is an adaptive
optical transport field, not a population of camera paths.

The proposition is viable, with one necessary refinement: a transport field
cannot always be a sparse graph of individual surface cells.  In an open room,
many surface pairs genuinely exchange diffuse light.  Deleting most of those
links changes the illumination.  Coherent far exchange must instead be stored
as a low-rank aggregate block, while visibility boundaries, corners, close
geometry, and angular/material discontinuities refine to smaller blocks or
exact links.

The geometric representation supports

```text
transport = coherent rank-one cluster blocks + exact uncertain leaf links.
```

Version 2 does not construct that complete potential field before solving.  It
discovers only the blocks relevant to the current residual light frontier.

It is diffuse-only and intentionally precedes camera rendering.  The purpose
of this stage is to establish the transport representation, conservation law,
and geometric scaling before primary visibility and glossy state complicate
the measurement.

## State and transport law

Let `q_i` be RGB power leaving surface cell `i`, `e_i` its emission, `rho_i`
its diffuse spectral reflectance, and `A[j,i]` the fraction of power leaving
`i` that reaches `j`.  The fixed point is

```text
q = e + rho * (A q).
```

`A` is nonnegative and column-substochastic in this convention.  Its missing
column mass is energy escaping the represented scene.  The implementation
rejects a centroid discretization whose outgoing fraction exceeds one; it does
not normalize an invalid geometry and thereby hide the error.

The multiple-bounce solution is the Neumann transport series

```text
q = e + R A e + R A R A e + ...
```

but no term is represented as a path.  The code marches one residual frontier:

```text
frontier -> gather -> block transfer -> scatter -> surface reflection.
```

Every depth reuses the same geometry hierarchy, but the active transport field
is rediscovered from the energy then present.  Nodes whose frontier is below
the certified error budget leave the active set.  Cycles are not duplicated
into a path tree; their diminishing residual simply returns in a later march.
The final ledger records absorbed, escaped, and unpropagated residual energy
separately.

The proposed one-dimensional forward line is thus real, but it is an operator
layout rather than a duplicated path list.  Rank-one blocks gather a cluster
to one RGB vector and scatter that vector to the receiver cluster.  A native or
GPU implementation can concatenate block factors and perform segmented
gather/scatter over fixed indices.

## Diffusion makes the field shrink

The earlier fixed-field formulation used the wrong complexity variable.  Let
`E_A^(d)` be the outgoing residual energy in source cluster `A` at depth `d`,
and let `U(A,C)` be a conservative upper bound on the fraction that can reach
an unopened receiver cluster `C`.  The complete receiver subtree is irrelevant
when

```text
E_A^(d) U(A,C) <= remaining incident-error budget.
```

No member link in that subtree is then constructed or tested.  The active size
is consequently

```text
S_d = number of cluster blocks whose deliverable-energy bound survives at d,
```

not the number of geometrically possible surface pairs.  Total propagation
work is proportional to `sum_d S_d`.  As reflectance, geometric dilution, and
escape contract the residual frontier, `S_d` eventually becomes zero.  In the
implemented march the final weak frontier is rejected at the root: one
cluster-pair visit, no blocks, and no visibility rays.

The pruning decision is collective.  Dropping every small leaf independently
would be wrong because many small recipients can contain important total
energy.  A cluster is removed only when an upper bound on the *whole unopened
subtree* fits inside one cumulative budget.  All pruned bounds are added, so
the sum of individually small omissions cannot silently exceed the requested
error.

If `rho_max < 1`, an omitted incident-energy bound `D` can contribute at most

```text
rho_max D / (1 - rho_max)
```

to all current and future outgoing diffuse energy.  Version 2 converts the
requested outgoing error to an incident budget with this relation and reserves
part of that budget for later diffusion depths.  Spending the entire allowance
at the first surface layer would perversely make later weak layers denser.

## Geometry compiler

Surface cells currently carry centroid, normal, area, bounding radius, RGB
albedo, and emission.  A binary spatial hierarchy is built by median splitting
along its longest coordinate extent.  Dual-tree traversal visits unordered
cluster pairs once.

A pair is aggregated only when all of the following hold:

1. cluster radii are small relative to their separation;
2. every member of each cluster faces the representative exchange direction;
3. a conservative swept-box beam hull certifies that every centroid-pair
   segment misses every spherical occluder.

The approximate Lambertian block is

```text
A[j,i] approximately equals
    cos(n_i, omega) * area_j cos(n_j, -omega) / (pi distance^2),
```

which separates into one source factor and one receiver factor.  If visibility
cannot be certified, the larger cluster bifurcates.  At leaves, one exact
centroid segment test decides the link.  Occlusion silhouettes consequently
create refinement without a global refinement schedule.

Coplanar equally oriented clusters are rejected by a support-interval
hemisphere certificate.  This is important: applying a leaf-level `front`
test alone still visits every pair on a flat wall and retains quadratic
construction cost even though none can exchange light.

The dense centroid compiler is retained only as an oracle.  It is not the
proposed rendering algorithm.

## What transfers from the repository

The implementation is a direct synthesis of already measured repository
ideas:

- FlowCells contributes tensor-conditioned population, coarse support first,
  interface-only refinement, and the principle that a causal graph is also
  the optimized assembly pattern.
- `denoiser/causal_ancestry.py` contributes the rule that accepted parents and
  barycentric fractions are the representation; paths need not be replayed.
- positive visibility ownership contributes nonnegative transported support
  and explicit accounting for unsupported regions.
- conservative multiresolution contributes the rule that coarse aggregation
  must preserve the physical zeroth moment and keep unresolved detail as a
  separate residual.
- reverse residual transport motivates an active illumination frontier rather
  than a fixed global bounce count.

The transfer is structural, not metaphorical.  Image ownership is not optical
visibility, and an image-space Eikonal metric cannot replace a 3-D occlusion
query.

## Surface color

A diffuse surface does not average its pigment into received RGB.  Its outgoing
reflected spectrum is

```text
reflected = albedo * incident.
```

Spatial/directional averaging occurs only when a coherent cluster or angular
basis represents a field.  Multiplication by reflectance is what creates red
or green color bleeding while respecting energy.  The focused tests include a
white emitter facing a red receiver to enforce this distinction.

RGB is sufficient only for Lambertian transport.  A glossy or specular node
must carry directional state, for example a sparse set of spherical Gaussian
lobes.  Then the same hierarchy stores blocks

```text
T[(receiver, outgoing_lobe), (source, incoming_lobe)],
```

and a mirror becomes a narrow angular routing operator rather than a false RGB
average.

The same limitation applies to source collimation.  Version 1 treats an
emissive patch through its diffuse cosine hemisphere.  A collimated luminaire
belongs in the directional source state, not in an RGB attenuation scalar.

## Complexity and the honest boundary

For `N` leaf patches and `S_d` energy-relevant source/receiver factors at depth
`d`:

- one frontier application costs `O(S_d * channels)` after discovery;
- transient transport storage is `O(S_d)`;
- total propagation work is `O(sum_d S_d * channels)`;
- residual depth is controlled by attenuation and tolerance, equivalently by
  the spectral radius of `R A`;
- favorable energy-gated dual-tree discovery is near `O(log N + S_d)` when
  upper clusters are rejected or accepted coherently;
- a bright adversarial frontier can still force `S_d = O(N^2)` at an early
  depth.

The last case is not theoretical.  A blocker through the center of a broad
beam makes the current conservative beam hull uncertain for many cluster pairs
and forces them toward leaves.  The experiment reports this loss of
compression rather than masking it.  A practical algorithm needs tighter
beam/frustum certificates, occluder hierarchies, and nested bases to make
silhouettes scale with boundary complexity rather than with all surface pairs.

## First measured checkpoint

The local NumPy reference run below is deliberately unoptimized.  Operator
error compares the hierarchical field with the quadratic centroid oracle;
fixed-point error compares residual marching with a dense linear solve of the
same compiled field.

| Cells | Scene | Dense compile | Field compile | Storage compression | Visibility rays | Operator error |
|---:|---|---:|---:|---:|---:|---:|
| 128 | open | 0.093 s | 0.0044 s | 16.0x | 0 | 2.63% |
| 512 | open | 1.539 s | 0.0183 s | 64.0x | 0 | 2.71% |
| 512 | central blocker | 1.475 s | 3.199 s | 5.69x | 23,552 | 1.19% |

Across these runs, five residual depths sufficed, maximum energy-ledger error
was below `5e-14`, and relative fixed-point error was below `3e-12`.  The open
case demonstrates the intended scaling.  The blocked case is the useful
failure: storage remains compressed, but Python compilation is slower than the
dense oracle because uncertain beams are refined without a coarse
fully-blocked certificate.

The corrected 128-cell energy-gated control requested an outgoing L1 error no
larger than `1.92e-4`:

| Scene | Active blocks by depth | Visibility tests by depth | Actual L1 error | Certified bound |
|---|---|---|---:|---:|
| open | 16, 16, 9, 0 | 0, 0, 0, 0 | `1.05e-5` | `1.35e-4` |
| central blocker | 72, 321, 148, 0 | 148, 820, 206, 0 | `2.57e-5` | `1.49e-4` |

The blocker initially expands the discovered field because its silhouette must
be resolved.  Diffusion then reduces 321 active blocks to 148 and finally zero.

## Run

```sh
python3 -m unittest experiments.photonic_transport_field.test_transport -v

python3 -m experiments.photonic_transport_field.run_experiment \
  --side 8 --out /tmp/photonic_transport_field_v2.json

python3 -m experiments.photonic_transport_field.render_standard_scene \
  --width 800 --height 600

make -C experiments/photonic_transport_field/native clean all test

experiments/photonic_transport_field/native/photonic_native \
  --width 3200 --height 2400 \
  --out /tmp/photonic_standard_16x.ppm
```

The standard scene is a Cornell-style open room with red, green, and blue
diffuse spheres and one broad overhead area source.  The camera interpolates
the solved outgoing radiance on 214 adaptive surface cells; it performs no
per-pixel secondary-light tracing.

## Subdivision theorem and native checkpoint

[`SUBDIVISION_THEOREM.md`](SUBDIVISION_THEOREM.md) gives the error criterion
used before the native renderer was written.  For source and receiver clusters
with radii `h_P,h_Q`, normal-cone deviations `eta_P,eta_Q`, and separation
floor `d`, it bounds the variation of the diffuse kernel over the complete
cluster product.  Multiplication by source energy and receiver area turns that
kernel variation into a delivered-energy error.  A cluster is accepted when
that error fits its allocated budget; otherwise only the uncertain child is
bifurcated.  A separate screen-space bound makes subdivision terminate at the
requested pixel resolution.  The executable randomized checks are in
`test_subdivision.py`; the native implementation repeats the kernel-bound test
under `--self-test`.

The native C++ checkpoint renders 3200x2400 pixels, exactly 16 times the pixel
count of the 800x600 reference.  The theorem selected 3,072 adaptive surface
samples for each sphere and 9,784 patches for the scene as a whole.  On the M4
Mini, the measured cold end-to-end run was 666.336 ms:

| Stage | Time |
|---|---:|
| subdivision decision | 0.366 ms |
| geometric field construction | 133.872 ms |
| nine-depth transport march | 84.904 ms |
| sphere-radiance lookup construction | 357.192 ms |
| 7.68-million-pixel raster | 71.461 ms |
| PPM write | 18.526 ms |

The requested outgoing-error allowance was 0.719 and the accumulated
certificate was 0.578.  Exact machine-readable measurements are in
`standard_scene_16x.json`.

This checkpoint deliberately distinguishes potential geometry from active
light.  It caches 11,364,273 directed geometric candidates once, while the
diffusion march admits an edge only while its source residual can deliver more
than the remaining error budget.  The next native geometry step is to retain
cluster blocks directly instead of flattening that cache; this is what removes
the large potential-edge list rather than merely avoiding repeated ray tests.

The controls require:

- form-factor reciprocity;
- compression of coherent open exchange;
- local refinement around occlusion uncertainty;
- residual-march agreement with a dense linear solve of the same operator;
- per-channel energy conservation; and
- spectral color bleeding through surface albedo.

## Next geometry gate

The next experiment should replace spherical blockers with a triangle scene
and build four connected pieces:

1. **Surface atlas.** Start from object/material patches; infer centroid density
   from curvature, normal spread, material variation, and visibility residual.
2. **Beam certificates.** Traverse a triangle BVH with cluster-to-cluster beam
   frusta.  Certify clear, blocked-by-one-occluder, or uncertain; subdivide only
   the uncertain case.
3. **Nested transport basis.** Reuse child-to-parent gathers across blocks
   rather than storing a fresh source and receiver factor in every block.
4. **Invalidation index.** Give each compiled block a beam volume and blocker
   witnesses.  A moving object's swept bounds query those volumes; only
   affected blocks and their ancestors are deleted and remarched.

Only after those geometry controls should a camera be attached.  Primary
visibility can begin with rasterized first hits or an adaptive projection of
visible surface cells.  The transport field eliminates repeated secondary
path work; it does not make the camera's first-surface visibility problem cease
to exist.

## Native camera journey

`regime_scene_native` can emit a raw RGB24 animation stream with
`--animation-frames` and `--animation-fps`.  The camera follows a closed
Catmull-Rom position/target/FOV curve reparameterized through an 8,192-sample
arc-length table, so frame spacing is uniform in scene distance rather than in
spline parameter.  The self-test samples 2,048 path segments and rejects a
journey that crosses scene geometry, leaves the safe vertical range, or lets
the target collapse onto the eye.

Frames are streamed directly to FFmpeg: no PPM sequence or full animation is
held in memory.  The web delivery render was produced on the M4 Mini with:

```sh
set -o pipefail
experiments/photonic_transport_field/native/regime_scene_native \
  --width 1280 --height 720 --terminal-error 1 \
  --animation-frames 1200 --animation-fps 30 2>journey.log | \
ffmpeg -y -loglevel error \
  -f rawvideo -pixel_format rgb24 -video_size 1280x720 -framerate 30 -i - \
  -an -c:v libx264 -preset medium -crf 21 -maxrate 3M -bufsize 6M \
  -g 60 -keyint_min 30 -pix_fmt yuv420p -profile:v high -level 4.0 \
  -movflags +faststart -threads 4 photonic_scene_journey_1200f_720p.mp4
```

The completed file contains exactly 1,200 decoded frames at 30 fps and is
40.000 seconds long.  It is H.264 High Profile, level 4.0, yuv420p, 1280x720,
3,988,950 bytes, and approximately 798 kbit/s.  Native rendering took
1,209.915 seconds (20 minutes 9.915 seconds) on the M4, averaging 1,007.365 ms
per frame: 212.770 ms adaptive evaluation, 34.254 ms edge discovery, and
697.622 ms exact boundary reconstruction.  The path length is 32.745 scene
units.  At playback speed the camera travels 0.819 scene units per second.

The large view-dependent spread is important profiling evidence.  The global
transport field is compiled once; most animation cost remains in terminal
camera-boundary reconstruction when a view exposes many overlapping analytic
edge signatures.  The animation therefore isolates the next optimization
target without coupling camera motion back into global illumination.
