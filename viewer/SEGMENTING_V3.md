# Segmenting version 3.0

Version 3.0 is a strict hierarchy rather than another refinement stage on the
canonical full-resolution partition.

## Transport audit

On a 1440 x 1799 cameraman demo, the former one-sweep causal path spent:

- about 6 ms emitting sites;
- about 550 ms preparing the full metric;
- about 2.78 seconds in the full-resolution causal front.

The front is expensive because 2.6 million vertices are accepted in strict
action order through a decrease-key heap, with multiple anisotropic simplex
updates per vertex. Population emission and Voronoi bookkeeping are not the
bottleneck.

Two discarded paths were removed:

1. With characteristic relaxation disabled, the pipeline still solved a
   restricted-grid front that was never consumed.
2. One optional characteristic pass could hide six rejected remarches.

The default is now zero characteristic passes. The optional control permits
at most one proposal/remarch.

## Surviving hierarchy

The implementation is `experiments/segmenting_v3.py`.

1. Resize the unchanged source to half scale.
2. Run one Meyer sweep there.
3. Populate and transport cartoon-only geometry. Texture and glass cannot
   manufacture cartoon cells.
4. Fit the cartoon at half scale and lift its owner IDs.
5. Upgrade the existing owner interfaces directly on the full-resolution
   unchanged-target metric. No owner is created or deleted.
6. Refit the upgraded cartoon once at full resolution and define texture
   exactly as target minus that refined cartoon.
7. Emit a curvature-limited full-resolution texture support population.
8. Assign every texture germ to exactly one cartoon parent, then transport
   only among sibling germs without crossing the parent.
9. Discard the parent identity. One reverse-residual pass splits hot texture
   cells locally, while adjacent pooled-affine model tests merge compatible
   flat texture IDs, including across the expired cartoon interfaces.
10. Fit an affine field and two paired one-sided ridges per cleaned texture
    microcell.

There is no full-resolution global front, candidate Meyer scoring, interface
proposal, owner-free diffusion, site relaxation, or texture intersection
graph.

The default owner upgrade is a fixed-sweep competition restricted to a
dilation of the lifted interfaces. Interior pixels seed their existing IDs;
an owner too small to retain an interior receives its closest owned pixel to
the original germ as a compulsory seed. Consequently the complete half-scale
ID set survives exactly. Full-resolution target edges guide the competition
but cannot emit a site or launch a global heap. A full-map germ-only refresh
is retained as an explicit geometry control.

## Cameraman result

For the 1440 x 1799 demo:

| Phase | Time |
|---|---:|
| Half-scale cartoon geometry | 346 ms |
| Half-scale cartoon transport | 1,038 ms |
| Cartoon fit | 12 ms |
| Full residual tensor | 172 ms |
| Texture affine fit | 42 ms |
| Four normal/tangent texture coordinates | 597 ms |
| Total | 3,358 ms |

The result uses 12,093 immutable cartoon cells and reaches 34.07 dB. The
one-sweep canonical pipeline used about 32,854 cells, took roughly 7 seconds,
and reached 35.82 dB on the same enlarged demo.

## Rejected experiments

Carrying the combined full-resolution population into the half-scale branch
created 32,855 cells. That was still a texture mesh and was rejected.

A literal product of global texture Voronoi IDs and cartoon IDs created
94,565 pieces: 7.8 fragments per cartoon cell, with median size 13 pixels and
10 percent only 2 pixels. It improved 32.09 to just 32.56 dB and was rejected
as patch manufacture.

Repeated splits on one cartoon-parent normal had diminishing returns and
produced fans across page glyphs and the cameraman tripod ground. It remains
available only as the parent-coordinate control.

## Interactive experiment

Run:

```sh
python viewer/segmenting_v3_app.py
```

The app keeps the texture topology and the older coordinate controls
independent:

1. **Texture topology** selects nested full-resolution supports or the former
   parent-cell ridge control.
2. **Nested ridge count** controls the bounded readout inside each texture
   microcell.
3. **Coordinate count** controls the older parent-cell ridge measurements.
4. **Axis schedule** compares repeated normal/tangent measurements with four
   literal directions: normal, tangent, and their two half-angle diagonals.
5. **Coordinate geometry** compares the straight cell tensor frame with
   owner-masked eikonal distance. The eikonal implementation uses a fixed
   number of four-direction causal raster sweeps over all owners at once. It
   has no per-cell queue and cannot cross a cartoon owner.

On the native cameraman, coffee, and Pikachu controls, four distinct straight
axes were respectively 0.021, 0.027, and 0.008 dB below repeated
normal/tangent coordinates at four slots. Owner-eikonal coordinates were
0.77, 0.45, and 1.26 dB below the straight paired basis. Additional eikonal
sweeps stopped changing the cameraman result after two. Thus neither
curvature nor angular coverage is presently the missing term; the app retains
both as explicit controls instead of making either an unearned default.

Raster-disconnected islands of a lifted owner have no legal path to that
owner's selected coordinate zero-line. The implementation reports their
count and uses the exact straight coordinate only on those pixels; it never
creates a path through another owner.

## Full-resolution owner upgrade result

With eight straight paired normal/tangent slots, upgrading the existing owner
map adds only 26--43 ms on the native controls. At radius 8 and edge strength
64 it improves PSNR by about 0.92 dB on cameraman, 1.13 dB on coffee, and
3.17 dB on Pikachu relative to the unchanged nearest-neighbour lift. Every
half-scale owner ID remains present. This stage is enabled by default.

The upgraded cells are then refitted once at full resolution before texture
exists. A complete target-affine replacement absorbs too much texture on
cameraman and coffee, so the retained default is a 0.5 blend with the lifted
cartoon. It improves the eight-coordinate result from 28.67 to 28.81 dB on
cameraman, 27.22 to 27.37 dB on coffee, and 27.22 to 28.54 dB on Pikachu.

The original BFFT cartoon metric is available as a normalized directional
term in the upgrade. Raw prolongation was rejected: its trace was roughly
1,000--5,000 and overwhelmed the full-resolution edge term. Increasing the
normalized direction control does make cells more elongated, but on the
three controls it traded away reconstruction accuracy. It therefore remains
an honest straight-cell geometry control with a zero default rather than an
unearned production setting.

Version 3 gives glass zero support weight. The former glass-sweep control was
therefore mathematically inert while its ROF projection still ran. The
control has been removed, and zero glass weight now bypasses that projection.

## Nested texture result

Printed page exposes why the former parent-coordinate model failed. V2 emits
2,904 full-resolution cells on that image, while the half-scale v3 cartoon
has only 418 parents. V2's interface and soft-support proposals are both
rejected; its letters survive because full-resolution texture participates in
population and one ridge acts only at glyph-cell scale.

The nested v3 branch reproduces that mechanism without multiplying global
texture IDs by cartoon IDs. It emits 3,082 texture microcells, assigns each
exactly one of the 418 parents, and constrains sibling transport to that
parent only during construction.

The subsequent cleanup is deliberately not an iterative region optimizer.
The existing predecessor forest carries affine residual energy backward once
and emits every eligible two-child split simultaneously. For merging, each
adjacent pair computes the increase in exact pooled affine SSE. That increase
must fit within the robust residual-variance allowance for the coefficients
removed. Every eligible cell contributes its best edge to one simultaneous
component graph, allowing an obvious smooth patch to collapse beyond a single
disjoint pair without a greedy candidate queue.

With the current affine component merge, page finishes at 2,353 texture IDs,
cameraman at 7,194, coffee at 5,034, and Pikachu at 1,344, after beginning
from 3,082, 8,752, 7,221, and 1,132 construction cells respectively. Hot-cell
splits are included in those final totals.

At native control sizes the additional full-resolution geometry and nested
transport cost roughly 30--135 ms each. Glass remains absent: texture support
is the population term that preserved page glyphs, while glass added little
population and required another ROF projection.

## Flat-texture speed pass

The accepted cleanup exposed one unnecessary exact remarch and one unnecessary
runner-up field:

1. The nested construction now uses a first-owner monotone bucket walk. For
   nonnegative source-independent graph costs, a source that loses at a vertex
   cannot win later by following the same path suffix. The first owner,
   distance, and predecessor forest therefore match the former two-label walk
   exactly; only the unused runner and second distance are omitted.
2. A residual split has exactly two children inside one established texture
   cell. Their squared-distance difference under a frozen 2-D metric is a
   linear half-space test. The default evaluates that paired metric directly
   at each incident pixel instead of launching a constrained Dijkstra. The
   former local-eikonal remarch remains in the viewer as a quality control.
3. Best-neighbour merge nominations use scatter reductions and one union-find
   edge scan rather than a Python adjacency loop or iterative region descent.
4. Nested mode reuses its already-computed boundary tensor for the cell frame
   instead of calculating a second residual Sobel tensor that no fitted basis
   consumed.
5. Exactly achromatic RGB targets filter only OKLab lightness. The two
   machine-scale chroma fields contain no source information and previously
   caused twelve redundant image passes.

Warm native runtimes fell from approximately 161 to 127 ms on page, 708 to
511 ms on cameraman, 605 to 397 ms on coffee, and 421 to 339 ms on Pikachu,
including the stronger affine component merge and interface correction.
The viewer exposes split metric strength and the legacy remarch so the small
speed/geometry trade remains measurable.

After split/merge, a single one-pixel interface band is refreshed only where
both full-resolution boundary confidence and incident residual ratio are
high. This addresses contours such as dark hair against a fine striped wall:
the nested cartoon parent may place the initial interface coarsely, but the
flat texture IDs can now move that interface instead of merely splitting or
unioning immutable pieces.

## SciPy-free execution path

Version 3 no longer calls or eagerly loads `scipy.ndimage`, `skimage`, or the
legacy SciPy-backed transport viewer. Gaussian smoothing, Sobel derivatives,
cross dilation, and anti-aliased linear resize use cached compiled kernels in
`port_needed/fast_image_ops.py`. The resize agrees with the former
`skimage.transform.resize` result to floating-point roundoff, including its
whole-sample reflected boundary.

These Gaussian supports are only 7--25 taps wide. A direct batched separable
pass moves less data than padding every row and column to a power of two,
calling the public one-dimensional BFFT, multiplying spectra, and transforming
back. BFFT convolution becomes the appropriate alternative when the support
is wide enough to amortize those transforms; it is not the fast path for the
current local tensor scales.

## Canonical-v2 structural quotient

The first hybrid reused the exact same curvature-limited measure and
deterministic quantizer phase for both layers. Below the 512-pixel allocation
restriction and both safety ceilings this collapsed the hierarchy: Pikachu
received 1,046 structural germs and 1,047 texture germs, the first 1,046
centres were bit-identical, and the two label maps agreed at 99.983% of
pixels. Before cleanup the nominal texture layer bought only 0.076 dB. The
large Golden Gate control looked better only because grid restriction and the
32k structural ceiling accidentally decorrelated its two populations.

The retained operator is now a literal population-measure decomposition. The
already-frozen Meyer cartoon, texture, glass, source reliability, and boundary
fields are reweighted without a second Meyer or ROF solve. A cartoon-only,
curvature-limited submeasure commands structural germs. One canonical
characteristic step relaxes those germs on the full composite metric. Each
relaxed structural germ is also its parent's first texture germ; only the
nonnegative remainder of the full support measure is quantized into surplus
detail germs with a distinct fixed phase:

```
rho_full = rho_structural + rho_surplus.
```

If the structural safety ceiling clips its commanded mass, only that
commanded mass is subtracted. Unrepresented structural demand therefore
returns automatically to `rho_surplus` instead of disappearing. At the
default 0.8 structural mass, native page, cameraman, coffee, and Pikachu use
1,759, 5,446, 4,550, and 704 structural germs. Their independently emitted
surplus populations are 1,165, 2,402, 1,834, and 361 germs.

Cross-structural texture merging and the post-cleanup hot-interface remarch
are now controls rather than defaults. On Pikachu the latter moved 9,634
pixels and made every tested ridge readout worse. The direct surplus
construction with one characteristic step and three measured normal ridges
reaches 34.26 dB at 1,357 final texture cells, compared with 32.62 dB at 1,415
cells for the aliased hybrid. The 1440 x 1799 Golden Gate control constructs
77,328 initial and 80,388 final cells at 27.45 dB; its structural ceiling is
accounted for rather than silently deleting roughly 22k cells of support
demand.

This experiment deliberately exposes two ID views. `Structural soft IDs`
renders the persistent v2 quotient through the owner-free diffusion cover;
`Texture micro IDs` shows the subordinate residual cells. The old half-scale
cartoon scaffold remains selectable as an A/B control.

## Transport fast path

The full-resolution readout now defaults to an exact monotone-bucket graph
front when the structural sites were emitted on a restricted allocation grid.
Small images, where allocation and readout share a grid, retain the continuous
Eikonal control. The viewer exposes `Automatic`, `Continuous control`, and
`Full bucket graph` explicitly.

This is not a reduced-iteration approximation. The bucket queue retains the
same first-owner shortest paths, predecessor forest, input-derived queue
geometry, and deterministic insertion order as its Numba reference. The
native first-owner kernel is bit-exact with that reference and omits the
runner-up state that the v3 hierarchy never consumes.

A characteristic site's circular core is also required to fit inside its
average allocation cell. When

```
allocation_pixels / structural_sites < pi * core_radius**2,
```

the coarse grid cannot represent the proposed core. Version 3 now skips that
coarse preparation and front instead of computing a characteristic candidate
that will be rejected.

Finally, the eight fixed edge families are streamed natively from the frozen
float32 precision and boundary tensors. Their output is bit-identical to the
former NumPy construction, but it avoids three full float64 metric images and
large arithmetic temporaries. On the 1365 x 2048 natural-image control this
put structural transport at about 0.59 s and nested texture transport at about
0.54 s, with unchanged 27.3767 dB reconstruction and 78,454 final cells. The
whole build measured about 5.56 s; frozen-geometry construction, rather than
transport, is now the dominant phase.
