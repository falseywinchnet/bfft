# Retained scene edits and source response

The subsequent [camera audit](CAMERA_DEMAND_GATHER.md) retains the mechanisms
below and reduces the 800×600 edited-frame times to 362–424 ms. The timings in
this note preserve the earlier baseline.

The September 5 work continues the existing `native/regime_scene_native.cpp`
engine. The governing destination is a CPU-rendered waterfall and grass scene,
with dynamic insertion, deformation, a freely moving observer, **at least
800×600 pixels**, and a complete **33.33 ms CPU frame**. This update establishes
and measures engine mechanisms; it does
not claim that the final demo or the frame-time requirement is complete.

The two preceding conversations were **Design photonic transport tracer** and
**Design photonic transport tracer (2)**. The first established scene-compiled
retained radiance transport. The second diverged into the standalone
`operation_pink_floyd` renderer. Its outputs do not demonstrate integration with
this engine. All results in this note exercise the existing geometry, beam
compiler, retained transport, area emitters, optical evaluator, and camera.

## The retained state is an operator and its current response

Let the current finite diffuse field satisfy

\[
L=b+KL,\qquad K=D T,\qquad b=D E,
\]

where `T` is the compiled incident-radiance transfer and `D` is the receiving
material response. After an edit, inject the signed defect

\[
r=b'+K'L-L.
\]

The correction is

\[
\delta L=\sum_{j=0}^{\infty}(K')^j r.
\]

This accounts for light removal as well as addition. Clamping the correction to
nonnegative values, or deciding activity from the signed sum of RGB channels,
would lose valid subtraction. The native retained block application already
accepts signed coefficients; the renderer now carries them through its entire
residual march. New node values start at zero, while existing values are mapped
by stable primitive IDs. Deleted transport nodes disappear from that mapping.

For the actual fused operator, let

\[
q=\|K'\|_1=\max_{s,c}\sum_r D_{r,c}T'_{rs}<1.
\]

The remaining correction has the bound

\[
\|\delta L_{\rm remaining}\|_1\leq \frac{\|r_{\rm remaining}\|_1}{1-q}.
\]

For a rank block `v u^T`, compute its column sums using
`u_s * sum_r(D_r v_r)`. No source/receiver pair expansion is needed to bound the
fused operator. The code checks this operator's bound, not the unfused CSR
surrogate. It stops at a numerical L1 tail below `1e-8`, then independently
reapplies the operator to check the final fixed-point defect. This is an exact
arithmetic tail theorem evaluated in double precision, not an interval-arithmetic
roundoff proof or a certificate of the renderer's physical discretization.

The packed native plan now stays alive with `TransportField`. Changing geometry
currently recompiles the fusion plan; retaining the plan's lifetime does not
mean its topology has become incrementally mutable.

## Keep the dependence on a changing source

The existing participating-volume model applies an affine action to each
spectral channel:

\[
I_{\rm out}=a I_{\rm in}+c J.
\]

Here `a` contains boundary transmission and extinction, and `J` is the volume's
registered source radiance. Composition preserves affinity. Area integration
also preserves it, so a registered irradiance sample has the representation

\[
E(x,J)=A(x)+B(x)J.
\]

The compiler now integrates `A` and `B` in the same traversal. The ordinary
source-valued integrator remains available and is checked independently against
the affine evaluation at zero and two nonuniform source spectra.

A source-radiance change therefore updates a retained sample with a channelwise
multiply-add. Geometry changes invalidate coefficients only where the old or new
primitive could enter the sample's numerical source partition. Adaptive cell
refinement is rerun from the updated values, and newly required samples are
constructed. Old refinement decisions are not silently reused with new light.

This is a small instance of the broader representation needed for the demo:
retain a geometry-dependent response to source signals, so changing a signal
does not trigger geometry integration. It is currently implemented for the
engine's existing single registered participating source; a general source basis
and source-region representation remain further work.

## Geometry, dependency ownership, and observer validity

`RetainedSceneState` owns the scene, beam field, and transport field together.
Its `apply_geometry` method accepts replacements and additions. Callers observe
const geometry and lighting. An empty edit preserves the field identity. Failed
field compilation makes the owner unavailable rather than publishing old
lighting over edited geometry. Updates are synchronous; concurrent snapshot
publication has not been implemented.

Primitive IDs remain stable. Replacement refreshes sphere radius/area,
rectangle Gram data, or triangle edges/normals before use. Existing BVH leaves
and their deduplicated ancestor union are refitted in reverse preorder. New
primitives remain directly queryable outside the old root through a delta list.
Once that list exceeds 32 primitives, the hierarchy is rebuilt. This is correct
buffered insertion, not a worst-case constant-time insertion guarantee.

The median tree's leaves contain at most four primitives, so reserving `N`
nodes is sufficient. The previous reservation was `2N`. Parent links and a
primitive-to-leaf map add retained metadata; the smaller reservation offsets
that cost for a newly built tree. Bounds still use doubles and primitive storage
is still the existing large tagged representation.

Changed relation tests retain the hull of each primitive's old and new bounds.
For linear motion that hull also encloses the intermediate geometry. For curved
motion it certifies the two evaluated states, not arbitrary unsubmitted motion
between them. Raw transport coefficients are retained before the existing
source normalization. Every pair, including an old zero pair, is considered for
reuse; a removed blocker can open a relation with no previous stored edge.
Normalization is recomputed across the complete outgoing source column.

The source partition now filters individual primitive bounds after querying
BVH leaves. Otherwise an unrelated primitive sharing a leaf could change
quadrature bands, making the numerical lighting depend on acceleration-tree
grouping. Sample invalidation preserves the numerical partition as well as
physical visibility: an otherwise unnecessary geometric cut can still affect a
finite Gaussian quadrature rule. A physically tighter pyramid rejection alone
failed the cold-compiler comparison and was not retained as the update rule.

The viewer-origin specular atlas now includes the retained field's generation.
Rotation about the same origin may reuse it while the field is unchanged. A
geometry edit rejects the old atlas even if the camera has not moved. Translation
still rebuilds directional observer state; a persistent spherical observer field
has not yet been implemented.

## Reproduction and checks

Run from the authoritative MacBook checkout:

```sh
sh experiments/photonic_transport_field/run_retained_updates.sh
```

The script uses `m4build` and `m4host`, builds and executes on the M4 Mini, runs
both native suites and all 38 Python tests, runs the standard-scene update and
large-geometry benchmarks, runs the update suite under ASan/UBSan, and copies the
JSON/PPM records back. Both the C++ geometry/transport and the linked C CONV
dependency are instrumented in the sanitizer run. Sanitizer compilation reports
expected failures to honor three
existing vectorization requests; the optimized build is warning-free.

The focused tests include motion, skewed rectangle updates, disabling a blocker,
new diffuse nodes, insertion outside the old root, delta compaction, invalid
input, an empty scene, finite ray lengths, no-op ownership, newly opened
previously zero transport edges, signed light removal, affine source response,
and BVH/linear/fresh-rebuild equivalence. The end-to-end benchmark uses a stale
viewer cache deliberately, and checks the edited camera images against fresh
scene compilation.

## A finite source window without projecting vertices through a horizon

A planar convex blocker has vertices `v_i` and a receiver point `p`. For each
edge, orient

\[
h_i=(v_i-p)\times(v_{i+1}-p)
\]

toward the blocker's interior. Every direction meeting the blocker satisfies

\[
h_i\cdot(q-p)\geq 0.
\]

An emitter point has the chart `q=o+u U+v V`, so every inequality is linear in
`(u,v)`. Clip the unit source square by these half-spaces. If `n` is the blocker
normal and `d=n·(v_0-p)`, also impose

\[
\operatorname{sign}(d)\,n\cdot(q-v_0)\geq 0,
\]

which restricts the intersection to the finite receiver-to-emitter segment.
This avoids dividing each blocker vertex by its distance to the receiver's
projection horizon. The resulting convex polygon supplies the existing source
row cuts and depth-order events. No per-pixel cosmetic term is introduced.

Independent ray/primitive checks cover a blocker spanning that horizon, with
three receiver positions and 1,599 source points per receiver. A separate dense
source visibility probe exposed the original chart failure: after making BVH
candidate filtering independent of leaf grouping, missing projected blockers
could make the cavity roof incorrectly receive direct light. The half-space
construction repairs that representation failure.

The corrected window also resolves a very small *real* direct contribution to
the historically named `indirect_only_target`: the floor emitter is visible past
the right edge of the receiver card. A 192×192 per-ray source probe measured
roughly `5.84e-5`, `3.92e-5`, `5.87e-5` in its three irradiance channels. The
old zero assertion missed that opening. The proof test now checks that this
specimen remains overwhelmingly indirect; an additional fully shadowed receiver
with a separately illuminated relay preserves the strict zero-direct,
positive-indirect transport invariant. No existing scene geometry was moved to
conceal the visibility finding.

## Measured September 5 M4 Mini results

The minimum-resolution record is
[`retained_updates_800x600_m4.json`](retained_updates_m4/retained_updates_800x600_m4.json).
It renders the existing standard scene at 800×600 through the retained backend
and the benchmark helper's default BVH traversal,
with the existing terminal-error setting of 1. These are three checked update
steps, not a sustained interactive frame-time distribution. Each frame cost is
the sum of measured geometry maintenance, beam/lighting recompilation, and
camera rendering, including reconstruction of invalidated viewer state. Oracle
compilation, comparison, and file output are excluded; display presentation and
input processing are not implemented by this benchmark.

| Edit | Geometry (ms) | Retained lighting update (ms) | Fresh lighting compile (ms) | Camera rendering (ms) | Engine frame (ms) |
|---|---:|---:|---:|---:|---:|
| First sphere move | 0.0030 | 77.54 | 478.12 | 564.00 | 641.54 |
| Second sphere move + translated observer | 0.0019 | 76.07 | 480.46 | 560.42 | 636.49 |
| Sphere insertion + translated observer | 0.0009 | 134.11 | 548.90 | 569.36 | 703.47 |

Motion lighting compilation is 6.17–6.32× faster than fresh compilation;
insertion is 4.09× faster. The two moves reuse about 91.5% of requested atlas
construction samples; insertion reuses about 88.0%. All three edited images
are **byte-identical** to fresh compilation of the same edited scene and camera.
That establishes update equivalence for these cases, not continuous physical
accuracy. The fixed-point differences are checked against the sum of the warm
and cold tail bounds: neither result is assumed to be the exact fixed point.

The signed residual requires seven waves versus nine from zero, but its
0.007–0.008 ms wall time is slightly greater than the 0.006 ms cold march here.
The small transport matrix is not the current bottleneck. The substantial
measured saving comes from reusing geometric source responses, not from claiming
that two fewer waves materially accelerate the frame.

The minimum-resolution frame takes **636–703 ms (1.42–1.57 fps)**, approximately
**19–21× over the 33.33 ms budget**. The waterfall/grass/free-flight goal remains
open. At 96×64 the same three frames still take 390–449 ms; reducing pixel count
by 78× does not remove the large observer/source construction cost. The next
priority is retained directional and geometric response, alongside removing
quadratic relation discovery before adding many grass primitives.

The separate geometry-only record is
[`retained_geometry_m4.json`](retained_updates_m4/retained_geometry_m4.json).
Each case uses nine repetitions with refit/rebuild ray-hit comparisons:

| Scene size | Edited primitives | Median refit (ms) | Maximum refit (ms) | Median full rebuild (ms) |
|---|---:|---:|---:|---:|
| 100,000 spheres | 1 | 0.001959 | 0.004917 | 18.39 |
| 100,000 spheres | 64 | 0.049417 | 0.056792 | 18.40 |
| 100,000 spheres | 1,024 | 0.569666 | 0.577250 | 18.58 |

These timings measure geometry maintenance only; no 100,000-node transport
field was compiled or rendered. They do not establish an equivalent dense grass
frame rate or worst-case latency after repeated large deformations.

The source-window correction intentionally changes some static lighting relative
to the pre-edit engine. The retained 160×100 static control has maximum channel
difference 28/255 and mean absolute channel difference 0.1734/255, with 4,880
pixels having at least one changed byte. See
[`static_comparison.json`](retained_updates_m4/static_comparison.json) and the
paired PPM/JSON files. This is therefore not a claim of byte preservation against
the old renderer. Independent clipping/ray tests and selected dense source
probes support the corrected visibility; they do not certify every pixel's
quadrature error. Files under `diagnostics/` preserve the rejected intermediate
candidate-filter experiment and must not be read as final output.

## Remaining work toward the waterfall and grass demo

The current operator still contains the earlier renderer's object/facet-scale
diffuse field, source-column normalization, three broad spectral channels, and
rank-block approximation with a 5% local coefficient admission tolerance. The
new residual bound certifies convergence of that finite operator. It does not
certify continuous spectral transport, every thin feature, every indirect-light
gradient, or the physical energy interpretation of that normalization.

The legacy rectangular surface atlases also remain. Keeping their construction
samples is a useful incremental repair, but it is not completion of the original
source-defined continuous-region representation. The sample map now retains two
RGB coefficients per entry and therefore trades additional persistent memory for
less integration. Full region-owned source responses should eventually replace
that cache. Pair admission still visits all candidate surface pairs, even when
most existing weights can be reused. The cold compiler remains quadratic in
surface-node count, and fused blocks are rebuilt after edits.

The next engine changes should preserve the demonstrated math while removing
those costs:

1. Build a hierarchy over **transport supports**, with source/receiver normal
   cones, energy envelopes, and visibility ownership. Discover block relations
   directly, and index both visible and occluded relations so insertion and
   disocclusion find the affected frontier without an all-pairs sweep.
2. Retain **responses on geometric regions**. Extend the implemented affine
   source response to the incident signals actually required by each receiver.
   Split a region where ownership or the response certificate changes; collapse
   it only when its observable response stays within the declared error.
3. Retain the observer's **world-direction visibility and optical state**. At a
   fixed origin `O`, `I(O,R,p)=F_O(R d(p))`; camera rotation changes the sampling
   chart. Translation requires fresh certificates for parallax, disocclusion,
   and directional response. There is no guarantee that a small translation
   changes only a small screen region near a close occluder.
4. Give grass and water **owned geometric state**. Grass patches can share
   shape data and carry local deformation parameters; a water surface carries
   time-dependent geometry and normals. Their geometry updates feed the same
   invalidation and transport paths, rather than substituting an image effect
   for water or blades. Repeated geometry needs an instance hierarchy and local
   rebuild policy before the buffered insertion scheme can scale cleanly.

The scene-signal/shader boundary is explicit: physical geometry and material
state determine intersections, visibility, normals, and transport. Optional
stochastic shader effects consume those registered signals and live above the
illumination engine. Such effects cannot fabricate geometric occlusion or be
counted as a substitute for the waterfall's optical response. The current work
adds no stochastic appearance effect. A later WebGPU port must preserve these
ownership and response boundaries; it is not part of this CPU measurement.
