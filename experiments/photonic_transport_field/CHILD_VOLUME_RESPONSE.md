# Child-volume response system

## Boundary contract

The parent scene does not contain the child's internal geometry. It contains a
directional boundary interface with the response

\[
L_{\partial V}^{\mathrm{out}}
=\mathcal R_V[L_{\partial V}^{\mathrm{in}}]+E_V.
\]

`BoundaryFiber` is one spatial/angular boundary region. It stores a position,
outward normal, central propagation direction, area, and solid angle. Every
geometric boundary region is represented twice by role:

- an ingress fiber points into the volume and can only be a transport source;
- an egress fiber points out of the volume and can only be a terminal receiver.

The duplication is semantic, not geometric subdivision. It prevents light
which has already left the child from re-entering through the same boundary
cell. The parent can observe only the compressed egress ledgers.

## Private transport

`ChildLink` connects ingress or internal sources to internal or egress
receivers. Each link stores a geometric transport fraction, distance, geometry
label, and `VolumeMedium`. For a mode family

\[
\ell_m=(w_m,\mu_m,\sigma_m^2,\theta_m,p_m,\chi_m),
\]

one link transports

\[
w'_m=g_e w_m\exp\!\left[-\rho_e d_e\left(
\sigma_{t,e}+\kappa_{L,e}p_m+\kappa_{C,e}|\chi_m|\right)\right].
\]

while \((\theta_m,p_m,\chi_m)\) remains invariant.

An internal receiver then applies one local kernel:

- identity: population attenuation only;
- diffuse: extinguish the incident family and generate an isotropic family;
- Mueller: use transient Stokes arithmetic to select/generate a new outgoing
  family, never to rewrite the incident family.

Internal emissions are already outgoing families. Internal feedback cycles are
integrated by the ordinary residual march until their transported population
falls below the declared cutoff. Egress has no outgoing adjacency, so the
cycle is sealed exactly at the child boundary.

## Response cache

For fixed child geometry, materials, and media, the system caches

\[
\mathcal R_V[e_i\otimes m],
\]

the response to one unit population in ingress fiber \(i\) and mode family
\(m=(\mu,\sigma^2,\theta,p,\chi,\text{provenance})\). Power is deliberately
absent from the key. A later intensity change scales the cached response and
performs no child march. New modes or newly illuminated ingress fibers compile
only their missing unit responses. Exact packet coalescing compresses each
egress ledger. Signed negative corrections use the same unit response as
positive population and therefore propagate topology invalidation through the
child without compiling a second kernel.

`E_V` is cached separately. In a parent/child coupled residual solve it is
injected into the parent once. Every later feedback visit evaluates only
\(\mathcal R_V[\Delta L]\), preventing phantom repeated emission.

## Parent coupling and invalidation

`ChildIngressBinding` consumes a declared fraction of one parent node's
frontier and assigns it to a child ingress fiber. `ChildEgressBinding` maps a
child egress fiber back to a parent node. Their gains participate in the same
unit outgoing-energy budget as ordinary parent links.

The coupled solver marches parent links and child responses in one residual
queue. Therefore a parent-to-child-to-parent feedback loop is not enumerated as
a path; it emerges from repeated application of the two sparse operators.

Invalidation is bounded as follows:

- changing only incident power needs no invalidation;
- a newly occupied mode/fiber pair adds one cached unit response;
- explicitly invalidating selected ingress-response entries evicts only those
  cached mode responses;
- changing internal geometry, a material kernel, or a medium installs a new
  topology revision and clears only that child's kernel and emission response;
- changing parent bindings rebuilds the cheap coupling index but does not
  invalidate an unchanged child response.

## Geometry dogfood

`compile_child_volume_topology` creates temporary, invisible centroid patches
for ingress and egress fibers, combines them with hidden `ChildSurface`
facets, and invokes the repository's existing visibility compiler. It then
retains only legal role-directed links. Boundary proxies and private surfaces
are never inserted into the parent scene.

`compile_child_volume_topology` still materializes the dense centroid oracle as
a small-domain correctness bridge. `compile_child_volume_rank_blocks` invokes
the hierarchical compiler and retains its role-filtered rank-one blocks
directly. It never calls `dense_matrix` and never expands a coherent block into
pair links.

## Retained rank blocks

A retained block is

\[
B=(S,R,u,v,d,\mathcal M),
\qquad T_B=v u^\mathsf T.
\]

For one residual wave the evaluator performs

\[
G_B=\sum_{s\in S}u_sL_s,
\qquad \widetilde G_B=\mathcal M_d[G_B],
\qquad \Delta L_r=v_r\widetilde G_B.
\]

Thus its storage and arithmetic are proportional to \(|S|+|R|\), rather than
\(|S||R|\). Medium extinction is evaluated once on the gathered mode ledger;
receiver-specific material kernels are applied after the scatter. Blocks are
indexed by source membership, so a residual wave visits only blocks touched by
active sources.

Role filtering acts on block factor vectors themselves. Egress members are
removed from the source vector and ingress members from the receiver vector;
the remaining outer product stays retained. Energy-budget validation uses
\(u_s\sum_rv_r\) directly, without pair expansion.

The evaluator deliberately uses two regimes:

- scalar-link children cache unit responses by ingress and mode;
- any topology containing retained blocks batches the current ingress field
  through those blocks and does not populate the unit-response cache.

The second rule prevents gradual illumination of all ingress fibers from
silently reconstructing a dense boundary response matrix.

## Native retained execution

`native/retained_transport.{h,cpp}` implements the retained operator behind a
small C ABI. `pft_retained_plan_create_f64` copies and validates immutable
topology once. Subsequent calls therefore do not repeat bounds, coefficient,
role, or finiteness checks for every camera frame. The plan stores:

- fixed-size block descriptors;
- one packed `uint32` source-index stream and one receiver-index stream;
- contiguous `double` source and receiver factor streams;
- per-block optical depth and the three unified extinction coefficients.

The public arrays are explicitly `restrict`-qualified. Blocks whose index
streams are consecutive carry compile-time plan flags. Their gather and
scatter use direct contiguous loads; the arm64 implementation issues explicit
two-lane NEON fused multiply-adds. Irregular sources use a two-value gathered
NEON multiply, while irregular receivers use scalar indexed accumulation
because arm64 NEON has no scatter instruction. Generated assembly contains
the expected `fmla.2d` instructions.

Mode-major power storage makes every large coherent spatial gather contiguous.
The kernel evaluates the medium exponent once per block and optical mode, then
scatters that transported value through `v`. Spectral centre, variance,
polarization coordinates, and provenance never enter the coefficient loop;
only linear peakedness and absolute chirality participate in extinction.
`pft_retained_mode_plan_create_f64` goes one step further for a stable optical
basis: it retains the block-by-mode extinction table, so subsequent frames
perform no exponential evaluations at all.

There are two operational adapters:

- `ChildVolumeResponseSystem.evaluate` uses the C++ block wave inside the full
  Python residual march. Diffuse and Mueller boundaries may generate modes,
  and internal feedback remains supported and sealed at egress.
- `evaluate_packed` is the renderer path for direct terminal volumes. It keeps
  a modes-by-boundary power array packed and allocates no `LightTensor6`
  objects in the timed path. `PackedBoundaryField.materialize` is deferred
  until a consumer actually requires object ledgers.

If the shared library is absent, `execution_backend="auto"` preserves the
Python oracle. `execution_backend="native"` fails explicitly rather than
silently benchmarking the fallback.

## M4 measurements

All fourteen child-volume invariants, two direct native-kernel invariants, and
all thirty-six photonic transport tests pass on the M4 Mini.

The slab response used two internal edge applications on its first unit mode.
Changing that mode's power by 2.5 produced an exactly 2.5-times larger output,
with one cache hit, zero kernel marches, and zero child edge applications. A
negative half-strength correction likewise reused the response with zero
marches and returned exactly `-0.5` times the original boundary output. A
simultaneous request on one cached ingress/mode and one new ingress/mode
produced one hit and one miss. Replacing the topology with half the egress gain
cleared the child cache and produced exactly half the former output.

The parent/child control has a 0.04 complete feedback-cycle gain. The coupled
residual march returned parent-node population `1.0416666666666665`; its closed
form is `1.0416666666666667`, an absolute discrepancy of
`2.22e-16`. The child response was requested ten times as the parent
residual returned, but the sole child edge was applied once: one cache miss
compiled the response and nine later nonzero revisits were cache hits.

The repository-geometry control compiled one hidden surface and two legal
role-directed links. The outgoing field contained one active egress fiber;
none of the hidden nodes was inserted into or returned to the parent.

The coherent retained-block probe measured:

| Ingress × egress | Retained blocks | Stored coefficients | Equivalent pairs | Compression | M4 compile | M4 evaluate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 × 16 | 1 | 32 | 256 | 8× | 3.980 ms | 0.828 ms |
| 64 × 64 | 1 | 128 | 4,096 | 32× | 6.073 ms | 1.459 ms |
| 256 × 256 | 1 | 512 | 65,536 | 128× | 33.155 ms | 9.180 ms |

Every size required one block application, zero scalar edge applications, and
zero unit-response cache entries. The 256-fiber case gathered 256 source
coefficients and scattered 256 receiver coefficients: 512 coefficient actions,
not 65,536 pair actions. The complete record, including the earlier child
controls, took 70.785 ms.

The native dogfood battery keeps six wavelength/polarization ledger modes
packed and compares the C++ result to the identical retained NumPy algebra.
These are warm medians of five 200-evaluation batches on the M4 Mini:

| Transport scene | Fibers in → out | Blocks | Stored coefficients | Equivalent pairs | NumPy packed | Native packed | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| open field | 1,024 → 1,024 | 1 | 2,048 | 1,048,576 | 0.1088 ms | 0.0278 ms | 3.91× |
| single spherical occluder | 64 → 64 | 509 | 1,913 | 2,588 | 4.2618 ms | 0.0271 ms | 157.48× |
| offset occluder pair | 64 → 64 | 587 | 2,098 | 3,171 | 4.8864 ms | 0.0301 ms | 162.61× |

The open field deliberately exposes the lower bound: NumPy also performs one
large vector operation there, so native execution removes only dispatch and
temporary-array overhead. Occlusion produces hundreds of short retained
blocks. NumPy pays one high-level dispatch per block; the compiled plan walks
the whole field once. Maximum relative disagreement was `2.29e-15` in the
large open reduction and below `4.20e-16` in both occlusion fields.

The scene image is not a camera render. It displays the actual six-mode
boundary solve: the RGB source distribution followed by the open, centrally
occluded, and offset-double-occluded egress fields. It therefore exercises the
same compiled visibility blocks measured in the table without substituting a
raster-space shadow effect.

## Current limits

- Boundary fibers are supplied explicitly; adaptive edge-driven boundary
  subdivision is not yet compiled automatically.
- Media are homogeneous per link. Spatial density integrals and directional
  scattering phase functions are not yet represented.
- Hidden geometry uses oriented centroid facets and spherical occluders. A
  triangle BVH and texture/displacement-driven normal compiler remain to be
  added behind the same boundary contract.
- Rank is retained only for geometry blocks already accepted by the existing
  hierarchical admissibility test. Cross-block recompression and low-rank
  spectral/angular approximation have not yet been introduced.
- The established `regime_scene_native` room now backward-fuses coherent
  source-by-recipient portions of its diffuse field into retained blocks and
  marches those blocks through the native plan. Its prism and camera keep
  their established sealed-optics and exact-boundary evaluators.
- `render_retained_room.py` is only an isolated parent/child integration probe;
  it is not a replacement room architecture or the production demonstrator.
- The generic child response has not replaced the established prism evaluator.
  Joining those implementations must preserve the prism's current directional
  dispersion, feedback sealing, and camera-visible boundary behavior.
