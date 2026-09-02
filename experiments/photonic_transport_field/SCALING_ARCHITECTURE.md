# Scaling the adaptive photonic transport field

The renderer now separates three complexities that were previously hidden
inside one small-scene implementation:

1. geometric visibility;
2. discovery and storage of transport relations;
3. propagation and camera gathering over those relations.

The first and third now have sparse implementations. Relation discovery is the
remaining quadratic component and must become hierarchical before scenes with
millions of independently transporting surfaces are possible.

## Implemented structural changes

### Geometry

The scene builds a flat median-split AABB hierarchy. The first-hit query
traverses that hierarchy near-first and retains deterministic primitive-ID tie
breaking. The same hierarchy supplies:

- conservative source-pyramid candidates for rectangular-emitter occlusion;
- conservative camera-frustum candidates for visible-edge discovery;
- automatic linear traversal for very small scenes, where a BVH costs more
  than testing the short primitive array.

The scaling harness is guarded by both --benchmark-primitives and the explicit
--max-primitives ceiling.

### Emitter occlusion

Planar blockers on one emitter row are represented by their actual horizontal
spans. A sorted active-span sweep now tests depth-order roots only for
overlapping blockers. Its work is

\[
O(B\log B + K),
\]

where \(B\) is the number of row blockers and \(K\) is the number of
overlapping span pairs that can actually exchange order. The old code
unconditionally reserved and tested \(O(B^2)\) pairs, even when no two
blockers overlapped.

### Transport storage and propagation

The dense matrix is replaced by compact incoming and outgoing edge arrays.
Storage is \(O(N+E)\), not \(O(N^2)\). Illumination is evaluated as a
source-forward residual march:

\[
\Delta L^{(k+1)}_r =
\rho_r \sum_{s\in\mathcal A_k} T_{rs}\Delta L^{(k)}_s,
\qquad
L_r = L_r^{(0)} + \sum_k \Delta L^{(k)}_r .
\]

Only recipients of an active source enter the next frontier. This is the
non-path-enumerating form of source -> propagate -> gather: a multi-bounce path
appears through repeated sparse operator application, while no path object is
created.

### Camera evaluation

The visible-path signature needed by the camera edge field is now produced
during the terminal-topology traversal instead of tracing the same optical
continuation twice. Boundary region buckets are fixed per worker rather than
heap-allocated for every crossed pixel.

## M4 Mini measurements

The synthetic scaling harness casts 4,096 deterministic rays into a dense
sphere field. Linear and BVH checksums agree.

| primitives | mode | BVH build | query time | time/ray | speedup |
|---:|:---|---:|---:|---:|---:|
| 1,000 | linear | 0.295 ms | 15.491 ms | 3.782 us | 1.0x |
| 1,000 | BVH | 0.208 ms | 1.891 ms | 0.462 us | 8.2x |
| 10,000 | linear | 3.769 ms | 96.439 ms | 23.545 us | 1.0x |
| 10,000 | BVH | 2.729 ms | 3.072 ms | 0.750 us | 31.4x |
| 100,000 | BVH | 30.276 ms | 3.398 ms | 0.830 us | — |
| 1,000,000 | BVH | 203.323 ms | 5.253 ms | 1.283 us | — |

The retained 1920x1280 scene has only 29 primitives, below the automatic BVH
crossover, and therefore uses linear traversal. Its matched optimized run took
3.822 s. Output-sensitive blocker ordering reduced emitter intervals from
48,735,053 to 45,740,374 (6.1%) and quadrature evaluations from 149,804,701 to
138,159,470 (7.8%). Relative to the retained image, 218 pixels changed and the
maximum channel change was 2/255.

## Remaining non-scalable representations

### 1. Pair discovery is still quadratic

The diffuse compiler stores only surviving links, but it still visits every
surface-source/surface-receiver pair and then every quadrature-sample pair.
Consequently its construction is still

\[
O(N^2 S_r S_s C_{\rm visibility}).
\]

Sparse storage cannot repair an exhaustive discovery procedure. This is the
next architectural target.

### 2. A primitive is an oversized tagged union

The current Primitive occupies 248 bytes because each entry stores rectangle,
triangle, and sphere fields simultaneously, plus a dynamic name object. One
million primitives consume 248 MB; the current BVH consumes another 132 MB.
Production geometry should use:

- compact shape-specific structure-of-arrays storage;
- 32-bit geometry and material handles;
- names and diagnostics in a cold side table;
- BLAS instances for repeated meshes and a TLAS over object instances.

This should reduce analytic leaf storage to roughly 48–96 bytes, and repeated
geometry to one transform and one instance handle.

### 3. Camera topology still asks illumination questions per pixel

The camera pass currently samples rectangular-light blocker topology while
classifying every first-surface pixel. That information belongs to the compiled
surface light field. A camera query should return

\[
(\text{surface region},\ \text{directional terminal},\ uv)
\]

and gather already-solved illumination from that region. It should not
reconstruct source occlusion.

### 4. Edge discovery oversamples projected perimeter

Analytic edges are tested at eight points per projected pixel before invisible
segments are discarded. BVH frustum culling removes off-camera objects, but a
large hidden in-frustum population still costs projected-perimeter work.
Projected line/conic intervals should instead be clipped against an occlusion
hierarchy and rasterized by root stepping.

## Required relation compiler: bounded dual-tree transport

Build a second hierarchy over transport regions, distinct from the object BVH.
Each node carries:

- spatial bounds and total area;
- a normal cone;
- material/BRDF class;
- emitted and residual spectral-energy bounds;
- low-order spatial and directional radiance moments;
- child ownership of exact source-induced surface regions.

Traverse source and receiver trees as pairs. For a cluster pair, bound the
largest possible diffuse transfer by a conservative form-factor envelope:

\[
\widehat T_{R\leftarrow S}
\le
\min\!\left(
1,
\frac{A_S c_S^+c_R^+\tau_{\max}}{\pi d_{\min}^2}
\right).
\]

If

\[
E_S^{\rm residual}\widehat T_{R\leftarrow S}<\varepsilon_R,
\]

the entire cluster pair is impossible to perceive and is pruned without a
visibility ray.

Otherwise:

1. if the clusters are well separated, normal-coherent, material-compatible,
   and visibility-certified, emit one low-rank block relation;
2. if an occlusion beam sweep finds a partial visibility boundary, bifurcate
   only the affected source/receiver region;
3. if the geometric or radiometric error bound is too wide, split whichever
   cluster has the larger projected uncertainty;
4. descend to exact region relations only at true discontinuities.

This produces work proportional to the number of resolvable transport
relations, not the square of object count. Smooth distant populations remain
aggregate nodes. A tiny visible corner becomes an exact child region because
the beam-sweep visibility certificate fails only there.

## Loops, motion, and camera independence

Strongly connected specular relation components should be detected once,
integrated as local feedback operators, and sealed when their residual energy
bound falls below budget. Diffuse propagation then sees one contracted
component rather than repeatedly rediscovering its paths.

For motion, maintain ownership from object/TLAS nodes to transport-tree
relations. Refit the geometric hierarchy, invalidate only relations whose
visibility cones overlap the changed bounds, and remarch their residual
budgets. Cameras remain terminal one-order observers and do not invalidate
global illumination.
