# Boundary information reuse

This September 5 pass audits and changes boundary reconstruction in the existing
retained CPU renderer. The target remains a complete 800×600-or-larger frame
within 33.33 ms, including geometry updates and observer motion. The changes
below preserve the current camera calculation exactly on the validation battery;
they do not complete that frame-time target.

## Discovery and shading are different costs

The boundary stage has five routes. Its optional `--boundary-audit` separates
label discovery from radiance evaluation within each route:

| Kind | Route | Default-view pixels | Label requests | Radiance evaluations |
|---|---|---:|---:|---:|
| 0 | Two-pixel optical filter, 8×8 queries | 785 | 50,240 | 1,940 |
| 1 | Filtered straight boundary | 3,690 | 7,380 | 7,380 |
| 2 | Analytically clipped straight boundary | 4,241 | 8,482 | 7,682 |
| 3 | Retained boundary fragments | 5,216 | 11,175 | 10,798 |
| 4 | 16×16 fallback after unsuccessful earlier routes | 550 | 140,858 | 2,089 |

The fallback label count includes unsuccessful earlier attempts. It is not
exactly 256 times its pixel count. The audit's `*_worker_ms` values sum elapsed
spans on individual workers; they are not frame wall times or isolated hardware
CPU time. They identify work concentration without being presented as frame
latency.

The distinction matters in harder scenes. In aperture canyon, 794 fallback
pixels lead to 4,087 radiance evaluations, 149,230 specular-area calls, and
10,288,940 source quadrature samples. Their accumulated shading span is about
5,040 worker-ms versus 499 worker-ms for discovery. Reusing labels alone cannot
remove this source-response cost.

## Compile the common filter domain

The existing two-pixel optical filter requests points

\[
x=p_x-1+(s_x+\tfrac12)/4,\qquad
y=p_y-1+(s_y+\tfrac12)/4,\quad s_x,s_y\in\{0,\ldots,7\}.
\]

Moving to an adjacent pixel shifts this query lattice by exactly four sample
positions. Therefore neighboring filters repeatedly ask for identical optical
labels. These coordinates are binary-exact quarter/eighth fractions; sharing
requires no rounding, tolerance, or approximate spatial key.

The new `BoundaryFilterField` first discovers the union of needed 4×4 query
blocks. A block owns sixteen labels. An 8×8 filter reads four blocks. The field
stores an integer block index per pixel-grid location and labels only for active
blocks, including the existing off-sensor filter halo. Unique block queries are
computed in parallel before the boundary pixels consume them.

The original sample order, filter weights, ownership buckets, weighted centroid,
and radiance evaluation are retained. This is reuse of the same finite coverage
calculation, not a claim that the samples certify an arbitrary continuous region.
No new interpolation is introduced.

The prepass is admitted only for at least 256 requests and at least a 10%
reduction in exact query count. Isolated filters fall back to the existing path.
This is a finite work-count admission rule; it is not a universal wall-time
optimality guarantee. All construction and allocation costs are included in the
reported boundary-stage timing.

For the default view:

- Original filtered requests: **50,240**.
- Unique shared queries: **17,456**, a **65.3% reduction**.
- Shared-field construction with query reuse: **4.29 ms** median.
- With this change alone, boundary reconstruction falls from **95.89 to
  86.34 ms** median on the CLI's linear traversal policy.

For mirror relay, 175,552 filter requests become 60,560 unique queries. The
independent and shared versions produce identical labels and images.

## Retain geometric answers across optical consumers

Ownership discovery, spectral probes, and radiance evaluation often submit the
same ray more than once. Different wavelengths can have identical reflected
continuations until an actual dispersive refraction changes their geometry.
The previous path functions repeated the scene intersection even in the
identical cases.

`OpticalQueryMemo` retains the exact geometric result of

\[
Q_G(o,d,t_{\max},i)=\operatorname{firstHit}(G,o,d,t_{\max},i),
\]

where `i` is the ignored primitive. Each key contains the full bit patterns of
all six origin/direction coordinates, the distance limit, and the ignored ID.
The bounded table uses a hash for addressing but compares the complete key
before reuse. A collision causes replacement, never an approximate hit. The
stored result is the full hit, including distance, position, normals, side,
and primitive identity. Misses, finite-distance queries, and ignored-primitive
queries remain distinct.

The default is **128 entries per worker**. The table lives within one immutable
boundary pass, including separate worker-local tables for shared-filter
construction and reconstruction. It does not survive geometry changes, camera
frames, or a different scene. Color, Fresnel weighting, volume response, and
spectral radiance are still evaluated with their own parameters; only identical
geometric intersection queries reuse a result. The ordinary intersection
routine remains the uncached oracle.

In the default view, the shared representation makes 2,740,317 covered optical
queries and reuses a median 1,086,249 results (39.6%). The independent filters
would issue 3,297,011 covered queries. Combining filter sharing and exact query
reuse leaves about 1,654,068 scene searches: **49.8% fewer** than that covered
baseline. These counters cover the context-owned optical query path, not every
primary packet ray or shadow/source integration query in the entire renderer.
The small variation in reuse counts between repetitions comes from worker
assignment and bounded-table eviction, not changed images.

A 512-entry screen slightly increased reuse but did not reliably improve the
harder-scene timings. The 128-entry table is the retained default.

## An experiment that did not earn its cost

A separate experiment extended surface-radiance memoization beyond a single
primary ray and tried 256, 1,024, and 4,096 entries per worker. It preserved
images but did not reliably reduce expensive specular integration or wall time.
The default-view boundary stage stayed around 94–95 ms in that screen, and
specular call counts sometimes increased because of cache replacement behavior.
This implementation was removed from the production engine.

The lesson is narrower than “caching helps”: exact surface responses at nearby
points are often different, while optical consumers really do repeat identical
geometric queries. The first needs a representation of a varying response over
a region; a larger exact-value table does not supply that representation.
Diagnostic JSON files and a re-applicable probe patch are retained under
`boundary_discovery_m4/diagnostics/` so this result is not mistaken for an
accepted acceleration.

## Paired measurements at 800×600

The main benchmark uses four modes, three repetitions, and a rotated mode order:

- Mode 0: independent filters, uncached optical queries (reference).
- Mode 1: shared filters, uncached queries.
- Mode 2: independent filters, 128-entry query reuse.
- Mode 3: shared filters and 128-entry query reuse (new default).

There are eight scene/pose/traversal cases, for **96 complete image comparisons**.
All images are byte-identical to mode 0 within their case. Both scene and
lighting are compiled before the camera timing. The table reports medians and
includes shared-field construction in the boundary stage.

| Scene / pose | Traversal | Boundary before → after (ms) | Complete camera before → after (ms) |
|---|---|---:|---:|
| Standard, default | Linear | 95.89 → **70.57** | 217.57 → **192.20** |
| Standard, rotated | Linear | 96.93 → **69.93** | 217.59 → **190.71** |
| Standard, translated | Linear | 92.83 → **67.37** | 209.87 → **184.19** |
| Standard, prism overhead | Linear | 86.80 → **64.36** | 256.76 → **233.65** |
| Standard, default | BVH | 127.20 → **93.19** | 296.24 → **262.11** |
| Aperture canyon | Linear | 815.03 → **775.43** | 1,157.75 → **1,116.47** |
| Mirror relay | Linear | 562.74 → **499.03** | 862.66 → **794.64** |
| Occlusion garden | BVH | 247.52 → **193.19** | 495.93 → **443.31** |

The default-view boundary reduction is **26.4%**, or 1.36× faster. The complete
camera reduction is **11.7%**. Shading-heavy aperture canyon improves much less,
consistent with its audit. No resolution or optical cutoff was reduced.

### Acceleration provenance correction

The earlier camera and retained-update benchmark helpers used `Scene`'s default
BVH traversal. The command-line renderer's `auto` policy selects linear
traversal for fewer than 64 primitives. This explains an important part of the
earlier difference between single CLI timings and the helper's camera timing;
it should not have been attributed simply to run-to-run host variation.
Previous within-pair comparisons used matching traversal and remain valid.
This benchmark explicitly records acceleration and includes both policies for
the standard scene. The report does not claim that switching policy is part of
the new boundary speedup, or that linear traversal is preferable for large scenes.

The existing integrated update benchmark retains its BVH policy:

| Edit | Lighting repair (ms) | Camera (ms) | Geometry + lighting + camera (ms) |
|---|---:|---:|---:|
| First move | 79.83 | 252.87 | **332.70** |
| Second move + observer translation | 79.90 | 254.22 | **334.12** |
| Insertion + observer translation | 140.31 | 254.48 | **394.79** |

The edited images again match fresh compilation exactly. These are three update
steps, not a sustained interactive latency distribution, and exclude display
presentation. They remain about **10–12× the complete 33.33 ms target**.

## What the next deeper representation must remove

For an optical ownership region `R`, a pixel response conceptually contains

\[
C_p=\sum_R\int_{R\cap P_p}w_p(u,v)\,
L_R(u,v)\,du\,dv.
\]

The present renderer discovers labels, assembles a finite approximation to the
coverage, and shades representative points. This pass shares repeated labels
and geometric queries. It does not yet retain the varying `L_R` over an optical
region, nor transport that region through observer motion.

A stronger construction would make an optical continuation region own both its
visibility constraints and its response to incident scene light. Then coverage
and shading could consume the same retained geometry, and distinct neighboring
rays could evaluate a shared response function. Admission would need to respect
silhouette changes, depth-order changes, refraction/total-internal-reflection
boundaries, and sharp source lobes. Equality of a few sampled labels alone is
not a sufficient certificate for that generalization.

The measured target is now clear: in the hard scenes, a small set of complicated
boundary pixels expands into many distinct spectral surface-response evaluations.
Reducing those distinct integrations requires more than remembering identical
rays. The source/observer separation developed in
[CAMERA_DEMAND_GATHER.md](CAMERA_DEMAND_GATHER.md) remains relevant, applied along
these optical continuation regions. A shimmer or image effect would not replace
the underlying geometric work.

## Reproduction and checks

```sh
sh experiments/photonic_transport_field/run_retained_boundary.sh
```

The script uses `m4build` and the route selected by `m4host`, builds into `/tmp`,
and copies JSON receipts back immediately. It runs native scene, retained-update,
camera, and boundary suites; the four-mode benchmark; updated-scene image checks;
and baseline work audits. ASan/UBSan and TSan instrument the C++ engine and its
C CONV dependency, then run the focused suite including a full boundary render.
All passed. The optimized build is warning-free; the sanitizer build reports
three existing unfulfilled vectorization requests.

Focused checks cover full-key cache equality with a one-entry collision-heavy
table and the default table, finite distance limits, ignored primitives, disabled
caching, linear and BVH traversal, exact overlapping filter labels, image-border
halos, isolated-filter rejection, and complete-image equivalence. Shared blocks
are written by one worker and read only after joining; geometric query tables
are private to a worker. The cache lifetime prevents reuse across geometry edits.

The independent modes remain callable with `--boundary-filter independent`
and `--boundary-path-cache 0`. `--boundary-audit` writes its class breakdown to
stderr and regular frame metrics to stdout. Benchmark timing disables that
per-pixel audit instrumentation.

Primary records:

- [Alternating paired benchmark](boundary_discovery_m4/boundary_discovery_benchmark.json)
- [Integrated updates](boundary_discovery_m4/boundary_discovery_updates.json)
- [Default-view work audit](boundary_discovery_m4/boundary_baseline_standard_audit.json)
- [Aperture-canyon work audit](boundary_discovery_m4/boundary_baseline_aperture-canyon_audit.json)
- [Validation log](boundary_discovery_m4/validation.log)
- [Source hashes](boundary_discovery_m4/source_hashes.json)
