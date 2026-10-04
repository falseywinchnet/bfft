# Computational formulation of CONV pixel-area integration

14 September 2026. An implementation study of the admitted tensor-quintic atlas and its projective pixel functional. The retained candidate is experimental; the public kernel has not been replaced.

The compact triangle rule reduces the measured 1536 × 1024 area render from **4480.52 ms to 2996.96 ms**, a **1.495× speedup**, with the same complete RGBA output hash. It preserves the current surface, clipping, reference subtraction, selected density-polynomial degree, subdivision partition, and reported series-tail bound. The immediate saving comes from integrating the same polynomial with fewer quadrature locations. A separately tested Green-boundary formulation also preserves the result but is slower in its present implementation.

## The computed functional

Let E be a target pixel basin, Q the visible image plane, F the inverse warp, and G = F⁻¹. The output is

\[
 I_E=\frac{1}{|E|}\int_{E\cap Q}U(F(q))\,dq.
\]

The browser uses endpoint-aligned Voronoi basins; the basins on the outside edge have half the ordinary extent along that axis. Exterior coverage is transparent. RGB and alpha are integrated in premultiplied sRGB coordinates, then unpremultiplied and encoded to bytes.

On each source cell C, the already admitted field is

\[
 U_C(x,y)=\sum_{i,j=0}^{5}P_{C,ij}B_i^5(x)B_j^5(y).
\]

The input to the integrator is this fixed surface. The integration study introduces no additional source analysis or admission. With local source coordinates and the coordinate scale absorbed into the homogeneous matrix, the projective area density has the form

\[
 J_G(x,y)=\frac{|\det H_G|}{|\ell(x,y)|^3},
 \qquad\ell\text{ affine}.
\]

The current implementation:

1. Clips E against Q in target coordinates and measures the covered target area.
2. Maps that polygon to source coordinates.
3. Enumerates source cells in its bounding box and clips against each cell.
4. Subtracts a per-output-pixel reference R from the 36 controls of each incident patch. The term R times target coverage is integrated directly.
5. Uses integrated Bernstein products for axis-aligned affine rectangles. Otherwise it triangulates each clipped source-cell polygon.
6. On each triangle T, expands the reciprocal cubic density about the midpoint of its vertex denominator range, selects degree K from 0 through 6 using the existing tail bound, and subdivides if required.
7. Constructs 36 moments, contracts them with the four residual channels, restores exact reference coverage, and encodes the output.

For the selected triangle approximation,

\[
 J_G\approx\rho_T S_K(z),\qquad
 z=\ell/c_T-1,\qquad
 S_K(z)=\sum_{k=0}^{K}(-1)^k\binom{k+2}{2}z^k.
\]

The moment integrand B_i⁵(x)B_j⁵(y)S_K(z) is polynomial of **total degree at most 10 + K**. This degree, rather than image dimensions or channel count, determines the necessary triangle quadrature order. The 36 resulting moments serve every channel.

## Where the work goes

The retained 1536 × 1024 synthetic perspective case performs:

- 1,572,864 output-pixel visits; 699,483 have positive plane coverage.
- 5,316,250 candidate cell clips; 4,725,780 clipped cell polygons survive the vertex-count check.
- 9,451,560 accepted triangle integrals and **zero recursive subdivisions**.
- 211,784 triangles at K = 0 and 9,239,776 at K = 1.
- 460,373,248 Gauss-Duffy locations, each contributing to 36 weights: about **16.57 billion scalar weight contributions**.

The original tensor rule uses n = 6 + floor((K + 1)/2) locations per Duffy axis. The quadratic n² count is paid for each triangle. The source atlas is cached, but these footprint moments are rebuilt on each warp.

Five-run medians on the M4 Mini:

| Diagnostic variant | 257 × 257 perspective | 1536 × 1024 perspective |
|---|---:|---:|
| Original production kernel | 179.39 ms | 4480.52 ms |
| Same kernel with counters | 181.16 ms | 4518.43 ms |
| Counters, comparison sampling omitted | 182.35 ms | 4499.13 ms |
| Counters, moment construction and contraction omitted | 29.02 ms | 728.08 ms |

The last row is an intentionally incomplete cost ablation and does not produce a valid integrated image. It retains clipping, patch/reference preparation, degree selection, tail bookkeeping, comparison, and output handling. The gap places roughly 84% of this measured runtime in the omitted moment construction and contraction. This is an experimental cost isolation, not an additive cycle-accurate profiler. Removing comparison produces no material speedup in these runs.

## Compact polynomial-exact triangle integration

The candidate uses positive, interior Xiao–Gimbutas rules for total degrees 10–16. The upstream tables and license are retained in `triangle-rules.json` and `COPYING.xiao-gimbutas` in the browser study. The pinned source is [the authors' table](https://github.com/zgimbutas/triasymq/blob/25a42d3287118cd9064b57de457947fb937691a9/triasymq_table.txt).

| Selected K | Polynomial degree | Original locations | Compact locations |
|---:|---:|---:|---:|
| 0 | 10 | 36 | 25 |
| 1 | 11 | 49 | 28 |
| 2 | 12 | 49 | 33 |
| 3 | 13 | 64 | 37 |
| 4 | 14 | 64 | 42 |
| 5 | 15 | 81 | 49 |
| 6 | 16 | 81 | 55 |

For a triangle with vertices a,b,d, a rule point (u,v) maps to a + u(b−a) + v(d−a), with weight multiplied by the absolute edge determinant. There is no Duffy factor (1−u). The contraction then remains

\[
 M_{ij}=\rho_T\sum_q w_q B_i^5(x_q)B_j^5(y_q)S_K(z_q),
 \qquad
 V_T=\sum_{i,j}(P_{ij}-R)M_{ij}.
\]

Both rules integrate the selected polynomial exactly in real arithmetic. Replacing the rule therefore changes neither the projective series approximation nor its tail criterion. Floating-point rule coordinates and summation still introduce roundoff, as they do in the original Gauss construction.

In the large case, the new location count is **264,008,328**, a **42.65% reduction**. The remaining clipping, control reads, and encoding explain why the full-frame speedup is smaller than the quadrature-location ratio.

## Measured candidate results

Times are milliseconds, medians of five warm runs on the M4 Mini, Node v25.2.1, arm64. The input is a deterministic synthetic RGB field with opaque alpha. Source preparation is excluded. Timings include area rendering, bilinear comparison, byte encoding, and both output copies; they exclude browser scheduling, PNG compression, and viewport reduction. The kernels use Float64 WASM SIMD and the same Float32 admitted atlas.

| Source | Output | Map | Original ms | Compact ms | Speedup |
|---|---|---|---:|---:|---:|
| 257 × 257 | 257 × 257 | identity | 37.50 | 39.15 | 0.96× |
| 257 × 257 | 257 × 257 | affine | 124.16 | 100.32 | 1.24× |
| 257 × 257 | 257 × 257 | perspective | 179.39 | 122.36 | 1.47× |
| 257 × 257 | 257 × 257 | strong | 179.50 | 134.23 | 1.34× |
| 257 × 257 | 65 × 65 | perspective | 81.94 | 54.72 | 1.50× |
| 257 × 257 | 513 × 513 | perspective | 369.72 | 257.67 | 1.43× |
| 1536 × 1024 | 1536 × 1024 | perspective | 4480.52 | 2996.96 | 1.50× |

All seven complete output hashes agree. Every case retains the same reported tail and triangle count. The unchanged axis-aligned identity path is a timing control; its variation is not an integration improvement.

The 54-case numerical battery covers smooth, checkerboard, and noisy partially transparent sources; identity, affine, projective, strong-projective, thin and near-affine maps; and minified and magnified rectangular outputs. It exercises all seven selected degrees and 2706 subdivisions. The compact candidate's largest sampled unencoded-channel deviation from the baseline is **1.3322676295501878 × 10⁻¹⁵**. All complete rendered outputs in that battery are byte-identical. These measurements establish agreement for the tested cases, rather than a universal bitwise identity theorem.

The rule importer independently checks every monomial through each claimed total degree against

\[
 \int_{u,v\ge0,\;u+v\le1}u^a v^b\,du\,dv=\frac{a!b!}{(a+b+2)!}.
\]

All weights are positive and all nodes lie inside the reference triangle. The maximum relative monomial residual across the retained rules is below 2 × 10⁻¹⁴.

## The tested boundary formulation

The paper's Green-boundary representation also applies to the selected polynomial. At fixed y, write S_K(αx+βy+γ) = Σ_r h_r(y)xʳ. Since

\[
 x^r B_i^5(x)=\frac{\binom5i}{\binom{5+r}{i+r}}B_{i+r}^{5+r}(x),
\]

each x-antiderivative follows from an integrated Bernstein basis. If

\[
 A_{i,r}(x)=\int_{x_*}^{x}s^r B_i^5(s)\,ds,
\]

then a moment is an oriented boundary integral

\[
 M_{ij}=\rho_T\oint_{\partial T}B_j^5(y)
             \sum_{r=0}^{K}h_r(y)A_{i,r}(x)\,dy.
\]

The line integrand has degree at most 11 + K. The existing 6–9 point Gauss rules therefore evaluate each nonhorizontal edge exactly for that polynomial. The prototype chooses x_* at the minimum triangle x-coordinate to reduce cancellation and reuses the moments across channels.

This form matched all rendered bytes in its 54-case battery, with a largest sampled unencoded deviation of 1.4432899320127035 × 10⁻¹⁵. It also matched every full-frame benchmark output. It was nevertheless slower: **230.41 ms versus 179.39 ms** at 257 × 257 and **5590.75 ms versus 4480.52 ms** at 1536 × 1024. Evaluating several integrated basis orders separately on each triangle edge costs more than the dense, SIMD-friendly tensor sum it replaces. Counting fewer geometric locations alone does not establish lower computational cost.

## Further formulations to investigate

The measured compact rule is a concrete first improvement. Larger gains should remove repeated primitive construction rather than merely rearrange each triangle in isolation.

**Whole-cell and shared-edge moments.** A clipped polygon contains internal triangulation edges, and neighboring target pixels share footprint boundaries. A common primitive permits opposite orientations of the same edge to cancel or reuse one evaluation. A source-cell/edge traversal can compile primitives once for the current map and distribute signed contributions to incident pixels. The present triangle-specific expansion centres differ, so its internal edge terms cannot simply be discarded: a common polynomial density or the exact rational primitive is required. Its accuracy and stability must be checked as part of that reformulation.

**Fully covered source cells.** Their weighted cell integrals can be computed once per map. Interior rectangular runs can then be accumulated from prefix sums, leaving explicit clipping for boundary cells. This is especially relevant under strong minification, where the current algorithm still visits source-cell interiors individually. The projective density makes these caches map-dependent; they are not a source-only summed-area table. Their construction and memory cost belong in the complete timing.

**Exact rational edge primitives.** The paper already expresses the true projective integral using rational powers and logarithms of denominator values. This can remove density-series evaluation and subdivision. A useful implementation needs stable recurrences across the affine limit and small denominator differences, and reuse across whole clipped polygons. The tested polynomial-boundary prototype does not establish the performance of that larger reformulation.

**Shared-atlas parallel tiles.** Once each tile has its own stack, scratch space and output region, independent pixel tiles can evaluate the same Float64 integrals in parallel over one read-only atlas. This is an additional implementation opportunity. Copying the roughly 877 MiB large-case atlas per worker is not a suitable memory design; actual shared memory and per-worker scratch are required. No multicore speedup was measured here.

An exact arbitrary-image reduction must account for the admitted coefficients covered by a pixel or for exact aggregates containing those coefficients. The useful objective is to amortize that dependence through shared primitives and aggregates, while retaining the pixel functional. The compact rule already does so within each triangle's fixed polynomial degree; the larger opportunities operate across triangles, cells and pixels.

## Reproduction and retained evidence

The browser-side study is `paymenottowork/research/conv-area-acceleration/`. `build.py` generates isolated kernels; it does not rewrite the production WASM or its provenance. Its generated baseline hash equals the shipped kernel:

`b1c90245331dace5a2e3032143f032520668482bd9569cd074698fc56c5d3052`.

`check.mjs` compares either candidate with that baseline. `profile.mjs` retains every timing, output hash, tail, and work count. `results/initial-profile-runner.mjs` is the snapshot used for the first run; the current runner additionally accepts `VARIANTS`. Counter indices 0–4 are visited pixels, covered pixels, candidate cells, surviving cell polygons, and subdivisions; 5–11 count selected K = 0–6. Index 12 is the equivalent original Gauss-Duffy node count, retained for all variants. Compact evaluation counts are derived from the degree histogram and the compact rule sizes.

The raw profiles are `results/initial-profile.json` and `results/compact-profile.json`; correctness receipts are `results/edge-check.json` and `results/compact-check.json`. `results/manifest.json` records the local source and evidence hashes. The mathematical operator and exact projective edge derivation remain in BFFT's `output/pdf/convstar_warp_addendum.tex`.


## Geometry-only quadrature follow-up

The geometry-selected rational integration formulation and its complete validation are recorded in [geometry-plan.md](geometry-plan.md). The fixed table uses denominator variation alone, with positive 25–55-node triangle rules, direct rational density evaluation, and a 1e-9 mass-relative operator budget established before rendering. The validated kernel is adopted in the local perspective instrument. Retained `results/geometry-*.json` contain analytic bank checks, content-independent plan checks, image comparisons and final Mini timings.
