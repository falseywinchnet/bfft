# BFFT roadmap: Chebyshev DCT/DST endpoints

This roadmap plans the next major capability: optimized DCT-I..IV and DST-I..IV
endpoints (forward and inverse) built on BFFT's Bruun/Chebyshev kernel, plus the
conversions between DCT/DST coordinates and standard bins, and between DCT+DST
and the real FFT. The release/maintenance track is preserved at the end.

## Why this is the right next target

The basis already pays for it. 

- The radix-4 Chebyshev evaluation node is a real cubic evaluation at the four
  real nodes `+/-u, +/-v` — 8 FMA, pure real arithmetic, no complex rotation.
  The Bruun complex-residue node is ~2x heavier because it carries the
  imaginary (conjugate-pair) half at every level.
- A real FFT bin is `X[k] = C[k] - i*S[k]` where `C` is a cosine projection
  (Chebyshev-`T`) and `S` is a sine projection (Chebyshev-`U`). The cosine tree
  alone is a DCT; the sine tree alone is a DST. Each is ~half an FFT.
- Native residue order -> standard bin order is an O(N) permutation
  (`residues_to_complex` is a permutation + conjugation, zero multiplies). The
  same cheap-pack structure carries over to DCT/DST.

So DCT/DST are not a side feature: they are the endpoints where the Chebyshev
basis does strictly less work (it never builds the half it does not need), and
the pack to standard DCT/DST bin order stays O(N). This is the win the basis
was always implying.

## Architecture mirror

Each transform follows the same three-layer shape the FFT path already uses:

```
time samples  --(radix-4 Chebyshev tree)-->  native residue order
native residue order  --(O(N) permutation/scale)-->  standard bin order
inverse: standard bin order -> native -> time, by the reverse schedule
```

- The tree body reuses the validated radix-4 Chebyshev node (all-FMA on FMA
  hardware; the 4-multiply `u^2+v^2=1` node where multiplies dominate).
- `T`-side (cosine) and `U`-side (sine) share one schedule and one constant
  table; they differ only in leaf initial conditions and endpoint handling.
- Native/standard split is an output-layout choice, exactly as for the FFT.

## Transform-type landscape

The Chebyshev tree's natural grid is the roots of `T_N`,
`x_k = cos((2k+1)pi/(2N))` (half-sample, no endpoints). Relative to that grid:

- **DCT-III / DCT-II**: native to the half-sample `T` grid. DCT-III is
  evaluation of a Chebyshev series at `T_N` roots; DCT-II is its transpose.
  Cheapest, first target.
- **DCT-IV / DST-IV**: half-sample in both index and frequency, no endpoints,
  cleanest energy distribution. A diagonal half-sample phase relates it to the
  base grid (O(N) scale). Strong benchmark target.
- **DST-II / DST-III**: the `U` (sine) siblings of DCT-II/III; same schedule,
  sine leaf conditions.
- **DCT-I / DST-I**: Lobatto / endpoint cases (`cos(k pi/N)`), natural degree
  `N-1`, two fixed endpoints. Most awkward (endpoint scaling, off-by-one
  length); scheduled last but required for the symmetric real-FFT fold.

## Phases

### Phase 0 - foundation (done)
- Radix-4 Chebyshev node family validated against direct evaluation and the
  `T_N`-root multipoint transform; multiply/throughput/stability floors mapped;
  hourglass and deferral paths resolved. Artifacts under `experiments/`.

### Phase 1 - DCT-II/III double, forward + inverse
- Forward DCT-II and DCT-III scalar reference, then the fused radix-4 tree.
- Native (Chebyshev-tower) order as the hot path; O(N) permutation to standard
  DCT bin order (`dct_residues_to_bins` / `bins_to_residues`).
- Inverse via the reverse schedule (DCT-III is the inverse of DCT-II up to
  scale).
- Validate vs a direct DCT matrix and vs FFTW `REDFT10/REDFT01` to ~1e-12.

**Status (in progress).** Reference layer landed in
`experiments/chebyshev_dct_kernel.hpp`, validated by
`experiments/test_chebyshev_dct.cpp` (`make radix4-test`):

- All eight types (DCT-I..IV, DST-I..IV) have direct forward + inverse kernels
  in FFTW `REDFT*`/`RODFT*` convention; forward->inverse round-trips to machine
  precision (~1e-14) over `n = 1..40`.
- The DCT-III <-> Chebyshev identity is proven numerically: `dct3_forward`
  equals Clenshaw evaluation of the series `c0=X0, cj=2Xj` at the roots of `T_n`.
- The flat-Chebyshev <-> nested power-of-`T_q` tower repack
  (`flat_to_tower`/`tower_to_flat`) round-trips to machine precision, and the
  shipped radix-4 node recursion (`eval_monomial`) is confirmed to compute
  Chebyshev multipoint evaluation.
- End to end, `dct3_via_tree` runs a real DCT-III through the radix-4 Chebyshev
  tree and matches the direct DCT-III to ~1e-12 for power-of-four `N` up to 4096.

The hot path is now realized in production shape in
`experiments/chebyshev_dct_radix4.hpp` (`ChebDCT`), mirroring the shipped FFT
kernel's discipline (`src/detail/bruun_kernel.hpp`):

- `init()` precomputes the per-node Chebyshev constants `{u,v,u2,v2,1/u,1/v,1/w}`
  into a 64-byte-aligned table indexed by pre-order node id (no `sqrt`/divide in
  the transform), the leaf->bin permutation in O(N) from the closed-form nodes,
  and reusable aligned scratch.
- `dct3_forward`/`dct3_inverse` run in place on a caller `work` buffer with zero
  per-call heap allocation; the body is the all-FMA even/odd node (capstone
  throughput optimum) with `RESTRICT` and the `[a0|a1|a2|a3]` block layout. The
  inverse node is multiply-only via the precomputed reciprocals.
- Validated: plan forward == direct DCT-III to ~1e-12, forward->inverse to
  machine precision, for power-of-four `N` up to 4096 (Tests 7-8).

Remaining for Phase 1: fuse `flat_to_tower` into the node store (kill the
separate O(N log N) repack pass), expose a standalone DCT-II forward (the inverse
schedule, scaled), add a float32 twin, and add the FFTW `REDFT10/REDFT01`
cross-check where the library is available.

### Phase 2 - DCT-IV/DST-IV double, forward + inverse
- Half-sample phase handling as an O(N) diagonal pre/post step on top of the
  Phase 1 tree.
- DCT-IV is self-inverse up to scale; DST-IV likewise. Validate vs direct and
  FFTW `REDFT11/RODFT11`.

**Status (done, reference + plan).** Landed in `ChebDCT`
(`experiments/chebyshev_dct_radix4.hpp`), validated by Tests 9-10 to ~1e-12 vs
the direct oracle and to machine precision on the self-inverse round-trip, for
power-of-four `N` up to 4096.

Correction to the original plan: a numerical search (`B = M3^{-1} M4`,
`A = M4 M3^{-1}`) shows DCT-IV is **not** a diagonal or bidiagonal step on the
single same-length cosine tree. The correct route is the capstone's C/S split
with a per-`k` half-sample rotation:

- `Ck = sum_j X_j cos(j th_k)` from the cosine (T) tree;
- `Sk = sin(th_k) * sum_m X_{m+1} U_m(x_k)` from the sine projection, obtained by
  converting the U-series to a T-series with an O(N) parity suffix-sum and
  running the *same* tree;
- `[DCT-IV_k; DST-IV_k] = 2 R(th_k/2) [Ck; Sk]`, an O(N) diagonal rotation.

So DCT-IV/DST-IV cost two cosine-tree evaluations plus O(N) boundary work — the
"two small real schedules" of Phase 7 — and DST-IV reuses every byte of the
DCT-IV plan (same tables, scratch, twiddles). The U-series sine projection built
here is also the groundwork for the Phase 3 DST endpoints.

### Phase 3 - DST-II/III double, forward + inverse
- `U`-side leaf conditions; reuse the Phase 1 schedule and constant table
  (joint `T`/`U` schedule where a caller wants both).
- Validate vs direct DST and FFTW `RODFT10/RODFT01`.

**Status (done, plan).** Landed in `ChebDCT`, validated by Tests 11-12 to ~1e-12
vs direct and to machine precision on the round-trip, for power-of-four `N` up
to 4096.

- DST-III (RODFT01) is the sine sibling of DCT-III: evaluation of a U-series at
  the *same* `T_n` nodes, scaled by `sin(th_k)`. Coefficients
  `d_p = 2 X_p (p<n-1), d_{n-1}=X_{n-1}`; the U-series is converted to a T-series
  by the O(N) parity suffix-sum (`u_to_t`) and run through the existing tree, so
  no `U`-specific node or table is needed — the cosine schedule is reused intact.
- DST-II (RODFT10) lives on a different grid (`cos((k+1)pi/n)`), so it is not a
  same-tree evaluation; it is realized exactly as `2n * (DST-III)^{-1}` via the
  inverse schedule (`tree_eval_inverse` + `t_to_u`). Inverse DST-II = DST-III/2n.

The inverse tree (`tree_eval_inverse`: gather + inverse node + `tower_to_flat`)
and the `t_to_u` inverse conversion added here complete the analysis direction
reused by every inverse endpoint.

### Phase 4 - DCT-I/DST-I double, forward + inverse
- Endpoint/Lobatto handling: fixed endpoints, degree `N-1`, endpoint half-weights.
- Validate vs direct and FFTW `REDFT00/RODFT00`.

**Status (done, FFT route).** Landed in `experiments/dct1_dst1_via_fft.hpp`
(`DCT1`, `DST1`), validated by Tests 13-14 to ~1e-14 vs direct and to machine
precision on the round-trip.

Finding: DCT-I/DST-I live on the Chebyshev-**Lobatto** grid `cos(pi k/(n-1))`
(extrema of `T_{n-1}`, endpoints included), not the first-kind `T_n` roots the
radix-4 tree evaluates at. There is therefore no in-tree fast path; the endpoint
cases need a different engine. The natural one -- and the "easy" route in the
difficulty matrix -- is the real FFT of the (anti)symmetric extension, which
encodes the fixed endpoints and half-weights exactly:

- DCT-I = `Re V[k]`, `V` = rFFT of the length-`2(n-1)` symmetric extension;
- DST-I = `-Im V[k+1]`, `V` = rFFT of the length-`2(n+1)` antisymmetric extension.

This reuses BFFT's own power-of-two real FFT (`bfft::plan`) with preallocated
buffers and no per-call allocation, so it is also the first concrete piece of
Phase 5. Size constraint: `n = 2^m + 1` (DCT-I) and `n = 2^m - 1` (DST-I) so the
extension length is a power of two; other `n` fall back to the direct oracle
(a general-`n` fast path needs a mixed-radix real FFT, out of scope).

### Phase 5 - FFT <-> DCT/DST assembly
- Real FFT from DCT+DST and back, via O(N) fold + combine. Build the two
  cheapest routes first (see difficulty matrix) and the rest as diagonal
  pre/post variants.
- Validate the assembled rFFT against the existing Bruun FFT and an independent
  DFT.

**Status (done, easy route + benchmark).** `experiments/cheb_dct_assemble.hpp`
(`AssembledRFFT`) builds the real FFT from the symmetric/antisymmetric folds:
`Re X = DCT-I(even fold)`, `Im X[b] = -DST-I(odd fold)[b-1]/2`. Validated by
Test 15 against both the shipped Bruun FFT and a direct DFT to ~1e-14, for
power-of-two `N` (8..1024).

Benchmark (`make dct-bench`, Apple arm64, best-of-trials, warm cache):

- The assembled rFFT runs the rFFT's Lobatto-grid projections through DCT-I/DST-I,
  which are themselves real FFTs of length ~`N`, so it does ~2 FFTs' work and
  lands ~3-5x the single fused Bruun kernel. This confirms the capstone's
  expectation that the fused path is the locality optimum for *bins*; the
  endpoint route is a correct assembly, not a faster one.

### Performance examination (optimization pass before Phase 5)

`make dct-bench` profiles the Chebyshev DCT path:

- **Node:** the radix-4 node is all-FMA. Adding `__restrict__` to the plan's
  forward node (`ChebDCT::forward_node`) flips it from scalar `fmadd` to 2-wide
  `fmla.2d` (verified in asm); ~1.09x at the node, but the node is not the
  bottleneck.
- **Bottleneck:** the `flat_to_tower` input repack (flat-Chebyshev -> nested
  tower) is **66-73% of the DCT-III wall-clock**. It is branchy, scalar, and
  scatter-write (`f[a+r]`, `f[a-r]` with `a-r` descending), so it does not
  vectorize. The tree evaluation itself is only ~30%.
- **Net (pre-optimization):** DCT-III was ~5-7x the tuned Bruun rFFT in
  wall-clock; the tree eval alone (minus repack) ~1.7x.

#### Optimization round (fusion + scheduling + de-branch)

Four changes brought DCT-III from ~6.5x to ~4x the Bruun rFFT (and ~3.2x at
N=16384), all validated against the oracle to ~1e-12 and round-tripping to
machine precision:

1. **`restrict` forward node** — flips the q-lane loop from scalar `fmadd` to
   2-wide `fmla.2d`; also fixed a latent `inv_sin_` leak found in passing.
2. **Repack fusion** — the separate `flat_to_tower` traversal is gone; each node
   does its local split then the node combine in one pass while the block is hot
   in cache. Justified by the *commute* of towering (within-block) and the node
   combine (across the four blocks per lane): `Fused = eval . flat_to_tower`,
   proven by induction and checked numerically.
3. **Flattened schedule** — the forward/inverse recursions are replaced by flat
   loops over a precomputed `(offset, q)` schedule (forward ascending pre-order,
   inverse descending post-order), removing all call overhead, worst at the many
   tiny leaf nodes.
4. **Branch-free `local_split`** — the Chebyshev fold's index ranges are worked
   out in closed form so the `add_Tpow` bounds checks become loop limits; the
   affine loops vectorize. This was the single biggest win.

Cumulative (Apple arm64, `make dct-bench`): DCT-III @N=1024 5975 -> 3439 ns
(1.74x); @N=16384 110us -> 74us (1.48x). The node is all-FMA and vectorized; the
residual gap to the FFT is the intrinsic ~6q/node repack work of the power-of-T_q
tower. Closing it further needs a repack-free factorization. The repack-free route is
NOT a generic fast-DCT (Lee / DIT) import -- that would be parity with *a* DCT,
not with this library. The correct path is to **sparsify the production Bruun
kernel**: the DCT is the Bruun real-residue tree with the imaginary (sine /
Chebyshev-U) half pruned (the node is already the real radix-4 cubic eval), fed
by the kernel's cheap binomial-fold input adaptation (O(N) adds) instead of the
tower repack. The tower repack was an artifact of the experimental
`eval_monomial`, not of the Bruun design, so deriving from the production fold
schedule removes it entirely; target ~half an FFT.

#### Radix probe for the real (DCT) node (`make dct-radix-probe`)

The capstone proved total multiplies are radix-invariant, so radix is purely an
instruction-level choice; the real node has a different shape from the complex
FFT node, so its optimum was re-measured rather than assumed. Result (Apple
arm64), full depth-r node tree, ns/output:

```
N         radix-2   radix-4   radix-8
64         2.33      1.98      6.30
4096       2.89      2.32      6.28
262144     3.45      2.74      6.83
```

**Radix-4 wins**, same as the FFT. The real node being lighter does not shift the
optimum to radix-8: per-output node cost grows with radix (r2 0.066, r4 0.085,
r8 0.144 ns/out isolated -- r8's serial Horner chains and 8 live blocks have poor
ILP), and radix-4's fewer-passes / 4-wide-ILP balance beats radix-2's extra
memory traffic. So the Bruun-native sparsified DCT-II is built on the radix-4
real node, fed by the forward binomial fold.

#### Repack-free kernel: algebra verified, naive cut is slow

`experiments/bruun_dct.hpp` (`BruunDCT`) implements the repack-free,
natural-order DCT-II and DCT-III. The algebra is the Bruun real DIF, verified to
machine precision (Test 16, both transforms vs the direct oracle and round-trip):

- DCT-III synthesis: even coeffs -> half DCT-III (via `T_{2i}=T_i o T_2`), odd
  coeffs -> bidiagonal `g` -> half DCT-III, butterfly `Y[k]=E +- 2 t_k O`.
- DCT-II = `2 * DCT-III^T`: the transpose turns the trailing butterfly into a
  leading mirror fold (the Bruun binomial fold), with a transposed-bidiagonal
  forward sweep on the way back up. The `T_2` composition and the `2cos` /
  bidiagonal (the U-companion handled with O(N) adds) keep it in the family --
  this is the sparsified Bruun, not a Lee/DIT import.

**But the first cut is a radix-2 *recursive* implementation with a scratch copy
per level, and it is slower than the optimized tower**: BruunDCT-III is ~13-15x
the Bruun rFFT vs the tower ChebDCT-III's ~4.5x (`make dct-bench`). The algorithm
has fewer total multiplies (no repack), but the naive implementation pays for
(a) radix-2 = 2x the passes, (b) two full scratch copies per level (de-interleave
/ interleave), (c) a serial bidiagonal sweep that kills ILP, (d) no schedule. The
win requires the same rework the tower got: radix-4, a single up-front
permutation then in-place iterative butterflies (no per-level copy), a vectorized
fold, and a restructured (non-serial) bidiagonal. That is the next pass; the
correct, verified algebra is the foundation it builds on.

### Phase 6 - float32 and API surface
- Float32 twins of the shipped DCT/DST kernels (mirroring the f32 FFT path).
- Public C and C++ endpoints following the existing naming
  (`bfft_dct2_forward`, `bfft_dct2_inverse`, native/standard variants, residue
  helpers), documented in `docs/api.md`.

### Phase 7 - competitiveness study
- Decide empirically whether the production real FFT should stay the fused
  `norm2` Bruun path or switch to DCT+DST assembly. The arithmetic is at parity
  (both ~the same total once the imaginary half is included); the open question
  is **locality**: DCT+DST runs two independent real schedules with smaller
  working sets and shared constants, which may cache better than one complex
  schedule. Benchmark both across N; report ns, FMAs, loads/stores.

## FFT <-> DCT/DST conversion difficulty matrix

The conversions are all O(N) boundary work, but they differ in how much:

- **Easy (reorder + trivial combine):**
  - `rFFT  =  DCT-I(even fold)  -  i * DST-I(odd fold)`. Symmetric fold is
    `N/2` adds + `N/2` subs; combine is `X[k] = C[k] - i*S[k]`, zero multiplies.
    Cleanest algebra; the cost is DCT-I/DST-I's endpoint handling (Phase 4).
  - `DCT-II <-> rFFT` (Makhoul): even/odd index reorder (permutation) + one
    O(N) post-twiddle. Easy and the most-used DCT in practice.
- **Moderate (diagonal half-sample twiddle):**
  - `DCT-IV / DST-IV <-> rFFT`: an O(N) diagonal of half-sample phase factors
    on top of the reorder. One multiply per element, still O(N).
- **Harder (endpoint and length mismatch):**
  - `DCT-I / DST-I <-> rFFT` as a *standalone* conversion (not the fold above):
    length `N+1` vs `N`, two fixed endpoints with half weights, and natural
    degree `N-1` rather than `N`. Correct but fiddly; sign/scale/endpoint care.

The guidance: ship the easy routes first (they cover the common cases), add the
diagonal-twiddle IV routes next, and treat the standalone DCT-I/DST-I conversion
as the last, most careful piece.

## Validation strategy

- Every kernel checked against (a) a direct matrix transform for small N and
  (b) FFTW's `REDFT*`/`RODFT*` where available, to ~1e-12.
- Round-trip (forward then inverse) to machine precision for all eight types.
- Assembled rFFT checked against the shipped Bruun FFT and an independent DFT.
- Native vs standard parity checks mirroring `tests/test_radix_parity.cpp`.

## Open question to settle (Phase 7)

Prove one of:
- the fused `norm2` Bruun real-FFT node is the locality optimum and DCT+DST
  assembly does not beat it; or
- DCT+DST assembly is competitive or better on locality grounds (two small real
  schedules, shared constants) at some N range, justifying a switch.

Either outcome is a result worth recording, like the capstone.

**Early empirical read (Phase 5 benchmark).** The first data point favors the
fused Bruun path: the DCT/DST-assembled rFFT is ~3-5x slower, and a single
Chebyshev DCT-III tree is ~5-7x the rFFT in wall-clock (dominated by the
`flat_to_tower` repack, not arithmetic). The locality case for DCT+DST is not yet
made; it hinges on first fusing the repack into the node store and scheduling the
recursion, after which the arithmetic (~half an FFT per tree) could show. Until
then the production real FFT should stay the fused `norm2` Bruun path.

---

## Release / maintenance track (carried over)

Still-relevant items from the prior roadmap, independent of the DCT/DST work:

1. Land the first CI workflow and get a green GitHub Actions run on `main`;
   teach the operator to inspect failures and require checks before merging.
2. Keep `pkg-config` and `find_package(BFFT CONFIG REQUIRED)` discovery smoke
   tests repeatable for staged installs; promote them into CI-visible checks.
3. Refresh `docs/release-checklist.md` with hosted CI evidence before tagging;
   decide the beta version label once CI history is stable.
4. Expand CI to macOS/Windows and aarch64 NEON once Makefile/CMake platform
   details are validated; add scalar-forced and AVX-class variants without
   exposing user-facing transform-selection flags.
