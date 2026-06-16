# BFFT roadmap: Chebyshev DCT/DST endpoints

This roadmap plans the next major capability: optimized DCT-I..IV and DST-I..IV
endpoints (forward and inverse) built on BFFT's Bruun/Chebyshev kernel, plus the
conversions between DCT/DST coordinates and standard bins, and between DCT+DST
and the real FFT. The release/maintenance track is preserved at the end.

## Why this is the right next target

The basis already pays for it. Established results (see `capstone.md`):

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

### Phase 2 - DCT-IV/DST-IV double, forward + inverse
- Half-sample phase handling as an O(N) diagonal pre/post step on top of the
  Phase 1 tree.
- DCT-IV is self-inverse up to scale; DST-IV likewise. Validate vs direct and
  FFTW `REDFT11/RODFT11`.

### Phase 3 - DST-II/III double, forward + inverse
- `U`-side leaf conditions; reuse the Phase 1 schedule and constant table
  (joint `T`/`U` schedule where a caller wants both).
- Validate vs direct DST and FFTW `RODFT10/RODFT01`.

### Phase 4 - DCT-I/DST-I double, forward + inverse
- Endpoint/Lobatto handling: fixed endpoints, degree `N-1`, endpoint half-weights.
- Validate vs direct and FFTW `REDFT00/RODFT00`.

### Phase 5 - FFT <-> DCT/DST assembly
- Real FFT from DCT+DST and back, via O(N) fold + combine. Build the two
  cheapest routes first (see difficulty matrix) and the rest as diagonal
  pre/post variants.
- Validate the assembled rFFT against the existing Bruun FFT and an independent
  DFT.

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
