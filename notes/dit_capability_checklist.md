# DIT_RFFT_kernel capability checklist (parity target with DIF_RFFT_kernel)

Scope decided 2026-07-02 (with user):
- Formats: **complex-interleaved only**. Mag/phase and magnitude will live in a
  dedicated kernel and tap off the complex bins; NOT implemented in DIT. Split
  re[]/im[] deferred.
- f32 means **genuine float32 compute** (not double-then-narrow).
- AVX2: **macro-abstraction + extension notes only**, no dead 256-bit code. The
  real 256-bit widening (bruun_v4 / v8f) comes from an x86 companion model.

Kernel: `bruun::DIT_RFFT_kernel` in `src/detail/bruun_dit_kernel.hpp`.
Normalized (unit-frame Givens) basis; standard-order output.

## Capability matrix

| axis | double | f32 |
|------|--------|-----|
| forward scalar   | [x] done | [x] done (genuine float, rt ~1e-6) |
| inverse scalar   | [x] done | [x] done |
| forward SIMD | [x] done | [x] wide v4f, even sizes (odd -> scalar); parity DIF |
| inverse SIMD | [x] done | [x] wide v4f, even sizes (odd -> scalar); parity DIF (organized row-tile scatter, matches double) |

Inverse transport (2026-07-03): ported the double's collect-scatter-then-organize
to the paired layout (`seed_scatter_row_tiles_paired_f32`). KEY: rev(s+eighth) =
rev(s)+2 (verified all sizes) makes a paired slot's cA/cB outputs contiguous, so
each chunk is ONE V4F store {cA.lo,cA.hi,cB.lo,cB.hi} -- structurally identical to
`seed_scatter_row_tiles_v` (V2->V4F, 2*col->4*col, row_base 2*->4*). Reads stride
vwork (scattered loads); writes land in cache-local output rows. Two-phase split
(full-array top levels + per-block cache-resident bottom levels) like the double.
Result: even inverse inv_x 0.97-0.98 at N>=65536 (was 1.7-2.0); rt_x ~0.96-1.00.
Small even N (nb<64) uses per-pair block scatter, like the double.

Guard (2026-07-03): the wide f32 path is under `#if BRUUN_LEVEL >= 1` (same guard
as the rest of the SIMD) and uses ONLY the 128-bit `V4F` macros -- exactly how
the DIF handles NEON *and* SSE (one 128-bit abstraction; `V4F_CATLO/CATHI/ZIPLO`
cover both). NEON is always 128-bit; AVX2 (256) is the only separate thing and
is the x86 companion's job. The terminals do the cA/cB lane crossing in V4F
(`CATHI` aligns aB/va, `ZIPLO` is the mirror crossing, `CATLO` restores the
paired layout); the seed gathers via `V4F_SET4`. No `float32x2_t`, no scalar
terminal, no arch-specific `#if`. Compiles + runs on NEON, x86-SSE2, x86-AVX2.
Forward fwd_x ~0.94-1.07 (parity DIF). Inverse still ~1.7-2.0x DIF at N>=65536 --
that is the scattered-`rev` seed-scatter *transport*, not the abstraction; fix is
the double inverse's output-order / row-tile scatter (task below).
| forward SIMD AVX2 256 | [ ] handoff | [ ] handoff |
| inverse SIMD AVX2 256 | [ ] handoff | [ ] handoff |

f32 wide NEON measured (dif_vs_dit_f32_benchmark, even sizes, DIT/DIF ratio):
```text
N       1024  4096  16384  65536  262144  1048576
fwd_x   1.01  1.03  1.01   0.95   0.82    0.91
inv_x   0.99  0.99  1.06   1.59   1.66    1.90
rt_x    0.99  0.99  1.05   1.29   1.28    1.37
```
Roundtrip 1.6e-7..1.4e-6 (matches f32 scalar, flat to 1M); forward matches the
double reference bin-for-bin at f32 precision.

Follow-ups (tracked): (a) wide inverse large-N transport -- port the double
inverse's bit-reversed seed-scatter block order + row-tile transport to the
paired seed_scatter (the split interior already vectorizes); (b) odd log2 sizes
-- need a paired radix-2 terminal (currently route to f32 scalar); (c) fold the
forward's fused-seed/bitrev block order into compute_vwork_v4f (currently a
plain schedule).

Public entry points on `DIT_RFFT_kernel`:
- double: `forward_scalar` / `forward_simd` / `inverse_scalar` / `inverse_simd`
  (complex_t, work = work_size() doubles)
- f32: `forward_scalar_f32` / `forward_simd_f32` / `inverse_scalar_f32` /
  `inverse_simd_f32` (complex_f32_t, work = work_size() floats)

Notes:
- double complex fwd/inv, scalar+NEON-SIMD: normalized basis, roundtrip ~1e-15,
  beats DIF both directions 512..1048576. Shipped.
- f32 scalar: templated walk (`*_t<float>`) reading the double twiddle tables
  cast to float at use (higher-precision twiddle constant is a free f32 win).
  Genuine float arithmetic; roundtrip 1.8e-7..9.5e-7 (flat across N).
- f32 SIMD: `forward_simd_f32`/`inverse_simd_f32` exist and are correct, but
  currently call the f32 scalar path. The 4-wide `bruun_v4f` walk is designed
  (see below) but not yet hand-rolled -- it is the same class of SIMD-widening
  work as the AVX2 256-bit port.

## Wide-float NEON SIMD (v4f) design -- next increment
- Packing: one `bruun_v4f` = two seed blocks x two mirror lanes =
  {A.lo, A.hi, B.lo, B.hi}. Pair two blocks that never merge with each other
  until the terminal, so they share every twiddle at every level.
- Interior merges: byte-identical to the double `*_v_norm` walk with
  V2_* -> V4F_*, V2_SET1 -> V4F_SET1 (pure vertical, twiddle broadcast to all
  4 lanes). pair_reduce_v4f_norm / pair_expand_v4f_norm are 1:1 transcriptions.
- Only new logic: seed (interleave two rev-gathered contiguous mirror pairs via
  vld1_f32 + vcombine_f32) and terminal (per-block lane crossing on lanes {0,1}
  and {2,3} -- two independent trn/zip pairs).
- Open sub-question: choosing the second (block) pairing axis so partners stay
  twiddle-independent through the global sweeps, analogous to how the double
  walk's b vs b+N/4 top-bit split does. Needs the same care as the double walk.

## AVX2 256-bit (companion model, x86) -- handoff
- Today all SIMD is behind the V2_/V4F_ 128-bit macros, which already compile on
  x86 SSE2 (BRUUN_LEVEL>=1) and AVX2 (>=2). The double DIT SIMD therefore runs on
  x86 128-bit as-is.
- 256-bit widening = add `bruun_v4` (4xf64) + `V4_*` macros and `bruun_v8f`
  (8xf32) + `V8F_*` macros in bruun_dif_kernel.hpp's abstraction block, then
  provide `*_v_norm` variants at that width. Insertion points are the same
  `compute_vwork_norm` / `terminal_*_v_norm` / `split4_v_norm` boundaries.
- User decision: no dead 256-bit stubs now; macros + these notes only.

## Not in scope for DIT (by decision)
- [n/a] mag/phase forward+inverse  -> dedicated kernel taps complex bins
- [n/a] magnitude-only forward      -> same
- [defer] split re[]/im[] arrays    -> scalar writer exists internally; finish later if needed
- [n/a] native / residue / filter   -> DIT is inherently standard-order; no Bruun heap order

## Vector abstractions
- `bruun_v2`  = 128-bit 2xdouble (NEON float64x2_t / SSE __m128d)  -- double SIMD
- `bruun_v4f` = 128-bit 4xfloat  (NEON float32x4_t / SSE __m128)   -- f32 SIMD
- (future, companion model) `bruun_v4` = 256-bit 4xdouble, `bruun_v8f` = 256-bit 8xfloat

## Open items / questions
- (answered) formats=complex-only, f32=genuine, avx2=macros+notes.
- f32 SIMD lane packing: block-parallel v4f (two mirror-pairs from two seed
  blocks at the same tree position, sharing one broadcast twiddle) keeps every
  merge purely vertical, same as the double v2 walk. TBD confirm vs measured.
