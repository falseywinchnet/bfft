# Radix-4 Standard Basis AVX2 Handoff

## Status 2026-07-02: normalized-basis SIMD forward + inverse (NEON)

The class is now `bruun::DIT_RFFT_kernel` in
`experimental_radix4_rfft_kernel.hpp`. Codex moved the scalar merge/expand to
the **normalized (unit-frame) basis**: instead of the raw residue pair with a
`c2 = 2 cos(theta)` twiddle (which grows toward the tree top and forms
`1 - c2^2` cancellations, giving ~e-12 roundtrip at N=1M), each merge now runs
a Givens rotation with `c^2 + s^2 = 1` exactly. At a leaf the residue pair is
directly `(re, -im)`, so the forward projection and the inverse deprojection
both collapse to a sign flip.

`forward_simd` / `inverse_simd` are now the **normalized mirrored-lane SIMD
path** (this was the work of this session), the exact-inverse pair of each
other:

- Forward: `compute_vwork_norm` -> `terminal_v_norm` / `terminal_radix2_v_norm`.
  Primitive: `pair_reduce_v2_norm` (rotation). Tables: `cs_` = `{c, s}` parallel
  to `twiddle_`; `fterm_cs_` = terminal `{cA,sA,cBlo,cBhi,sBlo,sBhi}`.
- Inverse: `terminal_inv_v_norm` / `terminal_radix2_inv_v_norm` ->
  `compute_vwork_inv_norm`. Primitive: `pair_expand_v2_norm` (inverse rotation),
  reusing the SAME `cs_` table (identical 2*(i-1) index offsets as the old
  `inv_twiddle_`). Seed scatter is basis-independent and reused verbatim.

Measured (NEON, `dif_vs_dit_benchmark`, DIT/DIF ratios; <1 means DIT wins):

```text
N        512   1024  2048  4096  8192  16384 32768 65536 131072 262144 524288 1048576
fwd_x   0.95  0.98  1.00  1.00  0.93  0.96  0.95  0.94  0.89   0.92   0.97   0.94
inv_x   0.81  0.83  0.88  0.89  0.70  0.80  0.87  0.86  0.89   0.88   0.92   0.98
rt_x    0.87  0.90  0.90  0.84  0.92  1.05  0.91  0.92  0.94   0.90   0.97   0.91
dit_rt (roundtrip maxerr): 6.7e-16 .. 2.4e-15 at every size (incl. odd log2)
```

The SIMD forward matches the scalar forward's error vs a long-double DFT
bin-for-bin (verified). Roundtrip is machine precision because inverse is the
exact algebraic inverse of forward. `make test` passes standard,
`BFFT_USE_DIT_RFFT=1`, and `BFFT_USE_DIT_RFFT=1 BFFT_USE_DIT_RFFT_INVERSE=1`
(policy `dit-forward-dit-inverse`).

The old unnormalized SIMD (`pair_reduce_v2`, `merge4_v`, `terminal_v`,
`split4_v`, `terminal_inv_v`, `compute_vwork_inv`) and the interpolation inverse
(`inverse_complex_interp_simd`) remain in the file as reference/oracles but are
no longer routed. AVX2 work below should widen the `_norm` variants.

## (Older notes)


The main `bfft_forward()` / `bfft_inverse()` double-precision standard path is
now the radix-4 basis pair in `radix4_rfft_kernel`:

- forward: mirrored-lane radix-4 DIT, standard bins out.
- inverse: CRT interpolation up the real factor tree, standard bins in,
  ordered real samples out.

The old `bruun::RFFT` object is no longer owned by `bfft_plan`. BODFT remains a
separate kernel and should not be touched as part of this RFFT migration.

Current production integration:

- `src/bfft.cpp`
  - `bfft_plan` owns `bruun::radix4_rfft_kernel standard`.
  - `bfft_plan_standard_policy()` reports `"radix4-dit-interp-standard"`.
  - `bfft_forward()` calls `standard.forward_complex_simd(...)`.
  - `bfft_inverse()` calls `standard.inverse_complex_interp_simd(...)`.
  - `bfft_inverse_workspace()` provides caller-owned inverse scratch.
- `src/detail/radix4_rfft_kernel.hpp`
  - Contains the optimized forward and the interpolation inverse.
  - Uses the shared 128-bit `bruun_v2` abstraction today, even on AVX2 builds.
  - Keeps the phase/sincos table helpers from `bruun_kernel.hpp` available for
    magnitude/polar conversion paths.

Public API shape:

- Standard complex, magnitude, and mag/phase APIs remain.
- Native/residue/filter APIs have been pruned from `bfft.h` / `bfft.hpp`.
- Float standard APIs currently bridge through the double radix-4 kernel with
  plan-owned scratch. They are correctness placeholders, not the final fast
  float basis.

## AVX2 Targets

### Forward DIT

1. `seed_block_v`
   - Current shape packs one mirrored slot per 128-bit vector:
     `{node s, node s + N/4}`.
   - AVX2 should process two slots per vector while preserving the key gather
     property `rev(b + N/4) == rev(b) + 1`.

2. `pair_reduce_v2` / `butterfly4_v` / `merge4_v`
   - Below the terminal, math is vertical SIMD with broadcast constants.
   - A 4-lane double form should avoid introducing per-stage shuffles.
   - Watch register pressure; the 128-bit loop is dense and intentionally
     avoids spills.

3. `terminal_v` / `terminal_radix2_v`
   - This is the only lane-crossing zone in the forward.
   - Widen only after measuring; output bins are structurally spread.

### Interpolation Inverse

1. `itp_head8_v` / `itp_head16_v`
   - This is the inverse's single shuffle zone, fused with deprojection.
   - Odd-log2 sizes benefit from `itp_head16_v`; keep that register fusion.

2. `itp_merge4_v`
   - This should be an ideal AVX2 target: constants vary per block, not per
     position, so interior CRT merges are vertical SIMD.

3. `itp_upper_walk`
   - Preserve the L2-sized tiering (`BRUUN_ITP_TIER2 = 32768` doubles).
   - L1-sized windows measured worse on NEON; remeasure on AVX2 before changing.

4. `itp_terminal_v`
   - Above `n = 16384`, the two-half-pass terminal avoids same-set aliasing.
   - AVX2 work should remeasure fused vs two-half terminal forms.

## Validation Commands

```sh
make test
make examples
make probes
build/examples/radix4_rfft_benchmark 256 1024 4096 16384 65536 1048576
build/examples/benchmark 65536 256
make asm-check BUILD_DIR=build-avx2
```

Expected current behavior:

- `bfft_plan_standard_policy()` returns `"radix4-dit-interp-standard"`.
- `radix4_rfft_benchmark` reports public `bfft_standard_ns` close to direct
  `radix4_simd_ns`, and public `bfft_inverse_ns` close to direct
  `radix4_inverse_interp_simd_ns`.
