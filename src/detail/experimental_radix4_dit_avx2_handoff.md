# Radix-4 Standard Basis AVX2 Handoff

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
