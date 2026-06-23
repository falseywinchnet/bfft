# BODFT AVX2 pathway notes

## 2026-06-23

- Added a 256-bit AVX2/FMA double-precision forward combine path for the odd-frequency transform.
- The AVX2 code handles four paired radix-4 positions per iteration, twice the double-lane width of the SSE2/NEON path.
- The path uses structure-of-arrays lane registers for child spectra and twiddle tables, performs the three child twiddle complex multiplies with FMA instructions, then stores the forward and conjugate-partner outputs through one packing helper.
- Kept the existing 128-bit SIMD double path as the fallback tail and non-AVX2 implementation. Float remains on the portable 128-bit SIMD path because its current path already handles four positions per iteration and needs a separate eight-position AVX2 packing pass to be worthwhile.
- Added a Makefile assembly target for `src/bodft.cpp` so `make asm-check` emits both the regular BFFT AVX2 assembly and the BODFT AVX2 assembly.

## Validation notes

- Correctness was checked with `make test BUILD_DIR=build-avx2 CXXWARNFLAGS='-Wall -Wextra -Wpedantic -Werror'`.
- BODFT-specific correctness and timing were checked with `make build-avx2/examples/bodft_benchmark BUILD_DIR=build-avx2 && build-avx2/examples/bodft_benchmark`.
- Assembly was checked with `make asm-check BUILD_DIR=build-avx2` and by scanning `build-avx2/src/bodft_avx2.s` for AVX/FMA instructions.

## Performance observations from this container

- Backend reported by the benchmark: `avx2-fma-256`.
- Double forward timings from the AVX2 path were approximately 0.83 us at N=256, 3.88 us at N=1024, 19.2 us at N=4096, 458 us at N=65536, and 18.3 ms at N=1048576.
- Float forward timings remained on the existing 128-bit path and were approximately 0.71 us at N=256, 3.13 us at N=1024, 14.7 us at N=4096, 313 us at N=65536, and 11.9 ms at N=1048576.
- The next performance task is an eight-position AVX2 float combine and, after that, an AVX2 inverse combine if inverse parity becomes a requirement.
- For a same-container point comparison, `build-avx2/examples/benchmark 65536 200` measured the regular BFFT native double path at about 217 us for N=65536; the BODFT double forward path measured about 458 us for N=65536 in the broader BODFT benchmark. The gap is now concentrated in BODFT's transform-specific data movement/scratch schedule and unvectorized inverse rather than missing AVX2 forward-combine code.
