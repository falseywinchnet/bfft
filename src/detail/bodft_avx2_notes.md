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

## 2026-06-23 follow-up

- Added a 256-bit AVX2/FMA double-precision inverse combine path that mirrors the forward paired radix-4 lane layout.
- The inverse path loads paired partner spectra in reverse lane order, reconstructs the child spectra in SoA registers, applies conjugate twiddle multiplies with FMA instructions, and writes all four child blocks through the shared AVX2 packing helper.
- Tightened the forward scratch schedule so the final forward combine writes directly into the caller output buffer instead of ping-ponging once more and copying `N/2` complex values at the end.
- BODFT double inverse timing improved in this container from about 748 us to about 611 us at N=65536, and from about 22.7 ms to about 20.6 ms at N=1048576. Smaller sizes saw larger relative gains where the final-copy removal and inverse AVX2 combine reduce overhead together.
- The remaining gap to the regular BFFT native path is now mostly scratch/data-order locality and float-specific AVX2 work rather than the absence of a double inverse vector combine.

## Float forward combine pass

- Added an eight-position AVX2/FMA single-precision forward combine path for the odd-frequency transform.
- The float path now deinterleaves eight packed complex twiddles/children into 256-bit SoA real/imaginary lanes, does all three child twiddle products with `vfmaddps`/`vfmsubps`, and writes the four paired output blocks through an AVX2 interleave helper.
- Partner bins are lane-reversed in registers with an AVX2 permute before storing, preserving the scalar/128-bit conjugate-pair layout while doubling the 128-bit float combine width on AVX2 hosts.
- The existing 128-bit SSE2/NEON float combine remains the fallback for non-AVX2 builds and for tiny levels that cannot fill an eight-position vector.
- Correctness was checked with the regular CTest suite and the standalone BODFT benchmark on an AVX2/FMA build; a separate SSE2/no-AVX build was also checked to keep the fallback path intact.

## Float inverse combine pass

- Added an eight-position AVX2/FMA single-precision inverse combine path that mirrors the existing double inverse algebra and reuses the float AVX2 deinterleave/interleave helpers.
- The path loads k/k+M and conjugate-partner blocks, reverses partner lanes in registers, reconstructs the four child spectra, applies conjugate twiddle rotations with `vfmaddps`/`vfmsubps`, and stores c0..c3 as packed complex children.
- In this container, a same-command BODFT benchmark comparison against the previous commit at N=65536 improved float inverse from about 1.13 ms to about 0.78 ms; N=1048576 improved from about 25.6 ms to about 18.4 ms. Forward timings remained noisy and mostly unchanged because this patch targets inverse combine work.
