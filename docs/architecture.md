# Architecture and implementation notes

## Layout

- `include/bfft/bfft.h` is the stable C ABI.
- `include/bfft/bfft.hpp` is the header-only C++ convenience wrapper.
- `src/bfft.cpp` owns the C ABI implementation and wraps the internal Bruun
  plan.
- `src/detail/bruun_kernel.hpp` contains the implementation kernel extracted
  from the original prototype.
- `examples/` contains runnable copy/paste style C and C++ demos plus the
  benchmark path.
- `tests/` contains correctness and public API tests used by `make test`.

## Backend selection

The user-facing library is not configured with transform-policy preprocessor
flags. The Makefile probes AVX2/FMA support and adds compiler switches when the
host compiler accepts them. Inside the kernel, SIMD level is derived from the
compiler target macros and falls back safely to SSE2, NEON, or scalar code.

## Packing policy

The prototype had compile-time packing choices. The library keeps both useful
paths internally and selects between them based on the requested public output:

- Native output: heap-optimized native order with fused scatter.
- Residue output: no spectrum pack.
- Standard output: fused scatter plus native-to-standard conversion for normal
  sizes; two-phase standard pack only when large FFT-order output is expected to
  win.

Current thresholds:

- AVX2/AVX-512: two-phase standard pack for `N > 8192`.
- SSE2/NEON: two-phase standard pack for `N > 1048576`.
- Scalar: fused scatter.

## Small-size compatibility

The prototype's Bruun index mapping is already standard ordered for tiny plans
below `N = 32`. The public conversion helpers preserve that behavior with a
straight copy for those tiny plans and use heap mappings for larger plans.

## Future portability work

The initial release validates Linux make/install. The directory layout avoids
Linux-specific API assumptions so future work can add CMake, Windows DLL export
annotations, macOS install names, package metadata, and CI matrices without
changing the public API shape.
