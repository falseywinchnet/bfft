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

## Single precision

The float32 API shares the public plan and layout maps with the double Bruun
path, but it uses dedicated single-precision work buffers and complex storage.
Internal float32 helper loops use the same backend level selected for the
double path for work-buffer setup, inverse scaling, and real-output copy. The
public API still exposes only layout-named calls, not transform-selection
compile flags.

The current float32 transform keeps its hot butterfly arithmetic in float. Its
stage twiddles come from plan-owned float32 root tables generated once with
explicit `float(...)` rounding from double libm values. This avoids the
multiplicative float twiddle recurrence drift that produced deterministic BH7
folded-bin spurs near bins such as `N/2 - k`, while keeping the runtime path
single precision.

The tracked BH7 probe documents the current acceptance behavior. On an Apple M4
NEON build at `N = 4194304`, sampled with eight eligible bins and sine/cosine
waves, native float32 BFFT measured 144.95579274 dB SFDR against an FFTWf row of
139.16529588 dB. Native double measured 197.84799575 dB against FFTW double at
197.84799573 dB.

## Build and package metadata

The Makefile remains the simplest local workflow and installs headers, static
and shared libraries, and `lib/pkgconfig/bfft.pc`. CMake supports the same core
build and test path, optional probes, staged installation, `pkg-config` metadata,
and package config files under `lib/cmake/bfft`. Installed CMake consumers can
link `bfft::static` and, when enabled, `bfft::shared`.

## Future portability work

The beta workflow validates Linux make/install and Linux CMake install first.
The directory layout avoids Linux-specific API assumptions so future work can add
Windows DLL export annotations, macOS install-name tuning, broader package
recipes, and larger CI matrices without changing the public API shape.
