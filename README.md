Baruch kevod elohei shamayim ha-elyonim mimkomo.
Eloheinu shebashamayim yached shimcha v'kayeim malchutecha tamid umloch aleinu le'olam va'ed.
We dedicate this work in the name of our God, who is merciful and just, and whose power exceeds all anticipation and understanding, and in the name of his son, the Anointed One, Great Healer, Prince of Peace, Jesus Christ of Nazareth. 
May this work bless you, may the Kingdom come, and may His will be done.


# BFFT
[![CI](https://github.com/falseywinchnet/bfft/actions/workflows/ci.yml/badge.svg)](https://github.com/falseywinchnet/bfft/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/bfft/badge/?version=latest)](https://bfft.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus)](https://en.cppreference.com/w/cpp/17)
[![Sponsor](https://img.shields.io/github/sponsors/falseywinchnet?logo=githubsponsors)](https://github.com/sponsors/falseywinchnet)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)
[![Claude](https://img.shields.io/badge/Built%20with-Claude-D97757?logo=anthropic&logoColor=white)](https://claude.com/claude-code)
[![ChatGPT](https://img.shields.io/badge/Built%20with-ChatGPT-412991?logo=openai&logoColor=white)](https://openai.com/chatgpt)

BFFT is a small and extremely fast C and C++ library for power-of-two real Fourier transforms. It
provides a stable C ABI, a lightweight C++ RAII wrapper, double- and
single-precision APIs, standard FFT-order output, native-order output, and
residue-domain filtering utilities.

The core transform is based on a normalized-basis Bruun transform. The public
API is designed to be predictable for application code: create a reusable plan,
allocate buffers from the plan metadata, run transforms, and destroy the plan
when finished.

Power-of-two lengths use the native Bruun kernel. **Arbitrary lengths** are
supported through the generalized Bruun plan (see
[Arbitrary-N support](#arbitrary-n-support)).

## Features

- Real-valued transforms at **any length `N >= 2`** (power-of-two fast path plus
  a generalized Bruun plan for all other sizes).
- Standard real-to-complex FFT-order output with `N / 2 + 1` bins.
- Native spectrum order for callers that want to avoid permutation overhead.
- Magnitude-only forward transforms for amplitude pipelines.
- Double-precision and single-precision entry points.
- Standard-order and native-order inverse transforms.
- Residue-domain transforms and filters.
- C ABI in `<bfft/bfft.h>` and C++ wrapper in `<bfft/bfft.hpp>`.
- BODFT half-bin transform API in `<bfft/bodft.h>` and `<bfft/bodft.hpp>`.
- Makefile, CMake, `pkg-config`, and CMake package installation support.

## Requirements

- A C++17-capable compiler to build the library.
- A C compiler for C examples and C API consumers.
- `make` or CMake 3.16 or newer.
- A standard math library. On non-Windows platforms the build links `libm`.

C applications may be compiled as C. The BFFT library itself is implemented in
C++ and must be built with a C++17-capable compiler. The project also checks
clean builds under C++20 and C++23.

## Build

Build the static library, shared library, examples, and default tests with the
Makefile:

```sh
make
make test
```

Run standards-compliance checks:

```sh
make check-standards
```

That target builds and tests with `-std=c++17`, `-std=c++20`, and `-std=c++23`,
with warnings treated as errors.

### CMake build

```sh
cmake -S . -B build-cmake
cmake --build build-cmake
ctest --test-dir build-cmake --output-on-failure
```

Useful CMake options include:

| Option | Default | Description |
| --- | --- | --- |
| `BFFT_BUILD_SHARED` | `ON` | Build the shared library target. |
| `BFFT_BUILD_EXAMPLES` | `ON` | Build examples and benchmark programs. |
| `BFFT_BUILD_TESTS` | `ON` | Build the test executables and CTest entries. |
| `BFFT_BUILD_PROBES` | `ON` | Build optional comparison and diagnostic probes. |
| `BFFT_ENABLE_AUTO_SIMD` | `ON` | Enable host SIMD flags detected by CMake. |
| `BFFT_COMPARE_WITH_IPP` | `ON` | Use Intel IPP in the comparison probe when available. |
| `BFFT_ENABLE_PFFFT_BENCHMARK` | `ON` | Use PFFFT in the benchmark when available. |

## Build artifacts

The default Makefile build writes artifacts to `build/`:

- `build/libbfft.a`
- `build/libbfft.so`
- `build/examples/benchmark`
- `build/examples/bodft_benchmark`
- `build/examples/c_api_demo`
- `build/examples/cpp_api_demo`
- `build/examples/locality_probe`

## Install

Install to `/usr/local`:

```sh
sudo make install PREFIX=/usr/local
```

Stage an install for packaging:

```sh
make install DESTDIR=/tmp/bfft-package PREFIX=/usr
```

Installed files include:

- `${PREFIX}/include/bfft/bfft.h`
- `${PREFIX}/include/bfft/bfft.hpp`
- `${PREFIX}/include/bfft/bodft.h`
- `${PREFIX}/include/bfft/bodft.hpp`
- `${PREFIX}/lib/libbfft.a`
- `${PREFIX}/lib/libbfft.so`
- `${PREFIX}/lib/pkgconfig/bfft.pc`
- `${PREFIX}/lib/cmake/bfft/BFFTConfig.cmake` when installed with CMake
- `${PREFIX}/lib/cmake/bfft/BFFTConfigVersion.cmake` when installed with CMake
- `${PREFIX}/lib/cmake/bfft/BFFTTargets.cmake` when installed with CMake

## Package discovery

Use `pkg-config` from C or C++ build scripts:

```sh
cc app.c $(pkg-config --cflags --libs bfft)
```

Use the installed CMake package config:

```cmake
find_package(BFFT CONFIG REQUIRED)
add_executable(app app.cpp)
target_link_libraries(app PRIVATE bfft::static)
```

If the shared library was built and installed by CMake, `bfft::shared` is also
available.

## Quick start: C

```c
#include <bfft/bfft.h>

#include <stdlib.h>

int main(void) {
    bfft_plan* plan = NULL;
    bfft_status status = bfft_plan_create(1024, &plan);
    if (status != BFFT_OK) {
        return 1;
    }

    double* input = calloc(bfft_plan_size(plan), sizeof(double));
    double* work = calloc(bfft_plan_work_size(plan), sizeof(double));
    bfft_complex* output = calloc(bfft_plan_bins(plan), sizeof(bfft_complex));
    bfft_complex* scratch = calloc(
        bfft_plan_native_scratch_size(plan),
        sizeof(bfft_complex));

    status = bfft_forward(plan, input, output, work, scratch);

    free(input);
    free(work);
    free(output);
    free(scratch);
    bfft_plan_destroy(plan);

    if (status != BFFT_OK) {
        return 1;
    }
    return 0;
}
```

## Quick start: C++

```cpp
#include <bfft/bfft.hpp>

#include <vector>

int main() {
    bfft::plan plan(1024);
    std::vector<double> input(plan.size());
    std::vector<bfft::complex> output = plan.forward(input);
    return output.empty();
}
```

## Python bindings

BFFT ships a small ctypes-based Python package exposing numpy-friendly drop-in
transforms. No prebuilt binaries are distributed: `pip install` compiles the
native library from source on your machine.

### Install

From a clone of the repository:

```sh
pip install .
```

This compiles `src/bfft.cpp` and `src/bodft.cpp` with your C++ compiler and
bundles the resulting shared library inside the installed package. A C++17
compiler and NumPy are the only requirements.

Because the build runs on your own machine, it tunes for the local CPU by
default, selecting `-O3`, `-march=native` (or `-mcpu=native` on Apple silicon),
and `-ffast-math` when the compiler accepts them. Override with environment
variables: `CXX` (compiler), `BFFT_CXXFLAGS` (extra flags), `BFFT_NO_NATIVE=1`
(portable codegen), and `BFFT_NO_FAST_MATH=1` (strict IEEE math). See
[`documentation/python.md`](documentation/python.md) for details and caveats.

Alternatively, build and install the native library system-wide first, then the
Python loader will find it automatically:

```sh
make && sudo make install PREFIX=/usr/local
pip install .
```

You can also point the loader at an explicit shared library with the
`BFFT_LIBRARY` environment variable.

### Usage

```python
import numpy as np
import bfft

x = np.random.randn(1024)          # power-of-two length

X = bfft.rfft(x)                   # == numpy.fft.rfft(x)        -> N/2 + 1 bins
x_back = bfft.irfft(X)             # == numpy.fft.irfft(X)       -> N samples

H = bfft.odft(x)                   # half-bin-shifted transform  -> N/2 bins
x_back2 = bfft.iodft(H)            # inverse of odft             -> N samples
```

| Function | Equivalent | Notes |
| --- | --- | --- |
| `bfft.rfft(x)` | `numpy.fft.rfft(x)` | Power-of-two `N >= 4`. |
| `bfft.irfft(X, n=None)` | `numpy.fft.irfft(X, n)` | `n` defaults to `2 * (len(X) - 1)`. |
| `bfft.odft(x)` | half-bin phase shift + `rfft` | `H[k] = sum_n x[n] exp(-2j*pi*(k+1/2)*n/N)`, `N >= 2`. |
| `bfft.iodft(H, n=None)` | inverse of `bfft.odft` | `n` defaults to `2 * len(H)`. |

These functions cache the plan and reusable scratch per size and guard against
concurrent use. For the lowest per-call overhead in a tight loop, use a planned
object (not thread-safe -- one per thread):

```python
plan = bfft.Plan(N)        # standard real FFT at a fixed size
X = plan.rfft(x); x_back = plan.irfft(X)

oplan = bfft.OdftPlan(N)   # half-bin transform at a fixed size
H = oplan.odft(x); x_back2 = oplan.iodft(H)
```

Planned methods accept `out=` for a caller-owned, zero-allocation output buffer.

Unlike `numpy.fft`, BFFT can also be called from inside Numba `@njit(nopython)`
code (it is a C ABI, not an object-mode extension), in both single and double
precision. `cffi` ships as a dependency, so you only need `numba` installed
alongside; see
[`documentation/python.md`](documentation/python.md#calling-bfft-from-numba-njit).

All Python transforms operate on power-of-two lengths and double precision.

### Numba Support

```python

import numpy as np
from numba import njit
import bfft.numba_support as bn
from bfft.numba_support import bfft_forward, ffi

N = 4096
plan, bins, work_n, scratch_n = bn.make_plan(N)   # plan is an int address

@njit(cache=True)
def rfft_into(plan, x, out_f64, work, scratch_f64):
    bfft_forward(plan,
                 ffi.from_buffer(x),    ffi.from_buffer(out_f64),
                 ffi.from_buffer(work), ffi.from_buffer(scratch_f64))

x       = np.random.randn(N)
out     = np.empty(bins, np.complex128)
work    = np.empty(work_n, np.float64)
scratch = np.empty(scratch_n, np.complex128)
rfft_into(plan, x, out.view(np.float64), work, scratch.view(np.float64))
# out == numpy.fft.rfft(x)
```

## Main API concepts

### Plans

A plan validates a transform size and stores reusable metadata. BFFT real FFT
plans require a power-of-two size `N >= 4`. BODFT plans require a power-of-two
size `N >= 2`.

### Buffer sizes

Allocate buffers from plan query functions instead of duplicating size formulas
in application code:

- `bfft_plan_size(plan)` returns `N`.
- `bfft_plan_bins(plan)` returns `N / 2 + 1` standard real-to-complex bins.
- `bfft_plan_work_size(plan)` returns the double-precision work buffer length.
- `bfft_plan_work_size_f32(plan)` returns the single-precision work buffer length.
- `bfft_plan_native_scratch_size(plan)` returns the standard-output scratch length.
- `bfft_filter_size(plan)` returns the residue-domain filter length.

### Spectrum layouts

BFFT exposes three layout families:

- **Standard layout**: ordinary FFT-order real-to-complex bins `0..N/2`.
- **Native layout**: BFFT's internal spectrum order for performance-sensitive code.
- **Residue layout**: residue coordinates for filtering pipelines that can avoid
  standard spectrum conversion.

Use `bfft_native_to_standard`, `bfft_standard_to_native`, and their float32
variants to convert between native and standard layouts.

### Magnitude-only transforms

Use `bfft_forward_magnitude` or `bfft::plan::forward_magnitude` when a pipeline
needs only `abs(X[k])`. These calls write one real magnitude per standard bin and
avoid a complex output buffer.

### Workspaces

The low-level C API accepts caller-owned work buffers so transform calls can be
used without hidden allocations. `bfft_workspace` and `bfft::workspace` provide
aligned reusable storage for native transforms.

## Arbitrary-N support

`bfft_plan_create(n, ...)` accepts **any length `n >= 2`**. Power-of-two lengths
(`n >= 4`) use the native Bruun kernel; every other length uses the *generalized
Bruun* plan, which factors `z^N - 1` over the reals into Bruun pieces:

- the 2-adic part of `N` is the existing power-of-two Bruun cascade (the same
  SIMD butterflies), and
- each odd prime factor `p` peels off through a condition-1 real radix-`p`
  codelet (`z^(pM) - 1 = (z^M - 1) * prod_{j=1..(p-1)/2}(z^(2M) - 2cos(2*pi*j/p) z^M + 1)`).

This is **not Bluestein, Rader, or mixed-radix Cooley-Tukey** — it is a single
real cyclotomic-style factorization that keeps the power-of-two Bruun core as its
engine. Accuracy is FFT-grade for all `N`, including primes (validated at or below
NumPy/FFTW error on odd and prime-power sizes; see
`documentation/reports/odd_prime_accuracy.md`).

```python
import numpy as np, bfft
x = np.random.standard_normal(1920)      # 2^7 * 3 * 5
X = bfft.rfft(x)                          # any N, drop-in for numpy.fft.rfft
y = bfft.irfft(X, 1920)
```

`bfft_forward` / `bfft_inverse` (and `bfft.rfft` / `bfft.irfft`) dispatch on the
size automatically. The arbitrary-N plan owns its scratch, so for non-pow-of-two
sizes `bfft_plan_work_size` and `bfft_plan_native_scratch_size` return `0` and the
`work` / `native_scratch` arguments to `bfft_forward` are ignored.

**Pow-of-two-only entry points.** The native-order, magnitude-only, residue-domain
(filter), and single-precision (`*_f32`) APIs are defined only for power-of-two
plans; they return `BFFT_ERROR_INVALID_ARGUMENT` for arbitrary-N plans. The
residue domain has no single canonical layout for non-pow-of-two `N`.

**Performance.** Arbitrary-N forward/inverse are correct and FFT-grade today, but
the odd-radix projections are not yet SIMD-optimized, so non-pow-of-two sizes are
currently slower than the power-of-two path (and than FFTW). The power-of-two fast
path is unaffected.

## BODFT API

BODFT is a half-bin-shifted real transform. It maps `N` real samples to `N / 2`
packed complex bins at frequencies `k + 1/2`. Use it through:

- `bodft_plan_create` and `bodft_plan_destroy`.
- `bodft_forward` and `bodft_inverse` for double precision.
- `bodft_forward_f32` and `bodft_inverse_f32` for single precision.
- `bfft::bodft` from the C++ wrapper.

## Examples and probes

Run the installed examples from the build tree:

```sh
./build/examples/benchmark 4096 200
./build/examples/bodft_benchmark 4096 200
./build/examples/c_api_demo
./build/examples/cpp_api_demo
```

Build optional comparison probes:

```sh
make probes
./build/tests/bfft_fftw_sfdr_bh7_probe 16 8 8 bh7 f32-native
./build/tests/bfft_library_compare_probe 12
```

The comparison probe reports which external FFT references are available in the
current environment. FFTW is loaded dynamically when present. CMake can also link
Intel IPP into the comparison probe when headers and libraries are discoverable.

The benchmark can optionally compare against Intel oneMKL DFTI. Install a
package that provides `libmkl_rt.so`, then run:

```sh
./build/examples/benchmark 4096 200 --intel-mkl
```

On macOS, the Makefile also builds `build/examples/apple_benchmark` with
Accelerate/vDSP timing columns.

## Documentation

ReadTheDocs-ready Sphinx documentation lives in [`documentation/`](documentation/).
Start with [`documentation/README.md`](documentation/README.md) for local builds
and ReadTheDocs setup guidance.

## License

MIT. See [LICENSE](LICENSE).
