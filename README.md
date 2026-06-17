# BFFT 

<img width="1089" height="590" alt="image" src="https://github.com/user-attachments/assets/9f33fd82-2624-4c01-9af8-6a840abd9db8" />

BFFT is a small C/C++ real FFT library based on a normalized-basis Bruun transform.
Invertibility is stable and guaranteed. SFDR tracks FFTW in double precision and meets or exceeds SFDR and accuracy precision needs.

- Power-of-two real transforms with `N >= 4`.
- Standard FFT-order real-to-complex output (`0..N/2`) for everyday use.
- Standard FFT-order magnitude-only output for amplitude pipelines that do not
  need phase or complex scratch buffers.
- Heap-optimized native spectrum order for performance-oriented code.
- Double-precision and single-precision transform entry points.
- Residue-domain transforms and filters for pipelines that can avoid spectrum
  permutation entirely.
- Linux `make`, `make test`, and `make install` workflow.

## Build

```sh
make
make test
```

For standards-compliance checks, run:

```sh
make check-standards
```

That target builds and tests with `-std=c++17`, `-std=c++20`, and
`-std=c++23`, treating warnings as errors. The default build remains C++17.

CMake builds are also supported:

```sh
cmake -S . -B build-cmake
cmake --build build-cmake
ctest --test-dir build-cmake --output-on-failure
```

BFFT is implemented against a C++17 baseline and exposes both a stable C ABI and a C++ convenience wrapper. C consumers can compile their applications as C, while the library itself must be built with a C++17-capable compiler. CI also checks clean builds under C++20 and C++23.

The benchmark can optionally compare against Intel oneMKL DFTI without adding
a build-time dependency. Install an Intel MKL package that provides
`libmkl_rt.so`, then pass `--intel-mkl` to `build/examples/benchmark`; the
program loads `libmkl_rt` dynamically and adds `MKL64_ns`, `MKL32_ns`,
`S/MKL`, `F32/M`, `mkl64`, and `mkl32` printouts. If `libmkl_rt` cannot be
loaded, those columns remain `n/a`.

On macOS, `make` also builds `build/examples/apple_benchmark`. It is a copy of
the general benchmark with extra Accelerate/vDSP columns using
`vDSP_fft_zripD` and `vDSP_fft_zrip` for split real-to-complex FFT timing.

Artifacts are written to `build/`:

- `build/libbfft.a`
- `build/libbfft.so`
- `build/examples/benchmark`
- `build/examples/c_api_demo`
- `build/examples/cpp_api_demo`

## Install

```sh
sudo make install PREFIX=/usr/local
```

For packaging, stage an install with `DESTDIR`:

```sh
make install DESTDIR=/tmp/bfft-package PREFIX=/usr
```

Installed files:

- `${PREFIX}/include/bfft/bfft.h`
- `${PREFIX}/include/bfft/bfft.hpp`
- `${PREFIX}/lib/libbfft.a`
- `${PREFIX}/lib/libbfft.so`
- `${PREFIX}/lib/pkgconfig/bfft.pc` when installing with the Makefile or CMake
- `${PREFIX}/lib/cmake/bfft/BFFTConfig.cmake` when installing with CMake
- `${PREFIX}/lib/cmake/bfft/BFFTConfigVersion.cmake` when installing with CMake
- `${PREFIX}/lib/cmake/bfft/BFFTTargets.cmake` when installing with CMake


## Package discovery

After installation, `pkg-config` consumers can compile with:

```sh
cc app.c $(pkg-config --cflags --libs bfft)
```

CMake consumers can use the installed package config:

```cmake
find_package(BFFT CONFIG REQUIRED)
add_executable(app app.cpp)
target_link_libraries(app PRIVATE bfft::static)
```

If the shared library was built and installed by CMake, `bfft::shared` is also available.

## Minimal C example

```c
#include <bfft/bfft.h>

#include <stdlib.h>

bfft_plan* plan = NULL;
bfft_status status = bfft_plan_create(1024, &plan);
if (status != BFFT_OK) {
    return 1;
}

double* input = calloc(bfft_plan_size(plan), sizeof(double));
double* work = calloc(bfft_plan_work_size(plan), sizeof(double));
bfft_complex* output = calloc(bfft_plan_bins(plan), sizeof(bfft_complex));
bfft_complex* scratch = calloc(bfft_plan_native_scratch_size(plan), sizeof(bfft_complex));

bfft_forward(plan, input, output, work, scratch);
free(input);
free(work);
free(output);
free(scratch);
bfft_plan_destroy(plan);
```

## Minimal C++ example

```cpp
#include <bfft/bfft.hpp>

#include <vector>

bfft::plan plan(1024);
std::vector<double> input(plan.size());
std::vector<double> work(plan.work_size());
std::vector<bfft::complex> output(plan.bins());
std::vector<bfft::complex> scratch(plan.native_scratch_size());

plan.forward(input.data(), output.data(), work.data(), scratch.data());
```

For amplitude-only analysis, use `bfft_forward_magnitude` or
`plan.forward_magnitude(...)` with a real output buffer sized to `plan.bins()`.
Those calls produce standard FFT-order `abs(X[k])` values without allocating a
complex spectrum or native complex scratch.

Run a complete benchmark/demo:

```sh
./build/examples/benchmark 4096 200
./build/examples/c_api_demo
./build/examples/cpp_api_demo
```

Build the optional comparison probes from the tracked `tests/` sources:

```sh
make probes
./build/tests/bfft_fftw_sfdr_bh7_probe 16 8 8 bh7 f32-native
./build/tests/bfft_library_compare_probe 12
```

`bfft_library_compare_probe` reports which external FFT references are available in the current environment. The Makefile build always compiles the probe without hard dependencies and uses FFTW dynamically when present. The CMake build additionally links Intel IPP into that probe when the IPP headers and libraries are discoverable, usually through normal search paths or `IPPROOT`.

The BH7 probe modes are `f64-standard`, `f64-native`, `f32-standard`, and
`f32-native`. For float32 BFFT modes the probe uses FFTWf when `libfftw3f` is
available, and falls back to the double-precision FFTW reference otherwise.
The CSV includes an `fftw_precision` column so mixed-precision and same-precision
runs are explicit.

## License

MIT. See [LICENSE](LICENSE).

We dedicate this work in the name of our God, who is merciful and just, and whose power exceeds all anticipation and understanding,
and of his son, the Anointed One, Jesus Christ of Nazareth. 
May this work bless you and may the Kingdom come, and his will be done.

Baruch kevod elohei shamayim ha-elyonim mimkomo
Eloheinu shebashamayim yached shimcha v'kayeim malchutecha tamid umloch aleinu le'olam va'ed
