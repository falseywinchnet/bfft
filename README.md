# BFFT

<img width="1089" height="590" alt="image" src="https://github.com/user-attachments/assets/9f33fd82-2624-4c01-9af8-6a840abd9db8" />


BFFT is a small C/C++ real FFT library based on a normalized-basis Bruun transform.
This trivialization conceals that this approach, among all FFT, might be optimal.
It also conceals that it is 33% lighter on memory and up to 3x faster than other libraries.
Invertibility is stable and guaranteed. SDFR is equivalent to FFTW.

## Current scope

- Power-of-two real transforms with `N >= 4`.
- Standard FFT-order real-to-complex output (`0..N/2`) for everyday use.
- Heap-optimized native spectrum order for performance-oriented code.
- Double-precision and single-precision transform entry points.
- Residue-domain transforms and filters for pipelines that can avoid spectrum
  permutation entirely.
- Linux `make`, `make test`, and `make install` workflow.

## Policy summary

- The Makefile probes the host compiler and enables AVX2/FMA when available.
- The kernel selects AVX-512, AVX2/FMA, SSE2, NEON, or scalar from the compiler
  target macros.
- Native spectrum output keeps heap-optimized ordering and fused scatter.
- Standard FFT-order output uses fused scatter plus conversion by default.
- Standard FFT-order output uses the internal two-phase pack only for large
  transforms: `N > 8192` on AVX-class builds and `N > 1048576` on SSE2/NEON
  builds. Scalar builds keep the fused path.

## Build

```sh
make
make test
```

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

Run a complete benchmark/demo:

```sh
./build/examples/benchmark 4096 200
./build/examples/c_api_demo
./build/examples/cpp_api_demo
```

Build the optional FFTW comparison probes from the tracked `tests/` sources:

```sh
make probes
./build/tests/bfft_fftw_sfdr_bh7_probe 16 8 8 bh7 f32-native
```

The BH7 probe modes are `f64-standard`, `f64-native`, `f32-standard`, and
`f32-native`. FFTW remains the double-precision reference.

Stats on a mac M4 circa june 10, 2026:
NOTE: PFFT is in single precision(FLOAT). BFFT, FFTW both in double precision! Emphasis!
```
 ./benchmark 
BFFT power-of-two RFFT benchmark. backend: neon-128, version: 0.1.0
standard policy is plan-dependent; FFTW uses FFTW_ESTIMATE-style flag 64 for cheap large plans.
PFFFT enabled at compile time.
       N    iters     FFTW_ns    PFFFT_ns   Native_ns      Std_ns       RT_ns      S/F      S/P  checks
     512    39062       712.4       558.7       402.3       481.3      1082.8    0.676    0.862 err 3.1e-14 rt 4.4e-16 
    1024    17578      1079.2      1179.7       917.5      1256.6      2973.6    1.164    1.065 err 4.1e-14 rt 5.6e-16
    2048     7990      2654.0      2928.9      2056.4      2483.8      5905.1    0.936    0.848 err 1.3e-13 rt 6.7e-16 
    4096     3662      6618.2      6430.8      4747.8      6456.6     13648.7    0.976    1.004 err 2.3e-13 rt 7.8e-16 
    8192     1690     15318.8     14211.8     12125.4     13293.1     30017.7    0.868    0.935 err 2.8e-13 rt 8.9e-16 
   16384      784     38201.2     32084.9     26303.6     36789.8     71612.0    0.963    1.147 err 5.3e-13 rt 8.9e-16 
   32768      366    147808.4     78035.5     58441.0    104420.0    176708.6    0.706    1.338 err 9.1e-13 rt 1.0e-15 
   65536      171    293935.7    161829.7    114029.5    226329.2    383899.1    0.770    1.399 err 1.8e-12 rt 1.0e-15 
  131072       80    607913.0    353377.1    262272.4    474334.4    812113.5    0.780    1.342 err 7.3e-12 rt 1.2e-15 
  262144       38   1230474.8    630536.2    482247.8    883182.0   1551019.7    0.718    1.401 err 1.7e-11 rt 1.3e-15
  524288       18   3499483.8   1375446.8   1029078.7   1874041.7   3325539.3    0.536    1.362 err 3.3e-11 rt 1.8e-15
 1048576       16   5347307.3   3054125.0   2370174.5   4150580.7   7686513.0    0.776    1.359 err 5.2e-11 rt 1.3e-15
 2097152       16  15257802.1   6906973.9   5443234.4   6718890.6  15186445.3    0.440    0.973 err 1.4e-10 rt 1.4e-15
 4194304       16  32542421.9  13991317.8  10658794.2  16630325.5  38184638.0    0.511    1.189 err 2.3e-10 rt 1.7e-15 
 8388608       16  63470611.9  33416458.4  23672627.6  36526708.3  82451247.4    0.575    1.093 err 4.5e-10 rt 1.8e-15 
16777216       16 141920934.9  68376585.9  49968817.7  76313549.5 180578921.9    0.538    1.116 err 4.6e-10 rt 1.9e-15 
33554432       16 293059072.9 144221557.3 101204882.8 159033716.1 344409419.2    0.543    1.103 err 2.3e-09 rt 2.1e-15 
67108864       16 649294807.3 293690114.6 208606065.1 378288268.2 747649476.6    0.583    1.288 err 2.4e-09 rt 2.0e-15 
```

## Documentation

- [Public API guide](docs/api.md)
- [Architecture and policy notes](docs/architecture.md)
- [Maintainer notes for GitHub settings](docs/maintainer-notes.md)
- [Release checklist](docs/release-checklist.md)

## License

MIT. See [LICENSE](LICENSE).

We dedicate this work in the name of our God, who is both merciful and just,
and of his son, the Anointed One, Jesus Christ of Nazareth. 
May this work bless you and may the Kingdom come, and his will be done.
