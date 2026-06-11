# BFFT

BFFT is a small C/C++ real FFT library based on a normalized-basis Bruun transform.
The first release turns the prototype into an installable library with a stable
public include path, examples, tests, and an automatic backend/packing policy.

## Current scope

- Power-of-two real transforms with `N >= 4`.
- Standard FFT-order real-to-complex output (`0..N/2`) for everyday use.
- Heap-optimized native spectrum order for performance-oriented code.
- Residue-domain transforms and filters for pipelines that can avoid spectrum
  permutation entirely.
- Linux `make`, `make test`, and `make install` workflow.

## Policy summary

Applications no longer choose transform packing with public compile flags. Build
and runtime policy are automatic:

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

bfft_plan* plan = NULL;
bfft_plan_create(1024, &plan);

double input[1024];
double work[1024];
bfft_complex output[513];
bfft_complex scratch[513];

bfft_forward(plan, input, output, work, scratch);
bfft_plan_destroy(plan);
```

## Minimal C++ example

```cpp
#include <bfft/bfft.hpp>

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
```

## Documentation

- [Public API guide](docs/api.md)
- [Architecture and policy notes](docs/architecture.md)
- [Maintainer notes for GitHub settings](docs/maintainer-notes.md)
- [Release checklist](docs/release-checklist.md)

## License

MIT. See [LICENSE](LICENSE).
