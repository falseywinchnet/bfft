# BFFT completion idea

BFFT is complete for the first public library release when it is a small,
installable Bruun real FFT library with a coherent C ABI, a lightweight C++
wrapper, tests, examples, docs, and a validated Makefile workflow.

## Product shape

- Power-of-two real transforms for `N >= 4`.
- Standard FFT-order real-to-complex output for everyday users.
- Native spectrum order for users who want the internal heap-optimized layout.
- Residue-domain transform and filtering helpers for pipelines that can avoid
  spectrum permutation.
- Automatic backend and packing policy derived from the build target and plan
  size.
- No public transform-selection compile flags.

## Completion checklist

- Public C declarations in `include/bfft/bfft.h` are documented and tested.
- Public C++ wrappers in `include/bfft/bfft.hpp` are documented and tested.
- Build outputs include static and shared libraries.
- `make clean`, `make`, and `make test` pass from a clean tree.
- `make install DESTDIR=... PREFIX=...` stages the documented files.
- A downstream smoke program can build against the staged install.
- Examples cover the C API, C++ API, and benchmark path without becoming large.
- `README.md`, `docs/api.md`, and `docs/architecture.md` match the shipped API
  and policy.
- Release notes or a release checklist identifies validated platforms and known
  limits.
- GitHub-only repository settings are recorded in `docs/maintainer-notes.md`.
- No placeholder TODO markers remain in source, public docs, examples, or tests
  unless they describe post-release work in the roadmap.

## Out of scope for 0.1.0

- CMake packaging.
- Windows DLL export work.
- macOS install-name tuning.
- Prebuilt release artifacts.
- Multi-platform CI beyond notes or a lightweight first workflow.

- direct ABS grab:
  
  ```
  With a real-only Bruun interior, fused depth-first traversal, and leaf codelets that currently materialize re/im complex bins at the bottom. The leaf packers are where X[k].re and X[k].im are finally produced, and the recursive transform bottoms out in fused depth-3 codelets.

The magnitude-only version should not do this:

Bruun RFFT -> complex output -> magnitude pass

It should do this:

Bruun RFFT interior -> leaf codelet computes re/im in registers -> store magnitude only

That means each leaf codelet computes:

double re = ...;
double im = ...;
P[k] = re*re + im*im;      // power / magnitude squared

or, only if you truly need amplitude magnitude:

M[k] = std::sqrt(re*re + im*im);

The first target should be magnitude squared, not sqrt, because most spectral work wants power or relative magnitude anyway. That avoids the complex output buffer and avoids a later read-complex/write-mag pass.

For FFTW or Cooley-Tukey, the usual route is:

real input -> complex RFFT bins -> magnitude pass

So FFTW still pays for complex bin materialization. Bruun can fuse the magnitude at the leaves and never store phase at all. That is exactly the kind of thing this structure is good at.

There are two variants worth implementing.

The fastest one is:

forward_power_native(input, Pnative, work);

This gives Bruun-native spectral magnitude-squared order. It should be close to native Bruun timing, maybe slightly slower or maybe even similar, because you replace two 8-byte stores per bin with one 8-byte store plus two multiplies and an add. At large N, halving output bandwidth may matter more than the extra arithmetic.

The drop-in one is:

forward_power_standard(input, Pstandard, work, Pnative_tmp);

This computes native power, then does a double-only permutation to standard FFT bin order. That permutation is half the bandwidth of the complex standardization pass:

complex standardization:  read 16 bytes/bin, write 16 bytes/bin
power standardization:    read  8 bytes/bin, write  8 bytes/bin

So even if standard complex output was only 1.5–1.9× faster than FFTW, standard magnitude-only should be better than that because FFTW must add its own magnitude pass while Bruun avoids complex output.

The API to aad add is:

void forward_power_native(const double* input,
                          double* power_native,
                          double* work) const;

void forward_power_standard(const double* input,
                            double* power_standard,
                            double* work,
                            double* power_native_tmp) const;

void native_power_to_standard(const double* power_native,
                              double* power_standard) const;

For DC and Nyquist:

power[0]     = dc * dc;
power[N / 2] = nyq * nyq;

For every other RFFT bin:

power[k] = re*re + im*im;

Then, only optionally:

for k:
    mag[k] = sqrt(power[k]);

Benchmark columns should be:

FFTW_r2c_ns
FFTW_r2c_plus_power_ns
Bruun_native_complex_ns
Bruun_standard_complex_ns
Bruun_native_power_ns
Bruun_standard_power_ns

The likely result:

native power:   close to native complex, possibly better at large N
standard power: much cheaper than standard complex
FFTW+power:     slower than FFTW alone

the route is real. The magnitude-only case strengthens Bruun’s advantage because you can throw phase away before it ever hits memory.
```
