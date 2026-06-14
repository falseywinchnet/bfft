BFFT is complete for the first public library release. it is a small,
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
- First-pass CI exists and the maintainer has instructions for enabling required
  checks on GitHub.
- Installed package metadata is available for CMake and `pkg-config` consumers.
- No placeholder TODO markers remain in source, public docs, examples, or tests
  unless they describe post-release work in the roadmap.

## Scope for 1.1

- Windows DLL export work.
- macOS install-name tuning.
- Prebuilt release artifacts.
- Package manager recipes beyond installed CMake and `pkg-config` metadata.

- direct ABS grab
  
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

- High-Speed DCT

```
  
  Yield a Chebyshev-Bruun kernel and API, with normal outputs using a DCT-order, but exposing chebyshev order endpoints also.
  
  DCTs are already the reciprocal-even half of a Fourier transform. Ordinary FFT-based DCT code pays for a cyclic complex object, then projects down to cosines. BFFT is already closer to the projection because its real quadratic leaves are conjugate-pair planes. A DCT kernel can go one step further: never build the conjugate-pair plane as a complex output. Work directly in the fixed reciprocal basis where a leaf is just evaluation at x = cos theta.
  
  For DCT-III, this is almost embarrassingly natural. The transform is evaluation of a Chebyshev series at roots of T_N. DCT-II is the transpose/inverse direction of the same object after scaling. DCT-IV is the half-shifted version, using basis functions cos((n + 1/2) theta) at theta = (k + 1/2) pi / N; it can be converted to adjacent Chebyshev terms with a diagonal endpoint scale. DCT-I is the endpoint/Lobatto case and is more annoying because of the two fixed endpoints and because the natural power is N - 1, not N.
  
  The modified kernel would replace the z-polynomial CRT with a Chebyshev CRT. Instead of factoring z^N - 1 into conjugate-pair quadratics, factor T_N(x) through T_{2m}(x) = 2 T_m(x)^2 - 1. The split is then T_m(x) = +sqrt(1/2) and T_m(x) = -sqrt(1/2). That is spiritually the same quarter-plane event you are staring at in the Bruun four-star, but now it lives inside a real cosine basis, not a complex residue basis.
  
  This is why it can be cheaper. The current BFFT low-level split still emits real quadratic residue coordinates, and then the public RFFT path has to convert residues to complex bins. The code is explicitly organized this way: it can run residue transforms first and then call residues_to_complex, while the standard path is an outer layout decision. The normalized split itself is also just real add/sub plus a c,s plane mix on a four-quarter block. A DCT kernel can keep the add/sub and local rotations, but its leaves are scalar cosine evaluations, so the final complex packing and imaginary/conjugate interpretation disappear.
  
  The best first target is DCT-III or DCT-IV, not DCT-I. DCT-III gives you the clean Chebyshev evaluation problem. DCT-IV gives you the cleaner no-endpoint energy distribution and probably the nicest benchmark story. DCT-II then comes from the inverse/transpose path once the scaling convention is nailed down. DCT-I should come later.
  
  The implementation experiment I would run is: build a scalar Chebyshev-Bruun DCT-III for power-of-two N, using the same half-angle table logic already in the constructor. The existing BFFT table is generated by the Bruun recurrence alpha(1) = pi/4, alpha(2m) = alpha(m)/2, alpha(2m+1) = pi/2 - alpha(m)/2, without a full cosine table. That recurrence is exactly the kind of angle tree a direct DCT wants.
  
  The expected win is not asymptotic. It is constant factor and memory traffic. You avoid materializing even extensions, avoid complex FFT bins that will be projected away, avoid part of the post-twiddle/post-pack path, and can keep native Chebyshev order as the hot output. Against a generic FFT-based DCT, this should be meaningfully better. Against FFTW-style hand-tuned REDFT kernels, the win is not automatic, but the native-pipeline version has a real shot because BFFT’s residue-native design already treats standard ordering as optional rather than sacred.
  
  The deeper answer to your basis question is: yes, DCT is probably where the “absorbed final rearrangement” idea becomes cleaner. In C2C, native quadratic leaves are two complex coefficients. In RFFT, they collapse to one complex bin under conjugacy. In DCT, they collapse further to one real Chebyshev evaluation under reciprocal-even symmetry. That is the basis where the skipped work is not a trick anymore; it is the actual transform domain.
  

```
