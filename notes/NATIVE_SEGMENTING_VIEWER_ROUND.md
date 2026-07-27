# Native segmentation viewer tightening

Date: 2026-07-27

This round moved measured hot loops behind stable C ABI boundaries without
changing the population law, enumerating candidates, or introducing an
all-pairs cell structure.

## Native kernels

`include/bfft/vision.h` and `src/vision.cpp` now expose:

- `bfft_vision_fast_march_first_label` — the continuous first-arrival walk,
  including source covectors, simplex parents, acceptance order, and heap
  diagnostics;
- `bfft_vision_hard_affine_fit` — fused cell mass/centroid/radius reductions,
  conditioned affine coefficients, and rendering;
- `bfft_vision_hard_basis_refit` — fused normal accumulation, direct
  fixed-small-system elimination, and rendering after residual ridge columns
  are appended;
- the previously ported curvature-population and soft-support diffusion
  kernels.

Every entry has an optional Python binding and an executable fallback. The
canonical viewer calls the native path automatically when the bundled library
contains the symbol.

## Numerical controls

On a nonconstant 73x91 tensor field with 47 sources, the native front and the
reference front had:

- identical owner labels;
- identical first and second parents;
- identical acceptance order;
- identical push and maximum-heap counts;
- maximum floating discrepancy below `2.8e-14` over distances, covectors,
  source covectors, and simplex fractions.

The native hard affine field agrees with the conditioned NumPy reference below
`3e-15`. The augmented hard-basis refit agrees below `3e-13` in regression
tests. The live path contains no `numpy.linalg.solve`; it performs the tiny
per-cell elimination directly while accumulating the measured normal terms.

Concurrent Meyer channel evaluation uses separate native plans. With four
requested lanes and the two-plane OKLab lightness/chroma representation, two
independent two-lane plans run concurrently. Outputs are bit-identical to the
serial channel evaluation.

## Measured result

The warm 475x475 Pikachu control used 1,194 cells, one characteristic refresh,
one residual ridge, curvature population, and 16 soft-support passes.

| measurement | executable fallback | native |
| --- | ---: | ---: |
| wall time | 1,063.3 ms | 1,036.1 ms |
| population/front/step | 354.5 ms | 326.1 ms |
| fit/ridge/score | 591.0 ms | 558.9 ms |

These are medians from three alternating warm runs in one process; the full
path improved by about 2.6%, with noticeable scheduler noise in the parallel
decomposition phase. Isolated kernel results are more revealing:

| kernel | executable reference | native | speedup |
| --- | ---: | ---: | ---: |
| conditioned hard affine fit | 24.94 ms | 2.23 ms | 11.18x |
| one-ridge augmented refit | 15.57 ms | 3.62 ms | 4.30x |
| continuous first-arrival walk | 181.76 ms | 164.83 ms | 1.10x |

The first-arrival port is therefore a clean ownership boundary, not the main
performance claim. Its reference was already efficiently compiled. The warm
profile now attributes roughly 0.38 seconds to the nine native Meyer splits
needed by target setup and three measured candidate scores, and roughly 0.27
seconds to two exact global first-arrival walks. Those are the remaining
structural costs; the formerly wasteful per-cell reduction and soft-support
bookkeeping are no longer dominant.
