# Sigma optimization backport

Date: 2026-07-26

This is the production boundary drawn after validating the unfinished
`experiments/sigma_opt` work.  The governing constraint is that production
operates on the one owner/runner graph measured from the current rendering.
It does not enumerate candidate graphs or reuse a pattern across hypothetical
geometry.

## Promoted

### Fused normal assembly and rendering

`bfft.vision.coownership_graph` constructs the interaction graph of the
current measured ownership.  `assemble_normal` scatters each pixel's four
rank-one block contributions directly into that graph and carries `A^T b` in
the same pass.  `render_partition` evaluates both cell predictions and their
blend without materializing gather arrays.

The C++ implementations are public through `include/bfft/vision.h`; the
Python API falls back to the exact Numba reference when an older shared
library is loaded.

Pikachu, 256 pixels on the long side, 2,400 cells:

| path | one exact field fit |
|---|---:|
| design matrix + generic sparse products | 32.7 ms median |
| fused BFFT core | 22.7 ms median |

Maximum rendered-field difference was `6.7e-16`.

### Residual ridge scan

The per-angle histogram loop is a single fused residual sinogram.  The core
kernel preserves first-angle/first-bin tie behavior.  The isolated benchmark
measured about 8.2x over the original Python implementation.

### Exact selected inversion

Takahashi selected inversion now exposes every diagonal cell block of
`G^-1`.  It replaces 48 stochastic probes plus a shortlist of explicit
columns with exact prices for all current cells.  Runtime guards verify the
SuperLU symmetric-factor assumptions and raise so a caller can fall back.

Measured exact all-cell price time:

| cells | selected inversion | previous shortlist path |
|---:|---:|---:|
| 700 | 6.9 ms | 19.3 ms |
| 2,400 | 34.8 ms | 177 ms |
| 5,400 | 120.4 ms | 1,476 ms |

Agreement with explicit inverse columns was approximately `1e-12`.

### Single-stage decomposition objective

`SingleStageDecompositionObjective` computes the immutable target split once
and evaluates RGB, cartoon, and texture MSE for each reconstruction.  The
receiver-guided experiment now uses this object; target decomposition is no
longer repeated inside line search.

## Kept experimental

### Broad-field FFT

Reflect-boundary FFT convolution becomes useful only for broad full-frame
expected-gain smoothing.  At 1080p and sigma 16.2 it measured 400 to 239 ms
per plane.  It is not useful for the cell-local ridge scan.  BFFT's current
power-of-two-only transform would pad the needed 1080p grid to roughly three
times its useful area, so this remains a study until a general-size backend
or overlap-save plan exists.

### Fresh SPD LDLT

A fresh AMD-ordered Eigen LDLT was about 1.4x faster than SuperLU for the
2,400-cell affine normal system.  This is a useful future C++ factor layer,
especially as the filled factor is naturally dense in 3x3 cell blocks.  It
does not eliminate an outer reach iteration.

## Rejected

### Candidate-pattern Schur batching

The apparent small-fixture win reverses at 2,400 cells (about 0.7x).  More
importantly, candidate-pattern enumeration is the wrong abstraction.  It is
not in production.

### Bucket Dijkstra replacement

The packed bucket walk was 1.7x at 128/700 but slower at 256/2,400.  The
production walk remains unchanged.

### One fixed linear solve for full reach

No such exact solve exists under the current model.  Ownership, sigmoid blend
weights, and variable-projected affine coefficients make the full objective
nonlinear in reach.  The current linear system is already the exact solve of
one local Gauss-Newton model.  Its scalar factorization is only about 4 ms;
formation and measured-objective evaluation dominate.

### Blind randomized inverse probes

Exact selected inversion is both faster and more accurate at the tested
sizes.  The unfinished JL interpretation also overstated its guarantee:
relative preservation applies to block quadratic forms, while individual
near-zero off-diagonal entries have only additive control.

## HD smoke test

The viewer's full-resolution option uses every source pixel and downsamples
only the displayed texture.  A 1280x720 synthetic image with four TGFD passes,
eight flow sweeps, and 64 cells initialized in 7.3 s; the exact coupled solve
then took 0.53 s.  Resident memory was approximately 1.1 GB, so full
resolution is explicit rather than the default.

