# Benefit of retaining the degree-six blend

The isolated degree-six blend term has negligible reconstruction benefit on
the declared paper family. Its error changes are small and mixed, while a
matching fixed RGBA evaluation loop costs approximately 35% more on the M4
Mini. This result does not support promoting this representation change into
the browser engine on quality grounds.

## Matched construction

The study uses the original Float64 signed-current factor construction and
the existing nodal order coordinate. Each factor order is sampled on the
global fifth-step lattice and represented by a tensor quintic, A and B.
The two alternatives are:

1. Quintic: blend those samples, then perform the existing quintic collocation.
2. Retained: compile (1-beta)A + beta B exactly into degree six.

They use the same source samples, factor states at the sampled phases, and
bilinear beta. Neither applies support-range clipping, joint-current admission,
nor experimental sharpening. No adaptive analysis is added to either method.
The study therefore measures blend representation alone. It does not assert
equivalence to the final deployed atlas after its additional admission steps.

The 56 cases comprise all 28 declared smooth paper cases at each source side
17 and 33: oriented edges, windowed carriers, curved transitions and crossings.
Both original order representations can still differ from their continuous
factor outputs between collocation nodes.

## Reconstruction benefit

Continuous-field MSE integrates squared error against the analytic field over
the complete source domain. Tensor Gauss quadrature uses order 32 per source
cell axis and is independently checked at order 16. Pixel-area MSE compares
65-by-65 outputs against analytic averages over exactly the same footprints.
Aggregate percentages below are geometric means of paired MSE ratios, minus
one. Negative values indicate improvement.

| Source samples | Continuous-field MSE change | Pixel-area MSE change |
|---|---:|---:|
| 17 × 17 | -0.003503% | -0.004732% |
| 33 × 33 | +0.001961% | +0.000370% |

The largest continuous-field improvement among individual cases is 0.034010%;
the largest worsening is 0.024949%. For pixel-area MSE the corresponding
values are 0.038679% and 0.027345%.

For the four displayed paper examples at source side 17:

| Field | Continuous-field MSE change |
|---|---:|
| Edge | -0.004082% |
| Carrier | -0.006753% |
| Curved | +0.003960% |
| Crossing | -0.003946% |

On the 32-point quadrature inventory, the largest value change across all
cases is approximately 0.2413 of one 8-bit level at source side 17 and 0.0798
of one level at side 33. These are maxima over evaluated sites, not certified
supremum bounds. The largest whole-domain RMS correction among cases is only
0.00671 and 0.000966 levels respectively. Display quantization is not part of
the measurements. The comparison plate uses a separate, magnified signed
scale for differences.

## Cost

The CPU benchmark uses shared control lattices, four Float32 channels,
Float64 accumulation, and otherwise matching degree-five/degree-six
Bernstein point loops. It evaluates 512-by-512 points, alternates method
order, warms both methods, and reports nine-repetition medians.

Host: M4 Mini, arm64, macOS 26.5; Apple clang 21.0.0, `-O3 -lm`.

| Source side | Quintic | Degree six | Ratio |
|---|---:|---:|---:|
| 17 | 4.947 ms | 6.681 ms | 1.3505 |
| 129 | 4.954 ms | 6.685 ms | 1.3494 |
| 513 | 5.039 ms | 6.803 ms | 1.3501 |

These are CPU evaluation-loop timings, not browser frame times, WASM SIMD
timings, preprocessing times, or integrated full-warp timings. The arithmetic
inventory rises from 36 to 49 local controls; asymptotic shared-lattice storage
rises from 25 to 36 controls per source cell. A browser implementation may
have different costs, but there is no observed quality gain large enough here
to motivate that implementation solely to retain this term.

## Interpretation

The degree-six contribution comes from multiplying the variable part of beta
by the highest-order parts of B-A. It vanishes when beta is constant, when the
orders agree, or when both order representations have bidegree at most (4,4).
Exact multiplication preserves this contribution without any new analysis.

Let E be the quintic reconstruction error against truth and D the retained
correction. The change in mean squared error is exactly

\[
\Delta\mathrm{MSE}=2\langle E,D\rangle+\langle D,D\rangle.
\]

The sign depends on the correlation with the existing reconstruction error.
There is no implication that an exact blend of approximated order fields
must be closer to truth, or even closer to the original continuous factor
blend. The retained term is small in this census and has mixed correlation
with those residuals. This explains the measured mixture of improvements and
regressions without attributing it to display interpolation.

This experiment isolates one identifiable representation loss. It does not
exonerate support-range clipping, establish a universal degree-five optimum,
or evaluate the benefit of preserving other existing current information.

## Verification and reproduction

- All 288 exact-rational coefficient identities in `test_faithful_blend.py`
  pass, with equal-order and exact-integral checks.
- Every case checks the vectorized compiler against the scalar coefficient
  reference and its evaluated field against the explicit blend of A and B.
- Both alternatives agree at their original collocation nodes to less than
  7e-15 in this run.
- Four- and eight-point quadrature agree on the retained polynomial's pixel
  averages to less than 6e-16.
- Analytic pixel-reference quadrature at orders 8 and 12 agrees to less than
  8e-11. Continuous MSE ratios at orders 16 and 32 differ by less than 3e-14.
- The four displayed fields also compare the Float64 direct factor reference
  to the existing Float32 native factor implementation. Those discrepancies
  are recorded separately; native output is not substituted for the Float64
  analytic study.

Run the bounded numerical study locally using the existing NumPy/SciPy
environment (the checked Mini Python environments did not supply it):

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CONV_NATIVE_THREADS=1 \
 /Users/ultimussecundai/.cache/pmnw-release-python/bin/python \
 -m experiments.conv_warp.support_range_audit.retention_study \
 output/support_geometry/conv_blend_retention 257
```

Run the CPU timing on the Mini from the authoritative checkout:

```sh
/Users/ultimussecundai/.local/bin/m4build -- /bin/sh -c \
 'clang -O3 experiments/conv_warp/support_range_audit/retention_cost.c -o /tmp/conv-retention-cost -lm && /tmp/conv-retention-cost'
```

The result packet stores numerical source hashes; `cost.json` stores the
timing source hash, compiler, host and observed timing ranges. The numerical
study is Float64; exact arithmetic is used for the algebraic coefficient tests.

Artifacts are in `output/support_geometry/conv_blend_retention/`. The report
is also served at `http://127.0.0.1:8899/retention/`. The original comparison
page and the production engine are unchanged.
