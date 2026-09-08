# First diagonal packet audit, 2026-09-04

The recovered N=64 construction extends exactly to N=4^r. Its carry has
a² independent b²-dimensional blocks, where a=2^floor(r/2), b=2^ceil(r/2).
The block operator Schmidt rank is exactly b in the displayed two-axis
partition. This is an algebraic Vandermonde result, not a numerical-rank
threshold claim. Blocks grow; fixed-size closure does not hold.

## Verified structure

| N | a² blocks | Block dimension | Separation rank |
|---:|---:|---:|---:|
| 4 | 1 | 4 | 2 |
| 16 | 4 | 4 | 2 |
| 64 | 4 | 16 | 4 |
| 256 | 16 | 16 | 4 |
| 1,024 | 16 | 64 | 8 |
| 4,096 | 64 | 64 | 8 |
| 16,384 | 64 | 256 | 16 |
| 65,536 | 256 | 256 | 16 |

Independent dense matrix checks through N=1024 reproduce the first five
support decompositions (`first_sweep.json`). The maximum monomial error in
that sweep is 6.50e-16, maximum carry action error 4.53e-15, and maximum full
normalized FFT error 2.78e-15. Larger block counts/dimensions follow from the
proved invariant labels; dense matrices were not built at those sizes.

The vector tests pass through N=65536, including the independently expressed
cancelled transition. Small-size carry basis-vector tests cover slopes 1,3,-1.
The native fused/cancelled/full routes all pass comparison to the same ordinary
complex DFT contract, with an independent direct DFT through N=256.

## What the cancellations reveal

The uncancelled factorization uses N(r+2 log2 b) radix-2 butterflies.
Two exact adjacent Fourier/inverse-Fourier cancellations reduce that to Nr,
the conventional count. Folding reversals into routing and fusing the carry
diagonal into butterfly reads removes standalone swaps and one data pass.

This is a useful executable autosorting schedule. It still consists of
a-DIT, inverse b-DIT, b-DIF, inverse a-DIF with three phase fields and
four routing passes. It therefore fails the stronger requested criterion of
an independent diagonal mechanism that does not pay through DIT/DIF-like work.
This conclusion concerns this subgroup/character family, not all diagonal walks.

## Native M4 results

Apple M4, Apple clang 21.0.0, -O3 -std=c++17 -fno-vectorize
-fno-slp-vectorize. Host load observed near the run: 1.64 / 1.50 / 1.45.
Numbers are median microseconds per ordinary complex transform from
`native_fused_m4.json`; nine batches, fixed method ordering. Plans and
allocation excluded. No method is a tuned production library.

| N | DIT | DIF | Stockham | Full packet | Cancelled | Fused |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.161 | 0.160 | 0.124 | 0.923 | 0.463 | 0.408 |
| 256 | 0.816 | 0.789 | 0.627 | 3.705 | 1.991 | 1.772 |
| 1,024 | 4.007 | 3.975 | 3.056 | 15.466 | 7.777 | 7.363 |
| 4,096 | 18.773 | 19.029 | 20.857 | 63.924 | 36.456 | 30.883 |
| 16,384 | 106.726 | 103.663 | 140.927 | 306.608 | 175.806 | 158.597 |
| 65,536 | 584.417 | 757.139 | 659.222 | 1,402.667 | 897.125 | 797.653 |

No size in this sweep beats the best scalar baseline. Prior raw measurements
are retained in `native_m4.json` and `native_cancelled_m4.json`; variation
between runs cautions against interpreting small timing differences.

At N=65536, fused and Stockham each use 524,288 butterflies and zero standalone
bit-reversal swaps. Fused transfers 2,621,440 complex sample elements versus
2,097,152 for Stockham: 25% more. Each element is 16 bytes. These are logical
sample accesses, not DRAM traffic; coefficient and index accesses are itemized
separately in README. `audit_counts.py` independently verifies all seven
dynamic arithmetic/data counter fields across all 48 native size/method cases.

## Completed NEON comparison

The explicit `[real, imaginary]` NEON implementation and assembly audit are
now complete; see `NEON_LEDGER.md`. It uses no boundary layout conversion.
All 48 exported native input/output cases match an independent NumPy FFT;
maximum absolute error is 7.9824e-12. The generated uninstrumented dispatch
and two packet kernel bodies contain no vector stack accesses. Unused counter
initialization discovered during this check was removed from timed code.

Final NEON timings from `native_neon_final_m4.json`:

| N | DIT | DIF | Stockham | Fused packet |
|---:|---:|---:|---:|---:|
| 64 | 0.173 | 0.166 | 0.106 | 0.476 |
| 1,024 | 4.126 | 3.892 | 2.700 | 8.628 |
| 4,096 | 19.816 | 18.830 | 19.359 | 39.969 |
| 16,384 | 118.042 | 115.479 | 81.816 | 163.750 |
| 65,536 | 629.917 | 793.972 | 462.097 | 843.722 |

Units are microseconds. The timing range retained in JSON is material,
especially for the largest Stockham case; these are schedule measurements,
not universal production-library ratios. No fused packet size beats the best
baseline in this final NEON sweep.

At N=65536, the fused route requires 475136 arithmetic lane exchanges versus
458753 for DIT/DIF/Stockham (+3.57%). It actually uses fewer general complex
multiplications: 360448 versus 425986, while quarter turns increase from
32767 to 114688. The smaller internal transforms expose more cheap rotations.
That arithmetic benefit does not overcome the routing, addressing, and kernel
execution costs in this implementation. It is not a new arithmetic lower bound:
these baselines deliberately share simple radix-2 schedules.

The family, transitions, growth analysis, scalar and explicit NEON movement
ledgers now address all four requested audit items. The completion evidence
and limits are listed in `COMPLETION.md`. No independently superior diagonal
FFT is claimed, and these results do not rule out a different packet family.
