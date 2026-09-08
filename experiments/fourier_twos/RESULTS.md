# Measured Fourier-by-twos representation costs

2026-09-05, Apple M4, Python 3.9.6. These are reference-Python timings, not a native BFFT speed comparison. Construction/plan metadata and Python object overhead are excluded from packed-bit capacity figures. All three executors return exact algebraic encodings; ordinary numeric projection is separate.

All 2,040 bins across the eight sizes below passed exact integer coefficient comparison to a direct polynomial DFT. All inverse, packed-fold and compact phase-delay checks passed. Numeric projections through N=256 had worst maximum absolute error / maximum reference magnitude 2.599910118337433e-16. Eleven focused unit tests passed.

| N | Expanded bits/bin | Expanded capacity (KiB) | Compact capacity (KiB) | Expanded FFT (ms) | Compact encode (us) | Packed phase delay (us) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 81 | 0.079 | 0.019 | 0.028 | 2.750 | 5.959 |
| 16 | 169 | 0.330 | 0.037 | 0.139 | 8.958 | 15.542 |
| 32 | 353 | 1.379 | 0.075 | 0.342 | 14.583 | 19.500 |
| 64 | 737 | 5.758 | 0.149 | 0.843 | 11.375 | 11.458 |
| 128 | 1537 | 24.016 | 0.297 | 1.057 | 42.250 | 27.417 |
| 256 | 3201 | 100.031 | 0.594 | 2.756 | 77.125 | 30.708 |
| 512 | 6657 | 416.062 | 1.188 | 10.536 | 68.625 | 18.292 |
| 1024 | 13825 | 1728.125 | 2.376 | 27.131 | 117.000 | 18.500 |

Input bound is |x[j]| <= 32,768. Times are medians of five warmed calls. The compact delay starts with packed packets and excludes encoding, packing and inverse reconstruction. It acts on all frequency orbits. The shifts move long integers; this is not constant-time delay or a faster alternative to changing a time-domain read index.

The direct packed-fold executor retains uniform guard widths. Its payload is larger than the per-orbit-width compact capacity above. Initial packing is measured separately:

| N | Input packing (us) | Packed fold (us) |
|---:|---:|---:|
| 8 | 1.500 | 4.750 |
| 16 | 5.291 | 12.459 |
| 32 | 10.125 | 15.125 |
| 64 | 9.416 | 9.416 |
| 128 | 43.625 | 21.500 |
| 256 | 118.792 | 27.084 |
| 512 | 188.875 | 17.542 |
| 1024 | 565.958 | 17.792 |

Python scheduling, allocation and host load make these short timings noisy. They establish a cost ledger, not a promotion gate. Horner input packing repeatedly grows an integer and can dominate the fold itself. No timing omits numeric projection while claiming an end-to-end numeric FFT speedup.

At N=1024 the expanded representation is 1.69 MiB, the variable-width compact representation has about 2.38 KiB of guarded payload capacity, and the compact state has N coefficient lanes. Resolving independent numeric complex bins remains required. No production kernel was changed.

Raw data and source hashes: [fourier_twos_results.json](m4_results/fourier_twos_results.json). Test log: [fourier_twos_tests.txt](m4_results/fourier_twos_tests.txt).
