# Normalized quartic NEON measurements

Three process runs per size, seven alternating-order chunks per policy per run. Table entries are medians across runs; times are microseconds. Lower walk/DIF is better. Both complete real transforms include standard-order output.

| N | DIF us (paired with separated) | Separated us | Walk/DIF | Sibling/DIF | Random/DIF | Unfused/DIF |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.044 | 0.042 | 0.956 | 0.963 | 0.956 | 1.227 |
| 128 | 0.086 | 0.094 | 1.096 | 1.099 | 1.106 | 1.374 |
| 256 | 0.181 | 0.211 | 1.169 | 1.157 | 1.168 | 1.426 |
| 512 | 0.400 | 0.469 | 1.174 | 1.161 | 1.169 | 1.411 |
| 1024 | 0.859 | 1.063 | 1.237 | 1.224 | 1.232 | 1.452 |
| 2048 | 1.892 | 2.382 | 1.274 | 1.253 | 1.271 | 1.464 |
| 4096 | 4.549 | 5.869 | 1.302 | 1.295 | 1.309 | 1.623 |
| 8192 | 11.766 | 16.042 | 1.363 | 1.391 | 1.356 | 1.409 |
| 16384 | 24.139 | 39.907 | 1.649 | 1.759 | 1.636 | 1.655 |
| 32768 | 54.599 | 87.558 | 1.597 | 2.118 | 1.741 | 1.594 |
| 65536 | 120.350 | 188.976 | 1.559 | 2.117 | 1.829 | 1.574 |
| 131072 | 263.038 | 397.805 | 1.508 | 2.008 | 1.851 | 1.518 |
| 262144 | 563.688 | 829.239 | 1.480 | 2.000 | 1.878 | 1.553 |
| 524288 | 1237.475 | 1888.419 | 1.510 | 1.904 | 1.800 | 1.534 |
| 1048576 | 2810.193 | 4607.885 | 1.640 | 1.860 | 1.798 | 1.644 |

Inspect the per-size ratios and run ranges before interpreting a crossover.

See `NATIVE_FACTORIZATION.md` for the algebra, scope, and benchmark contract. Raw results, complete benchmark output, host/compiler metadata, assembly, and checks are in `native_m4/`.
