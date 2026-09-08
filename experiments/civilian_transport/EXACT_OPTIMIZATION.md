# Exact elimination of the finite current history

The public streaming `RelationalKalman` now uses a four-state realization of the
same finite midpoint-current prior. It uses the compiled double-precision kernel
when locally built, and the same algebra in NumPy otherwise. The original dense
implementation is preserved byte-for-byte in `relational_reference.py` as the
independent reference used in the matched measurements.

## Why four coordinates suffice

For a cell width h and each retained length scale ell, the old current values
have covariance

    Cov(v_i,v_j) = 4 + 4 (1 + lambda |i-j| h) exp(-lambda |i-j| h),
    lambda = sqrt(3) / ell.

Write v_i = g + w_i with independent g ~ N(0,4) and a stationary Matérn-3/2
sequence w. Introduce r = w'/lambda and z_i = (w_i,r_i). Its exact one-cell law is

    a = lambda h
    A = exp(-a) [[1+a, a], [-a, 1-a]]
    z_{i+1} = A z_i + eta_i,
    Cov(z_i) = 4 I,  Cov(eta_i) = 4 (I - A A^T).

The first entry of 4 A^n is 4 (1+n a) exp(-n a). This exactly reproduces the
sampled kernel above at every pair of cell indices. The offset of the original
sample locations to cell midpoints cancels in their pairwise differences.

The state per spatial coordinate is s_i = (p_i, g, w_i, r_i), where p_i is
position relative to the noisy first anchor at the **start** of cell i. Thus

    p_{i+1} = p_i + h (g + w_i)
    H(t) = [1, tau, tau, 0],  tau = t - i h
    position(t) = anchor + H(t) s_i.

Initialize Cov(s_0) = diag(sigma^2,4,4,4) and mean zero. This reproduces the
old coefficient scaling c_i = sqrt(h) v_i and its piecewise-linear position
exactly. We do **not** integrate a continuous Matérn current between observations:
that would be a different prior from the finite cell model being optimized.

The known Markov realization of temporal Matérn priors is established machinery;
see Hartikainen and Särkkä, [Kalman filtering and smoothing solutions to temporal
Gaussian process regression models](https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf).
The particular elimination here retains the finite-cell integral, noisy anchor,
constant-current component, and five-branch evidence mixture used in this project.

Each branch shares its 4x4 covariance across the three spatial axes. One scalar
innovation variance S and vector pb=P H^T serve all three corrections:

    S = H pb + sigma^2
    m_new = m + (pb/S) innovation^T
    P_new = P - pb pb^T / S
    log likelihood = -1/2 [3 log(2 pi S) + ||innovation||^2/S].

All five lengths remain present, and the mixture covariance retains the
between-branch spatial term. Evidence normalization is calculated once per
observation and reused by readouts. Integer-cell jumps use cached powers F^n and
process sums Q_n, with Q_{n+1}=F Q_n F^T+Q. Current and future queries reuse these
tables and leave the filtering state unchanged. The native kernel combines the
small branch operations into a single call per update or forecast request.
Repeated fractional likelihood updates with total exponent one collapse to one
Gaussian update; they have the same posterior and once-counted evidence.

For `sever_at`, the split is still determined by the old midpoint rule. After
integrating the last old-group cell, reset (g,w,r) to an independent N(0,4I),
retaining the accumulated position marginal. This reproduces the block-diagonal
current kernel, including splits outside the interval and within a cell.

## Measured cost

M4 Mini CPU, double precision, no fast-math, one BLAS thread. Microbenchmarks
use 160 identical irregular observations and 31 repeats after one warm-up;
512-cell dense cases use 5 repeats. Numbers below are medians in microseconds.
The cold initialization measurement clears the transition-table cache every
repeat; the per-update figures do not include initialization.

| 128-cell implementation | Update | Current position | Current + 2 s forecast |
|---|---:|---:|---:|
| Original dense Zak coordinates | 452.68 | 43.50 | 136.80 |
| Same dense model in real current coordinates | 250.62 | 30.99 | 79.87 |
| Four-state elimination, NumPy | 23.38 | 2.37 | 22.97 |
| Four-state elimination, compiled | 12.18 | 2.32 | 22.58 |

The update is **37.2x faster** than the original; NumPy elimination alone is
**19.4x faster**. Changing coordinates alone yields only 1.8x. At 32/128/512 cells,
compiled updates cost 11.99/12.18/12.45 us, versus 150.63/452.68/5129.44 us for the
original. Cached cell jumps keep online work independent of the number of
represented cells. Cold table construction remains O(cells): 0.357 ms at 128 cells,
versus 4.744 ms for dense initialization. Warm cached native initialization in
the acquisition runs is about 0.028 ms.

A separate simple CV control uses only two states per axis, shared isotropic
covariance, fixed acceleration intensity q=1, and no length mixture. Its update
cost is 7.28 us in NumPy and 5.39 us compiled, using the same irregular stream,
validation style, precision, and Python-to-native interface. Its compiled
current-plus-2-s forecast costs 11.16 us. Our five-branch update is **2.26x** this
small compiled control, and its forecast is **2.02x**. These are implementation
cost controls, not matched-prior methods or an accuracy comparison. They do not
establish an advantage over every tuned native Kalman implementation.

Matched full acquisition replay, sigma=0.35, averages across five families and
eight seeds, with the same 12-second evaluation interval:

| Policy | Original total online ms | Optimized total online ms | Original / optimized |
|---|---:|---:|---:|
| Constant 2 Hz | 33.87 | 3.54 | 9.57x |
| Adaptive 2→32 Hz | 114.47 | 12.55 | 9.12x |
| Constant 32 Hz | 218.18 | 9.58 | 22.78x |

For adaptive 2→32 Hz, assimilation alone falls from 78.09 to 2.29 ms. Its 6.73 ms
scheduler is now the largest cost: the established 44-step phase inversion is
unchanged. Consequently adaptive acquisition still uses fewer sensor samples,
but its total CPU cost now exceeds always-dense acquisition in this implementation.
This reverses the old CPU ordering because filtering became cheap, not because
accuracy or the sample policy changed. Sensor acquisition energy, bandwidth,
and latency are not measured here. Sigma=1.4 shows the same cost pattern.

## Same-result evidence and limits

The frozen acquisition battery has 480 matched cases: five families, eight seeds,
two noise levels, six policies, each run through both implementations (960 runs).
Boundary triggers, sample counts, timestamps, and observation arrays match
**exactly in every case**. Maximum absolute differences:

- Evaluated current position: 7.29e-12 m.
- Two-second forecast: 8.13e-12 m.
- Position used at acquisition events: 6.26e-12 m.
- Reported confidence-radius diagnostic: 1.37e-13 m.

The model is algebraically identical; floating-point operation order changes,
so this is not a bitwise-posterior claim or a universal guarantee for a decision
exactly on its threshold. Accuracy improvements from the previous acquisition
study are retained; no new denoising or prediction accuracy improvement is claimed.

23 tests pass: 17 existing relational/acquisition tests plus 6 NumPy/native
realization tests. The latter cover kernel/stationarity reconstruction, 8/32/128
cells, both coordinates, six sever configurations, random irregular observations,
multiple fractional-update counts, evidence and every forecast field, historical
smoothing, coefficient export, and cache invalidation. The native simple CV
control independently agrees with its NumPy implementation across 160 updates
and 320 queries at 2e-12 absolute tolerance.

The fast filtering covariance is 16 entries per branch instead of 16,641 at
128 cells, a 1040x reduction in entries. The filtering means/covariances occupy
1120 bytes across the five branches, plus weights/evidence. This is **not** a
claim of constant total memory: the shared bounded cache retains O(cells)
transition tables, and the object retains O(observations) input history to support
old-history smoothing and complete coefficient exports. Those requests lazily
replay the dense reference and incur its costs; their cache is invalidated on
each new observation. Dense arrays are inspection/export results, not a supported
mechanism for mutating the compressed filter state. The measured fast scope is
streaming updates and current/future marginal queries inside the declared horizon.

Raw results: `optimization_results/results.json`, `micro.json`, and
`cv_cost_controls.json`. Reproduce with `run_optimization_m4.sh`. Build the local
optional library with `make -C experiments/civilian_transport/native libmarkov.so`.
