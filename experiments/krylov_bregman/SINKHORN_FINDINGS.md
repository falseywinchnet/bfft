# Transfer to entropic optimal transport

The existing Krylov polynomial transport accelerates a distinct established
Bregman problem. In a 24-case dense Sinkhorn study it is 1.91–2.73x faster than
ordinary Sinkhorn (median 2.40x). The existing discovery policy is faster in
19/24 cases, with median 1.18x speedup. Anderson is generally stronger here.
The new rational chart grammar does not transfer cleanly to this coupled map.

## Problem and relation to the earlier geometry

We solve the entropically regularized transport problem

    min_P <C,P> + epsilon * sum_ij P_ij (log P_ij - 1)
    subject to P 1 = p, P^T 1 = q, P >= 0.

For K=exp(-C/epsilon), Sinkhorn alternates row and column scaling, equivalently
alternating KL/Bregman projections. Its two scaling vectors have a redundant
common gauge. The complete future-driving state is the log column scaling y,
with its mean removed after each full sweep. The map is

    a(y) = log p - log(K exp(y))
    F(y) = gauge(log q - log(K^T exp(a(y)))).

Its exact tangent is a product of two conditional averaging maps, followed by
gauge removal. `sinkhorn_transport.py` applies those actions through the same
kernel matvecs used in the ordinary method. It does not substitute the scalar
rational construction or change the regularized objective. Stable rescaling
avoids unnecessary all-pairs exponentials; an independently tested log-domain
fallback covers extreme states. No timed run needed that fallback.

Sinkhorn and its small-regularization slowdown are established:

- Marco Cuturi, [Sinkhorn Distances](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html), 2013.
- Bernhard Schmitzer, [Stabilized Sparse Scaling Algorithms](https://epubs.siam.org/doi/abs/10.1137/16M1106018), 2019.
- Thibault et al., [Overrelaxed Sinkhorn–Knopp](https://www.mdpi.com/1999-4893/14/5/143), 2021, also discusses nonlinear acceleration of this problem. We claim no novelty for accelerating Sinkhorn generally.

## Predeclared protocol

- 128 and 512 source points, with the same number of target points.
- Independent irregular point supports in the unit square, with squared
  Euclidean costs. These are generic dense costs, not a regular-grid
  convolution implementation.
- Camera-to-Barbara image-intensity masses and displaced Gaussian mixtures.
  Positive floors keep every prescribed marginal positive.
- Two support seeds and epsilon=0.003, 0.01, 0.03: 24 distinct problems.
- Seven timed repeats per method per case, with warmup and shuffled method
  order, on the M4 Mini, CPU, single-thread BLAS.
- The target is L1 column-marginal error <=1e-8 for the exactly row-normalized
  plan. The same work-indexed checking cadence (eight map/action units) is
  used throughout. A tighter ordinary solution at 1e-11 independently checks
  the final plans and objective values outside the timing loop.
- Work budget 20,000 map/action units, never reached in the retained runs.
- No epsilon continuation, objective substitution, or target-dependent tuning.

The four methods are ordinary Sinkhorn; the existing fixed polynomial method
with depth 4, horizon 16, and one settling step; the existing discovery policy
with its default depths/horizons/tolerance and a 32-step rejection cooldown;
and the existing regularized type-II Anderson comparator of depth 4. Four
initial ordinary sweeps are used before acceleration. We use the Euclidean
metric in gauge-fixed dual coordinates for this first transfer test.

Every timed solver loop includes acquisition, tangent actions,
orthogonalization, discovery probes, rejected attempts, settling, residual
checks, and bookkeeping. Kernel/data setup is excluded and recorded separately;
final dense plan materialization and independent reference verification are
also outside solver timing. These are complete iterative solve timings, not
an end-to-end file-loading benchmark. The same implementation backs all four
methods. Specialized OT solvers, GPU kernels, epsilon scaling, and Newton
methods are not assessed.

Discovery probes only selected predicted states. They are an empirical screen,
not a global nonlinear trajectory certificate. Every accepted solve still has
to reach the actual marginal target.

## Results

Speedups are ratios of seven-repeat median solve times, summarized across the
24 cases; they are not ratios of iteration counts.

| Method | Median speedup | Range | Cases faster than ordinary |
|---|---:|---:|---:|
| Fixed polynomial transport | 2.397x | 1.914–2.728x | 24/24 |
| Discovered polynomial transport | 1.178x | 0.708–1.808x | 19/24 |
| Anderson, depth 4 | 3.303x | 1.684–9.430x | 24/24 |

All 672 timed runs and all tighter references reached their targets. Maximum
row error was 3.19e-15; maximum column error was 9.925e-9. Maximum L1 plan
difference from the tighter reference was 6.55e-8, and maximum absolute
regularized-objective difference was 5.71e-10. The slightly infeasible terminal
objective is not itself a certified objective-gap bound.

For image transport with 512 points, seed 0, epsilon=0.003, median times were
107.68 ms ordinary, 40.00 ms fixed polynomial, 71.51 ms discovered polynomial,
and 21.02 ms Anderson. Thus this representative difficult case gives 2.69x
fixed-polynomial and 1.51x discovery speedup, while Anderson remains faster.

Fixed polynomial transport is faster than Anderson in 5/24 median timings;
two are close small-case timings. There is no overall advantage over Anderson.
Discovery is slower than fixed transport in every case. It reduces map/action
work relative to ordinary execution but loses enough overhead on five easier
cases to increase wall-clock time. This isolates acquisition cost as a practical
limitation even when the transported dynamics are useful.

## Does the new rational coordinate discovery transfer?

A separate diagnostic applies the unmodified degree-(1,1) rational relation
and cross-ratio construction to each component of the gauge-fixed Sinkhorn
state. At 18 anchors (two distributions, three epsilon values, prefixes
4/32/128), it acquires eight transitions and predicts a 64-sweep horizon.
Actual future execution is used only to measure error in this diagnostic.

The current grammar rejects 16/18 anchors because some coordinate has nonreal
or repeated fixed points, or the relation is underidentified. Two charts fit,
with relative trajectory errors 2.44% and 1.92%; the corresponding depth-four
Krylov errors are 0.0190% and 0.0373%. A tiny local rational fitting residual
therefore does not establish the exact closure seen in the constructed control.
Near-stationary anchors report absolute errors and omit relative errors when
the ordinary displacement is below 1e-10.

This is a test of this coordinatewise grammar, not proof that Sinkhorn admits
no useful nonlinear chart. The evidence supports transfer of finite Krylov
transport and rejects treating the earlier exact rational example as an
already general discovery algorithm. Coupled observable relations remain the
next representation question; improving discovery amortization is a separate
cost question.

## Reproduction and retained records

Run the commands in the root AGENTS.md or the experiment README. Files:

- `sinkhorn_transport.py`: actual map, exact tangent, error, and timed solve.
- `sinkhorn_study.py`: problem construction and full repeated benchmark.
- `sinkhorn_chart_probe.py`: separate chart-transfer diagnostic.
- `test_sinkhorn_transport.py`: six new invariant and cross-method tests.
- `results/sinkhorn_pilot.json`: first one-repeat screen, not final timings.
- `results/sinkhorn_full.json`: all 672 final runs and independent checks.
- `results/sinkhorn_summary.json`: case medians and aggregate statistics.
- `results/sinkhorn_chart_probe.json`: all 18 chart probes, including rejections.
- `sinkhorn_report.py`: regenerates the summary and PNG/SVG timing figure.

The complete research suite passes 57 tests on the M4 Mini. The open LaTeX
paper was not edited during this empirical transfer experiment.
