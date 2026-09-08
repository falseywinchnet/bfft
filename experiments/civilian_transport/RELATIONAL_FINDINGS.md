# First relational-current Kalman experiment

This implements the explicit Gaussian structural special case of KALMAN_FOUNDATION.md. It does not implement or establish a general nonlinear relational BTB denoiser. No parameter was changed after inspecting this evaluation.

## Construction

The state contains an anchor and 64 orthonormal piecewise-constant current coefficients on [0,8] s. Position observations integrate these basis functions exactly. The current covariance is a shared constant component of variance 4 plus a Matérn-3/2 component of variance 4 (m/s)^2; coefficient covariance scales with cell width. Five fixed correlation lengths (0.3, 0.7, 1.5, 3, 6 s) receive a uniform prior. Their exact history marginal likelihoods determine posterior weights, and total covariance includes between-component uncertainty. This is a declared motion prior, not a law supplied by the motivating papers.

The first measurement conditions a flat absolute-position prior once. Subsequent measurements are independent Gaussian errors of known sigma. Zak analysis/synthesis retains full covariance and the real-signal conjugacy constraints. Sequential Kalman conditioning retains the complete current covariance as observations arrive and evaluates the Gaussian fixed point; it is tested against joint conditioning. A separate actual relaxed proximal iteration tests the BTB-compatible Gaussian map; it is not substituted into the reported forecast before convergence.

## Evaluation contract

Five existing motion families, six fresh seeds per family (80000+1000×family index+repeat), three paired conditions: 65 observations with sigma 0.35 m; 65 with sigma 1.4 m; 9 with sigma 0.35 m. History is 0–6 s; forecast is 6–8 s. Truth values after 6 s never reach an estimator. All 720 method/case runs are retained. Baselines use their previously frozen configurations, changing only supplied sensor sigma, with 2048 forecast draws. This is an exploratory synthetic screen; no real sensor or correlated-noise claim is made.

## Endpoint results

| Method | Dense σ=.35 RMSE | Dense σ=1.4 RMSE | Sparse σ=.35 RMSE |
|---|---:|---:|---:|
| Relational current | 2.958 m | 4.891 m | 3.788 m |
| Past–future severed | 4.868 m | 5.240 m | 4.932 m |
| Single length 1.5 s | 3.556 m | 4.762 m | 3.886 m |
| CV Kalman | 3.750 m | 5.357 m | 4.355 m |
| CA Kalman | 6.319 m | 7.811 m | 6.140 m |
| IMM | 3.443 m | 5.014 m | 4.263 m |
| Robust IMM | 3.269 m | 4.806 m | 3.940 m |
| Turn UKF | 4.243 m | 5.229 m | 4.653 m |

RMSE is sqrt(mean squared 3-D endpoint distance), with equal case weights.

## Paired uncertainty

Ratios below are relational RMSE / comparator RMSE. Pointwise 95% percentile intervals use 4000 paired bootstrap resamples, six seeds resampled within each of the five fixed families. They do not establish generalization over arbitrary motion families or simultaneous significance.

| Condition | Comparator | Ratio | Interval |
|---|---|---:|---|
| σ=0.35, n=65 | Robust IMM | 0.905 | [0.837, 0.981] |
| σ=0.35, n=65 | IMM | 0.859 | [0.779, 0.952] |
| σ=0.35, n=65 | Past–future severed | 0.608 | [0.537, 0.675] |
| σ=0.35, n=65 | Single length 1.5 s | 0.832 | [0.757, 0.908] |
| σ=1.4, n=65 | Robust IMM | 1.018 | [0.921, 1.124] |
| σ=1.4, n=65 | IMM | 0.975 | [0.851, 1.107] |
| σ=1.4, n=65 | Past–future severed | 0.933 | [0.847, 1.009] |
| σ=1.4, n=65 | Single length 1.5 s | 1.027 | [0.979, 1.078] |
| σ=0.35, n=9 | Robust IMM | 0.961 | [0.906, 1.021] |
| σ=0.35, n=9 | IMM | 0.888 | [0.762, 1.021] |
| σ=0.35, n=9 | Past–future severed | 0.768 | [0.687, 0.850] |
| σ=0.35, n=9 | Single length 1.5 s | 0.975 | [0.923, 1.029] |

## Interpretation and unresolved work

- The declared current model is competitive in this first screen. Read the paired intervals before describing the small differences versus robust IMM as a win.
- Severing current covariance across 6 s worsens point prediction and greatly enlarges forecast regions. Its future current is independent and zero mean, so this ablation intentionally removes the continuation mechanism. It establishes the importance of this prior connection, not its uniqueness or optimality.
- The length mixture improves over fixed 1.5 s at low noise but does not win every condition. Hyperparameter uncertainty is retained; there is no post-hoc choice of the best length using future truth.
- All five 64→128 cell audits change endpoint means by less than 0.022 m. This is a limited resolution check, not a convergence theorem or uniform truncation guarantee.
- Zak/current and four-step/single-step assimilation agree to less than 3e-11 in the retained audits. They are equivalent computations of the same posterior, not independent accuracy gains.
- The unpreconditioned relaxed proximal iteration is slow: after 2000 steps its relative whitened-state mean error is 55–83%, despite normalized residuals around 0.00016–0.00111. Its worst contraction factor is 0.9999795. Small residuals do not certify a recovered state. Exact Gaussian conditioning supplies the reported predictions; this iterative path has no demonstrated practical advantage.
- Moment ellipsoids use the chi-square-3 95% threshold. For mixtures and these out-of-model motion fixtures they are diagnostics, not exact 95% probability sets. The relational model covers 30/30, 28/30, 30/30 endpoints; its mean equal-volume radii are 5.77, 8.42, 6.61 m. Small-sample coverage does not prove a guarantee.
- Recorded timing is the actual complete query workload: exact analytical moments for the current model versus baseline Monte Carlo forecasts. It is not a matched-kernel latency or complexity claim. Dense covariance and growing current dimension remain scaling costs.
- A justified nonlinear relational operator, robust correlated sensor model, and calibration on external motion remain unresolved. This experiment tests an explicit integrated Gaussian-process prior with uncertain correlation length; it does not claim the full original mathematical objective has been solved.

## Reproduction

```sh
sh experiments/civilian_transport/run_relational_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.relational_report
```

The runner uses m4build/m4host and immediately copies results home. The retained JSON includes source hashes. Eleven tests cover streaming/batch equivalence, integration, Zak identity, likelihood accounting, spatial equivariance, covariance/information order, future nullspace, mixture variance, proximal convergence on a small case, and prior-predictive calibration.

![Comparison](relational_results/comparison.png)

![Trajectories](relational_results/trajectories.png)
