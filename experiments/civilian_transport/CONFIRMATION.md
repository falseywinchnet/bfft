# Position-only civilian tracking: measured findings

The fixed test has 80 independent synthetic trajectories: 8 seeds for each of five motion families under clean and corrupted observations. Configuration selection used ten separate development cases.

The approximately two-second forecast uses only past positions and requested future timestamps. Numbers below are vector position RMSE in meters; lower scores are better. No claims of a universal SOTA result or real-airspace validation follow from this battery.

| Method | Tracking RMSE | ~1 s forecast | ~2 s forecast | Energy score (~2 s) | Boundary crossing Brier |
|---|---:|---:|---:|---:|---:|
| CV Kalman | 0.664 | 2.212 | 4.495 | 2.825 | 0.0877 |
| CA Kalman | 0.698 | 2.733 | 6.488 | 4.280 | 0.1073 |
| IMM | 0.648 | 2.090 | 4.122 | 2.410 | 0.0849 |
| Robust IMM | 0.610 | 1.844 | 3.624 | 2.250 | 0.0836 |
| 3-D turn UKF | 0.683 | 2.010 | 3.757 | 2.525 | 0.0971 |
| Geometric transport | 0.684 | 1.929 | 3.495 | 2.222 | 0.0995 |

## Paired forecast comparison

Positive percentage means lower RMSE for geometric transport. Intervals resample whole trajectories within each family/noise stratum (4,000 paired bootstrap draws). They describe this synthetic case distribution, conditional on the frozen development choice.

| Comparator | Geometric RMSE reduction | 95% interval | Cases won |
|---|---:|---:|---:|
| CV Kalman | 22.2% | [18.4%, 26.0%] | 67/80 |
| CA Kalman | 46.1% | [42.5%, 49.5%] | 77/80 |
| IMM | 15.2% | [10.2%, 20.0%] | 50/80 |
| Robust IMM | 3.6% | [-2.1%, 8.5%] | 49/80 |
| 3-D turn UKF | 7.0% | [3.9%, 10.0%] | 55/80 |

## Posterior-coupling and straightening ablations

These start from the same geometric filtered posterior; only future propagation is changed.

| Forecast law | ~2 s RMSE | Energy score | Crossing Brier |
|---|---:|---:|---:|
| Intact joint posterior | 3.495 | 2.222 | 0.0995 |
| uncoupled | 3.468 | 2.234 | 0.1007 |
| straight | 3.783 | 2.312 | 0.0917 |

## Calibration, components, and cost

Containment is empirical coverage of a nominal 95% moment-Gaussian ellipsoid, not an exact mixture credible set. Bias/noise errors test attribution separately from tracking. Update times include current-state diagnostics; forecast times include 256 sampled paths over twenty irregular intervals and exclude the ablation calls. Timings are single-run Python measurements on the M4, not production-kernel or isolated-host speed guarantees.

| Method | Tracking coverage | Forecast coverage | Bias RMSE | Noise RMSE | Update ms | Forecast ms |
|---|---:|---:|---:|---:|---:|---:|
| CV Kalman | 88.9% | 85.6% | 0.190 | 0.620 | 0.09 | 1.79 |
| CA Kalman | 79.3% | 47.9% | 0.190 | 0.638 | 0.10 | 2.05 |
| IMM | 85.0% | 67.1% | 0.198 | 0.594 | 0.85 | 12.28 |
| Robust IMM | 93.7% | 96.7% | 0.190 | 0.572 | 0.99 | 12.28 |
| 3-D turn UKF | 89.3% | 76.2% | 0.195 | 0.629 | 0.17 | 2.93 |
| Geometric transport | 89.8% | 83.5% | 0.241 | 0.609 | 3.78 | 3.50 |

## Breakdown by motion family and noise

| Scenario | CV | CA | IMM | Robust IMM | Turn UKF | Geometric |
|---|---:|---:|---:|---:|---:|---:|
| line_False | 1.454 | 1.836 | 1.117 | 1.728 | 2.887 | 1.600 |
| line_True | 3.525 | 3.881 | 3.565 | 2.482 | 3.657 | 2.354 |
| helix_False | 4.399 | 5.441 | 2.827 | 2.749 | 4.008 | 3.796 |
| helix_True | 4.951 | 5.331 | 3.887 | 3.477 | 4.086 | 3.779 |
| figure8_False | 4.926 | 8.024 | 4.685 | 4.236 | 3.991 | 4.182 |
| figure8_True | 5.547 | 8.212 | 5.102 | 4.669 | 4.534 | 4.864 |
| stop_go_False | 2.187 | 3.427 | 2.460 | 2.235 | 2.931 | 2.194 |
| stop_go_True | 4.649 | 6.210 | 4.586 | 3.160 | 3.527 | 2.809 |
| switching_False | 5.024 | 8.848 | 5.052 | 4.534 | 3.567 | 3.912 |
| switching_True | 6.080 | 9.167 | 5.679 | 5.218 | 4.058 | 4.025 |

## Scope of the establishment

Eleven structural tests pass: exact finite-map composition, rotation equivariance, speed preservation, the zero-turn limit, the mean-generator boundary counterexample, positive process covariance, independent Gaussian conditioning, posterior signal/noise mean accounting, missing-observation handling, forecast isolation from filtering/RNG, and all-method smoke checks.

The inferred object is a distribution over a restricted continuous family of geometric continuation laws. The resistance functional itself is prescribed up to development-selected diffusivities. This is not unrestricted discovery of the geometry from one trajectory. Possible-history reduction is Monte Carlo resampling, so exact posterior support is not guaranteed. Independent future waypoint commands remain unavailable to all methods.

The baseline implementations and their limits are specified in THEORY.md. The test compares standard practical filter families plus a continuous-turn UKF; it does not reproduce every contemporary arXiv method or claim a universal leaderboard.
