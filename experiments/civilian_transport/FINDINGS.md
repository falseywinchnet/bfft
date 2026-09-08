# Position-only civilian tracking: measured findings

The fixed test has 80 independent synthetic trajectories: 8 seeds for each of five motion families under clean and corrupted observations. Configuration selection used ten separate development cases.

The approximately two-second forecast uses only past positions and requested future timestamps. Numbers below are vector position RMSE in meters; lower scores are better. No claims of a universal SOTA result or real-airspace validation follow from this battery.

| Method | Tracking RMSE | ~1 s forecast | ~2 s forecast | Energy score (~2 s) | Boundary crossing Brier |
|---|---:|---:|---:|---:|---:|
| CV Kalman | 0.656 | 2.061 | 4.365 | 2.762 | 0.1066 |
| CA Kalman | 0.714 | 2.762 | 6.632 | 4.381 | 0.1317 |
| IMM | 0.653 | 1.971 | 3.973 | 2.332 | 0.0940 |
| Robust IMM | 0.614 | 1.881 | 3.706 | 2.278 | 0.0970 |
| 3-D turn UKF | 0.642 | 2.007 | 3.769 | 2.533 | 0.1009 |
| Geometric transport | 0.874 | 2.149 | 3.804 | 2.439 | 0.1047 |

## Paired forecast comparison

Positive percentage means lower RMSE for geometric transport. Intervals resample whole trajectories within each family/noise stratum (4,000 paired bootstrap draws). They describe this synthetic case distribution, conditional on the frozen development choice.

| Comparator | Geometric RMSE reduction | 95% interval | Cases won |
|---|---:|---:|---:|
| CV Kalman | 12.9% | [8.7%, 17.0%] | 55/80 |
| CA Kalman | 42.6% | [38.6%, 46.3%] | 71/80 |
| IMM | 4.3% | [-0.7%, 8.8%] | 48/80 |
| Robust IMM | -2.7% | [-7.8%, 2.4%] | 39/80 |
| 3-D turn UKF | -0.9% | [-4.5%, 2.8%] | 43/80 |

## Posterior-coupling and straightening ablations

These start from the same geometric filtered posterior; only future propagation is changed.

| Forecast law | ~2 s RMSE | Energy score | Crossing Brier |
|---|---:|---:|---:|
| Intact joint posterior | 3.804 | 2.439 | 0.1047 |
| uncoupled | 3.786 | 2.481 | 0.1050 |
| straight | 4.141 | 2.556 | 0.1037 |

## Calibration, components, and cost

Containment is empirical coverage of a nominal 95% moment-Gaussian ellipsoid, not an exact mixture credible set. Bias/noise errors test attribution separately from tracking. Update times include current-state diagnostics; forecast times include 256 sampled paths over twenty irregular intervals and exclude the ablation calls. Timings are single-run Python measurements on the M4, not production-kernel or isolated-host speed guarantees.

| Method | Tracking coverage | Forecast coverage | Bias RMSE | Noise RMSE | Update ms | Forecast ms |
|---|---:|---:|---:|---:|---:|---:|
| CV Kalman | 89.2% | 85.4% | 0.187 | 0.613 | 0.09 | 1.81 |
| CA Kalman | 78.0% | 48.3% | 0.187 | 0.648 | 0.10 | 2.08 |
| IMM | 84.7% | 69.6% | 0.192 | 0.596 | 0.85 | 12.46 |
| Robust IMM | 94.3% | 96.8% | 0.187 | 0.574 | 0.99 | 12.45 |
| 3-D turn UKF | 91.7% | 81.0% | 0.191 | 0.602 | 0.17 | 2.94 |
| Geometric transport | 81.9% | 77.6% | 0.299 | 0.801 | 1.07 | 3.50 |

## Breakdown by motion family and noise

| Scenario | CV | CA | IMM | Robust IMM | Turn UKF | Geometric |
|---|---:|---:|---:|---:|---:|---:|
| line_False | 1.386 | 1.784 | 1.040 | 1.675 | 3.049 | 2.193 |
| line_True | 3.414 | 3.992 | 3.401 | 3.001 | 3.690 | 2.777 |
| helix_False | 4.364 | 4.845 | 2.613 | 2.601 | 4.160 | 4.020 |
| helix_True | 4.784 | 6.408 | 3.668 | 3.673 | 4.253 | 4.527 |
| figure8_False | 5.257 | 8.958 | 5.011 | 4.585 | 3.998 | 4.270 |
| figure8_True | 5.776 | 9.356 | 5.136 | 4.940 | 4.470 | 4.978 |
| stop_go_False | 2.416 | 3.652 | 2.582 | 2.412 | 3.158 | 2.706 |
| stop_go_True | 3.732 | 5.083 | 3.738 | 3.626 | 3.114 | 2.913 |
| switching_False | 4.837 | 8.454 | 4.907 | 4.389 | 3.586 | 4.428 |
| switching_True | 5.555 | 8.817 | 5.360 | 4.646 | 3.906 | 4.137 |

## Scope of the establishment

Eleven structural tests pass: exact finite-map composition, rotation equivariance, speed preservation, the zero-turn limit, the mean-generator boundary counterexample, positive process covariance, independent Gaussian conditioning, posterior signal/noise mean accounting, missing-observation handling, forecast isolation from filtering/RNG, and all-method smoke checks.

The inferred object is a distribution over a restricted continuous family of geometric continuation laws. The resistance functional itself is prescribed up to development-selected diffusivities. This is not unrestricted discovery of the geometry from one trajectory. Possible-history reduction is Monte Carlo resampling, so exact posterior support is not guaranteed. Independent future waypoint commands remain unavailable to all methods.

The baseline implementations and their limits are specified in THEORY.md. The test compares standard practical filter families plus a continuous-turn UKF; it does not reproduce every contemporary arXiv method or claim a universal leaderboard.
