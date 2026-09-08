# Acquisition density: measured benefit and integration cost

This experiment keeps one retained relational-current Kalman update law. The ordinary covariance-weighted innovation supplies the shortened correction; the independent-witness gate from the preceding exploration is not part of this acquisition test. Sampling density is the controlled variable. No tracker or correction method is selected at the boundary.

The result supports the proposed tradeoff: added irregular samples improve tracking smoothly, assimilation cost grows approximately with the number of observations, and individual corrections become smaller as more history is available. After settling, adaptive high-density sampling reaches the tracking accuracy of continuous high-density acquisition at substantially lower whole-run cost.

## Protocol

See [ACQUISITION_PROTOCOL.md](ACQUISITION_PROTOCOL.md). Position arrivals have +/-20% phase jitter around the requested cadence. The sampler begins at 2 Hz. When an acquired observation updates the estimated x position to x>=0, the request increases exponentially toward 4, 8, 16, or 32 Hz with a one-second time constant. The boundary affects acquisition only; the motion model and state are not reset. Constant 2 and 32 Hz profiles provide acquisition-cost references.

Five boundary-crossing fixture families, eight fresh seeds, and two Gaussian noise levels give 80 paired cases and 480 retained main policy/case runs. All accuracy measurements use the same 0.1 s physical evaluation grid. Acquisition is driven by the estimated crossing, never the true crossing. Adaptive profiles have exactly identical acquired histories and trigger times before their requested cadences diverge.

## Whole-run cost and post-crossing accuracy

Sample counts and costs cover the entire 12 s run. Position and forecast RMSE cover the interval after the true crossing; forecast horizon is two seconds. CPU integration time excludes initialization, readouts, scheduling, and diagnostic-only probes; total measured online time includes the first three. No sensor, radio, energy or bandwidth price is assigned to a sample.

| Noise σ | Acquisition profile | Mean samples | Integration ms | Total online ms | Position RMSE m | 2 s forecast RMSE m |
|---|---|---:|---:|---:|---:|---:|
| 0.35 | Constant 2 Hz | 24.5 | 10.8 | 33.4 | 0.681 | 3.249 |
| 0.35 | 2 → 4 Hz | 34.0 | 15.2 | 39.2 | 0.544 | 3.042 |
| 0.35 | 2 → 8 Hz | 53.0 | 24.0 | 49.7 | 0.441 | 2.804 |
| 0.35 | 2 → 16 Hz | 91.3 | 41.5 | 70.6 | 0.363 | 2.636 |
| 0.35 | 2 → 32 Hz | 167.8 | 76.4 | 112.1 | 0.307 | 2.493 |
| 0.35 | Constant 32 Hz | 383.9 | 174.7 | 213.5 | 0.209 | 2.331 |
| 1.4 | Constant 2 Hz | 24.5 | 10.8 | 33.4 | 1.833 | 4.395 |
| 1.4 | 2 → 4 Hz | 34.0 | 15.1 | 39.1 | 1.537 | 4.147 |
| 1.4 | 2 → 8 Hz | 52.9 | 23.8 | 49.5 | 1.291 | 3.867 |
| 1.4 | 2 → 16 Hz | 91.0 | 41.3 | 70.3 | 1.084 | 3.686 |
| 1.4 | 2 → 32 Hz | 167.2 | 76.1 | 111.8 | 0.918 | 3.464 |
| 1.4 | Constant 32 Hz | 383.9 | 175.1 | 214.0 | 0.646 | 3.209 |

The per-observation integration cost is approximately 0.46 ms throughout this fixed 128-cell experiment on the M4 CPU. This is a measured implementation cost, not a universal hardware or real-time guarantee. Dense covariance and the finite represented interval remain implementation constraints.

## Transition and retained history

| Noise σ | Profile | First second position RMSE m | 3–6 s position RMSE m | Later mean correction m | Later projected gain |
|---|---|---:|---:|---:|---:|
| 0.35 | Constant 2 Hz | 0.729 | 0.687 | 0.702 | 0.716 |
| 0.35 | 2 → 4 Hz | 0.694 | 0.497 | 0.412 | 0.524 |
| 0.35 | 2 → 8 Hz | 0.645 | 0.370 | 0.239 | 0.347 |
| 0.35 | 2 → 16 Hz | 0.608 | 0.275 | 0.135 | 0.216 |
| 0.35 | 2 → 32 Hz | 0.563 | 0.209 | 0.077 | 0.129 |
| 0.35 | Constant 32 Hz | 0.208 | 0.210 | 0.074 | 0.125 |
| 1.4 | Constant 2 Hz | 1.858 | 1.853 | 1.559 | 0.523 |
| 1.4 | 2 → 4 Hz | 1.783 | 1.440 | 1.003 | 0.370 |
| 1.4 | 2 → 8 Hz | 1.707 | 1.127 | 0.602 | 0.240 |
| 1.4 | 2 → 16 Hz | 1.637 | 0.859 | 0.351 | 0.147 |
| 1.4 | 2 → 32 Hz | 1.550 | 0.645 | 0.195 | 0.085 |
| 1.4 | Constant 32 Hz | 0.657 | 0.644 | 0.175 | 0.076 |

The projected gain is delta_position·innovation / ||innovation||², measured at each acquisition and then averaged. It describes the effective correction of this mixture model; it is not an imposed clipping coefficient. The rate command is continuous, while actual measurement updates remain discrete. No claim of pointwise state continuity is made.

## Adaptive high density versus continuous high density

At σ=0.35, adaptive 2→32 Hz uses 43.7% of the acquired samples, 43.7% of assimilation time, and 52.5% of total measured online time. After 3 s of settling its position RMSE is 0.209 m versus 0.210 m. The paired RMSE ratio is 0.996, with a pointwise 95% bootstrap interval [0.958, 1.034].

At σ=1.4, adaptive 2→32 Hz uses 43.6% of the acquired samples, 43.5% of assimilation time, and 52.3% of total measured online time. After 3 s of settling its position RMSE is 0.645 m versus 0.644 m. The paired RMSE ratio is 1.001, with a pointwise 95% bootstrap interval [0.951, 1.053].

The paired intervals use 4000 resamples of eight seeds within each of the five fixed motion families. They are pointwise, not simultaneous claims or generalization over arbitrary flights. Continuous high density is more accurate immediately after crossing because it has already accumulated dense history. The adaptive profile pays for its lower earlier acquisition cost with a settling interval.

## Marginal return

| Noise σ | Target increase | Extra samples | Extra integration ms | Position MSE reduction / ms | Position MSE reduction / sample |
|---|---|---:|---:|---:|---:|
| 0.35 | Constant 2 Hz to 2 → 4 Hz | 9.4 | 4.35 | 0.03864 | 0.01782 |
| 0.35 | 2 → 4 Hz to 2 → 8 Hz | 19.1 | 8.78 | 0.01152 | 0.00529 |
| 0.35 | 2 → 8 Hz to 2 → 16 Hz | 38.3 | 17.54 | 0.00357 | 0.00163 |
| 0.35 | 2 → 16 Hz to 2 → 32 Hz | 76.5 | 34.85 | 0.00107 | 0.00049 |
| 1.4 | Constant 2 Hz to 2 → 4 Hz | 9.4 | 4.31 | 0.23112 | 0.10574 |
| 1.4 | 2 → 4 Hz to 2 → 8 Hz | 18.9 | 8.69 | 0.07994 | 0.03671 |
| 1.4 | 2 → 8 Hz to 2 → 16 Hz | 38.2 | 17.48 | 0.02815 | 0.01290 |
| 1.4 | 2 → 16 Hz to 2 → 32 Hz | 76.2 | 34.85 | 0.00954 | 0.00436 |

These are observed returns on the unchanged update law. No runtime policy winner is selected. A hardware-specific acquisition cost can later be combined with the separate count and compute measurements. More samples improve current tracking more strongly than they improve two-second prediction in this screen; the continuation model is still substantive.

## Trigger, uncertainty and representation checks

At σ=0.35, median estimated-trigger delay is 0.207 s, with observed range [-0.272, 0.668] s and 90th percentile 0.476 s. Negative delay means the estimate triggered before the clean path crossed. These decisions were retained, not replaced with oracle crossing times.

At σ=1.4, median estimated-trigger delay is 0.183 s, with observed range [-0.916, 1.032] s and 90th percentile 0.879 s. Negative delay means the estimate triggered before the clean path crossed. These decisions were retained, not replaced with oracle crossing times.

All four 128→256-cell checks change post-crossing position RMSE by less than 0.00013 m and two-second forecast RMSE by less than 0.00084 m, with identical triggers and sample counts. These are limited checks, not a uniform resolution theorem. Moment-region sizes and empirical coverage are retained in summary.json; they are not exact 95% guarantees for these out-of-model trajectories.

Seventeen tests passed on the Mini: six cadence/end-to-end checks and eleven existing relational Kalman checks. Source hashes and all 480 unique main records are retained.

## Reproduction

```sh
sh experiments/civilian_transport/run_acquisition_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.acquisition_report
```

![Benefit and cost](acquisition_results/benefit_cost.png)

![Sampling transition](acquisition_results/sampling_transition.png)

![Marginal return](acquisition_results/marginal_return.png)
