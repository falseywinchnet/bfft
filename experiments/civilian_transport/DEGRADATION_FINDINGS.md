# Matched noise and observation-density study

The geometric predictor loses ground to IMM as observations become sparse. With independent noise σ=0.35 m, reducing observations from 65 to 5 increases its two-second RMSE from 3.373 to 4.201 m (+24.5%, paired 95% interval +11.1% to +36.5%). Robust IMM rises from 3.111 to 3.295 m. At five observations geometric RMSE is 27.5% higher than robust IMM (paired interval 5.2% to 50.8%).

Increasing noise from σ=0.35 to 2.80 m at 65 observations increases geometric RMSE by 53.1% (paired interval 41.3% to 65.7%). Relative performance is not uniformly decreasing: robust IMM degrades faster, while ordinary IMM remains more accurate at the highest noise level. The geometric method does not dominate these comparators.

All 540 analytic pair intersections contained the true endpoint. This is accompanied by large regions: the smallest certified pair ball has radius 27.11 m at the ordinary dense condition, rising to 71.44 m at σ=2.80 m. At fixed σ=0.35, thinning barely changes this simple certificate (27.11 to 27.13 m), because the sparse schedule still contains an almost optimal observation pair. The exact intersection volume is not measured. The geometric moment ellipsoid covers only 22 of 30 endpoints (73.3%) in the ordinary dense condition; its smaller size cannot be interpreted as a valid 95% guarantee.

30 fresh trajectories (six seeds in each of five families), two sensor regimes, nine unique conditions, six methods: 3,240 filter runs. All predict exactly two seconds after a fixed six-second history. The dense acquisition has 65 positions; nested thinning retains 33, 17, 9 or 5, including both endpoints. Independent noise per coordinate is 0.10, 0.35, 0.70, 1.40 or 2.80 m. Noise and acquisition sweeps are separate, not a full factorial grid. The second regime adds one shared bias of norm 0.75 m. All methods receive the same correct nominal sigma. Motion settings were frozen from the earlier development study. The geometric filter uses 6,144 particles; every method uses 1,024 forecast paths.

Rows use position RMSE across 30 endpoints. Ratios and pointwise 95% intervals use 4,000 paired bootstrap samples, stratified by family. They are not simultaneous confidence bands. Thirty cases provide coarse coverage estimates; analytic coverage follows from assumptions, not observed success.

| Bias | σ (m) | Observations | Method | Forecast RMSE (m) | Geometric / method [95% CI] | Moment-region coverage | Certified radius about method (m) |
|---|---:|---:|---|---:|---|---:|---:|
| False | 0.10 | 65 | CV Kalman | 2.790 | 1.127 [0.926, 1.364] | 93.3% | 19.08 |
| False | 0.10 | 65 | CA Kalman | 4.296 | 0.732 [0.530, 1.084] | 63.3% | 21.32 |
| False | 0.10 | 65 | IMM | 2.383 | 1.320 [1.062, 1.633] | 76.7% | 19.91 |
| False | 0.10 | 65 | Robust IMM | 2.384 | 1.319 [0.975, 1.871] | 100.0% | 20.27 |
| False | 0.10 | 65 | Turn UKF | 3.151 | 0.998 [0.883, 1.122] | 73.3% | 20.96 |
| False | 0.10 | 65 | Geometric transport | 3.145 | 1.000 [1.000, 1.000] | 70.0% | 20.61 |
| False | 0.35 | 65 | CV Kalman | 3.599 | 0.937 [0.820, 1.081] | 93.3% | 28.01 |
| False | 0.35 | 65 | CA Kalman | 5.877 | 0.574 [0.444, 0.794] | 60.0% | 30.18 |
| False | 0.35 | 65 | IMM | 3.139 | 1.075 [0.931, 1.266] | 66.7% | 28.81 |
| False | 0.35 | 65 | Robust IMM | 3.111 | 1.084 [0.903, 1.317] | 100.0% | 28.69 |
| False | 0.35 | 65 | Turn UKF | 3.826 | 0.882 [0.829, 0.935] | 63.3% | 29.63 |
| False | 0.35 | 65 | Geometric transport | 3.373 | 1.000 [1.000, 1.000] | 73.3% | 29.02 |
| False | 0.70 | 65 | CV Kalman | 4.260 | 0.939 [0.817, 1.084] | 90.0% | 36.82 |
| False | 0.70 | 65 | CA Kalman | 6.572 | 0.609 [0.495, 0.743] | 60.0% | 38.83 |
| False | 0.70 | 65 | IMM | 3.555 | 1.125 [0.984, 1.296] | 70.0% | 37.64 |
| False | 0.70 | 65 | Robust IMM | 3.749 | 1.067 [0.897, 1.303] | 100.0% | 37.46 |
| False | 0.70 | 65 | Turn UKF | 4.411 | 0.907 [0.841, 0.965] | 53.3% | 38.39 |
| False | 0.70 | 65 | Geometric transport | 3.999 | 1.000 [1.000, 1.000] | 73.3% | 37.70 |
| False | 1.40 | 65 | CV Kalman | 5.072 | 0.889 [0.791, 1.018] | 93.3% | 51.21 |
| False | 1.40 | 65 | CA Kalman | 7.658 | 0.589 [0.524, 0.655] | 60.0% | 52.97 |
| False | 1.40 | 65 | IMM | 4.032 | 1.119 [1.013, 1.238] | 83.3% | 52.03 |
| False | 1.40 | 65 | Robust IMM | 4.429 | 1.019 [0.905, 1.189] | 100.0% | 51.79 |
| False | 1.40 | 65 | Turn UKF | 5.023 | 0.898 [0.847, 0.945] | 50.0% | 52.41 |
| False | 1.40 | 65 | Geometric transport | 4.511 | 1.000 [1.000, 1.000] | 76.7% | 51.72 |
| False | 2.80 | 65 | CV Kalman | 5.995 | 0.861 [0.767, 0.992] | 93.3% | 74.98 |
| False | 2.80 | 65 | CA Kalman | 8.890 | 0.581 [0.517, 0.657] | 73.3% | 76.47 |
| False | 2.80 | 65 | IMM | 4.581 | 1.127 [1.051, 1.212] | 93.3% | 75.62 |
| False | 2.80 | 65 | Robust IMM | 5.231 | 0.987 [0.885, 1.121] | 100.0% | 75.27 |
| False | 2.80 | 65 | Turn UKF | 5.727 | 0.902 [0.843, 0.960] | 60.0% | 75.58 |
| False | 2.80 | 65 | Geometric transport | 5.164 | 1.000 [1.000, 1.000] | 76.7% | 75.37 |
| False | 0.35 | 33 | CV Kalman | 3.651 | 1.032 [0.889, 1.212] | 93.3% | 27.87 |
| False | 0.35 | 33 | CA Kalman | 5.738 | 0.657 [0.504, 0.911] | 60.0% | 30.06 |
| False | 0.35 | 33 | IMM | 3.127 | 1.205 [1.033, 1.410] | 70.0% | 28.88 |
| False | 0.35 | 33 | Robust IMM | 3.221 | 1.170 [0.956, 1.430] | 100.0% | 28.69 |
| False | 0.35 | 33 | Turn UKF | 4.041 | 0.933 [0.870, 0.993] | 63.3% | 30.00 |
| False | 0.35 | 33 | Geometric transport | 3.769 | 1.000 [1.000, 1.000] | 73.3% | 29.29 |
| False | 0.35 | 17 | CV Kalman | 3.767 | 1.022 [0.886, 1.175] | 90.0% | 28.02 |
| False | 0.35 | 17 | CA Kalman | 5.683 | 0.677 [0.539, 0.864] | 56.7% | 29.98 |
| False | 0.35 | 17 | IMM | 3.231 | 1.191 [1.010, 1.395] | 76.7% | 29.05 |
| False | 0.35 | 17 | Robust IMM | 3.282 | 1.173 [0.986, 1.393] | 100.0% | 28.91 |
| False | 0.35 | 17 | Turn UKF | 4.178 | 0.921 [0.868, 0.972] | 56.7% | 30.46 |
| False | 0.35 | 17 | Geometric transport | 3.849 | 1.000 [1.000, 1.000] | 76.7% | 29.57 |
| False | 0.35 | 9 | CV Kalman | 3.759 | 1.033 [0.917, 1.171] | 90.0% | 27.90 |
| False | 0.35 | 9 | CA Kalman | 5.469 | 0.710 [0.589, 0.853] | 60.0% | 29.82 |
| False | 0.35 | 9 | IMM | 3.219 | 1.206 [1.029, 1.416] | 80.0% | 29.06 |
| False | 0.35 | 9 | Robust IMM | 3.282 | 1.183 [0.981, 1.420] | 100.0% | 29.06 |
| False | 0.35 | 9 | Turn UKF | 4.271 | 0.909 [0.851, 0.960] | 46.7% | 30.89 |
| False | 0.35 | 9 | Geometric transport | 3.882 | 1.000 [1.000, 1.000] | 80.0% | 29.92 |
| False | 0.35 | 5 | CV Kalman | 3.871 | 1.085 [0.955, 1.220] | 90.0% | 27.34 |
| False | 0.35 | 5 | CA Kalman | 5.480 | 0.767 [0.646, 0.908] | 63.3% | 29.69 |
| False | 0.35 | 5 | IMM | 3.174 | 1.324 [1.133, 1.524] | 83.3% | 28.94 |
| False | 0.35 | 5 | Robust IMM | 3.295 | 1.275 [1.052, 1.508] | 100.0% | 28.68 |
| False | 0.35 | 5 | Turn UKF | 4.746 | 0.885 [0.786, 0.970] | 43.3% | 31.36 |
| False | 0.35 | 5 | Geometric transport | 4.201 | 1.000 [1.000, 1.000] | 73.3% | 30.01 |
| True | 0.10 | 65 | CV Kalman | 2.951 | 1.114 [0.938, 1.332] | 96.7% | 19.83 |
| True | 0.10 | 65 | CA Kalman | 4.339 | 0.758 [0.559, 1.086] | 63.3% | 22.07 |
| True | 0.10 | 65 | IMM | 2.574 | 1.278 [1.050, 1.569] | 66.7% | 20.66 |
| True | 0.10 | 65 | Robust IMM | 2.528 | 1.301 [0.991, 1.777] | 96.7% | 21.02 |
| True | 0.10 | 65 | Turn UKF | 3.336 | 0.986 [0.882, 1.101] | 70.0% | 21.71 |
| True | 0.10 | 65 | Geometric transport | 3.289 | 1.000 [1.000, 1.000] | 73.3% | 21.36 |
| True | 0.35 | 65 | CV Kalman | 3.738 | 0.950 [0.838, 1.079] | 93.3% | 28.76 |
| True | 0.35 | 65 | CA Kalman | 5.936 | 0.598 [0.469, 0.806] | 56.7% | 30.93 |
| True | 0.35 | 65 | IMM | 3.310 | 1.072 [0.937, 1.247] | 70.0% | 29.56 |
| True | 0.35 | 65 | Robust IMM | 3.245 | 1.094 [0.927, 1.309] | 100.0% | 29.44 |
| True | 0.35 | 65 | Turn UKF | 3.978 | 0.892 [0.843, 0.940] | 56.7% | 30.38 |
| True | 0.35 | 65 | Geometric transport | 3.550 | 1.000 [1.000, 1.000] | 70.0% | 29.77 |
| True | 0.70 | 65 | CV Kalman | 4.388 | 0.940 [0.821, 1.079] | 90.0% | 37.57 |
| True | 0.70 | 65 | CA Kalman | 6.663 | 0.619 [0.508, 0.741] | 56.7% | 39.58 |
| True | 0.70 | 65 | IMM | 3.705 | 1.114 [0.983, 1.278] | 73.3% | 38.39 |
| True | 0.70 | 65 | Robust IMM | 3.866 | 1.067 [0.908, 1.284] | 100.0% | 38.21 |
| True | 0.70 | 65 | Turn UKF | 4.567 | 0.903 [0.843, 0.958] | 53.3% | 39.14 |
| True | 0.70 | 65 | Geometric transport | 4.125 | 1.000 [1.000, 1.000] | 66.7% | 38.45 |
| True | 1.40 | 65 | CV Kalman | 5.195 | 0.893 [0.797, 1.021] | 93.3% | 51.96 |
| True | 1.40 | 65 | CA Kalman | 7.773 | 0.597 [0.534, 0.663] | 60.0% | 53.72 |
| True | 1.40 | 65 | IMM | 4.176 | 1.111 [1.009, 1.232] | 83.3% | 52.78 |
| True | 1.40 | 65 | Robust IMM | 4.531 | 1.024 [0.914, 1.187] | 100.0% | 52.54 |
| True | 1.40 | 65 | Turn UKF | 5.174 | 0.897 [0.846, 0.943] | 53.3% | 53.16 |
| True | 1.40 | 65 | Geometric transport | 4.641 | 1.000 [1.000, 1.000] | 70.0% | 52.47 |
| True | 2.80 | 65 | CV Kalman | 6.107 | 0.864 [0.775, 0.990] | 90.0% | 75.73 |
| True | 2.80 | 65 | CA Kalman | 8.995 | 0.587 [0.526, 0.658] | 70.0% | 77.22 |
| True | 2.80 | 65 | IMM | 4.703 | 1.122 [1.045, 1.213] | 90.0% | 76.37 |
| True | 2.80 | 65 | Robust IMM | 5.315 | 0.993 [0.896, 1.119] | 100.0% | 76.02 |
| True | 2.80 | 65 | Turn UKF | 5.854 | 0.901 [0.841, 0.961] | 56.7% | 76.33 |
| True | 2.80 | 65 | Geometric transport | 5.276 | 1.000 [1.000, 1.000] | 83.3% | 76.12 |
| True | 0.35 | 33 | CV Kalman | 3.755 | 1.038 [0.904, 1.206] | 93.3% | 28.62 |
| True | 0.35 | 33 | CA Kalman | 5.791 | 0.673 [0.527, 0.906] | 53.3% | 30.81 |
| True | 0.35 | 33 | IMM | 3.291 | 1.184 [1.028, 1.376] | 70.0% | 29.63 |
| True | 0.35 | 33 | Robust IMM | 3.320 | 1.174 [0.985, 1.415] | 100.0% | 29.44 |
| True | 0.35 | 33 | Turn UKF | 4.188 | 0.931 [0.870, 0.992] | 56.7% | 30.75 |
| True | 0.35 | 33 | Geometric transport | 3.898 | 1.000 [1.000, 1.000] | 66.7% | 30.04 |
| True | 0.35 | 17 | CV Kalman | 3.883 | 1.026 [0.891, 1.186] | 90.0% | 28.77 |
| True | 0.35 | 17 | CA Kalman | 5.756 | 0.692 [0.554, 0.865] | 56.7% | 30.73 |
| True | 0.35 | 17 | IMM | 3.410 | 1.168 [0.995, 1.364] | 76.7% | 29.80 |
| True | 0.35 | 17 | Robust IMM | 3.401 | 1.171 [0.991, 1.381] | 100.0% | 29.66 |
| True | 0.35 | 17 | Turn UKF | 4.318 | 0.923 [0.872, 0.973] | 56.7% | 31.21 |
| True | 0.35 | 17 | Geometric transport | 3.984 | 1.000 [1.000, 1.000] | 73.3% | 30.32 |
| True | 0.35 | 9 | CV Kalman | 3.880 | 1.043 [0.927, 1.184] | 90.0% | 28.65 |
| True | 0.35 | 9 | CA Kalman | 5.567 | 0.727 [0.607, 0.857] | 56.7% | 30.57 |
| True | 0.35 | 9 | IMM | 3.367 | 1.202 [1.030, 1.409] | 76.7% | 29.81 |
| True | 0.35 | 9 | Robust IMM | 3.401 | 1.190 [1.000, 1.417] | 100.0% | 29.81 |
| True | 0.35 | 9 | Turn UKF | 4.435 | 0.913 [0.851, 0.967] | 46.7% | 31.64 |
| True | 0.35 | 9 | Geometric transport | 4.048 | 1.000 [1.000, 1.000] | 70.0% | 30.67 |
| True | 0.35 | 5 | CV Kalman | 4.008 | 1.091 [0.985, 1.205] | 90.0% | 28.09 |
| True | 0.35 | 5 | CA Kalman | 5.591 | 0.782 [0.680, 0.897] | 60.0% | 30.44 |
| True | 0.35 | 5 | IMM | 3.331 | 1.313 [1.152, 1.492] | 80.0% | 29.69 |
| True | 0.35 | 5 | Robust IMM | 3.423 | 1.278 [1.091, 1.478] | 100.0% | 29.43 |
| True | 0.35 | 5 | Turn UKF | 4.914 | 0.890 [0.802, 0.970] | 46.7% | 32.11 |
| True | 0.35 | 5 | Geometric transport | 4.374 | 1.000 [1.000, 1.000] | 73.3% | 30.76 |

The certificates are a wrapper available to every method. They do not change the geometric point prediction. All-pair intersection coverage and the smallest pair ball are reported separately from Gaussian moment ellipsoids. The exact intersection volume is not computed. A smallest pair ball is an outer approximation, not a minimum enclosing ball of the intersection.

See [BOUND_EXAMINATION.md](BOUND_EXAMINATION.md) for proofs and scope, and `degradation_results/results.json` for raw paired results. Plots connect measured values only; no monotonic fit or smoothing is imposed.
