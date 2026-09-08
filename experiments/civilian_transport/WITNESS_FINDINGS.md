# Witnessed correction: findings

The experiment borrows the local-restoration observation from BTB; it implements no BTB iteration. It constructs a finite confidence geometry from independent history witnesses and tests its limits. The main result is a separation: local clean-error improvement can be supported without assuming the current prior is correct, but extending that correction through the same current relationships does not establish improved prediction.

## Construction and proof

See [WITNESS_GEOMETRY.md](WITNESS_GEOMETRY.md) for the derivation. Directions are the complete current-mean revisions induced by ten training observations. The witness values never enter the direction construction. For a rank-r witness space, the projected clean residual lies inside a chi-square confidence ball. The maximum guaranteed-gain correction is the projected observed residual shortened by that ball’s radius. This is an algebraic calculation, not repeated denoising. A separate single-direction procedure only shortens the combined training correction.

The guarantee is conditional on independent, zero-mean Gaussian witness errors of known sigma. It concerns summed clean squared position error at the ten witness times. It is uniform over all corrections in the training-defined space, so selecting the correction using the witness confidence ball is permitted. It makes no claim about the unobserved future or the physical cause of a residual.

## Protocol

Five existing motion families; 12 fresh seeds each; sigma 0.35 and 1.4 m; ordinary motion, one 3 m training-only sensor spike, a new maneuver, and coherent sensor drift. This gives 480 cases, 2880 retained method/case records, and 1440 certificate records. Prefix observations are 0–4 s; training observations are 4.2,4.4,...,6 s; witnesses are 4.1,4.3,...,5.9 s. All forecasts are made at 6 s for 8 s. Future observation values are replaced with NaN before proposal construction. No threshold or prior was tuned on these results.

Original Kalman uses all 61 history observations; training update uses the 41-point prefix plus ten training points; witnessed methods additionally use the ten witnesses in the support calculation. The original Kalman witness error is evaluated against clean truth at positions it has observed noisily, so it is a denoising comparison, not an independent validation score for that baseline.

## Pure-noise control

With 4000 pure-noise trials, false support is 5.05% for the honest ray and 5.10% for the honest correction space, consistent with their separate 5% levels. Reusing the noise that created the direction as its own witness produces 100% false support for both. This is a concrete self-confirmation failure; it is not a theorem about every denoiser or EM system.

## Local error and forecast error

Each entry is clean witness RMSE / two-second endpoint RMSE, both in metres. Each condition has 60 paired paths.

| Noise | Scenario | Training update | Witnessed ray | Witnessed space | Original Kalman, all history |
|---|---|---:|---:|---:|---:|
| 0.35 | Ordinary motion | 0.236 / 3.604 | 0.311 / 3.536 | 0.513 / 7.832 | 0.188 / 3.413 |
| 0.35 | Training sensor spike | 0.408 / 3.777 | 0.436 / 3.775 | 0.513 / 5.808 | 0.244 / 3.554 |
| 0.35 | New maneuver | 0.254 / 3.715 | 0.327 / 3.774 | 0.540 / 9.375 | 0.198 / 3.552 |
| 0.35 | Coherent sensor drift* | 1.057 / 7.238 | 0.990 / 6.772 | 0.950 / 10.241 | 1.063 / 7.363 |
| 1.4 | Ordinary motion | 0.795 / 4.811 | 1.108 / 4.624 | 1.679 / 9.277 | 0.651 / 4.638 |
| 1.4 | Training sensor spike | 0.787 / 4.800 | 1.105 / 4.648 | 1.676 / 9.305 | 0.638 / 4.635 |
| 1.4 | New maneuver | 0.828 / 5.558 | 1.173 / 6.091 | 1.792 / 9.747 | 0.672 / 5.179 |
| 1.4 | Coherent sensor drift* | 1.153 / 6.082 | 1.223 / 5.486 | 1.683 / 8.834 | 1.112 / 6.257 |

For ordinary motion at sigma 0.35, the prefix-only witness/forecast RMSE is 1.939 / 7.113 m. The witnessed space improves local error to 0.513 m but worsens the future RMSE to 7.832 m. Original Kalman reaches 0.188 / 3.413 m. A valid local improvement certificate therefore does not establish a competitive denoiser or an improved forecast.

## Bounds and transport conditioning

Among the 360 cases satisfying the witness contract, the ray lower bound fails in 18/360 records; the space lower bound fails in 0/360. The underlying space confidence event fails in 13/360. The latter event guarantees all directional lower bounds; a failure of that sufficient event need not violate the particular selected bound. These are measured counts, not replacements for the mathematical assumptions.

Across these cases, future amplification ||E(8)||/sigma has median 16.84, 90th percentile 32.75, and maximum 132.95. It measures endpoint change per unit stacked witness change within the proposed correction space. Weakly witnessed combinations can therefore generate large endpoint changes. This operator norm is a diagnostic, not a direct error bound on an arbitrary future path.

The one-direction correction avoids reweighting individually weak directions and forecasts much better than the full-space correction. It still does not consistently improve over the original all-history Kalman model. Its lack of a future guarantee remains unchanged.

## Identical observations, incompatible truth

The maneuver and drift cases have exactly identical histories and exactly identical corrected outputs, verified for all 120 matched pairs. Their clean endpoints differ by 6.400005 m. Any common point prediction must be at least 3.200003 m from one of those two truths. Any set containing both endpoints must have diameter at least 6.400005 m. These conclusions need no estimator-specific argument.

The drift breaks the witness-noise assumption. At sigma 0.35 the ray and space lower bounds fail in 48/60 and 46/60 drift cases respectively. Independent random noise on top of that drift does not make the drift independent zero-mean sensor error. The experiment cannot identify its cause from those observations alone.

## Consequence for the construction

The proof extends to any calibrated closed convex projected-noise region K: the maximum guaranteed-gain correction is v minus its projection onto K. An allowed coherent-nuisance subspace makes K a cylinder; its nuisance component cannot support a guaranteed correction. These are mathematical extensions, not additional fitted benchmark results. See Sections 3a and 6a of WITNESS_GEOMETRY.md.

A current prior can propose a correction, and untouched observations can establish a local benefit without treating that prior as truth. That is a useful interface. It does not identify an unrestricted corruption process, certify a physical continuation law, or justify transporting every locally supported change into the future. A further construction must specify which relationships support continuation, retain future-invisible directions, and account for coherent sensor alternatives. No new denoising loop is called for by this result.

## Reproduction and artifacts

```sh
sh experiments/civilian_transport/run_witness_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.witness_report
```

Nine focused tests cover the uniform confidence bound, basis and spatial equivariance, zero/rank-deficient fields, future nullspace, the ray bound, exact null rate, absence of witness/future leakage, and the indistinguishable observation pair. Existing relational Kalman tests run alongside them. Source hashes, full results, certificate records, traces, and paired bootstrap summaries are retained in `witness_results/`. The confidence theorem is pointwise per procedure/record; it is not simultaneous across the entire experiment.

![Local versus future](witness_results/local_vs_future.png)

![Evidence controls](witness_results/evidence_controls.png)

![Identical observations](witness_results/identical_observations.png)
