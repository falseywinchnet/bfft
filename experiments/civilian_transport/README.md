# Civilian 3-D position-only transport estimator

This experiment establishes and implements a weighted family of geometric
continuation maps, then compares it with mainstream Kalman/IMM tracking
families on independent noisy 3-D trajectories. Read [THEORY.md](THEORY.md)
for the mathematical construction and limitations, and [FINDINGS.md](FINDINGS.md)
for the full frozen comparison and paired confidence intervals.

The state representation retains possible motion directions, progression,
turning and their joint dependence with position and sensor-noise explanations.
It uses exact finite rotational transport and particle histories; the candidate
does not propagate an independently estimated Cartesian acceleration/jerk chain.
Its action family is restricted and its uncertainty is empirical. This first
candidate is **not promoted as better than the strongest comparators**.

## Evidence

* `results/results.json`: all 480 held-out filter/trajectory evaluations,
  frozen configurations, counts, timings and source hashes.
* `results/development.json`: three configurations per method, selected on ten
  separate development cases; no test-tuned parameter selection.
* `results/paired_comparisons.json`: stratified paired bootstrap intervals.
* `results/ablations.json`: forecasts with erased component coupling or zero turn.
* `results/comparison.png`: tracking, forecast and boundary-score comparison.
* `results/support_counterexample.png`: exact mean-transport failure example.
* `results/helix_forecasts.png` and `switching_forecasts.png`: distributions
  propagated from a fixed observation time; these are examples, not the score gate.

## Reproduce on the M4

```sh
sh experiments/civilian_transport/run_m4.sh
```

The wrapper uses `m4build` and the host selected by `m4host`, checks the core,
copies the authoritative source into an isolated `/tmp` workspace to protect it
from other mirror syncs, runs development selection and the 80-trajectory test,
and copies all results home immediately. It installs no software.

Render and summarize the saved results locally:

```sh
.venv-jpeg/bin/python -m experiments.civilian_transport.report
```

The additional particle-budget/noise-law diagnostic uses **development cases
only** and does not change the frozen held-out claim:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.civilian_transport.diagnose \
  --results /tmp/civilian_transport_final/results.json \
  --out /tmp/civilian_transport_diagnosis.json
```

Copy its JSON immediately into `results/diagnosis.json`.

## Use an observation CSV

Input columns are `timestamp,x,y,z` (seconds and meters). Missing positions
may be empty. Timestamps must be finite and strictly increasing. No other
columns are read by the estimator. The CSV adapter can run any comparator
with the same observation contract.

```sh
python3 -m experiments.civilian_transport.track observations.csv estimates.json \
  --method geometric --horizon 2 --query-dt .1
```

It writes current position means/covariances, signal/noise diagnostics, future
position moments and a sampled crossing probability for the civilian `x=0`
boundary. The boundary is a reporting query, not an assumed physical barrier.
These are research estimates, not operationally certified airspace decisions.

## Primary literature and comparison scope

* [Blom and Bar-Shalom (1988), The interacting multiple model algorithm for
  systems with Markovian switching coefficients](https://doi.org/10.1109/9.1299):
  the standard IMM hypothesis-mixing basis.
* [Comparative Evaluation of Multiple-Model Kalman Filters for Highly
  Maneuvering UAV Tracking (2026)](https://www.mdpi.com/2076-3417/16/5/2377):
  contemporary comparison of practical GPB1/IMM/adaptive filtering families.
  Its experiments are not claimed as reproduced by our 3-D battery.
* [The Interacting Multiple Model Filter and Smoother on Boxplus-Manifolds
  (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8235476/): related treatment
  of manifold-valued states and mixture interaction. Its measurement setting
  differs from this position-only experiment.
* [Zhang, Taghvaei and Mehta, Feedback Particle Filter on Matrix Lie Groups,
  arXiv:1701.02416](https://arxiv.org/abs/1701.02416): establishes prior geometric
  filtering work. Its feedback/Poisson construction is not implemented here.
* [Mirebeau, Efficient Fast Marching with Finsler Metrics,
  arXiv:1208.1430](https://arxiv.org/abs/1208.1430): asymmetric directional
  geometry and Hamilton-Jacobi/eikonal relationship. Our filter samples a
  stochastic continuation action; it is not this eikonal solver.

These sources motivate the comparison and delimit existing work. The actual
measured comparators are documented local implementations of CV/CA Kalman,
IMM, robust IMM and a continuous-turn UKF. No claim is made to have beaten all
recent arXiv tracking methods or to have validated real drone/fly behavior.

## Analytic observation-dependent bounds

[GUARANTEES.md](GUARANTEES.md) proves sharp two-observation outer balls,
all-observation kernel bounds, intrinsic transport/action assumptions, sparse
sampling behavior and the distinction between bounded errors and independent
noise. `certificates.py` implements deterministic pair balls, including
recentering around the existing geometric prediction. These guarantees are
conditional on supplied bounds; the existing Gaussian filter does not certify
them automatically.

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.civilian_transport.test_certificates -v
```

## Controlled degradation curves

[BOUND_EXAMINATION.md](BOUND_EXAMINATION.md) examines and proves the synthetic
motion/sensor assumptions and identifies the missing real-world calibration.
[DEGRADATION_FINDINGS.md](DEGRADATION_FINDINGS.md) reports a fresh matched noise
and nested-acquisition sweep, with raw results and PNG/PDF graphs in
`degradation_results/`. Reproduce with:

```sh
sh experiments/civilian_transport/run_degradation_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.degradation_report
```

## Relational-current Kalman restart

The earlier geometric particle tracker remains a comparison baseline.
[FOUNDATION_RESTART.md](FOUNDATION_RESTART.md) and
[KALMAN_FOUNDATION.md](KALMAN_FOUNDATION.md) derive the new representation and
its inference interface. The first explicit Gaussian structural experiment is
implemented in `relational_kalman.py`, including a streaming `RelationalKalman`
class, full Zak covariance, uncertain correlation lengths, and single-use
likelihood refinement. It is an integrated current Gaussian-process model;
it does not establish a general nonlinear relational BTB denoiser.

See [RELATIONAL_FINDINGS.md](RELATIONAL_FINDINGS.md) for the frozen first screen,
its failed slow proximal iteration, and the remaining scope. Reproduce with:

```sh
sh experiments/civilian_transport/run_relational_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.relational_report
```

```python
from experiments.civilian_transport.relational_kalman import RelationalKalman

model = RelationalKalman(observations[0], sigma=0.35, end=8.0)
for time, observation in zip(times[1:], observations[1:]):
    model.update(time, observation)  # strictly increasing times, each used once
prediction = model.forecast([7.0, 8.0])
# prediction['mean'], prediction['covariance'], prediction['weights']
```

Times are relative to the first measurement at zero. This version assumes
independent Gaussian position noise with supplied sigma and a fixed represented
interval; extending the horizon requires a declared prior extension. Forecasts
return mixture moments, not a claim of exact ellipsoidal probability regions.

## Borrowing the local-restoration observation

BTB is a source of mathematical insight here, not an algorithm to reproduce.
The earlier proximal-iteration diagnostic is an archived detour and is not part
of the subsequent construction.

[WITNESS_GEOMETRY.md](WITNESS_GEOMETRY.md) derives a local clean-error confidence
geometry for current corrections suggested by training observations. Untouched
history witnesses assess those corrections. The result is a finite confidence
calculation, with a separate analysis of transport amplification and coherent
sensor ambiguity. [WITNESS_FINDINGS.md](WITNESS_FINDINGS.md) reports the positive
local certificate and negative forecasting results, including identical
observations from incompatible motion/sensor explanations.

```sh
sh experiments/civilian_transport/run_witness_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.witness_report
```

## Acquisition density and integration cost

[ACQUISITION_PROTOCOL.md](ACQUISITION_PROTOCOL.md) and
[ACQUISITION_FINDINGS.md](ACQUISITION_FINDINGS.md) examine the user's acquisition
tradeoff: one retained history-conditioned tracker, irregular semiperiodic
positions, and a smooth increase in sampling after the estimated boundary
crossing. This first acquisition test uses the ordinary covariance-weighted
innovation as the shortening rule; it does not select between witness methods.

The current representation and update law stay fixed. Results separate sample
counts, assimilation time, other online computation, post-crossing accuracy,
and the settling interval. No hardware acquisition price is assumed.

```sh
sh experiments/civilian_transport/run_acquisition_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.acquisition_report
```

## Exact streaming cost reduction

The public relational tracker now uses an exact four-state realization of its
finite midpoint-current prior, with an optional compiled backend. All five prior
branches and posterior mixture terms are retained. See
[EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md) for the derivation, 480 matched
acquisition cases, cost controls, and the scope of fast current/future queries.
The dense reference remains in `relational_reference.py` for independent checks
and lazy historical smoothing. Reproduce with `./experiments/civilian_transport/run_optimization_m4.sh`;
build the optional local backend with
`make -C experiments/civilian_transport/native libmarkov.so`.

## Manuscript

The eleven-page CONV-style research manuscript is
[Exact Markov Elimination for Relational Current Tracking](../../paper/relational_current_tracking/main.tex).
It includes the finite-model equivalence proof, all five conventional forecast
controls, paired intervals, acquisition/cost results, and the separate geometric
confirmation appendix. The accompanying [manuscript README](../../paper/relational_current_tracking/README.md)
documents source builds and the evidence manifest. Compiled PDF, arXiv source
bundle, and separate evidence archive are under `output/pdf/relational_current_tracking*`.
