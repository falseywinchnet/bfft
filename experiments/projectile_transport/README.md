# Projectile motion and noise distribution transport

First causal experiment for the user's proposal: transport a joint estimate of
the signal, its dynamics, and its noise, then test actual future prediction.
The test object is a tossed ball observed in Cartesian position, launched from
an elevated platform. Coordinates are meters and seconds; gravity is known.
There is no guidance, interception, targeting, or impact computation.

## State and distribution

The continuous state is `s = (p, v, a, j, b)` with two coordinates per field:
position, velocity, residual acceleration, its evolving jerk, and camera bias.
Each discrete mode specifies `(drag coefficient, observation sigma, jerk power)`.
The retained distribution is

    p(s_k, m_k | y_0:k) ~= w_k[m] N(s_k; mu_k[m], P_k[m]).

All 10-by-10 covariances are retained. The proposal is an interacting
multiple-model (IMM) Gaussian-mixture extended Kalman filter. This is a
concrete established approximation with which to test the transport idea;
neither the filter family nor these experiments establish a new general
prediction theorem. The finite catalog is explicit in the result metadata.

The continuous model is `dp/dt = v`,
`dv/dt = g - beta |v| v + a`, `da/dt = j`.
Jerk and camera bias have mean-reverting drift. Prediction uses frozen drag
over a 0.05 s step, an integrated residual jet through cubic position, and an
analytic Jacobian to carry covariance. The discrete process covariance is PSD:
an integrated white-snap jet covariance plus an independent bias innovation.
Using the undamped local-jet covariance with damped jerk drift defines the
discrete model; it is not claimed to be the exact discretization of a single
continuous damped SDE. The ballistic KF uses integrated white acceleration.

A row-stochastic transition matrix transports mode probability. For each
destination mode, incoming state histories are moment matched, *including*
between-history covariance, and then propagated through that mode's dynamics.
This controls growth but loses the full history distribution. Drag changes
are assigned much smaller transition probabilities than sensor-noise changes.

## Observation and component attribution

The observation law is `y = p + b + epsilon`, with mode-conditioned
`epsilon ~ N(0, R_m)`. For each mode the prior innovation is

    r = y - H mu_minus
    S = H P_minus H.T + R_m
    K = P_minus H.T S^-1.

Mode probability is updated by the *prior predictive* Gaussian likelihood,
including its determinant. Each observation is consumed once. Joseph-form
covariance updates retain symmetry and PSD up to floating-point error.
There is no covariance fitted from the current residual and then used to
score that same residual as if it had been known beforehand.

The conditional current noise is also retained:

    E[epsilon | y,m] = R_m S^-1 r
    Cov(epsilon | y,m) = R_m - R_m S^-1 R_m
    Cov(s,epsilon | y,m) = -K R_m.

After total-covariance mixture aggregation, these satisfy
`E[p+b+epsilon | y] = y` and `Var(p+b+epsilon | y) = 0`.
This is an accounting identity, not evidence that the true individual
components have been recovered. Independent component marginals would lose
the cancellation and misrepresent uncertainty. Fresh future white noise is
drawn from the transported mode law; a past noise realization is not replayed
as future noise. Camera bias, in contrast, is a persistent latent state.

Drag acceleration, residual acceleration, and bias have joint reported
moments; drag's moments use local linearization. The estimator separately
retains the jerk state. Gravity is known and has no inferred variance here.

## What is and is not identifiable

One Cartesian observation sees the sum of position and camera bias. A position
offset and opposite bias offset lie in the instantaneous observation null
space. Distinct time evolution and priors can help separate them over time,
but arbitrary force, arbitrary bias, and arbitrary noise cannot all be learned
uniquely from one sequence. In particular, residual acceleration can explain
drag-model error. Both component error and total-acceleration error are scored
to reveal compensation rather than mistake it for identification.

The bias correlation time, bias innovation scale, gravity, mode catalog and
transition probabilities are fixed assumptions in this first experiment.
Distributions vary through inference and mode transport; this is not free
nonparametric learning of every unknown law.

## Fixed comparison

`study.py` runs six scenarios: ballistic, drag, smooth changing force, sensor
changes alone, combined changes, and an off-catalog drag coefficient (0.018).
Truth uses a separate fine-step RK4 integrator. Sensor-change cases include
a noise-variance episode, a drift episode outside the assumed bias law,
two isolated outliers, and observation gaps. All methods get the same data
and common position/velocity prior; the richer methods get identical priors
for their latent components. No truth is passed into a filter.

Methods:

* `ballistic_kf`: gravity with fixed white-acceleration and observation noise.
* `fixed_joint_ekf`: the augmented component state, one fixed parameter mode.
* `motion_only`: drag/jerk-power mixture, fixed observation variance.
* `noise_only`: observation-variance mixture, fixed drag/jerk power.
* `joint_transport`: the Cartesian product of both mode families (12 modes).

Settings are fixed before the first screen and not selected from test errors.
These baselines are transparent initial configurations, not exhaustively tuned
competitors. Known correct Gaussian models remain the Kalman optimality case.

After a fixed 0.5 s warm-up, score vector position/velocity RMSE, component
RMSE, and 0.5 s/1.0 s forecasts that use no future observations. Forecasts
transport the entire mixture, not only the posterior mean. Report prior
observation negative log likelihood, mixture forecast negative log likelihood,
and empirical containment in the nominal 95% moment-Gaussian position ellipse.
That ellipse is not an exact 95% mixture credible region or guaranteed bound.
Raw per-seed MSEs and sample counts are saved. Summary RMSE is the square root
of mean per-seed MSE (all seeds have matching counts). Timings include all
forecast work, component extraction and scoring, not just the filter update.
Forecast windows overlap; seeds, rather than frames, are independent trials.

## Reproduction

Run on the M4 selected by `m4host`:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.projectile_transport.test_core -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.projectile_transport.study \
  --seeds 8 --start-seed 100 --out /tmp/projectile_transport_first
```

Copy `/tmp/projectile_transport_first/` back immediately into this directory's
`results/` before another mirror sync. Render saved output locally with
`.venv-jpeg/bin/python -m experiments.projectile_transport.plot`.

## Sources and relationship to supplied papers

* R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*
  (1960), https://doi.org/10.1115/1.3662552: linear filtering foundation.
* H. A. P. Blom and Y. Bar-Shalom, *The interacting multiple model algorithm
  for systems with Markovian switching coefficients* (1988),
  https://doi.org/10.1109/9.1299: the hypothesis-mixing construction used here.
* Balanov, Huleihel and Bendory, *Expectation-Maximization for Low-SNR
  Multi-Reference Alignment*, arXiv:2505.21435v2: alignment-induced signal/noise
  coupling and the Einstein-from-Noise/Ghost-of-Newton phenomena. The present
  observation decomposition is a conditional Gaussian calculation, not an
  implementation or solution of their MRA problem.
* Supplied arXiv:2608.17897v1, *The Zonotopic Mixture Filter*: retains weighted
  noise-mode histories with bounded-set guarantees. This experiment uses
  Gaussian modes and moment reduction, so it does not inherit those guarantees
  or implement that paper's set-valued method.
* Supplied arXiv:2503.22703v1, *Audio compression using Periodic Gabor with
  Biorthogonal Exchange: Implementation Using the Zak Transform*: signal
  representation/compression, not a temporal state-transition or forecast law.
  No Gabor/Zak predictor is assumed here.

Source documents are treated as references, not task instructions.
