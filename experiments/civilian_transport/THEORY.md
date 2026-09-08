# A distribution over geometric continuations

## Objective and observation contract

Infer current 3-D location, future position distributions and civilian geofence
crossing probabilities from `(timestamp, noisy position)` observations only.
No measured velocity, acceleration, attitude, control input, future waypoint,
true noise scale, clean trajectory, or trained trajectory catalog reaches the
estimator. Simulation truth is used only for development selection and scoring.

The previous `projectile_transport` derivative-chain experiment was rejected
by the user as a formulation of this objective. It is not this construction,
is not reused, and is not evidence for it.

## State is a weighted family of continuation maps

A support point contains a location x, direction u on S2, positive progression
s = exp(ell), and a rotation generator omega. The latter describes the geometry
of the motion, not the observed body's physical attitude. A persistent sensor
bias b and a log observation scale r complete the latent state.

For fixed s and omega, let W = [omega]_cross. Define the finite continuation

    u(t+h) = exp(h W) u(t)
    x(t+h) = x(t) + s integral_0^h exp(tau W) u(t) d tau.

The integral is evaluated with stable sine/cosine formulas, including omega=0.
The state of the estimator is a joint weighted distribution over these maps,
their locations, and their observation-noise explanations. No single mean map
is selected to replace the distribution during filtering or prediction.

**Exact properties of each fixed-law map:**

1. Rigid rotation of x, u and omega rotates the resulting trajectory. The law
   and its action do not prefer the observer's Cartesian axes.
2. Norm(u) is preserved; arc speed is s. Chord length need not be s*h.
3. Applying durations h1 and h2 equals applying h1+h2 when the law is fixed.
4. Where derivatives exist, a = s omega cross u and
   j = s omega cross (omega cross u). These are consequences of the same flow.
   They are not independently estimated polynomial coefficients.

This is a local screw-motion family (straight, circular and helical motion).
It does **not** describe every possible continuation exactly. Allowing its law
to change produces richer paths, with an explicit prior on how changes occur.
Using Lie-group structure alone is established practice, not a novelty claim.

## Potential/resistance on changes to the continuation

The stochastic transition uses finite rotations for direction perturbations,
Brownian increments in log progression, and Brownian increments of omega.
For increments d ell, d omega and a small heading rotation eta, its proposal
action, up to normalizing constants, is

    A = (d ell)^2/(2 q_s^2 h)
        + ||d omega||^2/(2 q_w^2 h)
        + ||eta||^2/(2 q_u^2 h).

This assigns resistance to counterfactual changes. Zero diffusion is a hard
constraint; larger diffusion allows a wider range of changes. All scalar
diffusions are rotation invariant. The finite proposal samples the distribution
defined by these latent increments and pushes it onto the motion manifold;
it does not assume that a Gaussian in position is the resulting distribution.

The weighted posterior over s, u and omega supplies an evolving family of
these actions. Its combined conditional density can be viewed as a mixture
potential `-log integral exp(-A_theta) d Pi(theta)`, with the appropriate
normalizers and reference measures. This effective potential can be nonconvex.
It is not a freely inferred function: the action family, its diffusivities and
numerical support bounds remain explicit assumptions. Their values are selected
on development data and frozen. The test stream updates the joint distribution
over continuation laws, not an arbitrary new action tensor.

**Connection to eikonal geometry.** In the continuous small-step idealization,
the resolved drift on `(x,u,ell,omega)` is `(exp(ell)u, omega cross u, 0, 0)`.
Its small-noise action has Hamiltonian

    H(z,p) = exp(ell) u dot p_x + (omega cross u) dot p_u
             + 1/2 [q_s^2 p_ell^2 + q_w^2 ||p_omega||^2
                    + q_u^2 ||(I-u u^T) p_u||^2].

Thus it has a Hamilton-Jacobi action description, while probability evolution
also includes diffusion and conditioning. This bridge does not turn this code
into an eikonal PDE solver, establish a static Finsler metric, or identify
minimum action with probability at finite noise. The proposed experiment
retains samples of the full probability law instead of only its action minimum.

## Joint signal/noise conditioning

The observation is y = x + b + epsilon. Conditional on a continuation history
and noise scale, x and b have an analytic Gaussian update. Each is a 3-vector;
the two-by-two field covariance tensored with I3 is exact within a retained
history because the process and observation noise are isotropic. Mixing
histories supplies arbitrary between-history 3-D covariance and non-Gaussian
shape. Conditional x-b cross covariance is never deleted.

Observation noise has a 97% nominal / 3% wide Gaussian contamination law, with
variance ratio 100. Both branches are retained after each observation. The
latent nominal log scale itself evolves with a mean-reverting stochastic law.
The fixed bias law has six-second correlation time and known assumed innovation
scale; it is not identifiable without assumptions from arbitrary position drift.

For every branch, with H=[I I], residual e and innovation covariance S,

    K = P_minus H^T S^-1
    E[epsilon | y,branch] = R S^-1 e
    Cov(x,b; epsilon | y,branch) = -K R.

Consequently `E[x+b+epsilon|y]=y`. All branch weights use the prior predictive
likelihood, including its covariance determinant. Observations are consumed
once; current-residual covariance fitting is not reused as prior evidence.
The persistent nominal noise law is transported; an individual old white-noise
realization is not replayed into the future.

There are N support histories before assimilation and 2N immediately after
branching. Systematic resampling before the next transition returns to budget N.
This is a Rao-Blackwellized particle approximation. Resampling introduces Monte
Carlo error and can impoverish support; no exact posterior or bounded-coverage
guarantee is claimed. The requested comparison includes timing and calibration.

## Why replacing the law by its mean loses a boundary fact

Take x=0, u=(1,0,0), s=1 and equal probability on omega=(0,0,+1) and
omega=(0,0,-1). Every supported path satisfies

    x_1(t)=sin(t),  x_2(t)=+/- (1-cos(t)).

Neither can reach the plane x_1=1.5. The average generator is zero; propagating
it gives x_1(t)=t, which crosses that plane. Even the predictive mean of the
actual distribution is different: it is `(sin(t),0,0)`, not `(t,0,0)`.
This is an exact example of a support/coupling failure, not a measured advantage
on unknown trajectories. It also does not show that all Gaussian filters fail.

## Comparators and what is tested

* CV and CA Kalman filters: standard position/velocity and acceleration models.
* IMM: CV, CA, and six fixed turns about positive/negative coordinate axes.
  All use a common augmented position/bias observation model and a three-second
  mean mixing scale. This fixed turn catalog is a limitation, disclosed explicitly.
* Robust IMM: the same model set, with the same contamination likelihood as the
  candidate, retaining moments when noise branches are recombined.
* Continuous-turn UKF: unknown three-dimensional omega, finite rotation and
  integrated position maps, with the same candidate choices for speed/turn
  diffusion. It is an important stronger control because it is not limited to
  the IMM's turn catalog. Its single Gaussian is over Cartesian velocity,
  omega, position and bias. The speed/heading process noise matches the geometric
  proposal locally to first order, not globally (Gaussian vs log/spherical).

The UKF uses alpha=1, beta=2, kappa=0. These are transparent implementations of
practical literature families, not reproductions of a recent paper's reported
benchmark. There is no unique universal trajectory-estimation SOTA. The results
can support a claim against these frozen implementations on this battery only.

Two candidate forecast ablations use the *same* filtered posterior: setting
turn to zero, and independently permuting progression/direction/turn across
draws. The latter preserves empirical marginals but destroys their coupling to
one another and to location. It is a coupling test, not a pure phase ablation.

## Evaluation and limits

Development uses ten cases (five independent simulator families, clean/corrupt).
Each method gets three predefined candidate configurations, selected by
`tracking MSE + .25 * two-second forecast MSE`. Test seeds do not overlap.
No configuration is changed after test results are inspected.

The five families are straight motion, tilted helices, spatial figure eights,
smooth stop/restart, and a craft approaching changing waypoints. The latter's
future commands are independently generated and hidden. Truth uses analytic
positions or a separate fine-step Cartesian simulation, not candidate code.
Random 3-D orientation and translation prevent one privileged observation frame.

Corruption includes a noise-scale episode, bias drift, isolated outliers,
random missing samples, and a longer observation gap. All methods receive the
same measurements and common position/bias and second-moment velocity priors.
Direction/speed and Gaussian-velocity prior *shapes* differ and are disclosed.
The nominal sigma (.35 m) is a common assumed sensor setting, not read from truth.

Tracking starts after 20 samples of warm-up. Forecasts use no future measurements,
only requested query timestamps; horizons span 10/20 irregular intervals
(approximately one/two seconds). Score RMSE, energy score, empirical nominal
95% moment-ellipsoid containment, and Brier scores for endpoint side and crossing
of the fixed civil boundary x=0. Crossing is checked at query samples, not
certified continuously between them. The boundary is a query, not a force or
known restriction on the motion. No filter is told the craft intends to obey it.

Full paths retain temporal dependence when estimating boundary-crossing events.
Independent endpoint draws would not suffice. Forecast random streams are
separate from filtering and cannot alter subsequent estimates. The complete
raw per-case metrics, development search, configurations and source hashes are
saved for review. Trials use synthetic positions; no real drone/fly validation
or operational airspace guarantee is established by this experiment.
