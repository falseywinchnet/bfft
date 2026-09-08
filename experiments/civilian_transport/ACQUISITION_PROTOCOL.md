# Sampling density as the accuracy/cost control

The tracker is one retained relational-current Kalman system. Its update law,
current basis, and prior stay fixed across the boundary. Each acquired position
is integrated exactly once at its actual timestamp. The ordinary
history-conditioned Kalman gain supplies the continuously weighted correction.
This experiment does not choose between the earlier witness constructions.

## Cadence and trigger

The sampler begins at 2 Hz. Observation timing is semiperiodic: each interval
advances a phase amount chosen independently and uniformly in [0.8,1.2]. At a
constant rate this makes intervals within +/-20% of the nominal period.

After an acquired observation, the tracker evaluates its estimated x position.
The first estimate with x>=0 latches an increased sampling request. Clean truth
and the actual crossing time are not available to this controller. The boundary
is x=0, and it is not a force, physical barrier, or motion-law input.

For a target rate r1, after trigger time t0 the requested rate is

    r(t) = 2 + (r1-2)*(1-exp(-(t-t0)/1 s)).

The rate is continuous at the trigger. The next observation occurs when the
integral of that rate advances by the next random phase amount. The state and
its covariance are retained, with no reset at the trigger. Observation updates
are discrete corrections; a continuous sampling-rate command does not imply a
pointwise continuous estimated state at the moments measurements arrive.

The fixed comparison profiles are constant 2 Hz, adaptive 2->4/8/16/32 Hz, and
constant 32 Hz. These are acquisition-budget experiments using the same
integration rule, not alternative estimators selected at runtime. Adaptive
profiles share exactly the same observation history and estimated trigger time
until their cadences diverge, checked in the runner.

## Fixed representation and noise contract

The represented current has 128 orthonormal cells on 0–14 s. The evaluated
interval is 0–12 s with a two-second forecast at every 0.1 s evaluation time.
No future position values enter the estimator. Independent Gaussian position
noise has supplied sigma 0.35 or 1.4 m. The existing five-length current prior
and its evidence-weighted uncertainty are unchanged.

Because the next acquisition time depends only on past observations and
independent timing jitter, the acquisition policy does not introduce an extra
likelihood for the newly acquired position. It changes the observation design.
This statement uses the independent sensor-noise contract. No benefit from
repeated samples of an unrestricted coherent drift is asserted here.

The finite horizon is a study representation, not a claim of an indefinitely
running flight implementation. A sliding/extended representation needs its own
state and covariance transport. Four 128->256-cell checks examine sensitivity
of this retained experiment; they do not establish a universal resolution bound.

## Motion fixtures and matching

Five boundary-crossing fixture families use lateral motion from the existing
line, helix, figure-eight, stop/go, and waypoint-derived generators. The crossing
axis is x(t)=2*(t-6)+0.3*sin(0.7*t+phase), so the true crossing occurs near 6 s.
The physical motion does not react to the boundary or to the sampling request.
These are declared derivatives of the earlier fixtures, not unchanged runs of
the previous benchmark. A fixed 0.01 s truth grid is linearly interpolated at
all acquisition times, identically for all policies.

Eight fresh seeds per family and two noise levels give 80 paired cases and
480 main policy/case records. Noise and timing random streams are matched by
acquisition index. All profiles see the same underlying path; after their
cadences diverge they need not acquire at identical times. No threshold, rate,
or prior is selected by clean-truth evaluation.

## Benefit and cost accounting

Accuracy is evaluated at the same fixed physical times for every policy, not
weighted by how many observations a policy acquires. Results are split into
approach, first second after the true crossing, 1–3 s after, and 3–6 s after.
True crossing time is used only for terminal evaluation and delay reporting.

Report these separately:

* 3-D current-position RMSE and two-second forecast RMSE.
* Moment-ellipsoid size and empirical coverage, as diagnostics for this mixture
  and these out-of-model trajectories rather than exact 95% guarantees.
* Acquired sample counts: a direct acquisition/transmission burden.
* Measured assimilation milliseconds: calls that integrate new positions.
* Initialization, operational readouts, and cadence-scheduler milliseconds.
* Correction sizes and estimated crossing delay across the transition.

Diagnostic pre-update readouts used solely to measure correction size are timed
separately and excluded from the reported online workload. Every policy has the
same regular evaluation/readout frequency; acquisition readouts additionally
track the current estimate at actual observations. Timings are actual M4 CPU
scopes, not universal latency guarantees. No hardware acquisition cost, energy
per fix, radio price, or network delay is invented.

For policies a and b, the observed marginal return can be reported as

    (MSE_a-MSE_b)/(assimilation_ms_b-assimilation_ms_a)

and separately per extra acquired sample. Report signed values rather than
forcing monotonic improvement or selecting a winner. A deployment-specific
sample cost c_s can be combined later as T_online + c_s*N_samples when c_s is
expressed in time-equivalent units; neither axis is silently discarded here.

See ACQUISITION_FINDINGS.md and acquisition_results/ for retained measurements.
