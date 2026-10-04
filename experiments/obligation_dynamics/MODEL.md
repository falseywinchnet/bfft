# Obligation Dynamics: retained motion and equilibrium

This implements the user's proposed organization: each object keeps its motion
and the relationships that motion predicts. A deadline says when a prediction
must be revisited. Impact or changed support invalidates affected predictions;
an unchanged equilibrium does not generate periodic collision or solver work.

The dynamics and scheduler are new code. Convex cooking and manifold generation
are shared with the existing Wrench engine, so both engines can receive the same
general convex and compound geometry. Wrench's world, global solver and matrix
factorization are not linked into this engine.

## The equilibrium atomic

Let the shared contact normal `n` point from B toward A. At one shared world
point, the relative contact velocity is

```
u = (v_A + omega_A × r_A) - (v_B + omega_B × r_B)
k_n = 1/m_A + 1/m_B
    + (r_A × n)^T I_A^-1 (r_A × n)
    + (r_B × n)^T I_B^-1 (r_B × n)
```

Static bodies have zero inverse mass and inverse inertia. The normal impulse
atomic changes an accumulated nonnegative impulse by

```
lambda_new = max(0, lambda_old + (target_velocity - n·u) / k_n)
delta_p_A =  n (lambda_new - lambda_old)
delta_p_B = -delta_p_A
```

The equal-and-opposite impulses also change angular momentum by `r × delta_p`.
This is the precise form of the proposed mass-times-velocity-along-a-shared-vector
exchange. Momentum has units kg m/s; kinetic energy is quadratic in velocity.
For a scalar impulse increment `j`, its kinetic-energy change is
`j (n·u) + 0.5 k_n j^2`. A zero-restitution coordinate update minimizes that
quantity subject to the unilateral constraint. Tangential updates use the
two-dimensional effective-mass matrix and the Coulomb disk. A line search
prevents the inexpensive radial disk candidate from adding kinetic energy.

An impact changes velocities immediately. A standing obligation retains
**force**, including its tangential component. To update that field, the code
predicts free velocity over a short horizon, applies the saved forces times the
horizon as a starting impulse, and transmits corrections through neighboring
contacts. Dividing the resulting impulses by the horizon gives the next force
field. Actual present velocities are restored after this virtual calculation;
the event scheduler advances bodies under that field. This is a finite-horizon
support approximation, not an exact continuous acceleration solve. Gyroscopic
acceleration enters the predicted angular velocity; changes of contact geometry
are handled by the next refresh rather than a full differential contact model.

Near rest, a separate zero-velocity trial tests whether the retained forces
balance gravity and torque. Changed response enqueues adjacent atomics;
propagation ends at the measured correction tolerance, with a finite budget and
an explicit unconverged counter. No assumed attenuation through the floor is
used as proof that transmission has converged.

The code uses standard rigid-body impulse and constraint mechanics in this
retained, deadline-driven organization. Local projected impulse operations are
not a claim of a newly discovered conservation law.

## Pending and standing obligations

* **Pending:** a conservative cast predicts when two shapes might approach the
  contact skin. It retains the deadline and both motion revisions. A changed
  revision invalidates the event. A conservative early arrival is a recheck,
  not an invented impact.
* **Standing:** a contact keeps local witnesses, a normal and support/friction
  forces. A changing connected group updates support on a shared clock,
  currently capped at 1/240 second. New impacts transmit immediately through a
  local queue, without rebuilding the entire group's support field on every
  arrival. The previous support field persists until its next scheduled update;
  this bounded lag must be assessed in the geometric and stability tests.
  Unrelated bodies are not included in that solve.
* A compound body pair can have standing and pending obligations simultaneously:
  touching one convex part does not suppress casts against its other parts.
* **Certified equilibrium:** all bodies in the group have sufficiently small
  surface speed and force/torque acceleration residual, feasible contact forces,
  and acceptable contact gaps. Their tiny residual kinetic energy is removed
  and counted. The retained force field supports the stationary pose. No new
  event is scheduled until an arriving body, external impulse, teleport or
  support removal changes the relationship.

This last operation is a tolerance-based equilibrium certificate, not exact
symbolic statics. Its residual is independently reconstructible from gravity
and the saved contact forces. Merely moving slowly does not qualify.

## Motion, casts and scheduling

Translation follows a quadratic trajectory under its current constant force;
free fall is therefore analytic and independent of the observation frequency.
Angular momentum is transported with the retained torque. Orientation uses a
second-order exponential update, with angular-motion limits on its validity
horizon. This is a numerical angular integrator, not exact Euler-top motion.

A convex separating-axis gap is a lower bound on distance. A bound on relative
surface closure includes translation, acceleration and each body's angular
sweep. The cast advances by at most the gap divided by that bound, then checks
again. A cast budget produces an earlier recheck; it never licenses unchecked
motion through the rest of the interval. The heap orders deadlines. The API's
`advance` endpoint is an observation boundary, not a required dynamics tick.

Each body is indexed in a dynamic AABB hierarchy by an envelope of its future
trajectory. Quadratic translation extrema and a radius-based rotation expansion
bound that envelope. Only changed envelopes are reinserted. Queries create pair
obligations; no all-body-pair loop is used for discovery. Group traversal uses
visit generations, and atomic adjacency is allocated only for the affected
group. The free-flight envelope currently has a 0.1-second validity cap.

Each separated convex-part pair also retains a world-space separating plane.
Projected quadratic translation and a bound on rotational motion can certify
that this plane remains separating through the entire validity interval. That
certificate skips the part pair's future casts. Parts that remain candidates
are transformed individually; a single candidate does not require transforming
every other part of the compound at every cast sample.

For the orientation path `R(t) = exp(phi(t)) R0`, where `phi` is quadratic,
the instantaneous angular rate is `Omega = J_l(phi) phi_dot`. The integral
representation `J_l(phi) = integral_0^1 exp(s [phi]_x) ds` gives
`|Omega| <= |phi_dot|` and
`|Omega_dot| <= |phi_ddot| + 0.5 |phi_dot|²`. Consequently a point within radius
`R` has rotational acceleration bounded by
`R (|phi_ddot| + 1.5 max_interval |phi_dot|²)`. The plane certificate uses each
part's minimum/maximum initial projected rotational velocity and this quadratic
remainder. This bound applies to the implemented orientation prediction, not
an assertion that the prediction is the exact rigid-body trajectory. Motion
changes invalidate the certificate before it can be reused.

Fast motion does not inherently demand a high update rate. Fast motion near an
obstacle produces an early deadline; unobstructed flight can retain its analytic
trajectory. Friction is included in the finite-horizon support calculation.
Dense moving groups are batched through their standing-contact clock rather
than represented as infinitely many tiny impacts. This implementation still
rebuilds rows within a changing connected group at a support update; it does not
claim fully local force propagation in every dense scene.

## What the validation must establish

Tests cover hierarchy queries against exhaustive bounds, analytic flight and
observation-rate invariance, unequal-mass elastic impact, off-centre momentum
and energy, thin-slab impacts over several speed orders, rotation-only impact,
support removal, load-change wake-up and zero-work stationary scenes. Shared
packing scenes are scored by the independent geometry oracle from the previous
study. Report geometric error, retained residuals, rejected/budgeted updates,
equilibrium energy removal and wall time together.

Passing a finite speed sweep is evidence for those tests. It is not a proof of
correctness at arbitrary speed, arbitrary geometry, or unbounded floating-point
range. A smaller final gap alone also does not establish physical accuracy.

Background: [continuous collision and conservative advancement](https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf),
[contact forces and constraint mechanics](https://mujoco.readthedocs.io/en/latest/computation/).
