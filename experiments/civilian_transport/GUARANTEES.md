# Observation-dependent guarantees for geometric transport

These are analytic guarantees conditional on explicit physical and observation
bounds. They are independent of the number of filter particles. The existing
stochastic filter has unbounded Gaussian innovations and does **not** establish
these physical bounds or deterministic observation-error bounds. Its empirical
95% regions are not certified by this document. No additional Monte Carlo
trajectory battery is needed for the results below.

## 1. What can be guaranteed

Let observations be y_i = x(t_i) + e_i, with ||e_i|| <= epsilon_i. Define the
admissible path set by these observation balls and a specified geometric
transport law. Its evaluation at T is the exact reachable set R(T). Every
admissible true path lies in R(T), by construction. Adding an observation or
reducing an error bound can only shrink this set at a fixed T. Empty feasibility
means that the observations and assumptions conflict; it is not evidence that
an empty region safely describes the target.

No finite forecast bound follows from noisy positions alone with unrestricted
future motion: paths identical throughout the past can make arbitrarily large
smooth departures after the last observation. With a speed bound V alone, one
observation yields B(y_i, epsilon_i + V*(T-t_i)). Intersect these balls for all
observations. This already gives a count-independent, continuous error bound.
Learning a direction well enough for a sharper forecast requires a further
restriction on changes of transport.

One such intrinsic restriction is

    x' = s*u,  ||u|| = 1,
    0 <= s <= V,  |s'| <= a_parallel,  ||u'|| <= Omega.

Since u dot u' = 0, almost everywhere,

    ||x''||^2 = (s')^2 + s^2*||u'||^2 <= a_parallel^2 + V^2*Omega^2 = A^2.

An alternative curvature bound ||u'|| <= kappa*s gives
A = sqrt(a_parallel^2 + kappa^2*V^4). These restrict admissible continuations;
they do not introduce estimated Cartesian acceleration/jerk state variables.
The acceleration envelope is a conservative relaxation of the intrinsic law:
it loses directional restrictions but gives a coordinate-invariant outer
certificate. Assume x' is absolutely continuous, with ||x''|| <= A almost
everywhere on the observation and forecast interval. The fixed A must be
justified externally; selecting it from the same noisy observations without a
separate guarantee does not establish coverage.

## 2. A sharp two-observation theorem

Take t_a < t_b <= T; write Delta=t_b-t_a and h=T-t_b. Set

    c(T) = y_b + (h/Delta)*(y_b-y_a),
    r(T) = (1+h/Delta)*epsilon_b + (h/Delta)*epsilon_a
           + (A/2)*h*(Delta+h).

Then ||x(T)-c(T)|| <= r(T), in any Euclidean dimension.

Proof. Translation permits t_b=0, t_a=-Delta. The exact extrapolation residual
for the uncorrupted positions is the integral of x''(q) against

    K(q) = h*(q+Delta)/Delta,   -Delta <= q <= 0,
           h-q,                0 <= q <= h.

Integrating twice proves this identity; affine paths have zero residual. K is
nonnegative and its integral is h*Delta/2+h^2/2. The norm of this integral is
therefore at most A*h*(Delta+h)/2. The two observation errors enter with
coefficients -(h/Delta) and 1+h/Delta, whose absolute values give the remaining
terms. This proves the assertion.

Sharpness. In one dimension choose x(t)=A*t^2/2, e_a=+epsilon_a and
e_b=-epsilon_b. The future residual equals r exactly. Embedding this example
along any axis proves sharpness in 3-D for the relaxed acceleration class.
Additional intrinsic speed/direction limits can exclude this extremizer, so
sharpness is not claimed for every more restrictive transport class.

If a geometric estimator predicts g(T), it has the certified error envelope

    ||x(T)-g(T)|| <= r(T) + ||g(T)-c(T)||.

Thus c is a certificate center; the geometric estimator can remain unchanged.
Intersect the pair balls for every available pair and any speed-bound balls
for a tighter outer region. Membership is a finite set of norm inequalities;
no optimizer is needed. The intersection is not claimed to equal R(T).
Numerically evaluating the formulas in ordinary floating point is not a
machine-checked, outward-rounded interval certificate.

## 3. Sparse observations and smooth degradation

For equal errors epsilon the pair radius is

    r = epsilon + 2*epsilon*h/Delta + A*h*Delta/2 + A*h^2/2.

The terms expose measurement uncertainty, noisy direction inference, accumulated
past turning/speed change, and unpredictable future change. At positive Delta
this depends smoothly on errors, horizon and observation separation. With
A>0, epsilon>0 and h>0, the unconstrained best separation is
Delta*=2*sqrt(epsilon/A); choose an available separation within the history.
Extremely close noisy observations can be less informative for extrapolation.
For n uniformly spaced estimates over a fixed interval L, using the latest
estimate and any earlier one gives the explicit outer radius

    r_n = epsilon + A*h^2/2
          + min over m=1,...,n-1 of
            [2*epsilon*h*(n-1)/(m*L) + A*h*m*L/(2*(n-1))].

This scalar minimum is continuous and piecewise smooth for fixed n and positive
L. The candidate centers differ, so select the ball belonging to the minimizing
pair; do not attach its radius to an unrelated center. Keeping every old pair
constraint guarantees set inclusion when observations are added. Merely
replacing one uniform grid by another need not give nested observations or
monotonic radii. Hard feasible sets can change abruptly at inconsistency;
universal smoothness of exact feasible sets is not claimed.

Count alone cannot imply a 1/sqrt(n) deterministic reduction: an unknown common
bias e, ||e||<=epsilon, remains compatible with arbitrarily many samples.
Similarly, even exact knowledge of all past motion cannot remove the future
A*h^2/2 minimax floor in the relaxed class: the two continuations starting with
the same position/tangent and future accelerations +A and -A are separated by
A*h^2. These conditional futures need no derivative-state estimator.

## 4. All observations, individual errors, and action bounds

For any weights w_i fixed from timestamps and declared uncertainty, satisfying
sum(w_i)=1 and sum(w_i*t_i)=T, define c_w=sum(w_i*y_i) and

    K_w(q) = (T-q)_+ - sum_i w_i*(t_i-q)_+,
    B_w = integral from t_min to T of |K_w(q)| dq.

The same integral identity proves the general bound

    ||x(T)-c_w|| <= sum_i |w_i|*epsilon_i + A*B_w.

K_w is piecewise linear, so B_w has a finite algebraic evaluation, splitting
at timestamps and zero crossings. This is a theoretical alternative; the
current implementation supplies the simpler sharp pair certificates.
For an energy budget integral ||x''||^2 dq <= E, Cauchy-Schwarz instead gives
sqrt(E)*||K_w||_2 for the motion term. For a pair this is

    sqrt(E)*sqrt(h^2*Delta/3 + h^3/3).

An intrinsic action budget on (s')^2+s^2||u'||^2 is exactly this energy budget.
An action penalty in a stochastic filter is not a hard budget unless explicitly
restricted or assigned a justified tail probability.

## 5. When the count can improve confidence as 1/sqrt(n)

An additional statistical assumption makes the familiar rate legitimate.
Suppose independent centered isotropic Gaussian errors e_i with per-coordinate
standard deviations sigma_i. For deterministic weights above,

    tau^2 = sum_i w_i^2*sigma_i^2.

With probability at least 1-alpha, simultaneously across the three coordinates
at a single fixed target T,

    ||x(T)-c_w|| <= A*B_w + tau*sqrt(6*log(6/alpha)).

Proof: apply the scalar Gaussian tail bound to each coordinate at threshold
tau*sqrt(2*log(6/alpha)), union bound the three coordinates, and bound the
Euclidean norm by sqrt(3) times that threshold. This conservative constant
avoids a numerical chi-square quantile. Gaussian variance and independence
must be supplied or covered by a separate valid calibration theorem.

For equal sigma and ordinary linear-regression weights,

    w_i = 1/n + (T-t_bar)*(t_i-t_bar)/S_tt,
    S_tt = sum_i (t_i-t_bar)^2,
    tau^2 = sigma^2 * [1/n + (T-t_bar)^2/S_tt].

For n uniform times over [-L,0], T=h,

    tau^2 = sigma^2/n * [1 + 12*(n-1)/(n+1)*(h/L + 1/2)^2].

Only the noise term decays as 1/sqrt(n); the motion remainder A*B_w remains.
This is a bound for these timestamp-defined weights, not a claim that linear
regression captures the target's transport or that the total radius is optimal.
For correlated Gaussian errors replace tau^2 by w^T Sigma w per coordinate
(with suitable coordinate bounds); a shared bias does not average away.
Observation-dependent selection of the smallest stochastic region requires
simultaneous coverage or a separate selection argument. A confidence bound at
one target time does not certify a whole continuous forecast tube.

Alternatively choose observation-ball bounds with joint coverage 1-alpha and
apply the deterministic pair theorem to all pairs and all future times within
the physically bounded interval. Independence is not required if joint coverage
comes from a union bound. If each component error has known Gaussian marginal
standard deviation at most sigma_i, then

epsilon_i = sigma_i*sqrt(6*log(6*n/alpha)) supplies such coverage by a union
bound over 3*n coordinates. Unbounded outliers or unidentified bias invalidate
that particular error model. If the physical envelope itself fails with
probability beta, total coverage is at least 1-alpha-beta, without independence.

## 6. Civilian boundary queries

For a unit normal n and boundary n dot x = b, a ball B(c,r) proves the target
lies strictly on the positive side if n dot c-b > r, and strictly on the
negative side if n dot c-b < -r. Otherwise that ball alone is inconclusive.
Enforcing a strict same-side margin for every time in an interval certifies
no crossing there. Checking only sampled forecast times cannot do this.
For a single fixed pair, c(h) is affine and r(h) is a convex quadratic for
h>=0, so each same-side margin is concave. Positive margin at both endpoints
therefore certifies a positive margin throughout that interval. This provides
a finite continuous-time no-crossing test under the theorem's assumptions.
Neither uncertain membership nor a worst-case reachable crossing is a
calibrated crossing probability.

## Verification and context

`test_certificates.py` checks the sharpness identity with exact rational
arithmetic, finite rotational paths, retained pair constraints, error
monotonicity and invalid inputs. The integral arguments above are the proofs;
sampled path checks are supplementary implementation tests.

This is set-membership estimation and reachability, an established framework:
[Bertsekas and Rhodes, Recursive State Estimation for a Set-Membership
Description of Uncertainty (1971)](https://faculty.engineering.asu.edu/bertsekas/wp-content/uploads/sites/129/2020/03/RecursiveStateEstimation.pdf).
The explicit formulas here are derived above for the present position-only
transport setting; they are not claimed as a new general estimation theory.
