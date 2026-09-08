# Examination of the assumptions behind the certificate

## What has and has not been established

For the local synthetic generator, the motion bound can be proved uniformly
from the generator's parameter ranges, without observing a case's true path.
For the controlled sensor experiment, its declared Gaussian law gives a joint
probability bound. The added persistent bias has a declared norm bound. These
are justified properties of the experiment, not inferred properties of actual
drones, flies or real positioning hardware. No real sensor calibration records
or physical target specification have been supplied. An operational bound for
those targets therefore remains unestablished.

The existing geometric filter has Gaussian random changes of direction and
speed, an adaptive noise scale, and a numerical speed clamp. Those features
alone do not imply the continuous motion-change bound used here. Certificates
are external restrictions on admissible truth, available equally to all the
estimators; they neither certify every particle nor improve point estimates by
themselves. The exact pair-bound proof is in [GUARANTEES.md](GUARANTEES.md).

## Uniform motion proof

The generator draws speed s in [1.5,3.5] and frequency f in [0.35,0.7]. Rigid
rotation and translation preserve the following Euclidean derivative bounds.
They hold throughout continuous time, including between measured positions.

* Line: x=(s*t, .08*sin(.6*t), .05*sin(.9*t)), so
  A_line=sqrt((.08*.6^2)^2+(.05*.9^2)^2).
* Helix: circular radius s/f plus linear vertical progression, so
  A_helix=s*f <= 2.45.
* Figure eight: coordinates (s/f)*[sin(f*t+phase), .45*sin(2*f*t),
  .35*sin(.7*f*t)], so
  A_eight <= 3.5*.7*sqrt(1+1.8^2+(.35*.7^2)^2).
* Stop/go: write x=r(q(t)), q=t-1.5*tanh((t-4)/1.5)+constant.
  Then q'=tanh^2((t-4)/1.5) lies in [0,1] and
  |q''| <= 8/(9*sqrt(3)). The route satisfies
  ||r'|| <= sqrt(3.5^2+.72^2+.3^2),
  ||r''|| <= sqrt(.432^2+.3^2), hence
  ||x''|| <= sqrt(.432^2+.3^2) + sqrt(3.5^2+.72^2+.3^2)*8/(9*sqrt(3)).
* Waypoint switching: every integration substep explicitly clamps the
  acceleration vector to norm <=5. Define the between-sample path by the
  implemented constant-acceleration polynomial on each substep. Position and
  velocity match at substep boundaries, so velocity is absolutely continuous
  and the bound holds almost everywhere. Command changes can jump acceleration
  but do not jump velocity. This is a guarantee for the numerical fixture's
  continuous extension, not for a separate exact waypoint ODE solution.

The sweep uses the maximum of these five bounds, about 5.062 m/s^2, for every
case. It does not pass the family label, sampled speed/frequency or measured
true acceleration into any estimator or certificate. A family-specific bound
would require identifying that family with a separate guarantee and is not
used in the scored results. Using a maximum of sampled finite differences
would not establish a continuous bound and is also not used.

This common bound implies a motion-only two-second term A*h^2/2 of about
10.125 m. It explains why the broad worst-case certificate can be much wider
than typical forecast errors. The sharpness theorem concerns the relaxed
acceleration class, not the smaller union of these five trajectory families.

## Controlled sensor laws and simultaneous coverage

The main sweep supplies independent centered errors N(0,sigma^2 I_3). For the
65 available history timestamps, set

    epsilon = sigma*sqrt(6*log(6*65/.05)).

The Gaussian coordinate tail bound and union bound give simultaneous
||e_i||<=epsilon with probability at least 95%. The same event covers every
nested subset and every observation pair. It is deliberately conservative;
the plots label its guaranteed coverage as at least 95%, not exactly 95%.
The fixed 65-sample event also avoids changing the error allowance simply
because a measurement was removed. For each sigma this confidence statement
is over the noise; it is not a claim about frequencies across all plotted
conditions or across every repeated run. Shared normalized noise in the paired
experiment couples conditions for more precise comparisons.

The second regime adds a constant vector b with ||b||=.75 m, with a different
random direction per case, shared across all times and all paired conditions.
Since affine extrapolation weights sum to one, this bias contributes at most
.75 m once. It must not be treated as independent noise that decreases with
sample count, nor amplified twice as unrelated errors if its shared structure
is known. The certificate adds .75 m to each Gaussian-error pair radius. The
original point estimators retain their frozen bias treatment; they are not
retuned to this challenge. All receive the true nominal independent-noise sigma.
The known bias bound is available to the certificate wrapper around every
method. Sensor-law correctness in this experiment is an assumption provided by
the generator, not an estimation success.

## The earlier contaminated sensor model needs different allowances

The original `make_case(..., corrupt=True)` uses time-varying Gaussian standard
deviation up to 1.1 m, a 3.5% additive Gaussian spike with coordinate standard
deviation 3 m, an OU-like correlated Gaussian bias and a deterministic bias
injection. Thus a certificate assuming only independent 0.35 m Gaussian noise
is invalid for those cases. Robust likelihood branches do not repair that
coverage claim.

A conservative valid allowance is available if these generator specifications
are known. Conditional on a spike pattern the total error is Gaussian with a
deterministic mean. For the original 121 observations, dt<=.12 s and total
time<=14.4 s. Dropping all bias decay factors bounds its per-coordinate
variance by .06^2*14.4. Bounding every sample as a spike gives total coordinate
variance <= 1.1^2+3^2+.06^2*14.4. The deterministic mean's norm is at most
(2+.12)*sqrt(.18^2+.08^2+.05^2), where .12 accounts for the injection's discrete
right-endpoint selection. Therefore a simultaneous 95% allowance is

    epsilon_original =
        2.12*sqrt(.18^2+.08^2+.05^2)
        + sqrt(1.1^2+3^2+.06^2*14.4)*sqrt(6*log(6*121/.05)).

More precisely, declaring epsilon equal to the right-hand expression gives a
valid upper confidence allowance. It is roughly 25 m per observation and
illustrates how expensive this worst-case treatment is. A sharper allocation
could use the known mixture tails and recursive marginal bias variances, but
would still be conditional on that complete sensor law. The new plots use
controlled noise and shared-bias regimes so that each horizontal axis changes
one quantity. They do not claim to cover every corruption in this older model.

## What would establish the remaining real-world assumptions

A usable motion envelope requires a specified target class and a justified
speed/turn/progression constraint, or another justified reachable-motion law.
The sensor allowance requires metrology/specification bounds or independent
calibration valid for the relevant operating conditions, including persistent
bias, outliers, timing error and dependence. Nominal standard deviation alone
is insufficient. Future timestamps are treated as exact here.

If calibration is statistical, its uncertainty must be included in the total
failure budget. Finite noisy positions cannot establish a hard universal
motion bound for unrestricted future targets. Observations inconsistent with
an assumed envelope should invalidate the envelope's application, not be
silently assigned a narrower region. This experiment establishes the
conditional theorem and its synthetic instantiation; it does not close the
real-world identification problem by assuming measured maxima are hard limits.
