# Anchor, Euclidean descent, and transported Bregman state

Date: 2026-08-29  
Status: formal mechanism decomposition and research hypotheses

## Executive result

Anchor is not Bregman descent, and its spherical state transport is not the
same transport used by the BFFT Meyer solver.  It is a Euclidean optimizer
that imports one design principle from that work:

> A request learned at one stage should not be reused at another stage until
> its frame has been updated; geometry may restrain the incompatible part of
> the request, but acceleration is judged by total relaxation, not by the size
> of one step.

The difference between Euclidean and Bregman descent is not that one uses dot
products and the other does not.  Both consume the same first-order
differential returned by backpropagation.  The difference is the map that
turns that differential into a finite displacement and the rule used to
compare or carry state between base points.

Euclidean descent silently makes three unusually strong identifications:

1. a gradient covector is identified with a vector by one fixed inner product;
2. every tangent space is identified with every other tangent space by
   translation;
3. the finite step is the same vector as the infinitesimal steepest-descent
   direction.

For the quadratic potential `h(x)=||x||^2/2`, those identifications are exact.
For a general Bregman potential they separate.  Descent is a translation in
dual coordinates `y=grad h(x)`, while the corresponding primal displacement
depends on the current base point.  This is why first-order descent generalizes
cleanly and naive momentum does not.

Anchor remains on the Euclidean side of that divide.  It uses ordinary rowwise
dot products, norms, projections, and an orthogonal rotation.  Its novelty is
to construct a moving *observed gradient frame* inside the flat parameter
space, then condition historical momentum in that frame.

## 1. What backpropagation actually returns

At parameters `x`, backpropagation returns the differential

```text
df_x(delta x) = <g_x, delta x>.
```

Strictly, `g_x` is a covector: it tells the first-order change in the loss
caused by a trial displacement.  To call `g_x` a direction and subtract it
from `x`, an optimizer must choose a metric that raises the covector index.

With metric `G_x`, steepest descent is

```text
v_x = -G_x^{-1} g_x.
```

Ordinary SGD chooses `G_x=I` everywhere.  That choice is usually invisible
because the coordinate dot product identifies vectors and covectors.  It is
not coordinate-free.  Under a non-orthogonal reparameterization `x=phi(z)`,

```text
g_z = J_phi(z)^T g_x,
```

but the Euclidean vector steps `-g_z` and `-g_x` do not generally map into one
another.  The pairing `<g,delta x>` is invariant; treating the components of
`g` as a globally meaningful displacement is not.

Thus the user's “single-point pivot” intuition identifies a real limitation,
but not the defining difference.  Both SGD and mirror descent query the loss
at the current point.  Backpropagation is a memoryless first-order oracle in
both cases.  The divergence occurs *after* backpropagation:

- SGD uses the identity metric and a straight translation from the current
  point;
- mirror/Bregman descent translates the dual coordinate selected by `h` and
  maps the result back to the primal point;
- Anchor uses the Euclidean gradient, but adds temporal state that observes
  how its direction changes between points.

## 2. Euclidean descent is the quadratic Bregman special case

For a differentiable strictly convex potential `h`, define

```text
D_h(x,y) = h(x) - h(y) - <grad h(y), x-y>.
```

The mirror-descent step is

```text
x_(t+1) = argmin_x {
    eta <g_t,x> + D_h(x,x_t)
}.
```

Its first-order condition is

```text
grad h(x_(t+1)) = grad h(x_t) - eta g_t.          (1)
```

Writing `y=grad h(x)` gives

```text
y_(t+1) = y_t - eta g_t,
x_(t+1) = grad h*(y_(t+1)).                       (2)
```

The translation is simple in dual coordinates.  The primal displacement is
not.  With `G_t = Hess h(x_t)` and third derivative tensor `C_t`, expansion of
(1) yields

```text
delta x_t
  = -eta G_t^{-1} g_t
    - (eta^2/2) G_t^{-1}
        C_t[G_t^{-1}g_t, G_t^{-1}g_t]
    + O(eta^3).                                   (3)
```

For `h(x)=||x||^2/2`, `G=I`, `C=0`, and (3) is exactly

```text
x_(t+1) = x_t - eta g_t.
```

This exposes the real distinction:

- **Euclidean descent:** fixed Riesz map, translation-invariant squared
  distance, symmetric geometry, exact global identification of tangent
  spaces.
- **Bregman descent:** base-point-dependent divergence, generally nonlinear
  primal/dual map, asymmetric finite displacement, no Euclidean triangle law.

A Bregman Hessian geometry is *dually flat*: primal and dual affine
connections each have flat coordinates.  That does not restore one canonical
isometric transport compatible with every object.  The Levi-Civita metric
transport, primal affine transport, and dual affine transport answer different
questions.  A momentum scheme must say which object it carries and under which
connection.

## 3. Why naive acceleration separates from descent

Euclidean heavy-ball momentum uses

```text
m_t = beta m_(t-1) + g_t,
x_(t+1) = x_t - eta m_t.
```

This assumes the identity transport

```text
T_(x_(t-1) -> x_t) m_(t-1) = m_(t-1).
```

In a flat Euclidean vector space that is a coherent default.  Even there it is
not evidence that the old vector remains dynamically useful; it merely says
the vector can be compared without a coordinate contradiction.

In a general mirror geometry, several superficially similar quantities are
different:

- primal displacement `x_t-x_(t-1)`;
- dual displacement `grad h(x_t)-grad h(x_(t-1))`;
- tangent vector under the Hessian metric;
- constraint-dual or Bregman residual state generated by a splitting method.

Adding one of them unchanged to another silently chooses a transport law.
Successful accelerated mirror methods therefore use coupled primal-dual
sequences, estimate functions, or an explicit geometric transport.  “Add
momentum” is incomplete until its state space and connection are named.

## 4. Exact decomposition of Anchor

The implementation acts independently on each parameter row; a vector is one
cell.  Let

```text
g_t  current raw gradient in one cell
u_t  = g_t / ||g_t||
r_t  carried unit signature
s_t  carried slow request
```

and define

```text
c = exp(-1) = 0.367879...
a = 1-c     = 0.632121...
```

### 4.1 The signature is a spherical low-pass state

Away from zero and antipodal degeneracy,

```text
q_t = (r_t + u_t) / ||r_t + u_t||.                (4)
```

If the angle from `r_t` to `u_t` is `theta_t`, then `q_t` is their spherical
midpoint.  Anchor advances the carried frame halfway toward the current unit
gradient rather than replacing it with the current gradient.

The orthogonal basis

```text
b_t = (u_t+r_t)/||u_t+r_t||,
e_t = (u_t-r_t)/||u_t-r_t||                       (5)
```

has a useful exact interpretation:

```text
u_t = cos(theta_t/2) b_t + sin(theta_t/2) e_t,
r_t = cos(theta_t/2) b_t - sin(theta_t/2) e_t.    (6)
```

`b_t` is agreement; `e_t` is departure.

### 4.2 Slow state is moved by the minimum rotation

Let `Q_t` be the minimum orthogonal rotation satisfying

```text
Q_t r_t = q_t.                                    (7)
```

The slow request is transported before reuse:

```text
sbar_t = Q_t s_t.                                 (8)
```

The implementation uses the closed rank-two formula.  For unit frames `r,q`,
decompose `s = xi r + z`, where `z` is orthogonal to `r`.  Then

```text
Q s = xi q + z - [<z,q>/(1+<r,q>)](r+q).          (9)
```

It preserves `||s||` and the axial component carried from `r` to `q`.  At an
antipode, `1+<r,q>` collapses and no unique shortest rotation exists; Anchor
resets that cell rather than choosing an arbitrary plane.

This is spherical Euclidean transport.  It is not the transport generated by
a Bregman potential.

### 4.3 The lead--lag core is an exact two-timescale filter

Anchor forms the fast request and next slow request as

```text
f_t       = a sbar_t + c g_t,
s_(t+1)   = a sbar_t + c f_t.                     (10)
```

Substitution reveals the slow pole:

```text
s_(t+1)
  = (1-c^2) sbar_t + c^2 g_t
  = 0.864665 sbar_t + 0.135335 g_t.               (11)
```

The fast readout gives the new gradient weight `c`; the slow state gives it
weight `c^2`.  Their responsiveness ratio is exactly

```text
c / c^2 = e.                                      (12)
```

In a constant frame with constant gradient, both states converge to the raw
gradient and the DC gain is one.  During a change, `f` leads `s`.  This is the
surviving deterministic core of Wolf: not ordinary heavy-ball, and not an
extrapolated parameter point, but a slow EMA with a faster readout.

### 4.4 Restraint is a rank-one contraction of departure

Define the half-angle turn energy

```text
tau_t = (1-<u_t,r_t>)/2
      = sin^2(theta_t/2)
      = ||u_t-r_t||^2/4.                          (13)
```

With restraint strength `alpha in [0,1]`, Anchor applies

```text
R_t = I - alpha tau_t e_t e_t^T,
v_t = R_t f_t.                                    (14)
```

In the agreement/departure basis,

```text
f_t = f_b b_t + f_e e_t,
v_t = f_b b_t + (1-alpha tau_t) f_e e_t.          (15)
```

The agreement component passes unchanged.  Only the component crossing the
turn is reduced.  The spectrum of `R_t` is

```text
1                     on e_t^perp,
1-alpha tau_t         on e_t.                     (16)
```

Therefore

```text
||v_t||^2
  = ||f_t||^2
    - alpha tau_t(2-alpha tau_t)<e_t,f_t>^2
  <= ||f_t||^2.                                   (17)
```

This invariant must be stated precisely: restraint never amplifies the *fast
proposal*.  Anchor as a whole is not restraint-only, because `sbar_t` is a
historical momentum request and `||f_t||` can exceed `||g_t||`.  The original
BFFT-inspired no-new-force experiment obeyed `||g_eff||<=||g||`; standing
Anchor relaxes that stronger rule when momentum is introduced.

### 4.5 Restraint can sacrifice immediate descent

Because

```text
<e_t,u_t> = sin(theta_t/2) = sqrt(tau_t),          (18)
```

the alignment with the current raw gradient changes by

```text
<g_t,v_t>
  = <g_t,f_t>
    - alpha ||g_t|| tau_t^(3/2) <e_t,f_t>.         (19)
```

If the fast proposal has positive departure component, restraint may reduce
the instantaneous first-order decrease predicted by the current gradient.
That is not an algebraic defect.  It is the formal version of “slow down bad;
good speeds up on its own”: Anchor accepts less immediate descent in a
component hypothesized to produce later reversal.

For an `L`-smooth loss, the ordinary descent lemma gives

```text
F(x_t-eta lambda_t v_t) - F(x_t)
 <= -eta lambda_t <g_t,v_t>
    + (L/2) eta^2 lambda_t^2 ||v_t||^2.            (20)
```

The trust controller bounds the second term through the step norm, but it
cannot guarantee `<g_t,v_t> > 0`.  Anchor has a local displacement guarantee,
not a universal descent guarantee.

### 4.6 The learning rate is a local certificate, not Adam scaling

For cell parameters `p_t`, define

```text
B_t = max(||p_t||,1),
h_t = min(1, delta B_t / [eta_max ||v_t||]),
delta = 0.02.                                     (21)
```

The active multiplier brakes immediately and recovers slowly:

```text
lambda_t = h_t,
    if h_t < lambda_(t-1),

lambda_t = lambda_(t-1)
           + gamma[h_t-lambda_(t-1)],
    otherwise,

gamma = 0.05.                                     (22)
```

Since `lambda_t<=h_t`,

```text
eta_max lambda_t ||v_t|| / B_t <= delta.           (23)
```

The controller has one scalar per row.  It is not a coordinatewise second
moment, and it does not infer uncertainty.  It only certifies relative
displacement and gives each row its own recovery time.

## 5. What the lead--lag core does on a flat quadratic

The transport and restraint can be separated from the temporal filter.  On a
one-dimensional quadratic with gradient `g_t=lambda x_t`, constant signature,
and inactive trust controller, let `kappa=eta lambda` and

```text
beta = 1-c^2.
```

The state recurrence is

```text
[x_(t+1)]   [1-kappa c    -eta a] [x_t]
[s_(t+1)] = [c^2 lambda      beta] [s_t].          (24)
```

Its characteristic polynomial has

```text
trace       = 1+beta-kappa c,
determinant = beta-kappa a c.                      (25)
```

The scalar Jury conditions give the positive-step stability boundary

```text
0 < kappa < 2(1+beta) / [c(1+a)]
          = 6.21... .                              (26)
```

Plain gradient descent has `0<kappa<2`.  Thus even before spherical transport,
rank-one restraint, or the trust controller, the lead--lag recurrence has a
larger scalar-quadratic stability interval.  This is one plausible source of
Anchor's high nominal learning-rate tolerance.

Equation (26) is not a global convergence theorem.  In multiple dimensions,
signatures turn; in nonconvex problems, gradients are stochastic; at a
one-dimensional sign reversal Anchor's antipodal reset and full turn restraint
change the recurrence.  It is nevertheless a clean ablation prediction:
Anchor's speed is not attributable to transport alone.

## 6. What BFFT's Meyer acceleration actually transports

The motivating Meyer decomposition alternates

```text
u_(n+1) = ROF(f-v_n, lambda),
v_(n+1) = P_(G_mu)(f-u_(n+1)).                     (27)
```

The second variable is a bounded flux/transport object: `v=div q` with
pointwise-bounded `q`.  Each ROF subproblem is solved by Split Bregman, whose
`d,b` fields are exact shrinkage and constraint-dual state, not an empirical
gradient-direction memory.

The BFFT work obtained its large iteration reduction through a combination of

- carrying the inner Bregman fields instead of restarting them;
- interleaving one warm inner sweep per outer block;
- exposing the two-projector structure;
- in the reduced formulation, recognizing (27) as proximal gradient on the
  static composite

```text
J(u) + (lambda/2) dist^2(f-u,G_mu);                (28)
```

- recomputing transport at the extrapolated point when applying sound FISTA,
  rather than carrying a stale two-block primal velocity;
- accepting analytical Hodge closure only when the exact ROF objective falls.

The solver acceleration and Anchor therefore share an ancestry but not a
mathematical identity.

| Question | BFFT Meyer acceleration | Anchor |
|---|---|---|
| State | Split-Bregman dual/shrinkage fields or bounded flux | gradient signature, slow request, LR multiplier |
| State meaning | exact constraint and operator state | empirical temporal request |
| Frame | subproblem/operator and spatial flux frame | rowwise unit-gradient frame |
| Transport | warm operator state; flux/divergence geometry; re-solve at changed point | minimum spherical rotation inferred from gradients |
| Restraint | proximal projection, bounded G-ball, objective acceptance | rank-one contraction of departure component |
| Target | provably same convex objective/fixed point | changed trajectory; nonconvex implicit bias may change |
| Acceleration | eliminate restart and nested-solve debt | tolerate larger useful motion and reduce correction debt |
| Strong invariant | exact proximal/constraint structure | `||R_t f_t||<=||f_t||` and local 2% displacement |

The closest common statement is:

> Preserve semantically valid state, re-express it before reuse, and constrain
> the part of a proposal that does not survive the new frame.

## 7. Is the difference caused by dot products?

Partly, but “dot product versus no dot product” is too coarse.

Anchor's mechanism is explicitly Euclidean.  Every decisive object is built
from the rowwise coordinate inner product:

- normalization `g/||g||`;
- angle `<u,r>`;
- midpoint on the Euclidean unit sphere;
- orthogonal minimum rotation;
- projection `e e^T`;
- parameter and proposal norms in the trust certificate.

Consequently Anchor is equivariant under an orthogonal change of coordinates
inside a cell.  If `U^T U=I`, then

```text
g -> Ug, r -> Ur, s -> Us
```

preserves every dot product and norm, while

```text
Q -> U Q U^T,
R -> U R U^T.                                     (29)
```

The entire update rotates by `U`.

Anchor is not equivariant under a general scaling or shear.  Those transforms
change angles, normalized directions, row norms, and therefore the inferred
turn and restraint.  A genuine mirror or natural-gradient construction would
define those operations using a problem metric and transform that metric with
the coordinates.

The productive diagnosis is therefore:

1. backprop supplies a covector at one point;
2. SGD uses a fixed dot product to turn it into a globally reusable vector;
3. Anchor keeps that fixed dot product but observes temporal frame rotation;
4. Bregman descent changes the covector-to-step map itself;
5. BFFT's split solver additionally carries objective-derived dual state whose
   semantics are stronger than either gradient history.

## 8. Research consequences

### 8.1 Separate the three mechanisms

Standing Anchor contains three independent sources of behavior:

1. lead--lag filtering with the scalar stability interval in (26);
2. spherical transport of the slow state;
3. departure restraint plus the local trust controller.

The next formal assay should cross these factors rather than compare only full
optimizers:

| Arm | Lead--lag | State transport | Departure restraint | Trust controller |
|---|---:|---:|---:|---:|
| A | yes | no | no | no |
| B | yes | yes | no | no |
| C | yes | no | yes | no |
| D | yes | yes | yes | no |
| E | yes | yes | yes | yes |

This identifies whether the 75-step launch comes from the filter's stability
region, the transported frame, or restraint's settling behavior.  The existing
MLP result already suggests: transport produces the launch; output restraint
produces sustained residence.

### 8.2 Measure correction debt directly

The removed energy is observable:

```text
C_t = alpha tau_t(2-alpha tau_t)<e_t,f_t>^2.       (30)
```

Equation (17) shows that `C_t=||f_t||^2-||v_t||^2`.  Log it alongside future
gradient reversal, loss overshoot, and time to sustained threshold.  The core
hypothesis becomes falsifiable:

> Large removed departure energy predicts motion that an unrestrained
> optimizer later reverses.

If `C_t` does not predict reversal or settling debt, the transport story is
wrong even if Anchor remains an effective tuned optimizer.

### 8.3 Test the dot-product hypothesis by reparameterization

For the same function and initialization, compare:

- orthogonal within-row parameter rotations;
- diagonal rescalings spanning several orders of magnitude;
- an equivalent whitened parameterization.

Equation (29) predicts exact or numerical equivalence under the first and
failure of equivalence under the latter two.  This directly measures how much
Anchor depends on Euclidean coordinate geometry.

### 8.4 Construct a true mirror-Anchor only after the ablation

A genuine Bregman version would define

```text
y_t = grad h(x_t),
```

carry its request in a named primal or dual tangent bundle using the metric
`Hess h`, perform agreement and departure projections in that metric, update
`y`, and recover `x` through `grad h*`.  It would need to specify whether the
slow state is a primal vector, dual covector, or dual-coordinate displacement.

Without that declaration, “Bregman Anchor” would merely replace one ambiguous
transport with another.  The Euclidean decomposition above should be fully
falsified first.

## Conclusion

The BFFT result and Anchor are connected by a rule about state, not by a shared
objective.  BFFT carries exact dual transport state through changing convex
subproblems and removes solver restart debt.  Anchor carries empirical
directional momentum through changing gradient signatures and tries to remove
future correction debt.

Euclidean descent differs from Bregman descent because its quadratic potential
makes covector conversion, finite displacement, and cross-point transport all
look like the same subtraction.  Backpropagation's current-point gradient is
not the cause; it is the common input that makes the hidden optimizer geometry
easy to overlook.

Anchor exploits that overlook without leaving Euclidean space.  It adds an
observed moving frame, a precisely analyzable lead--lag filter, a rank-one
departure contraction, and a local displacement certificate.  Its next
research phase should determine which of those four objects actually buys the
iteration reduction and whether removed departure energy truly measures the
motion that another optimizer later has to undo.
