# Formal analysis of projected-signature SGD

## Local geometry

Work in one local parameter cell. Let the current gradient be `g = m u`,
where `m = ||g||` and `||u|| = 1`. Let `r` be the normalized carried Bregman
signature. For the nondegenerate case `u != -r`, write

```text
phi = angle(u, r),       a = phi / 2,
tau = (1 - <u,r>) / 2 = sin(a)^2,
d = u - r,              b = u + r.
```

The departure and compatible bisector are orthogonal:

```text
<d,b> = ||u||^2 - ||r||^2 = 0.
```

Moreover,

```text
P_d u = (<u,d> / ||d||^2) d = d / 2.
```

The family implemented by the shape arm is therefore

```text
v(k) = g - k P_d g = m (u - k d/2),       0 <= k <= 1.
```

This is a linear restraint of the current request. The historical signature
chooses a subspace but is never added as a force.

## Exact invariants

In the orthonormal bisector/departure frame,

```text
u    = cos(a) b_hat + sin(a) d_hat,
v/m  = cos(a) b_hat + (1-k) sin(a) d_hat.
```

Consequently,

```text
<g,v> / m^2 = 1 - k tau,
||v||^2 / m^2 = 1 - k(2-k) tau <= 1.
```

Thus the update is norm-nonincreasing and remains a descent direction for
the current objective except at the intentionally frozen exact-reversal
endpoint. Constant direction has `tau=0` and is exactly ordinary SGD.

For an `L`-smooth objective, the descent lemma gives

```text
f(x - eta v) <= f(x)
  - eta <g,v> + (L eta^2 / 2) ||v||^2.
```

Relative to the usual `2/L` sufficient bound for gradient descent, the local
admissible-step factor is

```text
R(k,tau) = (1-k tau) / (1-k(2-k) tau),
R - 1 = k(1-k) tau / (1-k(2-k) tau) >= 0.
```

Partial shape restraint can therefore enlarge the sufficient local step
region. Hard projection `k=1` has `R=1`; it removes motion but gains no such
allowance. This matches the measured ordering.

## Why the energy gate survives

The live restraint uses `k=tau`. Its correction magnitude is

```text
||g-v|| = m k sqrt(tau) = m sin(a)^3.
```

Useful coherent turning is disturbed only at cubic order, while an exact
reversal still receives full restraint. Its sufficient-step factor is

```text
R(tau,tau) = (1+tau) / (1+tau-tau^2),
```

which approaches `2` at reversal.

The ablations change the small-angle order:

| gate | k | correction near a=0 | 16D step to 1e-4 |
|---|---:|---:|---:|
| hard | 1 | O(a) | 4,800 |
| amplitude | sqrt(tau) | O(a^2) | 4,000 |
| **energy** | **tau** | **O(a^3)** | **2,900** |
| quartic | tau^2 | O(a^5) | unstable / did not reach |

The energy law is the observed boundary: stronger gates retard legitimate
valley motion; weaker high-order gating does not arrest the wall cycle.
Half-strength energy gating also fails because it gives up full restraint at
reversal.

## Fused and cycle variants

Repeating the same restraint `n` times has the closed form

```text
k_n = 1 - (1-tau)^n,
```

because the departure projector is idempotent. Two and four fused cells took
4,000 and 4,400 transitions respectively, versus 2,900 for one. Unlike the
Meyer inner/outer fusion, repeated action on the same SGD request is redundant
damping; the moving target already arrives at the next optimizer transition.

A two-phase signature observer was also rejected. Rosenbrock's wall motion is
not locked to optimizer-step parity, so parity states alias rather than reveal
the cycle. It did not reach the target.

## What is actually transported

The useful state is best interpreted as a moving normal frame, not an averaged
gradient. With old signature `r` and current request `u`, the next signature

```text
c = (u+r) / ||u+r||
```

is the spherical geodesic midpoint. The live departure `d=u-r` satisfies
`<c,d>=0`, so it is already a tangent vector at the new base. This is why the
one-step construction works without an explicit transport matrix: the chord
is naturally seated in the correct new tangent space.

Blindly retaining earlier departures in fixed coordinates is invalid. On the
16D oracle, untransported normal ranks 2 and 4 took 4,700 and 8,200 transitions
respectively, versus 2,900 for rank 1. Old normals had become stale constraints
on useful tangent motion.

Transporting an old tangent `z` from `r` to `c` along the shortest spherical
geodesic uses

```text
z_plus = z - (<z,c> / (1+<r,c>)) (r+c).
```

After this correction, rank 2 becomes exactly neutral on both assays: 2,900
transitions on Rosenbrock and 2,425 on the MLP, identical to rank 1.

This neutrality is algebraic. Both `u` and `r` lie in `span{c,d}`. Any
transported historical tangent, after orthogonalization against `d`, is
orthogonal to `c` and `d`, hence orthogonal to `u`. It has zero projection of
the only current request available to the optimizer. Therefore rank one is
complete for any method that sees one aggregated gradient vector per cell and
only restrains projections of that vector. More correctly transported history
cannot improve such an update.

The implication is structural for restraint-only SGD: a larger history of
aggregated gradients cannot add another active normal.  The present study does
not use microbatch or per-example channels; those would change the information
available to the optimizer rather than test transport of the ordinary SGD
trajectory.

## Anchor as transported lead--lag momentum

The useful core of the original Wolf experiment is not its random multiplier,
sign gate, or parameter-decay fallback.  Removing those leaves two coupled
requests.  With `c=exp(-1)` and `a=1-c`,

```text
f_t     = a s_t + c g_t,
s_{t+1} = a s_t + c f_t
          = (1-c^2) s_t + c^2 g_t.
```

Thus `s` is an EMA with pole `1-exp(-2)`, while the applied request `f` is a
faster readout that leads the slow state toward the current gradient.  It is a
two-timescale momentum cell, not ordinary heavy-ball accumulation.  For a
constant gradient both states converge to that gradient without steady-state
amplification.

The literal transport composition uses the same consecutive-gradient geometry
as signature SGD.  Let `r` be the old unit signature and `u` the current unit
gradient.  Define

```text
q   = (u+r) / ||u+r||,
d   = u-r,
tau = (1-<u,r>)/2.
```

First parallel-transport the slow state from `r` to `q`, then form Anchor's fast
and next-slow states.  Restrain only the applied fast request:

```text
s_bar = T[r -> q](s_t),
f_t   = a s_bar + c g_t,
s_next = a s_bar + c f_t,
v_t   = f_t - tau P_d f_t.
```

The chord `d` is tangent at `q`, so the transported state and live restraint
now inhabit the same frame.  Since `P_d` is an orthogonal projector,

```text
||v_t||^2 = ||f_t||^2 - tau(2-tau)||P_d f_t||^2 <= ||f_t||^2.
```

This is a no-amplification statement relative to Anchor's proposal.  Momentum
means it is not, in general, a descent-direction proof relative to the current
minibatch gradient.

Restraining the stored slow state as well is not valid.  The slow variable is
memory to be transported into the next frame, not a present parameter request;
projecting it with the current tangent destroys the recurrence.  Empirically it
diverged at the high learning rate, while output-only restraint remained
finite and produced the large iteration compression.

`ANCHOR.md` derives the subsequent dynamic trust controller.  It retains
`lr=1.0` as a hard ceiling while certifying a maximum local relative
displacement, with immediate braking and slow recovery.

## Capacity and coordinate controls

A closer Split-Bregman analogue retained the projected state's ball occupancy
and used

```text
k = ||q|| ||u-q||^2 / 4.
```

It was neutral on deterministic Rosenbrock and slightly slower on the MLP
(2,500 versus 2,425 transitions). Capacity is therefore not the source of the
present gain; angular moving-frame transport is.

Using one orthogonally equivariant tensor-wide cell instead of weight-matrix
rows produced the same MLP hitting times (2,425 restraint, 1,925 active) to the
25-transition measurement resolution. The result is not dependent on the
chosen neuron-row coordinates in this assay.

## Active shape transport

`mode="rotate"` renormalizes each nondegenerate local shaped cell back to its
original gradient norm. It is no longer restraint-only: it reallocates removed
transverse norm into the compatible direction, although it still uses no
momentum and never exceeds the local raw norm. It diverges on the deterministic
Rosenbrock stress case and is therefore not a replacement for the restrained
operator.

On the fixed stochastic MLP, however, it reaches plain SGD's attained test
floor in 1,925 transitions instead of 2,999 (36% fewer). The restraint-only
shape arm takes 2,425 (19% fewer); scalar damping does not reach the floor.
This active branch is promising but currently conditional, not a general
optimizer claim.

## Scope

These are local, per-transition statements. For minibatch training, `g` is the
batch gradient, so descent alignment is with that batch objective rather than
the population loss. The analysis proves the operator's geometry and a smooth
descent bound; it does not prove global acceleration on nonconvex or stochastic
objectives.
