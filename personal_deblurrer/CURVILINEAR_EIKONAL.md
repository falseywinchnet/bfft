# Curvilinear Eikonal exposure transport

## Why the tangent atlas was not enough

A straight displacement path is a one-parameter translation group. This is
why differentiating its exposure integral produces an exact endpoint
recurrence. A curved path does not have that closure:

```text
gamma(t + h) - gamma(t)
```

depends on `t`. No global image-coordinate warp can honestly turn arbitrary
curved translation blur into the same box recurrence used for a line. The
former three-tangent atlas was therefore only a local approximation.

The replacement is a lifted exposure tube. It keeps every positive PSF atom as
an actual transported state, but gives those states a continuous coordinate
along a fitted curve.

## Eikonal path coordinate and Jacobian

Let `e` be the principal displacement direction and `n` its normal. Every PSF
atom `d_j` has coordinates `(s_j,q_j)`. A weighted quadratic skeleton is fit:

```text
q_hat(s) = a (s^2 - E[s^2]) + b s + c.
```

Its arc coordinate and Jacobian are

```text
tau(s) = integral_0^s J(r) dr,
J(s)   = sqrt(1 + q_hat'(s)^2).
```

`tau` has unit gradient along the fitted path, so it is the one-dimensional
Eikonal coordinate of the exposure tube. The residual
`q_j - q_hat(s_j)` is retained as transverse model discrepancy. Nothing is
projected onto the skeleton: the original atom location and positive weight
remain the image-formation operator exactly.

## Exact reflected forward and adjoint

For the discrete reflection map `R`, the lifted forward operator is

```text
(A x)(p) = sum_j w_j x(R(p - d_j)).
```

Its adjoint is not guessed by applying another reflect convolution. It scatters
each output value back through the identical map:

```text
(A* z)(q) = sum_(p,j : R(p-d_j)=q) w_j z(p).
```

Consequently the implementation satisfies

```text
<A x, z> = <x, A* z>
```

to floating-point precision, including at image boundaries. Reflection can
send several output sites to one source site, so `A* 1` is not assumed to be
one. The positive update explicitly divides by this coverage:

```text
x_(k+1) = x_k [A*(y / A x_k)] / [A* 1].
```

This preserves a constant image and gives the boundary a real conservation
ledger.

## Reconstruction and support

The exact exposure solve starts from the existing stable
positive/Fourier-gated basin. It spends at most

```text
min(32, ceil(maximum positive passes / 2))
```

additional reflected characteristic passes. A robust residual-discrepancy law
usually stops sooner when the remaining residual is noise-consistent. The
resulting correction is projected once more by the original path OTF coverage
law. Near-null coefficients cannot be reintroduced merely because the spatial
operator is exact. This operator now applies to every nonidentity centered
positive measure; curve geometry changes diagnostics and conditioning, not
solver identity.

## Endpoint seed uncertainty

The two fitted path endpoints define extreme latent seed gauges: the entire
exposure could initially be attributed to either endpoint before consistency
transport. Each seed is propagated through the same normalized forward/adjoint
operator. Their terminal weights use only forward residual and spent descent
action.

The diagnostic uncertainty contains two terms:

1. disagreement between the transported endpoint branches; and
2. their common displacement from the accepted positive-basin refinement.

The second term matters because two poor endpoint branches can agree with each
other. Omitting it would make the uncertainty falsely narrow. This remains a
sensitivity field, not a calibrated Bayesian interval.

## Trust boundary

Known synthetic operators may activate the exact curve refinement. A
single-image phase estimate may not: estimated path geometry is insufficient
authority for a high-gain inverse. Projective motion, depth-varying transport,
rolling shutter, occlusion, and arbitrary non-quadratic exposure tubes remain
outside the present exactness claim. Their next operator must retain this same
forward/adjoint and boundary-normalization contract.
