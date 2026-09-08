# Muon, spectral descent, Bregman geometry, and restraint-shaped momentum

Date: 2026-08-29  
Status: reference implementation staged; formal bridge to Anchor research

## Result

Muon is a matrix optimizer, but it is not classical Bregman or mirror descent.
Its defining operation is an approximate polar factor of a momentum matrix.
In the exact limit, that factor is simultaneously

1. the nearest semi-orthogonal matrix in Frobenius distance;
2. a subgradient of the nuclear norm;
3. the linear minimization oracle over a spectral-norm ball; and
4. the instantaneous, no-accumulation matrix-Shampoo direction.

Those equivalences make Muon highly relevant to Anchor.  They do not make it
Bregman descent.  The distinction matters because a restraint fused with Muon
must deform the admissible matrix ball.  Multiplying a momentum matrix by a
restraint and then applying ordinary Muon can erase the restraint by restoring
the suppressed singular values.

The experiment implementation is `ML_experiment.optimizers.MuonWithAuxAdamW`.
It applies Muon to hidden two-dimensional weights and AdamW to input/output
weights, biases, gains, and other nonmatrix parameters, matching the reference
routing rule.

## 1. Exact Muon update

For a hidden weight matrix `W_t` with stochastic gradient `G_t`, form a
momentum state and optional Nesterov readout.  In the EMA convention used by
the staged implementation,

```text
M_t = beta M_(t-1) + (1-beta) G_t,
N_t = (1-beta) G_t + beta M_t.                    (1)
```

The overall scalar convention is not geometrically important to an exact
polar factor because `polar(cN)=polar(N)` for positive `c`.  It does matter to
a finite Newton--Schulz approximation, so the staged code follows the current
reference implementation literally.

If

```text
N_t = U Sigma V^T,                                (2)
```

then exact Muon uses

```text
P_t = U V^T.                                      (3)
```

For a wide matrix this has orthonormal rows; for a tall matrix it has
orthonormal columns.  Every nonzero singular value of the request is replaced
by one.  Muon therefore retains the singular subspaces but discards the
singular-value hierarchy.

The applied update is

```text
W_(t+1) = (1-eta lambda) W_t - eta rho_(m,n) P_t, (4)
```

where `rho_(m,n)` is the selected shape/RMS correction.  The staged comparator
uses the Moonshot/PyTorch `match_rms_adamw` rule

```text
rho_(m,n) = 0.2 sqrt(max(m,n)),                   (5)
```

so a learning rate tuned for AdamW remains dimensionally comparable.  The
original Keller scaling remains available as `muon_original`.

## 2. What Newton--Schulz is doing

Muon avoids an SVD.  It starts from

```text
X_0 = N_t / max(||N_t||_F, epsilon)               (6)
```

and performs five quintic iterations

```text
A_k     = X_k X_k^T,
X_(k+1) = a X_k + (b A_k + c A_k^2) X_k,          (7)

(a,b,c) = (3.4445, -4.7750, 2.0315).
```

The polynomial is intentionally aggressive.  Five iterations do not compute
an exact `UV^T`; they produce `U S' V^T` with singular values concentrated
near one.  That inexactness is part of standing Muon rather than a CPU error.

## 3. The spectral-norm variational problem

The polar factor solves

```text
P_t in argmax_D <N_t,D>_F  subject to ||D||_2 <= 1.  (8)
```

The maximum is the nuclear norm:

```text
max_(||D||_2<=1) <N_t,D>_F = ||N_t||_*,           (9)
```

and, away from rank degeneracies,

```text
grad ||N_t||_* = U V^T.                           (10)
```

Thus Muon replaces an elementwise or Frobenius steepest direction with a
matrix-level decision.  It asks which bounded linear operator has maximum
agreement with the complete momentum request.  This is why its update cannot
be decomposed into independent scalar momenta.

The constraint is on the operator norm, not on each entry.  All active
singular modes are permitted to move with the same spectral amplitude.  This
prevents one very large singular value from consuming the whole step, but it
also promotes weak and possibly noisy singular modes.

## 4. Exact relationship to Shampoo

Matrix Shampoo accumulates

```text
L_t = L_(t-1) + G_t G_t^T,
R_t = R_(t-1) + G_t^T G_t                         (11)
```

and applies

```text
D_t = L_t^(-1/4) G_t R_t^(-1/4).                 (12)
```

If accumulation is removed and the current request `N=U Sigma V^T` alone is
used, then on its support

```text
(N N^T)^(-1/4) N (N^T N)^(-1/4)
    = U Sigma^(-1/2) Sigma Sigma^(-1/2) V^T
    = U V^T.                                      (13)
```

So exact Muon is instantaneous Shampoo applied to the momentum request.  The
important differences are:

- Shampoo accumulates left and right second-moment geometry through time;
- Muon accumulates only first-moment momentum, then rebuilds its matrix shape
  from that single carried request;
- Shampoo retains a spectrum-dependent preconditioner;
- exact Muon flattens every supported singular value to one;
- standing Muon uses a cheap inexact Newton--Schulz polar factor.

Calling Muon “a form of Shampoo” is therefore mathematically defensible only
with the no-accumulation qualifier.

## 5. Does Muon use Bregman descent?

No, not in the classical sense.

Mirror/Bregman descent chooses a differentiable convex potential `h` and
updates through

```text
W_(t+1) = argmin_W {
    eta <G_t,W> + D_h(W,W_t)
},                                                (14)

grad h(W_(t+1)) = grad h(W_t) - eta G_t.          (15)
```

Muon specifies neither a global potential `h` nor a dual-coordinate state
`grad h(W)`.  It applies a support-map to the momentum and translates the
primal weight directly.

There is a tempting convex-duality construction.  Since

```text
polar(N) in partial ||N||_*,                      (16)
```

one can choose a dual function

```text
h*(N) = rho ||N||_*.
```

But its convex conjugate is

```text
h(D) = indicator{||D||_2 <= rho}.                 (17)
```

This is an indicator of a spectral-norm ball, not a smooth strictly convex
Legendre potential.  Its ordinary Bregman divergence is not the finite,
invertible mirror geometry required by classical mirror descent.  The exact
description is a norm-constrained linear minimization oracle, or normalized
spectral steepest descent with momentum.

Muon and Bregman descent still share the broader insight that the coordinate
dot product is not the whole optimizer.  Bregman descent changes the
primal--dual map through a potential.  Muon changes the admissible shape of a
matrix displacement through an operator norm.

## 6. Why naive restraint plus Muon fails

Suppose a restraint operator suppresses uncertain momentum modes:

```text
Z_t = R_t[M_t].                                   (18)
```

If ordinary Muon is then applied,

```text
D_t = polar(Z_t),                                 (19)
```

every nonzero singular value of `Z_t` is pushed back toward one.  A restraint
encoded only as reduced magnitude is therefore undone.  The order

```text
restrain -> ordinary polar
```

is structurally wrong unless restraint deletes a mode exactly or rotates the
singular subspaces.

The opposite order,

```text
ordinary polar -> restrain,
```

preserves attenuation but loses Muon's defining spectral optimality: the
result is no longer the optimizer of (8) for its stated ball.

The correct fusion must put restraint into the constraint itself.

## 7. Restraint as a deformable matrix ball

Let the current gradient be `G_t` and carried momentum be `M_t`.  Define a
positive matrix-space gauge `A_t` from their agreement, disagreement, and the
confidence of that observation.  The update should solve

```text
D_t in argmin_D <M_t,D>_F
      subject to ||D||_(A_t) <= rho.              (20)
```

For a two-sided positive-definite gauge

```text
||D||_(A_t,B_t) = ||A_t^(1/2) D B_t^(1/2)||_2,
```

the linear oracle is explicit:

```text
D_t = -rho A_t^(-1/2)
      polar(A_t^(-1/2) M_t B_t^(-1/2))
      B_t^(-1/2).                                 (21)
```

Ordinary exact Muon is the special case `A_t=I`, `B_t=I`.  This equation
preserves the restraint because the polar factor is computed *inside* the
deformed coordinates and then mapped back through the same gauge.

When the current evidence is certain that momentum reverses a live decision,
the ball contracts in that matrix mode.  When evidence is uncertain, the ball
locally expands toward Muon's spectral ball and lets momentum price the
gambit.  Restraint is then the shape of the admissible manifold rather than a
scalar brake applied to its output.

A concrete non-elementwise observation begins with polar signatures

```text
U_t = polar(G_t),
V_t = polar(M_t).                                 (22)
```

The symmetric operators

```text
C_in  = sym(U_t^T V_t),
C_out = sym(U_t V_t^T)                            (23)
```

measure agreement on the input and output action spaces.  Their negative
eigenspaces are genuine matrix-level reversals: momentum and the current
gradient send a shared mode in opposing directions.  Eigenvalues near zero
are uncertainty, not proof of reversal.

Soft restraint factors built from these spectra can define a two-sided gauge

```text
||D||_(A_t,B_t) = ||A_t^(1/2) D B_t^(1/2)||_2.    (24)
```

Equation (20) under (23) has a generalized polar solution.  This is the
natural candidate for a restraint-shaped Muon/Anchor fusion: momentum remains
matrix-valued, but its authority is bounded in the action spaces where it
contradicts current evidence.

This proposal should not yet be promoted to standing Anchor.  First measure
whether the negative agreement eigenspaces predict later correction, and
whether the uncertainty band is stable under minibatch noise.  If they do,
the metric deformation has an observed target.  If they do not, a generalized
polar will merely add expensive geometry to a tuned optimizer.

## 8. Staged comparison

The existing battery runner accepts Muon as an optimizer arm.  The intended
four-way invocation is

```sh
python3 -m ML_experiment.run_benchmark \
  --optimizers adamw,sgd,anchor,muon \
  --optimizer-lrs adamw=0.003,sgd=0.03,anchor=1.0,muon=0.003 \
  ...
```

The paired analyzer now accepts the corresponding comparator set without
changing the frozen three-way report:

```sh
python3 -m ML_experiment.analyze_anchor_battery RESULTS.json \
  --optimizers adamw,sgd,anchor,muon \
  --baselines adamw,sgd,muon
```

This stages Muon as a baseline.  It does not claim that `0.003` is optimal for
these small M-layer problems.  The `match_rms_adamw` scaling makes it the
faithful same-LR starting point; iteration-frontier work can follow only after
the mechanism comparison is sound.

## Conclusion

Muon supplies exactly the missing side comparison: a momentum method whose
decision is already a whole matrix rather than a collection of independent
coordinates.  It also exposes the hard part of the proposed fusion.  A
restraint that merely shrinks singular values is incompatible with ordinary
polar orthogonalization, because Muon destroys those magnitudes by design.

The likely object is not “Anchor plus Muon.”  It is a momentum-driven linear
minimization oracle over a restraint-shaped, uncertainty-deformable operator
ball.  That is closer to a time-varying Finsler geometry than to classical
Bregman descent.

## Primary references

- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
- [PyTorch Muon implementation](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a.html)
- [Towards Understanding Orthogonalization in Muon](https://openreview.net/forum?id=06a8f68e593c58f5f8a15c4a929073efbbd86c44)
