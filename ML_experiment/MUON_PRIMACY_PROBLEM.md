# A Muon primacy witness: anisotropic operator recovery

Date: 2026-08-29  
Status: exact population and rank-deficient block theorems implemented; numerical certificate reproduced on the M4 Mini

## Result

We now have a synthetic problem on which Muon's advantage is not a favorable
curve, a borrowed benchmark, or a learning-rate accident.  It is an algebraic
property of the update.

The problem asks a matrix `W` to recover a dense orthogonal operator `Q` from
anisotropic inputs.  Its gradient contains exactly the desired operator
multiplied by an irrelevant positive spectrum.  Muon's polar map removes that
spectrum while preserving the complete left/right operator frame.  SGD keeps
the spectrum and must work through its condition number.  Adam removes entry
magnitudes coordinate by coordinate, which generally destroys rather than
recovers the operator frame.

There are two exact statements.

1. With the population gradient, exact Muon reaches the optimum in **one
   update**, independently of the covariance condition number.
2. When the covariance is revealed as `m` mutually orthogonal rank-deficient
   blocks, exact Muon reaches the optimum in exactly **`m` updates**, one
   complete sweep.  The singular values inside every block can be arbitrarily
   anisotropic.

For the implemented `32 x 32` instance with condition number `10^4` and four
rank-8 blocks, Muon reaches relative loss below `10^-8` and relative operator
error below `10^-2` on update 4.  Neither tuned SGD nor tuned Adam reaches
either full-recovery criterion in 400 updates.

![Muon primacy certificate and curves](muon_primacy.png)

## 1. The population problem

Let

```text
Q in R^(d x d),       Q^T Q = I,
Sigma = R Lambda R^T, Sigma positive definite,
Lambda = diag(lambda_1,...,lambda_d),
kappa = lambda_max / lambda_min.
```

Define

```text
F(W) = 1/2 tr((W-Q) Sigma (W-Q)^T).               (1)
```

This is ordinary linear regression in population form.  If inputs have
covariance `Sigma` and labels are `y=Qx`, then

```text
F(W) = 1/2 E ||Wx-y||_2^2.                        (2)
```

The unique solution is `W*=Q`, and

```text
grad F(W) = (W-Q) Sigma.                           (3)
```

The construction uses a dense Haar-like orthogonal `Q` and an independently
rotated covariance.  It is therefore not diagonal in the ambient parameter
coordinates and cannot be solved by treating entries as independent scalar
problems.

## 2. One-step Muon theorem

Initialize `W_0=0`.  Then

```text
G_0 = grad F(W_0)
    = -Q Sigma
    = -(Q R) Lambda R^T.                          (4)
```

Because `Q R` and `R` are orthogonal and every entry of `Lambda` is positive,
equation (4) is already a signed singular-value decomposition.  Its polar
factor is

```text
polar(G_0) = -(Q R) R^T = -Q.                    (5)
```

The exact Muon step with operator learning rate one is therefore

```text
W_1 = W_0 - polar(G_0) = Q = W*.                 (6)
```

This holds for every positive spectrum.  The condition number can approach
infinity while `Sigma` remains nonsingular; equation (6) does not change.

The key operation is **spectral erasure**.  The gradient's singular vectors
contain the operator decision, while its singular values contain the input
frequency with which that decision was observed.  Muon keeps the first and
quotients out the second.

### Why SGD cannot make the same step

SGD with a scalar step `eta` produces

```text
W_1^SGD = eta Q Sigma.                            (7)
```

Equality with `Q` would require `eta Sigma=I`.  No scalar `eta` can satisfy
that unless `Sigma` is isotropic.  The best scalar first step is

```text
eta* = tr(Sigma^2) / tr(Sigma^3),                 (8)

min_eta F(eta Q Sigma) / F(0)
  = 1 - tr(Sigma^2)^2 / (tr(Sigma) tr(Sigma^3)). (9)
```

For repeated stable SGD steps,

```text
W_t-Q = -Q(I-eta Sigma)^t.                       (10)
```

At the best stationary minimax scalar rate,

```text
eta = 2/(lambda_max+lambda_min),
rho = (kappa-1)/(kappa+1).                       (11)
```

Thus tight operator recovery costs order `kappa log(1/epsilon)` iterations,
whereas exact Muon costs one on this problem.

### Why Adam's first step is structurally wrong

Ignoring its small numerical epsilon, Adam's bias-corrected first direction is

```text
D_0^Adam = sign(G_0) = sign(-Q Sigma).            (12)
```

For a generic dense `Q` and independently rotated `Sigma`, this coordinate
sign matrix is not a scalar multiple of `Q`.  Adam has removed magnitudes in
the ambient coordinate chart, but the nuisance spectrum lives in the
left/right singular chart.  A scalar LR cannot repair the discarded matrix
geometry.  This is a one-step separation, not a claim that Adam can never
converge after accumulating history.

## 3. Rank-deficient spectral block revelation

The population result is intentionally surgical: it isolates the polar map.
To make the optimizer act repeatedly without abandoning the theorem, partition
the covariance eigenbasis into `m=d/b` orthogonal blocks:

```text
C_j = R_j Lambda_j R_j^T,
P_j = R_j R_j^T,
P_i P_j = 0 for i != j,
sum_j C_j = Sigma,
sum_j P_j = I.                                   (13)
```

Each `C_j` is rank `b`.  Its eigenvalues are interleaved from the full
spectrum, so each observed block is internally anisotropic rather than a
scaled projector.  At update `j`, use the legitimate batch objective

```text
F_j(W) = 1/2 tr((W-Q) C_j (W-Q)^T).               (14)
```

Assume inductively that Muon has completed the previous supports:

```text
W_(j-1) = Q sum_(i<j) P_i.                       (15)
```

Orthogonality gives

```text
G_j = (W_(j-1)-Q) C_j = -Q C_j,
polar_support(G_j) = -Q P_j.                     (16)
```

The unit step is

```text
W_j = W_(j-1) + QP_j
    = Q sum_(i<=j) P_i.                          (17)
```

After `m` batches,

```text
W_m = Q sum_j P_j = Q.                           (18)
```

This is exact finite termination under rank-deficient observations.  Muon does
not invert `C_j`, estimate its eigenvalues, or remember a second moment.  The
supported polar factor simply recognizes the complete operator action on the
subspace that is currently visible.

## 4. Numerical certificate

Configuration:

- dimension: `32 x 32`;
- dense orthogonal target and independently rotated covariance;
- covariance condition number: `10^4`;
- block stream: four mutually orthogonal rank-8 blocks, repeated for 100
  cycles;
- exact Muon LR grid: `0.3, 0.5, 1.0`;
- SGD LR grid: `0.1, 0.3, 0.7, 1.0, 1.5, 1.9`;
- Adam LR grid: `0.001, 0.003, 0.01, 0.03, 0.1`;
- selection: first crossing of relative loss `10^-8`, then terminal relative
  loss when no run crosses.

### Best possible first scalar step from the population origin

| method | best scalar | relative loss after one step |
|---|---:|---:|
| **exact Muon** | **1.000000** | **6.10e-30** |
| SGD | 1.316699 | 0.244507 |
| Adam sign limit | 0.072464 | 0.320954 |

The Muon target error is `2.71e-13`, the float64 SVD floor.  The operator norm
of its request is exactly one.  The operator norm of Adam's sign request is
`15.80`, showing how entrywise normalization can create a large matrix-level
action even though every individual coordinate appears bounded.

### Population trajectory

| method | selected LR | updates to `10^-8` loss | final loss ratio at 2,000 | final operator error |
|---|---:|---:|---:|---:|
| **exact Muon** | **1.0** | **1** | **6.25e-30** | **1.69e-13** |
| SGD | 1.9 | not reached | 5.92e-5 | 0.684 |
| Adam | 0.01 | not reached | 1.05e-5 | 0.422 |

### Rank-deficient block trajectory

| method | selected LR | updates to `10^-8` loss | updates to 1% operator error | final loss ratio at 400 | final operator error |
|---|---:|---:|---:|---:|---:|
| **exact Muon** | **1.0** | **4** | **4** | **5.02e-28** | **1.85e-13** |
| SGD | 1.9 | not reached | not reached | 2.19e-3 | 0.981 |
| Adam | 0.03 | not reached | not reached | 1.18e-3 | 1.005 |

The objective can look moderately small while the learned operator is still
almost maximally wrong in its quiet directions.  This is exactly why both loss
and operator error are reported.  The latter prevents an optimizer from
receiving credit merely for fitting the covariance's loud subspace.

## 5. Why this is a valid witness—and what it does not prove

The target is not exposed to the optimizer.  Every method receives the same
gradients, batches, initialization, and declared scalar LR search.  The block
covariances are genuine positive-semidefinite batch covariances, and they sum
exactly to the population covariance.  No optimizer-specific data or metric is
used.

The problem is nevertheless designed.  It places signal in singular vectors
and nuisance in singular values because that is exactly the distinction Muon
makes.  Its purpose is analogous to a separating example in numerical linear
algebra: demonstrate a mechanism under conditions where its claimed invariant
has an exact consequence.

It does **not** establish that Muon is universally superior, that practical
five-step Newton--Schulz Muon has exact finite termination, or that arbitrary
neural gradients factor as `-Q Sigma`.  The theorem concerns the exact polar
primitive.  Momentum, weight decay, RMS matching, finite Newton--Schulz error,
and auxiliary AdamW parameters must be added one at a time if the problem is
used as a practical optimizer benchmark.

## 6. The random-minibatch negative control

The implementation also draws ordinary Gaussian minibatches from the same
population covariance.  Their supports overlap randomly instead of revealing
an orthogonal partition.  On the 400-update screen:

| method | selected LR | updates to `10^-2` loss | final loss ratio | final operator error |
|---|---:|---:|---:|---:|
| exact Muon | 0.03 | 99 | **0.00119** | **0.912** |
| SGD | 1.0 | **73** | 0.00181 | 0.962 |
| Adam | 0.01 | 122 | 0.00195 | 1.035 |

This is not a Muon-primacy result.  Muon ends best, but SGD crosses a loose
loss threshold sooner by following the loud covariance directions.  Random
overlap also means a unit partial-isometry step can disturb directions already
estimated from earlier batches.  That failure is scientifically useful: the
block theorem isolates spectral erasure, while the random stream reintroduces
the correction debt that transport, smooth Bregman response, and restraint
are intended to manage.

## 7. What the problem teaches us about Anchor and Lepton

The witness separates two jobs that were entangled in the pinwheel results.

```text
Muon:      erase nuisance spectrum inside the current matrix request.
Transport: preserve a valid historical request as the observed frame turns.
Lepton:    retain graded confidence instead of hard-promoting every supported mode.
Restraint: stop history from reversing a newly certified local decision.
```

On the population witness, transport cannot improve the one-step answer:
there is no historical frame yet.  On the orthogonal block sweep, it is also
unnecessary: completed and live supports do not interfere.  These are controls
in which Muon's own geometry should be sufficient.

The next useful family is obtained by continuously rotating and partially
overlapping the block supports, then injecting controlled noise into their weak
singular modes.  The static polar theorem remains the known solution at zero
rotation and zero overlap.  Increasing those two quantities creates a clean
phase diagram for the three new mechanisms:

1. transported Lepton with Anchor's lead--lag recurrence;
2. Anchor restraint encoded as a matrix mirror map;
3. restrained-momentum Anchor, where momentum may propose a gambit but cannot
   reverse a live certified component.

That family will tell us whether each mechanism preserves Muon's exact
spectral advantage while paying less correction debt once the frame itself
moves.

## Reproduce

The implementation and algebraic tests are:

- `ML_experiment/muon_primacy.py`;
- `ML_experiment/test_muon_primacy.py`;
- `ML_experiment/plot_muon_primacy.py`;
- `ML_experiment/results_muon_primacy.json`.

Run on the M4 Mini from the repository root:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m unittest ML_experiment.test_muon_primacy -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.muon_primacy \
  --out /tmp/muon_primacy.json --dimension 32 --condition 10000 \
  --steps 2000 --stream-steps 400 --batch-size 8 --block-cycles 100
```
