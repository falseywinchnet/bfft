# Eikonal recovery of the midpoint-Hessian parity

Date: 2026-08-13. Source under study:
`/Users/ultimussecundai/Downloads/2608.02478v2.pdf`, especially Lemmas 5.2,
6.4, 6.7--6.9 and Algorithms 4--5.

Implementation and measurements:

- `experiments/walsh_eikonal_recovery.py`
- `tests/walsh_eikonal_recovery_test.py`
- `experiments/out/walsh_eikonal_recovery.json`
- `experiments/out/walsh_eikonal_recovery_cutoff3.json`
- `experiments/out/walsh_eikonal_n9_cutoff3.json`

## 1. The eikonal state is a partial parity cell

Let `A(y)` be one matrix-valued empirical histogram on `F_2^ell`, and let
`Ahat(theta)` be its unnormalized Walsh transform.  Resolve a set `S` of
parity coordinates and assign them the value `p`.  The descendant energy is

    E_S(p) = sum_{theta: theta_S=p} ||Ahat(theta)||_F^2.

Writing `y=(x,z)` in the resolved and unresolved coordinates, Parseval gives

    E_S(p)
      = 2^(ell-|S|) sum_z ||sum_x (-1)^(p.x) A(x,z)||_F^2.

This is an exact conservative action:

    E_S(p) = E_{S union {b}}(p,b=0) + E_{S union {b}}(p,b=1).

It therefore supports the same irreversible first-arrival pattern used by
the image eikonal code.  A cell is accepted or discarded once.  No later
descent can create more energy than its ancestor contained.

For two independent sample panels, replace the squared bucket norm by the
inner product of the two panel buckets.  The resulting action is unbiased
and removes the diagonal self-noise.  The implementation tests conservation
both for the exact histogram and for fixed split panels.

## 2. Greedy marching is not enough

A width-limited front that follows the largest descendant energy is not a
correct recovery algorithm.  In the exact finite populations, the target
Hessian is often not the largest Frobenius-energy leaf.  At cutoff 3:

| n | Walsh outputs | width needed for every tested affine coset |
|---:|---:|---:|
| 6 | 16 | 8 |
| 7 | 32 | 8 |
| 8 | 32 | 8 |
| 9 | 64 | 16 |

Choosing the next bit adaptively by maximum energy concentration sometimes
helps and sometimes hurts.  It still optimizes total energy, not the
shortest-vector certificate.  This rejects the naive claim that fast
marching alone locates `theta_star`.

## 3. Certified marching is the useful construction

Suppose the target spike has Frobenius norm at least `tau`.  Every ancestor
of its leaf satisfies `E_S(p) >= tau^2`.  We may therefore discard exactly
the cells below that threshold and retain every other cell.  Unlike a beam,
this cannot lose the target when its action estimates are valid upper
bounds.

The cutoff-3 populations have boundary mass below 0.0033.  Their maximum
certified frontier widths were:

| n | N | threshold `tau^2` | `0.5 tau^2` | `0.25 tau^2` |
|---:|---:|---:|---:|---:|
| 6 | 16 | 8 | 16 | 16 |
| 7 | 32 | 8 | 15 | 26 |
| 8 | 32 | 16 | 31 | 32 |
| 9 | 64 | 16 | 31 | 44 |

The exact-threshold column is encouraging but not an asymptotic claim.  The
weaker thresholds show that a constant-factor loss in the certificate can
make the front dense.  Lemmas 6.4 and 6.9 are relevant because their target
is a rank-one spike with only `O(1/n)` relative operator error; the intended
asymptotic threshold approaches the first column rather than a fixed half.

## 4. A population-width theorem is already in the paper

Lemma 5.2 gives, with `g(r)=0.5 log2(r/t0)`,

    sum_u ||A_{r,d}(u)||_F^2
      <= mu_{r,d}^2 2^((kappa(r)+o(1))n),

    kappa(r) = 2r - min(g(r), 2g(r)).

Consequently, there are at most `2^((kappa(r)+o(1))n)` parity classes with
Frobenius energy at least `mu^2`.  For `h=0`, there is no affine alias sum:
the target class supplied by Lemma 6.4 is one of these heavy classes.  This
is the asymptotic certified-frontier bound that the eikonal formulation
needed.

The important distinction is that this is a bound on the number of accepted
leaves.  It does not by itself prove that their prefix actions can be
estimated and traversed in time linear in that number.

## 5. Rebalanced parameters point to the 0.5 floor

Set `h=0` and let the source width `R` approach `t0` from above.  There is no
need to pay for random affine-coset collection.  Choosing a small strict
margin

    r = 0.2240735...,  R = t0(1+o(1))

gives the diagnostics

    kappa(r) = 0.495,
    iota(r,R) + 2r = 0.4489...,
    2r = 0.4482....

Thus source-sample generation, histogram construction, and the retained
sample population are all below `2^(0.5n)`, while the certified heavy
population is below `2^(0.495n)`.  The existing preprocessing DGS/BDD cost
`2^(0.5n+o(n))` would dominate if heavy matrix-Walsh recovery can be made
near-linear in the certified population.

This is not a proof of a `2^(0.5n)` SVP algorithm.  It is a concrete
parameter regime in which exactly one recovery lemma would imply it.

## 6. Hash access does not give a useful Goldreich--Levin oracle

The empirical histogram can indeed be built as a hash table:

    A(y) = sum_{i: V_P(X_i)=y} w(X_i) traceless(X_i X_i^T) / M.

An arbitrary `A(y)` query is then an expected-constant-time lookup.  That fact
does **not** repair the access model.  Goldreich--Levin uses the normalized
Fourier coefficient of a bounded function.  If the domain has `2^(ell n)`
points, the histogram has `M=2^(m n)` occupied samples, and the unnormalized
target coefficient is `2^(-a n)`, then even the optimistic no-collision
normalization gives

    gamma <= 2^(-(a + max(ell-m, 0))n + o(n)).

A `gamma^-2` search therefore costs

    2^((2a + 2 max(ell-m, 0) + o(1))n).

For the `h=0` candidate this is about `2^(1.597n)`, not `2^(0.495n)`.
For the paper's parameters, where `ell` and `m` are balanced, it is still
about `2^(1.295n)` because the importance attenuation has exponent

    a = r + (1/2) log2(R/r) = 0.64729...

This also explains why the usual sparse-WHT theorems do not apply merely
because a zero-valued hash lookup is available.  Their bounded query signal
is dense enough that a normalized heavy coefficient remains heavy; ours is
an empirical random-example table whose nonzero density is exponentially
small in the promising `h=0` regime.

The executable calculation is in
`experiments/walsh_recovery_theorem_ledger.py`.

## 7. Quadratic eikonal energy has its own variance barrier

One might avoid the linear correlation search by hashing coefficients into a
prefix cell and estimating

    C_p = sum_beta vec(Ahat(p,beta)) vec(Ahat(p,beta))^T.

A random `d`-bit prefix isolates a target from a total spectral energy
`2^(kappa n) mu^2` once `d > kappa n`.  This is the correct population
eikonal picture.  But a split-panel collision estimator has a degenerate
noise term.  If `q=ell-d` bits remain unresolved and each individual Walsh
coefficient is estimated at target-scale variance, the standard deviation
of the cell tensor contains a factor `2^(q/2)`.  Removing it requires roughly
`2^(q/2)` more samples.  At `q` near `n/2`, that pushes the sampling exponent
well above `1/2`.

Thus neither ordinary Goldreich--Levin nor a naive quadratic prefix action is
the missing theorem.  The former pays occupancy; the latter pays unresolved
cell noise.

## 8. The cleaner theorem is target-width random-coset transport

There is a route that keeps the paper's recovery proof and its complete WHT.
The expensive endpoint importance sampling is needed because the paper can
efficiently sample the random affine dual-lattice coset only at a wide source
parameter `R`, where the coset sublattice is above smoothing.  It then changes
the width to `r` by scalar importance weights.  The `iota(r,R)` term and the
attenuation `(r/R)^(n/2)` are the price of that one-shot change of measure.

The strong theorem to seek is:

> **Random half-coset discrete-Gaussian transport.** Fix `r>t0` and
> `h=(1/2-o(1))n`.  For uniform `P in GL_n(F_2)` and uniform `j in F_2^h`,
> let `Lambda_j` be the affine parity coset from Section 6.2.  After
> `2^(n/2+o(n))` preprocessing, output
> `M=2^((2r+o(1))n)` samples jointly within `exp(-Omega(n^2))` statistical
> distance of `D_(Lambda_j,xi_r)^M`, in
> `(2^(n/2)+M)2^o(n)` time and `2^(n/2+o(n))` space, except with
> `2^(-Omega(n))` probability over `P,j`.

The exact independence and statistical-distance conclusion is stronger than
recovery needs.  A leaner, and possibly more provable, version is:

> **Transported Walsh-Hessian moment theorem.** In the same time and space,
> construct the matrix histogram whose complete `ell`-bit WHT simultaneously
> approximates every target-width matrix `G_(r,d,j)(theta)` to operator error
> `mu_(r,d)/poly(n)`, with the same success probability needed by Lemma 6.9.

The lean theorem permits dependent particles, signed control populations,
and cross-scale martingale corrections.  It asks only for the observable that
the recovery proof consumes.

## 9. Why either transport theorem gives the `1/2` constant

Take a fixed small `epsilon>0` and set

    r = t0 + epsilon,
    chi = 1/2 - epsilon,
    ell/n = 1/2 + epsilon.

Direct target-width samples have no endpoint importance penalty:

    R = r,  iota(r,r) = 0,  w = 1,
    M = 2^((2r+o(1))n) = 2^((0.46294+2epsilon+o(1))n).

The paper's target isolation condition is still strict:

    chi - 1 - min(g(r),2g(r)) + 2r < 0

for small `epsilon`, and `chi < 1/2 + g(r)` also holds.  The complete WHT and
all BDD verifications cost

    2^(ell+o(n)) = 2^((1/2+epsilon+o(1))n).

Consequently the preprocessing and WHT dominate, and letting `epsilon` tend
to zero gives `2^(n/2+o(n))` time and space.  No sparse-WHT recovery theorem
is required.

This identifies the mathematical core suggested by the eikonal analogy:
construct a causal heat-flow/transport on the fixed affine lattice coset that
moves the wide Gaussian population inward while rejuvenating support, rather
than multiplying endpoint likelihood ratios.  A proof must show either a
uniform mixing/spectral-gap statement along that flow or a direct martingale
concentration theorem for the Hessian observable.  Reweighting the same
particles through many small temperature steps is insufficient by itself;
without a genuine transport or mutation kernel, the endpoint effective
sample-size loss reappears unchanged.
