# Midpoint-Hessian Walsh noise audit

Date: 2026-08-13. Source under study:
`/Users/ultimussecundai/Downloads/2608.02478v2.pdf`, especially Eqs. (26)--(34)
and Lemmas 6.7, 6.8, 6.12.

Companions:

- `experiments/walsh_hessian_noise_audit.py`
- `experiments/walsh_cross_scale_control_audit.py`
- `tests/walsh_hessian_noise_test.py`
- `tests/walsh_cross_scale_control_test.py`
- `experiments/out/walsh_hessian_noise.json`
- `experiments/out/walsh_cross_scale_control.json`

## 1. The claim that failed

The initial objection to a support search was that the empirical
matrix-valued Walsh transform has a dense white noise floor.  "Dense" is
correct in the energy sense; "white" is not.

For one affine coset and one unsparsified sample family, put

    Y = V_P(X),
    Z = w(X) traceless(XX^T),
    T_theta = E[Z (-1)^(theta.Y)].

The one-sample cross-output covariance Gram matrix is exactly

    G[theta,phi]
      = K[theta xor phi] - <T_theta,T_phi>_F,

    K[delta]
      = E[||Z||_F^2 (-1)^(delta.Y)].

Thus whiteness is equivalent to the second-moment Walsh kernel `K` being
concentrated at `delta=0`, after the small mean correction.  The experiment
enumerates the finite dual-lattice cube and computes this object directly;
there is no Monte Carlo covariance estimate.

## 2. Why the floor is correlated

The paper defines `s` by

    1/xi_s^2 = 2/xi_r^2 - 1/xi_R^2,
    s = rR/(2R-r).

Because

    rho_R(X) w(X)^2 = rho_s(X),

the unsparsified second moment is a polynomially tilted discrete Gaussian at
the *narrower* effective width `s`.  Its parity mass is not uniform, so the
Walsh errors are correlated.  This is structural, not an accident of a
particular lattice basis.

For Horvitz--Thompson selection, most accepted samples lie in the branch
`pi(X) proportional to w(X)`.  There

    rho_R(X) w(X)^2/pi(X) proportional to rho_r(X).

The sparsifier therefore moves the second-moment kernel from effective width
`s` toward the wider target width `r`, whose parity mass is more uniform.  It
whitens the floor while increasing its marginal variance.  The measurements
show exactly that change.

## 3. Exact finite-dimensional measurements

Generic well-conditioned bases, the paper's parameters
`r=0.2222355`, `R=0.400613`, `chi=0.3961331`, and cutoff 4:

| n | Walsh N | boundary mass | unsparsified Neff/N | unsparsified K off-origin | HT Neff/N |
|---:|---:|---:|---:|---:|---:|
| 6 | 16 | 1.65e-7 | 0.373 | 0.647 | 0.700 |
| 7 | 32 | 1.08e-6 | 0.370 | 0.640 | 0.721 |
| 8 | 32 | 1.47e-5 | 0.370 | 0.637 | 0.738 |

Here `Neff=(tr G)^2/tr(G^2)`.  The unsparsified off-diagonal covariance is
about 79% of the Gram Frobenius norm.  Five additional random seeds at `n=7`
gave unsparsified `Neff/N` from 0.289 to 0.374 and off-origin kernel energy
from 0.635 to 0.720.  The identity-lattice control was more correlated
(`Neff/N=0.229`), not less.

Trace removal is still the correct spectral operation, but it does not change
the Walsh effective dimension: raw and traceless `Neff/N` agree within about
0.2% in these runs.  It removes irrelevant marginal variance, not an
exponential family of Walsh modes.

## 4. What this does and does not buy

The nonwhite floor falsifies an assumption, but the measured effective
dimension remains a constant fraction of `N`; it has not yet exposed an
exponential reduction.  Likewise, 90% of the unsparsified kernel energy needs
roughly 6.5 of 16 or 11--12 of 32 deltas.  A support certificate based only on
this covariance still appears to resolve nearly all parity bits in the worst
case.

The paper's importance exponent also appears structurally credible.  For a
fixed nonzero traceless test matrix `A`,

    E_R[w^2 <XX^T,A>^2]

is a fourth polynomial moment under the width-`s` Gaussian.  Removing the
identity changes polynomial factors but does not normally change its
exponential Gaussian-mass rate.  A proof needs matching lower bounds, but the
paper's `iota(r,R)` is not merely an artifact of an entrywise union bound.

## 5. Cross-scale Meyer control

Several target widths can be estimated from the same source samples.  The
second experiment computes their full cross-covariance and minimizes
integrated variance subject to preserving the known asymptotic spike scale

    a(r) proportional to (r/R)^(n/2) r^2 2^(-rn).

Five widths produced finite-n variance reductions of 1.28x, 1.28x, and 1.53x
at dimensions 6, 7, and 8.  Those combinations are high-order finite
differences with coefficient L1 norms in the thousands, however, and are not
stable evidence of an exponent change.

Stable two-width controls have coefficient L1 norms around 2.5--4.8 and reduce
variance only 1.03--1.06x.  Three widths reach 1.14--1.19x with L1 norms around
18--30.  Cross-scale cancellation is useful as a numerical constant-factor
control, not yet as the source of a lower asymptotic exponent.

## 6. The access-model trap

After a random traceless scalar projection, locating `theta_star` resembles
decoding a noisy Hadamard codeword.  Ordinary Goldreich--Levin does not apply
for free: it assumes query access to a bounded received function, while here
the natural data are weighted random examples or a sparse empirical
histogram with exponentially bad dynamic range after normalization.

Viewed as random labeled examples, the generic abstraction is Learning Parity
with Noise.  BKW-style collision reduction squares the bias at each merge.
Our initial correlation is already exponentially small, so a direct BKW
transplant can easily increase rather than decrease the exponent.  Any claim
that a heavy-coefficient routine replaces the full Walsh pass must explicitly
account for this access model and bias loss.

## 7. Current verdict and next theorem

The correct ledger is:

1. The unsparsified Walsh floor is dense but strongly correlated.
2. Horvitz--Thompson storage sparsification partially destroys that useful
   correlation.
3. The observed correlation is a constant-fraction reduction, not yet an
   exponential one.
4. Trace removal is spectrally necessary but does not remove the importance
   exponent.
5. Stable cross-scale controls improve constants, not the measured exponent.

The next proof target is the matrix-valued noisy-parity problem with this
specific Gaussian side information, not generic sparse WHT:

> Given samples `(Y_i,X_i,w_i)`, recover a parity `theta` for which
> `sum_i w_i traceless(X_iX_i^T)(-1)^(theta.Y_i)` has a rank-one spike, in
> time below `2^ell`, without squaring the exponentially small bias.

A second, independent route is proposal redesign: find a sampleable coset
distribution whose likelihood ratio to `D_{Lambda_j,xi_r}` has smaller
chi-square exponent than the single wider Gaussian.  The exact audit can test
any proposed distribution by replacing its first- and second-moment weights.

The follow-up `notes/WALSH_EIKONAL_RECOVERY.md` replaces the generic
noisy-parity formulation with a concrete partial-Walsh first-arrival action.
It derives a certified frontier from Lemma 5.2 and identifies a rebalanced
`h=0` parameter regime that reaches the `2^(n/2)` preprocessing floor if one
remaining near-linear, `l2`-robust sparse-WHT recovery lemma can be proved.
