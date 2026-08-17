# Accessible-scale soft-Hessian seed law

Date: 2026-08-13.

This note separates the constructed sampling/oracle statement from the one
remaining geometric conjecture.  The construction is implemented in
`experiments/walsh_accessible_seed_law.py`.

## 1. The law

Let `r=0.23675858...` be the target width.  The bare half-exponent equation

    2t + (1/2) log_2(r^2/[t(2r-t)]) = 1/2.

has a lower root `0.0978695...`, but that root is not uniformly justified:
the theta-mass estimate for the importance second moment requires

    q(t,r) := tr/(2r-t) > t0/2.

We therefore use the rigorously controlled scale `t_*=0.16`.  Numerically,

    q(t_*,r) = 0.1208271... > t0/2,
    iota(t_*,r) = 0.0801086...,
    2t_*+iota(t_*,r) = 0.4001086... < 1/2.

For the traceless periodic Hessian `T_t(z)`, choose a reference signal scale
`sigma_t` from the current length guess and define the gap-free score

    Phi_(t,beta)(z)
      = (sigma_t/beta) log tr exp(beta T_t(z)/sigma_t),

where `beta=poly(n)`.  The seed law `nu_hat` is sampled as follows.

1. Draw `Z_0` from Haar measure on `R^n/L`.
2. Follow a fixed polynomial-step discretization of the empirical vector
   field `grad Phi_(t_*,beta)`.
3. Express the terminal point as `Bc`, round `2c` to an integer `theta`, and
   output the exact half-grid point `B theta/2 mod L`.
4. Evaluate the target-width Hessian at that exact point, send its leading
   direction to BDD, and verify every returned lattice vector.

The log-trace-exponential is essential.  It is analytic when the leading
eigenvalue is multiple, whereas `lambda_max(T)` has no stable gradient there.
If `lambda_1(T)-lambda_2(T) >> sigma_t/beta`, the soft score and gradient
agree with the leading-eigenvalue score up to `exp(-poly(n))` relative error.

At a half-grid point every Fourier sine vanishes, so snapping is compatible
with every scale.  If `theta` is the parity of a shortest vector `v`, then
`B theta/2 = v/2 mod L`; the final target Hessian is exactly the midpoint
observable required by the BDD recovery step.

## 2. Uniform empirical derivative theorem

The stability half of the requested theorem can be proved independently of
the basin geometry.

> **Proposition (uniform narrowing-importance oracle).**  Fix `j<=2` and a
> rational length/scale guess.  Draw
>
>     M = n^C 2^((2t_*+iota(t_*,r))n)
>       = 2^((1/2+o(1))n)
>
> independent target-width dual DGS samples, for a sufficiently large fixed
> `C`.  There is a self-normalized importance estimator such that,
> simultaneously for every torus point `z` and `j=0,1,2`,
>
>     ||D^j T_hat_(t_*)(z)-D^j T_(t_*)(z)||_op
>       <= sigma_t/n^10
>
> except with probability `exp(-Omega(n^2))`.  The same panel supplies all
> adaptive queries.  Consequently the normalized soft score and its first two
> derivatives have inverse-polynomial uniform error.

Proof sketch.  Write `s_r` and `s_t` for the corresponding spatial Fourier
widths.  Since `t<r`, one has `s_t>s_r`, and the unnormalized likelihood ratio

    u_t(X) = exp(-pi(s_t^2-s_r^2)||X||^2)

lies in `[0,1]`.  No clipping or heavy-tail control of the likelihood ratio is
needed.  Its small mean is the source of the importance exponent.  The exact
population identity is

    E_r[u_t(X) A(X)] / E_r[u_t(X)] = E_t[A(X)].

A `j`th derivative is a matrix trigonometric average whose
single-sample coefficient has size at most

    C_j ||X||^(2+j).

Truncate the DGS norm at a polynomial multiple of its Gaussian radius.  This
loses `exp(-Omega(n^2))` probability and makes all polynomial moments bounded.
The relative second moment

    E_r[u_t^2] / E_r[u_t]^2

has exponent `iota(t,r)`.  More explicitly, if `Z_a` denotes the dual theta
mass at width parameter `a` and

    q = tr/(2r-t),

then

    E_r[u_t] = Z_t/Z_r,
    E_r[u_t^2] = Z_q/Z_r,
    E_r[u_t^2]/E_r[u_t]^2 = Z_q Z_r/Z_t^2.             (1)

The spherical theta bounds applied to (1) give

    log_2(E_r[u_t^2]/E_r[u_t]^2)
      <= (iota(t,r)+o(1))n.

At a fixed point, scalar concentration for the denominator followed by matrix
Bernstein for the bounded weighted numerator
requires `sigma_t^-2=2^((2t+o(1))n)` effective samples.  Importance sampling
multiplies this by `2^((iota(t,r)+o(1))n)`, giving the displayed sample count.
Cover a rational fundamental parallelepiped at resolution
`sigma_t/poly(n)` after bounding the physical Fourier frequencies by the tail
event.  Its logarithmic covering number is polynomial in `n` and the input bit
length.  The polynomial factor `n^C` absorbs this union bound.  The next
derivative supplies a deterministic Lipschitz bound that extends the estimate
from the net to the whole torus.  Applying the same argument through order two
proves the claim.  Analytic functional calculus for log-trace-exp then
transfers the matrix bounds to `Phi`, `grad Phi`, and `D^2 Phi` without any
eigengap assumption.

Because the event is uniform over the torus, adaptively selected ascent points
do not incur another sample or union-bound exponent.  Uniform gradient
accuracy does not by itself imply arbitrarily long trajectory shadowing near a
basin boundary; the basin conjecture below explicitly removes a boundary tube.
For paths whose accumulated flow sensitivity is `2^o(n)`, the estimator can be
tightened by a `2^o(n)` sample factor and standard discrete Gronwall bounds
give the required shadowing without changing the half exponent.

## 3. Amplification and exact remaining clause

Let `p_L` be the Haar measure of starting points whose exact soft flow snaps
to a shortest parity, excluding an inverse-polynomial boundary tube.  Run

    K = 2^o(n)

independent starts on the same uniformly accurate panel.  The success
probability is

    1-(1-p_L)^K-o(1).

The total work is still `2^((1/2+o(1))n)` because both the number of starts
and the number of steps are subexponential and every query reuses the same
panel.

Thus the requested algorithm follows from exactly one geometric statement:

> **Soft shortest-parity basin conjecture.**  For every lattice and one of
> the polynomially many valid length guesses,
>
>     p_L = 2^-o(n).

This conjecture is not proved here.  Calling the constructed law a completed
worst-case SVP algorithm without this inequality would be circular.

## 4. Finite audit

On the simplex-cancellation family, with `beta=16`, cutoff two, and 256 Sobol
starts, the soft law captured the unique all-ones shortest parity with
fractions

    n=3: 0.203125
    n=4: 0.140625
    n=5: 0.1015625
    n=6: 0.08984375.

The median ascent used 14, 16, 18, and 19 field/gradient evaluations.  The
maximum half-grid snapping error fell from `1.15e-3` at `n=3` to `1.57e-6`
at `n=6`.  These measurements show that smoothing the spectral maximum does
not destroy the observed shortest basin.  They do not establish the
dimension-uniform conjecture.

The finite ratio-estimator audit is in
`experiments/walsh_empirical_soft_oracle.py`.  On a fixed generic
four-dimensional lattice, 64 common probe points and sample counts 4096,
16384, and 65536 gave normalized RMS `(value,gradient,Hessian)` errors

    4096:  (1.32e-3, 8.57e-3, 9.74e-2)
    16384: (2.34e-4, 1.72e-3, 2.08e-2)
    65536: (1.15e-4, 1.04e-3, 1.13e-2).

The exact finite-population reweighting identity held to `1.4e-20`--`1.1e-16`
in dimensions two through four, and every weight was at most one.  These are
finite checks of the estimator and derivative formulas, not substitutes for
the covering argument.

## 5. Current boundary

What has now been constructed and proved:

- an explicit samplable law;
- an eigengap-free analytic ascent objective;
- exact parity snapping and target-scale lifting;
- a `2^(n/2+o(n))` uniform empirical oracle through two derivatives; and
- subexponential multi-start amplification conditional on basin mass.

What remains genuinely open is only the lower bound `p_L=2^-o(n)`, or a
lattice-adapted replacement for Haar initialization that satisfies the same
bound.  Exponential branch catalogs, ordinary unimodular basis descent, and
random scale continuation do not prove it.

## 6. Update: a target-scale absolute radial seed law

The Haar-start conjecture is no longer the only concrete route.  The radial
law in `notes/WALSH_GAUSSIAN_CELL_SEED.md` draws a physical unit direction,
evaluates the target Hessian on a polynomial radius grid in `[0.4d,3d]`,
retains a polynomial catalog of separated anisotropy peaks, and applies the
same gap-free centering flow to each.  It needs no Voronoi or boundary oracle.

For a fixed shortest direction, the cap `|cos(angle)|>=1/6` has exponent

    -1/2 log_2(35/36)=0.0203209922...

and its bisector lies within `3d`.  This is below the available
`0.02648284...` query exponent.  Reusing `n^C 2^(2rn)` target samples across
the radial probes and repeated directions costs
`2^((0.493838153+o(1))n)`, below preprocessing.

The remaining clause is the robust radial-capture lemma: a cap ray must
supply a retained radial peak whose soft centering reaches a shortest
midpoint.  Bisector intersection alone does not prove this because of
tangential interference.  Dimension-five stress tests over six generic and
six rotated-anisotropic lattices gave absolute-catalog success mass at least
`0.1146`, but this remains finite evidence rather than the required theorem.
