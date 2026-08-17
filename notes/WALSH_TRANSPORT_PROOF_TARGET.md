# Walsh recovery: the transport proof target

Date: 2026-08-13.  This note sharpens the recovery problem left in
`notes/WALSH_EIKONAL_RECOVERY.md`.

Companions:

- `experiments/walsh_radial_matched_filter.py`
- `experiments/out/walsh_radial_matched_filter.json`
- `tests/walsh_radial_matched_filter_test.py`

## 1. The strong sampler theorem is too broad

The first proposed theorem asked for `2^((2r+o(1))n)` samples from the narrow
Gaussian on a random half-parity affine coset in total `2^(n/2+o(n))` time.
That is a batch below-smoothing DGS statement.  Existing discrete-Gaussian
combiners obtain `2^(n/2)` arbitrary-parameter samples in `2^(n+o(n))` time;
their `2^(n/2+o(n))` batch algorithm requires the target lattice or shift to
be above smoothing.  The new one-sample arbitrary-parameter result does not
give the required batch: its Gaussian-mass comparison spends the full
`2^(n/2)` factor per accepted target-lattice sample.

This does not disprove the random-coset theorem, but it shows that proving it
as a black-box DGS statement would resolve a substantially broader sampling
bottleneck than the Hessian recovery actually needs.

## 2. Exact radial matched-filter lemma

Consider the continuous source density

    p_R(x) proportional to exp(-pi ||x||^2/xi_R^2)

and let `u` be a unit target direction.  Write

    H(x) = x x^T - ||x||^2 I/n.

Every rotation-equivariant traceless one-sample matrix kernel has the form

    F(x) = a(||x||) H(x)

for a scalar radial function `a`.  At the target frequency `u/2`, its signal
is a multiple of `u u^T-I/n`.  Conditional on `rho=||X||`, put

    c_n(z) = E_omega[
        ((u.omega)^2 - 1/n) cos(z u.omega)
    ],
    z = pi d rho.

Cauchy--Schwarz gives the exact optimum over all measurable radial `a`:

    sup_a signal(a)^2 / E_R[||F(X)||_F^2]
      = E_R[c_n(pi d ||X||)^2] / (1-1/n).

The polynomial factor is irrelevant to the exponential rate.  Equality is
attained by

    a(rho) proportional to c_n(pi d rho) / rho^2.

This is the rotationally matched Laguerre/Bessel filter.  It already contains
the closure of stable Gaussian mixtures and cross-scale radial controls, so a
radial Meyer window cannot beat its signal-to-noise ratio.

## 3. Closed form and asymptotic exponent

Put

    nu = n/2 - 1,
    x = 2 n R ln 2.

Funk--Hecke gives

    c_n(z)
      = -A_n z^(-nu) J_(nu+2)(z),

    A_n
      = 2^(nu-1) Gamma(nu+1) (2nu+1)/(nu+1).

The source-radius integral is then Weber's Bessel integral:

    E_R[c_n(pi d ||X||)^2]
      = A_n^2 / Gamma(nu+1)
        * (2x)^(-nu) exp(-x) I_(nu+2)(x).

This is an exact finite-dimensional formula in the continuous surrogate.
The uniform large-order asymptotic for `I_nu(nu*c)`, with

    c = 4 R ln 2,
    eta(c) = sqrt(1+c^2)
             + ln(c/(1+sqrt(1+c^2))),

gives the optimal sample exponent

    m_rad(R)
      = [1 + ln c - ln 2 - eta(c) + c] / (2 ln 2).

## 4. The paper's importance width attains the radial optimum

The Gaussian importance family has exponent

    m(r,R)
      = 2r + (1/2) log2(R^2/(r(2R-r))).

Its unique minimizer is `r=R*q`, where, for `a=2R ln 2`,

    q = [(2a+1)-sqrt(4a^2+1)]/(2a).

Direct substitution shows

    min_(0<r<R) m(r,R) = m_rad(R).

At the paper's source width,

    R = 0.400613,
    r_opt = 0.2222354295...,
    m_rad(R) = 0.603865753375... .

The target width printed in the paper, `0.2222355`, is the radial
matched-filter optimizer to the shown precision.  At the source width needed
by the elementary half-coset smoothing condition,

    R = 2 t0 = 0.46294...,
    m_rad(R) = 0.670255...

The quadrature audit converges to both values and checks the exact Bessel
formula at each finite dimension.

Therefore the `0.603867` constant cannot be lowered by any one-sample,
rotation-equivariant radial redesign.  This includes arbitrary radial
Gaussian mixtures, stable cross-scale controls, radial acceptance profiles,
and a direct radial transcription of the Meyer support window.

## 5. Minimal non-radial theorem

The surviving theorem should not demand exact narrow DGS.  Let

    pi_R = D_(Lambda_j,xi_R),
    pi_r = D_(Lambda_j,xi_r),

with `h=(1/2-o(1))n`, `R` large enough to sample `pi_R`, and `r>t0`.  Seek an
efficient non-radial Markov transport `K` on the fixed affine coset and let

    q = pi_R K.

It is enough to prove

> **Low-congestion half-coset transport.** The kernel can be sampled in
> `2^o(n)` amortized time per particle after `2^(n/2+o(n))` preprocessing, and
>
>     1 + chi^2(pi_r || q) <= 2^((delta+o(1))n)
>
> for some `delta < 1/2-2r`, uniformly at every scale used by the algorithm.

Then `2^((2r+delta+o(1))n)` transported importance samples give the uniform
Walsh-Hessian concentration required by Lemma 6.9, while the complete
half-dimensional WHT and BDD preprocessing cost `2^(n/2+o(n))`.

For `R=2t0` and the radially optimal `r=0.236758...`, endpoint radial
importance has

    delta_rad = 0.196738...,

whereas the theorem needs

    delta < 1/2-2r = 0.026482... .

Thus the transport must save about `0.170256n` in Renyi-2 action.  This is the
precise constant that a support-moving eikonal construction must account for.

## 6. What an eikonal proof must establish

The support machinery suggests a temperature ladder on one fixed affine
coset, with irreversible first-arrival ownership of transported particles.
For a proof, the ladder must include genuine non-radial mutation.  Reweighting
the same particles at many small temperature increments cannot help: the
Radon--Nikodym products telescope to the endpoint ratio and reproduce
`delta_rad`.

A sufficient proof can take either form:

1. a path-space Renyi contraction showing that each mutation step dissipates
   the radial likelihood action and that the total residual is below
   `(1/2-2r)n`; or
2. a Poisson-equation/control-transport construction for the traceless
   Hessian observable whose second moment satisfies the same exponent, even
   if the transported particle law is not close to `pi_r` globally.

Any local lattice walk also needs a worst-case conductance argument.  General
below-smoothing lattice Gaussians can be multimodal, so an unproved generic
mixing assumption would simply rename the theorem.  The useful structure
still available is the random parity subspace, the `2^(n/2)` BDD/DGS
preprocessing, and the fact that only the rank-one Hessian observable—not the
entire cold distribution—must be transported accurately.

## 7. Exact finite-coset contraction audit

`experiments/walsh_coset_contraction_transport.py` tests the simplest genuine
support-moving proposal.  In each enumerated affine coset it maps

    x -> nearest point in the same coset to sqrt(r/R) x,

then mixes with the wide source by the amount that minimizes the exact
Renyi-2 divergence to the cold target.  A soft version splits mass among the
16 closest points using an optimized local heat-kernel bandwidth.  This test
is deliberately generous: it uses exact nearest-neighbour search and tunes
the bandwidth against the known target distribution.

At `R=2t0`, `r=0.236758...`, cutoff 2, and dimensions 4 through 7:

- hard contraction saves only `0.00001` to `0.00165` bits of action per
  dimension;
- soft contraction saves `0.0066` to `0.0240` bits per dimension;
- the asymptotic ledger requires `0.170255` bits per dimension.

Soft contraction is a real improvement and covers the finite target support,
but remains far from the required action.  More revealingly, an oracle-tuned
temperature ladder with 2, 4, or 8 small projection steps becomes worse, not
better.  At dimensions 4 through 6 its saving collapses toward zero as the
number of steps grows.  Each contraction becomes smaller than the local
lattice spacing, and the optimal heat kernel approaches the identity; the
steps do not accumulate into macroscopic inward motion.

This is finite-dimensional evidence, not an asymptotic impossibility proof.
It nevertheless rejects the direct transcription of local eikonal marching.
A successful kernel needs nonlocal support rejuvenation across lattice cells,
not merely smaller temperature increments and local reprojection.

## 8. The exact combiner and its address tradeoff

There is one canonical nonlocal inward move.  If two wide Gaussian samples
from the same affine coset agree modulo `2L*`, their average lies in `L*` and
has width parameter `R/2`.  Thus choosing `R=2r` implements the desired cold
move exactly.  With `h=n/2`, the relevant intermediate lattice is

    2 Lambda_0 subset 2L* subset Lambda_0,

and dividing the pair sum by two relaxes all `h` affine parity constraints.
The fast above-smoothing ADRS combiner can therefore produce a cold batch on
`L*` in `2^(n/2+o(n))` time.

The price is address loss: the output now has `n` free parity bits, so the
paper's final transform returns to size `2^n`.  Keeping the collision bucket
as an `n/2`-bit address avoids that transform, but its Walsh coefficient is a
product of a scalar Gaussian coefficient and a Hessian coefficient.  The
shortest-vector signal is consequently squared, of order `2^(-2rn)`, and
sample-optimal concentration costs `2^(4rn)` (about `2^0.947n` at the
half-coset target).

This isolates the remaining combinatorial issue:

> transport inward while preserving a half-dimensional *linear* parity
> address.  Ordinary averaging either forgets that address or turns the
> linear Hessian signal into a quadratic one.

That address-preserving, low-congestion combiner is now the most concrete
form of the missing theorem.

## 9. Moving harmonic subspaces lower the constant directly

Chapter 2 of *Ten Advances in Mathematics and Theoretical Computer Science*
improves the asymptotic spherical-code bound by replacing the fixed harmonic
line attached to each point with a moving high-rank stabilizer subspace.  If
`P_x` is its projector, `tr(P_x P_y)` is still a scalar two-point kernel, but
the projection inequality pays the ambient-to-fiber dimension ratio `D/d_E`.
The one-row spherical construction has

    Gamma_row(a,b)
      = (a-b)(1+a+b)
        / ((1+2a) sqrt(a(1+a))),

    Phi_row(a,b) = H_sph(a)-H_sph(b),

and gives the exponent `Phi_row` whenever `2 Gamma_row > s`.  The classical
KL boundary is exactly `b=0`; positive `b` subtracts an exponential harmonic
fiber dimension while paying only a smaller spectral displacement.

This transfers directly into Hhan's Lemmas 6.2 and 6.3.  Replace both

    log2(beta) = B_KL(1/2)

and

    K_2(x) = B_KL(1-2/x^2),  x > sqrt(2),

by any smaller valid spherical-code exponent.  No recovery argument changes:
the Gaussian shell count improves pointwise, hence `g_2(R)` increases and the
allowable parity-prefix exponent `chi` increases.

`experiments/walsh_spherical_bound_transfer.py` implements the moving one-row
bound, including the spherical-cap optimization, and reoptimizes all of
`r,R,chi`.  Its exact nested-bound check gives

    classical KL used by the lattice paper:  0.60386572...
    classical cap optimization alone:         0.60385570...
    moving one-row bound:                      0.60246909...

at

    r = 0.22187205...,  R = 0.39935520...,
    chi = 0.39753094... .

Thus the new projection idea already lowers the SVP constant by
`0.00139663`.  This uses only the simplest one-row subfamily, so the full
spherical representation hierarchy can only improve the bound further.

The binary result itself does not directly replace the spherical count: the
lattice vectors in Lemma 6.2 form a spherical code, not a binary code.  Its
proof architecture nevertheless identifies a sharper recovery target.  The
exact combiner loses an `n/2`-bit scalar address.  Instead of preserving one
Walsh character (a one-dimensional fiber), attach a moving Boolean/Johnson
harmonic subspace to every parity address and propagate the projector overlap
through the combiner's three-term recurrence.  The theorem needed is an
algorithmic counterpart of the projection bound:

> **Harmonic address-preserving combiner.**  A nonlocal inward combiner on the
> random half-coset admits moving projectors of rank `2^((sigma+o(1))n)` whose
> overlap remains a scalar function of the collision distance, whose transfer
> operator has a polynomial-width tridiagonal representation with spectral
> gap bounded away from the rejection threshold, and whose retained Hessian
> observable is linear rather than squared.

To bridge the elementary half-coset endpoint, its dimension credit must
supply `sigma > 0.170255...`, the Renyi-action deficit from Section 5.  This
is the Boolean-harmonic entropy `H_2(b)` already at normalized degree
`b = 0.0253021...`; the required fiber is low-degree even though its rank is
exponential.  The spectral displacement of that degree, rather than raw
dimension availability, is therefore the quantity the combiner theorem must
control.  This
is not proved by the code-bound paper: its projection inequality is
nonconstructive as a sampler.  But it gives the first mechanism we have found
that attacks exactly the combiner's address loss without returning to a full
`2^n` transform or squaring the shortest-vector signal.

## 10. Correction: the harmonic fiber must implement a bridge

The exact mixed-Walsh calculation in
`notes/WALSH_COLD_BRIDGE_PROOF_TARGET.md` shows that a target-independent
rank-`d` harmonic sketch retains only `d/2^h` energy from the uniformly random
missing prefix.  Rescaling it restores the same factor in noise.  Thus the
rank condition in the preceding proposed theorem is not sufficient by
itself; the code-bound projection gain relies on an off-diagonal sign
constraint absent from coefficient estimation.

The surviving theorem is cold syndrome folding.  Exact averaging first gives
`D_(L*,xi_r)`.  A state-dependent Doob/eikonal bridge must then condition that
cold law onto the desired affine half-coset with Renyi action below
`(1/2-2r)n = 0.02648284...n`.  The complete corrected statement and its
elementary bridge proof are in the companion note above.
