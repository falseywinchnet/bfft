# Gaussian-cell transport seed for shortest parity

Date: 2026-08-13.

The verifier Gibbs law has overwhelming stationary shortest mass but no
proved fast entrance.  This note replaces Markov-chain entrance by a one-shot
transport through a fundamental coefficient cell.  The resulting law is
directly samplable and has an exact shortest-parity lower bound.

## 1. The law

Let `L=B Z^n`, let `C=[-1/2,1/2)^n`, and let `lambda=lambda_1(L)`.
For `tau>0`, draw

    Y ~ N(0,tau^2 I_n),       X=B^(-1)Y,
    Z=round(X) in Z^n.                                  (1)

Rounding ties have probability zero.  Repeat (1) only when `Z=0`, and output

    theta=Z mod 2.                                      (2)

The law may output the zero parity when `Z` is a nonzero even vector.  Treat
that event, and any event rejected by the exact syndrome/Hamming sieves, as a
failure symbol.  This does not reduce the mass of a shortest parity.

Define the central-cell median scale `tau_B` by

    Pr[B^(-1)Y in C] = 1/2.                              (3)

The probability in (3) is continuous and strictly decreasing from one to
zero, so `tau_B` is unique.  It is efficiently estimated to inverse-
polynomial relative accuracy: for `g~N(0,I)`, the largest scale at which
`tau B^(-1)g` remains in `C` is

    T(g)=1/(2 ||B^(-1)g||_infinity),                    (4)

and `tau_B` is the median of this one-dimensional random variable.  A
polynomial scale grid and independent batches avoid any reliance on an exact
real median.

## 2. Exact cell-shift inequality

Let `p_tau(z)=Pr[X in z+C]`.  Gaussian translation gives

    p_tau(z)
      = exp(-||Bz||^2/(2tau^2))
        integral_C phi_tau(Bx)
          exp(-<Bz,Bx>/tau^2) det(B) dx.                (5)

The central density and `C` are symmetric under `x -> -x`.  Pairing `x` and
`-x` replaces the last exponential by a hyperbolic cosine, which is at least
one.  Therefore

    p_tau(z) >= exp(-||Bz||^2/(2tau^2)) p_tau(0).       (6)

This is the required transport inequality: it moves the complete central
rounding cell to the coefficient cell of `z` without paying a determinant,
condition number, boundary-volume, or pathwise Jacobian factor.

Let `z_*` represent a shortest vector.  The cells of `z_*` and `-z_*` are
distinct and have the same parity.  At `tau=tau_B`, (3) and (6) imply

    Pr[Z mod 2 = z_* mod 2 | Z != 0]
      >= 2 exp(-lambda^2/(2 tau_B^2)).                  (7)

The right side may of course be truncated at one.  Rejection has expected
cost exactly two trials.  The final target-Hessian and BDD endpoint is then
applied to the parity in (2), with exact verification as before.

> **Theorem (Gaussian-cell solvable class).**  Suppose, for a correct length
> guess, that
>
>     tau_B^2 >= lambda^2/(2 delta n ln 2).              (8)
>
> Then one Gaussian-cell seed reaches a shortest parity with probability at
> least `2^(-(delta+o(1))n)`.  Reusing the common target DGS panel for
> `2^((delta+o(1))n)` independently generated cells and verifying every BDD
> output gives constant success probability in total time
>
>     2^((2r+delta+o(1))n).                              (9)
>
> Hence `delta<1/2-2r=0.02648284...` closes the half-exponential ledger.

Unlike the verified Gibbs construction, this is an initialized sampler, not
only a stationary law.

## 3. A checkable dual-basis sufficient condition

Let `b_i^*` be the rows of `B^(-1)`, viewed as dual basis vectors, and put

    D=max_i ||b_i^*||.

For `g~N(0,I)`, a union bound gives

    Pr[||B^(-1)g||_infinity > D sqrt(2 ln(4n))] <= 1/2.

Equations (3)--(4) therefore give

    tau_B >= 1/(2D sqrt(2 ln(4n))).                     (10)

Combining (8) and (10), it suffices that

    lambda D
      <= sqrt(delta n ln 2)/(2 sqrt(ln(4n))).           (11)

In particular,

    lambda D=o(sqrt(n/ln n))                            (12)

implies a directly samplable shortest-parity mass `2^(-o(n))`.  If
`lambda D=O(1)`, the explicit loss is only `O(ln(n)/n)` in the exponent per
dimension.

The basis can be replaced by `BU` for any unimodular integer `U`, and the
largest right-hand dual-basis norm can be minimized over efficiently
available reduced dual bases.  Verified short-dual quotients and the spectral
support-chain cover should be applied first; (11) is required only on the
remaining coefficient block.

Condition (11) is deliberately only sufficient.  The exact median (4) can be
much larger than its rowwise union-bound lower bound because the dangerous
dual directions are correlated.  For the simplex cancellation family the
dual-row product `lambda D` stays bounded; even the elementary union bound
therefore gives `tau_B=Omega(lambda/sqrt(ln n))` and shortest-parity mass
`2^(-O(ln n))=2^(-o(n))`.  Correlation can only improve this bound.

## 4. The new obstruction

The one-shot construction changes the remaining theorem.  We no longer need
a conductance bound for a low-temperature verifier chain.  It is enough to
prove the following basis/quotient dichotomy.

> **Median-cell dichotomy (proof target).**  In `2^(n/2+o(n))`
> preprocessing, either the short-dual/Hamming/spectral covers enumerate the
> shortest parity within the existing budget, or construct a unimodular basis
> of the residual block whose central-cell median satisfies (8) for some
> `delta<0.02648284`.

A counterexample must now have all earlier cover entropies large and must
make every efficiently obtainable fundamental parallelepiped have Gaussian
median width below about `5.22 lambda/sqrt(n)`.  This is a more concrete
geometric obstruction than Gibbs mixing: it is expressed entirely through
the Gaussian measure of one symmetric convex body and admits dual-basis,
successive-minimum, and transference attacks.

## 5. A volume obstruction to the median-cell certificate

The median-cell route is not universal.  This can be proved without choosing
a bad basis.  Let `L` be unimodular and let `C` be any measurable fundamental
domain, so `vol(C)=1`.  Let `r_n` be the radius of a Euclidean ball of volume
one, and let `m_n` be the median of the chi distribution `||N(0,I_n)||`.
Symmetric decreasing rearrangement gives

    Pr[N(0,tau^2 I_n) in C]
      <= Pr[||N(0,tau^2 I_n)|| <= r_n].                 (13)

Consequently, if the left side is `1/2`, then

    tau <= r_n/m_n
        = 1/sqrt(2 pi e) (1+o(1)).                     (14)

Now take an asymptotically extremal self-dual Conway--Thompson family, for
which

    lambda_1(L)^2 >= (1/(2 pi e)-o(1)) n.              (15)

For every volume-one fundamental cell, the exponent supplied by the
translation certificate (7) is therefore at least

    lambda_1(L)^2/(2 tau^2 n ln 2)
      >= 1/(2 ln 2)-o(1)
       = 0.721347...-o(1).                              (16)

This is a no-go theorem for the *constant-central-mass cell-shift
certificate*, not for every Gaussian rounding law: (6) can be very loose,
and aggregating all cells of one parity can add mass.  It does show that
changing from a parallelepiped to a rounder fundamental domain, or replacing
the median by any other fixed quantile, cannot close the `0.02648284` gap on
the dense self-dual obstruction.

## 6. What if the zero parity, rather than only the zero cell, is rejected?

Continuous Gaussian trials and coefficient rounding are cheap compared with
a target Hessian query.  It is therefore legitimate to choose a colder scale,
reject every output in `2 Z^n`, and query the Hessian only for an accepted
nonzero parity.  The experiment

    experiments/walsh_gaussian_cell_quantiles.py

audits this law over central-cell probabilities `0.99, 0.9, 0.5, 0.1`.
There is a sharp geometric limitation.  In the cold limit, exits from a
basis parallelepiped cross one of its `2n` facets, so their coefficient
addresses are the unit parities.  Cold rejection is therefore excellent when
a shortest vector labels a nearest basis facet (as in the rectangular
family), but suppresses the all-ones parity of the simplex cancellation
family: reaching it requires simultaneous crossing of every relevant basis
facet.

The corresponding intrinsic construction would replace the basis
parallelepiped by the Voronoi cell.  Its nearest facets are labelled by
shortest Voronoi-relevant vectors, so a cold first-exit law would concentrate
on shortest parities.  But sampling or even recognizing that exit law needs
a Voronoi-cell separation/nearest-lattice oracle at distance about
`lambda_1/2`; the available BDD radius does not provide it.  This exposes the
next viable object cleanly:

> **Punctured intrinsic-cell transport target.**  Using the common DGS/BDD
> advice, sample the first-exit label of a computable inner approximation to
> the origin Voronoi cell, with total distortion below
> `2^((0.02648284-o(1))n)`, and retain the Hessian/BDD verification step.

This is the eikonal version of the surviving theorem.  It transports boundary
flux rather than stationary cell mass.  A proof must construct the boundary
oracle without already solving CVP; a basis-cell implementation cannot work
on dense cancellation parities.

The finite geometry of this proposal is audited independently in

    experiments/walsh_voronoi_first_exit.py.

For a random unit ray `u`, the first Voronoi label is the lattice vector `w`
minimizing

    ||w||^2/(2 <u,w>)                                   (17)

over positive denominators.  Thus the experiment can measure the total solid
angle of shortest-vector facets without implementing CVP or a Hessian
surrogate.  Comparing this ideal mass with the basis-cell result separates a
geometric obstruction (too little shortest-facet angle) from an algorithmic
one (the available advice cannot expose the first boundary).

## 7. A universal first-exit bound, and why it is not enough

There is a basis-free cap that can be proved for every oriented shortest
vector.  Put `a=v/lambda`, and let `u` make angle at most `pi/6` with `a`.
For another lattice vector write `w=q lambda b`, `q>=1`.  The ray meets the
bisector of `0` and `w` at

    t_w=q lambda/(2 <u,b>),                              (18)

when the denominator is positive.  Lattice minimality gives, for `w!=v`,

    ||w-v||>=lambda, hence <a,b><=q/2.                  (19)

If `q>=2`, (18) is no smaller than the `v` boundary immediately.  If
`1<=q<2`, resolving `u` into its components parallel and perpendicular to
`a` and using `tan(pi/6)=1/sqrt(3)` gives

    <u,b> <= q <u,a>.                                   (20)

Thus `t_v<=t_w`: every direction in the `pi/6` cap first exits through the
facet labelled by `v`.  The open caps of distinct oriented shortest vectors
are disjoint because shortest vectors have mutual angle at least `pi/3`.
If `K(L)` is the number of oriented shortest vectors, then

    Pr[random ray exits through a shortest facet]
      >= K(L) Cap_n(pi/6)
       = K(L) 2^(-(1+o(1))n).                           (21)

Equation (21) is a genuine universal lower bound but is far outside the
`0.02648284` budget.  A proof through first-exit flux must exploit the
residual dense-core hypothesis, not only lattice minimality.

The cutoff-two audit with 20,000 directions per fixture found the following
ideal shortest-parity exit masses:

| family | n=3 | n=4 | n=6 |
|---|---:|---:|---:|
| rectangular | 0.8876 | 0.9357 | 0.9815 |
| simplex cancellation | 0.2133 | 0.1583 | 0.1085 |
| generic | 0.3020 | 0.2854 | 0.1772 |
| adversarial rotated | 0.5056 | 0.4078 | 0.1983 |
| needle/D shell | 0.3417 | 0.2228 | 0.1177 |

The finite coefficient cube is not a completeness proof.  Still, it shows
that the ideal intrinsic boundary law repairs the severe basis-cell failure
on the simplex family: its measured loss is roughly polynomial over these
dimensions.  This motivates the sharpened dichotomy:

> **Dense-core Voronoi-flux dichotomy.**  Either the short-dual/Hamming/
> spectral-tube machinery produces at most
> `2^((0.02648284-o(1))n)` parity candidates, or the union of origin-Voronoi
> facets labelled by shortest vectors has spherical first-exit measure at
> least `2^(-(0.02648284-o(1))n)`.  In the second case, construct from the
> common Hessian panel a polynomial-evaluation ray continuation that exposes
> the first label without a CVP oracle.

The first clause is already proved in the earlier cover theorems.  The
second geometric implication and its Hessian implementation are now the two
separate remaining obligations.  The cap lemma proves the same statement
with constant `1` in place of `0.02648284`; closing that constant gap is the
precise obstruction.

## 8. The first-exit oracle can be removed: an absolute radial catalog

The ideal Voronoi boundary is useful for diagnosis but is not required by
the strongest finite construction.  The following law uses only the length
guess `d` and the common target-width Hessian panel.

1. Draw a physical unit direction `u` uniformly from the sphere.
2. On a polynomial grid of radii in `[0.4d,3d]`, evaluate the traceless
   Hessian and its eigengap (or an eigengap-free soft anisotropy surrogate).
3. Retain the top polynomially many separated radial peaks, keeping all
   inverse-polynomial near ties.
4. From every retained point, run the gap-free periodic-Hessian centering
   flow, query BDD with the terminal direction, and verify the output.

This is a fully samplable seed law.  Each seed uses only polynomially many
evaluations of one common panel, and no coefficient rounding, Voronoi oracle,
target parity, or boundary time.  The uniform empirical derivative argument
already used for the accessible soft law applies directly at target scale
with

    M=n^C 2^((2r+o(1))n)                               (22)

samples; direct target samples require no importance factor.  Ties in the
radial catalog are handled by retention, while the centering flow uses the
analytic log-trace-exponential objective.

There is an exact exponent explaining the radius `3d`.  For a fixed shortest
vector `v`, put `a=v/lambda`.  A ray `tu` intersects the Euclidean bisector
of `0` and `v` at

    t=lambda/(2 <u,a>).                                 (23)

Thus `|<u,a>|>=1/6` puts one of the two oriented intersections within radius
`3lambda<=3d`.  Spherical-cap asymptotics give

    Pr[|<u,a>|>=1/6]
      =2^(-(delta_ray+o(1))n),
    delta_ray=-1/2 log_2(35/36)
             =0.0203209922... .                         (24)

This lies below `delta_query=0.02648284...`.  If a robust radial-capture
lemma turns the bisector event into a retained peak whose centering reaches
a shortest midpoint, repetition gives total target-panel work

    2^((2r+delta_ray+o(1))n)
      =2^((0.493838153+o(1))n),                         (25)

below the preprocessing half exponent, with `0.00616184n` exponent slack.
Equation (24) alone is not that lemma: an arbitrary bisector point can be far
along the facet tangent, and other lattice terms can dominate its Hessian.
This tangential interference is the remaining geometry.

`experiments/walsh_voronoi_hessian_ray.py` separates the pieces.  At the
enumerated first-boundary point, direct target-label alignment can degrade on
shortest exits (at `n=5`, median `0.876` for simplex and `0.648` for one
generic fixture).  Full centering repairs much of that loss.  More
importantly, the absolute 41-radius/three-peak catalog, which does not use the
boundary, achieved the following `n=5` success masses:

- simplex cancellation: `0.3125` in the main run and `0.3542` in the stress
  run;
- six generic fixtures: `0.1562`--`0.2604`;
- six rotated anisotropic fixtures: `0.1146`--`0.8542`;
- needle/D shell: `0.9648`--`0.9688`.

The median work was at most `137` population field evaluations per seed in
that panel.  These are finite truncated-population measurements, not an
asymptotic proof.  They identify the precise replacement theorem:

A second stress panel retained the union of three separated leading-score
peaks and three separated eigengap peaks.  This is still constant catalog
size.  At `n=5`, its minimum success mass was `0.250` over four generic
fixtures and `0.156` over four rotated-anisotropic fixtures; simplex rose to
`0.391` and the needle family stayed at `0.969`.  Thus peak ranking is not the
observed obstruction, although the theorem should retain polynomially many
near ties rather than assume a fixed catalog of three.

> **Robust radial-capture lemma.**  On every residual dense-core lattice and
> a valid length guess, for some shortest `v`, a subset of the cap
> `|<u,v/lambda>|>=1/6` having relative spherical measure `2^-o(n)` consists
> of directions with a polynomial-grid radial candidate whose gap-free
> centering flow reaches a robust shortest midpoint.  The radial catalog and
> flow sensitivity are `2^o(n)`.

This lemma plus (22)--(25) gives the requested `2^(n/2+o(n))` recovery.  It
is strictly narrower than the earlier global basin conjecture: it specifies
the seed geometry, the bounded radial interval, the catalog, and the exact
exponent available for failure.
