# Walsh recovery: reverse the dependency graph

Date: 2026-08-13.

The constant-rank no-go theorem rejects one proposed middle layer; it does
not say that cold fixed-coset sampling is intrinsically necessary.  Starting
again from the verified endpoint separates requirements from artifacts.

## 1. What the endpoint actually consumes

The final BDD step needs only one unit vector `q` with

    min_(tau in {+1,-1}) ||q-tau v/lambda_1(L)|| = O(n^(-1/2)).

Every returned lattice vector is verified, so false candidates are harmless.
Only the unknown target parity channel must work.  The endpoint does not
intrinsically require:

- independent samples inside a preselected affine half-coset;
- simultaneous accuracy of every Walsh channel;
- a complete Walsh transform; or
- mixing of an arbitrary function on a cold lattice coset.

Those are sufficient interfaces inherited from the paper's proof.

## 2. What is already available at the beginning

For any `t>t0`, one target-width DGS call on `L*` supplies up to `2^(n/2)`
samples from `D_(L*,xi_t)` in `2^(n/2+o(n))` time.  The rank-one Hessian
needs only `N=2^((2t+o(1))n)` samples.  With the current
`t=r=0.23675858...`, the unused exponent is

    delta_query = 1/2-2r = 0.02648284... .             (A)

The same samples give arbitrary-point, not merely parity-grid, access to the
periodic Gaussian.  For

    F_s(z)=rho_s(L+z)/rho_s(L),

Poisson summation gives

    F_(1/xi)(z)=E_(X~D_(L*,xi)) exp(2 pi i <X,z>).      (B)

After taking the real, traceless Hessian,

    T(z)=-4 pi^2 E[(XX^T-||X||^2 I/n)
                    cos(2 pi <X,z>)].                  (C)

At a midpoint `z=v/2` of a shortest vector, Lemma 3.4 says that `T(z)` has a
rank-one eigendirection `v/lambda_1(L)` up to exponentially small error.
The paper evaluates (C) at every half-lattice point.  Nothing in (B)--(C)
requires doing so.

## 3. New primary route: periodic-Hessian basin search

Use the full-lattice samples once and search the continuous torus `R^n/L`:

1. choose a seed `z`;
2. compute an extreme eigenvector `q(z)` of the empirical `T(z)` by Lanczos;
3. ascend the spectral anisotropy using

       grad_z [q^T T(z) q]
         = 8 pi^3 E[((q.X)^2-||X||^2/n) X
                     sin(2 pi <X,z>)];                 (D)

4. at a terminal rank-one ridge, query BDD at `d q` and verify the output.

One field/gradient evaluation costs `N poly(n)`.  Equation (A) permits
`2^((delta_query-o(1))n)` independent basins while retaining total exponent
`1/2`.  The exact replacement theorem is therefore:

> **Periodic-Hessian basin theorem.**  Construct a seed distribution and a
> polynomial-evaluation ascent/continuation rule such that, for every
> lattice and every correct scale, the union of attraction basins whose
> terminal eigendirections BDD-decode to a shortest vector has probability
>
>     at least 2^(-(delta+o(1))n)
>
> for some `delta<1/2-2r`.  Prove uniform empirical stability of (C)--(D)
> along the adaptive paths from `N=2^((2r+o(1))n)` samples.

This target bypasses the random half-coset, cold syndrome folding,
constant-rank Gibbs blocks, and the Walsh transform.  Its geometric burden is
a basin-volume theorem rather than a mixing theorem.

The exact traceless field has the equivalent primal representation, up to a
positive common scalar,

    T_s(z)=traceless(sum_(y in L) (y-z)(y-z)^T
             exp(-pi||y-z||^2/s^2)).                   (E)

The audit evaluates it through the dual Fourier representation, which is
periodic term by term under truncation.  Its leading eigenvalue has an
analytic gradient, and a shortest midpoint has the desired leading edge.  The
experiment measures the fraction of random torus seeds that ascend to a
shortest nearest-neighbour edge.

The first census used 48 random starts per lattice, cutoff 2, and dimensions
two through five.  Orthogonal controls reached a shortest edge on every
start, with 13--16 objective/gradient evaluations at the median.  Generic
well-conditioned fixtures reached a shortest edge with fractions

    n=2: 0.3750,  n=3: 0.1667,  n=4: 0.0833,  n=5: 0.0833. (F)

The terminal leading eigenvector was normally almost exactly aligned with
the nearest-neighbour edge, including false longer edges.  Thus the local
oracle is doing the intended geometric job, but the cold leading-eigenvalue
landscape has many legitimate longer-edge basins.  Some longer-edge terminal
scores even exceed the chosen shortest-midpoint score at these dimensions.
The experiment therefore supports basin continuation and BDD verification;
it does not support a theorem based on a unique global maximum.

Interpreting (F) as a finite exponential rate gives numbers much larger than
the asymptotic allowance `delta_query=0.02648...`, but dimensions this small
cannot distinguish a basin fraction converging to a constant from one losing
an exponential factor.  The census is encouraging enough to retain the
route, but it is not evidence for the required sharp asymptotic bound.  The
next audit should track critical branches from a smoother field and measure
when the shortest-edge branch first separates from the longer ones.

## 4. Why this route is not already a proof

There are `2^n` half-lattice stationary points.  A local method succeeds only
if the total basin of the shortest midpoints has the lower bound above.  A
worst-case lattice may give one shortest Voronoi facet very small solid angle,
and ordinary random initialization has no known dimension-only guarantee.
Near ties also make exact global ranking delicate.

The transport techniques suggest the right strengthening: do not rely on
one cold random seed.  Start from a smoother periodic field, preserve every
certified basin as the scale changes, and allow new basins to nucleate when a
spectral gap appears.  The conserved object is now basin support and its
Hessian eigendirection, rather than particles inside one affine coset.  A
proof must bound the number or total action of live basins by
`2^((delta_query+o(1))n)`.  Equivalently, seek a *first-bifurcation theorem*:
the continuation retains at least one shortest-edge branch, and the number
of branches with no smaller certified action stays within the available
query budget.  Calling it "first" describes the intended filtration; it is
not an assumption that the shortest branch is literally the first critical
point born.

Increasing the audit parameter to `t=1/4` spends the entire `2^(n/2)` sample
budget and suppresses each fixed longer edge more strongly.  It did not cure
the finite landscape: on the generic fixtures the shortest-basin fractions
became `0.2917, 0.2083, 0.0833, 0.0625`, and false longer-edge terminal scores
remained comparable to the shortest-midpoint score.  Temperature alone is
therefore not the missing selection mechanism.

## 5. Other paths after reversal

Three secondary routes remain, but each has a known tax:

1. **Matrix-valued light-bulb search.**  Split the unknown parity into two
   halves.  Finding the pair whose matrix correlation has a rank-one outlier
   is an outlier-correlation problem on two lists of size `2^(n/2)`.  The
   planted normalized correlation is itself `2^(-rn)`, so known constant-
   correlation light-bulb intuition does not immediately give near-linear
   time.
2. **Nonlinear moment aggregation.**  Parseval can sum squares or tensor
   powers of all Hessian channels without locating the channel.  This keeps
   the physical direction but squares the `2^(-rn)` signal, reproducing the
   `2^(4rn)` population barrier unless cross-channel geometry supplies an
   exponential gain.
3. **Use the BDD preprocessing internally.**  The paper treats preprocessing
   as a query oracle plus verifier.  If its DGS advice exposes a
   `2^(n/2)`-sized family of torus seeds whose basins cover every shortest
   midpoint, basin entry is solved.  No such coverage theorem is currently
   known, but this is more targeted than constructing a cold coset sampler.

The periodic-Hessian branch-budget theorem is the cleanest next target because it
uses only resources already present before the affine-coset reduction and
spends exactly the small exponent slack that the recovery constant leaves.

## 7. Continuation experiment: the obstruction moved

The continuation audit is implemented in
`experiments/walsh_periodic_hessian_continuation.py`.  It starts at a smoother
scale, transports every distinct local maximum along a scale ladder, admits
new branches from independent probes, and compares with target-scale cold
restarts using the same number of exact field/gradient evaluations.

The first high-budget comparison (dimensions four and five, six independent
lattices per schedule) was neutral: nucleating continuation found a shortest
branch in 47 of 48 trials, and equal-evaluation cold restart also succeeded in
47 of 48.  Under a low budget, continuation was worse: 35 of 60 successes
against 43 of 60 for cold restart.  Random nucleation is therefore not the
missing algorithm.  It relocates restart work to intermediate scales without
reducing it.

This negative result is not caused by an obvious exact-oracle/SNR illusion.
Every ascent evaluation now records its leading score and top eigengap.  In a
new continuation panel with four lattices in each of dimensions four, five,
and six and start scales 0.15 and 0.18, all 24 nucleating runs retained a
shortest branch.  Along the best shortest lineage, the worst observed
bottleneck was 0.172 of the same-scale shortest-midpoint score and 0.0513 of
its eigengap.  In an independent 24-start cold panel through dimension six,
the corresponding minima on successful paths were 0.0891 and 0.00953.  These
finite ratios do not prove empirical stability, but they give no evidence of
an exponential signal collapse before the shortest ridge is reached.

## 8. A branch census isolates a sharper conjecture

`experiments/walsh_periodic_hessian_branch_census.py` replaces random
nucleation by a nested scrambled-Sobol census.  It clusters converged maxima,
transports the complete discovered catalog to the next scale, and records
which current branches were not carried from the preceding catalog.  Counts
are lower bounds, so probe-budget saturation is essential.

For the fixed generic well-conditioned fixtures, the census stabilized as
follows.  At 512 probes the branch counts at scale 0.12 and target scale
0.23675858 were respectively

    n=2: 2 -> 2,   n=3: 3 -> 5,
    n=4: 5 -> 9,   n=5: 7 -> 5.                    (I)

The target shortest branch was carried from the smooth catalog in every
case.  At the lowest nearly accessible scale `t=0.10`, a 256-probe panel had
`n` discovered branches for dimensions three through five; again its
shortest branch survived to the target.  The target shortest-basin fractions
were 0.516, 0.176, 0.184, and 0.0703 in dimensions two through five.  In the
dimension-five fixture the five `t=0.10` branches were precisely the five
basis-edge families, with the shortest one ranked fifth by Hessian score.
Thus global score selection fails, but catalog continuation succeeds on this
family.

The scale `t=0.10` is not free.  Reweighting target samples at
`r=0.23675858` to scale `t` costs the Renyi-2 exponent

    a_r(t) = 2t + (1/2) log_2[r^2/(t(2r-t))].          (J)

The equation `a_r(t)=1/2` has the formal lower root

    t_access = 0.0978695099... .                       (K)

but this root is not uniformly justified by the available theta-mass bound.
The importance second moment also requires

    q(t,r)=tr/(2r-t)>t0/2.

For example, `t=0.16` gives `q=0.1208271...>t0/2` and
`a_r(t)=0.4001086...<1/2`.  We call this a rigorously controlled accessible
scale.  The `t=0.10` census remains exploratory evidence only.

It is not yet a worst-case theorem.  The fixtures are near orthogonal; at
`t=0.10` their initial branches are basis-edge families.  General lattices
can have many Voronoi-relevant directions, and the census does not exclude
exponentially small basins or exponentially many branches.  Nor does an
"uncarried" finite-census branch prove a topological birth: the preceding
probe set may merely have missed it.

The remaining proof target can now be stated without random-restart language:

> **Accessible-scale branch-catalog theorem.**  At some
> a rigorously controlled `t` satisfying `q(t,r)>t0/2`, construct from
> target-width DGS samples a catalog of
> `2^o(n)` certified periodic-Hessian branches.  For every lattice and correct
> scale, at least one catalog branch continues to a target-scale ridge whose
> leading direction BDD-decodes to a shortest vector.  Along one such lineage
> the normalized score and eigengap are `2^-o(n)`, and the number of certified
> bifurcations created during continuation is `2^o(n)`.

Together with (C)--(D), this theorem would make all field evaluations and
branch transports polynomial per DGS sample and incur only a `2^o(n)` lifting
detail.  The immediate falsification test is a stress census on reduced bases
with many non-basis Voronoi-relevant vectors.  If (I) becomes exponential,
the route dies; if the *certified high-gap* subcatalog remains subexponential,
that smaller object is the likely kernel of the proof.

## 9. Half-grid stationarity removes branch transport

The branch formulation above is stronger than necessary.  Write a torus
point as `z=Bc` and a dual vector as `B^(-T)k`, with `k in Z^n`.  Every entry
of the exact Fourier Hessian has the form

    sum_(k in Z^n) M_k(t) cos(2 pi k.c).               (L)

At a half-grid point `c=theta/2`, every first derivative vanishes because

    sin(2 pi k.(theta/2)) = sin(pi k.theta) = 0.       (M)

This is simultaneous for all matrix entries and all scales.  Thus a lattice
midpoint does not trace a moving critical branch: it is the same stationary
torus point throughout the scale homotopy.  Numerical branch matching was
tracking an object that algebraically cannot move.

This gives an exact snapping reduction.  Suppose accessible-scale ascent
returns `c` with

    ||2c-theta||_infinity < 1/2.                       (N)

Coordinate rounding recovers `theta mod 2`.  Evaluate the target-scale
Hessian at the exact point `B theta/2`.  If
`theta = coeff_B(v) mod 2` for a shortest vector `v`, then

    B theta/2 = v/2 mod L,                             (O)

so periodicity makes this exactly the midpoint Hessian used by the recovery
lemma.  Its leading direction can be sent directly to BDD and verified.  No
scale continuation, bifurcation count, or nearest-lattice-point computation
is required.

The adversarial experiments support the distinction.  A `D_(n-1)` shell
with `m(m-1)` almost-short root directions produced 7, 13, and 21 target
branches in dimensions four, five, and six, but the unique shortest needle
retained 54--68 percent of the observed basin and score rank one.  Adding a
nonorthogonal tilt did not change that conclusion.  Hence many competing
branches alone do not obstruct parity capture.

A search over 64 reduced anisotropic four-dimensional lattices initially
produced three apparent continuation failures.  Denser probing removed two;
raising the Fourier cutoff from three to four removed the third.  With cutoff
four, all smooth optima lay within `2.26e-5` of the half-grid.  A 64-probe
census missed one shortest parity, but a 512-probe rerun found it with basin
fraction `0.0234375`.  This is a useful warning: finite branch matching and
finite Fourier cubes can manufacture topological failures that the parity
invariant does not have.

The corrected theorem target is therefore:

> **Accessible-scale shortest-parity theorem.**  Fix a rigorously controlled
> scale such as `t=0.16`.  Construct a seed distribution and a
> `2^o(n)`-evaluation ascent rule for the reweighted periodic-Hessian field at
> reweighted periodic-Hessian field such that, for every lattice, with
> `2^-o(n)` success probability an endpoint satisfies (N) for the parity of a
> shortest vector.  Prove uniform empirical stability of the score and
> gradient along these adaptive paths from `2^((1/2+o(1))n)` target samples.

Equivalently, it suffices to prove that the total attraction mass of shortest
parity midpoints is `2^-o(n)` under an explicitly samplable seed law.  Once
that happens, (L)--(O) turn the endpoint into the target midpoint with no
lifting-detail loss beyond `2^o(n)`.  This is now the sole geometric
existence clause.  A uniform random torus seed is only one candidate; a proof
may use DGS/BDD preprocessing to place a lattice-adapted seed law near the
half-grid basins.

## 10. The catalog bound is false even when parity capture is easy

The simplex-cancellation stress family makes the distinction explicit.  Its
basis Gram matrix has diagonal one and a common negative off-diagonal chosen
so that the all-ones coefficient vector has length `0.97`; it is the unique
shortest pair, while every basis column has length one.  Its condition number
is only `Theta(sqrt(n))`, so the construction is not hiding an ill-conditioned
coordinate system.

At scale `t=0.10`, a 256-probe cutoff-two census found

    n=3: 8 branches,  n=4: 15,  n=5: 30,  n=6: 50.   (P)

The first three counts are already essentially all `2^n` parity addresses.
Therefore a theorem that explicitly retains every smooth local maximum cannot
have a `2^o(n)` branch budget in general.  The branch-catalog target in
Section 8 is rejected.

Nevertheless the shortest all-ones parity had observed basin fractions

    0.2109, 0.1289, 0.0859, 0.0977                     (Q)

in dimensions three through six.  Exponential branch count does not imply an
exponentially small shortest basin: the Hessian flow can allocate very unequal
basin masses.  This is why (P) rejects catalog enumeration but does not reject
the shortest-parity theorem.

The natural attempted no-go via many minimal vectors has a precise geometric
form and does not require transitivity.  Group `K` oriented shortest facets
into antipodal pairs.  Since the total direct solid-angle mass and total
cone-volume mass are each at most one, some pair has combined weight at most
`4/K`.  Applying
T_tau=I-tau(v/lambda)(v/lambda)^T makes one antipodal pair uniquely shortest;
the shortest-facet inball guarantees persistence, and the two weights vary
continuously.  Hence the perturbed lattice has

    p_min+q_min <= (4+o(1))/K.

This does not by itself refute the joint-core target: one must also prove that
the perturbed family survives the syndrome, spectral, exact-factor, and
minimum-cut escapes.  It does show that any proof using only local
shortest-facet geometry is capped by the relevant lattice kissing exponent.
The previously advertised large code-derived lattice exponents were
invalidated in arXiv:2410.16660; a later correction preprint,
arXiv:2411.07371, claims a much smaller positive exponent.  The perturbation
lemma is independent of that unsettled construction history.

Finally, a basis-update descent does not improve uniform-torus basin mass by
itself.  Replacing `B` by `BU` for unimodular `U` only applies a
measure-preserving torus automorphism to coefficient space.  It may improve a
finite deterministic probe set, but it cannot turn an exponentially small
physical basin into a subexponential one.  A useful support-chain descent must
therefore alter the seed law or combine parity information from several
ridges; ordinary basis reduction is not the missing kernel.

The log-theta line detector imposes an additional no-go.  At scale
`s=1/xi_t`, Hhan's weighted shell bound suppresses the non-colliding tail on
a homothetic shortest-facet core only if `eta t>t_0`.  Because `t<1/4`, this
retains at most `2^(-(3.7540+o(1))n)` of its area.  On an arbitrary
tail-suppressed facet subset visible above the `M^-1/2` empirical floor, with
`M=2^(mn)` and `alpha=m/2`, every point `x=v/2+y` obeys

    ||y||^2/lambda^2 <= (alpha/t-1)/4+O(log n/n).

The facet inball converts this to accessible area fraction at most

    [sqrt(3(alpha/t-1)+O(log n/n))]^(n-1).

At the complete `m=1/2` budget and the required `t>t_0`, this is at most
`2^(-(1.0289+o(1))n)`.  Retaining exponent `delta=0.0264828` would require
`m>=0.61169...`.  Thus direct-plus-cone mass is not sufficient for the
empirical log-theta scan, even if its geometric inequality is true.

The viable proof fork is now only:

1. prove `2^-o(n)` shortest-parity attraction mass at an accessible scale
   with directly observable endpoints;
2. prove response-weighted shortest-collision mass together with a
   variance-sensitive estimator beating the global `M^-1/2` floor;
3. construct a lattice-adapted seed law from DGS/BDD advice with that mass; or
4. combine a subexponential number of discovered parity addresses by a
   bias-linear support transport that crosses local length barriers.

The first is the clean geometric clause.  The latter two are information-
transport replacements if uniform attraction is false.

The explicit gap-free candidate law, the uniform empirical derivative
proposition, and the exact statement of the remaining basin conjecture are in
`notes/WALSH_ACCESSIBLE_SEED_LAW.md`.

## 6. Two generic shortcuts are exactly taxed

The reverse view also gives short no-go calculations for two tempting
alternatives.

First, direct BKW-style weak-parity decoding destroys the sample budget after
one merge.  The full-lattice Hessian label has correlation

    epsilon = 2^(-(r+o(1))n).

XORing two examples to eliminate a parity block replaces it by
`epsilon^2`.  Detecting that correlation needs

    epsilon^(-4) = 2^((4r+o(1))n) = 2^(0.9470...n),     (G)

already above the target before accounting for collision work.  Any useful
parity decoder must preserve the Hessian bias linearly; ordinary BKW does not.

Second, replacing a hard syndrome coset by a target-independent soft window
cannot improve the exponent.  For a window `w:F_2^h -> C`, normalize
`sum_j |w(j)|^2=1`.  Parseval gives

    average_alpha |what(alpha)|^2 = 2^(-h).             (H)

The missing head frequency `alpha_star` is uniform.  A chirp window avoids
rare samples but attenuates the target by `2^(-h/2)`; a delta window preserves
the coefficient but accepts only a `2^(-h)` fraction.  Both pay the same
`2^h` variance factor.  Multiple oblivious windows form a frame and conserve
this energy.  A successful window must therefore be state-dependent and
learn something about the target physical direction--again a transport, not
a static taper.

After these eliminations the meaningful fork is narrow:

1. prove a branch-budget theorem for the continuous periodic-Hessian field;
2. construct a state-dependent, bias-linear address transport; or
3. exploit internal structure of the BDD/DGS preprocessing to seed one of
   the certified branches directly.
