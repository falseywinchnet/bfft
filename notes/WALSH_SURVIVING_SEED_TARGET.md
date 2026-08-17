# Walsh recovery: the surviving lattice-adapted seed target

Date: 2026-08-13.

This note starts after the anisotropic product obstruction to the universal
Haar-seeded soft-Hessian law.  The obstruction does not survive a
lattice-adapted parity sieve.  Two exact coefficient-space lemmas remove it,
and a low-dimensional spectral-tube construction also handles the symmetric
all-ones cancellation family.  The remaining target is consequently a dense
joint core: high spectral-cover entropy, high exact-factor transport width,
and no accessible submodular min-cut representation, not arbitrary anisotropy.

Companions:

- `experiments/walsh_spectral_parity_sieve.py`
- `experiments/out/walsh_spectral_parity_sieve.json`
- `tests/walsh_spectral_parity_sieve_test.py`
- `experiments/walsh_gram_gdl_sieve.py`
- `experiments/out/walsh_gram_gdl_sieve.json`
- `tests/walsh_gram_gdl_sieve_test.py`
- `experiments/walsh_dense_core_transport.py`
- `experiments/out/walsh_dense_core_transport.json`
- `tests/walsh_dense_core_transport_test.py`
- `experiments/walsh_voronoi_causal_transport.py`
- `experiments/out/walsh_voronoi_causal_transport.json`
- `tests/walsh_voronoi_causal_transport_test.py`
- `experiments/walsh_obtuse_superbase_sieve.py`
- `experiments/out/walsh_obtuse_superbase_sieve.json`
- `tests/walsh_obtuse_superbase_sieve_test.py`

## 1. Short dual vectors are exact syndrome certificates

Let `L=B Z^n`, let `lambda=lambda_1(L)`, and suppose the length guess obeys

    lambda <= d <= (1+1/n) lambda.

Write a shortest vector as `v=Bz`, `z in Z^n`, and its parity as
`theta=z mod 2`.  For `k in Z^n`, put

    y_k = B^(-T) k in L*.

Then `<y_k,v>=k.z` is an integer.  This gives the following exact statement.

> **Short-dual annihilator lemma.** If `d ||y_k|| < 1`, then every lattice
> vector `v` of norm at most `d` satisfies
>
>     k.z = 0,
>
> and in particular every shortest parity satisfies
>
>     k.theta = 0 mod 2.                                  (1)

The proof is one line:

    |k.z| = |<y_k,v>| <= ||y_k|| ||v|| <= d ||y_k|| < 1,

and the left side is an integer.  Any collection of verified short dual
vectors therefore gives a binary syndrome matrix `K`; the shortest parity is
in `ker_F2 K`.  Completeness of the short-dual search is unnecessary.  Every
found row is a safe constraint, and its norm can be checked exactly for a
rational input basis.  The algorithm need only test a polynomial or
`2^(n/2+o(n))`-budgeted candidate list (for example a reduced dual basis and
its certified short combinations); exhaustive short-dual enumeration is not
being assumed.

For

    L_n = diag(1,2^n,...,2^n) Z^n,

the rows `k=e_2,...,e_n` have dual norms `2^-n`.  They force
`theta_2=...=theta_n=0`.  Since a shortest parity is nonzero, the sole
remaining candidate is `e_1`.  Thus the family that disproves universal Haar
ascent is solved deterministically by polynomial-time input geometry; no
exponentially small Hessian gradient is evaluated.

## 2. A basis-conditioned Hamming seed law

There is a certificate even when no individual dual row has norm below
`1/d`.

> **Coefficient Hamming lemma.** Put
>
>     A(B,d) = d^2 ||B^(-1)||_op^2,
>     q(B,d) = min(n, floor(A(B,d))).                    (2)
>
> Every shortest parity has Hamming weight at most `q(B,d)`.

Indeed,

    ||z||_2 <= ||B^(-1)||_op ||v|| <= d ||B^(-1)||_op.

Every odd coordinate of `z` is nonzero and contributes at least one to
`||z||_2^2`; hence

    wt(z mod 2) <= ||z||_2^2 <= A(B,d).                 (3)

A shortest parity cannot be zero: if `z=2w`, then the nonzero lattice vector
`Bw=v/2` is shorter.  Consequently consider the finite law

    theta uniform on {a in F_2^n \ {0}: K a=0,
                       wt(a)<=q(B,d)}                  (4)

always assigns positive, known mass to every shortest parity.  The elementary
count

    Q_q = sum_(j=1)^q binom(n,j)                        (5)

is an upper bound on the denominator in (4); the verified dual equations can
only reduce it.

The set in (4) can be constructed by filtering either the Hamming ball or
the binary kernel.  Thus its enumeration cost is at most

    E_(q,K) = min(Q_q, 2^(n-rank_F2 K)) poly(n),         (5a)

after which exact uniform sampling is immediate.  We do not assume an
output-sensitive enumerator for arbitrary low-weight codewords.

This is already an unconditional solvable-class theorem.  If `q=o(n)`, then
`log_2 Q_q=o(n)`.  Enumerate the candidates in (4), evaluate the
target-scale midpoint Hessian at `B theta/2`, invoke preprocessing BDD, verify
every output, and return the shortest verified vector.  The target estimator
cost is

    2^((2r+o(1))n) Q_q.

Thus the total time and space stay `2^(n/2+o(n))`.  More generally, if
`q/n -> rho`, the exponent paid by the candidate set is at most `H_2(rho)`.
At `r=0.23675858...`, the unused exponent is

    delta_query = 1/2-2r = 0.02648284...,

so the half-exponential ledger still closes whenever

    H_2(rho) < delta_query,
    rho < 0.00264781684... .                            (6)

The rectangular family has `q=1`, giving `Q_q=n`; the dual equations sharpen
this to one candidate.

More generally it is enough that `E_(q,K)<=2^((1/2+o(1))n)` for filtering and
that the retained set in (4) has exponent below `delta_query` for target
midpoint evaluation.

## 3. The Hamming sieve exposes the real hard family

The simplex-cancellation basis has unit diagonal Gram entries and common
negative off-diagonal chosen so that the all-ones coefficient vector has
length `0.97`.  Its unique shortest parity is `(1,...,1)`.  Here

    d^2 ||B^(-1)||_op^2 = n+o(n),

there are no short-dual annihilator rows in the audited coefficient cube, and
the Hamming sieve correctly retains every nonzero parity.  This is not a
failure of the lemma.  It identifies the surviving phenomenon: the shortest
vector is a dense integer cancellation in a low-energy coefficient direction.

That direction is visible before any periodic-Gaussian query.  If
`G=B^T B`, the normalized all-ones vector is the unique minimum eigenvector of
`G`, with eigenvalue `0.97^2/n`.  Scaling that eigenvector by `sqrt(n)` and
rounding recovers the shortest coefficient vector exactly.  This suggests
replacing local torus transport by a finite support chain through the
low-eigenvalue coefficient ellipsoid.

## 4. A proved spectral-tube parity cover

The support-chain idea has an exact geometric formulation.  Let `U` be a
`k`-dimensional subspace of coefficient space and let `P_U` be its orthogonal
projector.  Suppose every point in the length ellipsoid

    E_d = {z in R^n: z^T G z <= d^2}                    (7)

obeys

    ||z|| <= R,
    ||(I-P_U)z|| <= C.                                  (8)

For the span of the first `k` eigenvectors of `G`, one may take

    R = d/sqrt(lambda_1(G)),
    C = d/sqrt(lambda_(k+1)(G)),                         (9)

with `C=0` when `k=n`.

Intersect the radius-`R` ball in `U` with the coordinate rounding
hyperplanes

    x_i = m+1/2.

There are at most

    N <= n(2 ceil(R)+2)                                 (10)

relevant hyperplanes.  Their `k`-dimensional arrangement, including all
tie faces, has at most

    2^k sum_(j=0)^k binom(N,j)                          (11)

rounding labels.

Now take an integer `z in E_d`, put `x=P_U z`, and let `u=round(x)`.  The
integer residual `w=z-u` has two useful bounds.  If `w_i != 0`, then `z_i`
is not the nearest integer to `x_i`, so

    |z_i-x_i| >= 1/2.

Equation (8) gives

    wt(w) <= floor(4C^2),
    |w_i| <= C+1/2.                                    (12)

We have proved the following.

> **Spectral-tube parity-cover theorem.** Every shortest parity belongs to
> the set
>
>     {round(x)+w mod 2:
>        x in U, ||x||<=R,
>        wt(w)<=floor(4C^2), |w_i|<=ceil(C+1/2)}.       (13)
>
> Its cardinality is at most
>
>     2^k sum_(j=0)^k binom(N,j)
>       sum_(s=0)^floor(4C^2)
>         binom(n,s) (2 ceil(C+1/2))^s.                (14)

This proof is only an arrangement count plus the stable lifting detail (12):
`round(x)` is the coarse response transported on `U`, and `w` is the sparse
detail needed to reconstruct the integer coefficient vector.  It is the same
coarse-state/lifting-detail separation that was useful in the Meyer and
segmenter work, now applied to parity recovery.

A rational approximation to `U` can be certified by directly enlarging `R`
and `C` in (8).  Incremental hyperplane-arrangement enumeration then constructs
(13) in time polynomial in the displayed integer-label catalog and the input
bit length.  Boundary uncertainty is handled by retaining both adjacent
rounding labels, already covered by the factor `2^k` in (11).  This is an
output-sensitive statement: if `R` is enormous, scanning all crossed integer
cells is not silently treated as polynomial in `log R`.

The geometric cover is subexponential whenever

    k log(n(R+1)) + C^2 log(n(C+1)) = o(n).             (15)

For the full algorithmic corollary, the left side must be
`o(n)+O(log size(B))` in the uniform input model, so the catalog costs only
`2^o(n) poly(size(B))`.  The cover is also compatible with the smaller
exponential allowance: replace `o(n)` in (15) by any constant below
`delta_query n` after keeping the explicit binomial entropies in (14).

For the simplex-cancellation family, choose `U` to be the minimum Gram
eigenspace.  Then `k=1`, `R=Theta(sqrt(n))`, and `C<1`; (14) is polynomial.
For the rectangular family, the same construction has `k=1`, `R=O(1)`, and
`C=2^-Omega(n)`.  Thus one theorem covers both obstructions, while the
short-dual equations make the rectangular case even sharper.

## 5. What the finite audit says

The experiment checks the exact Hamming and short-dual claims on dimensions
three through seven.  Its short-dual cube is deliberately exhaustive only at
these small dimensions; it is not the proposed asymptotic search procedure.

- On every rectangular fixture, the Hamming radius is one, the enumerated
  short-dual rank is `n-1`, and the combined sieve has exactly one candidate.
- On the simplex-cancellation fixtures, the Hamming sieve has `2^n-1`
  candidates and no short-dual row; the minimum spectral ray recovers the
  all-ones parity in every dimension.
- On the generic fixtures, the combined proved sieve has between three and
  ten candidates in dimensions three through seven.
- The spectral-ray diagnostic finds the shortest parity on 24 of the 25
  audited family/dimension pairs.  Its one failure is useful: a union of
  eigenvector rays is a heuristic, whereas the spectral-tube cover (13) is
  the proved object and contains every audited shortest vector.

All assertions about inclusion are tested independently against exact finite
coefficient-cube shortest-vector enumeration.  The experiment is evidence
about the size of the covers, not an asymptotic substitute for (14).

## 6. The first surviving target was a dense spectral core

The spectral-cover reduction leaves lattices for which the displayed basis
and every *specified, efficiently enumerable* alternative description have
large cover entropy.  Concretely, the unresolved regime has no verified dual
syndrome family of rank `n-o(n)` and no coefficient subspace `U` satisfying
(15).  This is deliberately a statement about the actual entropy in (14),
not merely the dimension of a low-energy eigenspace: the arrangement term is
of order `k log(n(R+1)/k)`, so large cover entropy is not equivalent to
`k=Omega(n)` unless a suitable bound on the coefficient radius `R` is also
imposed.

At this stage the next theorem appeared to belong only to this residual
class:

> **Dense-core shortest-parity transport target.** After applying the exact
> short-dual syndrome quotient and the spectral-tube cover, construct from the
> common DGS/BDD advice a distribution on the remaining parity cells such
> that the total robust mass of shortest parities is `2^-o(n)`; or prove the
> weaker exponent below `0.02648284n`.  The construction may use projected
> soft-Hessian ascent, but its guarantee is required only when the residual
> coefficient ellipsoid has linear spectral dimension.

This formulation removes the now-solved long-direction localization problem.
It also avoids asking a local field to discover information already present
in the input basis.  In the residual dense core, one of two additional facts
should be exploitable:

1. the large low-energy spectral dimension forces many parity cells to carry
   comparable short-vector theta mass, giving a total-basin lower bound; or
2. the DGS panel supplies a nonlocal parity bias across that core that can be
   transported without squaring the Hessian signal.

Proving either statement would complete the lattice-adapted seed law.  A new
counterexample must now defeat the dual quotient, the sparse coefficient
sieve, and the spectral support chain simultaneously; the rectangular family
does none of these.

## 7. Update: the stationary law exists; entrance is the surviving clause

`notes/WALSH_VERIFIED_GIBBS_SEED.md` constructs a verifier-weighted Gibbs law
from the common target DGS panel and preprocessing BDD oracle.  Its pushforward
through exact BDD verification has shortest-parity mass at least
`1-2^(-2n)`.  For any block of at most `(1/2-o(1))n` parity bits, every exact
heat-bath conditional is computable within the half-exponential budget by one
binned matrix-valued Walsh transform of the common panel.

This resolves the stationary-mass part of the displayed target, but not the
algorithmic initialization.  A planted-parity control has the same
overwhelming stationary mass while a half-block update sees the target only
when its frozen complement is already correct.  The remaining theorem must
therefore prove `2^-o(n)` entrance in `2^o(n)` block updates using a
lattice-specific relation between neighboring parity cells and their physical
BDD outputs.  Stationary mass, Walsh sparsity, and signal amplitude alone do
not imply that entrance bound.

## 8. Update: one-shot cell transport and intrinsic boundary flux

`notes/WALSH_GAUSSIAN_CELL_SEED.md` tests whether entrance can be removed
altogether.  Physical isotropic Gaussian sampling followed by coefficient
rounding has an exact translated-cell inequality and gives
`2^(-o(n))` shortest-parity mass whenever

    lambda_1(L) max_i ||b_i^*||=o(sqrt(n/ln n)).

This is a proved initialized law on a substantial basis-reduced class.  It
is not universal: a rearrangement/volume argument on dense unimodular
Conway--Thompson lattices forces the median-cell certificate to lose at least
`1/(2 ln 2)-o(1)=0.721347...-o(1)` bits per dimension.  Rejecting all
zero-parity samples at a colder scale repairs sparse basis-facet targets but
empirically worsens dense cancellation targets.

Replacing the basis cell by the origin Voronoi cell removes that coordinate
artifact.  A random ray has a computable ideal first label

    argmin_w ||w||^2/(2<u,w>),

and the finite audit finds polynomial-looking total shortest-facet mass on
the simplex and needle families.  Every shortest vector universally owns a
`pi/6` angular cap, but that proves only `2^(-(1+o(1))n)` mass.  The live
geometric target is therefore a dense-core dichotomy improving this cap loss
below `0.02648284n`, followed by a Hessian ray-continuation implementation
that exposes the boundary label without presupposing CVP.

The boundary label can in fact be removed from the implementation.  An
absolute target-Hessian scan on a polynomial grid of radii `[0.4d,3d]`,
followed by a small separated peak catalog and gap-free centering, is a
directly samplable law.  The cap `|<u,v/lambda>|>=1/6` has exponent
`0.0203209922...`, and its shortest-vector bisector lies inside the scanned
interval.  Hence the new surviving theorem is a robust bisector-to-ridge
capture lemma.  If proved, its total target-query exponent is
`2r+0.0203209922...=0.493838153...<1/2`.  The finite dimension-five stress
panel found success masses `0.156`--`0.260` on six generic bases and
`0.115`--`0.854` on six rotated-anisotropic bases; the geometry, not the
sampler or empirical oracle, remains unproved.

## 9. Chip transport gives a second exact escape: Gram-factor GDL

The seventeen-square calculation suggests that the relevant size is not the
number of legal global configurations.  Its exact transport algorithm pays
the largest table induced by eliminating the local factor graph and then
recovers one realizing packet by a backward preimage pass.  The same statement
applies to the coefficient energy of a lattice.

> **Lemma (weighted Gram-GDL sieve).** Let `B in Q^{n x n}` be nonsingular,
> let `L=B Z^n`, and let `d >= lambda_1(L)`.  Put
>
>     R_i = floor(d ||row_i(B^{-1})||_2),
>     D_i = {-R_i,...,R_i}.
>
> Let `G=B^T B`, and join `i` and `j` when `G_ij != 0`.  For an elimination
> order `pi`, let `K_i(pi)` be variable `i` together with its remaining
> neighbors just before it is eliminated, including the fill edges generated
> earlier, and define
>
>     W_pi(B,d) = max_i sum_{j in K_i(pi)} log_2(2R_j+1).       (16)
>
> Then exact SVP on `L` is solvable in
>
>     2^{W_pi(B,d)} poly(n,size(B),log(d+1))                  (17)
>
> space and in `n` times this quantity in time.  A minimizing coefficient
> vector, not merely its length, is returned by a backward GDL pass.

**Proof.** If `v=Bz` and `||v||_2 <= d`, then

    |z_i| = |<row_i(B^{-1}),v>| <= d ||row_i(B^{-1})||_2,

so every shortest coefficient vector lies in `D_1 x ... x D_n`.  On this
finite product,

    ||Bz||_2^2
      = sum_i G_ii z_i^2 + sum_{i<j} 2G_ij z_i z_j             (18)

is a unary-and-pairwise factor graph with precisely the Gram interactions.
Exact min-sum elimination in the order `pi` forms a table only on the current
bag `K_i(pi)`.  Its number of entries is at most
`prod_{j in K_i(pi)} |D_j| <= 2^{W_pi(B,d)}`.  Exact rational arithmetic has
polynomial bit growth under addition, multiplication, comparison, and taking
minima of the displayed energies.  Store an attaining value of the eliminated
variable for each surviving table entry; the reverse pass reconstructs one
global minimizer.

The all-zero assignment is excluded by running the elimination once for each
`p in {1,...,n}` with the pivot domain replaced by `D_p` minus `{0}`, and
taking the best of the `n` answers.  Every nonzero vector occurs in at least
one such run.  The shortest vector occurs in the bounded product, while no
assignment can have energy below its energy, so the best returned vector is
globally shortest.  This proves (17).  `square`

The Gram expansion is only the first convenient factorization.  The exact
chip invariant is more general.  Suppose a finite collection of original and
auxiliary variables has domains `E_x`, local objective factors and exact
constraint factors (with value zero or infinity) represent `||Bz||_2^2`, and
projecting the finite-objective assignments onto `z` gives exactly
`D_1 x ... x D_n`.  For a tree decomposition `T` of this factor graph, put

    J(T) = max_{bag A in T} sum_{x in A} log_2 |E_x|.          (19)

The same min-sum and backward-preimage proof returns a shortest vector in
`2^{J(T)} poly(input)` time and space, up to the `n` zero-exclusion runs.  In
particular, after clearing rational denominators, each row
`(Bz)_k = sum_i B_ki z_i` can be represented by a binary tree of exact partial
sums and one unary square factor.  A partial sum over `S` needs only the
integer interval

    [-sum_{i in S}|B_ki|R_i, sum_{i in S}|B_ki|R_i].

This auxiliary encoding can remain narrow when the pairwise Gram expansion
is dense.  Conversely its state intervals can be expensive, so the certified
quantity is the least displayed weighted junction width, not graph
treewidth with domain sizes suppressed.

The exact invariant is

    W_*(B,d) = min_pi W_pi(B,d),                              (20)

the heterogeneous weighted induced width of the displayed Gram
factorization.  No claim is made that finding the best order is free.  Any
efficiently produced order with

    W_pi(B,d) <= (1/2+o(1))n                                  (21)

already gives an unconditional half-exponential algorithm, because the
backward GDL pass returns the shortest vector itself rather than merely a
parity candidate.  The stronger threshold `W_pi <=
(0.02648284-epsilon)n` lies inside the later parity-query allowance but is
not the actual factor-algorithm threshold.  Basis preprocessing can be
included by explicitly naming an enumerable family of candidate bases; the
vague phrase "every efficiently available basis" is not a mathematical
invariant.

The audit `experiments/walsh_gram_gdl_sieve.py` computes the coordinate
domains and a weighted min-fill upper bound.  Its floating zero test is only a
fixture diagnostic; (16) uses exact Gram zeros.  In dimensions three through
six, the un-tilted needle-D fixtures expose the intended phenomenon: at
dimension six the old spectral/Hamming cover costs `5.977` bits, whereas the
displayed Gram order costs `4.644` bits.  Generic and simplex fixtures have a
dense Gram graph in their displayed bases, so this escape is complementary to
the earlier cover, not a replacement for it.

There is a second exact factor escape that does not require low width.  Let
`b_1,...,b_(n+1)` be a superbase: the first `n` vectors are a lattice basis,
their total sum is zero, and

    <b_i,b_j> <= 0       for i != j.                            (21a)

Put `w_ij = -<b_i,b_j> >= 0`.  For any integer coefficients `c_i`, defined
modulo adding a common integer because the superbase sums to zero,

    ||sum_i c_i b_i||^2 = sum_{i<j} w_ij(c_i-c_j)^2.            (21b)

Normalize `min_i c_i=0`, and set `S_t={i:c_i>=t}`.  Since an integer `d`
satisfies `d^2>=|d|`,

    ||sum_i c_i b_i||^2
      >= sum_t sum_{i<j}w_ij|1_(i in S_t)-1_(j in S_t)|.       (21c)

Every term on the right is the capacity of a nontrivial cut.  Conversely the
subset vector `sum_(i in S)b_i` has squared norm exactly that cut capacity.
Therefore a global minimum cut returns a globally shortest lattice vector.
This is the obtuse-superbase/min-cut theorem for lattices of Voronoi's first
kind, proved here directly from the graph Laplacian identity; see also
[McKilliam--Grant](https://arxiv.org/abs/1201.5154).

The displayed superbase is obtained by adjoining `-sum_i b_i` to a basis and
its obtuseness is exactly checkable for rational input.  The audit
`experiments/walsh_obtuse_superbase_sieve.py` includes a self-contained
Stoer--Wagner implementation.  It recognizes the rectangular and
simplex-cancellation fixtures through dimension six and returns the enumerated
shortest length in every recognized case.  In particular, the exponential
subset family in the simplex fan is submodular and polynomial-time solvable;
it is not evidence about the surviving joint core.

## 10. What information survives in boundary transport

The other chip lesson is to inspect relations between legal states rather
than only the states themselves.  For coefficient parities `theta`,

    theta(w_1-w_2) = theta(w_1) xor theta(w_2).               (22)

Thus two Voronoi first-exit labels, or two labels on neighboring rays, induce
a parity that can be sent directly to the target Hessian/BDD recovery step.
The experiment `experiments/walsh_dense_core_transport.py` measures the ideal
finite-cube version of this law.

There is an exact low-rank alternative before asking for mass on one
difference.  Let `A` be the set of parities whose exact Voronoi first-exit
regions have positive spherical measure.  Every shortest vector is Voronoi
relevant, hence every shortest parity belongs to `A`.  Indeed, if a vector
`w` in the coset `v+2L`, other than `v` and `-v`, had `||w||<=||v||=lambda`,
then the two nonzero lattice vectors `(v-w)/2` and `(v+w)/2` would have squared
norms summing to `(||v||^2+||w||^2)/2<=lambda^2`, although minimality makes
their sum at least `2lambda^2`.  Thus `v` and `-v` are the unique shortest
vectors in their mod-`2L` coset, the Voronoi relevance criterion.  Fix `a in
A` and put

    H_A = span_F2 {b xor a : b in A}.                          (23)

Then

    A subset a+H_A,    shortest parities subset a+H_A.        (24)

This proves the **boundary affine-cover lemma**: if `dim H_A=r`, exhaustive
knowledge of the active labels gives a shortest-parity cover of size `2^r`.
The vectors generating `H_A` are exactly transports from the base label `a`.
Thus low transport rank is a cover and only high transport rank can be a
geometric obstruction.  This is an existence statement until the active
label set is exposed or an outer affine hull is certified.  A finite random
sample supplies an inner span and therefore only a lower bound on `r`; it
cannot certify (24) for unseen labels.

The dimension-five census also rules out affine rank alone as the desired
lower bound.  The sampled active sets for the simplex, both generic, both
adversarial, and needle fixtures all had full affine rank five.  Nevertheless
the needle's shortest pair-XOR mass was exactly zero.  Rank records which
transports are generated after arbitrary compositions; it does not lower-bound
the occupation of a shortest transport in one accessible step.

At dimension five, an angular displacement of `0.1` radians gave conditional
shortest-XOR masses between `0.0778` and `0.1429` on the simplex, generic, and
adversarial fixtures.  This is genuine shortest-vector information carried by
a boundary transition.  It is not a universal theorem: the rectangular and
needle fixtures had zero shortest-XOR mass, because their shortest parity was
not a difference of two active shell parities.  Those same fixtures have high
direct first-exit mass (`0.9630` and `0.1572`, respectively), which points to a
direct-flux-or-transition alternative rather than an XOR-only law.

There is a second implementation gap.  The ideal experiment knows both
Voronoi labels.  A realizable Hessian procedure must recover their physical
BDD representatives robustly before (22) becomes an accessible parity.  The
identity (22) does not itself provide that recovery.

## 11. The corrected dense-core theorem

The remaining object is therefore not merely a dense spectral core.  It is a
**joint dense core** at allowance `delta` if, after every explicitly allowed
short-dual quotient and basis preprocessing step,

1. every certified Hamming/spectral-tube parity cover has log-size greater
   than `delta n`, including every certified boundary affine cover; and
2. every certified Gram elimination order and auxiliary exact factorization
   has weighted width greater than `delta n` after accounting for its
   construction cost, and no certified submodular/obtuse-superbase
   representation gives a polynomial minimum-cut solution.

Only this intersection needs a geometric seed theorem.  A form aligned with
the two transport mechanisms is:

> **Joint-core flux-or-transition target.** There is a polynomial-size,
> samplable catalog of radial probes and short physical line segments such
> that, on every joint dense core, the total robust mass of trials for which
> either (i) one radial probe centers to a shortest parity or (ii) one segment
> crosses a translated Voronoi facet carrying a shortest parity is at least
> `2^{-(delta+o(1))n}`, for some `delta<0.02648284`.  The centering and
> collision tests must remain stable under the common empirical target panel
> of size `2^{(1/2+o(1))n}`.

This is strictly narrower than the previous dense-core target.  Low-width
exact factor transport is now solved by the junction-width lemma; low
spectral entropy is solved by the existing cover; direct radial flux and
transition flux may be pooled rather than forced into one mechanism.  What
remains to prove is a lower bound on the relevant occupation/congestion
functional in the simultaneous high-cover, high-width regime.  Counting
boundary cells is insufficient: the desired certificate must price how much
spherical or Gaussian flux they carry, just as the chip transport proof prices
induced table width instead of the raw number of configurations.

## 12. The noncausal neighboring-fan kernel

The occupation functional can be made intrinsic.  The finitely many
Voronoi-relevant vectors partition `S^{n-1}` into spherical polyhedral
first-exit cells.  Give a cell the coefficient parity of its vector.  For a
parity `theta`, let `Sigma_theta` be the union of regular codimension-one
interfaces whose two cell labels have XOR `theta`, counting each geometric
interface once, and put

    P_theta = H^{n-2}(Sigma_theta).

Draw `u` uniformly on `S^{n-1}`, draw `tau` uniformly on the unit sphere in
the tangent space at `u`, and set

    u_alpha = cos(alpha)u + sin(alpha)tau.

If `F_alpha(theta)` is the probability that the two first-exit labels have
XOR `theta`, then

> **Lemma (spherical boundary-transport kernel).** For every nonzero
> `theta`, away from the immaterial finite union of degenerate interfaces,
>
>     lim_{alpha -> 0+} F_alpha(theta)/alpha
>       = c_n P_theta / H^{n-1}(S^{n-1}),                    (25)
>
> where
>
>     c_n = Gamma((n-1)/2)/(sqrt(pi) Gamma(n/2))
>         = Theta(n^{-1/2}).                                (26)

**Proof.** Discard the codimension-two intersections of the spherical
polyhedral interfaces; their `alpha`-tubes have measure `O(alpha^2)`.  At a
regular point of an interface, use signed geodesic normal coordinate `s` and
let `nu` be its unit tangent normal.  For fixed tangent direction `tau`, the
short geodesic crosses the interface precisely on an interval of starting
coordinates of length

    alpha |<tau,nu>| + O(alpha^2).

The spherical area element is `1+O(alpha)` in this tube.  Integrating over
the interface, summing the interfaces with XOR `theta`, and dividing by the
area of `S^{n-1}` leaves the expectation of the absolute first coordinate of
a uniform point on `S^{n-2}`.  That expectation is exactly `c_n` in (26).
Dominated convergence gives (25).  `square`

For each fixed lattice, the angle loses only the explicit linear factor once
it is below that fan's geometric feature scale.  A uniform algorithmic use
with `alpha=1/poly(n)` additionally requires a polynomial robust-tube scale;
the pointwise limit (25) does not supply that uniformity.  If `Theta_min` is
the set of shortest parities and

    Phi_min = sum_{theta in Theta_min} P_theta
              / H^{n-1}(S^{n-1}),                          (27)

then the ideal direct-or-transition law has, to first order, success mass

    p_min + alpha c_n Phi_min + o(alpha),                   (28)

where `p_min` is direct first-exit shortest-parity mass.  The empirical fields
`shortest_xor_flux_per_radian` in the transport audit estimate the second
term before multiplication by `alpha`.

This identifies exactly what the neighboring-ray experiment measures.  One
might try to prove on every joint dense core that

    p_min + Phi_min >= 2^{-(delta+o(1))n}                   (29)

for some `delta<0.02648284`.  The chip/flow-cell analogy shows why this is the
wrong primary target: these interfaces separate two possible *facets of the
same origin cell*, not two first-arrival owners.  It is a runner-up quantity.
The needle audit makes the distinction concrete: the origin fan has abundant
transitions but none in the shortest Walsh channel.  The causal interface is
instead a translated Voronoi facet separating two lattice owners, treated
next.

## 13. A proved gap-dependent first-exit cap

There is a stronger direct law when the shortest line is separated from all
noncollinear competitors.  Fix an oriented shortest vector `v`, put
`a=v/lambda`, and define

    gamma(v) = min {||w||/lambda : w in L, w notin R v}.         (30)

Positive multiples of `v` never beat `v` in a first-exit comparison, so they
are correctly omitted.  Define

    a_gamma = sqrt(1-gamma^2/4),       1 <= gamma <= sqrt(2),
              1/gamma,                gamma >= sqrt(2).          (31)

> **Lemma (gap-dependent winner cap).** If a unit direction `u` satisfies
> `<u,a> >= a_gamma`, its first Voronoi exit is labelled by `v`.  Consequently
> the two caps belonging to `v` and `-v` have total asymptotic mass
>
>     2^{-(delta_gap(gamma)+o(1))n},
>     delta_gap(gamma)=-1/2 log_2(1-a_gamma^2).                  (32)

**Proof.** Write a noncollinear competitor as `w=q lambda b`, where `b` is
unit and `q>=gamma`.  Minimality of `v-w` gives

    <a,b> <= q/2.                                               (33)

Put `s=<u,a>`.  When `q<=sqrt(2)`, maximizing `<u,b>` under (33) and comparing
`<u,b>/q` with the score `s` of `v` gives the threshold
`s>=sqrt(1-q^2/4)`.  When `q>=sqrt(2)`, the unconstrained alignment `b=u` is
the worse regime and gives `s>=1/q`.  Both thresholds decrease with `q`, so
their supremum over `q>=gamma` is (31).  The standard fixed-height spherical
cap asymptotic is `(1-a_gamma^2)^{n/2}` up to subexponential factors, proving
(32).  `square`

For the query allowance `delta_query=0.02648284...`, the second branch of
(31) closes the ledger once

    gamma > (1-2^{-2 delta_query})^{-1/2}
          = 5.2670066... .                                    (34)

This is an exact solvable class, but it deliberately does not pretend to
handle a dense near-short shell.  The earlier value `1.9636` resulted from
incorrectly continuing the first branch of (31) past `sqrt(2)`; competitors
near length `2lambda` expose that error.

## 14. The causal Voronoi-tiling transport kernel

Let `V` be the origin Voronoi cell.  For every oriented Voronoi-relevant
vector `w`, write

    F_w = V intersect {x : <x,w>=||w||^2/2},
    A_w = H^{n-1}(F_w).                                       (35)

For a parity `theta`, let `A_theta` be the sum of `A_w` over facets whose
coefficient parity is `theta`.  Draw `X` uniformly from `V`, draw `U`
uniformly from `S^{n-1}`, and let `Q` be nearest-lattice-point decoding away
from its measure-zero ties.  For a dimensionless step `eta`, define the
causal label difference

    C_eta = Q(X+eta lambda U)-Q(X).                            (36)

> **Lemma (causal tiling kernel).** For every nonzero parity `theta`,
>
>     lim_{eta -> 0+} Pr[parity(C_eta)=theta]/eta
>       = d_n lambda A_theta/det(L),                           (37)
>
> where
>
>     d_n = Gamma(n/2)/(2 sqrt(pi) Gamma((n+1)/2))
>         = Theta(n^{-1/2}).                                  (38)

**Proof.** Outside an `O(eta^2)` tube around codimension-two faces, a segment
of length `eta lambda` crosses at most one Voronoi facet.  At a regular point
of `F_w`, the interval of starting points inside `V` that cross outward has
normal thickness `eta lambda (<U,w/||w||>)_+ + O(eta^2)`.  Integrate over the
facet, average over `U`, and divide by `vol(V)=det(L)`.  Rotational symmetry
gives `E[(U_1)_+]=d_n`.  Summing exactly the facets of parity `theta` proves
(37).  `square`

The experiment can draw `X` without first constructing `V`: draw a point
uniformly in any fundamental parallelepiped and subtract the two nearest
lattice labels before and after the displacement.  Translation covariance
makes their difference equal in law to (36).  This observation is geometric,
not an algorithmic CVP implementation.

The kernel has a useful cone-volume normalization.  Put

    q_w = ||w|| A_w/(2n det(L)).                                (39)

The pyramids `conv(0,F_w)` partition `V` up to measure zero, so

    sum_w q_w = 1.                                             (40)

There is also an exact matrix conservation law:

    (1/(2 det(L))) sum_w A_w (w w^T/||w||) = I_n,              (40a)

where the sum is over both orientations.  To see it, write the periodic
Voronoi residual as `R(x)=x-Q(x)`.  Haar translation invariance gives, for
every physical displacement `h`,

    E[Q(X+h)-Q(X)] = h-E[R(X+h)-R(X)] = h.                      (40b)

Differentiate at zero.  A fixed infinitesimal `h` crosses `F_w` with rate
`A_w (<h,w/||w||>)_+/det(L)` and transports the label by `w`.  Pairing `w`
with `-w` turns the positive parts into the linear map in (40a).  Taking the
trace recovers (40).  Equivalently, for `a_w=w/||w||`,

    sum_w q_w a_w a_w^T = I_n/n.                               (40c)

Thus the Voronoi collision vectors form an exact isotropic transport frame;
the missing theorem concerns how much of this conserved frame can sit in its
minimum-length atoms.

If `q_min` is the sum of (39) over the oriented shortest facets, then the
shortest-parity derivative in (37) is at least

    2n d_n q_min = Theta(sqrt(n)) q_min.                        (41)

Thus causal crossing pays only a polynomial factor relative to shortest
cone-volume mass.  Unlike (25), it compares the two actual first-arrival
owners, and the crossing vector itself labels the transported parity.

## 15. Shortest facets retain a robust constant fraction of their area

The causal kernel does not lose its exponent when interfaces must be kept
away from higher-order collisions.

> **Lemma (shortest-facet robust core).** Every shortest facet `F_v` contains
> in its affine hyperplane the ball of radius `lambda/(2sqrt(3))` centered at
> `v/2`.  More strongly, for `0<epsilon<1` the homothetic core
>
>     F_v(epsilon)=(1-epsilon)F_v+epsilon(v/2)                  (42)
>
> has area `(1-epsilon)^{n-1}A_v`, lies at tangential distance at least
> `epsilon lambda/(2sqrt(3))` from every ridge, and every `x` in this core
> satisfies
>
>     ||x-w||^2-||x||^2 >= epsilon lambda^2/2                  (43)
>
> for all `w in L` other than `0` and `v`.

**Proof.** On the hyperplane of `F_v`, the distance to the constraint induced
by another `w` is

    (||w||^2-<v,w>)/(2||P_(v-perp)w||).                         (44)

After scaling `lambda=1`, set `A=||w||^2` and `b=<v,w>`.  Minimality gives
`A>=1`, `b<=A/2`, and `b^2<=A`.  Directly,

    3(A-b)^2 >= A-b^2,                                        (45)

with equality at `A=1,b=1/2`; for `1<=A<=4` minimize the left-minus-right
at `b=A/2`, and for `A>=4` at `b=sqrt(A)`.  Equations (44)--(45) give the
claimed inball.

Every defining slack of `F_v` is affine, is nonnegative on `F_v`, and at
`v/2` is at least `lambda^2/2`.  Taking the convex combination (42) proves
(43).  Applying the same combination to the inball proves the ridge-distance
claim, and homothety multiplies `(n-1)`-area by
`(1-epsilon)^{n-1}`.  `square`

Taking `epsilon=1/n` retains a constant fraction of every shortest-facet
area and gives an inverse-polynomial positional reach and squared-distance
margin.  Taking a small constant `epsilon` loses exactly
`-log_2(1-epsilon)n+o(n)` bits of area but gives a constant physical margin.
Hence any strict exponent slack in a lower bound for `q_min` can be converted
into a robust collision set without changing the qualitative algorithm.

The M4 finite-cube audit `experiments/walsh_voronoi_causal_transport.py`
confirms the distinction at dimension five.  At step `0.03lambda`, the
shortest-parity mass divided by the step ratio was `0.212`--`0.305` on all
seven fixtures.  In particular the needle value was `0.301`, whereas its
neighboring-fan shortest-XOR flux was zero.  These constants are evidence
about the kernel, not an asymptotic bound or a CVP implementation.

The corrected surviving statement is now:

> **Joint-core cone-volume target.** After all certified parity covers,
> exact low-width factor transports, and submodular min-cut escapes, prove
>
>     p_min + q_min >= 2^{-(delta+o(1))n}                       (46)
>
> for some `delta<0.02648284`, or construct a counterexample inside that
> joint core.  Then implement the robust causal crossing by a polynomial
> line scan of the common empirical Hessian field, without calling CVP.

The geometric robustness part of this target is supplied by (42)--(43).  The
two live obligations are (46) and the analytic assertion that the available
theta/Hessian line scan exposes the collision vector on that robust core.
The latter cannot be inferred from nearest-pair geometry alone without
bounding the total contribution of all other lattice Gaussian terms.

## 16. The log-theta Hessian removes tangential contamination

There is a better analytic observable at a causal crossing.  Put

    Z_s(x) = sum_(w in L) exp(-pi ||x-w||^2/s^2)

and let the posterior on lattice sites have weights proportional to the same
summands.  Direct differentiation gives the exact identity

    C_s(x) = s^4/(4 pi^2) [nabla^2 log Z_s(x) + 2 pi I/s^2]
           = Cov_x(W).                                        (47)

At a point x on the facet F_v, the sites 0 and v have equal weight.  If all
other sites are deleted, (47) is exactly vv^T/4, independently of the
tangential displacement x-v/2.  This is the key correction to a raw-Hessian
line scan: the raw two-site Hessian retains a tangential xx^T component,
whereas the log-Hessian converts site location into posterior covariance and
cancels it.

Let a be the common weight of 0 and v on F_v, and define

    eps_0 = [sum_(w notin {0,v}) exp(-pi||x-w||^2/s^2)]/(2a),
    eps_2 = [sum_(w notin {0,v}) ||w||^2
             exp(-pi||x-w||^2/s^2)]/(2a lambda^2).             (48)

If eps_0,eps_2 <= eps <= 1, raw-moment comparison and Cauchy--Schwarz give

    ||C_s(x)-vv^T/4||_op <= 6 eps lambda^2.                     (49)

Indeed the normalized remainder first moment is at most
sqrt(eps_0 eps_2), the second moment at most eps_2, and normalization changes
the two-site mean u/2 and raw moment uu^T/2 by at most the displayed
eps_0 terms.  Subtracting the outer products gives (49).

Along a transverse segment x(t)=x_0+t xi crossing F_v at t_0, the exact
two-site law is logistic:

    p(t) = [1+exp(-2 pi <xi,v>(t-t_0)/s^2)]^(-1),
    C_s(x(t)) = p(t)(1-p(t)) vv^T.                              (50)

Its ridge is centered at the facet and has width
s^2/|<xi,v>|.  The same moment comparison gives O(eps lambda^2) perturbation
when (48) holds uniformly relative to the combined two-site weight.  Hence a
polynomial grid resolves the collision vector whenever the crossing angle,
the inverse tail, and s^2/lambda^2 are polynomially controlled.

The common DGS panel encodes Z_s and its first two derivatives, so (47) does
not require CVP.  It does require relative rather than absolute empirical
accuracy.  The remaining analytic clause has therefore been reduced to two
explicit inequalities on enough of the robust facet core:

    eps_0+eps_2 <= n^(-Omega(1)),                               (51)
    Z_s(x) lies above the empirical relative-error floor.       (52)

Lemma (42)--(43) attenuates each individual non-nearest site on the robust
core, but does not by itself bound their total weighted second moment.  The
next calculation shows that the aggregate tail/response conjunction is not a
surviving target at all: it is incompatible with the desired exponent under
the present sample budget.

There is also a sharp stress test on (46).  Suppose a lattice has `K`
oriented shortest vectors, grouped into `K/2` antipodal pairs.  Their total
direct-exit mass is at most one and their total cone-volume mass is at most
one, so some pair has combined weight at most `4/K`.  Fix that shortest `v`
and perturb by

    T_tau = I-tau (v/lambda)(v/lambda)^T.

For small positive tau, strict Cauchy--Schwarz makes plus-or-minus T_tau v the
unique shortest pair, while the gap to the next shell prevents an overtake.
The shortest-facet inball guarantees that its facets persist, and their solid
angles and cone volumes vary continuously.  Therefore, for every zeta>0,
an arbitrarily small perturbation satisfies

    p_min+q_min <= (4+zeta)/K.                                 (53)

Rational lattices admit rational perturbations.  Thus an exponential
kissing family with exponent kappa would forbid any universal
version of (46) with delta<kappa unless its perturbations are excluded by an
exact joint-core escape.  This does not presently refute the target, but it
shows that the proof must use the joint-core hypotheses and cannot follow
from local shortest-facet geometry alone.

## 17. The tail/response conjunction is impossible at exponent 0.0264828

The log-theta kernel solves identification, but not observability.  Let
`x` lie in the homothetic core `F_v(eta)` and put `s=1/xi_t(d)`.  For every
non-colliding lattice site `w`, the squared-distance slack is affine and
satisfies

    ||x-w||^2-||x||^2 >= eta ||w||^2/2.

Hhan's weighted shell estimate with Gaussian dilation `sqrt(2/eta)` now
gives the rigorous aggregate bound

    eps_0(x)+eps_2(x)
      <= 2^((1/2 log_2(t_0/(eta t))+o(1))n).                  (54)

For fixed parameters, polynomial tail suppression therefore requires
`eta t>t_0`.  Since the target scale must have `t<1/4`, one needs
`eta>4t_0=0.9258...`.  The core then retains at most

    (1-4t_0+o(1))^(n-1) = 2^(-(3.7540+o(1))n)                (55)

of the facet area.  Homothetic robustification cannot preserve the target
mass.

There is a stronger bound independent of the core shape.  Suppose
`M=2^((m+o(1))n)` common samples are available and put `alpha=m/2`.  On any
shortest facet subset where `eps_0<=n^-c` and the normalized theta response
is above the empirical floor,

    F_(1/xi_t)(x) >= n^-C 2^(-alpha n),                       (56)

write `x=v/2+y`, `y` perpendicular to `v`.  The two distinguished weights
are

    a=exp[-pi xi_t(d)^2(lambda^2/4+||y||^2)].

Tail suppression gives `F_s(x)<=Z_s(x)<=3a`; hence (56) and the definition
of `xi_t` imply

    ||y||^2/lambda^2
      <= (alpha/t-1)/4 + O(log n/n).                          (57)

The shortest-facet inball has radius `lambda/(2sqrt(3))`.  Comparing ball
volumes shows that, when the leading radius in (57) is useful, the accessible
area fraction is at most

    [sqrt(3(alpha/t-1)+O(log n/n))]^(n-1).                    (58)

For `alpha<=t` this is superexponentially small.  Even spending the complete
`2^(n/2+o(n))` panel on one query gives `alpha=1/4`; combining tail
suppression `t>t_0` with (58) gives at most
`2^(-(1.0289+o(1))n)` of a shortest facet.  To retain
`2^(-(delta+o(1))n)` instead requires

    m >= 2 t_0 (1+2^(-2delta)/3)
      = 0.61169...                 for delta=0.0264828.        (59)

This exceeds the half-exponential budget before paying for multiple line
queries.  Therefore (51) and (52) cannot hold together on enough robust
cone-volume mass for the proposed empirical log-theta scan.  The
direct-plus-cone inequality (46), even if true after fixing an explicit
certificate catalogue, is not sufficient.  The surviving target must be a
response-weighted shortest-collision bound paired with a variance-sensitive
estimator, or a nonlocal observable that avoids dividing by the local theta
value.
