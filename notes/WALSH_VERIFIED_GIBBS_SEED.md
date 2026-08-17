# Verified Gibbs parity law from the common DGS/BDD advice

Date: 2026-08-13.

This note separates two questions that were conflated in the dense-core
target.  The common DGS panel and the preprocessing BDD verifier do define a
parity law whose stationary shortest-vector mass is overwhelming.  Moreover,
every linear-size block conditional of that law is computable by one binned
Walsh transform of the common panel.  What is not yet proved is a uniform
subexponential entrance or mixing theorem for those block conditionals.

## 1. The cold primal theta law is exact but too weak

Put `s=1/xi_r(d)` and let `theta(y)` be the coefficient parity of `y in L`.
If `Y~D_(L,2s)`, then

    Pr[theta(Y)=theta]
      = rho_(2s)(B theta+2L) / rho_(2s)(L)
      = rho_s(L+B theta/2) / rho_(2s)(L).               (1)

Thus the normalized midpoint theta masses are already a genuine probability
law: they are exactly the parity law of a centered primal discrete Gaussian.
For a shortest `v in B theta_*+2L`, the two points `+v/2,-v/2` give

    Pr[theta(Y)=theta_*]
      >= 2 exp(-pi lambda_1(L)^2/(4s^2)) / rho_(2s)(L)
      = 2^(-(r+o(1))n).                                 (2)

At the target `r=0.23675858...`, (2) is far below the available parity-query
exponent `0.02648284...`.  Kim's arbitrary-parameter one-sample DGS theorem
does not remove this loss: at this cold scale the sample is zero with
overwhelming probability, and conditioning it to be nonzero costs the
missing exponential factor.  Equation (1) is therefore a correct baseline,
not the desired seed law.

## 2. A verifier energy on parity cells

Fix the correct length guess `lambda<=d<=(1+1/n)lambda` and a uniformly
accurate target-width empirical Hessian panel.  For a parity `theta`, compute
the empirical midpoint Hessian, take the prescribed leading directions, run
the preprocessing BDD queries, and retain only exactly verified nonzero
lattice outputs of norm at most `d`.

Let `G=B^T B`.  For rational `B`, choose a positive integer `D_G` such that
`D_G G` is integral.  Define the integer verifier energy

    E(theta) = min D_G ||w||^2,                          (3)

where the minimum is over the retained outputs.  If there is no retained
output, put

    E(theta)=floor(D_G d^2)+1.                           (4)

Every finite energy in (3) is an integer.  Distinct squared lattice norms
therefore differ by at least `1/D_G`.

Define the verified Gibbs law

    pi(theta) = 2^(-3n E(theta)) / Z.                   (5)

When a cell has several minimum-energy verified outputs, use any fixed exact
tie rule, or split its mass among them.  The output of the law is the parity
of the selected verified vector, rather than necessarily the queried address.

> **Proposition (stationary robust mass).**  On the uniform empirical event
> used by the midpoint recovery theorem, the pushforward of (5) through the
> verified BDD output has shortest-vector mass at least `1-2^(-2n)`.

Proof.  Every actual shortest parity supplies a verified output of energy
`E_*=D_G lambda^2`.  Conversely, every output with energy `E_*` is a shortest
lattice vector.  Every other cell has energy at least `E_*+1`.  There are at
most `2^n` cells, so their total Gibbs weight relative to one minimum cell is
at most

    2^n 2^(-3n) = 2^(-2n).                              (6)

This also proves robustness under the empirical perturbation: the only
analytic fact needed is that every true shortest cell still supplies its
verified endpoint.  False endpoints cannot lower (3), because lattice
membership and squared norm are checked exactly.

The coefficient `3n` in (5) is inessential.  Any inverse temperature above
`(1+delta)n` in base two gives error at most `2^(-delta n)`.

## 3. Every block conditional is available from the common panel

Write the empirical traceless Hessian in coefficient parity form as

    T_hat(theta) = (1/M) sum_i A_i (-1)^(k_i.theta),     (7)

where `k_i in F_2^n` is the coefficient parity of the `i`th dual DGS sample
and `A_i` is its weighted traceless quadratic matrix.  Fix a block
`I subset [n]` of size `b`, and fix the outside bits `theta_(I^c)`.  Form the
`2^b` matrix bins

    C_u = sum_(i: k_(i,I)=u)
            A_i (-1)^(k_(i,I^c).theta_(I^c)).           (8)

Then for every block word `a in F_2^b`,

    T_hat(theta_(I^c),a)
      = (1/M) sum_u C_u (-1)^(u.a).                     (9)

Equation (9) is one length-`2^b` matrix-valued Walsh transform.  It computes
all Hessians required by the exact heat-bath conditional of (5).  After the
transform, run BDD and exact verification on the `2^b` directions, form their
integer energies, and sample with weights `2^(-3nE)`.

The cost of one block update is

    M poly(n) + 2^b poly(n) + 2^(b+o(n)),               (10)

where the last term is the collection of preprocessing-BDD queries.  With
`M=2^((2r+o(1))n)` and any `b<=(1/2-o(1))n`, (10) fits in
`2^((1/2+o(1))n)` time and the existing space budget.  No fresh DGS samples
are used, and adaptively selected blocks are allowed by the uniform empirical
event.

Thus (5) is not merely a formal posterior: it has exact, implementable
linear-size block conditionals derived from the common advice.

## 4. The remaining clause is entrance, not stationary mass

Let `K_I` be the heat-bath projection for block `I`, and compose or randomize
a fixed family of blocks of size at most `(1/2-o(1))n`.  A complete theorem
would follow from either of the following uniform statements on the residual
dense spectral core:

    (a) a warm start reaches pi in 2^o(n) block updates; or
    (b) before mixing, 2^o(n) updates already output a minimum-energy
        verified vector with probability 2^(-o(n)), or at least
        2^(-(delta+o(1))n), delta<0.02648284.

Neither statement follows from stationarity.  A planted-parity control makes
the obstruction exact: take one hidden word `theta_*`, give it energy zero,
and give every other word energy one.  A block of size `b` can see the hidden
word only when all `n-b` frozen bits already agree with `theta_*`.  From a
uniform start this event has probability `2^(-(n-b))`.  For `b<=n/2`, a
subexponential number of updates does not reach the target.  Yet the Gibbs
law (5) puts mass `1-2^(-Omega(n))` on it.

The same control is the one-sparse, exponentially weak Walsh/LPN model.  A
general random-sample sparse-Walsh decoder here would solve an LPN instance
with bias `2^(-(r+o(1))n)` from `2^((2r+o(1))n)` examples.  Sparse-WHT
algorithms that query deliberately chosen time-domain samples do not apply:
querying a chosen dual parity means sampling a cold shifted DGS coset, which
is the unavailable operation.  Squaring correlations again costs
`2^((4r+o(1))n)`.

Therefore the construction proves the requested robust mass at stationarity
and supplies every large-block conditional within budget.  The surviving
theorem is now the following narrower geometric statement.

> **Dense-core verified-block entrance theorem (proof target).**  For the
> verifier energy (3)--(4) arising from an actual residual dense-core lattice,
> construct blocks of rank at most `(1/2-o(1))n` and an initialization from
> the DGS/BDD advice such that the corresponding heat-bath process reaches a
> minimum verifier-energy cell with probability `2^-o(n)` in `2^o(n)` block
> updates.  It is enough to obtain exponent below `0.02648284`.

This theorem must use a genuinely lattice-specific relation between nearby
parity cells and their physical BDD outputs.  It cannot be proved from the
Walsh signal amplitude, sparsity, or stationary mass alone.

