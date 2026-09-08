# Elliptic rank support transport

## Target and status

The smallest conductor-rank record whose global minimality is not known is
the rank-five curve

\[
E:\quad y^2+y=x^3-79x+342,
\qquad N=19\,047\,851.
\]

Its rank is exactly five.  The uncertainty is whether a rank-five curve of
smaller conductor exists.  The experiment makes a theory-first local attack
on that question.  It does not enumerate Weierstrass coefficient boxes or
walk a bounded curve database.

No smaller-conductor rank-five curve was certified.  The finite local family
tested here is closed with an exact negative result.

## 1. The correct geometric object

Completing the square gives

\[
Y^2=4x^3-316x+1369.
\]

If a cubic `g` takes signed curve values at six distinct rational
coordinates `x_i`, then

\[
P(x)=g(x)^2-(4x^3-316x+1369)
\]

has the six `x_i` as roots.  Replacing `g` by `g+c` gives a pencil

\[
Y^2=g_c(x)^2-P(x),\qquad g_c=g+c,
\]

on which all six rational sections persist.  The divisor of `Y-g_c` forces
the exact relation

\[
P_1+\cdots+P_6=O.
\]

This is the arithmetic version of transporting curve support: a complete
degree-six principal divisor moves as one object.  The group-law current is
not allowed to be projected point by point.

If the leading coefficient of `g_c^2-P` is `u^2`, the rational change

\[
X=u^2x,\qquad Z=u^2Y
\]

produces a monic integral model whenever the remaining coefficients are
integral.  The original curve is the fiber `u=2`.  The nearest nontrivial
integer square-leading fibers are `u=1` and `u=3`.

## 2. Finite exact audit

The interval `-100 <= x <= 100` contains 23 nonnegative-sheet integral
points of the record curve.  Exact interpolation, with global sign reversal
quotiented out, gives 196 distinct cubics that meet at least six signed
points.

The audit then applies three gates.

1. **Base-rank gate.** Reduction in `E(F_p)/2E(F_p)` at good primes proves
   that 102 pencils carry five independent sections at `u=2`.  Pencils whose
   selected divisor spans only rank four are discarded.
2. **Conductor gate.** Exact discriminant factorization retains only fibers
   whose tame conductor lower bound lies below `19,047,851`.  At `p>=5`, a
   prime dividing both `c4` and the discriminant is counted twice, as required
   for additive reduction.  This leaves 52 fibers.
3. **Saturation gate.** Every surviving mod-two relation is summed in the
   exact rational group law.  Its duplication quartic is factored over
   `Q`; rational halves are adjoined and the test repeats.  All 52 fibers
   end in an exact nonzero integral relation.  There are no unresolved cases
   and no rank-five certificates.

The result is therefore

\[
\boxed{196\to102\to52\to0.}
\]

This is not evidence that the conductor record is globally minimal.  It is a
proof that the nearest square-leading fibers of every small-support pencil
through the record curve fail to improve it by transporting five independent
sections.

## 3. A representative false positive

One pencil uses

\[
g(x)=53+8x-\frac92x^2-\frac12x^3
\]

at `x=-10,-8,-6,-1,3,4`.  Its `u=1` fiber is

\[
Y^2=X^3-27X^2-268X+1696.
\]

It has six visible integral points, but they satisfy both the principal
divisor relation and

\[
P_0=P_1+P_2+P_4.
\]

This example is the central falsification: geometric support count is not
Mordell--Weil rank.

## 4. Relation to the repository's geometric machinery

The signed-Eikonal owner construction in
`../EIKONAL_TWO_CURRENT_DERIVATION.md` transports one exact differential from
a zero-line and derives the companion current from integrability.  Its useful
arithmetic analogue is:

| geometric construction | elliptic arithmetic analogue |
|---|---|
| zero-line support | principal divisor of `Y-g` |
| exact current `dF` | divisor class / rational section |
| commuting square `d^2F=0` | exact elliptic group law |
| cut locus | singular discriminant stratum |
| derived companion current | saturation/halving forced by the section lattice |

The Meyer implementation is a convex cartoon/texture decomposition with a
screened-Poisson step.  It can inspire a numerical regularizer for smooth
coefficient motion, but convex separation does not preserve rationality,
local reduction type, or section independence.  It cannot certify rank.

The `zeta/` canonical-system experiment supplies a second warning.  Its
Hankel/DIP operator bridge is correct, but the measured compressed phase law
is not zeta-specific and its refinement observables do not converge to a
distinguished arithmetic law.  Replacing zeta by `L(E,s)` would therefore
produce a numerical observer, not a zero theorem.  A proof about the order of
vanishing still needs modularity plus a sign-definite explicit formula,
certified modular symbols, or another rigorous analytic-rank argument.

## 5. Theory suggested by the failure

The next object should not be another pencil obtained by moving a divisor on
one isolated curve.  It should be an elliptic surface over `Q(t)` with:

1. five sections proven independent in the generic Mordell--Weil lattice;
2. an explicit Shioda height matrix whose determinant stays nonzero away from
   a known finite set;
3. a discriminant polynomial with controlled factor strata and a Tate-local
   conductor formula;
4. a low-degree parameter map so algebraically distinguished critical,
   collision, or symmetry fibers can be examined without scanning integers.

On such a surface, “continual integration of differentials” has a precise
form: transport the invariant differential and the five Abel--Jacobi section
classes through the Gauss--Manin connection, while the height determinant is
the no-caustic certificate.  The conductor descent is then performed on the
discriminant divisor, and specialization is rejected whenever the height
determinant vanishes.  This adds the arithmetic invariant missing from the
Eikonal analogy.

For analytic zeros, the corresponding family must be pushed through the
weight-two newforms of the specialized conductors.  The Hasse--Weil
`L`-function, not the Riemann zeta function, is the relevant spectral object.
The algebraic rank-five certificate remains the five independent rational
sections; BSD may guide the search but is not needed for the lower bound.

## 6. Reproduce

The light exact tests run locally:

```sh
python3 experiments/elliptic_rank_support/test_elliptic_rank_support.py
python3 experiments/elliptic_rank_support/run_base_rank_gate.py
python3 experiments/elliptic_rank_support/run_nearby_fibers.py
```

The factorization and saturation audit uses the Mini's SymPy installation:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_rank_support/run_factor_screen.py
```

Expected terminal summary:

```text
tame-sub-record candidates=52
dependent 52
unresolved 0
rank5 0
rank-five certificates= 0
```

## Sources

- ICARM leaderboard, curve 41: <https://elliptic-rank.icarm.cloud/curve/41>
- Elkies--Watkins, *Elliptic curves of large rank and small conductor*:
  <https://arxiv.org/abs/math/0403374>
- LMFDB record data: <https://www.lmfdb.org/EllipticCurve/Q/19047851/a/1>
- LMFDB reliability statement:
  <https://www.lmfdb.org/knowledge/show/rcs.rigor.ec.q>

