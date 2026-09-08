# Audit of the closest-wall uniqueness argument

## Audited theorem

The exact result proved by the arithmetic-current chain is the following.

> Let `B,C` be positive coprime integers on the characteristic
> `P/Q=1/6`, `2C-5B=-1`.  If the seven-section surface admits the rational
> dilation making its support exactly `(r,s)=(B,C)`, then `(B,C)=(3,7)`.

Equivalently, the positive integral points selected by

\[
Y^2=210B(5B-1)(7B-1)
\]

have only `B=3` after imposing the primitive-support interpretation.  No
height bound or finite walk through the Pell orbit occurs in the proof.

This establishes uniqueness of the record fiber on the complete primitive
closest-wall characteristic.  It does **not** establish global conductor
minimality among all elliptic curves, or even among every rational fiber of
the two-parameter seven-section surface.

## Arithmetic base certificate

For

\[
E:y^2+y=x^3-79x+342
\]

the generalized Weierstrass invariants are

\[
c_4=3792,\qquad \Delta=-19047851.
\]

Deterministic trial division proves `19047851` prime.  The model is therefore
minimal, has good reduction away from that prime, and has multiplicative
reduction of exponent one at it.  Its conductor is exactly `19047851`.

The five displayed generators are independent modulo `2`: every nonzero
binary combination fails to be twice a point after reduction at some good
prime.  The completed monic cubic

\[
X^3-1264X+21904
\]

has no root modulo 5, hence the rational curve has no rational 2-torsion.
Therefore the modular certificate proves Mordell--Weil rank **at least five**.
There is no rank upper bound in this repository, so “rank exactly five” is not
an internally proved statement.

## Reduction audit

| Stage | Status | Exact scope |
|---|---|---|
| Seven-section parametrization | equivalence inside the construction | Two split horizontal cubics plus the center point give seven sections and two forced relations. |
| Five-section certificate | proved lower bound | Trivial rational 2-torsion plus independence in `E(Q)/2E(Q)` proves rank at least five. |
| Six wall forms | complete only for visible cross-support relations | Avoiding them does not exclude every possible higher-coefficient Mordell--Weil dependence. |
| Homogeneous carrier | exact discriminant identity | It is not itself a conductor formula; minimalization and local reduction types remain necessary. |
| Primitive square gate | equivalence for exact primitive support in this chart | It is not known to be necessary for every sub-record minimal model on the surface. |
| Unit-offset characteristic | complete one-dimensional branch | No argument presently forces every sub-record candidate onto `P/Q=1/6`, `2C-5B=-1`. |
| Control-curve two-descent | complete | Pairwise coprimality partitions the squarefree primes `2,3,5,7`; local conditions leave `(3,7,10)`. |
| Positive Pell flows | complete | Strict predecessors classify all positive solutions, not a sampled prefix. |
| Defect conic and covers | complete after repair below | All four residues modulo 70 must be retained. |
| Biquadratic norm and currents | complete for the six covers | Relative norms recover the three Pell units exactly; the current contradictions are unbounded. |

## Repair: the missing zero residue

The first version proved

\[
h\pmod {70}\in\{0,14,50,64\}
\]

but later retained only `14,50,64`.  A positive integer can still be zero
modulo 70, so that deletion was invalid.  Restoring it adds two covers:

\[
(g,d)=(1,35),\qquad(3,70).
\]

The complete cover list is

\[
(1,2),(1,7),(1,35),(3,5),(3,14),(3,70),
\]

with universal norm levels

\[
6,21,105,5,14,70.
\]

The added `sqrt(2)` norm seeds are

\[
\begin{array}{c|c}
70&70,\ 110\pm60\sqrt2\\
105&105,\ 165\pm90\sqrt2.
\end{array}
\]

Divisibility by `d=70` or `35` excludes the split seeds, just as divisibility
by `14` or `7` excludes the split seeds at levels 14 and 21.  Their viable
squareclass roots are especially simple:

\[
\rho_{70}=\beta,
\qquad
\rho_{105}=\sqrt{105}.
\]

Level 105 uses the same reduced current `P` as levels 5 and 21 and has odd
negative-Pell index; its current is strictly negative.  Level 70 uses `Q` as
levels 6 and 14 and has even index; every reduced term has the same strict
sign.  Thus the repaired branches contain no solution.

## Completeness of the unit reduction

The relative norms of `xi=A-beta*b` are not merely analogous to the Pell
flows.  Substitution of the defect reconstruction gives

\[
\begin{aligned}
N_{K/\mathbf Q(\sqrt2)}(\xi)
 &=c\{(50w-49v)-35(v-w)\sqrt2\},\\
N_{K/\mathbf Q(\sqrt{210})}(\xi)
 &=-c\{(15u+14v)-(u+v)\sqrt{210}\},\\
N_{K/\mathbf Q(\sqrt{105})}(\xi)
 &=-c\{(21u+20w)-2(u+w)\sqrt{105}\}.
\end{aligned}
\]

The bracketed terms are exactly

\[
(99-70\sqrt2)^n,
\quad(29-2\sqrt{210})^{j+1},
\quad(41-4\sqrt{105})^{k+1}.
\]

Their product identity is therefore exact and gives the global squareclass
equation, rather than a necessary heuristic approximation.

The `sqrt(2)` seed list is complete because `Z[sqrt(2)]` is Euclidean:
`3` and `5` are inert, `2` is ramified, and
`7=(3+sqrt(2))(3-sqrt(2))`.  The three allocations of the two factors above
7 give precisely the rational seed and the displayed conjugate split pair.

The four unit parity classes are distinct in `K*/K*^2`.  Neither
`29-2sqrt(210)` nor `41-4sqrt(105)` is a square in `K`; solving for a square
in the relevant quadratic subfield reduces respectively to rational squares
among `{14,15,7,15/2}` and `{20,21,10,21/2}`, of which there are none.  Their
product is not a square because `70` times it is `beta^2`, while `sqrt(70)` is
not contained in the biquadratic field whose quadratic subfields have
squareclasses `2,105,210`.  Hence exhibiting one `rho_c` also proves the
uniqueness of its parity pair.

Finally, the negative-Pell index is even exactly when `g=3`.  Modulo 3, the
orbit alternates between `3|(v-w)/2` and `3` not dividing it; the conic
`H(H+3u)=70D^2` then gives `3|H` in precisely the even case.  Since `g=9` is
already impossible, this fixes the sign used in every reduced current.

## What remains before “unsnipable” is justified

The current theorem closes one distinguished branch.  A global statement
would additionally require all of the following.

1. Prove an exact Mordell--Weil upper bound if the target category requires
   rank exactly five rather than rank at least five.
2. Prove that every rank-at-least-five curve with conductor below `19047851`
   is represented by this surface, or introduce a separate exhaustive
   classification theorem.
3. Within this surface, prove that every sub-record conductor candidate must
   have primitive support and must lie on the audited unit-offset
   characteristic.  The homogeneous carrier alone does not imply this.
4. Replace the six visible cross walls by a certificate controlling every
   Mordell--Weil dependency locus relevant to the rank threshold.
5. Compute minimal local conductor exponents symbolically on every remaining
   arithmetic chamber, not merely the discriminant carrier.

Until those steps exist, “unsnipable” should mean *unsnipable on the complete
primitive closest-wall characteristic*, not globally minimal among elliptic
curves.

## Higher-rank construction suggested by the mechanism

The proof mechanism is more useful constructively than a direct coefficient
search.  The present surface has seven visible points and two forced triple
relations, so its built-in rank capacity is five.  A rank-six construction
needs one additional independent section; a third split horizontal cubic
would add three sections and one forced relation, raising the visible capacity
from five to seven.

The cheapest extra section is already explicit.  At `x=r-s`, rationality
reduces to the conic

\[
Z^2=(c+1)\{(c+1)(1+t^2)^2+12(c-1)t(1-t^2)\}.
\]

It contains the symmetric point `(c,Z)=(1,2(1+t^2))`.  Intersecting with

\[
Z=2(1+t^2)+\lambda(c-1)
\]

gives the second point

\[
c=1+
\frac{4(1+t^2)^2+24t(1-t^2)-4(1+t^2)\lambda}
{\lambda^2-(1+t^2)^2-12t(1-t^2)}.
\]

Thus the rank-six-capacity locus is rationally parametrized; no point or
coefficient search is needed.  The midpoint slope `lambda=-2(1+t^2)` and
`t=1/3` give `c=149`, `Z=-980/3`.  After the available 10-dilation, the
short model is

\[
y^2=x^3-7939432816x+4928844010000.
\]

Six displayed sections are independent modulo 2, and its 2-division cubic
has no root modulo 3, so this specialization has rank at least six.  Its
discriminant is

\[
2^{12}149^6\cdot14627\cdot48839731007.
\]

The large final multiplicative prime means this example is a construction
certificate, not yet a conductor competitor.  It does, however, turn the
higher-rank problem into conductor descent on an explicit rational subfamily.

For a zero-sum support triple `R={r1,r2,r3}`, write

\[
K_R=r_1r_2r_3,
\qquad
r_1r_2+r_1r_3+r_2r_3=-m.
\]

It is a horizontal triple on

\[
y^2=x^3-mx+a_6
\]

exactly when `h_R^2=a_6+K_R`.  Thus adding a third triple is not a point
search: choose three zero-sum triples with the same quadratic support `m` and
require

\[
h_1^2-K_1=h_2^2-K_2=h_3^2-K_3=a_6.
\]

This is the higher-rank analogue of the three-square progression that created
the record family.  Its primitive dilation, wall offsets, and conductor
carrier can be subjected to the same norm/current descent before any
specialization.  A cheaper rank-six route is to impose one extra rational
section on the existing two-parameter surface; that is one square condition
over `(c,t)` and produces a double-cover surface whose primitive fibers can be
filtered by the audited current machinery.

The key design criterion is therefore: seek an additional split support whose
new squareclass current is *compatible*, rather than annihilated, at 2.  The
level-5 contradiction shows exactly what to avoid—two product currents whose
2-adic slopes differ by a positive constant.
