# Seven-section collision family

## Outcome

The rank-five conductor record belongs to an internally derived algebraic
family with seven rational sections.  The construction uses two horizontal
cubic-support triples and a three-square arithmetic progression.  It is not a
database fit or a coefficient search.

The family supplies the missing bridge from geometric support to five
generically independent Mordell--Weil currents.  It also explains two earlier
failures: some especially symmetric fibers acquire a third collinearity
relation and fall to rank four, while generic fibers retain rank at least five
but usually acquire much larger conductor support.

The completed current descent proves that the record is the unique primitive
fiber on the closest unit-offset characteristic.  The precise theorem, the
repair of an omitted `h=0 mod 70` cover class, and the boundary between this
branch theorem and global conductor minimality are recorded in
[`AUDIT.md`](AUDIT.md).  Internally, the section certificate proves rank at
least five; no rank upper bound is claimed.

The audit also extracts a rational conic for an eighth section at `x=r-s`.
One canonical specialization has six independent sections and gives an exact
rank-at-least-six construction; it is structurally useful but not yet
conductor-competitive.

## 1. Two support triples

Choose rational `r,s` and put

\[
K=rs(r+s),
\qquad m=r^2+rs+s^2.
\]

If

\[
A^2=q^2-4K,
\qquad C^2=q^2+4K,
\]

then

\[
E:\quad y^2=x^3-mx+\frac{q^2}{4}
\]

contains

\[
(0,q/2),
\]

the lower horizontal triple

\[
(r,A/2),\quad(s,A/2),\quad(-r-s,A/2),
\]

and the upper horizontal triple

\[
(-r,C/2),\quad(-s,C/2),\quad(r+s,C/2).
\]

Each triple sums to the origin.  Seven sections minus those two forced
relations leave five possible independent currents.

The square conditions say that `A^2,q^2,C^2` are in arithmetic progression.
Equivalently, the right triangle

\[
(C-A,\ C+A,\ 2q)
\]

has area `4K`.

## 2. The record

Take

\[
r=3,\qquad s=7,\qquad K=210,
\]

and the right triangle

\[
(24,70,74).
\]

It gives

\[
A=23,\qquad q=37,\qquad C=47,
\]

and therefore

\[
y^2+y=x^3-79x+342.
\]

Five selected sections have no surviving dependency in the exact reductions
`E(F_p)/2E(F_p)`, proving rank at least five.  Since this is one specialization,
the corresponding generic family has rank at least five as well.

## 3. Record-passing rational family

Let `z` be rational and set

\[
M=\frac{2z(2z+1)}{4z-3},
\qquad r=\frac M2,
\qquad s=M+z.
\]

Then

\[
Mz(M-z)(M+z)=rs(r+s).
\]

The Euclidean triangle with dilation two,

\[
2(M^2-z^2),\quad4Mz,\quad2(M^2+z^2),
\]

has the required area and produces all seven sections.  At `z=1`, it is the
record curve.

Writing

\[
m=\frac{z^2(84z^2-6z+1)}{(4z-3)^2},
\qquad
q=\frac{z^2(32z^2-8z+13)}{(4z-3)^2},
\]

the rational discriminant is

\[
\Delta=64m^3-27q^4.
\]

Its numerator is `-z^6` times an irreducible degree-ten polynomial.  At
`z=1`, `Delta=-19,047,851`.

The exact logarithmic derivative of `abs(Delta)` at the record points toward
increasing `z`, but the canonical adjacent integral-chart fiber `z=5/4`
introduces additive reduction at 5 and the large multiplicative prime
`2,626,021,261`.  Its five sections remain independent, so rank transport
works while conductor support expands.

## 4. Congruent base curve and complete integral orbit

For fixed `r=3,s=7`, the arithmetic-progressions form the elliptic curve

\[
v^2=u^3-840^2u.
\]

The record is `(1176,28224)`.  The other small integral triangle
`(40,42,58)` is `(1960,78400)` and gives

\[
y^2+y=x^3-79x+210,
\qquad \Delta=12,457,909.
\]

Although its discriminant beats the record, it has the extra exact relation

\[
(0,29)+(7,1)+(-3,41)=O
\]

on the completed-square `(x,Y)` model, and only four visible independent
currents.

Euclid's formula classifies all integral right triangles of area 840:

\[
(15,112,113),\quad(24,70,74),\quad(40,42,58).
\]

They give center heights `113/2`, `37`, and `29`.  The first has rank at least
five but much larger conductor, the second is the record, and the third is the
rank-four trap.  Thus the record is exactly minimal within the complete
integral orbit of its fixed six-point support.

Adding the record base point to the rank-four base point gives `q=-113/2` and
the rank-five integral presentation

\[
Y^2=X^3-1264X+51076.
\]

Its discriminant factors as

\[
-2^8\cdot227\cdot17,169,193,
\]

so it also misses the conductor record decisively.

## 5. Interpretation

This family supplies three exact states that the earlier support pencil could
not distinguish:

1. two forced principal-divisor relations, compatible with rank five;
2. an accidental third cross-line relation, forcing rank loss;
3. five independent sections whose local discriminant support expands.

The full normalized surface, its six rank-loss divisors, its rank-safe
discriminant contour, and its nodal-torus limit are derived in
[`SEVEN_SECTION_SURFACE.md`](SEVEN_SECTION_SURFACE.md).  The resulting attack
does not continue sampling rational curves.  The nodal torus proves an exact
2-adic height obstruction, while a rational cusp blow-up exposes all five
section currents in Abel orders `0,1,2,3,5`.  Those jets now form the frame for
a symbolic no-caustic transport into the smooth rank-five surface.

The continued arithmetic attack is in
[`ARITHMETIC_EIKONAL.md`](ARITHMETIC_EIKONAL.md).  It replaces affine distance
by homogeneous lattice distance, completes the signed rank-wall arrangement,
expresses the conductor carrier as a quadratic norm, and reduces primitive
support on the closest-wall characteristic to one explicit elliptic control
curve.

## 6. Reproduce

```sh
python3 experiments/elliptic_collision_family/test_elliptic_collision_family.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_collision_family/run_cusp_abel_jet.py

python3 experiments/elliptic_collision_family/run_control_curve_descent.py

python3 experiments/elliptic_collision_family/run_single_recurrence_attack.py
```
