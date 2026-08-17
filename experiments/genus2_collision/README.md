# The m=3 collision cover

This experiment starts from Cal Aldred's July 2026 poster family of
three-dimensional Keller counterexamples.  It keeps `m=3` distinct from the
degree-three modular isogeny used in `experiments/elliptic_pi`.

## 1. The actual map data

For `m=3`, the poster gives

\[
R_3(t,p)=p^2-5(1+t)p+10(1+t+t^2),
\]

\[
A_3(p)=p^2-5p+10,
\qquad
\rho_3(t)=R_3(t,1-t)=16t^2+8t+6.
\]

If \(\alpha^2-5\alpha+10=0\), the Hensel polynomial is

\[
S(t)=\alpha+\frac{5(\alpha+2)}3t
       +\frac{25(\alpha+2)}9t^2.
\]

With

\[
t=1+xy,\quad h=t^3z+y^2S(t),\quad p=x^2h,
\]

the map is

\[
F_3=(th,\ y+xh,\ xR_3(t,p)/t^3),
\]

and exact reduction modulo \(\alpha^2-5\alpha+10\) gives
\(\det JF_3=-30\).  Its maximum component degree is 13.

## 2. Generic branch divisor

Write a target as \((A,B,C)\).  The first two output equations imply

\[
t^2-(1+Bx)t+Ax^2=0,
\]

and the third gives

\[
Ct^5=x\bigl(A^2x^4-5Ax^2t(1+t)+10t^2(1+t+t^2)\bigr).
\]

Eliminating \(t\) produces an extraneous chart factor \(A^2x^5\) and the
actual inverse-coordinate quintic

\[
H_{A,B,C}(x)=Kx^5+Lx^2+Mx+N,
\]

where

\[
\begin{aligned}
K={}&A^3C^2-30A^2BC+256A^2+10AB^3C-95AB^2-B^5C+10B^4,\\
L={}&10AC-10B^2C+90B,\\
M={}&180-15BC,\qquad N=-6C.
\end{aligned}
\]

For a generic target, `H` has five simple roots.  Those roots plus infinity
give six simple branch points, hence

\[
\mathcal C_{A,B,C}:Y^2=H_{A,B,C}(X)
\]

has genus 2 by Riemann-Hurwitz.

## 3. The symmetric collision axis is a false positive

At the poster's collision target \((A,B,C)=(c,0,0)\),

\[
H_{c,0,0}(x)=4x(64c^2x^4+45).
\]

After constant scaling this is

\[
v^2=u(u^4+1).
\]

It has the non-hyperelliptic involution

\[
(u,v)\longmapsto(1/u,v/u^3).
\]

The two elliptic quotients are

\[
V_+^2=(U+2)(U^2-2),\qquad
V_-^2=(U-2)(U^2-2),
\]

and are isomorphic over \(\mathbb C\).  Their invariant is \(j=8000\), the CM
value of discriminant \(-8\).  Thus

\[
\operatorname{Jac}(\mathcal C_{c,0,0})\sim E_{-8}^2.
\]

The scaling orbit in `c` is isotrivial, so its periods are only algebraic
powers of `c`; it has no new rank-four Picard-Fuchs dynamics.  Looking only at
the six symmetric points would therefore have produced exactly the elliptic
collapse that the falsification program was meant to reject.

## 4. A genuinely genus-2 slice

The target slice \((A,B,C)=(0,1,z)\) gives

\[
H_z(x)=(10-z)x^5+10(9-z)x^2+15(12-z)x-6z
\]

and

\[
\operatorname{disc}_x H_z
=101250000(z-10)^2(20z^2-405z+2052)^2.
\]

At `z=1`, reduction modulo 7 has

\[
\#C(\mathbf F_7)=9,\qquad \#C(\mathbf F_{49})=45,
\]

so the Frobenius polynomial is

\[
T^4+T^3-2T^2+7T+49.
\]

It is irreducible and ordinary.  With Howe-Zhu notation
\((q,a,b)=(7,1,-2)\), none of

\[
a=0,\quad a^2=q+b,\quad a^2=2b,\quad a^2=3b-3q
\]

holds.  The reduction is absolutely simple.  Specialization of endomorphisms
then certifies that the characteristic-zero generic Jacobian is not an elliptic
product.

## 5. Picard-Fuchs systems

For \(f=H_z\), use the de Rham basis

\[
\omega_i=x^i\frac{dx}{y},\qquad i=0,1,2,3.
\]

The script performs Hermite reduction with

\[
d(q/y)=\left(q'f-\frac12qf_x\right)\frac{dx}{y^3}
\]

to obtain the exact rank-four Gauss-Manin connection.  A cyclic component has
an order-four scalar Picard-Fuchs equation.  After removing its cubic-derivative
term, both symmetric-cube identities fail, so this operator is not a disguised
elliptic \(\operatorname{Sym}^3\).

The cup product has only

\[
\langle\omega_0,\omega_3\rangle=\frac4{3(10-z)},\qquad
\langle\omega_1,\omega_2\rangle=\frac4{10-z}.
\]

Consequently the primitive exterior square has the constant basis

\[
01,\quad02,\quad3(03)-12,\quad13,\quad23
\]

and rank 5.  The first component is cyclic of full rank five.  This is the
correct higher-period object to test for a `1/pi^2` identity.

This is a positive structural result, not yet a fast pi algorithm.  A genus-2
primitive exterior square is a weight-two, K3-type variation with Hodge numbers
`(1,3,1)`; a Ramanujan-like `1/pi^2` evaluation still requires a suitable
arithmetic/CM point and a period relation.  Genus 2 alone does not supply either.

## 6. The two-node boundary exists only at infinity

A quintic with no (x^4,x^3) terms and two double roots must, after scaling
the roots by (S), have the form

\[
(x+2S)(2x^2-2Sx+3S^2)^2/4
=x^5+5S^3x^2-\frac{15}{4}S^4x+\frac92S^5.
\]

Trying to match this directly to (H_{A,B,C}), with common coefficient
scale \(\mu\), leaves the exact obstruction

\[
K-\mu=-\frac{125}{1296}S^4\mu^2.
\]

Thus no nondegenerate finite target realizes two nodes.  The missing boundary
is nevertheless reached by a rational arc whose target coordinates diverge.
With

\[
\begin{aligned}
A(z)&=-\frac{5(2916z^2-3375z-3125)}{26244z^2},\\
B(z)&=-\frac{27z+125}{81z},\\
C(z)&=-\frac{972}{125}z,
\end{aligned}
\]

the inverse quintic is (1296z/125) times

\[
f_z(x)=(1-z)x^5+5x^2-\frac{15}{4}x+\frac92.
\]

Its discriminant is

\[
\operatorname{disc}_x f_z=\frac{20503125}{16}z^2(z-1)^2.
\]

The rank-four residue at (z=0) is nilpotent with Jordan type `(2,2)`.
On the primitive exterior square the residue has ranks

\[
\operatorname{rank}N=2,qquad
\operatorname{rank}N^2=1,qquad
N^3=0,
\]

so its type is `(3,1,1)`.  The connection has only the three true singular
points (0,1,\infty).  This is the best degeneration available in the
genus-2 collision chart, but it also exposes the limitation: the variation is
weight two and its logarithmic tower has length three.  The order-five
Calabi--Yau operators behind the strongest `1/pi^2` series instead have a
length-five maximally-unipotent tower.  Therefore the naive
`genus 2 -> primitive wedge^2 -> 1/pi^2` ladder does not reach the desired
binary-splitting family by itself.

One can force weight four by taking a symmetric square of the rank-five
K3-type system.  That does create a length-five nilpotent block, but it is not
a hidden order-five equation.  After subtracting scalar trace, the residues at
(0) and (1) generate the full ten-dimensional Lie algebra
(\mathfrak{so}_5).  Consequently

\[
\operatorname{Sym}^2(\mathbf5)=\mathbf1\oplus\mathbf{14}
\]

and the nonconstant weight-four piece is irreducible of rank 14.  Its larger
recurrence and coefficient burden make it a poor candidate for a large
constant-factor speedup over Chudnovsky.  A viable successor must produce a
weight-four, rank-five motive directly (or another comparably small
tree-evaluable recurrence), rather than manufacturing weight four from this
generic genus-2 variation.

## 7. Exceptional CM-factor tower

The symmetric fiber can be constructed for every (m) without the full
inverse map.  Put

\[
Q_m(u)=\operatorname{Res}_t\bigl(\rho_m(t),u-t(1-t)\bigr).
\]

After scaling the collision target, its branch curve is

\[
C_m:\quad y^2=xQ_m(x^2),\qquad g(C_m)=m-1.
\]

The next polynomials after the split-CM (m=3) case are

\[
Q_4=256u^3-140u+175,
\qquad
Q_5=16384u^4-16800u+11025.
\]

For (m=4), point counts at the good prime (p=13) are

\[
(\#C(\mathbf F_{13}),\#C(\mathbf F_{13^2}),\#C(\mathbf F_{13^3}))
=(10,210,2170),
\]

giving the ordinary irreducible Weil polynomial

\[
T^6-4T^5+28T^4-100T^3+364T^2-676T+2197.
\]

Its maximal-real polynomial is

\[
g_4(U)=U^3-4U^2-11U+4,
\qquad \operatorname{disc}(g_4)=11020.
\]

It is irreducible and non-Galois, so the Howe--Zhu criterion certifies that
the reduction, and hence the characteristic-zero Jacobian, is absolutely
simple.

For (m=5), the counts over (\mathbf F_{13^k}), (1\leq k\leq4), are

\[
(18,180,2094,28288).
\]

The resulting degree-eight Weil polynomial is ordinary and irreducible.  Its
maximal-real polynomial

\[
g_5(U)=U^4+4U^3-39U^2-160U-144
\]

is irreducible with Galois group (S_4).  Its quartic field has no proper
subfield and is not cyclotomic-real, giving the same absolute-simplicity
certificate.

Consequently neither (C_4) nor (C_5) has an elliptic isogeny factor over
the algebraic closure.  The low-height sequence (D=-4,-8,\ldots) stops
rather than continuing.  Moreover, the (E_{-8}) factor at (m=3) lies on
an isotrivial symmetric axis.  Making a nonconstant
(\operatorname{Sym}^4H^1(E)) variation requires importing an external
modular elliptic family, which returns to the classical Ramanujan/CM setting
rather than supplying a new collision engine.

## 8. The internal rank-two eigensystem test

The (m=3) symmetric curve has

\[
\sigma:(x,y)\longmapsto(-x,iy),\qquad \sigma^2=(x,-y),
\]

so its two (\pm i) eigenspaces in (H^1) each have rank two.  If this
automorphism survived on a non-isotrivial collision slice, then
(\operatorname{Sym}^4V_i) would indeed be rank five, weight four, and would
turn a (J_2) cusp into (J_5).

For the exact inverse quintic

\[
H=Kx^5+Lx^2+Mx+N,
\]

preserving the displayed automorphism requires (H) to be odd.  But

\[
N=-6C,
\qquad
L\big|_{C=0}=90B.
\]

Thus oddness forces (B=C=0), leaving only

\[
H_{A,0,0}=4x(64A^2x^4+45),
\]

the known isotrivial scaling axis.

Allowing a parameter-dependent Möbius conjugation does not open another
branch.  Write a general smooth (C_4)-curve as

\[
y^2=x(x^4+bx^2-1).
\]

Choose a nonfixed branch point (r), so

\[
b=\frac{1-r^4}{r^2},
\]

send (r) to infinity, and depress the resulting quintic.  Its cubic
coefficient is exactly

\[
-\frac{4(r-1)^2(r+1)^2(r^2+1)^2}{5r^2(r^4+1)}.
\]

The collision normal form requires this coefficient to vanish.  Away from
the singular denominator this gives (r^4=1), hence (b=0).  Choosing either
fixed branch as infinity gives the same condition directly.  A conjugate
(C_4)-locus is tangent to the collision surface at first order, but the
contact is quadratic and contains no non-isotrivial local slice.  Therefore
the collision family supplies a rank-two eigenspace only at the isolated
isotrivial (E_{-8}^2) point; there is no collision-native variation on which
to perform the proposed fourth symmetric power.

## 9. Higher-rank automorphism eigensystems

Absolute simplicity does not remove the built-in action

\[
\sigma:(x,y)\mapsto(-x,iy)
\]

from any symmetric \(C_m\). For \(m=4\), a rank-three \(i\)-eigensystem with
principal \(J_3\) monodromy could in principle produce \(J_5\oplus J_1\) in
its symmetric square. For \(m=5\), a rank-four eigensystem could analogously
produce \(J_5\oplus J_1\) in its exterior square. These possibilities require
a non-isotrivial collision deformation preserving \(\sigma\).

Eliminating the general \(F_m\) inverse equations gives

\[
\begin{aligned}
H_4={}&K_4x^7+L_{4,3}x^3+L_{4,2}x^2+L_{4,1}x+20C,\\
H_5={}&K_5x^9+L_{5,4}x^4+L_{5,3}x^3+L_{5,2}x^2
       +L_{5,1}x-70C.
\end{aligned}
\]

For \(m=4\), setting \(C=0\) makes the \(x^2\) coefficient \(2800B\). For
\(m=5\), it becomes \(66150B\). Hence fixed-coordinate oddness again forces
\(B=C=0\). The remaining axes are

\[
H_4(A,0,0)=4096A^3x^7-2240Ax^3+2800x,
\]

\[
H_5(A,0,0)=65536A^4x^9-67200Ax^3+44100x.
\]

In both cases \(x=u/\sqrt A\) removes \(A\) up to an overall factor, so these
axes are isotrivial.

The coordinate-independent infinitesimal test uses degree-\(2m\) binary
forms. It adds the full three-dimensional \(\mathrm{PGL}_2\) orbit to the
odd-form deformation space and intersects that with the three target
directions \(A,B,C\). The exact ranks are

\[
\begin{array}{c|ccc}
 & C_4\text{-preserving} & \text{collision} & \text{intersection}\\ \hline
m=4 & 6&3&1\\
m=5 & 7&3&1.
\end{array}
\]

The sole intersection direction is \(A\)-scaling. After quotienting
coordinates, the tangent intersection in moduli is zero-dimensional. Thus
neither symmetric point lies on a non-isotrivial collision slice preserving
the automorphism. There is no rank-three or rank-four eigenspace
Gauss--Manin system to test for \(\mathrm{SO}_3\) or \(\mathrm{Sp}_4\), and no
\(J_5\) representation can be manufactured by the proposed operations.

## Reproduce

The only non-standard dependency is SymPy:

```sh
python3 -m pip install sympy
python3 experiments/genus2_collision/test_genus2_collision.py
python3 experiments/genus2_collision/genus2_collision.py
python3 experiments/genus2_collision/test_symmetric_collision_tower.py
python3 experiments/genus2_collision/symmetric_collision_tower.py
```

The test file uses plain assertions and can also be collected by `pytest`.
