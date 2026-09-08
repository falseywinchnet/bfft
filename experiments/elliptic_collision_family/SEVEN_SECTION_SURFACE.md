# The normalized seven-section surface

## Result

The two-parameter surface exposes two different boundaries that were merged
in the one-parameter experiments:

1. **rank-loss walls**, where an accidental third collinearity relation
   appears;
2. **the nodal wall**, where the elliptic curve reaches a simpler
   multiplicative support geometry.

The steepest discriminant descent from the rank-five record heads toward the
nearest rank-loss wall.  An exact contour can avoid that wall and descend
toward the nodal wall, but its first simple rational specialization acquires a
genuine billion-sized bad prime.  Thus shape descent alone is not conductor
descent.

At the nodal wall, the elliptic differential integrates exactly to a
logarithm on a norm-one torus over `Q(sqrt(6))`, but a two-adic obstruction
prevents a nondegenerate rational seven-section fiber from reaching it.  The
useful rational representation is instead the cusp blow-up at `t=0`: five
Abel jets recover all five Mordell--Weil currents on simple additive support.

## 1. Normalized coordinates

Put

\[
c=s/r,\qquad T=t(1-t^2),
\]

and retain a rational dilation `w`.  The complete surface is

\[
\begin{aligned}
r&=Tc(1+c)w^2, &s&=cr,\\
\lambda&=Tc^2(1+c)^2w^3,\\
A&=\lambda(1-2t-t^2),
&q&=\lambda(1+t^2),
&C&=\lambda(1+2t-t^2).
\end{aligned}
\]

These formulas identically satisfy

\[
A^2=q^2-4rs(r+s),\qquad
C^2=q^2+4rs(r+s).
\]

Consequently

\[
E_{c,t,w}:y^2=x^3-(r^2+rs+s^2)x+q^2/4
\]

has all seven sections.  The coordinate `w` is only the Weierstrass dilation
`x -> w^2 x`, `y -> w^3 y`; `(c,t)` is the genuine shape surface.

The record is

\[
(c,t,w)=\left(\frac73,\frac16,\frac{54}{35}\right),
\]

which gives `(r,s,A,q,C)=(3,7,23,37,47)`.

## 2. Discriminant geometry

After removing dilation and degenerate-boundary factors, the discriminant is

\[
\boxed{
D(c,t)=64t^2(1-t^2)^2(1+c+c^2)^3
-27c^2(1+c)^2(1+t^2)^4.
}
\]

The full short-Weierstrass discriminant is

\[
\Delta=T^4c^6(1+c)^6w^{12}D(c,t).
\]

At the record,

\[
D=-\frac{23333617475}{34012224}.
\]

Its logarithmic shape gradient is

\[
\partial_c\log|D|=\frac{441804177}{666674785},\qquad
\partial_t\log|D|=-\frac{10197960336}{666674785}.
\]

So the unconstrained descent overwhelmingly increases `t`.

## 3. The six rank-loss divisors

A center point, lower point with normalized abscissa `a`, and upper point with
normalized abscissa `b` are collinear precisely when

\[
b(1+t)+a(1-t)=0.
\]

For the six nontrivial lower/upper pairings this gives

\[
\begin{gathered}
-2c+t-1,\quad 2c+t+1,\\
ct-c-2,\quad ct+c+2,\\
ct-c+t+1,\quad ct+c+t-1.
\end{gathered}
\]

Their product, apart from the degenerate factors, is the complete visible
cross-relation divisor.  The fixed-support `q=29` rank-four fiber is

\[
(c,t)=\left(\frac73,\frac25\right)
\]

and lies on

\[
L(c,t)=ct-c+t+1=0.
\]

At the record `L=-7/9`.  The discriminant gradient points toward this wall,
explaining why the apparently favorable fixed-support descent loses a rank.

## 4. Exact safe contour

Preserve the record's exact algebraic clearance from the nearest wall:

\[
L(c,t)=-\frac79.
\]

This gives the rational contour

\[
\boxed{t=\frac{9c-16}{9(c+1)}}.
\]

Its tangent has `dt/dc=1/4` at the record.  Along it,

\[
D(c)=\frac{P_{10}(c)}{1594323(c+1)^6},
\]

where

\[
\begin{aligned}
P_{10}(c)={}&2460532464c^{10}-2056269888c^9-11137238208c^8\\
&+2813556384c^7-17135648520c^6+39958431984c^5\\
&-15590632608c^4+16986307512c^3-13032197761c^2\\
&-4919040000c+1505280000.
\end{aligned}
\]

The logarithmic derivative at `c=7/3` is

\[
-\frac{2107685907}{666674785}<0.
\]

There is no stationary point between the record and the first positive nodal
root

\[
c\simeq2.52585029033177.
\]

Thus this contour gives a genuine rank-safe descent in geometric shape.

The first simple rational point near that endpoint is

\[
(c,t)=\left(\frac52,\frac{13}{63}\right).
\]

It still has five independent visible sections.  Its smallest integral short
presentation containing those sections uses scale `1134` and has

\[
y^2=x^3-48181857750000x+132214754848556250000.
\]

Its discriminant factors as

\[
-2^{12}3^{11}5^{16}13^4 19^4\cdot911\cdot1046925311.
\]

Dividing the model by the available `5`-dilation removes `5^12` but cannot
remove the prime `1046925311`: it occurs once in the discriminant and does not
divide `c4`.  It is genuine multiplicative reduction, already forcing the
conductor above the record.  This exactly demonstrates the arithmetic-height
wall.

## 5. The simpler support at the nodal wall

On `D=0`, write the singular cubic as

\[
y^2=(x-a)^2(x+2a)
=x^3-3a^2x+2a^3.
\]

Rationality of the center height forces

\[
q^2=8a^3,\qquad a=2v^2
\]

for some rational `v`.  The two tangent directions at the node are defined
over the fixed quadratic field

\[
K=\mathbf Q(\sqrt{3a})=\mathbf Q(\sqrt6).
\]

Set

\[
z=\frac{y}{x-a},\qquad d=\sqrt{3a},\qquad
U=\frac{z-d}{z+d}.
\]

Then

\[
x=z^2-2a,\qquad y=z(z^2-3a),\qquad
N_{K/\mathbf Q}(U)=1.
\]

The group law on the smooth part of the nodal cubic becomes multiplication:

\[
U(P+Q)=U(P)U(Q).
\]

For the center section `(0,q/2)`, one has `z=-2v`; hence its torus coordinate
is independent of every surface parameter:

\[
\boxed{U_0=-5-2\sqrt6.}
\]

It is the negative of the fundamental norm-one unit `5+2sqrt(6)`.

Most importantly, the invariant differential becomes

\[
\boxed{
\omega=\frac{dx}{2y}
=\frac{dz}{z^2-3a}
=\frac1{2d}\,d\log U.
}
\]

This is the continual-integration limit sought in the motivating idea.  As a
smooth seven-section fiber approaches the nodal wall, its Abel--Jacobi
coordinates become logarithms of seven norm-one torus coordinates.  The two
horizontal triple relations become two products equal to one; a rank-loss
wall is exactly an additional product equal to one.

There is also an exact support map.  Substitution of

\[
z=d\frac{1+U}{1-U}
\]

gives

\[
y=4d^3H(U),\qquad
H(U)=\frac{U(1+U)}{(1-U)^3}.
\]

A horizontal triple of height `4d^3 h` is therefore the root set of

\[
h(1-U)^3-U(1+U)=0.
\]

The constant and leading coefficients show immediately that the product of
its three roots is one.  Thus the two forced elliptic relations survive as
two tautological torus-product relations.  Rationality is also built in:
Galois conjugation sends `U` to `U^{-1}`, while
`H(U^{-1})=-H(U)` and `d` changes sign.

## 6. Why the rational nodal wall cannot be the base point

There is an exact arithmetic obstruction to carrying a nondegenerate rational
horizontal triple all the way to the nodal cubic.  Write `a=2v^2` and use the
rational nodal parameter `z=y/(x-a)`.  Three rational points at one height
would give three rational roots of

\[
z^3-6v^2z-y=0.
\]

If two normalized roots are `X=z_1/v` and `Y=z_2/v`, the missing quadratic
coefficient and the fixed linear coefficient force

\[
\boxed{X^2+XY+Y^2=6.}
\]

This conic has no rational point.  Indeed, after clearing denominators to a
primitive integer equation

\[
x^2+xy+y^2=6n^2,
\]

reduction modulo two forces `x,y` both even.  Reduction modulo four then
forces `n` even, contradicting primitivity.  Hence the nondegenerate rational
seven-section surface never actually intersects its nodal divisor.

This explains the observed phenomenon rather than merely reporting it: every
rational approach to the attractive nodal shape must pay arithmetic height.
The billion-sized prime at `c=5/2` is one instance of that unavoidable
separation between geometric distance and rational height.

The torus remains the correct limiting representation over the algebraic
closure, and its `S`-unit formulation may yield lower bounds.  It cannot,
however, be used as a rational point from which to lift directly.

## 7. Rational cusp blow-up and the five invisible jets

The degenerate boundary `t=0` *is* rational.  It initially looks less useful
because all coefficients vanish, but the weighted blow-up

\[
X=x/t,\qquad Y=y/t
\]

retains the complete support.  Put

\[
R=c(1+c),\qquad h=1+c+c^2.
\]

The blown-up curve is

\[
Y^2=tX^3-tR^2h(1-t^2)^2X
+\frac{R^4}{4}(1-t^4)^2.
\]

At `t=0`, all seven sections lie on the simple additive support

\[
Y=R^2/2,
\]

with abscissae

\[
0,\ R,\ cR,\ -(1+c)R,\ -R,\ -cR,\ (1+c)R.
\]

The elliptic differential has the finite limit

\[
\frac{dx}{2y}=\frac{dX}{2Y}\longrightarrow\frac{dX}{R^2}.
\]

The transverse height jets immediately form three oriented shells:

\[
\partial_tY|_{t=0}=-R^2,\ 0,\ +R^2
\]

for the lower triple, center, and upper triple.  This is the same minimal
three-shell structure forced independently by the two annihilated Mellin
moments of a rank-five curve.

To see all five currents, define for each noncentral support role

\[
a\in\{1,c,-1-c,-1,-c,1+c\}
\]

the formal Abel difference

\[
J_a(t)=\int_0^{aR(1-t^2)}\frac{dX}{2Y(X,t)}.
\]

At `c=7/3`, the exact coefficient matrix has ranks

\[
\boxed{1,2,3,4,4,5}
\]

through orders `t^0,...,t^5`.  The only left relation is

\[
-J_1-J_c-J_{-1-c}+J_{-1}+J_{-c}+J_{1+c}=0,
\]

the difference of the two forced horizontal-triple relations.  Thus orders

\[
0,1,2,3,5
\]

are an exact five-current jet basis.

The generic determinant of those layers, using any five of the six section
differences, factors in the positive support chamber as

\[
\boxed{
\det J_{0,1,2,3,5}
=\frac{12(c-1)^2(c+2)^2(2c+1)^2(1+c+c^2)^4}
{35c^8(1+c)^8}.
}
\]

Consequently the positive chamber loses its five-current cusp geometry only
at the symmetric support `c=1`.  The record `c=7/3` lies in a connected
no-caustic chamber extending from the rational additive boundary to the
smooth elliptic fiber.

## 8. The next theoretical attack

The rational construction should now run from the cusp outward, while the
nodal torus supplies the opposing arithmetic-height obstruction.

1. Use the exact Abel layers `0,1,2,3,5` as the transported five-current
   frame; do not represent rank by point enumeration.
2. Derive its Gauss--Manin connection in `(c,t)` and restrict to paths that
   avoid the jet divisor `c=1`, the six cross-relation divisors, and `D=0`.
3. Couple that connection to the Mellin two-moment current.  The desired path
   must rotate the prime phase toward the measured sub-record chamber while
   retaining nonzero jet determinant.
4. Use the nodal no-rational-point theorem as a height barrier in the path
   functional.  Minimizing `abs(D)` alone is forbidden; the homogeneous
   numerator after rational denominator clearing must be minimized.
5. Demand symbolic factor control of that homogeneous discriminant along the
   path before taking any specialization.  This is the arithmetic analogue
   of a no-caustic Eikonal path.
6. Specialize only the distinguished endpoint, then apply exact section
   independence and local conductor certificates.

The point of the cusp chart is that five-current transport is now explicit
and finite: no invisible rank information is lost when the curve is reduced
to simple support.  The point of the nodal chart is that it proves why naive
discriminant descent must fail.  A successful path has to live between those
two boundaries.

## 9. Earlier nodal lift formulation

Over the algebraic closure, the nodal formulation remains:

1. Choose two anti-invariant heights `h_A,h_C` for which both explicit cubics
   `h(1-U)^3-U(1+U)` split into norm-one `S`-units, with no cross-product
   relation.  The center unit is already fixed as `-5-2sqrt(6)`.
2. Restrict all unit numerators and denominators to a deliberately small prime
   set `S`.  This is the toric form of controlling conductor support.
3. Invert `U=(z-d)/(z+d)` to obtain the limiting six support points and impose
   the required opposite-abscissa pairing between the two triples.
4. Solve the remaining height-progression compatibility on the nodal surface.
5. Lift transversely with one smoothing parameter `epsilon`.  Transport the
   sections by integrating `omega`; require `epsilon` and all lift
   denominators to remain `S`-units.
6. Only after the symbolic lift exists, specialize once and apply the exact
   five-section independence and local conductor certificates.

This algebraic-closure formulation is retained as a boundary obstruction, not
as the rational lifting route.  Insufficient torus-unit rank or an unavoidable
new prime remains informative; the rational construction itself starts from
the cusp jet frame in Section 7.
