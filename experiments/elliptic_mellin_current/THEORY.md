# Mellin-current conductor obstruction

This note isolates the exact part of the theoretical attack from the numerical
probes.  It uses no enumeration of elliptic curves.

## 1. The transported differential

Let `E/Q` have conductor `N`, normalized weight-two newform

\[
f(z)=\sum_{n\ge1}a_ne^{2\pi inz},
\]

and modular parametrization `phi:X0(N)->E`.  Up to the rational normalization
of the Neron differential,

\[
\phi^*\omega_E=2\pi i f(z)\,dz.
\]

On the Fricke geodesic

\[
z(t)=\frac{i e^t}{\sqrt N}
\]

the pullback is a constant multiple of

\[
G(t)\,dt,
\qquad
G(t)=e^t f(z(t)).
\]

Thus `G dt` is an exact transported elliptic differential.  It is not a
surrogate signal attached to the curve.

For root number `-1`, Fricke symmetry makes `G` odd and

\[
\Lambda(1+z)
=2\sum_{j\ge0}\frac{z^{2j+1}}{(2j+1)!}
\int_0^\infty t^{2j+1}G(t)\,dt.
\]

Analytic rank at least five is consequently equivalent to the two current
constraints

\[
M_1=M_3=0.
\]

## 2. Minimal-shell theorem

Put

\[
u=t^2,
\qquad d\nu=2tG(t)\,dt.
\]

Analytic rank at least `2m+1` gives

\[
\int u^j\,d\nu=0,
\qquad 0\le j<m.
\]

**Proposition.** A nonzero signed measure satisfying these `m` equations has
at least `m` sign changes.

**Proof.** If it had `r<m` sign changes, place the roots of a degree-`r`
polynomial at those changes and choose its global sign to match the measure.
Its integral against the measure is strictly positive.  But its degree is
below `m`, so the moment equations say the same integral is zero.  This is a
contradiction.

Therefore analytic rank `2m+1` needs at least `m+1` oriented Fricke shells.
The record curve has rank five and exactly three shells, so it is minimal in
this representation.

If its two changes occur at `u=a,b`, the quadratic `(u-a)(u-b)` matches the
current's `+,-,+` sign.  Since the lower moments vanish,

\[
M_5=\int u^2\,d\nu
=\int (u-a)(u-b)\,d\nu>0.
\]

The order-five terminal zero is therefore forced to be exact once the
minimal-shell moment balances hold.

## 3. Conductor as a derived boundary

Set

\[
\alpha=\frac{2\pi}{\sqrt N},
\qquad s=\alpha e^t,
\qquad F(s)=\sum_{n\ge1}a_ne^{-ns}.
\]

Then

\[
G(t)\,dt=\frac1\alpha F(s)\,ds
\]

and rank five requires

\[
\int_\alpha^\infty
\left(\log\frac{s}{\alpha}\right)^kF(s)\,ds=0,
\qquad k=1,3.
\]

For a fixed multiplicative coefficient current, the first equation selects a
logarithmic balance boundary `alpha`; the third is a zero-skew condition about
that same boundary.  If both hold, conductor is recovered from

\[
N=(2\pi/\alpha)^2.
\]

This is the inverse-design coordinate: construct or exclude a coefficient
current first and derive its conductor, rather than walking through levels.

## 4. One-crossing annihilator

Termwise integration gives

\[
M_k=\sum_{n\ge1}a_nW_k(\alpha n),
\qquad
W_k(a)=2\int_1^\infty(\log x)^k e^{-ax}\,dx.
\]

Define

\[
R(a)=\frac{W_3(a)}{W_1(a)}.
\]

Normalize `log(x) exp(-a*x) dx` to a probability measure.  Then `R(a)` is the
expectation of `(log x)^2`.  Differentiating with respect to `a` gives

\[
R'(a)=-\operatorname{Cov}_a(x,(\log x)^2)<0,
\]

because both functions are strictly increasing.  Thus

\[
K_\rho(a)=W_3(a)-\rho W_1(a)
\]

has exactly one crossing as a function of `a` whenever `rho` is in the range
of `R`.  It is the dual analogue of a single Eikonal cut between coefficient
shells.

## 5. Universal coefficient envelope

The prime-power recurrence and the Hasse bound imply

\[
|a_{p^r}|\le(r+1)p^{r/2}.
\]

Coprime multiplicativity therefore gives

\[
|a_n|\le d(n)\sqrt n.
\]

If rank is at least five, then for every `rho`

\[
0=\sum_{n\ge1}a_nK_\rho(\alpha n).
\]

Since `a1=1`, every such curve must obey the separator inequality

\[
\boxed{
|K_\rho(\alpha)|
\le
\sum_{n\ge2}d(n)\sqrt n\,|K_\rho(\alpha n)|.
}
\]

Any `rho` violating this inequality excludes analytic rank five at that
conductor.  This conclusion is curve-independent.

For a rigorous finite evaluation, use `d(n)*sqrt(n)<=2n` after a cutoff `L`.
The omitted tail is bounded by

\[
2\sum_{n>L}n\bigl(W_3(\alpha n)+|\rho|W_1(\alpha n)\bigr).
\]

After interchanging sum and integral, the geometric-arithmetic tail is
explicit because, for `q=exp(-alpha*x)`,

\[
\sum_{n>L}nq^n
=\frac{q^{L+1}((L+1)-Lq)}{(1-q)^2}.
\]

The remaining one-dimensional positive integral can be enclosed by interval
quadrature.

## 6. What the first relaxation says

The current numerical separator retains the exact divisor bound through
exponent 42 and the larger `2n` envelope through exponent 70.  Before the
explicit final `exp(-70)` remainder, its normalized margin is

- `+0.614` at conductor 3000;
- `+0.097` at conductor 4250;
- `-0.0077` at conductor 4500.

This is not yet recorded as a rigorous conductor theorem: the kernel
quadrature, final tail, and the interval between tested levels still need
certified enclosures.  It demonstrates that the current inequality has real
excluding power, but the independent Hasse box becomes too loose near 4500.

## 7. Multiplicative depth

The loose box permits long consecutive blocks of unrelated coefficients to
sit simultaneously at their negative limits.  Real coefficients instead
satisfy `a_mn=a_m*a_n` for coprime indices.

For the record curve, decomposing by total prime-factor degree `Omega(n)`
gives the dominant first-moment layers

\[
+8290, -39181, +61775, -44259, +14610,\ldots
\]

for degrees zero through four.  The rank-one control has only a small prime
layer and negligible higher multiplicative depth.  The record also has

\[
a_p<0\quad\text{for every prime }p\le107,
\]

with the first positive prime trace at 109.  Its prime phases form a coherent
negative chamber; coprime multiplicativity turns that chamber into alternating
Euler-degree shells.

A constant common Frobenius phase was tested and cannot solve both moment
equations, even at its extremal value.  The next theoretical object must
therefore be a scale-dependent phase field over the primes, not one global
phase and not independent coefficient boxes.

## 8. Next exact target

The desired refinement is a multiplicative separator: bound

\[
\sum_{n\ge2}a_nK_\rho(\alpha n)
\]

using the prime-power recurrences before applying absolute values.  Taking
absolute values term by term destroys the alternating Euler layers that make
rank five possible and is precisely where the present bound loses strength.

The target is therefore an Euler-degree majorant whose state is the transported
prime-phase field and whose derived components are all composite coefficients.
That mirrors the repository's exact two-current rule: prime phases are the
free current; composite coefficients must not be admitted independently.

## 9. Two-zone phase frontier

As a minimal coherent model, assign every prime up to a cutoff `P` the same
normalized trace

\[
\frac{a_p}{2\sqrt p}=c<0,
\]

assign the later primes neutral phase `c=0`, and derive every prime power and
coprime product from the exact Chebyshev/Hecke recurrences.  This is not an
elliptic curve, but unlike the independent Hasse box it is one lawful
multiplicative current.

Solving `M1=M3=0` produces a phase--conductor frontier:

| cutoff `P` | chamber phase `c` | derived conductor |
|---:|---:|---:|
| 67 | -0.985381 | 1,631,161 |
| 71 | -0.948674 | 2,088,234 |
| 83 | -0.856021 | 4,099,436 |
| 97 | -0.807885 | 6,083,104 |
| 107 | -0.747295 | 10,443,303 |
| 113 | -0.712079 | 14,591,455 |
| 127 | -0.697046 | 17,012,983 |
| 149 | -0.644528 | 30,080,405 |

The record's uninterrupted negative chamber ends at 107, but its actual mean
normalized phase through that prime is only `-0.6181` (weighted mean
`-0.5696`).  The coherent model says that strengthening the same chamber to
about `-0.747`, or extending a roughly `-0.70` chamber through 127, is enough
at the moment level to move below the conductor record.

The very low-conductor end of the frontier demands phases almost at the Hasse
boundary at every small prime and is unlikely to be arithmetically realizable.
The segment from `P=97` through `P=127` is the useful target: it supplies a
quantitative local signature rather than a coefficient or curve search box.

The remaining hard bridge is algebraic realization.  A useful elliptic
surface must carry five generically independent sections and make this
negative prime chamber a consequence of one geometric parameter or symmetry.
Imposing the prime conditions independently by CRT would merely move the
exhaustive search into another coordinate and is not the proposed route.

## 10. Algebraic realization achieved

The adjacent `elliptic_collision_family` experiment constructs the required
rank-five surface internally.  Put

\[
K=rs(r+s),\qquad m=r^2+rs+s^2,
\]

and choose three rational squares in arithmetic progression

\[
A^2=q^2-4K,\qquad C^2=q^2+4K.
\]

Then

\[
y^2=x^3-mx+q^2/4
\]

has a point at zero and two horizontal triples at

\[
r,s,-r-s
\quad\text{and}\quad
-r,-s,r+s.
\]

The two triples give two forced relations among seven sections.  The record
specialization proves that five remaining sections are generically
independent.

A rational record-passing subfamily is

\[
M=\frac{2z(2z+1)}{4z-3},
\qquad r=M/2,
\qquad s=M+z,
\]

with the dilation-two Euclidean triangle generated by `(M,z)`.  At `z=1` it
is exactly the conductor record.  The canonical adjacent chart fiber `z=5/4`
retains rank five but introduces a large new discriminant prime and loses the
long negative Frobenius chamber.

For fixed support `(r,s)=(3,7)`, the progression parameter is itself the
congruent-number curve

\[
v^2=u^3-840^2u.
\]

Its three integral-triangle fibers are completely classified.  The smaller
discriminant `12,457,909` fiber acquires a third collinearity relation and has
only four visible independent currents; the other rank-five fibers have
larger conductor.  Hence the record is minimal on its complete integral
fixed-support orbit.

The algebraic bridge is therefore no longer missing.  The remaining theorem
must couple the two coordinates: identify a rational path on the rank-five
support surface for which the transported prime-phase field stays within the
sub-record two-zone frontier while the discriminant divisor does not acquire
new large components.

## 11. Nodal-torus representation of the seven-section current

Normalizing the surface by `c=s/r` and the Euclidean angle `t` gives the
shape discriminant

\[
D(c,t)=64t^2(1-t^2)^2(1+c+c^2)^3
-27c^2(1+c)^2(1+t^2)^4.
\]

The steepest descent from the record runs into an accidental cross-relation
wall.  Holding exact clearance from that wall yields

\[
t=\frac{9c-16}{9(c+1)},
\]

which decreases the shape discriminant until it reaches `D=0`.  The first
simple rational point on this safe contour keeps five independent sections
but acquires the genuine multiplicative prime `1046925311`; denominator
height defeats geometric descent.

At `D=0`, however, the curve is

\[
y^2=(x-a)^2(x+2a),\qquad a=2v^2.
\]

With `z=y/(x-a)`, `d=sqrt(3a)`, and

\[
U=\frac{z-d}{z+d},
\]

the group becomes the norm-one torus over `Q(sqrt(6))`, and

\[
\frac{dx}{2y}=\frac1{2d}d\log U.
\]

Thus the continually integrated elliptic current reaches a literal
multiplicative support geometry at the nodal boundary.  The seven sections
become seven norm-one coordinates, the forced Mordell--Weil relations become
products, and conductor control becomes support control for their prime
divisors and for the transverse smoothing parameter.  The next inverse
problem is therefore an `S`-unit construction on this fixed torus followed by
one symbolic lift, rather than another search through Weierstrass models.

There is a crucial rationality correction.  A nondegenerate rational
horizontal triple on the nodal cubic would force

\[
X^2+XY+Y^2=6,
\]

which has no rational solution by a two-adic infinite descent.  Thus the
nodal torus is an exact limiting representation and height obstruction, not a
rational base fiber.

The rational base is instead the weighted cusp chart at `t=0`.  After
`X=x/t`, `Y=y/t`, the differential tends to `dX/[c^2(1+c)^2]`.  Expanding the
six section differences as Abel jets gives rank growth

\[
1,2,3,4,4,5
\]

through orders zero to five, with only the difference of the two forced
triple relations remaining.  The five-current minor at orders `0,1,2,3,5`
has determinant

\[
\frac{12(c-1)^2(c+2)^2(2c+1)^2(c^2+c+1)^4}
{35c^8(c+1)^8}
\]

in the positive chamber.  This realizes the five invisible Mordell--Weil
currents as a finite jet frame on simple additive support.  The current attack
is to transport that frame by its Gauss--Manin connection while treating the
nodal two-adic obstruction as the arithmetic-height term in the Eikonal path.

The arithmetic continuation makes that height term exact.  For
`c=C/B,t=P/Q`, put

\[
H=B^2+BC+C^2,\qquad K=BC(B+C).
\]

The conductor-bearing homogeneous current is

\[
\mathcal F=64P^2(Q^2-P^2)^2Q^2H^3
-27K^2(Q^2+P^2)^4.
\]

It is a quadratic norm from `Q(sqrt(H))`.  Completing all sign choices also
reveals the nearest rank-loss wall `(2C+B)P-BQ=0`; the record has offset
exactly `-1`, the smallest nonzero lattice distance.  Requiring its support to
remain primitive adds the square gate

\[
BQP(Q^2-P^2)C(B+C)=\square.
\]

On the record's unit-offset characteristic this gate is the elliptic curve

\[
V^2=X(X-1050)(X-1470),
\]

with record point `(22050,3087000)`.  The snipe problem on the closest possible
rank-bearing path has therefore become an integral-point theorem on this
control curve, rather than a search through elliptic-curve models.
