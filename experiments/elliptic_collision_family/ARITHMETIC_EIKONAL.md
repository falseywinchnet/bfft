# Arithmetic Eikonal transport on the seven-section surface

## Outcome

The cusp attack now has an arithmetic connection, a complete signed wall
arrangement, and a denominator-cleared discriminant carrier. Together they
explain why the known record is unusually efficient:

- it is exactly one lattice unit from the nearest rank-loss wall;
- its seven-section support can be made primitive;
- after minimalization, all predictable support primes disappear;
- the remaining carrier is the single prime `19,047,851`.

Several geometrically closer fibers were derived rather than searched. One
has normalized carrier `16,434,603` but lies on a previously invisible
rank-four wall. Another is rank five with multiplicative core `4,703,737`,
but failure of the primitive-support square condition forces additive
reduction at 5 and puts its conductor above the record.

No snipe survives all three gates.  The six-cover norm descent below proves
that the record point is unique in the positive unit-offset chamber, so the
complete primitive closest-wall characteristic is eliminated.  This is a
branch theorem, not a classification of all elliptic curves or all rational
fibers of the seven-section surface; the exact scope is audited in
[`AUDIT.md`](AUDIT.md).

## 1. Closed five-jet transport

Let `B_k(c)` denote the six section-current coefficient vector at Abel order
`t^k`. The exact cusp computation showed that

\[
B_0,B_1,B_2,B_3,B_5
\]

form the five-current frame. The missing fourth layer closes inside it:

\[
B_4=\alpha_0(c)B_0+\alpha_2(c)B_2,
\]

where

\[
\alpha_0(c)=
-\frac{2(280c^6+840c^5+1637c^4+1874c^3+1637c^2+840c+280)}
{429c^2(c+1)^2},
\]

and

\[
\alpha_2(c)=
\frac{7(80c^6+240c^5+401c^4+402c^3+401c^2+240c+80)}
{429c^2(c+1)^2}.
\]

At the record,

\[
(\alpha_0,\alpha_2)=
\left(-\frac{1945066}{135135},\frac{163451}{12285}\right).
\]

Thus the order-five transported section current has finite coordinates

\[
\left(1+\alpha_0t^4,\ t,\ t^2+\alpha_2t^4,\ t^3,\ t^5\right)
\]

in the cusp frame. No rank information needs to be reconstructed by walking
points.

The volume part of the `c`-connection follows from the generic jet determinant
`J(c)`:

\[
\begin{aligned}
\partial_c\log J={}&\frac2{c-1}+\frac2{c+2}+\frac4{2c+1}
+\frac{4(2c+1)}{c^2+c+1}\\
&-\frac8c-\frac8{c+1}.
\end{aligned}
\]

Its poles are exactly the cusp-frame caustics and coordinate boundaries.

## 2. Homogeneous conductor carrier

Write

\[
c=C/B,\qquad t=P/Q
\]

in primitive projective coordinates and put

\[
H=B^2+BC+C^2,\qquad K=BC(B+C).
\]

The denominator-cleared shape discriminant is the bi-homogeneous form

\[
\boxed{
\mathcal F=
64P^2(Q^2-P^2)^2Q^2H^3
-27K^2(Q^2+P^2)^4.
}
\]

It has degrees `(6,8)` in the support and angle coordinates. At the record,

\[
\mathcal F(7,3;1,6)=-210^2\cdot19047851.
\]

At the old `q=29` rank-four trap it is

\[
+210^2\cdot12457909.
\]

For the failed safe-contour point `(c,t)=(5/2,13/63)`, it exposes the genuine
new primes `911` and `1,046,925,311`. This is why minimizing the real shape
factor `D(c,t)` was misleading: the homogeneous carrier contains the
arithmetic height.

## 3. Quadratic-norm structure

Set

\[
S=P^4+Q^4,\qquad T=P^2Q^2,
\]

and

\[
L=(B-C)(B+2C)(2B+C).
\]

As a quadratic in `z=S/T`, the carrier has coefficients

\[
A=-27K^2,\quad
D_1=64H^3-108K^2,\quad
D_0=-128H^3-108K^2.
\]

Their discriminant factors by the support identity

\[
D_1^2-4AD_0=1024H^3L^2,
\qquad
4H^3-27K^2=L^2.
\]

Consequently, with

\[
\beta=32H^3-54K^2+16HL\sqrt H,
\]

one has the exact norm law

\[
\boxed{
\mathcal F=\frac1{-27K^2}
N_{\mathbf Q(\sqrt H)/\mathbf Q}
\left(-27K^2S+\beta T\right).
}
\]

The conductor carrier is therefore a norm in the field selected by the
six-point support. At fixed support, conductor control is a norm-equation
problem, not an octic search in `P,Q`.

## 4. Complete signed rank walls

Allowing signs on the center, lower, and upper sections gives 24
nondegenerate cross-relation divisors. Only six can meet the chamber
`c>1,0<t<1`. In homogeneous coordinates their offsets are

\[
\begin{aligned}
W_1&=(2C+B)P-BQ,\\
W_2&=(C+B)P-BQ,\\
W_3&=CP-BQ,\\
W_4&=(C+2B)P-CQ,\\
W_5&=(C+B)P-CQ,\\
W_6&=(C+B)P-(C-B)Q.
\end{aligned}
\]

The previously missed wall is `W_1=0`, or `(2c+1)t=1`. At
`(c,t)=(5/2,1/6)`, exact saturation finds

\[
P_0+P_{-r}^{\rm upper}+P_{-s}^{\rm upper}
=2P_s^{\rm lower},
\]

and the normalized carrier would otherwise be only `16,434,603`.

The record has wall vector

\[
(W_1,\ldots,W_6)=(-1,-8,-11,-29,-32,-14).
\]

It is already at the smallest possible nonzero homogeneous distance from
rank loss. This replaces the earlier affine-clearance contour.

## 5. Theory-selected neighboring fibers

The following are the endpoints forced by the wall lattice, not a coefficient
or conductor sweep.

| `(c,t)` | section status | arithmetic outcome |
|---|---|---|
| `(2,1/6)` | rank at least 5 | core `383*61909=23,711,147`, above record |
| `(5/2,1/6)` | rank 4 | lies on hidden wall `W_1=0` |
| `(12/5,1/6)` | rank at least 5 | genuine prime `5,209,030,907` |
| `(5/2,1/7)` | rank at least 5 | core `277*16981=4,703,737`, additive at 5 |
| `(3/2,1/5)` | rank 4 | lies on `W_6=0` |

For `(5/2,1/7)`, the integral short presentation has

\[
\Delta=-2^4 3^7 5^6\cdot277\cdot16981.
\]

Since `v_5(c4)=2`, reduction at 5 is unavoidably additive and contributes
`5^2` to the tame conductor. Even before the wild primes, its conductor is at
least

\[
25\cdot277\cdot16981=117593425.
\]

## 6. Primitive-support square gate

The dilation needed to realize primitive support `r=B,s=C` is rational
exactly when

\[
\boxed{
\Sigma=BQ\,P(Q^2-P^2)C(B+C)
}
\]

is a square. The record has

\[
\Sigma=210^2.
\]

The neighboring failures do not satisfy this condition; their forced support
dilation introduces the additive primes that the real discriminant flow did
not see.

On the closest-wall characteristic through the record,

\[
P/Q=1/6,\qquad 2C-5B=-1,
\]

so `C=(5B-1)/2`. The primitive-support gate becomes

\[
\boxed{
Y^2=210B(5B-1)(7B-1).
}

With

\[
X=7350B,\qquad V=7350Y,
\]

this is the elliptic control curve

\[
\boxed{
V^2=X(X-1050)(X-1470).
}

The record is

\[
(B,Y)=(3,420),\qquad (X,V)=(22050,3087000).
\]

Doubling it gives `B=24649/29400`; adding the three rational two-torsion
points gives `B=1/105,7/50,10/49`. None is another positive integral support
on this characteristic.

## 7. Local two-descent

The integral-point analysis begins with an exact two-descent on

\[
V^2=X(X-1050)(X-1470).
\]

For odd `B`, put

\[
a=(5B-1)/2,\qquad d=(7B-1)/2.
\]

Then `B,a,d` are pairwise coprime and

\[
(Y/2)^2=210Bad.
\]

Their squarefree parts therefore form an ordered partition of the primes
`2,3,5,7`. There are 54 partitions compatible with odd `B`. Local solubility
reduces them without placing any bound on `B`:

\[
54\xrightarrow{\bmod 8}14
\xrightarrow{\bmod 3}5
\xrightarrow{\bmod 5}2
\xrightarrow{\bmod 7}1.
\]

The sole surviving square class is the record's:

\[
B=3u^2,\qquad (5B-1)/2=7v^2,\qquad (7B-1)/2=10w^2,
\]

with `(u,v,w)=(1,1,1)`. Equivalently,

\[
15u^2-14v^2=1,\qquad 21u^2-20w^2=1.
\]

Thus no other 2-Selmer class can produce a primitive fiber at minimum wall
distance. The remaining theorem is singular and precise: prove that the two
simultaneous Pell equations above have only `u=v=w=1` in positive integers,
or extract their next common solution. Either conclusion controls the entire
closest-wall characteristic, not a bounded list of fibers.

## 8. Residual Pell recurrence

Eliminating `u` from the surviving system gives

\[
49v^2-50w^2=-1.
\]

Every positive solution belongs to the single negative-Pell orbit

\[
7v_n+w_n\sqrt{50}
=(7+\sqrt{50})(99+14\sqrt{50})^n,
\qquad n\ge0.
\]

Equivalently,

\[
v_{n+1}=99v_n+100w_n,
\qquad
w_{n+1}=98v_n+99w_n,
\]

starting from `(v_0,w_0)=(1,1)`.  The record is `n=0`.  The entire unresolved
integral-point theorem is now the square-value question

\[
\boxed{
\frac{14v_n^2+1}{15}=u^2.
}
\]

Thus the next step is not another elliptic-curve specialization. It is an
infinite-descent or elliptic-divisibility argument proving that this recurrence
has no square value for `n>0`, or producing its first exception.

## 9. Unit-flow lift and the predecessor defect

The residual recurrence is itself a compressed ordinary Pell current:

\[
7v_n+5w_n\sqrt2=(1+\sqrt2)^{6n+3}.
\]

This is useful because the two square conditions can also be integrated into
complete positive unit flows. Every positive solution of the first equation is

\[
\sqrt{15}\,u_j+\sqrt{14}\,v_j
=(\sqrt{15}+\sqrt{14})^{2j+1},
\]

and every positive solution of the second is

\[
\sqrt{21}\,U_k+\sqrt{20}\,w_k
=(\sqrt{21}+\sqrt{20})^{2k+1}.
\]

This classification is elementary and descending. Away from `(1,1)`, the
strict positive predecessors are

\[
(u,v)\longmapsto(29u-28v,\ 29v-30u)
\]

and

\[
(U,w)\longmapsto(41U-40w,\ 41w-42U).
\]

The inequalities needed for positivity follow directly by comparing
`v/u` with `30/29,29/28` and `w/U` with `42/41,41/40`. Thus there are no
unclassified positive Pell orbits hidden behind the recurrence.

A simultaneous solution is now exactly a collision `u_j=U_k`. The first flow
has `u_j=1 mod 28`, while the second has `U_k=1 mod 40`. Reducing the opposite
flow gives the exact index gates

\[
u_j\equiv(-1)^j(1-3j)\pmod5,
\qquad
U_k\equiv(-1)^k(1-5k)\pmod7,
\]

and hence

\[
j\equiv0,-1\pmod {10},\qquad k\equiv0,-1\pmod {14}.
\]

Consequently a nonrecord solution cannot begin before `j=9,k=13`, and hence

\[
u\ge U_{13}=7472659988568811532839281.
\]

This is a theorem from the unit geometry, not the endpoint of a bounded scan.

There is also a natural transverse coordinate between the two descending
flows. Put

\[
u_1=29u-28v,\qquad u_2=41u-40w,
\]

and

\[
\boxed{h=10w-7v-3u=(u_1-u_2)/4.}
\]

Direct elimination on the two quadrics gives

\[
3(uu_1-1)=2h(h+14v+6u),
\qquad
3(uu_2-1)=2h(h+14v).
\]

Both predecessors are positive. The first identity shows that `h=0` if and
only if `u=u_1=1`, so the zero-defect fiber is exactly the record. Every
counterexample has `h>0`. Since `w/u<41/40` and `v/u>1`, it also satisfies

\[
0<h<u/4.
\]

Thus the defect is a genuinely smaller positive height, not merely a renamed
coordinate. Moreover `h` is even, and reduction at 5 and 7 gives

\[
h\equiv0,-1\pmod5,\qquad h\equiv0,1\pmod7.
\]

Thus

\[
h\pmod {70}\in\{0,14,50,64\},
\]

and every nonrecord fiber has `h>=14`. The remaining attack is therefore no
longer an unstructured square-value test: it is a descent on a positive,
strictly smaller mismatch between the two Pell predecessors. The exact
certificates are implemented in
`run_single_recurrence_attack.py`.

## 10. Genus drop of the defect surface

Eliminating `v,w` in favor of `(u,h)` first gives the quartic

\[
\begin{aligned}
0={}&4h^4+48h^3u-1044h^2u^2+68h^2\\
&-7128hu^3+408hu+9u^4-18u^2+9.
\end{aligned}
\]

This quartic contains a hidden lower support. Put `x=h/u` and `z=1/u^2`.
It is quadratic in `z`, and its discriminant factors as

\[
4480x(x+3)^2(x+6).
\]

The square factor is not an accident. On the original intersection of
quadrics it lifts to the exact linear-square identity

\[
\boxed{h(h+6u)=70(v-w)^2.}
\]

Both `h` and `v-w` are even. With

\[
H=h/2,
\qquad
D=(v-w)/2,
\]

the new support equation is the conic gate

\[
\boxed{H(H+3u)=70D^2.}
\]

Its factors are almost coprime. If a prime divides both `h` and `u`, the two
Pell equations and `10w=7v mod gcd(h,u)` imply that it divides 3. Therefore

\[
\gcd(h,u)\mid3,
\qquad
\gcd(H,H+3u)\mid9.
\]

All primes away from 3 must consequently occur to even order in the two
factors except for a partition of `2*5*7`. The three positive defect classes
already found distinguish the following support branches:

| `h mod 70` | `H=h/2 mod 35` | support branch |
|---:|---:|---:|
| `14` | `7` | `7` divides `H`, `5` does not |
| `50` | `25` | `5` divides `H`, `7` does not |
| `64` | `32` | `H` is coprime to `35` |

This is the desired simplification: the unresolved elliptic recurrence has
been transported to three almost-coprime conic covers, each at height
`0<H<u/8`. The next attack is to combine their 3-adic branch with the two
Pell predecessor identities; a contradiction in those three covers would
complete the closest-wall uniqueness theorem.

More explicitly, let

\[
g=\gcd(H,H+3u),\qquad de=70,
\]

and split the coprime factors as

\[
H=gda^2,\qquad H+3u=geb^2.
\]

Then `g` initially divides 9 and `D=gab`. The original variables reconstruct
as

\[
\begin{aligned}
u&=\frac g3(eb^2-da^2),\\
v&=\frac g3(eb^2+da^2+20ab),\\
w&=\frac g3(eb^2+da^2+14ab).
\end{aligned}
\]

Substitution into the first Pell equation gives the quartic cover

\[
e^2b^4+d^2a^4-560ab(eb^2+da^2)-9660a^2b^2
=\frac9{g^2}.
\]

The left side is integral, so `g=9` is impossible and only `g=1,3` remain.
There are initially sixteen ordered `(g,d)` covers. Exact local solubility at
5, including the coprimality of `da^2` and `eb^2`, leaves only

\[
\begin{aligned}
g=1:&\quad d\in\{2,7,10,35\},\\
g=3:&\quad d\in\{1,5,14,70\}.
\end{aligned}
\]

This cut is implemented by `mod5_defect_covers()`. It does not bound or scan
`a,b`; it removes entire conic covers over the finite residue field.

Finally impose all four proved classes
`h mod 70 in {0,14,50,64}`, the divisibility by 3 needed in the reconstruction,
and coprimality at `2,3,5,7`. The combined CRT gate modulo 210 leaves

\[
\boxed{(g,d)\in\{(1,2),(1,7),(1,35),(3,5),(3,14),(3,70)\}.}
\]

The last two covers are precisely the formerly omitted positive class
`h=0 mod 70`; positivity does not permit that residue to be discarded. Tests
at higher powers of the bad primes do not remove a further cover, so these six
are the honest boundary of the local method. Completing the proof now means
finding a height descent or a global norm incompatibility on these six
quartics, not trying more moduli or advancing farther along the original
recurrence.

## 11. One universal biquadratic norm

The six quartics are scaled copies of one form. Set `A=da`. Then

\[
d^2F_d(a,b)=G(A,b),
\]

where

\[
\boxed{
G(A,b)=A^4-560A^3b-9660A^2b^2-39200Ab^3+4900b^4.
}
\]

The six covers become the six small levels

\[
G(A,b)=c^2,
\qquad
c\in\{6,21,105,5,14,70\},
\]

in the order `(1,2),(1,7),(1,35),(3,5),(3,14),(3,70)`, together with `d|A`
and the previous coprimality conditions. Their identical discriminant is

\[
\operatorname{disc}(G)=2^{20}3^6 5^8 7^8.
\]

Thus no new prime is concealed in a large candidate.

The relevant real root is also explicit:

\[
\alpha
=2+\frac32\sqrt2+\frac15\sqrt{105}+\frac17\sqrt{210},
\]

and `beta=alpha^{-1}` is the algebraic integer

\[
\beta=140+105\sqrt2-14\sqrt{105}-10\sqrt{210}.
\]

In the biquadratic field `K=Q(sqrt(2),sqrt(105))`,

\[
\boxed{G(A,b)=N_{K/\mathbf Q}(A-\beta b).}
\]

Taking relative norms to the three quadratic subfields gives three simultaneous
Pell-type identities. If `G(A,b)=c^2`, then

\[
\begin{aligned}
(A^2-280Ab+70b^2)^2-2(210Ab)^2&=c^2,\\
(A^2-280Ab-2870b^2)^2
-105(28Ab+280b^2)^2&=c^2,\\
(A^2-280Ab-2030b^2)^2
-210(20Ab+140b^2)^2&=c^2.
\end{aligned}
\]

This is the important global structure. A huge solution remains a valid
candidate: the lower bound on `u` is not evidence against it. But it must be
the same small-norm element in all three quadratic directions while also lying
in the two-dimensional lattice `A-beta*b`. The problem is therefore a common
unit-orbit intersection in a biquadratic field, rather than six unrelated
large searches. The next descent should act on these three relative norms
simultaneously.

The first relative norm can be classified completely because
`Z[sqrt(2)]` is Euclidean. Up to multiplication by the norm-one unit
`3+2sqrt(2)`, the possible seeds at the six levels are

\[
\begin{array}{c|c}
c&\text{seeds }x+y\sqrt2\\ \hline
5&5\\
6&6\\
14&14,\ 22\pm12\sqrt2\\
21&21,\ 33\pm18\sqrt2\\
70&70,\ 110\pm60\sqrt2\\
105&105,\ 165\pm90\sqrt2.
\end{array}
\]

For the `d=7,14,35,70` covers, `d|A` and `gcd(b,d)=1` make both coefficients of
the relative norm divisible by `d`; this excludes the split seeds
at levels `21,14,105,70`. Hence every one of the six core covers lies on the
rational-seed orbit

\[
U+V\sqrt2=c(3-2\sqrt2)^m.
\]

The exponent is not being bounded. In fact it reconnects exactly to the
original negative-Pell index. Direct substitution of `H,D` gives

\[
\frac{U+V\sqrt2}{c}
=(50w-49v)-35(v-w)\sqrt2,
\]

and the residual orbit satisfies the identity

\[
\boxed{
(50w_n-49v_n)-35(v_n-w_n)\sqrt2
=(99-70\sqrt2)^n.
}
\]

Thus `m=3n`. The large branch is retained exactly, rather than rejected by a
size cutoff. It also explains why the `sqrt(2)` norm cannot finish the proof
alone: it is precisely the original recurrence in a different embedding. Any
new obstruction must couple it to at least one of the `sqrt(105)` or
`sqrt(210)` relative norms.

The coupling starts with an exact squareclass condition. For an element
`xi=A-beta*b` in a biquadratic field, the product of its three relative norms
is `xi^2*N(xi)`. Since `N(xi)=c^2`, the element

\[
c(99-70\sqrt2)^n
\,(29-2\sqrt{210})^{j+1}
\,(41-4\sqrt{105})^{k+1}
\]

must be a square in `K`. The `sqrt(2)` unit is already a square, so only the
parities of `j+1,k+1` matter. Exactly one parity pair works for each level:

\[
\begin{array}{c|c|c}
c&(j+1\bmod2,\ k+1\bmod2)&\rho_c\\ \hline
5&(0,1)&-10+\sqrt{105}\\
6&(1,1)&-42-30\sqrt2+4\sqrt{105}+3\sqrt{210}\\
14&(1,0)&-14+\sqrt{210}\\
21&(0,1)&21-2\sqrt{105}\\
70&(1,1)&140+105\sqrt2-14\sqrt{105}-10\sqrt{210}\\
105&(0,0)&\sqrt{105}.
\end{array}
\]

Here `rho_c` is the displayed square root after removing the even unit powers.
Combining these parities with
`j mod 10 in {0,9}` and `k mod 14 in {0,13}` gives, on every one of the six
levels,

\[
\left\lfloor\frac{j+1}{2}\right\rfloor\equiv0\pmod5,
\qquad
\left\lfloor\frac{k+1}{2}\right\rfloor\equiv0\pmod7.
\]

Thus the surviving unit lattice is generated by the fifth power of the
`sqrt(210)` unit and the seventh power of the `sqrt(105)` unit; this conclusion
does not place an upper bound on either exponent.

Thus any solution, however large, obeys the explicit global unit equation

\[
\boxed{
A-\beta b
=\pm\rho_c(7-5\sqrt2)^n
\,(29-2\sqrt{210})^{\lfloor(j+1)/2\rfloor}
\,(41-4\sqrt{105})^{\lfloor(k+1)/2\rfloor}.
}
\]

The right side must have coefficient vector proportional to

\[
(A-140b,\ -105b,\ 14b,\ 10b)
\]

in the basis `(1,sqrt(2),sqrt(105),sqrt(210))`. In particular its last three
coordinates must satisfy the exact two linear cuts

\[
2q_{\sqrt{105}}-7q_{\sqrt{210}}=0,
\qquad
2q_{\sqrt2}+21q_{\sqrt{210}}=0.
\]

This is now the global target: prove that those two coefficient currents never
vanish simultaneously for the allowed index congruences, or extract the
resulting large `(A,b)` if they do.

## 12. Vanishing-current descent

Write the three positive unit pairs as

\[
\begin{aligned}
(7-5\sqrt2)^n&=a-b\sqrt2,\\
(29-2\sqrt{210})^i&=c-d\sqrt{210},\\
(41-4\sqrt{105})^\ell&=e-f\sqrt{105},
\end{aligned}
\]

where `i=floor((j+1)/2)` is a multiple of 5 and
`ell=floor((k+1)/2)` is a multiple of 7.  No exponent is bounded here.
Expanding the two linear cuts and eliminating `f/e` produces only two
polynomials in the ratios `x=b/a` and `y=d/c`:

\[
\begin{aligned}
P(x,y)={}&42x^2y-7x^2+420xy^2-2x+735y^2-21y,\\
Q(x,y)={}&26460x^2y^2+3444x^2y+112x^2+840xy^2-4x\\
&\quad-11760y^2-1722y-63.
\end{aligned}
\]

Levels 5, 21, and 105 require `P=0`; levels 6, 14, and 70 require `Q=0`.
Clearing `a^2c^2` and using

\[
a^2-2b^2=(-1)^n,\qquad c^2-210d^2=1
\]

collapses the apparent quartics to bilinear currents.  If
`s=(-1)^n`, their numerators are

\[
\begin{aligned}
\widetilde P_s&=-2ab-7b^2-s(21cd-735d^2),\\
\widetilde Q_s&=-4ab-14b^2-s(1722cd+24990d^2+63).
\end{aligned}
\]

This immediately removes five covers.

* Level 21 has `s=-1`, so
  \[
  \widetilde P_{-1}=-2ab-7b^2-21d(35d-c)<0.
  \]
  For `d>0`, `c<35d` follows from `c^2=210d^2+1`; `d=0` is also
  strictly negative for a nonrecord point.
* Level 14 has `s=1`, and every term in `widetilde Q_1` is negative.  The
  same argument removes level 70.
* Level 105 has `s=-1` and is removed by the same strict
  `widetilde P_{-1}` sign as level 21.
* On level 6 the second, uneliminated current factors as
  \[
  3\{ae(c+14d)-14bf(c+15d)\}.
  \]
  Here `a,c,e` are odd and `d,f` are even.  Thus `ae(c+14d)` is odd
  while `14bf(c+15d)` is even, so this current cannot vanish.

Consequently the only quartic support still capable of carrying a nonrecord
solution is

\[
\boxed{(g,d,c)=(3,5,5).}
\]

For that level, simultaneous vanishing is equivalent to the two exact product
currents

\[
\boxed{
\begin{aligned}
b(2a+7b)&=21d(35d-c),\\
bce&=105adf.
\end{aligned}}
\]

These retain arbitrarily large candidates but expose a new divisibility
geometry.  The standard doubling identity for Pell coefficients gives

\[
v_2(b)=v_2(n),\qquad
v_2(d)=v_2(i)+1,\qquad
v_2(f)=v_2(\ell)+2.
\]

Since `a,c,e` are odd, the second product current therefore forces

\[
\boxed{v_2(n)=v_2(i)+v_2(\ell)+3.}
\]

In particular `8|n`.  Therefore `v2(b)>=3`, and the other factors in the first
product current have

\[
v_2(2a+7b)=1,\qquad v_2(35d-c)=0.
\]

Taking valuations in that first current gives

\[
v_2(n)+1=v_2(i)+1,
\quad\text{hence}\quad
v_2(n)=v_2(i).
\]

Combining the two current identities would require

\[
v_2(i)=v_2(i)+v_2(\ell)+3,
\]

which is impossible.  Thus level 5 also carries no nonrecord point.  All six
global squareclasses have now been eliminated without an upper bound on
`n,i,ell` and without walking the Pell orbits.  Within the primitive-support
reduction above, the record point `u=v=w=1` is therefore the unique positive
simultaneous solution.
