# Rank-six conductor attack

## Target and outcome

The live ICARM rank-six conductor to beat on 2026-08-29 is

```text
5,187,563,742
```

No rank-six snipe is certified here.  The attack does produce several
sub-record conductor or discriminant shapes, but exact Mordell--Weil
saturation exposes an additional relation in every arithmetically competitive
one.  Conversely, several hidden rank-six fibers are now certified exactly;
their conductors remain above the record.  These are useful hard negatives:
the conductor mechanism is working, and the obstruction is the coupling
between arithmetic size and the sixth height current.

## 1. Compatible support and sixth-section currents

Let

```text
T = t(1-t^2),   A = (1+t^2)^2.
```

The primitive normalized support current and the extra section at `x=r-s`
are

```text
U^2 = T*c*(c+1),
Z^2 = (c+1)*((c+1)*A + 12*(c-1)*T).
```

Putting `V=Z/U` eliminates `c` linearly.  The two square gates reduce to the
single norm conic

```text
W^2 = (A-12*T)*(V^2-24),
c   = (A-12*T)/(T*V^2-A-12*T).
```

This is the decisive reduction: rank-six capacity on a fixed `t` slice is a
conic, not a two-parameter search.  The wall-descent script parametrizes that
conic, pulls back the discriminant-zero divisor, and evaluates only continued
fraction convergents to its real branches.

The sharpest fibers were:

| `t` | generalized model `[0,0,a3,a4,0]` | discriminant | exact height result |
|---|---:|---:|---|
| `1/4` | `[0,0,833,-5341,0]` | `-7^6*541*51047` | `P_extra=P0-P2` |
| `1/5` | `[0,0,637,-3871,0]` | `-7^6*17^2*21563` | dependent after two halvings |
| `1/7` | `[0,0,125,-541,0]` | `3541990069` | dependent after one halving |
| `1/9` | `[0,0,287,-2221,0]` | `517988142757` | `r-s` dependent; hidden exact rank 6 |
| `1/11` | `[0,0,427,-2731,0]` | `406017834517` | `r-s` dependent; hidden exact rank 6 |

The `t=1/7` discriminant is itself below the target, so it would have been an
unconditional snipe if six sections were independent.  Its failure proves
that the closest discriminant characteristic meets a previously invisible
rank-loss current.

## 2. The second edge-difference current

The other support-edge abscissas give simpler square carriers.  At
`x=r+2*s`, rationality is equivalent to

```text
z^2 = (1+t^2)^2 + 12*T*(1+2*c),
```

which is linear in `c`.  The reciprocal edge `x=2*r+s` replaces `c` by
`1/c`.  This family was also pulled back to the discriminant wall.  Its
low-denominator fibers do not integralize competitively; even the first
`t=1/2` and `t=1/7` characteristics are far above the target.

## 3. Three split horizontal cubics

A more structural route is to add an entire third horizontal triple.  A
zero-sum support triple

```text
R=(a,b,-a-b)
```

has common quadratic support and product current

```text
m = a^2+a*b+b^2,
K_R = -a*b*(a+b).
```

It lies at height `h_R` on

```text
y^2 = x^3-m*x+a6
```

exactly when `h_R^2=a6+K_R`.  Thus three split cubics are found by solving
two differences of squares.  For fixed product currents this is a divisor
factorization, not a walk over Weierstrass coefficients.

The first norm `m=91` gives

```text
K = -330, -90, 90, 330.
```

Its smallest three-square solution is

```text
y^2 = x^3 - 91*x + 346,
Delta = -3,488,768,
```

with horizontal triples

```text
(-11,5,6; y=4), (-10,1,9; y=16), (-6,-5,11; y=26).
```

It has one additional exact relation and rank at most five.  The next solution
at `a6=1114` fails similarly.

An exact Eisenstein-norm/divisor classification for `m<=2000` found 51
integral three-split curves whose model discriminant is below the rank-six
target.  Exact PARI rank and conductor reduction was completed for all 51;
their ranks are between two and five.  Thus the complete classified layer is
closed, rather than merely screened modulo 2.

## 4. Symbolic factorization of the exposed height loss

Let `E` be the extra section at `x=r-s`, and take the rank-five basis

```text
P0, P_r, P_s, Q_-r, Q_-s.
```

For a base point `P=(x,h/2)`, the chord law gives

```text
x(P0+P) = (h-q)^2/(4*x^2)-x,
x(P0-P) = (h+q)^2/(4*x^2)-x.
```

Subtracting `x(E)=r-s`, factoring over `Q[c,t]`, and deleting only the
degenerate boundary `c*(c+1)*t*(t-1)*(t+1)=0` gives eight irreducible
first-shell factors:

| lattice direction (up to the sign of `E`) | primitive factor |
|---|---|
| `P0+P_r` | `c^2*t^2+c^2*t+c*t^2+c+2*t-2` |
| `P0-P_r` | `-c^2*t+c^2+c*t^2+c-2*t^2-2*t` |
| `P0+P_s` | `c*t^2+2*c*t-c+t^2+t` |
| `P0-P_s` | `c*t^2+2*c*t-c+t-1` |
| `P0+Q_-r` | `c*t^2-c*t+t^2-2*t-1` |
| `P0-Q_-r` | `-c*t-c+t^2-2*t-1` |
| `P0+Q_-s` | `-2*c^2*t-2*c^2+c*t^2+c+t^2-t` |
| `P0-Q_-s` | `2*c^2*t^2-2*c^2*t-c*t^2-c-t-1` |

Their product is the primitive first-shell sixth-height current
`D6^(1)`.  Equal abscissas are enough here: `E` is already on the curve, so
on any one of these divisors it equals the indicated lattice point or its
negative.  Thus `D6^(1)=0` is an exact dependence certificate, not a
reduction heuristic.

All three direct sixth-section failures in the wall descent are now
explained:

| fiber | vanishing primitive factor | exact consequence |
|---|---|---|
| `t=1/4, c=-12/7` | `c*t^2+2*c*t-c+t-1` | `E=+/-(P0-P_s)` |
| `t=1/9, c=-49/4` | `c*t^2-c*t+t^2-2*t-1` | `E=-(P0+Q_-r)` |
| `t=1/11, c=6/49` | `c*t^2+2*c*t-c+t^2+t` | `E=+/-(P0+P_s)` |

The `t=1/5, c=-10/7` fiber lies on the same `P0-P_s` sixth-loss sheet as
the quarter, but also on the independent base-five sheet

```text
B5 = 2*c*t^2-c*t-c-3*t-1 = 0,
P0+Q_-r = P_r+P_s+Q_-s.
```

This transverse double loss is exactly why its saturation needed two
halvings.  The `t=1/7, c=21/4` fiber misses all eight sixth factors; its sole
failure is instead the base sheet

```text
B7 = c*t^2+c*t+t-1 = 0,
P0+P_r = P_s.
```

The factorization remains simple after pullback to the norm conic.  For the
three sixth sheets actually hit by the descent, boundary units removed, it is

```text
P0+P_s:    [t(t+1)V]^2 - [5t^2+6t-1]^2,
P0-P_s:    [t^2+6t-5]^2 - [(t-1)V]^2,
P0+Q_-r:   t(t-1)(t^2-2t-1)V^2 - [5t^2-6t-1]^2.
```

So the closest discriminant characteristics did not encounter a mysterious
analytic height collapse.  They hit explicit rational norm walls.  These
walls can now be excluded before any model is integralized or saturated.

This is a complete factorization of the first coefficient shell
`E=+/-(P0+/-P)` for the four nonredundant adjacent base sections.  It is not a
claim that arbitrary higher-coefficient Mordell--Weil relations have a finite
algebraic product; those remain the only possible invisible part of the full
canonical-height determinant.

## 5. Height-safe descent and the reciprocal-edge near miss

Removing the six cross walls, the four adjacent base-shell walls, `B5`, and
all eight factors of `D6^(1)` before saturation changes the arithmetic scale
abruptly.  The smallest surviving rank-six specialization on the established
norm-conic slices is

```text
t=1/2, c=32/43,
[0,0,184900,-7856401,0],
Delta = -2^6*43^6*479*2699869.
```

Even its discriminant radical is `55,609,201,793`, already above the rank-six
target.  At `43`, `v(c4)=2` and `v(Delta)=6`, so the sixth power cannot be
removed by minimalization; the height-safe fixed-slice survivor is genuinely
too expensive locally.

The fixed-shape deformation explains why.  On the seventh slice, writing
`k=22+e`, the simultaneous square currents give

```text
c(e) = 4*(e^2+44e+2016)^2 /
       (17e^4+1664e^3-15808e^2-2644992e+3096576).
```

The numerator quadratic and denominator quartic have resultant

```text
2^26*3^4*7^2*383^2.
```

Therefore every transverse denominator prime outside `{2,3,7,383}` survives
reduction.  The primitive neighbors `k=21,23` retain rank six but inject the
new denominator primes `5,724,113` and `145,819`, respectively.  This is the
exact arithmetic reason that moving off the height wall is costly.

Holding `c` fixed instead turns the support square into a congruent-number
curve

```text
E_N: y^2=x^3-N^2*x,  t=-x/N.
```

The five low fibers give `N=15,30,21,5,330`.  Exact multiplication through
twelve levels produces no recurrence of either the `r-s` or `r+2s` square
current beyond the seed.  Translating by rational two-torsion merely applies
the Pythagorean transformations of `t`: for example
`(c,t)=(21/4,3/4)` reconstructs the identical model
`[0,0,125,-541,0]` as `(21/4,1/7)`, with the relabeled base relation
`P0+P_s=P_r`.

The previously omitted reciprocal edge `x=2r+s` has carrier

```text
C_(2r+s) = c^2*(1+t^2)^2 + 12*T*c*(c+2).
```

It is square on the quarter, fifth, and eleventh low fibers.  The quarter is
the strongest arithmetic near miss:

```text
y^2+833*y=x^3-5341*x,
Delta = -7^6*541*51047,
N = 7^2*541*51047 = 1,353,204,923.
```

That conductor would win.  Exact saturation nevertheless gives
`P_r+P_s+E=P0`.  Symbolically this is not a new accident: comparing the
reciprocal edge with `P0 +/- P_(-r-s)` gives precisely the already-factored
`R_-` and `R_+` sheets.  The fifth and eleventh reciprocal-edge points are
dependent for the same root-lattice reason.

Geometrically, `r-s`, `r+2s`, and `2r+s` form the three natural `A2` root
directions.  Their extra sections remain on the Weyl orbit of the original
support lattice.  The next orbit is the weight lattice, beginning with

```text
x=(2r+s)/3,  x=(r+2s)/3.
```

Their carriers are genuinely new rather than relabelings; neither is square
on any of the five low fibers.  A further attack must therefore construct a
weight-current conic deliberately, rather than testing more root edges.

### 5.1 The first weight-current quotient

For the lower weight `omega_2=(r+2s)/3`, interpolate opposite ordinate signs
between the two dependent lower-support coincidences `c=1` and `c=-4/5`.
With

```text
R = c*(c+1)*(1-2t-t^2)*(10c-1)/9,
```

the carrier defect factors as the known boundary components times one
quadratic `Q(c,t)`.  The discriminant of `Q` becomes, after
`u=t-1/t`,

```text
v^2 = 25u^4 + 20u^3 - 84u^2 + 80u + 400.
```

This quotient has the rational lifted point

```text
(u,v)=(-3/2,49/4),  t=1/2 or -2.
```

Its exact chord with `(0,20)` gives

```text
(u,v)=(-8/3,196/9),  t=1/3 or -3.
```

At both positive parameters the residual quadratic splits as

```text
Q(c,t) = unit*(c+9)*(5c-4).
```

Thus the first non-root orbit produces four extremely small arithmetic
presentations:

| `t` | `c` | generalized model | discriminant radical | saturation |
|---|---:|---:|---:|---|
| `1/2` | `-9` | `[0,0,2430,-53217,0]` | `1,203,594` | dependent |
| `1/2` | `4/5` | `[0,0,121500,-4446900,0]` | `707,910` | dependent |
| `1/3` | `-9` | `[0,0,46080,-2691072,0]` | `1,203,594` | dependent |
| `1/3` | `4/5` | `[0,0,288000,-14054400,0]` | `707,910` | dependent |

The two `t` values are Pythagorean presentations of the same two curves.  At
`c=-9`, the base lattice already has the coefficient-two relation
`P0+P_s=-2Q_-r`; at `c=4/5`, the weight point satisfies
`P0+P_s+Q_-s=-2E`.  The first relation and its `t -> (1-t)/(1+t)` mate are now
part of the symbolic pre-saturation gate.

The upper `omega_2` quotient is the sign transform of this quartic.  The
distinct upper `omega_1` quotient is

```text
v^2 = u^4 - 8u^3 + 60u^2 - 32u + 16.
```

Its complete first chord shell from `(0,4),(2,12),(-2,20)` has no rational
Pythagorean lift.  Fermat interpolation at infinity, boundary tangents and
boundary secants on the two lifted octics likewise leaves only irreducible
quadratic or cubic residuals.  This closes the low-degree weight interpolation
shell; the next move must use a genuinely new quotient-group point or a
different non-root orbit.

### 5.2 Doubled and mixed weight currents

The next non-root orbit begins with

```text
2*omega_2 = (2r+4s)/3.
```

Opposite-sign interpolation between its lower coincidences `c=1/4,-2`
factors the carrier defect as

```text
c*(c+1)*(c+2)*(4c-1)*Q_2(c,t),
```

where `Q_2` is quadratic in `c`.  Its discriminant quotient is

```text
v^2 = 16u^4 + 56u^3 + 537u^2 + 224u + 256.
```

Exact support secants give `(u,v)=(3/2,91/2),(8/3,728/9)`, hence
`t=2,3`.  At either lift

```text
Q_2(c,t) = unit*(7c+9)*(14c+5).
```

The arithmetically interesting fiber is

```text
c=-9/7, t=2,
[0,0,-476280,-38292912,0],
rad(Delta)=495892194,
tame conductor lower bound=3471245358.
```

It is below the rank-six record threshold, but its doubled-weight section
satisfies the exact relation

```text
P_r + Q_-r + Q_-s + E = O.
```

The next primitive mixed orbit

```text
2*omega_1+omega_2 = (5r+4s)/3
```

has quotient

```text
v^2 = u^4 - 46u^3 + 249u^2 - 184u + 16.
```

The same structural secants lift, now splitting the shape quadratic as
`(7c+9)*(7c+16)`.  They expose the very small model

```text
c=-16/7, t=2,
[0,0,-30481920,-7059612672,0]
  ~ [0,0,-17640,-340452,0],
rad(Delta)=5309178,
tame conductor lower bound=37164246.
```

Again the sixth section vanishes in the base lattice:

```text
P0 + Q_-r + Q_-s + E = O.
```

At `c=-9/7` the mixed section instead satisfies `E=P0+P_s`.  Thus these are
genuine conductor wins but not rank wins.

### 5.3 The exact simultaneous-lift recurrence

Both quotient quartics obey

```text
(u,v) -> (4/u,4v/u^2).
```

On `d^2=u^2+4`, put

```text
w=u+4/u,  z=v/u,  D=d*(u+2)/u.
```

Then `D^2=w*(w+4)`.  Parametrizing this conic by `rho=D/w` and converting
the remaining even quartic to a Jacobi 2-isogeny model gives

```text
E_d: y^2=(x-594)*(x^2-721476),  P_d=(1074,14400),
E_m: y^2=(x-666)*(x^2-425124),  P_m=(618,-1440).
```

The final lift back to rational `u` is exactly the square current
`w^2-16`.  Ordinary multiples `nP` and their rational 2-torsion cosets were
followed exactly through `n=28`; numerator and denominator sizes reach about
500 decimal digits.  Only `n=1` splits, giving `w=25/6` and
`u=3/2,8/3`; the torsion coset adds only the degenerate `w=-4` point and the
same lift.  This is a single Mordell--Weil recurrence, not a rational grid
search.  The reproducible implementation is `run_weight_lift_quotient.py`.

The next primitive orbit `3*omega_1+omega_2=(7r+5s)/3` has discriminant
quotient

```text
v^2 = 25u^4 - 1420u^3 + 5784u^2 - 5680u + 400.
```

Its support points and their involution closure generate no nondegenerate
Pythagorean lift in the complete first secant-and-tangent shell.  Unlike the
two prior orbits, it therefore supplies no initial simultaneous-lift
direction to saturate on the original quotient.  Reducing the true lift
quotient nevertheless exposes

```text
z^2 = 349*rho^4 - 1053*rho^2 + 729,
E_3: y^2=(x-1053)*(x^2-1017684),
P_3=(1728,36450).
```

The generator is exceptional at `rho=1` (`w=infinity`), but its higher
multiples are finite.  Exact evaluation of `nP_3` and `nP_3+T` through
`n=64` reaches roughly 1,345-digit coordinates and produces no split of
`w^2-16`; the torsion point contributes only the degenerate `w=-4` lift.

### 5.4 The rational A2 root-lattice family

The first integral root-lattice section not already represented by the seven
support points is

```text
x=(3+2c)r.
```

Opposite-sign interpolation between its lower coincidences `c=-3,-4/3`
leaves only

```text
3c*(t^2+2t-1)^2 + 50t*(t^2-1)=0.
```

Thus

```text
c(t)=50t*(1-t^2)/(3*(t^2+2t-1)^2),
R=-c*(c+1)*(1-2t-t^2)*(6c+13)/5.
```

This is a genuine rational rank-six family: the six-section saturation is
rank six at every tested nondegenerate structural specialization.  Its first
support point `t=2` has primitive model

```text
[0,0,-202300,-8671156,0],
rank=6,
conductor=1307806352018008.
```

The conductor current has two exact involution quotients.  With

```text
u=t-1/t,  w=u+4/u,
```

its degree-24 factor becomes `t^12 P12(u)=t^12*u^6*S6(w)`, where

```text
S6(w) = 1355211*w^6 - 23389236*w^5 - 46417860*w^4
        + 1345762080*w^3 + 1107138240*w^2
        + 15642192384*w - 136635218944.
```

The sextic is irreducible over every quadratic subfield suggested by its
discriminant and has full Galois group `S6`; it is not a concealed cubic norm.
The curve-shape quotient is especially simple:

```text
c=-50/(3*(w+4)).
```

In the true shape coordinate the sole essential factor is

```text
Q(c)=16c^6+48c^5-12c^4-1004c^3-3687c^2-4602c-1859
    =16*(c^2+c+1)^3 - 3*(c+1)^2*(6c+25)^2.
```

The canonical generalized model is

```text
q=-2*(c+1)*(6c+25)/3,
E_c: y^2+q^2*y=x^3-q^2*(c^2+c+1)*x,
Delta=4*q^6*Q(c).
```

The map `t -> c` is Klein-four Galois, generated by
`t -> -1/t` and `t -> (t+1)/(t-1)`.  Full deck-orbit traces of all eight
visible sections span only one rational direction.  At the exceptionally
smooth ramification endpoint `c=-25/12`, all quadratic traces likewise
collapse to one point.  Rational `t`, not merely rational `c`, really carries
the sixth rank direction.

### 5.5 The complete adjacent-root ray

The preceding family is the first member of the exact ray

```text
x=(n+(n-1)c)r,
c_n(t)=(n-1)*(n+2)^2*t*(1-t^2)
       /(n*(n-2)*(t^2+2t-1)^2).
```

The carrier root is

```text
ell_n=-(2c*n^2-4c*n+2n^2-n-2)/(n+2),
R_n=c*(c+1)*(1-2t-t^2)*ell_n.
```

This is a symbolic family in the lattice index, not an orbit enumeration.
At `t=2` its first decisive members are:

| `n` | A2 norm | `c` | primitive model | exact rank | exact conductor |
|---:|---:|---:|---|---:|---:|
| 3 | 7 | `-100/49` | `[0,0,-202300,-8671156,0]` | 6 | `1307806352018008` |
| 4 | 13 | `-81/49` | `[0,0,-3780,-44937,0]` | 6 | `1822615821216` |
| 5 | 21 | `-8/5` | `[0,0,-250,-1225,0]` | 4 | `4872100` |

The `n=5` denominator resonance cancels `(n+2)^2=7^2` against the support
denominator and gives a spectacular conductor win, but exact PARI descent
proves rank `[4,4]`; both nominal extra directions have collapsed.  The
lowest-height off-wall deformation `n=9/2` restores rank six but introduces
the bad prime `972245776321`.  Chords on its fixed-carrier cubic cannot repair
the jackpot: that cubic is the elliptic curve itself, so chords of known
points remain in the known rank-four subgroup.

### 5.6 The hidden half-root carrier

The exact generators on the ninth characteristic expose a section at

```text
x=3*r/2.
```

At `t=1/9` its current is

```text
z^2 = 4*c*(c+1)*(601*c^2+601*c+1350)/6561.
```

Putting `u=c(c+1)`, `y=81z/2`, and `k=y/u` gives

```text
u=1350/(k^2-601),
d^2=(k^2+4799)/(k^2-601),
c=(d-1)/2.
```

The biquadratic carrier is birational to

```text
Y^2=x^3+4198*x^2+11536796*x+48431469608.
```

It has exact rank three and rational two-torsion.  Its three small generators
pull back to the near miss `c=1/8`, the interpolation anchor `c=3/2`, and the
support point `c=1/2`.  The near miss has

```text
[0,0,820,-7300,0], rank=5, conductor=2266035800.
```

The unique quadratic through the anchor tangent at this point leaves the
rank-loss node and gives

```text
c=1215/23434, rank=6.
```

Its primitive discriminant contains the new prime
`7908773963869849067`; the tame screen is about `5.30e36`.  The next tangent
descendant has denominator

```text
2*233*376009*3451088033.
```

At each odd denominator prime the primitive model has
`v(c4)=2,v(Delta)=6`, so this is a monotone local obstruction, not merely
large coordinates.  The complete first three coefficient shells on the
fixed bad-support cover

```text
3*419*3863*v^2 =
76800*u^3-2595361*u^2+230400*u+76800
```

contain only the original point and its Weyl mate.

### 5.7 The rank-four `r-5s` carrier and its torsion coset

On the eleventh characteristic the generator at `x=19` is the section

```text
x=r-5*s.
```

After removing the visible square `c^2`, its carrier is already an elliptic
cubic.  At `t=1/11` it is birational to a rank-four curve with independent
points pulling back to

```text
c=1/4, 1/5, 17/88, 1/6
```

and rational two-torsion.  The torsion coset is essential: the known exact
rank-six seed is

```text
G1+G2+G3+T,
c=6/49,
[0,0,427,-2731,0],
conductor=406017834517.
```

Without `T`, the same coefficient vector lands at `c=123/220`.  This explains
why canonical-height calculations alone hid the useful carrier class.

The first shell has one apparent conductor win outside a support coincidence:

```text
c=-9/4,
[0,0,24156,-265716,0],
tame screen=4973362086.
```

Exact reduction gives rank five and conductor `59680345032`.  Its complete
dyadic/triadic twist adjustment has best conductor `6631149448`, at rank
zero.  The next independent rank-six shell point `c=35/64` has tame screen
`648430571366634`.  Thus both the ordinary and torsion-translated first shells
are closed below the record.

### 5.8 A rational conductor-factorization locus

For `U=c(c+1)` the essential discriminant is, up to a square unit,

```text
(U+1)^3 - lambda*U^2,
lambda = 27/64 * ((1+t^2)^2/(t*(1-t^2)))^2.
```

It acquires a rational linear factor at `U0=3*s^2-1` when

```text
w=-(1+t^2)^2/(t*(1-t^2))=-8*s^3/(3*s^2-1).
```

The angle lift factors as

```text
w^2-16 =
16*(s^2-1)^2*(4*s^2-1)/(3*s^2-1)^2.
```

Writing

```text
s=(r^2+1)/(4*r),
r=(p^2-1)/(2*p)
```

makes the entire locus rational, with one angle branch

```text
t=(p+1)*(p^2-4*p+1)/((p-1)*(p^2+4*p+1)).
```

The first nondegenerate slice is the deck orbit of `t=13/9,11/2`.  Its
adjacent-root specializations have exact rank six, including the formerly
rank-four index `n=5`, but their new wall prime `23` makes every conductor
screen enormous.

Trying to cancel the remaining wall factor leads to

```text
y^2=p^4-8*p^3+2*p^2+8*p+1.
```

Its Jacobian has minimal model `[0,-1,0,-4,4]`, exact rank zero, and torsion
`Z/4 x Z/2`.  All rational classes are boundary/torsion classes.  Hence the
most direct second square-factor cancellation has no nondegenerate rational
angle.

More generally, write the wall as

```text
p^4-8*p^3+2*p^2+8*p+1 = D*y^2
```

with squarefree `D`.  A projective local audit of the homogeneous binary
quartic modulo `2^8,3^5,5^3` and the squares of every prime through `23`
proves that every nontrivial squarefree `|D|<23` is locally obstructed.  The
only survivor is `D=1`, already closed by the rank-zero calculation above.
Thus `D=-23`, seeded by `p=2,3`, is the smallest possible nondegenerate wall
squareclass.

The `D=-23` Jacobian has minimal model

```text
[0,-1,0,-2292,-30780], rank=1, torsion=(Z/2)^2.
```

Its first canonical tangent descendants are

```text
p=-279/1363, -542/821,
```

which are deck presentations of the same adjacent-root fiber.  Although the
wall prime remains exactly `23`, the multiplicative discriminant current
acquires a 200-digit composite cofactor after the small factors
`7^2*373*4931*31957` are removed.  Hence preserving the minimal wall
squareclass alone does not preserve the conductor norm.

An explicit birational map to

```text
Y^2=X^3+22/23*X^2+4/23*X-280/12167
```

turns the tangent construction into the actual rank-one Mordell--Weil
recurrence, including all four two-torsion cosets.  For the adjacent `n=5`
rank-six direction, the first four recurrence classes have certified
conductor divisors or floors

```text
1: 411441293837
2:  96362009819
3: 575262767427105436499
4: >400000040000001.
```

The fourth certificate does not factor its 1,918-digit residual.  Exact trial
division removes every prime through `20,000,000`; the clean residual is
coprime to both generalized coefficients, divides the full discriminant, has
Fermat compositeness witness `2`, and is not a perfect power.  It therefore
contains at least two distinct multiplicative conductor primes larger than
the trial bound.  This closes the complete low-height portion of the minimal
wall recurrence without a global factorization.

There is a complementary attempt to square the *quadratic norm* factor first.
The norm conic has the universal point `U=-1`; to seed it at the rational
shape boundary `U=0`, one must impose simultaneously

```text
a^2=4*s^2-1,
h^2=1-3*s^2.
```

After parametrizing the first hyperbola, the remaining genus-one curve is

```text
Y^2=-3*q^4+10*q^2-3.
```

Its Jacobian has minimal model `[0,1,0,-4,-4]`, conductor `48`, exact rank
zero, and torsion `(Z/2)^2`.  All rational classes are the boundary/torsion
classes.  Thus neither the linear wall factor nor the quadratic norm factor
can be squared away through its natural universal seed.

## 6. Four split cubics from torus doubling

The sixth-height obstruction can be removed identically rather than avoided.
On the Eisenstein norm-one conic write

```text
X^2+3*b^2=4,             K=b^3-b,
S=X*(1-3*b^2),           S^2=4-27*K^2.
```

Doubling `(X+b*sqrt(-3))/2` replaces `b` by `X*b`; its cubic product
current is exactly `K2=S*K`.  The compatibility curve for the four heights
above `+/-K,+/-K2` is

```text
W^2=z*(z-1)*(z-S^2).
```

It has the universal rational point `(z,W)=(4,18*K)`.  The induced scale

```text
lambda=36/(16-S^2)
```

makes both within-pair height gates rational, and the remaining cross-pair
gate factors identically as

```text
(1+lambda*K2)/(1+lambda*K)
 = ((2*t^6+18*t^5+15*t^4-40*t^3-45*t^2-6*t+2)
    /(2*t^6-3*t^4+14*t^3+27*t^2+12*t+2))^2.
```

Thus every nondegenerate parameter carries four complete horizontal cubics,
twelve rational points, and no sixth-height square condition remains.
At `t=1/2`, exact PARI descent gives

```text
y^2=x^3-21609*x+1350441,
rank=6,
conductor=3945982521276
        =2^2*3^4*373*839*38917.
```

This is a genuine new rank-six family but not a conductor snipe.  The next
parameter `t=1/3` has exact rank seven and conductor
`9798274501605756`.

### 6.1 The remaining conductor is a pair of quadratic norms

For `t=p/q`, put

```text
D=p^2+p*q+q^2,       R=p*q*(p+q),
A=2*D^3,             B=9*R^2.
```

An integral representative is

```text
y^2=x^3-9*D^4*x + 9*D^6 + 81*N^2/4,
N=-p*q*(p-q)*(p+q)*(p+2*q)*(2*p+q).
```

Its complete discriminant factorization is

```text
Delta=3^6*F1*F2,
F1=(A-6*B)^2-3*(3*B)^2,
F2=A^2-3*B^2.
```

Both currents are norms from `Q(sqrt(3))`.  Since primitive `(p,q)` has
`gcd(D,R)=1`, `F1` and `F2` are coprime away from `2,3`; every other prime
in either current must split in `Q(sqrt(3))`.  This replaces two opaque
degree-twelve forms by one deterministic Pell action

```text
(a,b) -> (2*a+3*b,a+2*b).
```

The ideal fixed-current escape `F1=+/-4` or `F2=+/-4` is completely closed.
The negative cases fail elementary local congruences.  In the positive cases,
factoring the two neighboring Pell factors reduces the support gate to the
four Thue equations

```text
y^4-3*x^4=1,       3*y^4-x^4=1,
y^4-27*x^4=1,      27*y^4-x^4=1.
```

Exact PARI Thue certificates give only `(y,x)=(+/-1,0)` in the first and
third equations and no points in the other two.  These are precisely the
degenerate boundary `R=0`.  The fixed-norm unit orbit of the smallest
nondegenerate fiber was also audited through both coefficient directions to
height 64; the seed `(D,R)=(7,6)` is the sole cube/square return.

The apparent pair of norm currents is itself one current on a
determinant-three edge.  The exact transformation

```text
T(p,q)=(p-q,p+2*q)
```

satisfies

```text
F2(T(p,q)) = -3^5*F1(p,q).
```

If `T(p,q)` has content three, primitive reduction contributes its expected
degree-twelve factor `3^12`.  Projectively `T^2` is the ordinary Weyl deck
map `(p,q)->(-q,p+q)`.  Thus the two bad factors are the same quartic current
at the two vertices of one 3-isogeny edge, not independent degree-twelve
events.

Writing

```text
r=p*q*(p+q)/2,
g=(p-q)*(p+2*q)*(2*p+q)/2,
D^3=g^2+27*r^2,
```

gives the especially small forms

```text
F1/4=g^4-162*g^2*r^2-2187*r^4,
F2/4=g^4+ 54*g^2*r^2- 243*r^4.
```

Both have binary-quartic invariant `I=0`; this is the equianharmonic
3-neighbor behind the conductor split.  After the boundary norm `1`, the
next admissible core norms are `-11` and `13`.  Their complete unit orbits
are generated by `1+/-2*sqrt(3)` and `4+/-sqrt(3)`.  Full recurrence periods
modulo `9`, `8`, and `7`, respectively, contain no index at which the first
coefficient is a cube and the second is `18` times a square.  Hence both
norm classes are obstructed at every height, not merely through a bounded
integer search.

There is no rational fixed-point jackpot that identifies the two vertices.
Equating `T(p,q)` projectively with each of the twelve signed Eisenstein Weyl
matrices gives twelve binary quadratics.  Their discriminants are only
`-12,-3,12`, never rational squares.  Consequently no nonzero rational
support parameter can make `F1` and `F2` the same current by symmetry.  The
prime-support collapse exists only over `Q(sqrt(3))` or `Q(sqrt(-3))`.

The most powerful remaining conductor escape would make either quartic core
a square.  Both square covers are rank-zero genus-one curves:

```text
z^2=g^4+54*g^2*r^2-243*r^4   <->   y^2=x^3+1,
z^2=g^4-162*g^2*r^2-2187*r^4 <->   y^2=x^3-27.
```

Exact PARI descent gives ranks zero, with torsion `Z/6` and `Z/2`.  The
second cover has only boundary points.  The finite nonboundary torsion orbit
on the first has `g/r=+/-3`.  Pulling those two ratios back to the Eisenstein
support parameter gives

```text
t^3-3*t-1=0,       t^3+3*t^2-1=0.
```

Both are irreducible over `Q` by the rational-root test.  They describe
cyclic cubic supports, not three rational roots.  Hence neither conductor
core can be a nondegenerate rational square on the split-support family; in
particular, no even perfect power can produce the missing radical collapse.

Odd perfect powers are also constrained.  If `F2/4=C^3`, set `X=D^2`.
The coprime factorization

```text
(X-C)*(X^2+X*C+C^2)=972*r^4
```

has a unique 2- and 3-adic allocation:

```text
X-C=324*u^4,       X^2+X*C+C^2=3*v^4.
```

Eliminating `C` leaves

```text
(X-162*u^4)^2=v^4-8748*u^8.
```

For `u!=0` this is the genus-one quartic `w^2=x^4-8748`.  Its Jacobian
minimalizes to

```text
y^2=x^3+27*x,
rank=0, torsion=Z/2.
```

The torsion point is the quartic boundary at infinity.  Hence `F2/4` cannot
be a nontrivial cube on the primitive split-support family.  Together with
the square-cover result, the direct even- and odd-perfect-power mechanisms
for deleting the second conductor radical are closed globally.

The second current admits a sharper exact lower bound that does not assume
it is a perfect power.  Write

```text
c=F2/4=D^6-3*(18*r^2)^2.
```

For every fixed `c`, PARI's exact `bnfisintnorm` gives all elements of norm
`c` in `Z[sqrt(3)]` modulo multiplication by `2+sqrt(3)`.  Modulo an odd
prime, each resulting unit recurrence is finite.  It can therefore be
compared with the complete local image of the *coupled* split-support map

```text
(p,q) -> ((p^2+p*q+q^2)^3, 18*(p*q*(p+q)/2)^2).
```

This retains the relation between the cube and square coordinates that the
earlier separate residue tests discarded.  On the canonical primitive deck
`c=1 (mod 36)`.  There are 2,162 such signed classes with `|c|<38917`,
comprising 6,560 signed unit orbits.  Complete periods modulo primes at most
199 obstruct 6,559 of them.  The sole survivor is `c=1`, represented by
`1+0*sqrt(3)`, and is exactly the degenerate boundary `r=0`.

The bound is sharp.  On including `c=38917`, the first nonboundary survivor
is

```text
200-19*sqrt(3).
```

One unit step sends it to

```text
(200-19*sqrt(3))*(2+sqrt(3))
    =343+162*sqrt(3)
    =7^3+18*3^2*sqrt(3),
```

which is precisely the `t=1/2` rank-six fiber.  Thus `38917` is not merely
the smallest current seen in a parameter scan: it is the exact first
nondegenerate value of the second conductor current throughout this
four-split family.  Since that fiber's conductor is already
`3,945,982,521,276`, this family is sealed below the rank-six record.

### 6.2 Three asymmetric split cubics

Four horizontal cubics are unnecessary for rank six.  Retaining only the
currents

```text
+K, -K, S*K,       S^2+27*K^2=4,
```

leaves one height conic free.  Put `rho=c*K`, `z=1/(2*c)`.  The first two
heights are `rho^2*(1+/-z)`, while the torus-double height is rational on

```text
eta^2=4*c^2+4*S*c+1.
```

The chord `eta=1+ell*c` gives the complete rational family

```text
c=2*(2*S-ell)/(ell^2-4).
```

It carries three complete horizontal triples, nine visible points and
generic rank six.  This genuinely breaks the determinant-three pairing of
the four-split family.  At `t=1/2`, the structural chords give exact ranks

```text
ell=0: rank 8,
ell=4: rank 6.
```

The second fiber minimalizes to

```text
y^2=x^3-240100*x+42471025,
N=1066059801781300
 =2^2*5^2*131*251*421*770113.
```

It is not a snipe, but it is a new rank-bearing family rather than another
rank-loss wall.

The third support triple can be any torus multiple, not only the double.
With

```text
K_n/K=U_(n-1)(S/2),
```

the same height conic gives an exact adjacent multiplier ray.  At the
equianharmonic support cusp its distinguished chord is `ell=2*n`.  The first
new member, `n=3`, remains exact rank six at `t=1/2`, but its conductor rises
to

```text
3855023416647473402475.
```

Thus the first move along the support-multiplier ray is arithmetically
uphill.  Higher members are screened by conductor before rank descent; their
large unresolved discriminant factors are not treated as candidate evidence.

The chord `ell=4` is the exceptional equianharmonic blow-up of the support
cusp `t=-1/2`.  With `U=t+1/t=P/Q`, its first essential discriminant current
is the dihedral sextic norm

```text
A^2-7*B^2,
A=P^3+3*P^2*Q-21*P*Q^2-47*Q^3,
B=6*(P+2*Q)*Q^2.
```

For `t=p/q`, use `P=p^2+q^2`, `Q=p*q`.  Primitive local support forces the
norm residues used by the exact sieve.  Among all admissible `|c|<55151`,
there are 2,614 signed `8+3*sqrt(7)` unit orbits.  Complete coupled-support
periods modulo primes through 199 obstruct 2,610.  The four remaining signed
orbits have norms `1` and `729=3^6`; exact Thue solution shows every lift is
a support cusp.  At

```text
c=-55151=-131*421
```

the first nonboundary class appears.  Exact Thue solution gives only the
deck orbit of `t=1/2`.  Thus the sextic current of the displayed rank-six
fiber is the sharp first nondegenerate value, not an artifact of a short
parameter search.  The canonical Farey recurrence away from the cusp grows
immediately: its next fiber already has rank eight and conductor about
`1.64e23`.

### 6.3 Square-discriminant torus currents

The discriminant of the asymmetric family depends on

```text
B_curve=K*(c+1/(4*c)).
```

Setting this equal to a torus-multiple current makes
`4-27*B_curve^2` a rational square identically.  The second torus current
reduces to the double lift

```text
eta^2=2*(x^2+1),
H^2=-3*(x^4-14*x^2+1).
```

Eliminating `x^2` gives the rank-one quartic

```text
Z^2=-3*(eta^4-32*eta^2+64),
```

with useful 2-isogenous recurrence

```text
Y^2=X^3+96*X^2+576*X,
rank=1, torsion=Z/4.
```

A recurrence point lifts only when both `-X/3` and `(-X/3)/2-1` are
rational squares.  Exact multiples through 128 levels in all four torsion
cosets return only the two cusp classes.

The apparent height cutoff can in fact be removed.  Every simultaneous lift
projects before elimination to

```text
H^2=-3*(x^4-14*x^2+1).
```

This genus-one quartic has Jacobian

```text
y^2=x^3-39*x+70,
rank=0, torsion=(Z/2)^2.
```

Its leading coefficient `-3` gives no rational points at infinity, while
the four points `(x,H)=(+/-1,+/-6)` already account for all four Jacobian
torsion classes.  Therefore these are all of its rational points.  Each is
a cusp/rank-loss class, so the rank-one double recurrence is globally
closed at every height.

The third torus-current attempt is likewise closed without a height cutoff:
after the first conic reduction its two
remaining homogeneous quartics have no common nonzero projective point
modulo 11.  Thus the first two natural square-discriminant cancellations do
not yield a snipe, and this branch is sealed rather than merely screened.

The next multiplier is closed as well.  For the fourth torus current put

```text
L=S^3-2*S.
```

The rational-root and height gates are

```text
d^2=L^2-1,
2*c=L+d,
h^2=4*c*(L+S),
S^2+27*K^2=4.
```

The complete reduction modulo five has only four signed classes.  Every one
has `S=+/-1`, `L=-S`, and `d=h=0`.  This is not merely a mod-five screen.
Writing `delta=S-epsilon`, any nonzero 5-adic lift has, after deleting the
common even valuation of `delta`,

```text
d^2/h^2 = 1/2 (mod 5),
```

which is impossible because `1/2` is a nonsquare modulo five.  Hence every
`Q_5` lift has `S=+/-1` identically.  Pullback to the rational split-support
parameter gives the four monic cubics

```text
t^3-3*t-1,
t^3+3*t^2-1,
t^3-3*t^2-6*t-1,
t^3+6*t^2+3*t-1.
```

None has a rational root.  Thus the fourth-current cancellation has no
rational split-support fiber.  The double, triple and fourth torus-current
square cancellations are now globally closed.

The fifth and sixth multipliers close even faster.  For

```text
L_5=S^4-3*S^2+1,
```

the discriminant-root gate factors as

```text
d^2=S^2*(S^2-1)*(S^2-2)*(S^2-3).
```

After deleting the visible square and putting `X=S^2`, this is

```text
Y^2=(X-1)*(X-2)*(X-3)
   ~ y^2=x^3-x.
```

The elliptic curve has exact rank zero and torsion `(Z/2)^2`.  Its only
finite rational `X` values are `1,2,3`; only `X=1` is a rational square.
Thus again `S=+/-1`, which has no rational split-support preimage.

For

```text
L_6=S^5-4*S^3+3*S,
```

the complete support conic, root gate and height gate have no common class
over `F_11`.  Since `-27` is a nonsquare modulo eleven, the homogeneous
support conic has no missing points at infinity.  Consequently the sixth
multiplier is locally impossible.  The
square-discriminant torus multipliers `2,3,4,5,6` are all globally sealed;
the seventh is the next unclassified orbit.

## 6.4 Universal three-current conic

Three arbitrary equal-norm split supports with currents `K1,K2,K3` admit a
complete height parametrization.  Put

```text
A=K3-K1, B=K2-K1,
x=2*(B*s-A)/(A-B*s^2),
h2=1+x, h3=1+s*x,
lambda=(h2^2-1)/B.
```

After dilating every support root and height by `lambda`, all nine points lie
on

```text
y^2=x^3-m*lambda^2*x+lambda^2*(1-lambda*K1).
```

The first shell with three distinct absolute currents is `m=637`, with
currents `2484,4116,5916`.  The pole-adjacent chord `s=3/2` gives the exact
rank-seven minimal model

```text
y^2=x^3-4283188*x-1358839712,
N=7866100357847360.
```

Thus the conic genuinely escapes the old rank-four relation, although its
first independent member is too large arithmetically.

The self-collision lift `m=3*637^2` is exceptional: its residual degree-eight
discriminant current splits completely into

```text
(3s-14)(3s+8)(38s-13)(75s-143)
(152s+91)(209s-243)(209s-175)(729s-1001).
```

The cheapest boundary neighbor is `s=2`, with conductor `1698523941345`, but
exact rank five.  Its six visible representatives obey

```text
P1-P2+P3-P4+P5-4*P6=O.
```

Function-field addition leaves `s-2` and one irreducible degree-fifteen
factor after boundary removal, so this particular rational rank loss does
not recur elsewhere on the conic.

## 6.5 Four rational supports on the small `m=91` shell

The first square pullback `91*7^2=4459` has four integral current orbits.
Dividing their supports by seven gives four rational supports still of norm
91, with currents

```text
90, 63510/343, 87450/343, 330.
```

Requiring all four heights reduces to

```text
H^2=18496*s^4-128248*s^3+358401*s^2-323449*s+117649.
```

Its invariants and exact-rank-one minimal Jacobian are

```text
I=30118645593,
J=-6610138671777930,
y^2=x^3-10039548531*x+244819950806590.
```

The first nonboundary group-law translate is forced by
`s^2*(s-1)*(5*s-77)`, hence `s=77/5` (equivalently `392/139` from the other
tangent).  It gives twelve rational support points and an exact rank-eight
curve.  Its conductor is `19322156386220234804295192`, however, with residual
primes `24942037` and `93924767`.  The first four-height recurrence layer is
independent but not a conductor snipe.

## 6.6 Generalized base height and the rank-five walls

The normalized choice `h1=1` is not intrinsic.  For a fixed provisional
base height `H`, the same conic is

```text
x=2*H*(B*s-A)/(A-B*s^2),
h2=H+x, h3=H+s*x,
lambda=(h2^2-H^2)/B,
A0=H^2-lambda*K1.
```

This lets the universal family pass exactly through the small integral
rank-five curves before taking a transverse Farey step.

The smallest exact rank-five wall is

```text
y^2=x^3-532*x+4420,
N=37396136,
(K,h)=(-3744,26),(-1056,58),(1056,74).
```

It is `(H,s,lambda)=(26,3/2,1)`.  Its first exits expose the governing
anticorrelation: `s=2` and the octic-square return `s=10/7` both minimize to
the rank-four curve `[0,0,0,-133,493]`, while the independent side has rank
six or seven only above `10^13`.

The strongest later wall occurs at `m=217`:

```text
y^2=x^3-217*x+1585,
N=107827292,
(K,h)=(-1224,19),(-624,31),(624,47),(1224,53).
```

The primitive exit `s=3` has four complete horizontal cubics and minimizes
to

```text
y^2=x^3-1953*x+48177,
N=2921869260,
rank=5.
```

It is arithmetically below the rank-six record but retains one extra section
relation.  Requiring all four heights gives the quartic

```text
Y^2=625*s^4-7700*s^3+34016*s^2-31416*s+10404.
```

Its invariants are

```text
I=509408656,
J=-22046274731392,
```

and its minimal Jacobian

```text
[0,-1,0,-10612680,12761798400]
```

has exact rank two.  The complete first group-law shell collapses to two new
curve classes.  The useful class is represented by the integral slope
`s=-2` and gives

```text
y^2=x^3-313348*x+67164772,
N=350937240520
 =2^3*5*19^2*2797*8689,
rank=6.
```

This is the best independent conductor produced by the current attack.  The
other first-shell class and the complete tangent/adjacent second shells have
conductors above `10^39`; their many slope presentations collapse to the
same two isomorphism classes.  Thus the rank-two recurrence is bounded by
curve classes rather than by a denominator scan.

## 6.7 Horizontal-support reservoirs and the quadratic norm recurrence

For a fixed wall

```text
y^2=x^3-m*x+a6,
```

a horizontal product current `K` has split-cubic discriminant and height

```text
delta^2=4*m^3-27*K^2,
h^2=a6+K.
```

Eliminating `K` gives the genus-one support reservoir

```text
delta^2=4*m^3-27*(h^2-a6)^2.
```

The complete exact rank-five wall audit gives the following reservoir ranks
in increasing base-conductor order:

```text
(m,a6)       supports  reservoir rank
(532,4420)       3          3
(481,3961)       3          4
(1204,15844)     4          5
(217,1585)       4          4
(364,4084)       3          3
(1519,23026)     3          3
(589,5629)       3          3
(301,2509)       3          4
(637,5965)       4          4
(949,11509)      3          3
(301,3349)       4          4
(1957,33205)     3          5
(469,4861)       3          4
(469,2341)       3          5
```

Square discriminant is not enough: the depressed cubic can remain a cyclic
irreducible cubic.  Exact factorization of every adjacent Mordell--Weil deck
shell filters through this three-cover.  Only two walls expose new split
supports.  The `m=217` wall has

```text
(5407,-183,-5224)/19^2,
h=282377/19^3,
K=5169048744/19^6.
```

The `m=481` wall has two:

```text
(2441,1824,-4265)/13^2,
h=11383/13^3,
K=-18989417760/13^6,

(34176,-12041,-22135)/37^2,
h=4389953/37^3,
K=9108845036160/37^6.
```

All primitive transverse exits from the `m=217` support have conductor above
`10^104`.  The complete `13`-adic, `37`-adic, and mixed positive-height exits
on `m=481` are also enormous.  The independent signed height chambers are
arithmetically better but still fail sharply.  The smallest signed member is
the `s=-391/117` exit; its `j` denominator factors as

```text
682861904975160824507603
 * 14215412254771516087192583.
```

Both factors are proven multiplicative conductor primes, already far above
the rank-six target.

The useful theoretical residue is not the Farey chord but the dilation
`lambda`.  For the `13`-adic support with integral currents `-3600,-2280`, the
residual discriminant current is

```text
R(lambda)=27*H^4-54*H^2*K1*lambda
          +(27*K1^2-4*481^3)*lambda^2.
```

It is a norm from `Q(sqrt(1443))`, whose fundamental unit is
`38+sqrt(1443)`.  Scaling a norm factor by its unit powers gives an exact
squareclass-preserving recurrence `lambda_n`.  None of the first four powers
on either side returns to either rational height gate; only `n=0` gives the
wall `lambda=1`.

Globally, the fixed-squareclass curve has

```text
u^2=H^2+B*lambda,
v^2=H^2+A*lambda,
w^2=R(lambda)/R(1).
```

Its genus-three Jacobian splits into the elliptic quotients carried by
`u*w`, `v*w`, and `u*v*w`.  Their exact PARI rank bounds are

```text
u*w:     [1,3]
v*w:     [0,2]
u*v*w:   [2,2]
```

The last curve is the Prym.  Its minimal model is

```text
[0,1,0,-1088332527181202692852209675776,
 -357197621008566738848718735836593172099301360]
```

and its exact rank-two generators have denominator patterns `41^2/41^3`
and `19^2/19^3`.  The first maps back to the wall `lambda=1`; the second is a
genuine hidden Prym direction but makes both lift values nonsquares.  The
complete first and second coefficient shells contain only the wall deck
returns

```text
plus/minus G_41,
plus/minus (G_41+2*G_19),
```

and one degenerate `u=0` boundary.  Thus the adjacent support-reservoir and
fixed-squareclass norm layers are both bounded without a rational-height
walk.  The best proven rank-six conductor remains `350937240520`.

## 6.8 Conductor-first split-discriminant shell

The integral wall classification above used a model-discriminant ceiling.
That is not a conductor theorem: a much larger discriminant can still have a
small radical.  The first conductor-first correction is the surface

```text
m=3*d^2,
4*m^3-27*a6^2=27*(2*d^3-a6)*(2*d^3+a6).
```

The support-reservoir attack itself selects `d=13,19,37,41`.  Exact
difference-of-squares height compatibility leaves only three curves across
those four norms, with exact PARI results

```text
d=13: [0,0,0,-507,28730],   N=511056,   rank=3
d=19: [0,0,0,-1083,21818],  N=1599120,  rank=3
d=19: [0,0,0,-1083,165818], N=45012240, rank=4
d=37,41: no three-split height fiber.
```

Thus the split-discriminant surface does produce exceptionally small
conductors, but its entire first structure-selected shell stops below rank
five.  The strongest base curve is the last `d=19` member.  Its own support
reservoir has exact rank two, with minimal model

```text
[0,0,0,-82063414443,-9048324342385942],
N=285077520.
```

There is therefore no reservoir-rank surplus to furnish the two missing
directions.  The `N=45012240`, rank-four base is arithmetically excellent,
but it needs an additional independent support mechanism rather than another
horizontal-current recurrence.

## 6.9 Fixed-current height recurrence at `d=19`

The three supported currents on the `m=1083` wall are

```text
-13718, -10582, +10582.
```

Writing `c` for the height over `-13718`, the remaining two heights satisfy

```text
u^2-c^2=3136,
v^2-c^2=24300.
```

Putting `p=u+c` leaves the genus-one gate

```text
Y^2=p^4+90928*p^2+3136^2,
c=(p-3136/p)/2.
```

Its minimal Jacobian is

```text
[0,1,0,-10919160,13009804308],
N=8888880,
rank=2,
torsion=(Z/2)^2.
```

The first non-wall class is `p=5488/153`, `c=-27610/1071`.  Its target
minimizes to

```text
[0,0,0,-17591437151883,29774631841839625482],
N=1531196716427262062640,
rank=5.
```

The recurrence therefore genuinely gains one rank, although not at record
arithmetic.

Raw denominator size is not the obstruction.  If `c=e/d` is reduced and
`q` is away from `2,3,19`, the canonical integral model has

```text
v_q(a4)=4*k,
v_q(a6)=4*k,
v_q(Delta)=8*k,       k=v_q(d).
```

Scaling down `floor(2*k/3)` times leaves

```text
k mod 3:             0    1    2
v_q(Delta_min):      0    8    4
conductor exponent:  0    2    2.
```

Thus arbitrarily large cube denominators disappear completely.  The exact
remaining current is

```text
Delta=-432*c^2*(c^2+27436).
```

The natural fourth support would require `w^2=c^2+27436`.  Every such lift
would give an affine rational point on

```text
Z^2=(c^2+3136)*(c^2+27436).
```

This pointed quartic has minimal Jacobian

```text
[0,-1,0,-40981640,-76514264400],
N=31920,
rank=0,
torsion=Z/2.
```

Its two rational classes are exactly its two points at infinity.  Hence the
fourth-height lift is closed at every Mordell--Weil level.

## 6.10 The paired `d=13` recurrence

The other conductor-first wall has currents `-4394,-506,+4394` and base
heights `156,168,182`.  Its fixed differences are

```text
u^2-c^2=3888,
w^2-c^2=8788,
```

and its parameter quartic is

```text
Y^2=p^4+27376*p^2+3888^2.
```

The minimal Jacobian

```text
[0,1,0,-1212036,101211264],
N=10920,
rank=1,
torsion=(Z/2)^2
```

produces the first new height `c=828/55`.  Its target is

```text
[0,0,0,-4639366875,127902256996250],
N=26159086800,
rank=4.
```

The next class `c=31975684/477369` also has exact rank four and conductor
`2404621935171114329542789008`.  The missing `+506` height is globally
closed by the rank-zero, `Z/2` pointed quotient

```text
[0,1,0,-6371736,-6192519840],
N=212520.
```

The first new `d=13` discriminant norm exposes the split primes `7` and
`373`.  The attack promotes the smaller prime `7` to the next Eisenstein
support direction instead of treating it as an accidental conductor factor.

## 6.11 Norm current becomes support geometry: the `d=7` branch

For `d=7`, the support norm and four currents are

```text
m=147,
K=+/-686, +/-286,
686-286=400=20^2.
```

The three-height recurrence is

```text
V^2=p^4+3088*p^2+400^2.
```

Using the rational boundary `p=0` as origin gives

```text
Epar: Y^2=X^3+X^2-14916*X+205884,
N=34320,
Epar(Q)=Z*G + (Z/2)^2,
G=(-30,792).
```

The pointed-quartic map factors to the small height current

```text
c=(24300-(X-114)^2)/Y.
```

Every torsion translate of `n*G` changes only the sign of `c`, so there is
one target class per absolute recurrence index.  The generator maps to the
exceptionally small rank-three curve

```text
c=9/2,
[0,0,1,-147,706],
N=50121.
```

The double maps to

```text
c=76719/1148,
[0,0,0,-15957501882672,184268088879537135620],
```

with the exact certificate

```text
rank=6,
N=1750328941213617099804
 =2^2*3^3*7^2*41^2*107*239*7693969249.
```

This is not a conductor snipe, but it is the cleanest rank jump in the
conductor-first attack: one parameter doubling moves directly from rank
three to exact rank six.

The sixth direction is invisible to the horizontal supports.  The five
visible representatives have canonical-height rank five, while

```text
x=1187446/82369,
y=7342183887/94559612
```

raises the target to rank six and occurs explicitly in the exact PARI basis.
The missing `+686` height is nevertheless impossible: its pointed quotient
has minimal model

```text
[0,-1,0,-202616,-34012320],
N=1680,
rank=0,
torsion=Z/2.
```

For the rank-six double, the complete away current is

```text
e=76719=3*107*239,
d=1148=2^2*7*41,
e^2+1372*d^2=7693969249 (prime).
```

Reduction of `G` modulo `7,41,107,239` places every even index on the bad
divisor `c*(c^2+1372)=0` or a height pole.  The orbit orders are `4,4,8,8`.
A pole valuation divisible by three can still minimalize away, so this exact
orbit statement is not promoted to a false global conductor theorem.

The bad primes form a branching elliptic divisibility tree, not a numerical
sequence.  For example, the apparently plausible continuations `471` and
`503` resolve exactly as follows.  The meaningful prime in `471=3*157` is
`157`; `G` has order ten modulo `157`, and the pole divisor occurs at indices
`5,10`.  Indeed `157` divides the denominator of `c(5*G)`.  Modulo `503`, the
order is `130` and the pole occurs only at `65,130`.  Thus `503` is a real
deeper division label, but not a chronological successor to `239`.

Finally, the conductor-`50121` seed has residual norm prime `5569`.  Since
`5569=5^2+5*72+72^2`, promoting it again gives support currents

```text
K=+/-345431270018, +/-324684513218,
345431270018+324684513218=818606^2.
```

Two associated three-height parameter Jacobians have exact rank one.  This
is the active next norm-transport branch.

## 6.12 The `471` label resolves through `d=157`

The proposed continuation `471` is not itself a support norm.  Its useful
part is `471=3*157`, and the split prime has Eisenstein representation

```text
157=1^2+1*12+12^2.
```

Promoting it gives `m=3*157^2=73947` and currents

```text
C=7739786, K=7082714,
C-K=657072=3*468^2,
C+K=14822500=3850^2,
2*C=15479572=4*157^3.
```

The faithful three-height parameter curve has minimal model

```text
[0,1,0,-4383308336316,3507799008524937984],
N=425387802840,
rank=1,
torsion=(Z/2)^2,
G=(-18774951/16,169757713125/64).
```

The first generator height is

```text
c=-601029/292.
```

Its target minimizes to

```text
[0,0,0,-33599435635632,115996701569790580580],
N=21537291821479500445236
 =2^2*3^2*11*13*37*73^2*277*467*164024401,
rank=5.
```

Thus `157` is a genuine division branch, but its first target is already
sealed arithmetically.  Its smallest new split residual prime is `37`, which
opens a much tighter surface.

## 6.13 The rank-two `d=37` surface

For the representation `37=3^2+3*4+4^2`, the four currents give

```text
C=101306, K=89206,
A=C-K=12100=110^2,
B=C+K=190512.
```

The simultaneous support-height gate

```text
u^2=c^2+A,
v^2=c^2+B
```

has minimal Jacobian

```text
[0,1,0,-711167436,7193953224864],
N=412131720,
rank=2,
torsion=(Z/2)^2,
G1=(-15516,3807000),
G2=(223461/4,95270175/8).
```

The pointed map is exact.  If `(X,Y)` lies on this minimal model, put

```text
U=16*X-245944,
V=64*Y,
p=V*24200/(U^2-24200^2),
c=(p-12100/p)/2.
```

The complete first coefficient shell collapses to four target classes:

```text
c=231/2,          N=1795858911,                  rank=4,
c=2379/10,        N=13874778166275,              rank=5,
c=50589/68,       N=204456469404651156,           rank=5,
c=1361899/28272,  N=18806397523562338875666045873, rank=6.
```

This is the sharpest transition yet: one rank-two shell crosses exact ranks
four, five, and six, and its rank-four endpoint is already below the rank-six
record conductor.  The next two complete box shells have no conductor near
the target; the smallest second-shell conductor is
`21613716697783014670071996`, and every third-shell conductor is larger
still.  This seals the direct parameter recurrence without treating a small
coefficient as a proxy for arithmetic size.

The small endpoint is

```text
E: y^2+y=x^3-4107*x+114646,
N=1795858911,
rank=4.
```

An exact Mordell--Weil basis is

```text
(4,313), (26,159), (37,115), (70,412).
```

In the completed short model the two non-support points have coordinates

```text
(4,627/2), (70,825/2).
```

The line through them can be written

```text
x=66*n-128,
y=231/2+99*n.
```

Substitution factors identically as

```text
y^2-(x^3-4107*x+458585/4)
 =-3267*(n-2)*(n-3)*(88*n-75).
```

The third intersection is `(-287/4,1599/8)`, exactly
`-(P_4+P_70)`.  Thus the newly exposed current is an ordinary chord and not
a fifth direction.

The useful information is instead the pair of invisible height currents

```text
D4 =84942,
D70=156816.
```

Preserving both first gives the even quartic

```text
Y^2=p^4+457380*p^2+84942^2,
c=(p-84942/p)/2.
```

Its exact rank-two Weierstrass model is

```text
[0,457380,0,-28860573456,-13200249087305280],
N=82368,
G1=(-387684,91998720),
G2=(-339768,100911096).
```

The base height `231/2` is the primitive generator `G2`, up to rational
two-torsion.  A non-base point on this joint quotient for which both
`c^2+12100` and `c^2+190512` are squares would retain the two invisible
directions while escaping the visible rank-loss wall.  It is therefore the
smallest exact remaining rank-six problem.

Finite Mordell--Weil reduction makes the obstruction visible without a
rational-height walk.  The primes

```text
37,41,73,107,157,239,503
```

leave the base and first admit the non-base coefficient vector `(18,31)`.
That vector has canonical height about `2790`, but `31` and `47` immediately
make its second support current a nonsquare.  Adding those two exact gates
pushes the first locally admissible vector to `(120,90)`, of canonical height
about `60470`; `17` then kills its first support current.  The underdetermined
prime labels are therefore real local branches, but their intersection
creates a large forbidden moat around the only known rational lift.

Finally, the only structure-supported quadratic twist of the small endpoint
that retains sub-record conductor is the `-3` twist.  It has the same
conductor `1795858911`, root number `-1`, and exact rank one.  The
conductor-neutral twist escape is closed.

## 7. Interpretation

The root-lattice ray exposes the coupled obstruction sharply.  Arithmetic
height can collapse by six orders of magnitude in one lattice step, but here
the sixth-height determinant loses two ranks at exactly that step.  Moving
off the wall restores rank immediately and restores a huge discriminant
prime just as immediately.

The compatible-current obstruction is now factored on its entire adjacent
coefficient shell.  The next specialization should impose
`D6^(1)*B5*B7 != 0` symbolically, before arithmetic presentation.  If a
surviving fiber still loses rank, its saturation trace supplies the next
coefficient shell rather than sending the attack back into a blind search.

## Reproduce

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_collision_family/run_rank_six_wall_descent.py \
  --slice seventh --max-denominator 100 --show 8 --certify-count 0

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_collision_family/run_rank_six_edge_wall_descent.py \
  --current edge --q 7 --max-denominator 100 --show 12

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_collision_family/factor_height_loss_current.py

PYTHONPATH=experiments/elliptic_collision_family \
  python3 experiments/elliptic_collision_family/run_fixed_shape_orbit.py \
  --multiples 12

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_collision_family/run_height_safe_wall_descent.py \
  --max-denominator 100

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=experiments/elliptic_collision_family \
  python3 experiments/elliptic_collision_family/run_weight_current_attack.py

python3 experiments/elliptic_collision_family/run_weight_lift_quotient.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/elliptic_collision_family/run_root_lattice_rank_six_descent.py

python3 experiments/elliptic_collision_family/run_quadratic_trace_endpoint.py

python3 experiments/elliptic_collision_family/run_root_lattice_deck_descent.py

gp -q -f experiments/elliptic_collision_family/jackpot_curve_pari.gp

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/elliptic_collision_family/test_weight_current_attack.py

/opt/homebrew/bin/gp -q \
  experiments/elliptic_collision_family/four_split_torus_pari.gp

python3 \
  experiments/elliptic_collision_family/run_rank_five_support_reservoir_audit.py

python3 \
  experiments/elliptic_collision_family/run_support_reservoir_attack.py \
  --gp-split-shell | /opt/homebrew/bin/gp -q

python3 \
  experiments/elliptic_collision_family/run_m481_split_lift_attack.py | \
  /opt/homebrew/bin/gp -q

python3 \
  experiments/elliptic_collision_family/run_m481_norm_unit_attack.py

/opt/homebrew/bin/gp -q \
  experiments/elliptic_collision_family/m481_norm_unit_pari.gp

python3 \
  experiments/elliptic_collision_family/run_split_discriminant_surface_audit.py

python3 \
  experiments/elliptic_collision_family/run_m1083_fixed_current_height_recurrence.py

/opt/homebrew/bin/gp -q \
  experiments/elliptic_collision_family/m1083_fixed_current_pari.gp

python3 \
  experiments/elliptic_collision_family/run_d7_norm_transport_attack.py

/opt/homebrew/bin/gp -q \
  experiments/elliptic_collision_family/d7_norm_transport_pari.gp

python3 \
  experiments/elliptic_collision_family/run_d37_rank_two_transport_attack.py

/opt/homebrew/bin/gp -q \
  experiments/elliptic_collision_family/d37_rank_two_transport_pari.gp

python3 \
  experiments/elliptic_collision_family/run_four_split_norm_class_sieve.py \
  --bound 38916 --out /tmp/four_split_norm_below_38917.txt

PYTHONPATH=experiments/elliptic_collision_family \
  python3 experiments/elliptic_collision_family/run_asymmetric_three_split_attack.py

PYTHONPATH=experiments/elliptic_collision_family \
  python3 experiments/elliptic_collision_family/run_asymmetric_sextic_norm_sieve.py \
  --bound 55150

/opt/homebrew/bin/gp -q -s 200000000 \
  experiments/elliptic_collision_family/asymmetric_three_split_pari.gp

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_collision_family/factor_three_current_rank_loss.py

/opt/homebrew/bin/gp -q \
  experiments/elliptic_collision_family/three_current_height_conic_pari.gp
```

The exact section and saturation helpers are exercised by
`test_elliptic_collision_family.py`.
