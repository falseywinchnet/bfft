# Elliptic rank as a Mellin current

## Outcome

Among the representations already present in this repository, the modular
Mellin current retains the most useful information about both sides of the
rank/conductor problem.

- A Weierstrass model exposes rational points but hides the conductor behind
  local minimization.
- A Mordell--Weil height lattice exposes rank but largely forgets how the bad
  fibers are spatially arranged.
- A transported divisor pencil exposes persistent support but can silently
  lose section independence, as the adjacent experiment demonstrated.
- The Mellin current puts conductor into its spatial dilation and rank into
  exact current moments.

For the rank-five conductor record, that current has a particularly simple
three-shell form.

## 1. Current coordinate

Let `f` be the weight-two modular form of an elliptic curve of conductor `N`.
Put

\[
G(t)=e^t f\!\left(\frac{i e^t}{\sqrt N}\right).
\]

After centering the completed `L`-function at `s=1`, its Mellin transform becomes

\[
\Lambda(1+z)=\int_{-\infty}^{\infty}G(t)e^{zt}\,dt.
\]

For root number `-1`, the Fricke involution makes `G` odd.  Hence

\[
\Lambda(1+z)
=2\sum_{j\ge0}\frac{z^{2j+1}}{(2j+1)!}
\int_0^\infty t^{2j+1}G(t)\,dt.
\]

Analytic rank at least five is therefore exactly

\[
M_1=2\int_0^\infty tG(t)\,dt=0,
\qquad
M_3=2\int_0^\infty t^3G(t)\,dt=0.
\]

Rank exactly five additionally requires `M_5 != 0`.  The root number already
supplies the first central zero; the two current constraints supply the other
four.

The Fourier expansion

\[
G(t)=e^t\sum_{n\ge1}a_n
\exp\!\left(-\frac{2\pi n e^t}{\sqrt N}\right)
\]

shows why this representation couples the two quantities we care about:
`sqrt(N)/(2*pi)` is the current's spatial scale, while the integer Hecke
coefficients determine its signed population.

This current is also literally an elliptic differential.  If

\[
\varphi:X_0(N)\longrightarrow E
\]

is the modular parametrization, then, up to its rational normalization,

\[
\varphi^*\omega_E=2\pi i f(z)\,dz.
\]

Along the Fricke geodesic `z=i*exp(t)/sqrt(N)`, this becomes

\[
\varphi^*\omega_E=-\frac{2\pi}{\sqrt N}G(t)\,dt.
\]

Thus continual integration of `G(t) dt` is exact Abel--Jacobi transport of
the invariant differential, not a visual analogy.  The Fricke involution
folds the family at `t=0` and supplies its orientation symmetry.

Define the repeated tail integrals

\[
A_j(t)=\int_t^\infty
\frac{(u-t)^{j-1}}{(j-1)!}G(u)\,du.
\]

Then `A_2(0)=M1/2` and `A_4(0)=M3/12`.  The rank-five condition says that the
second and fourth integrated differential currents both arrive at the Fricke
boundary with zero trace, while the sixth does not.  This is the precise
version of “continually integrate the differentials until simpler support is
reached.”

## 2. Measured record signature

The probe computes the curve's `a_n` directly from exact finite-field point
counts and Hecke recurrence.  It uses 30,000 coefficients; it does not query a
curve table or search other models.

For

\[
y^2+y=x^3-79x+342,
\qquad N=19\,047\,851,
\]

the relative cancellation ratios are

| moment | `abs(M_k) / integral abs(2 t^k G)` |
|---|---:|
| 1 | `4.95e-17` |
| 3 | `6.59e-17` |
| 5 | `2.65e-1` |
| 7 | `5.07e-1` |

The rank-one control `y^2+y=x^3-x`, of conductor 37, has cancellation ratio
one at all four moments: its positive-half current never develops a genuine
sign change.

Ignoring the truncation-sized value at `t=0`, the record current changes sign
at

\[
t_1=3.37978554\ldots,
\qquad
t_2=6.59276076\ldots.
\]

It has exactly three shells, with signs `+,-,+`.  Their moment contributions
are

| moment | inner `+` | middle `-` | outer `+` | sum |
|---|---:|---:|---:|---:|
| `M1` | 2334.1406 | -4286.7078 | 1952.5672 | `-2.3e-13` |
| `M3` | 10656.7619 | -117844.9835 | 107188.2216 | `-4.0e-11` |
| `M5` | 63690.3007 | -3492327.0715 | 5953057.9078 | 2524421.1370 |

The first two sums vanish at double-precision noise; the fifth does not.

There is an even cleaner scale coordinate.  Put

\[
\alpha=\frac{2\pi}{\sqrt N},
\qquad s=\alpha e^t,
\qquad F(s)=\sum_{n\ge1}a_ne^{-ns}.
\]

Then

\[
G(t)\,dt=\frac1\alpha F(s)\,ds,
\]

and the moment laws become

\[
\int_\alpha^\infty
\left(\log\frac{s}{\alpha}\right)^kF(s)\,ds=0,
\qquad k=1,3.
\]

The zero locations are properties of the coefficient current `F`; conductor
only chooses its left boundary and logarithmic origin.  For the record the two
dimensionless zero locations are approximately

\[
s_1=0.042275,
\qquad s_2=1.05065.
\]

For a proposed multiplicative coefficient current, the first equation pins
`alpha` as its logarithmic balance point.  The third asks whether the current
has zero logarithmic skew about that same boundary.  When both hold,

\[
N=\left(\frac{2\pi}{\alpha}\right)^2
\]

is derived rather than scanned.  This is the useful inverse-design form of
the conductor problem.

A second probe decomposes the same moments into dyadic coefficient packets.
Those packets alternate with large nonlocal cancellations: for example the
`n=1` and `n=2:3` contributions to `M1` are about `8290` and `-14097`, and
subsequent bands continue to reverse sign.  The tail beyond `n=8192` is
negligible, but there is no packetwise telescope.  This rejects the DIP packet
tree as the privileged coordinate for this curve.  The Fricke spatial
coordinate is much simpler: the whole coefficient population becomes only
three oriented shells.

## 3. Exact support interpretation

Set

\[
u=t^2,
\qquad d\nu=2tG(t)\,dt.
\]

Then rank five says

\[
\int d\nu=0,
\qquad
\int u\,d\nu=0,
\qquad
\int u^2\,d\nu\ne0.
\]

Thus the rank-five curve is represented by three signed support shells whose
total current and first support barycenter both vanish, but whose curvature
moment survives.  Two sign changes are the minimum permitted by these two
orthogonality constraints.  The record realizes that minimum rather than a
complicated oscillatory cancellation.

This gives a general shell theorem.  If the root number is `-1` and the
analytic rank is at least `2m+1`, then

\[
\int_0^\infty u^j\,d\nu=0,
\qquad j=0,\ldots,m-1.
\]

A nonzero signed measure annihilating every polynomial of degree below `m`
must change sign at least `m` times.  Otherwise a polynomial with roots placed
at all its sign changes would have degree below `m` and the same sign as the
measure, making its integral strictly positive.  Consequently:

\[
\boxed{\text{analytic rank }2m+1
\Longrightarrow\text{ at least }m+1\text{ Fricke shells}.}
\]

For the record, `m=2` and equality holds.  If `a=t1^2` and `b=t2^2`, then
`(u-a)(u-b)` has the same sign pattern `+,-,+` as (d\nu).  The two vanished
lower moments give

\[
\int (u-a)(u-b)\,d\nu=\int u^2\,d\nu=M_5>0.
\]

So the positive fifth moment is forced by support ordering once the first and
third moments cancel.  It is not an incidental floating-point observation.

This corrects the earlier geometric analogy.  Rational points are not the
support units.  The modular current is the support, and Mordell--Weil rank is
shadowed by exact annihilated moments of that support.

Strictly, these moments see analytic rank.  A record submission still needs
five independent rational points (or another algebraic-rank certificate).
The current representation is an inverse-design and exclusion coordinate,
not a replacement for that final arithmetic proof.

## 4. Pathway suggested by the representation

The current moments can be written

\[
M_k(N)=\sum_{n\ge1}a_n W_k\!\left(\frac{2\pi n}{\sqrt N}\right)
\]

where the kernels are explicitly

\[
W_k(a)=2\int_1^\infty (\log x)^k e^{-ax}\,dx.
\]

A theory-first conductor
descent can therefore work in coefficient-current space:

1. combine `M3-rho*M1` to make an annihilating kernel localized at one shell;
2. use the Hasse bounds and Hecke recurrences to enclose the unobserved tail,
   rather than enumerate curves;
3. ask whether a three-shell `+,-,+` current can satisfy both exact balances
   when `N < 19,047,851`;
4. if the inequalities isolate an admissible Hecke signature, reconstruct
   the curve only at the end.

The immediate theoretical target is a variation-diminishing inequality for
the exponential kernel.  If two annihilated odd moments force a minimum
Fricke scale for every integral multiplicative Hecke current, it becomes a
conductor lower bound.  If the inequality has a sharp escape case, that escape
case supplies a candidate coefficient signature without walking through
Weierstrass models.

This is also where the repository's earlier ideas meet cleanly:

- the Eikonal two-current law contributes exact moment conservation;
- the boundary-motive work warns that branch count is not independent depth;
- the zeta experiment warns against generic operator compression;
- the elliptic-pi endpoint collapse suggests seeking a telescoped shell law,
  not carrying the full interior state.

The multiplicative refinement gives a concrete inverse target.  A two-zone
Frobenius model, with one negative phase through a prime cutoff and neutral
phase afterward, solves both moments along the following sub-record branch:

| cutoff | normalized phase | derived conductor |
|---:|---:|---:|
| 83 | -0.8560 | 4.10 million |
| 97 | -0.8079 | 6.08 million |
| 107 | -0.7473 | 10.44 million |
| 127 | -0.6970 | 17.01 million |

The known curve is negative at every prime through 107, but its mean phase is
only `-0.6181`; this explains its larger conductor.  The actionable snipe
signature is to retain five independent rational sections while strengthening
that chamber or extending an approximately `-0.70` chamber through 127.  The
model is a moment solution, not yet a realizable elliptic curve.

The companion experiment `../elliptic_collision_family/` now supplies the
realizable algebraic side: a record-passing rational family with seven
sections and generic rank at least five.  Its first canonical neighbor
preserves rank but expands conductor and breaks the negative phase chamber.
The remaining attack is to constrain a rational path on that family by the
Mellin-current frontier, so phase support and discriminant support are
transported together.

## 5. Reproduce

```sh
python3 experiments/elliptic_mellin_current/test_mellin_current.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_mellin_current/run_probe.py

python3 experiments/elliptic_mellin_current/analyze_shells.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_mellin_current/run_packet_probe.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_mellin_current/run_relaxed_cone.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_mellin_current/run_separator_bound.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_mellin_current/run_coherent_phase_model.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_mellin_current/run_two_zone_phase_model.py
```

The checked-in JSON files are copied M4 outputs of the focused current,
coefficient-packet, Euler-degree, separator, and phase-model probes.
