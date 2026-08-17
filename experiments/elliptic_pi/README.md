# Radical-free elliptic pi iterations

The four boundary points `0, infinity, a, -a`, with `a=sqrt(3)/2`, map under
`X=t/(t-a)` to `0, 1, infinity, 1/2`.  Their double cover is the Legendre curve

```text
Y^2 = X(X-1)(X-1/2),
```

with `lambda=1/2` and `j=1728`.  This experiment implements the resulting
elliptic contraction using polynomial branch selection rather than explicit
radicals.

## Stable implicit maps

The direct quartic elimination

```text
(1-x)^4 - (1-y^4)(1+x)^4 = 0
```

subtracts numbers close to one at the cusp.  Expanding and collecting gives
the stable equivalent

```text
y^4(1+x)^4 - 8x(1+x^2) = 0,       x ~ y^4/8.
```

The cubic Borwein map similarly becomes

```text
s^3(1+2x)^3 - 9x(1+x+x^2) = 0,   x ~ s^3/9.
```

Both select the small positive root by precision-doubling Newton.  Thus every
root extraction costs `O(Mul(n))`, while only `O(log n)` isogeny steps are
needed.  This puts the construction in the `O(Mul(n) log n)` AGM class.

### Optimized implicit quartic kernel

Direct Newton on the quartic was spending one division per precision stage
and solving each vanishing root to full *relative* precision.  The optimized
native path uses a better auxiliary polynomial.  Put `c=y^4`, select the root
`s` near one of

```text
(1-c)s^4 - 1 = 0,
```

and iterate

```text
s_next = s(5-(1-c)s^4)/4.
```

This Newton step uses only squarings and multiplications.  With
`r=(1-c)s^3`, the desired small root is reconstructed without cancellation:

```text
x = c/((1+r)^2(1+r^2)).
```

The implementation also exploits four pieces of precision structure:

1. Since `x~c/8`, a root of binary exponent `e` needs only `p+e` relative
   bits to provide `p` bits of absolute accuracy.
2. The seed `s=1` already has about `-log2(c)` correct bits, so each deep-cusp
   solve starts at that inherited precision instead of restarting at 96 bits.
3. Once `c^2` is below the guard threshold, `x=c/8+O(c^2)` is returned
   immediately.
4. The outer update carries its computed `y^4` into the next root solve,
   avoiding duplicate full-precision squarings.  The fixed initial quadratic
   is likewise solved by multiplication-only inverse-square-root Newton.

The operation is still polynomial branch selection plus rational arithmetic;
no library square root or fourth root is used by the optimized implicit path.

The period corrections are the classical Borwein cubic and quartic updates;
only the radical evaluations have been eliminated.  This distinction matters:
the polynomial presentation changes the available fusion and implementation
constants, but not the mathematical sequence or its asymptotic complexity.

## Direct degree-3 isogeny on the collision curve

The cubic Borwein iteration above uses the signature-3 parametrization; it
should not be confused with the degree-3 modular equation for the original
Legendre `lambda` coordinate.  The latter has now also been implemented.
Eliminating the auxiliary fourth root from the classical degree-3 relation
gives `P_3(u,v)=0`, where

```text
P_3(u,v) = v^4
 + (-256u^3 + 384u^2 - 132u)v^3
 + ( 384u^3 - 762u^2 + 384u)v^2
 + (-132u^3 + 384u^2 - 256u)v
 + u^4.
```

The selected root is `v=lambda(3*tau) ~ u^3/256`.  Starting at `u=1/2`, it is
`0.001290359062227362895...`, agreeing with the theta-function definition of
`lambda(3i)`.  This is the first direct odd-degree isogeny map on the actual
four-point collision curve.

### Direct period-alpha correction

Let `v=f_N(u)`, `q=f_N'(u)`, and let `K(u)` denote the complete elliptic
integral with Legendre parameter `u`. Differentiating the modular lambda
coordinate with respect to `tau` gives the squared period multiplier

```text
M = (K(u)/K(v))^2 = N v(1-v)/(u(1-u)q).
```

For Ramanujan's Legendre alpha,

```text
alpha(r) = pi/(4K(u)^2) - sqrt(r)(E(u)/K(u)-1),
```

the correction in the collision-curve normalization is

```text
alpha(N^2 r) = M alpha(r)
             + sqrt(r)[N v - M u + u(1-u) dM/du].
```

For an implicit map `P(u,v)=0`, no elliptic functions or radicals enter:

```text
q  = -P_u/P_v,
q' = -(P_uu + 2P_uv q + P_vv q^2)/P_v,

d(log M)/du = q(1-2v)/(v(1-v))
              - (1-2u)/(u(1-u)) - q'/q.
```

The implementation verifies the first degree-3 update independently against
the defining `K,E` expression and iterates it to pi.

### Alpha-free terminal period transport

The separate alpha state can be removed. Define

```text
F(u) = (2/pi)K(sqrt(u)),  G=F^2,  h=G'/G,
R = G(v)/G(u) = q u(1-u)/(N v(1-v)),
S = d(log R)/du
  = q'/q + (1-2u)/(u(1-u)) - q(1-2v)/(v(1-v)).
```

Then `G(u)=G(v)/R` and `h(u)=q h(v)-S`. Streaming `N` degree-3
steps uses

```text
R_product *= R
B          -= Q_product S
Q_product *= q
```

and reconstructs

```text
G_0 = G_N/R_product,
h_0 = Q_product h_N + B.
```

At the deep cusp, `F`, `G`, and `h` come from a few terms of

```text
F(u) = sum_k binom(2k,k)^2 u^k/16^k.
```

Finally, the collision value `u_0=1/2` is self-complementary. Legendre's
relation gives `G(1/2)h(1/2)=4/pi`, hence

```text
pi = 4/(G_0 h_0).
```

This identity and the complete backward transport have been verified through
10,000 decimal digits.

### Telescoping the complete correction chain

The apparent products and weighted sum above are not independent state.  Let
`f=f_3` be the selected branch of `P_3(u,f(u))=0`, compose it `k` times, and
write

```text
w = f^k(u_0),       D = (f^k)'(u_0),       E = (f^k)''(u_0),
L = 3^k,            A(x) = (1-2x)/(x(1-x)).
```

The period ratios telescope exactly:

```text
R_total = product_j R_j
        = D u_0(1-u_0)/(L w(1-w)).
```

Their complete logarithmic correction is just the logarithmic derivative of
that endpoint expression:

```text
T = d(log R_total)/du_0 = E/D + A(u_0) - D A(w).
```

Consequently the backward period reconstruction is product- and sum-free:

```text
G_0 = G(w) L w(1-w)/(D u_0(1-u_0)),
h_0 = D[h(w)+A(w)] - E/D - A(u_0).
```

For the collision value `u_0=1/2`, this simplifies to

```text
G_0 = 4 G(w)Lw(1-w)/D,
h_0 = D[h(w)+A(w)] - E/D,
pi  = D/(G(w)Lw(1-w)h_0).
```

Only the two-jet of the composite map remains.  It can be streamed without
forming any correction factors:

```text
D_next = q D,
E_next = q' D^2 + q E,
```

starting from `(D,E)=(1,0)`.  This removes the explicit `R_product`,
`Q_product`, and weighted `S` accumulation.  It does not remove sequential
composition: `D` is mathematically the same derivative product in compressed
endpoint form.

### Terminal nome: no period factors or derivatives

There is a stronger collapse special to the starting point.  Let
`nome(u)` be the elliptic nome inverse to the modular lambda function.  The
degree-3 branch obeys

```text
nome(f_3(u)) = nome(u)^3.
```

Since `u_0=lambda(i)=1/2`, `nome(u_0)=exp(-pi)`.  Thus, after `k` descents,

```text
nome(w) = exp(-pi 3^k),
pi = -log(nome(w))/3^k.                         (exact)
```

At the cusp,

```text
lambda(q) = 16q(1-8q+44q^2+...),
nome(w) = (w/16)(1+w/2+O(w^2)).
```

Therefore the fully product-free extraction used by the implementation is

```text
pi = -log(w/16)/3^k + O(w/3^k).
```

The descent depth is chosen so that this omitted term lies beyond all guard
bits.  This version carries only the terminal lambda value and one integer
power of three; it needs no `alpha`, `z`, periods, derivatives, products, or
weighted additions.

### Incremental nome correction: exact identity, wrong precision geometry

The normalized root also gives an exact logarithmic telescope.  With

```text
v = u^3 r(u)/256,       L(u) = -log(u/16),
```

one has

```text
L(v) = 3L(u) - log(r(u)),
L(u) = [L(v)+log(r(u))]/3.
```

The second form reconstructs `L(u_0)=log(32)`, not pi.  Solving the finite
telescope in the other direction and taking the cusp limit gives the useful
identity

```text
pi = log(32) - sum_{j>=0} log(r(u_j))/3^(j+1).
```

This formula was implemented both directly and with each `log1p(r-1)` rounded
to the minimum precision required by its weighted contribution.  It is
correct, but it does **not** create the proposed geometric precision schedule.
If `e_j=log2|r(u_j)-1|`, then an `n`-bit final result requires approximately

```text
p_j = n + e_j - (j+1)log2(3) + O(1)
```

relative bits in the `j`-th logarithm.  The weight saves only `O(j)` bits; the
small magnitude saves `-e_j` bits additively.  At a one-million-bit target the
first measured requirements are

```text
999999, 999987, 999959, 999875, 999629, 998893, ... bits.
```

Thus several logarithms remain essentially full precision.  On the M4 Mini,
the native relaxed implementation took 3.57 seconds at one million bits,
versus 0.985 seconds for the single terminal logarithm (the deliberately
full-precision incremental prototype took 5.58 seconds).  At 10,000 decimal
digits, the Python versions took 0.637 seconds incremental versus 0.140
seconds terminal.

Algebraically, the incremental formula merely factors the terminal logarithm
of an algebraic product into a sum of algebraic logarithms.  Without a faster
shared or batched logarithm evaluator, it increases work rather than changing
the `O(Mul(n) log n)`-type bottleneck.

### Normalized cusp root

Putting `v=u^3 r/256` changes `P_3=0` into

```text
1 - (33u^2-96u+64)r/64
  + 3u^3(64u^2-127u+64)r^2/32768
  - u^6(64u^2-96u+33)r^3/4194304
  + u^8 r^4/4294967296 = 0.
```

Its cusp limit is `1-r=0`. This improves the coordinate conditioning, although
mpmath's floating exponent already handles the vanishing unnormalized root
well: over six 10,000-digit descents the current Python implementation took
0.074 s normalized versus 0.050 s raw. It is expected to be more useful in a
fixed-point or carefully fused native implementation.

## Self-correction audit

The classical forward period state is not self-correcting. For the quartic update

```text
z_next = z(1+x)^4 - C x(1+x+x^2),
```

the propagation derivative is

```text
d z_next / d z = (1+x)^4 -> 1.
```

Thus an error already present in `z` survives essentially unchanged; the cubic
update has the same issue with derivative `(1+2x)^2 -> 1`. The terminal
`(G,h)` reconstruction above fixes this particular defect: uncertainty in
`h_N` is multiplied by `Q_product`, and `q ~ 3u^2/256`, so that uncertainty is
rapidly annihilated.

The formulas above answer the algebraic telescoping question, but do not prove
an `O(Mul(n))` algorithm.  The endpoint two-jet still takes `Theta(log n)`
sequential isogeny/derivative evaluations.  The terminal-nome form removes
those derivatives as well, but its final high-precision logarithm belongs to
the same currently known `O(Mul(n) log n)` elementary-function regime.  A
genuine asymptotic improvement would require either a faster logarithm or a
way to obtain its specially structured value while sharing work with the
modular descent.

## Run on the M4 Mini

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_pi/test_elliptic_pi.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_pi/elliptic_pi.py --digits 1000
```

The C++ GMP/MPFR benchmark now separates all relevant quartic root kernels,
the optimized and legacy AGM loops, terminal nome, and three binary-split CM
series.  Use the reduced, interleaved suite for stable comparisons:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  make -C experiments/elliptic_pi run \
  BITS=1000000 REPEATS=7 SUITE=core
```

The harness warms every algorithm, rotates their order each repetition, and
reports minimum, median, maximum, correctness bits, and paired median ratios
to AGM.  This matters on the Mini: short independent runs show substantial
frequency-state variation even when within-run ratios remain stable.

## Two-sheet and Bruun sine/cosine audit

The BFFT source contains two different sine/cosine mechanisms.  The
table-polynomial routine in `src/detail/MAG_REPRESENT_KERNEL.hpp` selects one
of 256 precomputed `(sin,cos)` pairs and applies a cubic residual rotation.
It is an effective binary64 phase kernel, but the table was generated with
high-precision `pi` and its range reduction already uses `2*pi`; it consumes
pi rather than constructing it.

The constructor in `src/detail/bruun_dif_kernel.hpp` is more structurally
relevant.  It builds the complete Bruun angle tree from

```text
alpha -> alpha/2,  pi/2-alpha/2
ce = sqrt((1+c)/2),  se = s/(2*ce).
```

Thus one stable root produces both complementary sinusoidal children.  If
used backward as a pi formula, however, this is a Viete-type small-angle
limit: halving the angle gains only about two error bits per level and needs
`Theta(bits)` dependent roots.

The exact analogue for the degree-2 lambda descent is revealing.  Put

```text
s = sqrt(1-u), r = (1-s)/(1+s), v = r^2.
```

The two radical sheets are `v` and `1/v`, with root-free symmetric data

```text
v + 1/v = 16/u^2 - 16/u + 2,     v(1/v)=1.
```

For the normalized period `F(u)={}_2F_1(1/2,1/2;1;u)`, the two algebraic
Landen multipliers are `1+r` and `1+1/r`; remarkably, both their trace and
their norm equal `4/u`.  This initially looks like a product-free two-branch
transport.

The period continuation prevents that simplification.  The large sheet lies
beyond the hypergeometric branch cut and satisfies

```text
F(1/v) = sqrt(v) [F(v) - i F(1-v)].
```

It therefore carries the complementary logarithmic period, not another
independent contracting copy of `F(v)`.  As `v -> 0`, `F(v)` becomes a cheap
power series while `F(1-v)` contains the full logarithmic cusp information.
Keeping both sheets reconstructs the same two-dimensional period system—and
the same terminal-logarithm obstruction—as the terminal-nome method.  The
new regression `test_lambda2_two_sheet_bifurcation` verifies all trace, norm,
Landen, and continuation identities numerically.

### Computational bifurcation

The useful bifurcation is inside each AGM step rather than between the two
Landen sheets.  Once

```text
m=(a+b)/2, d=(a-b)/2
```

are available, these branches are independent:

```text
period lane:    d^2 -> t - 2^n d^2
geometric lane: a*b -> sqrt(a*b)
```

`pi_agm_parallel_hybrid` keeps one persistent worker on the geometric lane
while the calling thread computes the period lane, then uses the 65% terminal
defect switch.  It falls back to the scalar hybrid below 100,000 bits.  This
is coarse two-core parallelism, not the one-core NEON/SSE mechanism used by
BFFT: arbitrary-precision limb multiplication already occupies the integer
backend and cannot be packed into the fixed-width sine/cosine lanes directly.

The wall-clock improvement is real:

| requested bits | direct AGM | scalar hybrid | two-lane hybrid | gain vs direct |
|---:|---:|---:|---:|---:|
| 100,000 | 0.004670 | 0.004472 | 0.003744 | 19.9% |
| 1,000,000 | 0.1040 | 0.1021 | 0.0858 | 17.4% |
| 10,000,000 | 1.7668 | 1.6975 | 1.4051 | 20.5% |

An equal-resource comparison is less favorable.  Chudnovsky's balanced split
tree accepts the same top-level two-way decomposition:

| requested bits | two-lane AGM | two-way Chudnovsky | AGM/Chudnovsky |
|---:|---:|---:|---:|
| 1,000,000 | 0.0861 | 0.0259 | 3.32x |
| 10,000,000 | 1.4051 | 0.4516 | 3.11x |

Thus the BFFT-style observation does improve the lemniscatic engine, but it
does not improve its standing against a naturally tree-parallel series.  Use
`SUITE=parallel` for this comparison.

### Single-core shared-transform candidate

At FFT-multiplication sizes, the two independent AGM lanes have additional
structure.  If fixed-point mantissas are represented as coefficient vectors
`A` and `B`, the period lane needs `(A-B)^2` while the geometric lane needs
`A*B`.  Both outputs can share the same two forward transforms:

```text
P_hat = A_hat * B_hat
D_hat = (A_hat - B_hat)^2
```

followed by two inverse transforms.  Separate multiplication and squaring
need three forward and two inverse transforms; the paired form needs two
forward and two inverse transforms.  This is the closest arbitrary-precision
analogue of the Bruun complementary sine/cosine construction: two children
are obtained from one transformed parent pair.

`bfft_paired_convolution.cpp` benchmarks exactly this transform graph using
BFFT native-order spectra.  It verifies the shared outputs against separate
`A*B` and `(A-B)^2` convolutions before timing them.  At length 262,144 the
product outputs agreed exactly in the test data and the square paths differed
by about `5.7e-16` relatively, consistent with alternate FFT roundoff:

| BFFT length | separate | shared | shared/separate | saving |
|---:|---:|---:|---:|---:|
| 262,144 | 0.003729 s | 0.002189 s | 0.587x | 41.3% |
| 1,048,576 | 0.012678 s | 0.010409 s | 0.821x | 17.9% |

Run it with, for example:

```sh
make paired_convolution CONV_N=1048576 REPEATS=7
```

This is not yet an MPFR replacement.  A production kernel must choose a
roundoff-safe signed digit base, zero-pad for linear rather than cyclic
convolution, normalize carries in both outputs, and pass the product into the
root engine without converting through MPFR.  But unlike generic
`mpn_mul_n -> mpn_sqrtrem` fusion, it demonstrates a concrete single-core
operation that GMP cannot infer from two independent calls: the forward
transforms are shared across the geometric and period branches.

There are two further algebraic reductions.  First put

```text
U_hat = A_hat + B_hat
V_hat = A_hat - B_hat.
```

Then

```text
(A*B)_hat       = (U_hat^2 - V_hat^2)/4
((A-B)^2)_hat   = V_hat^2.
```

A complex square has bilinear rank two,
`z^2=(zr-zi)(zr+zi) + i(2*zr*zi)`, so both requested quadratic forms use four
real pointwise multiplications per bin.  The direct shared formula uses seven
(a generic complex product plus a complex square).  At length 262,144 this
smaller pointwise graph was about 5--10% faster than the direct shared kernel.
At length 1,048,576 it was approximately tied or slightly slower: the extra
add/subtract traffic matters more than three multiplications once the
transforms and memory traffic dominate.  Thus four is the useful arithmetic
lower bound, but not an unconditional wall-clock win.

The larger reduction streams state across AGM iterations.  Linearity gives

```text
A_hat_next = U_hat/2
```

without another forward transform.  Furthermore, the period state is only
used after the loop, so accumulate

```text
C_hat += 2^j V_hat^2/4
```

and inverse-transform the complete correction once.  With a stopping guard
that makes the last `d^2` smaller than half an output ulp, the already
materialized last `A*B` is also an adequate numerator because
`((A+B)/2)^2=A*B+d^2`.  For `k` direct-root steps the transform counts are
therefore

| schedule | forward transforms | inverse transforms |
|---|---:|---:|
| eager shared | `2k` | `2k` |
| streamed, exact numerator | `k` | `k+2` |
| streamed, guarded numerator | `k` | `k+1` |

The transform of `a_0=1` is analytic and is not included.  A 16-step BFFT
simulation, including all pointwise work and retaining the exact numerator,
measured:

| BFFT length | eager shared | deferred stream | deferred/eager | saving |
|---:|---:|---:|---:|---:|
| 262,144 | 0.04361 s | 0.02179 s | 0.500x | 50.0% |
| 1,048,576 | 0.22074 s | 0.12159 s | 0.551x | 44.9% |

The deferred correction and numerator agreed with eager materialization to
about `1.3e-15` and `8.8e-16` relatively in these binary64 tests.  Production
use still needs a roundoff-safe signed digit base and linear-convolution
padding.  Retaining the mean spectrum also means retaining a redundant dyadic
digit representation: each division by two consumes one guard bit, but only
`O(log bits)` such bits are needed and coefficient magnitudes contract rather
than grow.

This is the current minimal degree-2 transform graph.  A reciprocal-square-root
Newton step can formally be composed spectrally as

```text
r_next = r*(3-x*r^2)/2
```

while retaining `x_hat`, and the final `sqrt(x)=x*r_next` then needs no new
forward transform.  Doing all of that without intermediate carry
normalization, however, changes a sequence of degree-2 products into a
degree-4 convolution.  It needs wider zero padding and a smaller safe digit
base, so it is not yet an operation reduction at the limb level.  The next
production experiment should retain transforms across *normalized* Newton
products (where safe) rather than fuse the entire quartic polynomial blindly.

There is a useful lower-bound interpretation.  At one non-real frequency bin,
the four scalar quadratic forms

```text
Re(A*B), Im(A*B), Re((A-B)^2), Im((A-B)^2)
```

are linearly independent.  A real bilinear straight-line program therefore
needs at least four real multiplications; the `U,V` construction attains that
bound.  Across iterations, carry normalization after the square root is
nonlinear.  A normalized `B_next` must be transformed once, and the product
must be inverse-transformed once before root extraction.  Hence `1F+1I` per
recurrent step is the safe transform floor for a degree-2 convolution backend.
Beating it requires absorbing carry/truncation into a different number
representation, not another Fourier identity.

## Refined lemniscatic formulations

The optimized Gauss--Legendre loop uses

```text
a_0=1, b_0=sqrt(1/2), t_0=1/4,
a'=(a+b)/2, b'=sqrt(ab),
t'=t-2^n(a-a')^2, pi=a^2/t at convergence.
```

Three implementation details are material:

1. `2^n` is applied by an exponent shift, not a full multiplication.
2. Successive states are swapped rather than copied.
3. Once the current correction is below the target plus guard bits, the
   arithmetic mean already suffices for `a^2/t`; the final multiplication and
   square root are skipped.

### Root-kernel audit

The geometric-mean root was also isolated and tested four ways.  Besides the
direct `sqrt(ab)` path, the C++ harness contains:

```text
reciprocal:       r = 1/sqrt(ab),  b' = ab*r
warm reciprocal: reuse r and refine x <- x(3-ab*x^2)/2
reduced defect:   b' = m - d^2/(m + sqrt(m^2-d^2))
                  m=(a+b)/2, d=(a-b)/2
```

The last identity is evaluated cancellation-free.  Since its defect has
exponent approximately `2*exp(d)-1`, all nonlinear operations use only the
precision needed to make the final defect accurate in absolute terms.  It
also reuses the `d^2` already required by the period correction and does not
compute the otherwise-unused product `ab`.

Interleaved M4 Mini medians show that none beats MPFR's direct square-root
kernel:

| requested bits | direct sqrt | reciprocal | warm reciprocal | reduced defect |
|---:|---:|---:|---:|---:|
| 100,000 | 1.000x | 1.679x | 2.285x | 1.397x |
| 1,000,000 | 1.000x | 1.482x | 1.740x | 1.523x |

All four results pass the requested-bit accuracy check.  Warm reuse loses
because each reciprocal-root Newton refinement requires three large
multiplications; the changing AGM argument prevents the previous reciprocal
root from being a free full-precision seed.  The reduced-defect route really
does shrink the working precision late in the chain, but its extra square and
division dominate before that saving becomes substantial.  Direct
`sqrt(ab)` is therefore not merely the shortest source formulation: it is the
fastest measured root kernel after state reuse and defect scaling are taken
into account.

That statement applies to using one kernel for every iteration.  A hybrid is
better: retain direct `sqrt(ab)` while the defect calculation would still be
near full precision, then use the cancellation-free defect only after its
required precision falls below 65% of the target.  The `pi_agm_hybrid`
implementation enables this above 4096 bits and keeps the direct path for
smaller inputs.

| requested bits | hybrid/direct paired ratio | improvement |
|---:|---:|---:|
| 1,000 | 1.003x | none; direct path selected |
| 10,000 | 0.926x | 7.4% |
| 100,000 | 0.959x | 4.1% |
| 1,000,000 | 0.983x | 1.7% |
| 10,000,000 | 0.961x | 3.9% |

The varying gain is expected: quadratic convergence makes the final defect
precision jump by roughly a factor of two, so changing the requested size
changes whether one or two terminal roots cross the cutoff.  The 65% value is
an empirical M4/GMP 6.3/MPFR 4.2.2 crossover, not a machine-independent
mathematical constant.

### Limb-level fusion audit

The benchmark also exposes two diagnostic suites:

```sh
make run BITS=1000000 REPEATS=7 SUITE=profile
make run BITS=1000000 REPEATS=9 SUITE=limb
```

`profile` times the direct AGM loop by operation.  At one million bits the
mean decomposition was:

| component | seconds | share of total |
|---|---:|---:|
| product multiplications | 0.02247 | 21.7% |
| recurrent square roots | 0.05461 | 52.6% |
| initial root | 0.00325 | 3.1% |
| period corrections | 0.01794 | 17.3% |
| terminal square/division | 0.00521 | 5.0% |
| means and loop overhead | 0.00025 | 0.2% |

Recurrent multiply-plus-root work occupies 74.3% of the run.  Deleting it
completely gives only a 3.89x ceiling, so a sixfold improvement from a root
kernel alone is impossible.  More sharply, Chudnovsky took about 0.0373
seconds in the paired run.  Non-kernel AGM work already costs about 0.0267
seconds, leaving 0.0106 seconds for all 17 fused geometric means.  The
existing multiplications alone cost 0.0225 seconds.  Beating Chudnovsky
therefore requires each fused product-root to cost less than half of a
multiplication, not merely a free square root after an ordinary product.

`limb` tests the direct GMP construction at limb-aligned precision:

```text
mpn_mul_n(A, B) -> exact 2n-limb product
mpn_sqrtrem(product) -> exact n-limb integer root
```

| requested bits | MPFR product+root | exact mpn product+root | ratio |
|---:|---:|---:|---:|
| 1,000,000 | 0.004591 s | 0.004572 s | 0.996x |
| 10,000,000 | 0.064582 s | 0.063899 s | 0.989x |

The end-to-end experimental `agm_exact_mpn` writes that integer root directly
into a limb-aligned MPFR mantissa.  Against an equally aligned direct AGM
control it was only about 0.3% faster at one million bits, with overlapping
ranges.  It uses MPFR representation internals and directed truncation under
guard bits, so it is a research control rather than the portable default.

GMP's integer-root machinery has already absorbed almost all of the generic
fusion opportunity, while MPFR's reciprocal square root already uses
limb-level Newton refinement.  A materially new generic kernel would need a
high-half algorithm that avoids forming most of the product.  AGM does expose
the special near-square defect, but only at the end of the chain: using it at
every step is 1.52x slower at one million bits, whereas the hybrid terminal
switch yields the small improvements above.

At one million bits this takes 18 iterations.  Counting the initial root, it
uses 18 square roots, 17 general multiplications, 19 squarings, and one final
division.  The explicit quartic takes 10 iterations but two square roots per
iteration plus the initial root: 21 roots total, followed by a more expensive
period-state update.  Replacing MPFR `rootn(4)` by two square roots helps, but
does not overturn that operation count.  Cancellation-free reconstruction

```text
y = c/((1+r)^2(1+r^2)),  r^4=1-c,
```

preserves extra guard bits but costs more than the direct quotient.

The tree-evaluable alternative inside the same CM field is the conductor-two
point `tau=2i`, with `j=66^3`:

```text
1/pi = 4/(11 sqrt(33)) sum_n
       (6n)!/((3n)!(n!)^3) (63n+5)/66^(3n).
```

Its binary splitter is exact, but the asymptotic term ratio is
`1728/66^3`, only about 7.378 useful bits per term.  It therefore needs
135,535 leaves at one million bits and 1,355,329 at ten million bits.

## Refined M4 Mini measurements

The entries are repeated-run medians in seconds.  Large runs had narrow
min--max ranges; at one million bits the measured ranges were AGM
`0.120--0.137`, quartic `0.224--0.246`, and lemniscatic binary split
`0.244--0.257` seconds.

| requested bits | AGM | quartic, two sqrt | lemniscatic binary split | Chudnovsky |
|---:|---:|---:|---:|---:|
| 1,000 | 0.0000105 | 0.0000201 | 0.0000883 | 0.0000166 |
| 10,000 | 0.000294 | 0.000571 | 0.00126 | 0.000198 |
| 100,000 | 0.00550 | 0.00910 | 0.0125 | 0.00209 |
| 1,000,000 | 0.1259 | 0.2296 | 0.2495 | 0.0451 |
| 10,000,000 | 2.070 | 3.812 | 4.950 | 0.796 |

The later fusion-only suite keeps direct AGM, hybrid AGM, and Chudnovsky in
the same short round-robin, avoiding interference from the slower exploratory
methods:

| requested bits | direct AGM | 65% hybrid AGM | Chudnovsky | hybrid/Chudnovsky |
|---:|---:|---:|---:|---:|
| 1,000,000 | 0.10411 | 0.10223 | 0.03740 | 2.73x |
| 10,000,000 | 1.76471 | 1.69584 | 0.67438 | 2.51x |

The computationally best formulation of the lemniscatic family is therefore
the quadratic AGM with a terminal near-square defect switch, plus concurrent
period/geometric lanes when a second core is available—not the quartic
iteration and not the CM product tree.  Quartic convergence halves the outer
iteration count but fails to reduce the total square-root count.  The
conductor-two product tree avoids sequential isogenies but pays for too many
low-yield terms and loses relative ground by ten million bits.

This is still not competitive with Chudnovsky.  On one core the scalar hybrid
is about 2.73 times slower at one million bits and 2.51 times slower at ten
million; with two-way parallelism the corresponding gaps are 3.32 and 3.11.
Generic limb fusion supplies at most about one percent, the terminal defect a
few percent, and concurrent AGM lanes about twenty percent of wall time.  A
large further improvement must also compress the period-correction state,
because even perfect deletion of the recurrent product-root chain has only a
3.89x measured ceiling.  Merely increasing Landen order or switching to the
available rational CM series cannot supply it.
