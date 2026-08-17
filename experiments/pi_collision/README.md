# Geometric tangent collisions versus Chudnovsky

This experiment starts from the first positive solution of

```text
D_M(t) = tan(M atan(t)) = 1,       M = m^N,
```

so the selected algebraic root is exactly `t = tan(pi/(4M))`.  Repeated
multiple-angle transport gives

```text
A_j = 4 m^j tan(pi/(4 m^j)),       j = 0,...,N.
```

Writing `r=m^-2`, these are samples at the geometric nodes `r^j` of the
analytic function

```text
g(z) = 4 tan((pi/4)sqrt(z))/sqrt(z),       g(0) = pi.
```

The Richardson triangle is therefore polynomial interpolation at zero.  Its
closed weights are

```text
lambda_j = (-1)^(N-j) r^((N-j)(N-j+1)/2)
           / ((r;r)_j (r;r)_(N-j)),

(r;r)_k = product_(h=1)^k (1-r^h).
```

## The fused recurrence

Index from the deepest sample with `s=N-j`.  Starting with

```text
x_0     = t
scale_0 = 4 m^N
w_0     = 1/(r;r)_N,
```

stream one weighted term at a time and update

```text
sum     += w_s scale_s x_s
x_(s+1) = D_m(x_s)
scale_(s+1) = scale_s/m
w_(s+1) = -w_s r^(s+1) (1-r^(N-s))/(1-r^(s+1)).
```

This is `O(N)` multiprecision operations and `O(1)` live storage.  It is the
right implementation, but it does **not** improve the asymptotic bit
complexity of this family.

The C++ path goes one step further and reuses the final root-Newton tower for
the sum.  If `F` is the extrapolate and `y=D_M(x)`, its pending Newton update is

```text
F(x_new) = F(x) - 4(y-1) H/(1+y^2) + O((x_new-x)^2),
H = sum_s w_s(1+x_s^2).
```

The same geometric interpolant makes `H=1+O(error_N)`, since its samples are
`sec^2((pi/4)sqrt(z_j))` and its value at zero is one.  Thus the implementation
uses `H=1` and gets the final Newton correction without another transport
tower.  Precision doubling pushes both discarded errors beyond the requested
bits.

The first unannihilated interpolation term contains
`product_j r^j = m^(-N(N+1))`, hence the useful accuracy is
`Theta(N^2 log_2(m))` bits and `N=Theta(sqrt(n/log m))`.  On the other hand,
forming `D_(m^N)` by powering or composition has addition-chain depth
`Theta(N log m)`.  Even with precision-doubling Newton and decreasing
precision for negligible weights, the resulting bound is approximately

```text
Theta(Mul(n) sqrt(n log m)),
```

minimized at a small fixed radix.  A fixed-precision Newton loop adds another
`log n` factor.  Fusing the recurrences saves the quadratic extrapolation and
large constants, but it cannot reach the AGM bound `O(Mul(n) log n)`, much
less beat it asymptotically.  Achieving that would require a new way to remove
the sequential multiple-angle/root depth, not another rearrangement of the
weights.

## Run

The Python file is the readable oracle.  From the repository root, use the M4
Mini environment:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 \
  experiments/pi_collision/test_pi_collision.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 \
  experiments/pi_collision/pi_collision.py --digits 1000 --m 2
```

The C++ benchmark uses GMP/MPFR and compares the fused method with binary-split
Chudnovsky and Gauss-Legendre AGM; see `pi_collision.cpp`.

## M4 Mini measurements

These are single-process timings from the optimized C++ implementation.  They
are scale indicators, not record-quality benchmarking; the gap is too large
for harness noise to affect the conclusion.

| requested bits | radix/depth | fused collision | Chudnovsky | AGM |
|---:|---:|---:|---:|---:|
| 1,000 | 2 / 35 | 0.0010 s | 0.000081 s | 0.000053 s |
| 10,000 | 2 / 103 | 0.0140 s | 0.000272 s | 0.000388 s |
| 100,000 | 2 / 320 | 0.610 s | 0.00191 s | 0.00590 s |
| 1,000,000 | 2 / 1,003 | 34.33 s | 0.0383 s | 0.1278 s |

At 100,000 bits, approximately 0.349 s is spent reaching the root and 0.261 s
in the fused final tower/weights.  Median radix sweeps at this size gave 0.68 s
for `m=2`, 0.73 s for `m=3`, 0.84 s for `m=4`, and 0.98 s for `m=8` before the
final Newton/tower fusion.  The small fixed radix predicted by the operation
count also wins in practice.

At one million bits the fused method is about 897 times slower than this
compact Chudnovsky baseline.  The gap grows with precision, matching the
`sqrt(n)` versus `log(n)` iteration-factor analysis; there is no plausible
high-precision crossover in favor of this construction as it stands.
