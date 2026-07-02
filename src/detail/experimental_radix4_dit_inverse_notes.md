# DIT-Shaped Standard Inverse Candidate

## RESOLVED: The Interpolation Basis (`inverse_complex_interp_simd`)

The search below ended in a fourth factorization that satisfies all three
desires at once - real-only, DIT-shaped transport (spectral head shuffle ->
forward-flow merges -> ordered real output), and as fast or faster than the
reverse-walk inverse at every size. It is implemented in
`experimental_radix4_rfft_kernel.hpp` as `inverse_complex_interp_simd` /
`inverse_complex_interp_scalar`.

### Why the three prior candidates could not satisfy

- **Reverse-walk inverse** (`inverse_complex_simd`): algebraic reverse of the
  DIT forward, so its permutation sits at the *time* tail (order -> chaos at
  output), the opposite transport shape.
- **Compact split-tree inverse** (`inverse_complex_dit_compact_*`, below):
  decimates *time* (a DIF of the inverse) - ordered head, ping-pong split
  flow, ~2x slower.
- **Bruun DIF inverse** (bruun_kernel): right shape, pre-mirrored-lane
  implementation vintage.

The missing corner of the square was **decimation in frequency of the
inverse**: CRT interpolation up the real factor tree of z^N - 1.

### The basis

Quad node Q(K, S) = z^S - 2 cos(pi K/n) z^{S/2} + 1 with children
Q(K/2, S/2), Q(n/2 - K/2, S/2); R-chain R(S) = z^S - 1 with children
R(S/2), Q(n/4, S/2). Leaves are the standard bins in the kernel's residue
convention X_k = a + b e^{-i theta_k}. The CRT merge of siblings at angles
(psi, pi - psi) into their parent at 2 psi is, per coefficient position i
with the single constant c = 2 cos(psi):

```text
pair_expand(r1[i], r1[i+S/4], r2[i], r2[i+S/4], 0.5/c, 1 - c^2)
  -> parent[i], parent[i+S/2], parent[i+S/4], parent[i+3S/4]
```

i.e. exactly the reverse walk's primitive, but per-BLOCK with a broadcast
constant instead of per-position twiddles. This is the exact dual of the
forward mirrored-lane discovery:

| | forward DIT | interpolation inverse |
|---|---|---|
| twiddles | per position, uniform across blocks | per block, uniform across positions |
| what that forces | mirror-lane packing | nothing - vertical SIMD is free |
| single lane-crossing | spectral terminal (fused projection) | spectral head (fused deprojection) |
| permutation | time-side gather (loads absorb it) | spectral gather (loads absorb it) |
| specials (DC/Nyquist/sqrt2) | mirror-lane DC chain | R-chain + prologue; interior constants never singular |

Butterfly read set == write set per position (clobber-free in place, no
palindrome pairing needed), the R-chain merge is a half-scaled add/sub, CRT
is exact so the 0.5s compose to 1/N with no normalization pass, and the root
merge streams ordered time samples straight to the output.

### Transport walk

- Work layout: P(S) subtree at work[S : 2S), lo|hi recursive; R-chain
  accumulates in the prefix. work[n] total, in place.
- Head: fused deprojection + two merge levels per 8-block (bins
  {j, n/4-j, n/4+j, n/2-j}, one shared constant record); odd-parity areas
  fuse their radix-2 16-level in registers (`itp_head16_v`).
- Global head phase: tiles of ALL areas sorted by comb phase (min leaf bin),
  so X is swept exactly once; per-area upper sweeps run afterwards
  (`itile_ord_`/`itile_meta_`). Ordering areas separately re-fetches every
  X line once per area that owns a bin in it (~4x) - measured disaster.
- Upper sweeps: fused radix-4 `pair_expand` merges, cache-tiered
  (`itp_upper_walk`, BRUUN_ITP_TIER2 = 32768 doubles measured best; L1-sized
  tiers measured WORSE - the win is grouping levels within L2).
- Terminal: R(n) from R(n/4), P(n/4), P(n/2) fused two R-levels; above
  n = 16384 it runs as two half-passes of <= 5 power-of-2-strided streams
  (~20% faster than the fused 12-stream form at L2 sizes).

### Measured (NEON, min-of-5, same-run ratios vs reverse-walk inverse)

```text
n:      16    32    64    128   256   512   1024  2048  4096  8192
ratio:  0.72  0.93  1.00  0.99  0.97  0.90  0.91  0.85  0.90  0.75

n:      16384 32768 65536 131072 262144 524288 1048576
ratio:  0.84  0.82  0.99  0.89   0.94   0.88   0.96
```

Roundtrip error identical order to the reverse walk at every size; scalar
path agrees. Odd-log2 sizes are the largest wins (the fused head16 absorbs
their extra radix-2 level for free).

---

## Historical search notes (superseded by the above)

The current `radix4_rfft_kernel::inverse_complex_simd` is the exact algebraic
reverse of the mirrored-lane forward:

```text
standard X -> terminal split -> inverse radix-4 splits -> seed output transport
```

That is correct, but it is not the desired transport shape. The desired inverse
should instead look like a normal DIT inverse FFT:

```text
standard X -> bit/digit-reversed spectral ingest -> forward-flow SIMD cascade -> ordered output
```

The clean candidate is the standard real-IFFT split through a length `M=N/2`
complex inverse FFT. Define complex samples

```text
z[n] = x[2n] + i*x[2n+1],  n in [0, M)
```

and let `Z = FFT_M(z)`. Given standard real-spectrum bins `X[0..M]`, reconstruct
`Z` by:

```text
Z[0] = 0.5*(X[0].re + X[M].re) + i*0.5*(X[0].re - X[M].re)

S = X[k]
T = conj(X[M-k])
Z[k] = 0.5 * ((S + T) + i*exp(+i*2*pi*k/N) * (S - T)),  k = 1..M-1
```

Then run a length-`M` complex inverse DIT:

```text
Z ordered -> digit-reversed gather into complex work -> inverse-sign radix cascade -> ordered z
```

Finally:

```text
out[2n]   = real(z[n])
out[2n+1] = imag(z[n])
```

This has the head-shuffle shape we want. The final output is ordered and needs
no tail bit reversal.

## Prototype Results

Out-of-tree probes validated the split equation against the current inverse to
roundoff:

```text
N=16    err ~1e-16
N=512   err ~5e-15
N=8192  err ~8e-14
```

A scalar radix-2 complex DIT with precomputed tables and caller-owned work is
correct but not competitive yet:

```text
N=8192    ~26.4 us
N=32768   ~166.7 us
N=65536   ~353.2 us
N=131072  ~743.9 us
```

A scalar radix-4 complex DIT with radix-4 digit reversal is correct for pure
radix-4 complex sizes and gives:

```text
N=8192    ~22.3 us
N=32768   ~187.5 us
N=131072  ~842.7 us
```

The mixed radix case (`M=2*4^L`, e.g. `N=65536`) still needs the correct
radix-2/radix-4 digit-reversal and stage ordering. The first prototype was
correct for `M=4^L` but not for `M=2*4^L`.

These scalar numbers are not the final performance target. The value of this
candidate is transport shape: ordered spectrum enters through a planned
digit-reversed gather, the cascade runs forward, and ordered time samples fall
out naturally.

## Implementation Work

- Build a planned complex inverse DIT helper for `M=N/2`.
- Precompute:
  - real-IFFT split coefficients `i*exp(+i*2*pi*k/N)`,
  - radix-4 inverse twiddles,
  - digit-reversal table for the actual mixed radix stage sequence.
- SIMD layout decision:
  - NEON/SSE 128 probably want two real lanes as one complex number for the
    simplest complex multiply, or a two-complex SoA tile if the shuffle cost is
    controlled.
  - AVX2 can likely carry two complex numbers per vector in a cleaner SoA form.
- Fuse spectral split with digit-reversed ingest so `Z[k]` never exists in
  ordered memory.
- Final output is just ordered complex-to-real deinterleave:
  `z[n].re -> out[2n]`, `z[n].im -> out[2n+1]`.

## Relationship To Current Inverse

Keep the current reverse-walk inverse as an oracle/fallback while developing
this path. Its row-tile transport is a repair for the reverse walk, not the
long-term target for the DIT-shaped standard inverse.

## Bruun-Style Real Residue Equivalent

The complex split above is not the form we want to implement. It can be
descended into Bruun quadratic residues directly.

For a parent length `N`, let `M=N/2`, `theta = 2*pi*r/N`, and
`x = exp(-i*theta)`. A parent residue bin is

```text
X[r]     = a + b*x
X[M-r]   = A + B*exp(-i*(pi-theta)) = A - B*x^-1
conj(X[M-r]) = A - B*x
```

The even/odd time-subsequence spectra are:

```text
E = 0.5 * (X[r] + conj(X[M-r]))
O = 0.5 * exp(+i*theta) * (X[r] - conj(X[M-r]))
```

The child length is `M`, so its residue basis is

```text
y = exp(-i*2*theta) = x^2
```

Use the identity

```text
x = (1 + y) / (2*cos(theta))
x^-1 = (4*cos(theta)^2 - 1 - y) / (2*cos(theta))
```

Then the child residues are all-real:

```text
inv4c = 1 / (4*cos(theta))

E_b = (b - B) * inv4c
E_a = 0.5*(a + A) + E_b

O_b = -(a - A) * inv4c
O_a = 0.5*(b + B) + (a - A) * (4*cos(theta)^2 - 1) * inv4c
```

Special bins:

```text
E[0].a = 0.5*(X[0].a + X[M].a)
O[0].a = 0.5*(X[0].a - X[M].a)

E[M/2].a = X[M/2].a
O[M/2].a = X[M/2].b
```

Base case for length 2:

```text
out[0] = 0.5*(X[0].a + X[1].a)
out[1] = 0.5*(X[0].a - X[1].a)
```

This gives the desired all-real DIT inverse:

```text
ordered standard bins
-> deproject/fuse into root Bruun residues
-> recursively split residues into even/odd child spectra
-> ordered real leaves
```

No arbitrary complex FFT is required. The recursive scalar probe validated this
against the current inverse:

```text
N=4     err 0
N=64    err ~1e-15
N=512   err ~2e-15
N=8192  err ~8e-15
```

The naive recursive probe is slow because it allocates child arrays and
recomputes cosines at every node:

```text
N=8192   ~733 us
N=32768  ~2.94 ms
```

Those timings are not meaningful for the final kernel. The implementation work
is to flatten/fuse this split tree into the existing radix-4 mirrored-lane
transport:

- Precompute `inv4c` and `(4*c*c - 1)*inv4c` per split stage.
- Fuse two radix-2 residue splits into one radix-4 split so it matches the
  current radix-4 forward granularity.
- Fuse the standard-bin deprojection with the first split/gather so the root
  residue array is not separately materialized in ordered form.
- Use digit-reversed spectral ingest to place child residue leaves in the order
  consumed by the forward-flow SIMD cascade.
- Keep the current reverse-walk inverse as a correctness oracle while this DIT
  residue inverse is built.

## Flattening The Recursion

The recursive proof should not become the kernel. It can be flattened with a
compact residue layout and a Stockham-style stage schedule.

Use the same compact Bruun residue layout as the old kernel:

```text
length L spectrum, H=L/2

v[0]             = bin 0 real
v[1]             = bin H real
v[2*r],v[2*r+1] = residue (a,b) for bin r, 1 <= r < H
```

This is exactly `L` doubles. Splitting one length-`L` parent produces two
length-`L/2` children, also exactly `L` doubles total:

```text
parent[L] -> even_child[L/2] + odd_child[L/2]
```

Therefore the full tree can be run with only:

```text
work[N]  <->  output[N]
```

No recursive allocation and no extra transport buffer are needed.

The flattened schedule that preserves ordered final time indices is:

```text
ingest standard bins into compact root residues in work

nodes = 1
L = N
cur = work
next = output

while L > 2:
    M = L / 2
    for node in 0 .. nodes-1:
        parent = cur  + node * L
        even   = next + node * M
        odd    = next + (node + nodes) * M
        split parent -> even, odd
    swap(cur, next)
    nodes *= 2
    L = M

for node in 0 .. nodes-1:
    parent length-2 spectrum -> output[node], output[node + nodes]
```

The child placement is intentional: after `d` stages, `node` is already the low
`d` bits of the final time index. The final length-2 base case supplies the
remaining high bit.

One scalar probe with precomputed constants and compact ping-pong storage
validated the flattened form:

```text
N=4     err 0
N=512   err ~1.6e-15
N=8192  err ~8.3e-15
```

Rough scalar timings:

```text
N=8192   ~26.5 us
N=32768  ~185 us
N=65536  ~413 us
```

An experimental compiled path now exists as:

```text
radix4_rfft_kernel::inverse_complex_dit_compact_simd(input, output, work)
```

It implements the compact flattened schedule with a V2 interior split loop. It
is covered by `tests/correctness.cpp` and reported by
`examples/radix4_rfft_benchmark.cpp` as `radix4_inverse_dit_compact_ns`.

Current NEON benchmark sample:

```text
N=8192    compact DIT  26.33 us   current reverse-walk  14.05 us
N=32768   compact DIT 141.33 us   current reverse-walk  63.77 us
N=65536   compact DIT 303.66 us   current reverse-walk 125.70 us
N=131072  compact DIT 616.07 us   current reverse-walk 279.93 us
```

This is a transport-correct development target, not a replacement yet. The
remaining gap is expected from running one radix-2 split stage at a time and
from the final parity copy on some sizes.

These timings are still not the target. The flattened scalar form is mostly a
storage/scheduling proof. Next performance steps:

- Fuse two radix-2 split stages into one radix-4 stage so the stage count and
  constants match the mirrored-lane forward kernel.
- SIMD the split node over multiple residue positions; the hot loop is
  add/sub/mul only once constants are precomputed.
- Fuse standard-bin deprojection directly into the first one or two split
  stages, avoiding a standalone root-residue ingest pass.
- Remove the final copy hazard by choosing stage parity or writing the length-2
  base case into the inactive buffer and treating that as the final output when
  the API can allow it. Otherwise the last copy is only `N` doubles but should
  be measured.
