# Bruun/BODFT kernel sequentiality map

This note describes the hot kernels as dependency flows rather than as C++
syntax. The goal is to make optimization decisions explicit: what has to happen
first, what can overlap, what should be kept out of the live set, and where SIMD
width changes leave tail work.

## Symbols

- `A0, B0, A1, B1` are the four quarters of one normalized Bruun node.
- `C0, C1, D0, D1` are the same storage locations viewed after a forward split,
  so inverse kernels reconstruct `A/B` values from `C/D` pairs.
- `c, s` are a twiddle pair. A rotation is always
  `R = c * x - s * y`, `I = s * x + c * y`.
- `h = 0.5`. In inverse code, `h` is algebraically movable into twiddle constants
  when the half-scaled value is only used by a rotation and is not stored itself.

## Forward one-level node: `norm_q_fwd`

Per lane group, the essential flow is:

```text
load B0, B1
R = c * B0 - s * B1
I = s * B0 + c * B1
load A0, A1
C0 = A0 + R
C1 = A1 + I
D0 = A0 - R
D1 = I - A1
store C0, C1, D0, D1
```

Sequentiality observations:

1. `R/I` depend only on `B0/B1` and the twiddle. `A0/A1` are not needed until the
   store frontier.
2. Loading `A0/A1` after launching the rotation shortens live ranges and lets the
   FMA/multiply chain start earlier.
3. The add/sub store frontier has two independent pairs: `(A0, R)` and `(A1, I)`.
4. Wide x86 should run AVX2 first, then the 128-bit V2 tail, then scalar.
   Dropping from AVX2 directly to scalar wastes the natural two-double tail.

## Forward two-level fused node: `norm2_fused`

The fused kernel is two `norm_q_fwd` levels with the high-half children consumed
immediately by the second level.

```text
frontier 0: load B0n, B1n, B0h, B1h
frontier 1: Rn/In = rotate(c, s, B0n, B1n)
            Rh/Ih = rotate(c, s, B0h, B1h)
frontier 2: load A0n, A1n, A0h, A1h
frontier 3: u0 = A0n + Rn, w0 = A1n + In
            v0 = A0n - Rn, x0 = In - A1n
            uh = A0h + Rh, wh = A1h + Ih
            vh = A0h - Rh, xh = Ih - A1h
frontier 4: R0/I0 = rotate(c0, s0, uh, wh)
            R1/I1 = rotate(c1, s1, vh, xh)
frontier 5: eight stores from u/w/v/x plus R0/I0/R1/I1
```

Sequentiality observations:

1. The first rotation frontier is independent of all `A` loads.
2. The second rotation frontier uses only the high-half derived values; low-half
   derived values can stay close to the store frontier.
3. `c0/s0` and `c1/s1` are independent rotations. They can be adjacent in source
   but should not force unnecessary temporary arrays or long-lived duplicated
   constants.
4. On x86, FMA-form V2/V4F macros matter in the 128-bit tail because scheduled
   transforms intentionally run wide, then narrow, then scalar.

## Inverse one-level node: `norm_q_inv`

The inverse first reconstructs half-sums and half-differences, then applies the
conjugate rotation for the B outputs.

```text
load C0, C1, D0, D1
t0 = C0 + D0        ; stored A0 needs h * t0
r  = C0 - D0        ; rotated B0/B1 need h * r
i  = C1 + D1        ; rotated B0/B1 need h * i
t1 = C1 - D1        ; stored A1 needs h * t1
A0 = h * t0
A1 = h * t1
B0 = (h*c) * r + (h*s) * i
B1 = (h*c) * i - (h*s) * r
store A0, B0, A1, B1
```

Sequentiality observations:

1. `A0` and `A1` genuinely need explicit half multiplies because they are stored.
2. `r` and `i` do not need to become separately half-scaled vectors before the
   rotation. Folding `h` into `c/s` removes two vector multiplies per lane group.
3. The same algebra works for double and float, and for AVX2 and 128-bit tails.
4. Scalar code can remain written in the direct mathematical form because the
   compiler has no SIMD register-pressure problem there.

## Inverse two-level fused node: `norm2_inv_fused`

This is the densest dependency graph in the kernel. It reconstructs two child
nodes, rotates their high halves, then reconstructs the parent node.

```text
frontier 0: load A0n, B0n, A0h, B0h, A1n, B1n, A1h, B1h
frontier 1: u0 = h * (A0n + B0n), r0 = A0n - B0n
            w0 = h * (A0h - B0h), i0 = A0h + B0h
            v0 = h * (A1n + B1n), r1 = A1n - B1n
            x0 = h * (A1h - B1h), i1 = A1h + B1h
frontier 2: uh/wh = rotate(h*c0, h*s0, r0, i0)
            vh/xh = rotate(h*c1, h*s1, r1, i1)
frontier 3: a0n = h * (u0 + v0), a1n = h * (w0 - x0)
            a0h = h * (uh + vh), a1h = h * (wh - xh)
            rn  = u0 - v0,       in  = w0 + x0
            rh  = uh - vh,       ih  = wh + xh
frontier 4: B0n = (h*c) * rn + (h*s) * in
            B0h = (h*c) * rh + (h*s) * ih
            B1n = (h*c) * in - (h*s) * rn
            B1h = (h*c) * ih - (h*s) * rh
frontier 5: store A0n, B0n, A1n, B1n, A0h, B0h, A1h, B1h
```

Sequentiality observations:

1. Many apparent `h * (...)` values feed only a rotation. Those half factors can
   be absorbed into the corresponding twiddle constants.
2. Values stored as A outputs still need explicit half scaling.
3. The two child rotations are independent; the parent low/high rotations are
   also independent after frontier 3.
4. This kernel is register-pressure sensitive. Optimizations that reduce a
   multiply but extend too many live ranges can lose on SSE2, so changes should
   be benchmarked per backend.

## BODFT forward combine

BODFT's paired radix-4 combine uses the same scheduling idea but with complex
children and paired conjugate partners.

```text
load twiddle streams t, t2, t3
load child spectra c0, c1, c2, c3
b0 = c0
b1 = t  * c1
b2 = t2 * c2
b3 = t3 * c3
e0/e1 = b0 +/- b2
o0/o1 = b1 +/- b3
write k, k+M, partner(k), partner(k+M)
```

Sequentiality observations:

1. The three child rotations are independent once twiddles and child lanes are
   deinterleaved.
2. The partner stores need reversed lane order but no new arithmetic.
3. Eight-position AVX2 f32 is worthwhile only when deinterleave/interleave stays
   in registers. If packing spills, the 128-bit path can be competitive.

## Practical optimization checklist

1. Identify the first dependency frontier. Do not load values that are not needed
   until a later frontier if doing so increases live range pressure.
2. Prefer accumulator-first FMA abstractions where the source math is
   `accumulator +/- lhs * rhs`; the intrinsic argument order must preserve this
   shape.
3. Let wide SIMD fall through to narrower SIMD tails before scalar cleanup.
4. Absorb invariant scales into twiddle constants when the scaled temporary is
   only consumed by a rotation.
5. Wider registers do not automatically
   win for these kernels because the DAG is short, store-heavy, and sensitive to
   frequency and register pressure.

## Benchmark gate for proposed rewrites

The branch already contains the x86 128-bit FMA macro fix, wide-to-narrow double
SIMD tails, float forward B-load scheduling, float one-level inverse half folding,
and AVX2 f32 BODFT forward/inverse combines. Additional Bruun rewrites should be
kept only after same-command before/after checks because the kernel DAG is often
limited by register pressure and stores rather than pure multiply count.

Two tempting changes are explicitly gated:

1. **Double `norm_q_fwd` B-first load scheduling.** This is algebraically sound and
   matches the dependency frontier, but same-container AVX2 benchmark runs were
   mixed/noisy rather than a clear win after the existing FMA-tail work. Keep it as
   a per-backend experiment instead of a default change until it wins consistently.
2. **`norm2_inv_fused_f32` quarter-scale delay.** The delayed-scale form reduces
   intermediate half multiplies, but it also changes live ranges in the densest
   inverse fused kernel. In local AVX2 benchmark samples it improved some larger
   runs but regressed smaller/store-hot runs, so it should not replace the current
   code without more per-backend evidence.

This benchmark gate is part of the intended workflow: the flow diagrams identify
legal motion, but legality is not the same as throughput. Keep changes only when
they either improve the measured backend or are needed to expose a later proven
optimization.
