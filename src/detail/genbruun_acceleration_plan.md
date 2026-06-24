# GenBruun acceleration plan

Target: make the arbitrary-N (`bruun::GenBruun`) path competitive with FFTW on
odd / prime / prime-power sizes. Today it is **FFT-grade accurate but ~3.5-5x
slower** than FFTW on prime powers (see `benchmarks/odd_prime_power_fftw_benchmark.cpp`).
The power-of-two path (`bruun::RFFT`) is already SIMD-tuned and is out of scope.

This document is the work breakdown. **Division of labor:** GPT takes the **AVX2**
codelets (it has the natural x86 hardware to profile on); we pick up **NEON**
here later. Keep a scalar reference path and gate SIMD behind the existing
`BRUUN_LEVEL` macro (see `bruun_kernel.hpp` lines ~28-51) so both backends and the
portable build stay correct.

Golden rule for every change: **`make test` (incl. `genbruun correctness`) stays
green, and re-run the FFTW benchmark to confirm a real win.** No speedup is real
until measured; no change ships if it loses a bit of accuracy on the probe
(`scripts/odd_prime_accuracy_probe.py`).

---

## 0. Profile first — find the hot ops

Where the time goes for non-pow2 N (`bruun::GenBruun::exec_fwd` /
`exec_inv` in `genbruun_kernel.hpp`):

- `MINUS_POW2` -> `bruun::RFFT` : already fast, leave it.
- `BRUUN_POW2` -> rooted `norm_q_fwd` cascade : already SIMD, minor wins only.
- **`MINUS_SPLIT`** and **`BRUUN_ODD`** : scalar `O(p*M)` projection loops. **This
  is the hot path.** Prime-power N (3^k, 5^k, 7^k) are *pure* chains of these —
  exactly the sizes where we lose to FFTW.

Action:
1. Add a cheap per-op-type cycle counter (or use `perf stat` / Instruments) and
   dump the time split for a few representative sizes (e.g. 729, 2401, 3125, and
   a prime-near-2^k like 4093). Confirm the projections dominate before tuning.
2. Disassemble the hot loops (`-S -fverbose-asm`, or `objdump -d` the `.o`) and
   read them (see §4). This is fast and usually shows obvious wins.

---

## 1. Restructure the projection loops for vectorization (portable, do FIRST)

Current `MINUS_SPLIT` inner kernel (per branch j):

```c++
for (int m = 0; m < M; ++m) { double a = 0, b = 0;
    for (int i = 0; i < p; ++i) { double r = in[i*M + m]; a += r*C[i]; b += r*S[i]; }
    lo[m] = a; hi[m] = b; }
```

The vectorizable axis is **m** (M can be large; the twiddles `C[i],S[i]` are loop
invariants). Swap the loops so `m` is the inner, contiguous, vector axis and the
twiddle is a broadcast scalar:

```c++
for (int m = 0; m < M; ++m) { lo[m] = 0; hi[m] = 0; }
for (int i = 0; i < p; ++i) {
    const double ci = C[i], si = S[i];          // broadcast
    const double* Ri = in + i*M;                 // contiguous length-M block
    for (int m = 0; m < M; ++m) { lo[m] += Ri[m]*ci; hi[m] += Ri[m]*si; }  // FMA over m
}
```

This is the form a vectorizer (and later hand-NEON/AVX) can turn into broadcast +
FMA over `m`. Do the same for `BRUUN_ODD` (complex: two input planes `A,B`, two
output planes per child, four real FMAs per i). Sub-tasks:

- 1a. Reorder `MINUS_SPLIT` (fwd + inverse adjoint) to `i`-outer / `m`-inner.
- 1b. Reorder `BRUUN_ODD` (fwd + inverse) the same way.
- 1c. Mark block pointers `__restrict` (kernel uses the `RESTRICT` macro) and
      hoist `twp_`/`ctp_` base pointers out of the loop (kill the pool indirection
      on the hot path).
- 1d. Use multiple accumulators (split the `m` loop into 2-4 independent FMA
      chains) to hide FMA latency — even scalar this helps; see §3.

Expect a solid win from this alone, before any intrinsics, because the current
`m`-outer form recomputes `lo/hi` scalars with a serial dependency.

---

## 2. Small-radix codelets (the big win) — AVX2 (GPT) + NEON (us)

Prime-power sizes are dominated by fixed small radices. Write specialized,
fully-unrolled **radix-3 / radix-5 / radix-7** projection codelets (like FFTW's
codelets) that:

- hold the `p` twiddles in registers (compile-time `p`, unrolled `i` loop),
- vector over `m` (broadcast twiddle * vector block, FMA accumulate),
- come in fwd and inverse (adjoint) variants for both `MINUS_SPLIT` (real DFT,
  `(p-1)/2` outputs + DC) and `BRUUN_ODD` (complex DFT, `p` outputs).

Follow the kernel's existing backend pattern exactly:
- Gate on `BRUUN_LEVEL` (3=AVX-512, 2=AVX2+FMA, 1=SSE2/NEON, 0=scalar) and reuse
  the `bruun_v2` / `V2_*` (and `_mm256_*`) vocabulary from `bruun_kernel.hpp`.
- AVX2: `_mm256_fmadd_pd` over 4 doubles of `m` per lane; NEON: `vfmaq_f64` over 2.
- Keep a scalar tail and a scalar reference codelet for `BRUUN_LEVEL==0`.

Sub-tasks:
- 2a. radix-3 fwd/inv codelets (covers 3^k and the `*3` factor).
- 2b. radix-5 fwd/inv.
- 2c. radix-7 fwd/inv.
- 2d. generic `p` fallback = the §1 vectorized loop (for large odd primes).
- 2e. dispatch in the plan: store the codelet kind per `MINUS_SPLIT`/`BRUUN_ODD`
      op at plan time so `exec_*` just calls a function pointer / switch.

These three radices cover the overwhelming majority of smooth odd sizes.

---

## 3. Scheduling / latency hiding

The projection is FMA-bound. To approach peak FMA throughput:

- **Multiple accumulators:** an FMA has ~4-cycle latency but 2/cycle throughput,
  so a single dependent accumulator chain runs at ~1/4 peak. Unroll the `m` loop
  by 2-4 vectors with independent accumulators, sum at the end.
- **Software pipeline** the radix codelets: issue all `p` broadcast loads up
  front, interleave the FMAs so the scheduler can overlap independent chains.
- **Block for L1:** for large `M`, tile the `m` loop so each `R_i` block stays hot
  across the `i` sweep; or transpose the access so one pass over input feeds all
  outputs (fuse the s-branch sum with the branch projections — one read of `in`).
- Avoid store/reload of `lo/hi` between `i` iterations — keep accumulators in
  registers across the whole `i` loop (the §1 rewrite already does this per m-tile).

---

## 4. Read the assembly — cheap, high-yield

For each hot codelet, dump and eyeball the asm (`g++ -O3 -march=native -S
-fverbose-asm`, or `objdump -d build/.../genbruun_kernel.o`). Look for:

- **Vectorization actually happened:** `fmla`/`vfmaq` (NEON) or `vfmadd*pd`
  (AVX2) in the inner loop, not scalar `fmul`/`fadd`. If scalar, the loop didn't
  vectorize — find why (aliasing? non-contiguous? pool indirection?).
- **No spills:** no stack `str/ldr` of accumulators inside the loop. If spilling,
  reduce live twiddles / unroll factor.
- **Broadcast hoisted:** the twiddle broadcast (`ld1r`/`vbroadcastsd`) is outside
  the `m` loop, not reloaded each iteration.
- **Tight loop, no scalar fallback:** the autovectorizer often emits a scalar
  remainder + a vector body; make sure the vector body is the common case (align
  M handling) and the remainder is just the tail.
- **`restrict` taking effect:** with `__restrict__` the compiler should not
  reload `in`/`lo`/`hi` across iterations.

A 10-minute asm read on the radix-3 codelet usually finds the difference between
"vectorized" and "vectorized well".

---

## 5. Cascade + plan-time polish (secondary)

- 5a. Use `norm2_fused` / `norm2_inv_fused` (2 tree levels per pass, already in
      `bruun_kernel.hpp`) for the `BRUUN_POW2` rooted cascade instead of
      one-level `norm_q_fwd`, when `q` is large enough (the kernel's own driver
      shows the size gate). Halves load/store traffic on the pow2 tails.
- 5b. Hoist the rooted-cascade `S1 = tsin(fhalf(o.f))` and the `s_tw` lambda out
      of `exec_*` into plan-time storage on the op (one transcendental per call
      removed).
- 5c. Pairwise / 2-accumulator summation in the projections (pairs with §3's
      multiple accumulators) — also tightens accuracy (fixes N=3000's ~8e-14 ->
      ~1e-15). Vector partial sums are already more accurate than serial.
- 5d. Consider an interleaved `(A,B)` layout for the `BRUUN_ODD` complex planes if
      profiling shows the split-plane loads are the bottleneck.

---

## 6. Accuracy guardrails

- Re-run `scripts/odd_prime_accuracy_probe.py` after each codelet. The condition-1
  structure must be preserved: twiddles are unit-modulus, so do not factor them
  in a way that reintroduces `1/sin` growth.
- Keep the high-precision twiddle generation model (plan-time tables generated at
  high precision, stored as correctly-rounded doubles — see
  `documentation/reports/odd_prime_accuracy.md`). The current C++ uses `std::cos`;
  if a sensitive angle shows up, switch the table build to a higher-precision
  generator (long double is *not* extended on arm64 — use a polynomial/Cody-Waite
  or a small bignum at plan time).

---

## 7. Sequence & ownership

1. §0 profile + §1 loop reorder (portable) — **do first, here**; should already
   close part of the gap and makes the SIMD step clean.
2. §2 codelets: **GPT -> AVX2**, **us -> NEON later**. Shared scalar reference +
   `BRUUN_LEVEL` gating so both land independently.
3. §3/§4 schedule + asm read per codelet (whoever owns that backend).
4. §5 cascade/plan polish.
5. Re-benchmark vs FFTW at each step; §6 accuracy probe each step.

When NEON codelets land and the benchmark shows parity-ish on smooth sizes
(and we accept the large-prime O(p^2) tail), **we are ready to push.**
