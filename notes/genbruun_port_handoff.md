# Generalized Bruun (arbitrary-N) — port handoff & TODO

**Status (2026-06-24):** COMPLETE. Arbitrary-N real FFT is landed in the library
(`bfft.rfft/irfft` any N>=2, FFT-grade; pow2 fast path intact), benchmarked vs
FFTW, tested (`make test` -> `genbruun correctness ok`), and documented (README
"Arbitrary-N support"). All tasks #1-#14 + #4/#5 done.

Artifacts added: `src/detail/genbruun_kernel.hpp` (bruun::GenBruun),
`tests/genbruun_correctness.cpp` (+ Makefile GENBRUUN_TEST),
`benchmarks/odd_prime_power_fftw_benchmark.cpp` (FFTW baseline; runs all N).

Open follow-ups = ACCELERATION (perf/polish, none blocking correctness). The work
breakdown lives in **`src/detail/genbruun_acceleration_plan.md`** — read it before
optimizing. Summary: hot path is the odd-radix projections in
`GenBruun::exec_fwd/exec_inv` (`MINUS_SPLIT`, `BRUUN_ODD`); reorder loops to
`i`-outer/`m`-inner, then radix-3/5/7 SIMD codelets gated on `BRUUN_LEVEL`
(**GPT -> AVX2**, **us -> NEON later**), schedule + read the asm, then
`norm2_fused` cascade + pairwise sums. bfft is ~3.5-5x slower than FFTW on prime
powers today, N=3000 ~8e-14 vs ~1e-15 elsewhere. Also: float32 arbitrary-N path;
fully guard bfft_workspace_create / bfft_plan_standard_policy for non-pow2
(currently benign). **When NEON codelets land -> ready to push.**

Library wiring (DONE, #14): `src/detail/genbruun_kernel.hpp` = `bruun::GenBruun`
(owns scratch). `src/bfft.cpp` `bfft_plan` is tagged `{bool pow2; RFFT impl;
GenBruun* gen}`; `bfft_plan_create` dispatches (pow2>=4 -> RFFT, else GenBruun);
`bfft_forward/inverse/size/bins` dispatch; 19 pow2-only entry points guarded.
`bfft/_core.py` `_check_rfft_n` relaxed to N>=2. Existing tests still pass.

Original (pre-#14) status note: the algorithm was designed, validated FFT-grade in
Python and C++, and flattened into a plan-time schedule with SIMD-butterfly reuse.

This file is the cold-start context. Read it before touching the port.

---

## 1. What this is

Extend BFFT's power-of-two normalized-basis Bruun real FFT to **any N**, the
"nice" way: one real factorization of `z^N - 1`, reusing the existing pow2 Bruun
core as the 2-adic subtree, with small condition-1 odd-radix codelets. **No
Bluestein, no Rader, no chirp, no mixed-radix Cooley-Tukey.** (Those were all
explored and explicitly rejected — see §6.)

Accuracy achieved: **uniformly FFT-grade (~4e-16), including primes** — 0.33x to
2.0x of NumPy across odd/prime/prime-power sizes (see
`documentation/reports/odd_prime_accuracy.md`).

## 2. The algorithm (what a node does)

Factor `z^N - 1` by repeatedly peeling a prime. Two node kinds:

- **minus-node** `z^D - 1` at output stride `sigma` (covers bins `sigma*k`):
  - `D` power of two → call the existing `bruun::RFFT` (pow2 kernel); place bins
    at stride `sigma`. (D=1,2 handled inline — RFFT requires n>=4.)
  - else peel smallest odd prime `p`, `M=D/p`, blocks `R_0..R_{p-1}` (len M):
    - **s-branch** = `sum_i R_i` → recurse `reduce_minus(sum, M, sigma*p)`.
    - **branch j** (`j=1..(p-1)/2`), angle `g=j/p` turns: CONDITION-1 cascade
      seed (bounded coeffs, NO 1/sin growth):
      `seed_lo = sum_i R_i*cos(2*pi*i*j/p)`, `seed_hi = sum_i R_i*sin(2*pi*i*j/p)`.
      If `M==1` (prime leaf): `X[bin] = seed_lo - i*seed_hi`. Else recurse
      `reduce_bruun_cascade(seed_lo, seed_hi, M, g)`.

- **Bruun node** in the **cascade frame** (value `lo - i*hi`), half-size `M`,
  angle `f` turns:
  - `M==1` leaf → `X[bin] = lo[0] - i*hi[0]`, `bin = round(N*f)`.
  - `M` power of two → **rooted normalized cascade** (see §3): two-plane seed
    `[lo|hi]` straight into `norm_q_fwd`, no basis change.
  - else peel odd `p`, `Mp=M/p`; children `t=0..p-1` at `phi=(f+t)/p`:
    `child_lo = sum_i A_i*cos(2*pi*i*phi) - B_i*sin(2*pi*i*phi)`,
    `child_hi = sum_i B_i*cos(2*pi*i*phi) + A_i*sin(2*pi*i*phi)`
    where `A_i,B_i` are the `p` block-slices of `lo,hi`. (This is the length-p
    complex DFT across blocks with unit-modulus twiddles = condition 1.)

**Key identity that makes it FFT-grade:** the stage-0 basis map
`B = [[I, cos*I],[0, sin*I]]` applied to the natural Chebyshev residue equals the
bounded projection `(sum R_i cos(i*phi), sum R_i sin(i*phi))`. So we compute the
cascade seed directly with `|cos|,|sin|<=1` and never form the `1/sin(phi)`-
growing natural residue. This is what fixed primes (3e-12 -> 4e-16).

## 3. Rooted normalized cascade (SIMD reuse — the crux of #13)

A pow2 Bruun tail `B(2M, theta_root)` is run with the kernel's `bruun::norm_q_fwd`
butterfly, reusing the kernel's half-angle C-table **recurrence**, with TWO
adaptations:

1. **Seed** `C[1] = cos(theta_root/2)` (kernel uses `cos(pi/4)`). Recurrence
   unchanged: `C[2m]=sqrt(0.5(1+C[m]))`, `C[2m+1]=s_tw(m)/(2*C[2m])`.
2. **Root sine twiddle** `S1 = sin(theta_root/2)` set EXPLICITLY. The kernel's
   `s_twiddle(1)=C[1]` trick is only valid at the pi/2 root (where sin=cos).
   Deeper nodes use the complementary-sibling bit-flip `C[m^1]` unchanged.
3. **Stage loop:** `for jj=0..log2(D)-2: s=D>>jj, q=s>>2, nb=1<<jj;`
   `for m=0..nb-1: norm_q_fwd(v + m*s, q, C[nb+m], s_tw(nb+m))`. No binomial
   (pure quadratic tree, unlike the full kernel which has a block-0 minus lineage).
4. **slot -> bin:** slot `m`'s leaf angle = `m`'s binary path applied to
   `theta_root`, **MSB-first**, bit 0: `f->f/2`, bit 1: `f->1/2 - f/2`;
   `bin = round(N*f)`. (Validated by value-match against the scalar reference.)

## 4. File map

| file | what |
| --- | --- |
| `scratch_genbruun_exact.py` | **reference** — Python, exact-phase, condition-1, FFT-grade all N. The spec. (GPT + this session) |
| `scratch_normbruun.py` | normalized pow2 Bruun in Python, mirrors the C kernel |
| `scratch_genbruun.cpp` | scalar C++ port; rooted cascade uses `norm_q_fwd` |
| `scratch_genbruun_plan.cpp` | **flattened plan** — `GenBruunPlan` (init/forward), pooled tables, bump arena, allocation-free. Start #14 from here. |
| `scripts/odd_prime_accuracy_probe.py` | mpmath-90-digit accuracy probe |
| `documentation/reports/odd_prime_accuracy.md` | accuracy audit + bounds |
| `src/detail/bruun_kernel.hpp` | existing pow2 kernel — `bruun::RFFT`, `norm_q_fwd`, `norm_q_inv`, `norm2_fused`, `norm2_inv_fused` |
| `src/bfft.cpp` | C ABI; `bfft_plan` wraps `bruun::RFFT impl`; `bfft_plan_create` currently REJECTS non-pow2 |
| `include/bfft/bfft.h` | C ABI header |
| `bfft/_core.py` | ctypes shim; `bfft.rfft/irfft` + `_check_rfft_n` (pow2-only) |
| `Desktop/oddbenchmark.cpp` (user's Desktop) | FFTW-baseline benchmark with a `BfftRfftShim` that currently returns "skip-qhso-todo" for non-pow2 |

Reference DFT gotcha: **arm64 `long double == double`**, so a naive
`cosl(large arg)` DFT reference lies (~7e-13). Use integer-reduced phase
`((k*n)%N)` + Kahan summation (done in `scratch_genbruun.cpp` / `_plan.cpp` mains).

## 5. Build & validate

```sh
# Python reference
python3 scratch_genbruun_exact.py                 # vs numpy
PYTHONPATH=. python3 scripts/odd_prime_accuracy_probe.py   # vs mpmath

# C++ scalar + flattened plan (need the pow2 lib built once for nothing here;
# these include the kernel header directly)
g++ -O3 -march=native -std=c++17 -Isrc scratch_genbruun.cpp      -lm -o /tmp/g  && /tmp/g
g++ -O3 -march=native -std=c++17 -Isrc scratch_genbruun_plan.cpp -lm -o /tmp/gp && /tmp/gp
# pow2 native lib (for bfft.rfft ctypes path):  make   ->  build/libbfft.so
BFFT_LIBRARY=$PWD/build/libbfft.so python3 -c "import bfft, numpy as np; ..."
```

All sizes should print `OK` at <1e-12 (most ~1e-15..1e-16). Known loose spot:
N=3000 at ~7e-14 (ordered-sum on the 5^3 chain; pairwise summation would tighten).

---

## 6. TASK #14 — GenBruunPlan + C ABI dispatch + inverse (START HERE)

### 6a. Plan object + buffer contract
- Promote `GenBruunPlan` (from `scratch_genbruun_plan.cpp`) into the library
  (new header e.g. `src/detail/genbruun_kernel.hpp`, or a section of bfft.cpp).
- **Buffer contract:** `init(N)` computes `arena_doubles` (the `hi` high-water
  mark) and a `work_doubles` for the pow2 subplans (currently a `thread_local`
  temp in `_plan.cpp` MINUS_POW2 — route it through a plan-owned/`forward`-passed
  buffer instead). Expose sizes like the pow2 path (`bfft_plan_work_size` etc.).
- Keep `norm_q_fwd` as the only heavy kernel. Optional: `norm2_fused` (2 levels/
  pass) for the cascade; pairwise-sum the odd-split projections (helps N=3000).

### 6b. C ABI dispatch
- `bfft_plan_create`: currently asserts pow2 (`(ni & (ni-1)) != 0` -> error at
  `src/bfft.cpp:~100`). Change to: pow2 -> existing `bruun::RFFT`; non-pow2 ->
  `GenBruunPlan`. Make `bfft_plan` a tagged union / hold either.
- `bfft_forward` / `bfft_inverse`: dispatch on the tag. `bfft_plan_bins` =
  `N/2+1` for both.
- `bfft/_core.py`: drop the pow2-only `_check_rfft_n` for forward/inverse; let
  the native plan handle any N>=... (decide min N; the recursion handles N>=1
  but keep parity with kernel's N>=4 where it matters).
- **Residue-domain filter API stays pow2-only** — a non-pow2 transform has no
  single residue layout. Document and reject non-pow2 in those calls.

### 6c. Inverse — DONE & validated (Python `scratch_genbruun_inverse.py` +
###      C++ `scratch_genbruun_plan.cpp::inverse`). Roundtrip & vs numpy.irfft
###      FFT-grade all N (<=1.7e-15). Reverse op-walk; each op fills its INPUT
###      region from its outputs. Exact adjoints (validated):
- MINUS_POW2: `bruun::RFFT::inverse` (gather `Xf[sigma*k]`). D=1 -> r[0]=Re X[0];
  D=2 -> r[0]=(X0+Xsig)/2, r[1]=(X0-Xsig)/2.
- BRUUN_POW2: gather leaves `v[2m]=Re Xf[perm[m]], v[2m+1]=-Im`; run stages in
  REVERSE with `bruun::norm_q_inv` (same C-table/S1) -> seed [lo|hi]. (norm_q_inv
  exactly implements the inverse butterfly: A0=(L+R)/2, R=(L-R)/2, A1=(L1-R1)/2,
  I=(L1+R1)/2, B0=cR+sI, B1=cI-sR.)
- BRUUN_ODD (1/p adjoint): `lo[i]=(1/p)sum_t(clo_t cos+chi_t sin)`,
  `hi[i]=(1/p)sum_t(chi_t cos-clo_t sin)`, cos/sin = same per-child twiddle table.
  (SIGN: forward is A cos - B sin, so inverse cross-terms are +sin. Easy to get
  wrong -- this was the one bug.)
- MINUS_SPLIT (2/p adjoint, 1/p DC): `r_i=(1/p)[sum + 2 sum_j(lo_j cos+hi_j sin)]`,
  cos/sin = same branch twiddle table cos(2pi ij/p).
- Result x = arena[0..N) (root residue lives at offset 0).

Original derivation note (kept for context): every op is orthogonal-up-to-scale,
so the inverse is the schedule run **in reverse** with each op's adjoint/inverse:
- **MINUS_POW2**: `bruun::RFFT::inverse` (place real samples at stride sigma's
  inverse gather). D=1,2 inline.
- **BRUUN_POW2 cascade**: the kernel already has `norm_q_inv` (and
  `norm2_inv_fused`) — run the rooted cascade backward (reverse stage order,
  `norm_q_inv` with the same C-table), then the seed->(lo,hi) is the cascade
  frame value; reverse the leaf placement (gather bins -> v).
- **MINUS_SPLIT / BRUUN_ODD projections**: these are real-DFT-like maps with
  orthonormal columns up to a known scale. The inverse is the scaled adjoint
  (transpose) of the same cos/sin table. Concretely the forward minus split maps
  `p` real blocks -> DC sum + (p-1)/2 (lo,hi) pairs; that's an orthogonal real
  transform of the `p` block-values per sample index, so inverse = (suitably
  scaled) transpose. Derive the scale from unitarity (forward has a sqrt(p)-ish
  norm; account explicitly). Validate against `numpy.fft.irfft` AND roundtrip.
- **Validation:** `irfft(rfft(x)) == x` to ~1e-14 for all N; and forward `rfft`
  vs `numpy.fft.rfft`; inverse `irfft` vs `numpy.fft.irfft` (Hermitian input).

Suggested order: get the **inverse working in Python first** (add `irfft_gen` to
`scratch_genbruun_exact.py`, reversing each step) — same de-risk pattern that
worked all along — then port to C++.

### 6d. Don't forget
- Conjugate symmetry: forward only fills bins via `place` (sets `X[k]` and
  `X[N-k]=conj`). For irfft input is `N/2+1` Hermitian bins.
- `bfft_plan_create` integer overflow guard already caps N at 2^31-1.
- The odd-split twiddle tables and rooted C-tables/perms are all plan-time; the
  inverse reuses the SAME tables (adjoint), so build once.

## 7. Remaining after #14
- **#4** — FFTW-baseline benchmark. NOTE: task #4's title still says "QHSO" —
  STALE; QHSO is dead (see below). Use `oddbenchmark.cpp`: replace the
  `BfftRfftShim` "skip-qhso-todo" path with the real non-pow2 call; time fwd/inv
  double (then float) vs FFTW on prime/prime-power/odd-composite groups. FFTW is
  the ONLY baseline (no numpy). It `#include`s `../bruun_fastplan_safe_api.cpp`
  which does NOT exist in this repo — reconcile (adapt benchmark to this repo's
  C ABI, or provide that single-file API).
- **#5** — tests + README: document the arbitrary-N contract; that this is NOT
  Bluestein/Rader/mixed-radix but a real `z^N-1` factorization; accuracy
  expectations; residue API stays pow2-only.

## 8. Dead ends (do NOT revisit — already disproven this session)
- **QHSO half-coset sidecar** — elegant, exact algebra, but its forward assembly
  reduces to the Dirichlet separator (O(T*eps) sidelobe cancellation) and its
  stable `apply_window` collapses to Bluestein. Net loss vs plain Bluestein.
- **Bluestein/Rader** — work and are FFT-grade (modular-chirp Bluestein in
  `scratch_modchirp.py` is validated), but rejected on philosophy: don't recreate
  FFTW/pocketfft. Kept only as a possible quarantined leaf for huge prime factors.
- **Phase-shifted-FFT block** (compute a 256-block of a 511-DFT with twiddles):
  provably impossible — the block kernel `exp(-2pi i j r/T)` is not `D1 F D2`
  (separability fails unless T=L). Confirmed with an SVD rank check.
- **Adjoint-state / natural-Chebyshev odd split** — catastrophic
  cancellation/conditioning; replaced by the condition-1 cascade frame (§2).
