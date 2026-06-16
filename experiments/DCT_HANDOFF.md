# DCT/DST work — handoff

Onboarding note for whoever (human or another Claude) picks this thread up next.
Read this first, then `ROADMAP.md` for the full reasoning. Everything here is
under `experiments/` and is research-grade (not yet wired into the public
`bfft` C/C++ API — that is Phase 6).

## What this is

Optimized DCT-I..IV / DST-I..IV endpoints built on BFFT's Bruun/Chebyshev basis.
The thesis (see `capstone.md`): a real FFT bin is `X[k] = C[k] - i S[k]`; the
cosine (Chebyshev-T) projection alone is a DCT, the sine (Chebyshev-U) projection
alone is a DST — each ~half an FFT because it never builds the half it doesn't
need. So the DCT/DST are the Bruun real-residue tree with the imaginary half
pruned.

## Current state (all validated)

Two DCT engines exist, both correct; they differ in maturity:

1. `chebyshev_dct_radix4.hpp` — `ChebDCT`. Radix-4 Chebyshev *tower* kernel.
   DCT-III, DCT-IV, DST-IV, DST-II, DST-III (all fwd+inv). Production-shaped:
   precomputed aligned tables, no per-call heap alloc, all-FMA `restrict` node,
   fused single-pass repack + iterative schedule + branch-free split. **~4x the
   Bruun rFFT** in wall-clock. Power-of-FOUR N. This is the one that ships today.
   It carries an intrinsic `flat_to_tower` repack (~6q/node) because it consumes
   the nested power-of-T_q tower — that repack is the reason it isn't ~half-FFT.

2. `bruun_dct.hpp` — `BruunDCT`. **Repack-free**, natural-order DCT-II + DCT-III
   (fwd+inv). Power-of-TWO N. The Bruun real DIF: `DCT-II = 2*(DCT-III)^T`; the
   transpose turns the eval butterfly into the Bruun mirror fold. Algebra VERIFIED
   to machine precision. **BUT the current cut is naive (radix-2 recursive, a
   scratch copy per level, serial bidiagonal, no schedule) and is ~13-15x the FFT
   — slower than the tower.** It has fewer multiplies (no repack); the
   implementation just hasn't been optimized yet. This is the higher-ceiling path.

Supporting files:
- `chebyshev_dct_kernel.hpp` — direct-matrix reference for all 8 types (the
  ORACLE everything is checked against; FFTW REDFT*/RODFT* conventions) + Clenshaw
  + flat<->tower helpers.
- `dct1_dst1_via_fft.hpp` — DCT-I/DST-I (Lobatto grid) via the shipped real FFT of
  the (anti)symmetric extension. `n = 2^m +- 1`.
- `cheb_dct_assemble.hpp` — `AssembledRFFT`: real FFT reassembled from DCT-I/DST-I
  folds (validated vs the Bruun FFT and a direct DFT).
- `cheb_dct_radix_probe.cpp` — empirical radix-2/4/8 probe for the real node.
- `cheb_dct_bench.cpp` — throughput benchmarks.
- `test_chebyshev_dct.cpp` — all correctness tests (16 groups, 136 checks).

## THE NEXT STEP: optimize BruunDCT

Give `BruunDCT` the same treatment that made `ChebDCT` fast (4x), which should
take the repack-free kernel toward the "~half an FFT" target it's capable of
(it has fewer multiplies than the tower). In rough priority:

1. **Eliminate the per-level scratch copies.** The recursive de-interleave /
   interleave (`analysis`/`synthesis` in `bruun_dct.hpp`) copy the whole block
   twice per level. Replace with a SINGLE up-front permutation (bit-reversal-like)
   then in-place iterative butterflies, exactly like `src/detail/bruun_kernel.hpp`
   does for the FFT.
2. **Radix-4.** The radix probe (`make dct-radix-probe`) settled radix-4 for the
   real node (r8 loses on ILP/register pressure, r2 on memory traffic). The
   current BruunDCT is radix-2 = 2x the passes. Fuse two levels.
3. **Iterative schedule.** Precompute the `(offset, size, twiddle-base)` schedule
   in `init()` and run a flat loop — no recursion. (See how `ChebDCT` did this:
   `sched_off_`/`sched_q_`.)
4. **Restructure the serial bidiagonal.** `od[j]=od[j]-od[j+1]` (synthesis) and
   the transposed forward sweep (analysis) are serial dependency chains that kill
   ILP. Find a parallel/blocked form, or fold it into the butterfly.
5. **Vectorize the fold** with `restrict` (the mirror fold should go 2-wide).

Target: `make dct-bench` "BruunDCT-III" column < the "ChebDCT-III" column, ideally
approaching ~0.5-1.0x the "Bruun rFFT" column. Then route DCT-IV/DST-* through the
repack-free engine and retire the tower (or keep the tower only for power-of-4).

Do NOT regress to a generic Lee/Cooley-Tukey DCT — keep it Bruun-family (real,
Chebyshev, the T_2 composition + binomial fold). That distinction is why the
verified algebra in `bruun_dct.hpp` is the right foundation.

## Checks to run (how to know you haven't broken anything)

From the repo root:

- `make radix4-test` — runs `test_chebyshev_dct` (among others). **Expect
  "136 passed, 0 failed".** This is the correctness gate for ALL the DCT/DST work
  (Tests 1-15 cover the reference + ChebDCT + endpoints + assembled rFFT; Test 16
  covers BruunDCT vs oracle + round-trip). Any BruunDCT change must keep this green.
- `make dct-bench` — throughput table (Bruun rFFT | ChebDCT-III | BruunDCT-III |
  asm rFFT). This is where you watch the BruunDCT-III column drop as you optimize.
  Also prints the node restrict/vectorization microbench.
- `make dct-radix-probe` — the radix-2/4/8 real-node probe (already concluded
  radix-4; re-run if you change the node).
- `make test` — the existing FFT suite; **expect "all tests passed"**. The DCT work
  must not touch it.
- Warning-clean check (CI uses -Werror across C++17/20/23):
  `c++ -Iinclude -O3 -std=c++17 -Wall -Wextra -Wpedantic -Werror \
   experiments/test_chebyshev_dct.cpp build/libbfft.a -lm -o /tmp/t`
  (do the same for `cheb_dct_bench.cpp` and `cheb_dct_radix_probe.cpp`).
  All three are currently -Werror clean.

Build prerequisite: `make` (builds `build/libbfft.a`) before the experiment
targets if `build/` is empty.

## Validation philosophy used throughout

Every transform was first derived and checked in Python (numpy) against the FFTW
convention BEFORE writing C++ — that caught all the sign/index/normalization
traps. The C++ is then checked against the direct-matrix oracle in
`chebyshev_dct_kernel.hpp` to ~1e-12 and round-tripped to machine precision. Keep
this discipline: verify the algebra numerically, then implement.
