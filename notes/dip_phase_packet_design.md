# DIP → transport-optimal phase-packet FFT (design)

Companion: `experiments/phase_fft.py` (the real diagonal walk — the target),
`experiments/dip_transport_symmetry.py` (the transport-conservation + real-Givens
proofs), `src/detail/bruun_dip_kernel.hpp`. `experiments/dip_phase_packet.py`
is the RETIRED four-step (see the 2026-07-04 correction below).

## Where the DIP is (2026-07-04)

- The stages are now column-vectorized (isoclinic rotation: twiddle broadcast
  along columns, contiguous V2/AVX2 loads/stores, scalar tail — `cell_fwd`,
  `cell_inv`, `ridge_*`). Roundtrip machine-precision at all sizes. This closed
  part of the gap to DIF ("closer").
- Remaining gap: DIP is ~2× DIF in cache (compute/out-of-place overhead) and
  ~2.8× at N≥64K (transport). The transport component is the ceiling and it is
  structural, not codegen — see below.

## The transport catastrophe (why the shape, not the code, is the problem)

The current walk is breadth-first and out-of-place: each of ~log2(N) stages
reads all N and writes all N. DRAM read+write passes therefore grow as
**2·log2(N)** — measured/modeled:

| N | sweep (current) | radix-2 depth-first | four-step |
|---|---|---|---|
| 65536 | 24 | 10 | 2 |
| 1048576 | 32 | 18 | 2 |
| 16777216 | 40 | 26 | 2 |

No amount of vectorization fixes a walk that touches DRAM 32× at 1M. The whole-
array double-buffer sweep is the thing to delete.

## The transpose-four-step was a wrong turn (2026-07-04 correction)

An earlier draft of this note proposed a Cooley–Tukey FOUR-STEP (factor `N=P·Q`,
length-Q leg, phase twiddle, length-P leg, transpose out) with a "complex row
leg." Two objections, both correct, kill it (verified in
`experiments/dip_transport_symmetry.py`):

1. **Boundary transport is not conserved.** The disorder in an FFT is the
   bit-reversal permutation: exactly `log2(N)` single-bit moves, an INVARIANT
   you can only place, not destroy. DIT piles them on the input (1 boundary
   sweep), DIF on the output (1 sweep). The four-step needs a **reshape-in AND a
   scatter-out — 2 boundary sweeps.** That doubling is the tell that it re-pays
   transport a correct walk should absorb. Acceptance law for any phase-packet
   candidate: `reshape_in + scatter_out ≤ DIT single-sided disorder (= 1)`.

   | walk | where the log N bit-moves go | boundary sweeps |
   |---|---|---|
   | DIT | all on the input | 1 |
   | DIF | all on the output | 1 |
   | transpose-four-step | two end transposes | **2 ✗** |
   | diagonal (phase-packet) | one per stage, into the store address | **0 ✓** |

2. **It shouldn't touch complex space — it's Bruun.** The four-step's "complex
   row leg" applied to a Bruun pair `(a,b)=(Re,−Im)` is *bit-identically* a real
   `pair_reduce_cs` Givens (‖diff‖ = 1.1e-16). The complex framing was an
   artifact; in the `(a,b)` encoding it is the SAME real cell already run at
   every interior stage, with a general angle. No new "complex cell" exists.

## The right object: block the REAL diagonal walk (natural order both ends)

The walk that satisfies the law above already exists and is validated:
`experiments/phase_fft.py :: phase_fft_real` — the r2c transform built ONLY from
`pair_reduce_cs` cells + the walking DC/Nyquist ridge, in diagonal-major packet
order, angle `θ(δ)=2πδ/2^{t+1}` a function of the output row alone. It reads `x`
in natural order and writes `X` in bin order: **boundary transport = 0.** The
bit-reversal is spent one bit-move per stage as store addressing (Stockham), so
it never piles into a boundary pass. `A[δ,j]` is literally a packet — coarse
frequency δ, fine offset j, rotated diagonally; the shear is free (address
meaning, not data motion: `phase_fft_sheared == phase_fft_ordered` bit-for-bit).

The one honest gap is the INTERIOR, not the boundary: `phase_fft_real` as
written is out-of-place, `log2(N)` passes (a ping-pong). That is what must be
cache-blocked. The four-step *looked* attractive only because its blocking was
visible — but its blocking arrived bundled with the two artifacts above. The
correct move is to block the real diagonal walk depth-first, with the cross-
array re-addressing carried by the free diagonal shear rather than a transpose:
interior → ~2 DRAM passes, boundary stays 0, cells stay real.

## The blocking, made concrete (2026-07-04, validated)

`experiments/dip_blocked.py` builds and MEASURES the depth-first blocking under
an LRU cache-line simulator. Three walks, all exact to machine precision:

| N | ping-pong (flat log-N) | blocked + transpose | **self-sorting** |
|---|---|---|---|
| 16384 | 28.0 | 4.9 | **3.0** |
| 65536 | 32.0 | 5.0 | **3.0** |
| 262144 | 36.0 | 5.0 | **3.0** |

The winner is **self-sorting** (`blocked_ss`): recursive four-step on the column
length `L = L1·L2`, real Givens cells only, with NO interior transpose:

```
blocked_ss(off, L, stride):
  if L <= leaf:  in-cache DFT (the existing vectorized stages);  return
  L1, L2 = split(L)           # ~sqrt factor
  for i2 in [0,L2):  blocked_ss(off + i2, L1, L2·stride)      # column sub-transforms
  for k1,i2:  elem at (i2 + perm(L1)[k1]·L2) *= W_L^{k1·i2}    # PHASE twiddle, perm-indexed
  for k1 in [0,L1): blocked_ss(off + perm(L1)[k1]·L2, L2, stride)  # row sub-transforms
  # output bin k lands at off + perm(L)[k]·stride  (perm composes: no data move)
```

`perm(L)[k2·L1+k1] = perm(L1)[k1]·L2 + perm(L2)[k2]`. The interior never moves
data to reorder; the twiddle simply indexes through `perm(L1)`, and the single
output permutation `perm(L)` is resolved in the boundary bin-extraction the
kernel already does (`output[d].re = src[...]`). So: interior shuffle 0,
boundary = the extraction that already exists, transport flat at 3 passes. The
middle twiddle is the packet rotation — this is the phase-packet object.

## The r2c composition (2026-07-04, validated)

`dip_blocked.py :: r2c_composed` proves the whole thing composes for REAL input,
machine precision (1e-14..1e-13, N up to 16384). Factor `N = N1·N2`, index map
`n = n1 + N1·n2`, `k = N2·k1 + k2`:

```
inner   A[k2, n1] = Σ_{n2} x[n1 + N1·n2] · W_N2^{n2·k2}    real leg (real (a,b) cells)
twiddle A[k2, n1] *= W_N^{n1·k2}                            phase Givens
outer   R[k1]     = Σ_{n1} A[k2, n1] · W_N1^{n1·k1}         self-sorting complex leg
        X[N2·k1 + k2] = R[k1]                               (k1 slow, k2 fast)
```

The inner legs are `phase_fft_real` (real cells, and — for the half spectrum —
the r2c fold already validated); the outer legs are `blocked_ss` (self-sorting,
real Givens, transport-flat). So all four properties hold together: exact, real
cells only, boundary 0, DRAM flat at ~3 passes.

## C++ implementation plan

1. **No gather in, no scatter out.** `x` enters in natural order; `X` leaves in
   bin (or diagonal-major) order. This is the property to preserve at all costs
   — it is what makes the walk beat DIT's single-sided boundary cost.
2. **Real diagonal cells only.** Every butterfly is `pair_reduce_cs` on `(a,b)`
   with angle `θ(δ)` (the DC/Ny ridge is its δ→ridge limit). Column-vectorized
   exactly as the shipped stages already are — the angle is broadcast along the
   column (isoclinic), no permute. There is NO separate complex leg.
3. **Cache-block the interior depth-first.** Split the m stages so a contiguous
   band of diagonal-major rows (a packet block) is transformed while resident;
   the shear re-addresses across bands for free. Target: total DRAM passes ≈ 2,
   independent of N, with (1) and (2) intact. This replaces the log(N)-pass
   ping-pong — the actual source of the DIP's large-N gap to DIF.
4. **Inverse is the mirror**: the same real cells run in reverse order (expand
   instead of reduce), same diagonal-major addressing, boundary still 0.
5. **Small-N (N ≤ leaf) stays on the current in-cache vectorized stages** — it
   is already fine there and needs no blocking.

## SHIPPED (2026-07-05): the in-place diagonal span walk

The C++ that landed in `src/detail/bruun_dip_kernel.hpp` supersedes both the
Python four-step prototype AND the self-sorting variant above. The realization
that unlocked it: writing each cell's two children INTO THE PARENT'S OWN SPAN
makes every span self-contained forever — fully in-place (work = n, down from
2n; no ping-pong, no scratch, concurrency-safe), depth-first (cache transition
emergent, no block knobs), the Nyquist-promotion copy free, and the SIMD purely
vertical (4/8 contiguous streams, broadcast twiddles, zero permutes; radix-4
fused two-level cells above the boundary granularity, `norm2_fused`-shape).

The boundary (the user's dual-polarity insight, validated hard): pay the
long-range disorder as COMB STREAMS at both ends, confine all scramble to L1.
The leaves of any subtree `(d, E)` are the bins `{jE ± d}` (two combs), and
the tree-order → bin-order permutation within a subtree is UNIVERSAL
(d-independent; leaf value = `m·E + s·d` with `(m,s)` path-only), so one tiny
shared table `T_w` per span size drives a streaming per-subtree egress
(forward) / ingress (inverse). Asymmetries that measurement forced:
- inverse ingress = comb reads + L1 scatter (a separate injection pass and a
  fused per-leaf gather both lose);
- forward egress = L1 gather + comb writes, with a radix-2 parity step so the
  RFO-paying writes land on the densest combs the L1 span budget allows (the
  inverse skips the parity step — reads prefetch at any density);
- seed at level 8, not 16: 16+16 power-of-two-strided streams alias L1 sets
  past associativity (measured 9× the 8-stream seed at 1M).

Measured vs DIF (M-series NEON, min-of-5, `dif_vs_dip_benchmark`):

| n | fwd DIP/DIF | inv DIP/DIF |
|---|---|---|
| 1024 | 1.28 | 1.00 |
| 4096 | ~1.1–1.5 | **0.57–0.89 (wins)** |
| 65536 | 1.56 | 1.10 |
| 262144 | ~2.3 | 1.11 |
| 1048576 | 1.89 | 1.17 |
| 4194304 | 2.22 | 1.65 |

Starting point was fwd 2.6×/inv 1.6–2.4× (breadth-first ping-pong). The
inverse is at DIF parity (or wins) through 1M.

### Forward optimization pass (2026-07-05)

Two wins landed, both grounded by isolation profiling (`headcmp`/`wcmp`
harnesses, in-process back-to-back to cancel this machine's severe thermal
noise — DIF-ratio benchmarks swung ±30% run-to-run and could not be trusted):
- **`phase_table_index` divide → shift.** `d*n/(2e)` was a 64-bit `sdiv` at
  every node; `2e` is always a power of two and exactly divides `d*n`, so it is
  `(d*n) >> (ctz(e)+1)`. (Index compute was NOT the dominant cost — `noidx`
  ≈ `nocells` — but the sdiv still cost real cycles at small/mid n.)
- **Inline the `span8` leaf codelets** (`BRUUN_ALWAYS_INLINE`). They were real
  calls; inlining removed the leaf-call overhead. Net: 4096 forward 1.5→1.11×.

What profiling RULED OUT (don't retry):
- **Egress is cheap** — isolating it (`wcmp`) shows 0–11% of the forward,
  mostly noise; it is NOT the gap. A split-comb (two monotonic streams) egress
  measured a dead wash back-to-back.
- **The WALK is the whole forward cost**, and the forward walk is ~35% slower
  than the inverse walk at 64K (both cache-resident, same cell count) yet
  converges by 4M — i.e. a COMPUTE/scheduling asymmetry in the forward cell's
  dependency chain (`cell4_fwd_ip`), not transport. It hides once DRAM-bound.
- The forward radix-2 parity step at `2*kEgressW` is net-neutral (its
  egress-density rationale is void now that egress is cheap); kept only because
  removing it is within noise and changes span sizing.

Remaining forward levers (larger efforts, uncertain payoff on this noisy box):
flatten the recursion into a DIF-style precomputed op schedule (kills the
residual call overhead), or restructure `cell4_fwd_ip` to shorten its two-level
critical path (the 64K compute asymmetry). The inverse needs nothing.
