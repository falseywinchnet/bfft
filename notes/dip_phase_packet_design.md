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

## Status

Corrected: the transpose-four-step is retired (fails transport conservation +
introduces complex arithmetic Bruun does not need). The real diagonal walk
`phase_fft_real` is the target — exact r2c, real cells only, boundary transport
0. The remaining build is the DEPTH-FIRST CACHE-BLOCKING of that walk (interior
log N → ~2 passes) with natural order preserved at both ends; the vectorized
real cells and the ordering theorem are in hand. `experiments/dip_phase_packet.py`
(the four-step prototype) is kept only as the retired reference.
