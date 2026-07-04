# The phase-packet FFT: a rotating local frame for the Bruun walk

Companion: `experiments/phase_packet_fft.py` (every claim below is verified
there to machine precision). Context: `src/detail/bruun_dif_dit_kernel_audit.md`
§1 (the walks in phase space), and the 2026-07-03 ChatGPT exploration
("DIT vs DIF Performance") that reached the torus/carry-block impasse.

---

## 1. Why the Stockham kernel needs no repack

The 512-point unrolled kernel maintains one invariant: the stage-t state
`A_t[k, j]` (shape `[2^t, N/2^t]`) is the 2^t-point DFT, **at natural frequency
index k**, of the time comb `{x[j + r·N/2^t]}`. Verified directly (1b in the
script). The index ledger is:

```
index = ( k : frequency bits accumulated so far, natural significance
        | j : time bits not yet consumed )
```

Each stage consumes the top remaining time bit of `j` and appends a new top
frequency bit to `k` — by **writing into a buffer of a different shape**
(`(e, 2q) -> (2e, q)`). Bit reversal is exactly the product of those m
single-bit moves; Stockham pays one per stage as *store addressing*, so no
layer ever holds data in a scrambled order and no separate permutation pass
ever exists. DIT/DIF-in-place instead keep the buffer shape fixed, so all m
bit moves pile up into one scrambled boundary (input side for DIT, output side
for DIF).

What it costs: the transform is out-of-place (ping-pong buffers), and the
column axis shrinks stage by stage — the last `log2(SIMD width)` stages have
columns narrower than a vector. Note also that the given kernel computes the
full complex transform of real input and truncates to 257 bins (2× redundant
arithmetic), allocates its nine buffers per call, and its `M` matrix is dead.

## 2. Yes: Stockham transplants onto our DIT with the cells unchanged

`real_stockham_rfft` in the companion script runs **our exact primitive** —
`pair_reduce_cs` on `(a, b) = (Re, −Im)` pairs plus the DC/Nyquist chains, the
same cells as `merge4_v_norm` / `terminal_radix2_v_norm` — under Stockham
addressing: state = folded real comb spectra per column, stage = broadcast-
`(c,s)` cells uniform along columns. Machine precision at all sizes, natural
order in **and** out. The fold algebra maps one-to-one: `pair_reduce`'s
`(ha, hb)` output is precisely the palindromic mirror-node write.

What this buys and costs relative to the mirrored-lane walk:

| | mirrored-lane (current) | real Stockham |
|---|---|---|
| input transport | bit-reversed comb gather (rev_ table, blocked) | natural order, none |
| output | natural (terminal fuses projection) | natural |
| interior SIMD | vertical via manufactured pairing axes (mirror, cA/cB) | vertical via layout; **width-agnostic, zero pairing axes** |
| in-place | yes (palindromic pairing) | no (ping-pong; the output buffer can serve, as in the odd-f32 forward) |
| crossing zone | last 1 (2-lane) / 2 (4-lane) merges | last 1 / 2 stages (columns < width) — **same depth: the pairing-axis law reappears as the column tail** |

The deep point: the mirrored-lane packing and the Stockham layout manufacture
the *same* symmetry (twiddle-sharing elements made vector-adjacent) — one by
lane packing at the seed, one by storage shape per stage. Since NEON already
measures at parity with the DIF and FFTW, the practical recommendation is:
**Stockham is the natural substrate for the AVX2/512 companion port**, because
its interior is width-agnostic — no new pairing axis per width doubling, no
cA/cB machinery, insertion points unchanged (the cells are identical).
Benchmark-gated as always.

## 3. The phase-packet FFT (the diagonal walk) — it exists, exactly

The object asked for: a factorization whose intermediate coordinates are
neither time samples nor frequency bins but chirped packets, with butterflies
pairing packets along time–frequency diagonals, reaching the Fourier frame by
progressive turning. Construction:

**Gauge the Stockham tree by quantized shears.** For `N = 2^m`, integer σ, the
time chirp `C_σ : x[n] ↦ e^{iπσn²/N} x[n]` acts on the stage-t mixed state as
a **monomial map** (pure relabel + per-coordinate phase):

```
A^σ_t[k, j] = e^{iπσj²/N} · A_t[(k − σj − c·2^{t−1}) mod 2^t, j],   c = σq²/N
```

**iff `σ·q_t² ≡ 0 (mod N)`** (`q_t = N/2^t`). This is the discrete symplectic
quantization of the torus: the admissible shear group at stage t has size
`2^min(t, m−t)` — a **diamond**. The frame is pinned to the time axis at the
input, pinned to the frequency axis at the output, and maximally free to tilt
mid-transform. (This is the exact finite answer to "how much can the frame
rotate at each scale.")

Pick any admissible schedule `σ_0 = 0, σ_1, …, σ_m ≡ 0` and run every stage in
its own sheared frame (`fused_sheared_stage`). Verified properties, any
schedule, machine precision:

- **Exactness**: the product of stages equals the DFT. No carry, no residual.
- **Sparsity**: every stage is 2-sparse — one Givens/reflection cell per pair,
  the same primitive class as the kernels.
- **Diagonal pairing**: partners sit at offset `(σ_t·q′, q′)` — a diagonal of
  slope σ_t in the packet lattice, not a coordinate axis.
- **The crisp test — answered YES**: the butterfly rotation is exactly
  `const · e^{−2πi(κ − Δσ_t·j)/2^{t+1}}` with `Δσ_t = σ_{t+1} − σ_t`:
  **constant along one diagonal coordinate**, the *increment* diagonal — the
  twiddles ride the direction the frame is currently turning. There is a
  second, separable 1-D law: a `j`-only chirp rotation `e^{iπΔσ_t j²/N}`,
  which vanishes whenever the frame does not turn that stage.
- **Packet anatomy**: intermediate atoms are combs with linear phase ramps
  proportional to their offset — genuine chirped (sheared-lattice) packets,
  partly time-local, partly frequency-local, at every interior stage.

DIT is the `σ ≡ 0` point of this gauge family; Stockham is its addressing.

## 4. The honest ledger (no free lunch, and what is actually gained)

- **Cost**: a turning stage (`Δσ ≠ 0`) pays one extra rotation per point (the
  j-chirp) on top of the DIT butterfly. So `σ ≡ 0` is the cost-minimal gauge
  point, and the phase-packet family **cannot reduce flops** — the per-stage
  angle-class count (2^{t+1} diagonal classes) is gauge-invariant. This is now
  proven by construction, not suspected.
- **Why the ChatGPT torus route hit a carry**: splitting Z_64 as an 8×8 torus
  demands the self-dual frame behave as a true product group at one scale;
  Z_64 ≠ Z_8×Z_8, and the defect (the `e^{−2πi tu/64}` carry chirp) is exactly
  an *inadmissible* jump in the diamond. The rank-4 carry blocks are the
  shadow of that quantization violation. The staged path never generates a
  carry because it only ever applies shears the current lattice can represent.
- **Boundary disorder**: with Stockham addressing, *no* schedule demands
  scrambled data at either end — shears are write-address remaps. The
  "half-disorder at both ends" trade only appears if you insist on in-place
  fixed-shape buffers.
- **What the freedom is good for**:
  1. *Chirped/LCT-flavored outputs at O(N) extra*: stop-and-chirp — since
     intermediate frames are chirped anyway, a dechirped or chirp-modulated
     DFT costs one boundary diagonal (per-pair Givens pass) instead of
     Bluestein's convolution. Natural fit for an audio library (sliding pitch,
     Doppler/dechirp, chirplet analysis).
  2. *Conditioning/scheduling knob*: angle tables and address strides can be
     reshaped along the path without touching exactness — unexplored territory
     for cache-geometry experiments.
  3. *Conceptual closure*: DIT, DIF, Stockham, and every "no-repack" variant
     are points of one family — a rotating-frame walk on the hypersphere whose
     twiddle law always collapses along the direction currently being swept.

## 5. Follow-ups if pursued

- Real-frame production form: the complex construction lifts verbatim (chirp
  = per-pair Givens; the folded demo in §2 shows the cell mapping). A
  production phase-packet kernel = real Stockham kernel + two angle tables per
  stage (diagonal + chirp) built from integer-reduced quadratic residues
  `σj² mod 2N`.
- The AVX2 companion should evaluate real-Stockham first (σ≡0); the shear
  schedule is a later free upgrade on the same substrate.
- Open question (small, well-posed): among admissible schedules, is there one
  whose *accumulated* angle tables minimize worst-case rotation error for the
  deep small angles — i.e., use the gauge freedom purely numerically?

---

## Addendum 2026-07-04: the phase FFT delivered, and two sharpenings

`experiments/phase_fft.py` is the standalone phase-packet FFT (complex +
real Bruun-cell forms), canonical diagonal schedule
`sigma_t = 2^max(0, 2t-m)` — unit diagonal through the self-dual half-depth
scale, then steepening into the frequency axis, riding the admissibility
boundary (`sigma*q^2 = N`) at every level past half depth.

**Sharpening 1 (correction).** The earlier "packet anatomy" probe printed a
*constant* per-step phase ramp and I labeled the atom "chirped" — wrong: a
constant ramp is a frequency SHIFT. Admissible shears move packet CENTERS
along diagonals (the lattice tilts); each atom's interior stays an upright
time-frequency rectangle. That is not incidental — it *is* the meaning of the
diamond: `sigma*q_t^2 = 0 mod N` says exactly "the slope step must look linear
across every packet," i.e. no intra-atom chirp is ever created. An exact
2-sparse walk can shear the lattice, never the tiles; demanding tilted tiles
is what generates dense carry corrections (the torus route's rank-4 blocks).

**Sharpening 2 (the ordering theorem).** With packets phase-referenced at
their own centers (absorbing the accumulated per-packet gauge phase
`e^{i pi sigma_t j^2/N}` into the packet definition — the natural convention),
the j-only chirp law vanishes and the cell rotation depends on the single
diagonal coordinate `delta = kappa - sigma_{t+1} j`. Storing packets
diagonal-major (row = delta) then collapses the address shear entirely:
`phase_fft_sheared` and `phase_fft_ordered` are **bit-identical**. So the
answer to "find a packet ordering where the Bruun angles depend on one
diagonal coordinate only" is: order by the output-frame diagonal; the angle
is `2*pi*delta / 2^{t+1}`, a function of the storage row alone — and in that
ordering the diagonal walk costs zero extra arithmetic. The walk lives in
what the coordinates denote (`packet_center`), not in the flop count: the
axis-aligned classical walks are this same object viewed in a frame that
hides the diagonal.
