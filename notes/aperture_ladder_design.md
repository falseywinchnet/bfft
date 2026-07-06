# The Aperture Ladder: design for a high-efficiency multi-scale TF fusion system

Status: theory validated to machine precision, reference implementation
benchmarked (`experiments/aperture_ladder.py`, 2026-07-06).  This document is
written so a sibling instance can build the production system from it without
re-deriving anything.  Companions: `experiments/dip_numba.py` (stage-capture
DIP + partial operators), `notes/dip_zak_fusion.md` (the Zak identity and
alignment law), `src/detail/bruun_dip_kernel.hpp` (the shipped C++ walk).

Prior art being superseded:
- arXiv:2604.15055 (Valdivia et al.): UOT barycenter fusion of 2 spectrograms;
  transport plan rebuilt per MM step (acceleration discarded), ~30 min for a
  sub-second signal, magnitudes-only output, cannot fuse >2 scales cleanly.
- codex's `zak_suffix_paused_level_solver.py` / `zak_middle_stack_vectorized_
  solver.py`: fast (0.9754 corr @ 0.209 s) but informal — circular-shift
  consensus (measured 45% relative residual on real audio), atom-patch gauge
  fits rediscovering the Weyl group empirically, seed of unknown provenance
  doing most of the work, no phase-truth (SNR) accounting.

Reference results (first second of daveandsimon.wav @ 8192 Hz, codex's exact
protocol: reassigned-spectrogram readout, cosine of sqrt-energy maps;
baselines long 0.7208 / short 0.4924 / geomean 0.6534):

| operating point                | corr    | SNR      | wall    |
|--------------------------------|---------|----------|---------|
| codex suffix-paused            | 0.9754  | (n/a)    | 209 ms  |
| codex vectorized middle-stack  | 0.97544 | (n/a)    | 268 ms  |
| **ladder FAST** (2 apertures)  | 0.9820  | −2.7 dB  | 162 ms  |
| **ladder QUALITY** (2 ap.)     | 0.9951  | +17.8 dB | 794 ms  |
| **ladder 3-aperture**          | 0.9971  | +27.8 dB | 1094 ms |
| blind 3-way geomean fusion     | 0.5341  | —        | —       |

The 3-aperture row is the aperture-scaling discovery: a third (wide, 4096)
aperture DEGRADES blind fusion (0.534 < 0.653) but adds ~10 dB to the ladder
— domain extension instead of contradictory blurring.  Fine-frequency readout
(4096-grid reassigned vs its own oracle): 0.9707 (2-ap) → 0.9794 (3-ap).

--------------------------------------------------------------------------

## 1. The objects

Everything lives on one dyadic frame walk.  For a frame u of length N = 2^m,
the DIP level-t state is the finite **Zak transform** on the lattice
(e, q) = (2^t, N/2^t):

    Z_t[d, j] = sum_{r=0}^{e-1} u[j + r q] e^{-2 pi i d r / e}
              = (e/N) sum_{k == d (mod e)} U[k] e^{2 pi i k j / N}

- **prefix** (time -> level t): one e-point FFT per column (batched
  `fft(u.reshape(I, e, q), axis=1)`), i.e. levels 0..t of the DIP walk;
- **suffix** (level t -> spectrum): the remaining butterflies, and it acts on
  each level-t ROW independently (nesting: the suffix on row d is a twisted
  q-point sub-walk; verified — zeroing all rows but d leaves spectrum support
  exactly on the comb {k == d mod e});
- every stage map obeys P P^H = 2 I: fixed, perfectly conditioned geometry.

`dip_numba.py` provides all of these with full stage capture; the C++ kernel
(`bruun_dip_kernel.hpp`) computes the same walk in place — pausing it at
level t is "stop after t levels", so paused states are FREE in kernel terms.

## 2. Exact covariance (validated to 1e-14; never approximate these)

1. **Weyl translations are twisted lattice translations.**  Time shift by
   s = a q + b: j-roll by b with phase e^{2 pi i d (a + carry)/e} per row;
   modulation by bin 1: d-roll with phase e^{2 pi i j / N} per column.
   Circular only — see (3) for real frames.
2. **Equal-q aperture coupling.**  Pause every aperture at the level with the
   SAME column count q*: aperture N_a sits at t_a = log2(N_a) − log2(q*),
   state shape (e_a, q*).  For rect frames centered at the same point with
   (N_b − N_s)/2 divisible by q*, the small state is a LINEAR, COLUMN-LOCAL,
   j-INDEPENDENT image of the big one:
       Z_s[:, j] = M Z_b[:, j],   M = F_{e_s} · crop_rows · F_{e_b}^{-1}
   (one e_s x e_b Dirichlet-band matrix for ALL columns; windows enter as a
   diagonal weight in comb coordinates).  This is the "shared lattice" that
   makes multi-scale fusion consistent: small apertures share the fine-time
   axis exactly; large apertures EXTEND the d axis (fine frequency).  There
   is nothing to transport and nothing that can contradict.
3. **Frame consistency is exact in comb coordinates.**  When q | hop, frames
   of one record satisfy C_i[r + hop/q, j] == C_{i+1}[r, j] IDENTICALLY
   (C = dewindowed frame reshaped (e, q); a reshape, no transform, no
   boundary error).  Codex's circular twisted-shift consensus is this
   relation minus the wrap correction — measured 45% relative residual on
   real audio.  Consequence: **overlap-add of dewindowed frames IS the exact
   consensus**, so the canonical state should be THE RECORD, with paused
   per-aperture states as cheap views (prefix = one small batched FFT).

## 3. The estimation problem (what fusion actually is)

All observations are magnitudes of fixed linear analyses of ONE record x:

    y_{a,i} = | F_{N_a} diag(w_a) S_{o_i} x |        (aperture a, frame i)

Fusion = solve for x.  This dissolves every difficulty the OT formulation
fights: grids never need aligning (each aperture keeps its own), scales never
contradict (they share x), and the geometry never moves (operators constant),
so momentum/curvature information PERSISTS across iterations — the precise
thing the per-step transport-plan rebuild destroys.

The problem is multi-window STFT phase retrieval.  Redundancy here (~4x per
aperture, several apertures) makes it well posed; the algorithmic questions
are only (a) where to get a basin-correct seed cheaply, (b) what descent
engine to use, (c) how to schedule apertures.

## 4. The phase field (why a seed exists in closed form)

For the Gaussian window g(t) = e^{-pi t^2 / lam}, the STFT factors as
V(tau, w) = e^{-pi lam w^2} G(tau − i lam w) with G ENTIRE.  Cauchy–Riemann
on log G = s + i phi gives the **phase-gradient laws** (window-centered
convention; derived and verified this session):

    d phi / d tau = 2 pi w + (1/lam) ds/dw          (time law)
    d phi / d w   = − lam ds/dtau                   (frequency law)

The magnitude surface DETERMINES the phase up to one constant per connected
component and vortex holonomy (zeros of V).  Hann behaves as lam ≈ 0.25645 n²
(regression-confirmed on real audio: fitted 0.25685 n²).

**Measured accuracy regime (do not re-derive, this cost a day):**
- time law: excellent everywhere (0.0006 rad/sample synthetic; 0.19 rad/hop
  weighted RMS on real polyphonic audio at n=1024, hop=n/8);
- frequency law: 20% residual on a clean chirp (discretization: central
  differences across hops of a curving ridge), 53–84% residual on real
  polyphonic audio (multiple components per mainlobe break magnitude-based
  group delay).  IRREDUCIBLE by calibrating lam.

Design consequence: integrate phases mainly along TIME (per-bin instantaneous
frequency), let the descent engine supply cross-ridge coupling through the
signal domain.  Two integrators are implemented:
- `pghi_phases`: magnitude-ordered heap integration (PGHI); convention
  conversion to frame-local phases is exactly −2 pi m c0 / n (c0 = (n−1)/2 —
  the global-time advance is already inside the 2 pi w term; adding a frame-
  offset term is a bug we made and fixed);
- `gauge_phases`: one weighted-least-squares (graph Poisson) solve of the
  gradient field — errors average over cycles instead of accumulating along
  a spanning tree.  On polyphonic audio the frequency law's bias currently
  makes the heap version better; the Poisson version becomes the right tool
  once the frequency-edge weights are set from measured coherence (open
  problem 10.2).

**The basin law (measured):** the SHORT-aperture PGHI seed (dense frames,
time law reliable) leads L-BFGS to the phase-TRUE basin (SNR +17.8 dB); the
LONG-aperture seed reaches high readout-corr fastest (0.982 @ 60 iters) but
stalls phase-frozen (SNR −2.7 dB).  Offer both operating points.

## 5. The descent engine

Joint amplitude loss over ALL apertures, record variable u:

    L(u) = sum_a w_a sum_i || |F_a W_a S_i u| − y_{a,i} ||^2,  w_a = N_ref/N_a

- Exact gradient: for each aperture ONE batched FFT + one batched IFFT +
  a scatter-add (`make_loss_grad`); 1.2 ms/evaluation for 8192 samples,
  3 apertures, numpy.
- Engine: L-BFGS (maxcor ≈ 8).  Quasi-Newton is the *point*, not a detail:
  the fixed geometry is what lets curvature memory accumulate — the
  structural advantage over Bregman-MM/OT whose plan changes per step.
  Plain GL/alternating projections stall on dense polyphony (measured: corr
  0.27 after 80 sweeps from random); L-BFGS from the same start reaches
  0.983 in 400 iterations.
- Momentum-GL sweeps (`Family.project`) are kept for cheap warm phases and
  for the streaming variant, but they are not the workhorse.

**Coverage discipline (a bug class to design out):** record samples with
near-zero total window coverage (edges) are unobservable; synthesis must
zero them, never divide by vanishing coverage (`Family.covered`).  Symptom
when violated: loss improves while readout corr collapses.

## 6. The aperture ladder (multi-scale scheduling)

- Dyadic apertures N_a, hops h_a = N_a/8 (denser is better for the seed).
- Seed on the SMALLEST aperture (phase-true basin; its time law is densest).
- Descend on all apertures JOINTLY (the record couples them; the equal-q
  algebra of S2.2 is the proof this is consistent, and supplies the
  column-local layout for implementations without a global record).
- Weights w_a = N_ref/N_a equalize per-coefficient scale across apertures.
- Adding apertures helps monotonically (measured 2→3: SNR +17.8 → +27.8 dB,
  loss 25 → 3.0).  Contrast: blind geomean of 3 scales DEGRADES (0.653 →
  0.534) because magnitude marginals of different scales are increasingly
  dissimilar views — exactly the user's aperture-scaling observation.  The
  nesting figure ("prefix crops reveal nested versions of the same
  structure") is theorem S1-nesting in disguise: dyadic scale families are
  suffix levels of one another.
- Readout: any product, computed once, feed-forward (reassigned spectrogram
  on the F1 x T2 grid for the paper protocol; the recovered waveform itself
  is the stronger deliverable — magnitude-fusion methods cannot produce it).

## 7. System architecture (production implementation map)

Reference (numpy/scipy, this repo): `experiments/aperture_ladder.py`
    Family (batched analysis/synthesis/projection, coverage mask)
    pghi_phases / gauge_phases / pghi_seed  (closed-form seeds)
    make_loss_grad + scipy L-BFGS-B         (engine)
    solve_ladder                            (pipeline)
    aperture_coupling / time_shift_zak / zak_prefix (exact algebra + tests)

Kernel-level (C++/BFFT) mapping for the high-efficiency system:
1. **Analysis/gradient FFTs** → the Bruun DIF/DIP kernels.  Real input:
   the real Bruun residue-pair walk halves flops (`dip_rfft_stages` math).
2. **Paused states for free** → the in-place DIP walk pauses at level t by
   construction (stage capture); suffix-only endpoint projections cost
   (m−t)/m of a transform, and the magnitude projection is diagonal at the
   endpoint.  In numpy this saving is invisible (one library call either
   way); in the kernel it is real flops and bandwidth.
3. **Consensus/OLA** → comb-coordinate scatter (S2.3) is a strided-stream
   operation of exactly the shape the DIP egress machinery (T_w tables,
   comb writes) already implements in `bruun_dip_kernel.hpp`.
4. **Cross-aperture coupling without a global record** (streaming/blocked):
   equal-q ladder, one small Dirichlet matrix per aperture pair, applied
   column-wise — SIMD-trivial, j-parallel.
5. **PGHI heap** → numba/C++ with a bucket queue (magnitudes quantized) is
   O(cells); the 40–70 ms python heapq is the current bottleneck of the
   FAST point (162 ms total).  Target < 10 ms.
6. **L-BFGS** → own two-loop recursion (fixed small memory, no scipy),
   fusing the gradient's IFFT scatter with the direction update.

Budget model per L-BFGS iteration, record length L, apertures a with
overlap factor v_a = N_a/h_a:  flops ≈ sum_a v_a · c_fft(N_a) · L/N_a · 2,
memory traffic ≈ a few streams over L·sum v_a.  For the benchmark problem
that is ~1.2 ms/iter in numpy; the kernel path should land 3–5x lower.

## 8. Evaluation protocol (keep results comparable)

`load_first_second` (48 kHz → 8192 Hz, first second, mean-removed, 0.95
peak), apertures long n=1024 h=128 / short n=128 h=32 (+ wide n=4096 h=512),
Hann; oracle = reassigned spectrogram of the true record, n=1024, at short-
frame centers; corr = cosine of sqrt-energy maps.  Baselines must reproduce
0.7208 / 0.4924 / 0.6534 before any comparison.  ALWAYS also report interior
SNR (cut 256): corr alone saturates and hides phase-frozen solutions.

## 9. Honest negatives (measured; do not re-litigate without a new mechanism)

- Circular twisted-shift consensus on windowed frames: 45% relative residual
  on real audio.  Use comb/OLA consistency (exact) instead.
- The frequency phase-law on polyphonic audio: 53–84% residual; lam
  calibration does not help (fitted constant ≈ published 0.25645 n²).
- Plain alternating projections (any momentum) from cold starts on dense
  audio: stalls (corr ≤ 0.27 at 80 sweeps).  Needs the seed + quasi-Newton.
- A magnitude-consistent record with WRONG phases does NOT score high on the
  reassigned-corr metric (0.03) — the metric is phase-sensitive through the
  cross-spectra; codex's 0.974-scoring seed was genuinely phase-coherent.
- Naive Poisson gauge integration with amplitude-squared weights on both
  edge families: worse than heap (frequency-law bias is systematic, not
  zero-mean; weighting must reflect LAW quality, not just magnitude).

## 10. Open problems (ranked)

1. **Vectorized/bucket PGHI** (S7.5) — the FAST point's remaining cost.
2. **Coherence-weighted gauge solve**: estimate per-edge law reliability
   (e.g., local ridge count / lobe interference detector), then the Poisson
   integrator should beat the heap everywhere; it also yields uncertainty.
3. **Vortex accounting**: spectrogram zeros carry ±1 holonomy; the LS field
   smooths them.  Detect via residual circulation on plaquettes; correct
   with branch cuts.  Relevant at high noise.
4. **Streaming ladder**: sliding-window records with the equal-q coupling
   carrying state across blocks (no global solve); the DIP kernel's paused
   states make the per-block work suffix-only.
5. **Windows in comb coordinates**: S2.2 coupling stated for rect; Hann is a
   diagonal weight in comb space — fold it into M to couple the OBSERVED
   families directly (removes the rect-frames caveat).
6. **>3 apertures / full pyramid**: measured monotone gain 2→3; establish
   the saturation curve and the redundancy/noise tradeoff.
7. **Noise robustness**: magnitude noise → loss floor; L-BFGS on the
   Poisson-seeded field with early stopping is the natural regularizer;
   quantify against the OT method's smoothing.

## 11. What to tell a sibling in one paragraph

Treat every spectrogram, of every window size, as a magnitude observation of
one latent record through a fixed linear operator.  Seed the phases in
closed form by integrating the time phase-law on the smallest aperture
(PGHI); then run L-BFGS on the joint amplitude loss of all apertures at once
(batched FFTs; fixed geometry means curvature memory persists — this is the
whole advantage over transport methods).  Zero unobservable edge samples.
Read out any product once, feed-forward.  The Zak/DIP algebra (twisted Weyl
translations, comb-exact consistency, equal-q Dirichlet coupling, suffix
nesting) is what makes the kernel-level implementation cheap and the
multi-scale consistency provable — use `dip_numba.py` for the operators and
`aperture_ladder.py` as the executable specification.
