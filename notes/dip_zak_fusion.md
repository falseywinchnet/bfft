# DIP intermediates are Zak states, and the mid-level alignment law

Companions: `experiments/dip_numba.py` (the Numba DIP with full stage
capture — the kernel deliverable), `experiments/dip_zak_fusion.py` (the
super-resolution fusion demonstration; figures in `experiments/out/`).
Context: Valdivia–Cazelles–Févotte, "Enhancing time-frequency resolution
with optimal transport and barycentric fusion of multiple spectrograms"
(arXiv:2604.15055) — UOT barycenter of a long-window and a short-window
spectrogram by block MM; the transport plan changes every step (acceleration
info discarded), ~30 min for a sub-second signal.

## The identity (exact, measured to 1e-14)

Unrolling the ordered DIP walk (`phase_fft.py :: phase_fft_ordered`) gives
the closed form for the level-t state, q = N/2^t, e = 2^t:

    B_t[delta, j] = sum_r x[j + r q] e^{-2 pi i delta r / e}          (time side)
                  = (e/N) sum_{k == delta mod e} X[k] e^{2 pi i k j / N}   (bin side)

i.e. **every DIP level is the finite Zak transform of the frame on the
dyadic lattice (e, q)** — the walk is a curve through the full family of
fractional time-frequency representations, time axis (t=0) to frequency
axis (t=m). The bin-side comb {k ≡ delta mod e} is the Bruun {jE±d}
signature seen from the other end. Coordinates at level t:

- row delta = fine frequency (LOW bits of the bin index k)
- col j     = fine time (LOW bits of the sample index n)
- coarse frequency = oscillation of the row along j (lives in PHASE)
- coarse time      = phase pattern down the column across rows

Corollaries: (1) two tones Δk bins apart occupy different rows as soon as
e ∤ Δk — close-tone separation happens in the FIRST levels and never trades
against time localization; (2) every stage map is a scaled unitary
(P_t P_t^H = 2^t I) — descent/adjoint/momentum geometry between levels is
fixed and perfectly conditioned; (3) each level packs to exactly N scalars,
so the full stage stack is an (m+1, N) array.

## The alignment law (the discovery, measured)

Fair game: give every level its natural rigid-motion group — the e×q
lattice translations, exactly N motions at EVERY level (equal DOF) — and
measure the best-aligned cosine between two captures of the same content
under different settings (Hann frames 320 samples apart, N=1024, fs=2048):

| content    |        | t=0   | t=3   | t=4   | t=5   | t=7   | t=10 (FFT) |
|------------|--------|-------|-------|-------|-------|-------|------------|
| two tones  | \|mix\|| 0.997 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| paper mix  | \|mix\|| 0.658 | 0.826 | 0.862 | 0.854 | 0.827 | 0.808 |
| slow chirp | \|mix\|| 0.808 | 0.845 | 0.834 | 0.702 | 0.553 | 0.553 |
| fast chirp | \|mix\|| 0.824 | 0.865 | 0.858 | 0.852 | 0.748 | 0.539 |
| paper mix  | cplx   | 0.571 | 0.512 | 0.506 | 0.423 | 0.291 | 0.259 |

1. **Complex states decohere with depth** — the "dissimilar phase at the
   final representation point" that makes final-form fusion an OT problem.
2. **Magnitudes of the mixes align best at INTERIOR levels**, and the
   interior advantage over final products grows with how fast the content
   wanders (slow chirp 0.845 vs 0.553; fast chirp 0.865 vs 0.539).
   A mid-level lattice translation has BOTH a time and a frequency axis, so
   it absorbs a diagonal (evolving) move that no frequency-shift of a final
   spectrum can express.
3. **Stationary content saturates everywhere** (1.000 from t=4 up) —
   final-product fusion is adequate exactly and only for content that
   needs no fusing.
4. Optimal depth scales inversely with wander rate (slow chirp peaks t≈3,
   fast chirp t≈3–6 plateau, stationary flat) — depth is a tunable
   "fractional" parameter matched to signal dynamics.

## The demonstration (experiments/dip_zak_fusion.py)

Paper-style problem: 100+120 Hz beating pair, 50 ms global pause, 300→380 Hz
chirp; observations ONLY the two Hann power spectrograms (1024-window hop
128; 64-window hop 16). Method: dual-family alternating magnitude
projections with over-relaxation momentum (valid because operators are
fixed), restarts ranked by consistency loss, short amplitude-flow polish.
Final products are pure feed-forward DIP finishes from the aligned
intermediates — no descent in final FFT form.

Results (M-series laptop, ~15 s wall vs paper's ~30 min):
- interior recovery SNR +19 dB (first/last ~128 samples get near-zero Hann
  weight — genuinely unobserved);
- fused product (reassigned long-window spectrogram of the recovered
  signal, F1×T2 grid): cosine-to-ideal **0.794 vs oracle 0.792** — at the
  oracle; baseline geometric-mean fusion 0.521, best single observation
  0.649;
- the structural win magnitude fusion cannot reach: the baseline imprints
  the 20 Hz beating pulses as false gaps on both tone lines
  (interference≠silence is invisible to any magnitude barycenter, OT
  included); the aligned mixes know the lumps are superposition — the
  fused lines are continuous, the real pause sharp;
- momentum persistence: loss 8.1 vs 67.0 at 400 iterations with/without
  momentum — the fixed-geometry contrast to per-step OT plan rebuilds.

## Honest negatives (measured, don't re-litigate without new mechanism)

- Soft-threshold (ℓ1) priors interleaved with the projections — in time,
  mid-level Zak, or final FFT domain, calibrated to equal survival
  quantiles — are NOT reliably helpful at these observation densities;
  basin luck dominates and the ordering flips between regimes. The working
  levers are redundancy, momentum, and restart selection.
- Windowing the j-axis of a mid-level state cannot beat the Gabor limit by
  itself: the representation is critically sampled (unitary), alias
  discrimination costs the full row. Super-resolution must come from priors
  or multi-observation fusion, not from re-windowing intermediates.
- ℓ0/energy compaction is NOT better at mid levels for quasi-stationary
  content (final FFT compacts tones best); the mid-level advantage is
  specifically ALIGNABILITY across settings, not sparsity.

## Next levers

1. Operationalize the law inside recovery: cross-frame consensus of
   mid-level |mix| (overlapping long frames agree at level t* after a known
   lattice roll — a temporal-coherence regularizer unavailable at final
   level for moving content).
2. Real-world audio: noise robustness, multi-component, mel-grid targets
   (the paper's supplementary case).
3. The short-window family can also be lifted to ITS mid-levels — mutual
   alignment of the two families' Zak states at matched lattices
   (16×64-of-1024 vs 64-frame states) instead of through the time domain.
4. Proof track: the law as a theorem about the Weyl–Heisenberg orbit of
   evolving atoms — the lattice-translation orbit approximates the atom's
   phase-space flow best when the lattice aspect matches the flow's slope
   (cf. the twisted-walk quantization theorem in `notes/dip_twisted_walk.md`).
