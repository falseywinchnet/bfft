# Waterfall math audit: STFT, reassignment, and intrinsic FCT

Date: 2026-07-10

This note separates the three viewer measurements and records the corrections
that were necessary when the STFT experiments moved into a streaming complex-IQ
waterfall.

## 1. The deployed FCT contract

For complex IQ `x[t]`, define

    C(k,tau) = sum_{t=0}^{tau-1} x[t] exp(-2 pi i k t / N).

The public FCT returns the globally maximizing support

    tau_k = argmax_{tau in [t_min,N]} |C(k,tau)|^2 / tau

and emits the complex value `C(k,tau_k)`. Its phase is therefore the requested
`atan2(A,B)` of the sine/cosine correlation pair at the selected support.

This is now the intrinsic phase-disk walk, not the former dyadic proxy plus
local endpoint refinement. The half-edge disk tests are certified upper bounds;
bins at one time node are joined into SIMD-mask packets, while an incremental
fractional-frequency provider supplies the demanded block totals. Brute-force
tests cover real and complex inputs, wrapped negative frequencies, adversarial
events, and non-dyadic `t_min` boundaries.

`activity=0` returns the exact optimum for every bin. A positive activity floor
can return the ordinary full-support Fourier value for a bin whose proven
optimum is below that floor; it does not substitute a guessed support.

## 2. Why the viewer declares a minimum aperture

The unconstrained mathematical reference has `t_min=1`. At `tau=1`, however,
all frequency phasors equal one, so one complex sample contains no frequency
identity. On wideband IQ the normalized objective can consequently choose one
sample for most bins and produce a frequency-flat image. That is a correct but
unhelpful answer to an underconstrained visualization question.

The viewer declares `t_min=N/32`, equal to its dense display hop. The intrinsic
walk then certifies the optimum over `[N/32,N]`; it does not try to predict where
the support should be. This boundary states the minimum aperture at which a
cell is allowed to claim frequency identity. The unrestricted public transform
remains available as the finish-line reference.

## 3. Timing and intensity are not STFT conventions

An STFT frame owns one center time. An FCT frame owns one origin and a distinct
endpoint `origin + tau[k]` for each frequency. The viewer therefore analyzes
preceding origins, scatters every cell to its measured endpoint, and keeps the
strongest claimant in each display cell. The sample-coordinate calculation
retains the residual when a scrub position is not aligned to the hop.

The displayed FCT amplitude is

    M(k) = |C(k,tau)| / sqrt(tau),

whose square is exactly the optimized score. Multiplication by `sqrt(N)` would
only add a transform-size-dependent dB offset and previously saturated the
normal waterfall range. `tau/N` remains a second measurement and is encoded as
a warm tint; it is not folded into intensity.

## 4. What actually caused the blurry "super-res" waterfall (2026-07-10)

The 2026-07-09 conclusion — that the active-delta fusion was intrinsically
"weaker and blurrier" than reassignment — was wrong.  Two concrete defects in
the fusion path caused the blur, and both are now fixed; the two-aperture
DIP-Zak fusion is again the viewer's super-resolution mode, with reassignment
kept as a separate observable.

### 4a. The direct seed was not spectrum-consistent (Hann² defect)

`SolverC::direct_seed` built the state as `tap3_adj(back_to_level(Y))`.  But
`tap3` is pointwise periodic-Hann in frame time, so the readout round-trip
`tap3 . tap3_adj` windowed every frame by Hann², widening the mainlobe from 4
to 6 bins.  The adjoint is not the inverse.  Measured on a 1.5 kHz tone pair
at Fs=456 kHz, NB=1024: plain Hann STFT valley −17.5 dB, state readout −4 dB —
the pair was effectively erased before any refinement began.

Fix: the direct seed is now the paused state OF THE RECORD ITSELF,
`prefix_rect(z)`.  Then `forward_long` reproduces the measured Hann spectra
exactly (verified 4e-14) and the short endpoints attach in-state through
`M_delta` exactly (verified 1e-13, the T3' identity, through the production
path).  The fusion is thus the gated product of two exact measurements
computed through one DIP-Zak state.

### 4b. Owned-band gate normalization caused the transient halo

The F1xT2 gates were normalized across each frame's OWNED fine rows only.  A
frame whose long window contains an event outside its owned band still has the
event in `|L|²`, and per-frame conservation forces that off-band energy into
its owned rows: a ±NB/2 halo (~26 dB in the event's band on the fixture).
A row-energy normalization fixes totals but not band shape (halo persists).

Fix (default, `norm="claim"`): normalize the gates over the frame's ENTIRE
delta range, each delta weighted by the long window's squared value at that
offset (its contribution to `|L|²`).  Off-band energy goes unclaimed by this
frame — the frame that owns it emits it.  Denominator shorts are measured
spectra (equal to the state-attached shorts to 1e-13).  A fixed per-row gain
restores the steady-tone convention exactly.  Halo: ~26 dB → ~0 dB.

### 4c. Measured on the known-truth SR fixture

`make_sr_fixture.py`: 1.5 kHz tone pair (long-aperture job), 60-sample click
train (short-aperture job), a crossing chirp, noise.  `sr_fusion_study.py`,
rows on the HS=32 lattice:

| config | click width (ms) | halo (dB) | tone dip (dB) |
|---|---:|---:|---:|
| STFT long N=1024 | 0.77 | +26.0 | −17.5 |
| STFT short N=128 | 0.07 | −0.4 | 0 (unresolved) |
| reassigned N=1024 | 0.07 | −1.2 | −20.8 |
| fusion frame-norm NB=1024 | 0.18 | +25.8 | −17.5 |
| fusion claim NB=1024/NS=128 | 0.07 | −0.6 | −16.8 |
| **fusion claim NB=2048/NS=128** | **0.07** | **−0.9** | **−38.0** |

The claim-gated fusion at NB=2048 reaches single-row time localization AND
the full long-window frequency separation simultaneously — joint resolution
neither single window nor reassignment provides — at ~55 ms per 160 rows.
Frame-norm conservation (`sum_e |F|² = |L|²`, verified 4e-8) remains available
as a toggle.

## 5. The three viewer modes

- **Streaming STFT:** center-timed complex spectrum; cheap and continuous.
- **Super-resolution (two-aperture fusion):** user-set long/short windows,
  claim-gated F1xT2 readout off one refined active-delta state.
- **Reassigned STFT:** phase derivatives relocate long-aperture energy to
  measured centroids; sparse/speckled by construction; kept as a separate
  observable.

The intrinsic-FCT viewer mode is archived (`viewer/archive/fct_view.py`).
The exact FCT transform, C ABI, and tests are unchanged; only the
endpoint-timed display product was set aside.

## 6. Open research

- Cross-frame constraints in the fusion state (the refinement currently has
  nothing to do for a known record; its value begins when constraints exceed
  measurements, e.g. denoising or partial data).
- A third aperture in the ladder (aperture_ladder measured +28 dB where blind
  two-window fusion fails) as an additional claim scale.
- Calibrate `gate_floor`/`beta` against a desired false-alarm probability on
  noise-only rows.
- For the archived FCT product: scale-aware phase bounds for coarse walk
  demand, two-sided (onset, offset) apertures, provider vectorization.
