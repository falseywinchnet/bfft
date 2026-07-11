# Waterfall math audit: STFT, reassignment, and intrinsic FCT

Date: 2026-07-10

This note separates the three viewer measurements and records the corrections
that were necessary when the STFT experiments moved into a streaming complex-IQ
waterfall.

## 0. Current correction: shared seed, unified projection

Sections 4a–4f below preserve the audit of the former magnitude-claim viewer,
but that construction is no longer the live super-resolution transform. It
multiplied a long spectrum by short-spectrum row weights. The original Python
operator instead solves for one latent waveform from two independent magnitude
families and then reads a reassigned long-window spectrum on the short lattice.

The live reference now composes the two useful state-domain results:

1. `active_delta_center5_fast1` supplies the smooth **shared** seed: PGHI record
   seed, one shared L5 long/attached-short step, rectangular consensus.
2. One literal `P_short(P_long(u))` alternating projection supplies the
   **unified** correction on independent long and short frame lattices.
3. Reassignment is read from that latent waveform. No row claim, gate,
   per-frame conservation, or image-space blend is involved.

The combined result is close to the shared raster while moving energy in the
unified direction. On the checked IQ tile, amplitude-image cosine between
shared and combined is 0.981/0.989/0.993 for N=1024/2048/4096; a random-seeded
one-projection unified solve gives only 0.467/0.555/0.759 and produces the
reported coral texture. This is why iteration count alone was the wrong knob:
the coherent seed dominates the first visible result.

Viewer geometry is fixed to long `N`, short `N/4`, and an internal short-COLA
hop `N/8`. Display readout decimates this by two to `N/4`, so SR N aligns with
conventional STFT N/2 on the same schedule (4096/1024 SR aligns with 2048/1024
STFT). Streaming STFT and reassigned STFT use the same `N/2` Hann COLA grid.
Changing those derived hops never mutates the position marker; completed SR
tiles also remain cached across mode toggles.

SR reassignment is evaluated directly at the external `N/4` centers. The
earlier implementation rendered on `N/8` and selected even raster rows, which
discarded energy reassigned to odd rows and manufactured holes. Continuous
time/frequency destinations are now bilinearly splatted to their nearest 2×2
cells. The four weights sum to one, so this fills only subpixel quantization
gaps and preserves total power exactly; there is no baseline or floor.

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

## 4. Historical audit of the rejected magnitude-claim waterfall

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
path).  The fusion is thus a magnitude-partition product of two exact measurements
computed through one DIP-Zak state.

### 4b. Owned-band gate normalization caused the transient halo

The F1xT2 gates were normalized across each frame's OWNED fine rows only.  A
frame whose long window contains an event outside its owned band still has the
event in `|L|²`, and per-frame conservation forces that off-band energy into
its owned rows: a ±NB/2 halo (~26 dB in the event's band on the fixture).
A row-energy normalization fixes totals but not band shape (halo persists).

Fix: normalize each owned short power by measured short power over the frame's
entire delta range. Off-band energy goes unclaimed by this frame — the frame
that owns it emits it. Denominator shorts are measured spectra (equal to the
state-attached shorts to 1e-13). The later long-Hann claim weighting and
per-row restoration gain were removed together: at NB=1024/NS=512 those gains
spanned 4.7 dB and AM broke their stationary-tone cancellation, producing a
periodic row/vertical astigmatism. The live claim is now simply
`P_owned/sum_delta(P)`: no floor, exponent, weighting, gain, or conservation
toggle. Halo remains approximately 0 dB.

### 4c. Cross-frame support and center timing

The attachment identity only requires a short interval to be contained in
*some* overlapping long state. The former viewer required it to be contained
in its display owner. At NB=1024/NS=512 the desired local deltas are 14..17,
while valid crops are 0..16. The final global short interval is exactly delta
13 of the next long state:

    j*4 + 17 = (j+1)*4 + 13.

The router now applies this identity generally and reserves trailing parent
states at tile boundaries. The only remaining aperture condition is the DIP
attachment margin `NS <= NB-128`; two independent transform streams are not
needed.

Fusion output indices denote short-window starts. Waterfall axes denote
centers. The missing `NS/2` term displaced NB=1024/NS=512 by 256 samples and
made AM structure look vertically astigmatic. Display lookup is now
`position/HS - lo - NS/(2*HS)`.

### 4d. The remaining horizontal texture

At projection step zero the paused state is the known record, and it already
satisfies both measured apertures. Step one is therefore identical: alpha and
iteration controls cannot improve this waterfall. Tests at the reported IQ
position also show that floor, exponent, and frame/row normalization do not
remove the horizontal texture. It is inherent to the magnitude-partition
readout: a high-variance short-window time sequence modulates a whole band of
the narrow long spectrum. NB=4096/NS=64 makes this easiest to see. The live UI
no longer exposes those non-causal tuning controls; phase relocation is the
path that avoids this construction.

### 4e. FCT-phase-guided reassignment

The archived endpoint-scatter FCT image was not restored. Instead Hann-STFT
energy is reassigned using two phase jets. Ordinary Hann derivatives establish
the local branch. At FCT's exact selected support,

    t_F = origin + Re(M/C),
    M = sum_{n<tau} n*x[n]*exp(-i*omega*n),

and a one-sample shift at that same fixed support gives frequency without
comparing two adaptive supports:

    C1 = exp(i*omega) * (C - x[0] + x[tau]*exp(-i*omega*tau)),
    omega_F = arg(C1*conj(C)).

An FCT coordinate replaces the ordinary coordinate only inside its local
phase basin (one dense time cell / 1.5 frequency bins); remote multimodal
branches retain ordinary reassignment. Energy is accumulated on a `2N`
half-bin frequency raster. There is no magnitude/coherence gate. Sparse and
pending `-120 dB` cells are excluded from display auto-ranging, preventing the
former solid-yellow view.

### 4f. Measured on the known-truth SR fixture

`make_sr_fixture.py`: 1.5 kHz tone pair (long-aperture job), 60-sample click
train (short-aperture job), a crossing chirp, noise.  `sr_fusion_study.py`,
rows on the HS=32 lattice:

| config | click width (ms) | halo (dB) | tone dip (dB) |
|---|---:|---:|---:|
| STFT long N=1024 | 0.77 | +26.0 | −17.5 |
| STFT short N=128 | 0.07 | −0.4 | 0 (unresolved) |
| reassigned N=1024 | 0.07 | −1.2 | −20.8 |
| FCT-guided reassignment N=1024, 2N raster | 0.07 | −1.15 | −20.1 |
| fusion frame-norm NB=1024 | 0.18 | +25.8 | −17.5 |
| fusion claim NB=1024/NS=128 | 0.07 | −0.6 | −16.8 |
| **fusion claim NB=2048/NS=128** | **0.07** | **−0.9** | **−38.0** |

The claim fusion at NB=2048 reaches single-row time localization AND
the full long-window frequency separation simultaneously — joint resolution
neither single window nor reassignment provides — at ~55 ms per 160 rows.
Frame-norm conservation remains a research diagnostic, not a viewer control.

## 5. The three viewer modes

- **Streaming STFT:** center-timed complex spectrum; cheap and continuous.
- **Super-resolution (two-aperture):** shared one-step DIP seed, one independent
  long/short waveform projection, then reassigned readout; N/N/4/N/8 geometry.
- **Reassigned STFT:** ordinary local Hann reassignment on the streaming COLA
  time grid.

The intrinsic-FCT viewer mode is archived (`viewer/archive/fct_view.py`).
The exact FCT transform remains shipped. Its C/Python API now also exposes the
selected-support first moment used by the hybrid; the endpoint-timed display
product remains archived.

## 6. Open research

- Port the single NumPy independent-family projection to C++/SIMD; the shared
  seed is already the validated dynamic C++ port of the Python fast1 baseline.
- Compare shared-only, combined, and unified-only quanta without display-domain
  blending; preserve smoothness through the latent waveform/state.
- A third aperture in the ladder (aperture_ladder measured +28 dB where blind
  two-window fusion fails) as an additional claim scale.
- For the archived FCT product: scale-aware phase bounds for coarse walk
  demand, two-sided (onset, offset) apertures, provider vectorization.
