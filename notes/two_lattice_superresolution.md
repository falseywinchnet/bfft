# Two-lattice super-resolution: algorithm, mathematics, measurements

Date: 2026-07-10.  Status: comprehensive record of the deployed algorithm,
its formal structure, measured properties, and the research program beyond it.

Companions: `viewer/two_lattice.py` (executable NumPy specification),
`viewer/dip_algo.cpp` (`MagnitudeProjectorC`, `iqw_dip_unified`),
`viewer/test_two_lattice.py` (C++ == Python to 2e-12),
`experiments/two_lattice_theory.py` (all measurements below),
`experiments/two_lattice_image.py` (2D prototype).

## 1. The deployed pipeline, exactly

Per 8192-sample tile (tiles overlap 50%, Hann OLA across tiles):

1. **Shared DIP seed** (`SolverC::run`, `shared_steps=1`): measured long and
   short windowed magnitudes; PGHI phases on the long lattice; record OLA
   seed; paused-DIP state via `prefix_rect`; ONE gradient step (scale
   2.5e-4) on the joint attached-state magnitude loss; rect consensus back
   to a waveform u0.
2. **One unified step** (`iqw_dip_unified`, `unified_steps=1`): momentum
   trial u + 0.88(u - u_prev) (inert on the first step), then
   `P_short(P_long(trial))`, then a 75%-relaxed final long projection
   `u += 0.75 (P_long(u) - u)` — the symmetric finish that protects weak
   chirp terminals.
3. **Global phase**: one rotation aligning the latent to the observation so
   neighboring tiles OLA coherently.  Nothing else uses observed phase.
4. **Readout**: reassigned STFT of the latent at the long aperture,
   bilinear 2x2 splat, evaluated directly at the display centers.

Geometry (one knob): long window `NB`; `NS = NB/4`; `h_long = NB/8`;
`h_short = NS/2 = NB/8`.  Windows are symmetric Hann (`np.hanning`).

One family's projector `P_i` is: extract every complete frame on the
family's lattice, windowed FFT, replace each coefficient's magnitude by the
measured target keeping its phase (radial projection), inverse FFT,
window-weighted overlap-add divided by the accumulated window-squared
(with the previous iterate carried where the Hann has no support).

## 2. Formal structure

**Objects.** Let `A_i` be the analysis operator of family i (windowed STFT
on lattice i), `m_i = |A_i z|` the measured magnitudes.  The constraint
sets are

    M_i = { u : |A_i u| = m_i },

nonconvex real-analytic varieties in C^L.  The projector implemented is

    P_i(u) = A_i^+ ( m_i * A_i u / |A_i u| ),

i.e. the coefficient-space radial projection followed by the least-squares
synthesis (`A_i^+` = win^2-weighted OLA, exact on COLA-supported samples).
This is the classical Griffin–Lim projection pair, expressed in waveform
space; it is a surrogate for (not equal to) the metric projection onto M_i.

**Iteration.**  Heavy-ball-accelerated alternating projection seeking
`M_long ∩ M_short`.  Since the targets are measured from the observation z,
**z itself is a member of the intersection and an exact fixed point**
(measured: seeding at z gives residual 0.0, corr 1.0).  The deployed
one-iteration algorithm is therefore a *local polisher of the DIP seed
toward the observation-consistent intersection*, not a global phase
retriever.

**Gauge structure (measured, `exp_b`).**  Which phase directions do the two
families pin?  Rotating one component of a tone-pair + chirp + click
fixture by 90 degrees and measuring family residuals:

| rotated component | long resid | short resid |
|---|---:|---:|
| global phase | 0.00000 | 0.00000 |
| isolated click | 0.008 | 0.002 |
| crossing chirp | 0.048 | 0.084 |
| one tone of a 1.4-bin pair | 0.264 | **0.522** |

- Global phase is the exact gauge (chosen once at the end).
- The absolute phase of a time-frequency–disjoint component is a *near
  gauge*: invisible to both families — and equally invisible to any
  magnitude display, so it is benign for the viewer.
- **The close pair's relative phase is pinned, and the SHORT family pins it
  twice as hard as the long.**  This is the formal heart of the method: in
  a short frame both tones land in one bin, so the interference term
  `2 Re(a conj(b) W)` — the beat — is fully visible in the magnitude; the
  long window separates the pair into different bins and suppresses the
  interference.  Conversely the long family measures that there *are* two
  tones.  Fine frequency (long magnitudes) and beat/event timing (short
  magnitudes) are complementary constraints on the same latent degrees of
  freedom.  That complementarity, not any image-space blending, is the
  super-resolution.

**Convergence (measured, `exp_a`).**
- Cold random start stagnates: asymptotic residual ratio ~0.996–0.998 per
  iteration, corr-to-truth <= 0.7 after 96 iterations — the well-known
  Griffin–Lim stagnation regime.  The viewer never operates here.
- Near-seed regime converges cleanly: from a jitter-0.5 phase-degraded
  seed, fidelity 0.81 -> 0.93 after ONE iteration -> 0.99 at 24.
- Momentum beta=0.88 measurably helps in both regimes.
- Hence the DIP/PGHI seed is the load-bearing component: it buys the basin;
  the projections buy the joint consistency.

**What iterations actually push (measured, `exp_c`) — honest statement.**
Iterations push fidelity toward the *observation*, by removing seed error.
When the targets are measured from a noisy record, the noisy record is in
the intersection, so magnitude consistency cannot denoise past the input
SNR (measured: fidelity-to-clean approaches 0.92 = the observation's own
fidelity, from below).  "Multiple iterations push SNR" is true relative to
the seed, capped at the observation.  True SNR gain beyond the observation
requires processing the *targets* (see research program).

## 3. Intuition

One latent waveform is photographed by two instruments that blur different
axes.  Each projection re-exposes the current guess through one instrument
and corrects exactly what that instrument can see, leaving everything else
(including the phase it cannot see) untouched.  Alternating exposures force
the guess into the set of waveforms both instruments would have measured
identically — triangulation in time-frequency.  Phase is the courier: the
long window's phases encode fine timing it cannot show in magnitude; the
short window's magnitudes (beat envelopes) pin relative phases the long
window cannot see.  The reassignment readout then displays the recovered
phase structure as sharpened time/frequency centroids.

## 4. 2D: the image-fusion foundation (measured)

`experiments/two_lattice_image.py` runs the identical algebra on a 2D
scene: two windowed-DFT magnitude families (48x48 patches, hop 12 = fine
spatial frequency; 12x12 patches, hop 4 = fine localization), radial
projection + win^2 OLA, momentum, **plus positivity** — scenes are
nonnegative, and clipping at zero kills the local *sign* gauge that is the
real-image analog of the per-component phase gauge (without it, recovered
points/lines come out polarity-flipped inside per-patch sign domains).

Scene: a two-frequency grating pair resolvable only by the large window,
plus point sources and a thin line localizable only by the small window.
From a phase-scrambled seed, 80 iterations:

| recovery | PSNR (dB) |
|---|---:|
| seed | 21.6 |
| large window only | 21.1 (grating warped, ghosted points) |
| small window only | 23.3 (pair unresolved, patch domains) |
| **alternating** | **28.5 (both structures clean)** |

This is the skeleton of fusing "two captures with different properties of
one scene": each capture contributes one constraint family on a shared
latent.  A real-capture demo needs (a) two registered images, (b) a
forward model per capture (PSF/exposure/noise) so each family constrains
what that capture measured reliably, (c) reliability weights per
coefficient.  Candidate pairs: short-exposure sharp-noisy + long-exposure
clean-blurred; two different PSFs; panchromatic + low-res multispectral.

## 4b. Relation to the OT-barycenter lineage (arXiv 2604.15055)

Valdivia, Cazelles & Fevotte, "Enhancing time-frequency resolution with
optimal transport and barycentric fusion of multiple spectrograms" (IRIT,
v1 2026-04-16) is where this work branched.  Their method: treat the long-
and short-window POWER SPECTROGRAMS as nonnegative measures, compute their
unbalanced-OT barycenter (lambda=1/2) on a target grid F_long x T_short,
with structured cost matrices that only let long-window energy move along
time and short-window energy move along frequency (+inf elsewhere), solved
by a block-MM algorithm without entropic regularization.  Runtimes: 0.43 s
(single packet), 3.78 s (packet mixtures), 9.36 s (5 s speech at 8 kHz on a
5 ms x 8 Hz grid); 57-945 MM iterations; same-grid variants 53-149 s.

The essential differences:

1. **Averaging pictures vs intersecting evidence.**  Their fused object is
   a barycenter — a geometric compromise BETWEEN two measures, living in
   image space; ours is (an iterate toward) the INTERSECTION of two
   constraint sets in waveform space.  When the two inputs describe the
   same signal, the intersection contains the true signal exactly (our
   measured fixed point); a barycenter is a new picture that is generally
   not the spectrogram of any signal.
2. **Where the joint t-f information comes from.**  Their fine-time x
   fine-frequency coupling is supplied by the transport prior (energy
   moves to the nearest agreement).  Ours is measured: the short family's
   magnitudes contain the interference (beat) terms of components the long
   window separates — the gauge probe showed the short family pins the
   close pair's relative phase 2x harder than the long.  Phase is the
   courier; no geometric prior fills the gap.
3. **Output type.**  Theirs is a display-only measure (to get audio their
   own earlier work appends Griffin-Lim — i.e. a single-family version of
   our step).  Ours IS a waveform: any readout (reassignment, arbitrary
   windows, demodulation, further iterations) applies afterward.
4. **Computation.**  Their cost is an iterative optimization over sparse
   transport plans (1e4-1e7 finite entries x 1e2-1e3 iterations) on a
   downsampled grid, batch over the whole signal.  Ours is a few windowed
   FFT passes per tile ((L/HB + L/HS) frames per projection), one unified
   step from the DIP seed, streaming with tile OLA and warm start —
   ~real-time on full-rate 456 kHz IQ in the viewer.  Architecturally
   O(L log L) per step vs plan-sized MM.
5. **What theirs does that ours cannot.**  It needs no phase, no common
   latent, and no forward operator — it fuses ANY two nonnegative images,
   even mutually inconsistent ones, with graceful barycentric compromise
   and per-block monotone convergence.  Ours requires the families to be
   magnitudes of one latent under known operators, and its convergence is
   local (seed-basin).  Their unbalanced marginals suggest our missing
   analog: relaxed/weighted projections with denoised targets when the
   families are inconsistent (research items 2 and 5 below).
6. **Domain freedom.**  Because our latent is the signal itself, the
   constraint families can act in any linearly-reachable domain — e.g.
   fuse two images through their 2D FFTs (measure families in the Fourier
   domain, invert at the end), or mix a time-domain family with a
   Fourier-domain one.  Caveat: unwindowed global-FFT magnitude alone
   sheds localization (classic phase retrieval); patched/windowed families
   or side constraints (positivity, support) restore the grip — exactly
   what the 2D prototype uses.

## 4c. Relaxed/weighted projections for INCONSISTENT families (first working form)

`experiments/two_lattice_mff.py`, run on MFFW3/4 (three differently-focused
captures of one scene, github shuangxu96/MFF-SSIM).  A focus stack is the
inconsistent-family setting: each capture measures the latent scene
correctly only where it is in focus.  The audio theory's hard projection is
wrong there; the generalization is the **relaxed radial projection**

    P_i^w(u):  S = A_i u;   S <- (1 - w_i) . S + w_i . m_i . S/|S|,

equivalently `u <- u + A_i^+ (w_i . radial residual)`: a per-coefficient
convex relaxation with reliability weights w_i in [0,1].  w=1 recovers the
consistent theory's hard constraint; w=0 ignores the family.  This is the
operator-space analog of unbalanced-OT marginal relaxation (their eta):
fixed points settle at w-weighted compromises wherever families disagree.

Reliability is measured, not assumed: defocus is a local low-pass, so
per-patch high-band energy identifies the in-focus capture.  Weights =
softmax (gamma=2) of high-band patch energy across captures; families =
{3 captures} x {32x32 Hann patches, hop 8}; positivity; RGB channels share
luminance weights; seed = reliability-weighted pixel average.

Measured (Laplacian variance; sources 0.0018-0.0040):

| image | lapvar |
|---|---:|
| best source | 0.0040 |
| pixel average | 0.0010 |
| argmax-Laplacian (hard classic) | 0.0082 |
| weighted-average seed (iters=0) | 0.0097 |
| **weighted projections (16 iters)** | **0.0101** |

Zoom inspection: fused shell-center matches the sharpest source there,
fused rim matches the other source, no ringing or ghost seams despite
visible focus-breathing misregistration between captures.  Honest split of
credit: the reliability WEIGHTING contributes most of the gain; the
projections add ~3.5% lapvar (converged by ~4 iterations).

## 4d. The complementarity law (controlled measurements with ground truth)

`experiments/two_lattice_blur_study.py`: ground truth = the sharp shell
crop; two captures through known blurs + noise; PSNR to truth.  These
measurements replace the two speculations at the end of 4c.

**Measured law: the projections' gain over the best capture tracks the
SPECTRAL COMPLEMENTARITY of the captures.**

| capture pair | best capture | pixel avg / seed | fused (max-consensus) |
|---|---:|---:|---:|
| identical disk r=4 (control) | 24.39 | 24.39 | 24.39 — exactly no hallucination |
| disk r=3 vs r=5 (weak compl.) | 26.15 | 24.39 | 25.85 — beats seed +1.5 dB, NOT best capture |
| h-box vs v-box 11px (strong compl.) | 24.74 | 24.50 | **26.30 — beats BOTH captures** |

- Orthogonal motion blurs: each capture perfectly preserves what the other
  destroys; the fusion recovers structure NEITHER capture contains (the
  pit's true shape, band edges in both orientations).  Projections alone:
  seed 24.50 -> 26.35 by iteration 4.  Residual cross-hatch remains where
  BOTH blurs kill the same band (the high-fx AND high-fy corner) — phase
  reconciliation there is the visible frontier (a Douglas-Rachford
  candidate).
- Disk pair: different radii share most of their attenuation; the extra
  coverage is sidelobe crumbs, so fusion lands between average and best
  capture.  Complementarity, not multiplicity, is what pays.
- Weights ladder under spectral complementarity orders as theory demands:
  per-patch 25.83 < per-coefficient 25.94 < max-consensus 26.30 —
  reliability must be resolved at the frequency scale where the captures
  actually differ.

**Honest negative (seam regime):** with spatially varying focus and a wide
ambiguous transition band, the projection stage adds nothing over the
weighted-average seed (±0.05 dB at every gamma; seam-local RMSE unchanged).
The earlier 4c speculation that projection value concentrates at soft-weight
seams is RETRACTED; the MFF lapvar bump was contrast, not fidelity.  Where
weights can decide, weighting is sufficient; where they cannot, magnitudes
carry no additional verdict either.

Audio corollary: "spectral complementarity" is the image-domain face of the
transversality angle between constraint manifolds (research item 1) — the
aperture-pair design rule and the capture-pair design rule are the same
mathematics.

## 4e. The cross-hatch is information-limited (DR/RAAR null result)

`experiments/two_lattice_dr.py`.  Hypothesis tested: the cross-hatch
residual of the orthogonal-blur fusion is AP stagnation, escapable by
Douglas-Rachford / RAAR reflections.  Measured (same budget, same seed):

| operator | PSNR | dead-band err | live-band err |
|---|---:|---:|---:|
| AP + positivity (baseline) | 26.30 | 1.0413 | 0.0017 |
| DR (100 it, best iterate) | 26.37 | 1.05 | 0.0016 |
| RAAR beta=0.75/0.9 (best) | 26.37 | 1.05 | 0.0016 |
| AP + absolute reliability gate | 26.28-26.30 | 1.04 | 0.0017 |

Null across the board.  The diagnosis is in the dead-band number: with
tau=0.3, the band where BOTH box OTFs are small is 74.5% of the frequency
plane, and its relative error is ~1.04 = 1.00 (content entirely missing:
neither capture measured it) + ~0.04 (small Rayleigh-max bias from taking
the max of two noise-level magnitudes).  The live band is essentially
solved (0.0017).  There is no optimization trap to escape and no
enforcement garbage worth gating: the hatch is the structured ABSENCE of an
unmeasured band.  No splitting operator can create missing information.

The absolute gate (enforcement weight t^2/(t^2+(kappa*floor)^2), floor
estimated from the outer-frequency shell) remains the conceptually correct
completion of the weighted theory — softmax weights are relative (who to
trust), the gate is absolute (whether to trust anyone) — but at these noise
levels the bias it removes is invisible.

**Constructive resolution (measured): add measurement, not machinery.**  A
third capture with 45-degree motion blur covers part of the corner both
others kill: original-dead-band error 1.041 -> 0.813, PSNR 26.30 -> 26.41,
with the identical AP operator.  Only new information moved the number that
operators could not.  Angular blur diversity samples the frequency plane
like tomographic projections; the capture-design rule is coverage of the
joint dead zone, and the audio analog is the third aperture of the ladder
(research item 4).

## 5. NEXT PROJECT — dead-zone coverage: the aperture ladder across domains

**The unifying law, now measured in both domains:** super-resolution
quality is set by how completely the family set COVERS the separation/
frequency plane; operators (AP, DR, RAAR, gates) only polish toward what
is covered (§4e: all operators identical; §4d: only complementarity pays).
A residual that ignores better operators is a coverage map of the
apertures' joint blindness.

Audio foundation (`two_lattice_theory.py::exp_e_ladder`, packet width 200):
a window N pins a packet pair's relative phase iff `dt + w <~ N` and
`df - 1/w <~ 1/N`.  Measured against the deployed {1024, 256} ladder:

| pair (dt, df) | 2048 | 1024 | 512 | 256 | 128 |
|---|---:|---:|---:|---:|---:|
| covered (400, 1/NB) | .48 | .29 | .06 | 0 | 0 |
| DEAD slow (1600, 1/NB) | **.021** | 0 | 0 | 0 | 0 |
| DEAD fast (40, 20/NB) | .014 | .015 | .020 | .022 | **.138** |
| interior, ratio 4 (400, 7/NB) | .28 | .18 | .04 | 0 | 0 |
| unpinnable (1600, 12/NB) | .01 | 0 | 0 | 0 | 0 |

(0 = gauge floor ~0.0008.)  Three measured facts: the deployed pair is
blind to slow structure (only a LONGER rung pins it) and to fast/wide
structure (only a SHORTER rung); adjacent rungs at scale ratio <= ~4 leave
no interior hole; and dt*df >> 1 is invisible to every window — the
uncertainty boundary no rung can cross.  Image analog already measured:
third blur angle cut the dead band and beat every operator upgrade (§4e).
The audio "capture angle" is the window scale; ladders must extend BOTH
ways, and the upward rungs (long windows, few frames) are computationally
cheap.

### Task series

STFT (controlled theory):
1. Full coverage map: residual heatmaps over a (dt, df) grid for ladder
   sets {NB,NS}, {2NB,NB,NS}, {NB,mid,NS}; verify the visibility law and
   the ratio-4 no-hole claim quantitatively; packet-width dependence.
2. Transversality vs ladder: measure the local AP convergence rate with 2
   vs 3 rungs — does a rung raise the manifold angle (fewer iterations),
   or only extend coverage?  Separates the two currencies.
3. Dead-zone recovery: fixtures with slow pairs (dt > NB); 2-rung vs
   3-rung recovery fidelity — the audio analog of the +1.6 dB image gain.

Waterfall (deployed engineering):
4. Extend `iqw_dip_unified` to P families (a loop; +L/H FFTs per rung per
   tile).  Rung-set knob in the viewer: {4NB or 2NB, NB, NS} — the slow
   dead zone argues the third rung goes UP, and long windows are cheap
   (few frames, big FFTs -> bfft/SIMD track).
5. Dead-zone fixture in the viewer: AM pairs / click trains with beat
   periods near 2NB; verify the readout shows the beat only with the
   third rung; measure the realtime budget at NB=4096 with 3 rungs.
6. Ladder-aware readout: reassignment at the rung that owns each cell's
   scale (coarse rungs own slow structure).

Images:
7. The tomographic curve: PSNR and dead-band fraction vs number of blur
   angles (2..8) at fixed budget; angle ladder x patch-scale ladder cross
   product.
8. Reinterpret the MFFW focus stack as a focal ladder: estimate each
   source's OTF-proxy coverage, compute the joint dead zone, and test
   whether a 3rd focus slice fills measured holes (it is capture angle in
   depth).
9. Fourier-domain families for registered capture pairs (patched, with
   positivity) — the domain-freedom demo.

Theory:
10. Ladder theorem candidate: a geometric ladder with ratio r covers the
    pinnable plane within its scale span with no interior holes iff
    r <= r*(w); connect the coverage function to the transversality angle
    (they should be the same object) and to frame/wavelet theory.

### First results (tasks 1-3, `experiments/aperture_ladder_coverage.py`)

**Task 1 — coverage is real, upward, and graded.**  Over a (dt, df) grid of
packet pairs (w=200), with pinnable = any rung 256..2048 above threshold:

- deployed ladder {1024, 256}: covers **53.2%** of the pinnable set;
- add the 2048 rung: **100%**;
- add an interior 512 rung instead: **53.2% — exactly nothing**, the
  ratio-4 no-hole claim confirmed as redundancy of interior rungs;
- the deployed ladder's entire hole is the upward band dt in (800, 1500).

The heatmaps (`out/ladder_coverage.png`) show each rung as a rectangle
dt <~ N - w whose pin strength FADES toward the cell edge: coverage is not
binary, and the maps are the local-rate landscape.

**Task 3 — the strength->rate law (the project's central dynamical fact).**
Slow pair dt=1600, started on its gauge orbit (packet B rotated), AP with
each ladder; residual rotation vs iterations:

| ladder | pin strength | 43 deg start -> it20 -> it300 | retained |
|---|---:|---|---:|
| {1024,256} | 0.000 | 43.1 -> 43.1 | 100% (blind forever) |
| {2048,1024,256} | 0.016 (cell edge) | 41.0 -> 13.1 | 36% (crawling) |
| {4096,1024,256} | 0.286 (deep) | **0.8 -> 0.0** | 0% (done by it20) |

Bare coverage says whether recovery is possible; pin strength (= local
transversality) sets the iteration budget, roughly rate ~ strength^2.
**Ladder design rule: place expected content DEEP in some rung's cell —
rung length ~2-2.5x the content separation.  Cell-edge coverage is
coverage-in-principle only.**  Engineering bonus: upward rungs are nearly
free (a 4096 rung is 9 frames per 8192 tile).

**Task 2 — honest partial.**  Constraint violation contracts fast (~7x in
early iterations) then floors at the OLA-surrogate's fixed-point mismatch
(~1e-4); tail-rate fitting measures the floor, not the angle.  The proper
instrument is the linearized composed-projector spectrum (power iteration)
— queued.  Task 3's strength->rate law already supplies the design-relevant
version of the answer: a rung both extends coverage AND raises the local
angle for content deep in its cell.

### Task 7 — the tomographic angle curve (`experiments/two_lattice_tomo.py`)

K captures through 11-px line blurs at angles i*180/K, max-consensus AP
fusion (the operator family already at the information limit), PSNR vs K:

| K | joint dead fraction | fused (fixed per-capture noise) | fused (fixed TOTAL budget) | best capture |
|--:|--:|--:|--:|--:|
| 1 | 0.855 | 25.26 | 25.26 | 25.26 |
| 2 | 0.732 | 26.94 | 26.93 | 25.26 |
| 3 | 0.630 | 29.10 | 29.04 | 25.26 |
| 4 | 0.493 | 30.55 | 30.41 | 25.25 |
| 6 | 0.382 | **32.00** | **31.63** | 25.23 |

Two conclusions.  (a) Fusion gain tracks the dead-band fraction almost
linearly (+6.7 dB for a 0.47 coverage gain) — the coverage function is a
quantitative predictor of fusion value, not just a qualitative one.
(b) The fixed-TOTAL-budget column is the tomography result: splitting one
exposure across six angles beats concentrating it in one by +6.4 dB —
**angular diversity is worth far more than per-capture SNR.**  No
diminishing returns yet at K=6 (the joint dead zone is an intersection of
wedges and keeps shrinking).  This is the image-domain quantitative form
of the ladder design rule.

### Deployed (tasks 4-6): the ladder in the viewer

- `iqw_dip_unified_ladder` (C ABI) + `iqwaterfall.dip_unified_ladder` +
  `two_lattice.recover_ladder` (executable spec): P magnitude families in
  application order, final relax on rung 0, same shared fast1 seed.  C++ ==
  Python to 2e-12; the pair ABI is verified as a special case
  (`test_two_lattice.py::test_cpp_ladder_kernel_matches_python_specification`).
- Viewer: "slow rung" combo (off / 2x / 4x) in the super-res panel;
  `TwoLatticeStream(rung_mult=...)` caps the rung at the tile length.
- **Budget (measured)**: the upward rung costs +0.5 ms/tile at NB=1024
  (15.8 -> 16.3 ms) and +0.06 ms at NB=4096 — effectively free.
- **In-hole recovery through the production spec (measured)**: packet pair
  at dt=1200 (inside the deployed hole, deep in the 2048 cell), started
  with a 70-degree gauge error: {1024,256} retains 70.0 deg after 100
  iterations; {2048,1024,256} recovers to **0.0 deg**.
- **Two honest nulls that sharpen the scope.**  (a) Steady sub-bin tone
  pairs are NOT dead-zone content: every long frame contains both tones,
  so the deployed pair already pins them (seed lands at 1.7 deg; both
  solvers hold it).  The hole is specifically TRANSIENT pairs at
  dt in (800, 1500).  (b) The NB-scale reassignment readout cannot display
  what only >NB windows see (its frames never contain both packets), so
  the rung's gain lives in the LATENT — waveform-domain uses (demod,
  re-analysis) and the future ladder-aware readout (task 6b), not the
  current display.  The rung is on by choice, not default, until the
  readout learns to consult it.

## 6. Background research items

Math:
1. **Transversality rate law.**  Local AP convergence rate = cos of the
   angle between the two constraint manifolds at z.  Measure it by power
   iteration on the linearized composed projector; derive the aperture
   design rule (which NS, hops maximize the angle for given content).
2. **Target-domain denoising.**  Shrink/average the measured magnitudes
   before projecting (the intersection then no longer contains the noisy
   observation) — the only route to SNR *beyond* the input.  Iterations
   then genuinely denoise; quantify vs sigma.
3. **Better splitting operators.**  Douglas–Rachford / RAAR instead of
   plain AP: standard phase-retrieval upgrades known to escape GL
   stagnation; drop-in at the same cost per iteration.  Worth measuring in
   the near-seed regime too (fewer iterations for the same polish).
4. **Third aperture.**  The aperture-ladder work measured +28 dB from a
   3-aperture scheme where blind 2-window fusion failed; the unified loop
   extends to three families at linear cost.
5. **Gauge accounting.**  Per-component phase gauges are invisible to
   magnitude displays but matter for waveform-domain uses (demod after
   super-resolution): enumerate them (T-F-disjoint components) and pin
   them with a third constraint if needed.

Engineering (SIMD/large-NB track):
6. Per unified step and tile: (L/HB + L/HS) windowed FFTs each way.  At
   NB=4096: 15 frames of 4096 + 127 of 1024 per projection — the FFTs are
   the whole cost.  Route `fft_pow2` to bfft kernels; vectorize the radial
   update (pure elementwise); frames are embarrassingly parallel; the
   momentum/OLA passes are memory-bound axpy.
7. PGHI seed at large NB: seed quality controls everything in the
   one-iteration regime; profile seed error vs NB and consider seeding the
   long phases from the previous tile (warm chain already exists).

## 7. Test inventory

- `viewer/test_two_lattice.py`: lattice geometry, AP convergence on a
  smooth fixture, real-signal path, C++ unified == Python spec (2e-12),
  bilinear reassignment power conservation.
- `experiments/two_lattice_theory.py`: convergence/momentum, gauge probe,
  seed-vs-iterations fidelity, why-two-lattices (run it; prints the tables
  quoted above).
- `experiments/two_lattice_image.py`: 2D prototype and PSNR table;
  writes `experiments/out/two_lattice_image.png`.
