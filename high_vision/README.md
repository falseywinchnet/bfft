# High Vision

High Vision is the temporal-imaging framework beside the BFFT OBS plugin. It
turns a stream of ordinary frames into a moving belief about scene radiance.
The first working mode is synthetic HDR; the same transport owns the landing
point for night/astronomy and inverse-diffusion experiments.

This is deliberately not a single-image detail hallucination system. Synthetic
HDR can recover information only when it was measured in at least one frame:
camera auto-exposure, gain changes, small motion, and sensor noise provide
different observations. A region clipped in every frame contains no recoverable
radiometric value.

## What exists now

- A dependency-free C++17 engine in
  `include/high_vision/high_vision.hpp`.
- A second OBS asynchronous filter named **BFFT High Vision**. It ships in the
  existing `bfft-cartoon` plugin module, so the OBS build and installation
  procedure is unchanged.
- Scene-linear exposure normalization from camera telemetry when available.
  Without telemetry, the fallback changes gauge only for a coherent,
  registered global multiplicative step; low-light frame ratios are not
  allowed to random-walk the accumulated radiance.
- Exposure-invariant census registration: one global camera translation plus a
  smoothly interpolated grid of local translations. Synthetic HDR can use the
  local field; Night conservatively transports its long-lived radiance in the
  global camera gauge so noisy tile flow cannot recursively turn the belief
  into an elastic surface.
- Night registration is preconditioned by the in-house allocation-free
  TGFD/Meyer split when High Vision is built inside BFFT. Census matching uses
  the cartoon carriers of the incoming frame and accumulated radiance belief,
  so sensor-fixed oscillation does not force the zero-motion solution. A
  dependency-free pooled witness remains available to standalone builds.
- The reliability law is physically composed in variance:
  `sigma² = read_noise² + shot_noise² * radiance`. Read and shot standard
  deviations are no longer added, which had introduced a false cross-term and
  made old shadow support too difficult to release.
- A second, zero-mean nuisance belief remains fixed in detector coordinates
  while scene radiance moves through the camera gauge. Registered camera motion
  provides the diversity needed to estimate fixed-pattern noise; a stationary
  scene does not falsely claim that the two fields are identifiable.
- Night's OBS YUV adapter owns a separate transported chroma belief. Chroma
  earns saturation from its own evidence and its lifetime is capped by luma
  support, preventing current-frame green/purple noise from being painted onto
  accumulated luminance. A slow dark-population estimate corrects global U/V
  black bias before fusion.
- Organic support replacement. A leaky signed innovation fusor distinguishes
  temporally coherent scene change from zero-mean sensor noise. Its scale is
  relative to local signal and to the uncertainty of the accumulated mean, so
  a dark occluder can rapidly bankrupt stale bright support without forfeiting
  stable shadow denoising.
- Highlight-safe evidence fusion. Near-black and near-clipped samples have low
  precision; a fully clipped sample cannot overwrite an earlier usable
  highlight estimate.
- Percentile AGC and a bounded logarithmic display transform. These affect only
  display output, never the stored linear belief.
- A long-persistence night integrator baseline and an `ExperimentalStage`
  interface for future inverse-diffusion estimators.

## Data flow

```text
OBS/camera frame
    -> transfer decode
    -> TGFD/Meyer carrier witness (Night in a BFFT build)
    -> global + local registration
    -> detector-fixed nuisance update
    -> conservative transport (belief, support, variance, signed innovation)
    -> exposure normalization
    -> reliability/change-weighted fusion
    -> independent chroma evidence fusion (OBS YUV Night adapter)
    -> optional ExperimentalStage
    -> display AGC + tone map
    -> OBS frame
```

The separation matters: an experimental night estimator receives the current
radiance observation, transported support, and mutable belief, but does not
have to implement capture, motion, reset behavior, or display.

## Build and test the engine alone

```sh
cmake -S high_vision -B build-high-vision \
  -DHIGH_VISION_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-high-vision --parallel
ctest --test-dir build-high-vision --output-on-failure
```

The test rig covers exact bypass, translated-frame registration, telemetry
exposure anchoring, clipped-highlight retention, experimental-stage injection,
low-confidence shadow accumulation, high-support dark-object replacement, and
the Night persistence control.

For a repeatable CPU measurement:

```sh
cmake -S high_vision -B build-high-vision \
  -DHIGH_VISION_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-high-vision --target high_vision_benchmark --parallel
./build-high-vision/high_vision_benchmark 512 256 60
```

`high_vision_night_dynamic_benchmark` is the operator-level dogfood rig. It
builds high support on a structured 128x96 shadow scene with heteroscedastic
Poisson-Gaussian read/shot noise, reports temporal noise reduction, then
inserts a persistent dark rectangle and measures how much of its contrast has
appeared after 1/2/4/8 frames. It also moves the scene across a detector-fixed
random/row pattern and compares corrected and uncorrected scene beliefs. Local
tile search is deliberately enabled in the stationary section, making
recursive elastic transport an adversarial failure rather than an aesthetic
judgment.

```sh
./build-high-vision/high_vision_night_dynamic_benchmark
```

## Registration-free photon-orbit demo

`experiments/poisson_orbit_demo.py` is the first falsification rig for the
opposite low-light regime: every individual frame is below registration, and
the frame translations are never estimated.

It downsamples `skimage.data.camera`, randomly translates it, samples detected
photoelectrons with an exact Poisson camera model, adds configurable dark
current and read noise, and thins photons into three independent virtual
gatherers. Their cross-bispectrum is invariant to the unknown translation and
contains no same-photon self product. A small biphase solve reconstructs the
image orbit from accumulated power and third-order evidence.

`--photons` is the expected detected photoelectrons at a white pixel in one
frame: incident photon flux × exposure time × quantum efficiency. `--dark` is
expected dark electrons per pixel per frame, and `--read-noise` is RMS
electrons per full frame. The default is deliberately shot-noise limited
(`--read-noise 0`) so the first experiment isolates the claimed obstacle;
nonzero read-noise controls are supported.

```sh
.venv/bin/python high_vision/experiments/poisson_orbit_demo.py
```

The rendered comparison and full numeric ledger are written to
`high_vision/out/poisson_orbit_demo.png` and
`high_vision/out/poisson_orbit_demo.json`.

The seeded default uses 4,096 frames at 0.75 expected detected
photoelectron/white pixel/frame. Only 31.3% of pixels in the example frame are
nonzero. Measured results:

| Reconstruction | PSNR | SSIM |
|---|---:|---:|
| One photon frame | 4.78 dB | 0.001 |
| Unregistered average | 11.63 dB | 0.271 |
| Average using unrelated shift guesses | 11.64 dB | 0.267 |
| Translation-orbit cross-bispectrum | **21.80 dB** | **0.539** |

The reconstruction uses no clean-image frequency selection, learned prior, or
per-frame registration. The reference is used only afterward to choose the
unidentifiable global cyclic translation for scoring. This baseline establishes
registration-free orbit recovery, not spatial super-resolution yet. The next
experiment must add known downsampling and physical subpixel sampling phases.

### A rejected shortcut

The two unit-frequency bispectrum planes formally contain the wrapped x/y
gradient of Fourier phase. We tested replacing the many-constraint circular
phase fit with a coherence-weighted Poisson phase integration. It was fast, but
phase-wrap residues made even the noiseless integration path-dependent; the
512-frame smoke reconstruction fell from about 21 dB to 13.8 dB. The
implementation remains as `solve_phase_poisson` for controlled follow-up, but
it is not the default. The correct cheap solver must remain circle-valued
rather than linearizing phase before its topology is resolved.

`experiments/orbit_projection_sweep.py` now measures the missing dwell instead
of guessing about it:

```sh
.venv/bin/python high_vision/experiments/orbit_projection_sweep.py --resume
```

At 32² and 4,096 frames, increasing distinct non-antipodal projection steps
from 4 to 48 improves the orbit reconstruction from 22.09 to 22.82 dB and SSIM
from 0.668 to 0.700. Holding 32 steps and increasing the dwell from 1,024 to
8,192 frames improves 20.62 to 23.53 dB; the circle-valued phase residual falls
by about 65×. A matched 64-step full-circle stencil reaches 22.75 dB, nearly
identical to the 32-step half-circle stencil: antipodal measurements are
redundant here, while distinct radii and directions are useful.

The linear Poisson result was also tested as an initializer rather than a final
answer. It reaches 20.20 dB after 400 full-circle corrections and 21.34 dB
after a 1,400-iteration dwell, still below the ordinary circle-valued seed at
22.82 dB. It is entering the wrong winding basin, not merely stopping early.
The retained `--poisson-seed` control makes a future support-aware unwrapping
experiment reproducible.

## Budgeted full-resolution fusion

`experiments/budgeted_fullres_demo.py` separates the expensive
below-registration bootstrap from ordinary full-resolution accumulation:

1. stream frames once into a 128² block-sum thumbnail and a raw mean;
2. bound camera motion to a physical local support;
3. pool a few temporally adjacent frames for registration evidence;
4. register only those small piles;
5. apply each pile displacement once to its original 512² photon frames;
6. rebuild the thumbnail from the sharper fusion and remarch once.

No full-resolution registration FFT is performed per frame. Memory is
`O(image + registration_group × image)` rather than `O(frames × image)`.
The experiment can regenerate its deterministic synthetic stream for a second
remarch without storing all frames. A live camera cannot do that: one round is
the strictly streaming mode, while additional rounds require either a bounded
raw-frame ring buffer or a replayable recording. The one-round sweep below is
therefore the direct live-capture budget.
The synthetic camera also has an exact sparse-event path: draw the total
Poisson photon count, sample its source-weighted pixel locations, and bin the
events. This is distributionally identical to drawing every pixel separately
and is about 3× faster at 0.02 electron/white-pixel.

```sh
.venv/bin/python high_vision/experiments/budgeted_fullres_demo.py
.venv/bin/python high_vision/experiments/budgeted_fullres_sweep.py --resume
.venv/bin/python high_vision/experiments/extended_snr_sweep.py --resume
```

The default 512² run uses 512 frames, 0.1 expected detected
electron/white-pixel/frame, ±12-pixel smooth camera support, eight-frame
registration piles, and two bounded rounds:

| Quantity | Measured |
|---|---:|
| Nonzero pixels in one frame | 5.08% |
| Unregistered reconstruction | 18.25 dB / SSIM 0.161 |
| Bounded registered fusion | **19.60 dB / SSIM 0.249** |
| Cross-supported circular fusion | **25.87 dB / SSIM 0.658** |
| Independently measured noise reduction | **13.58 dB** |
| Final median registration error | 1.0 pixel |
| Three streaming passes | 4.49 s total |
| All-pass budget per input frame | 8.77 ms |

Profiling localizes the remaining cost. At 512², all thumbnail registration
calls together take only tens of milliseconds; synthetic photon generation is
the dominant loop. In live capture that simulation cost disappears.

### Cross-supported projection circles

The registered stream is divided into even and odd half-stacks. Their Fourier
cross-power estimates scene energy that independently survives both stacks;
one quarter of their squared spectral difference estimates the noise in their
mean. Pooling those measurements on Fourier circles produces a data-derived
Wiener support. It suppresses unsupported noise without a clean reference,
learned prior, or preferred image direction. The same split also measures
noise reduction without consulting the known cameraman source.

The one-round extended sweep is the direct live-capture budget:

| Flux and dwell | Raw registered | Circular support | SSIM | Noise reduction |
|---|---:|---:|---:|---:|
| 0.10 e⁻, 256 frames | 16.99 dB | 24.58 dB | 0.623 | 16.26 dB |
| 0.10 e⁻, 512 frames | 19.50 dB | 25.62 dB | 0.657 | 13.99 dB |
| 0.10 e⁻, 1,024 frames | 21.84 dB | 26.28 dB | 0.693 | 12.50 dB |
| 0.10 e⁻, 2,048 frames | 24.32 dB | **27.30 dB** | **0.738** | 11.23 dB |
| 0.04 e⁻, 2,048 frames | 20.69 dB | 25.35 dB | 0.669 | 14.23 dB |

Ring widths from one to eight pixels—roughly 363 down to 46 projection
circles—produce almost identical image scores. At 512², each circle already
contains enough Fourier samples; longer photon dwell matters much more than
finer radial binning.

### Physical moonlight benchmark

The Cameraman result is an algorithmic falsification rig, not the realism
claim. `experiments/realistic_moonlight_bench.py` is the single adversarial
benchmark for that claim:

```sh
.venv/bin/python high_vision/experiments/realistic_moonlight_bench.py
```

It uses one 0.05-lux-equivalent, 120 fps monochrome camera model: 0.86 detected
electron at a white pixel, 1.5 e⁻ RMS read noise, temporal row and column
noise, residual DSNU and PRNU, hot pixels, and a 12-bit ADC. The scene is a
floating-point 13.4-stop radiance field with continuous shadow texture.
Camera motion crops a larger canvas, so nothing wraps cyclically, and every
registered accumulation carries an explicit overlap mask.

Every reconstruction is compared against the same controls:

| 1,024-frame reconstruction | Log PSNR | Log SSIM | Shadow RMSE |
|---|---:|---:|---:|
| Raw unregistered mean | 9.29 dB | 0.005 | 0.107 |
| Same circular filter, no registration | 9.68 dB | 0.006 | 0.094 |
| Estimated registration + circular support | 9.68 dB | 0.006 | 0.094 |
| Oracle registration + circular support | **21.57 dB** | **0.308** | **0.006** |

The estimated bounded registration has 7.07-pixel median error and contributes
nothing over the filtered mean. This is intentional: the mean has no scene
support, and fixed sensor structure repeats between the two half-stacks, so a
naive cross-power test misclassifies it as scene signal. Oracle transport
decorrelates sensor-fixed structure and demonstrates that useful evidence is
present. This gap—not the favorable 27 dB Cameraman result—is the operative
night-vision benchmark.

The full sweep is recorded in
`out/budgeted_fullres_sweep.json`:

| Case | Unregistered | Bounded | SSIM | Median error | All-pass budget |
|---|---:|---:|---:|---:|---:|
| 128 frames | 14.22 dB | 14.45 dB | 0.092 | 2.24 px | 8.02 ms |
| One registration round | 18.25 dB | 19.50 dB | 0.239 | 1.41 px | 5.31 ms |
| Baseline, two rounds | 18.25 dB | **19.60 dB** | **0.249** | **1.00 px** | 7.92 ms |
| 0.04 e⁻, 16-frame piles | 15.30 dB | 15.87 dB | 0.114 | 2.24 px | 4.56 ms |
| 0.02 e⁻, 1,024 frames | 14.98 dB | 15.59 dB | 0.105 | 2.24 px | 3.32 ms |
| 0.15 e⁻ read noise | 16.82 dB | 18.38 dB | 0.189 | 1.00 px | 12.02 ms |
| 0.30 e⁻ read noise | 15.03 dB | 15.95 dB | 0.114 | 1.41 px | 12.24 ms |
| ±32 px support | 17.29 dB | 18.86 dB | 0.182 | 3.00 px | 8.39 ms |

The result exposes a useful boundary: below about 0.04 electron in this
configuration, more frames improve radiance SNR but do not automatically
restore registration. Registration-pile support must grow too, and its motion
diameter then becomes the competing blur. That is the state the orbit-invariant
bootstrap should resolve.

### Pre-mean orbit extraction

`experiments/realistic_orbit_bootstrap.py` accumulates translation-invariant
power and third-order phase evidence from individual calibrated frames. It
never first forms an image-space mean and it receives neither motion records
nor clean-image phase. The default is deliberately 64×64: 32×32 does not
contain enough Cameraman structure for this test, while the current evidence
does not yet support the additional rings of a 128×128 solve.

```sh
.venv/bin/python high_vision/experiments/realistic_orbit_bootstrap.py
```

Two temporal halves are independent phase witnesses. The first implementation
reduced their agreement to radial gates, marched outward through those rings,
and reconciled the selected wraps with one sparse least-squares factorization.
That path remains as a rejected control.

The earlier 22–23 dB descriptions were not valid reconstruction claims. The
HDR oracle has 90.3% of its pixels below 0.025 radiance; a black image already
scores 31.79 dB in linear PSNR, and the simulated unregistered mean scores
25.08 log dB. The reported orbit images preserve only low-frequency blobs and
do not visually recover the object. Log PSNR is retained only as a labeled
full-field diagnostic. Support-matched oracle images and structural residuals
are the operative controls.

### Third-order evidence and multi-orbit nucleation

The realistic path originally used a same-frame cubic moment. A Poisson photon
can then occupy two or all three factors, adding three power-spectrum
collisions and a DC collision. `realistic_orbit_bootstrap.py` now includes the
corresponding factorial-moment subtraction and a stronger physical control:
each 8×8 sensor block is divided into four disjoint 2×2 sublattices, and all
24 ordered triples of distinct gatherers are averaged in one symmetric
expression. A sensor sample can never collide with itself in that
cross-gatherer estimator.

The collision-free cross estimator improves low-frequency closure coherence,
but it does not by itself recover the image. Restricting cubic products to a
declared evidence disk remains a valid computational bound; unmeasured
frequencies are explicitly unsupported.

`experiments/realistic_multisource_consensus.py` tests a different topology:

```sh
.venv/bin/python high_vision/experiments/realistic_multisource_consensus.py
```

`experiments/closure_transport_consensus.py` now replaces that post-hoc phase
average with the Split-Bregman analogue. For phase chart `z`, every
bispectral triad carries

```text
beta(k,s) = z(k) z(s) conj(z(k+s)).
```

The triad closure defects are persistent transport state. Each local sweep
sends all three adjoint messages back to its frequency nodes, performs a
bounded circular residual projection, and updates its Bregman defect. A second
circular constraint couples the local charts to one consensus chart.
Translation is the exact linear-phase nullspace; convolutional correlation
synchronizes every source gauge to one anchor.

Nucleation no longer marches in a chosen radial direction. The two
unit-frequency closure families form a weighted connection Laplacian, and its
lowest mode initializes every frequency in the connected support
simultaneously. Exact-data tests recover a random supported image orbit to
numerical precision.

The realistic benchmark uses four disjoint temporal sources of 1,024 frames.
Each source retains the full sensor aperture; the four physical sublattices
are used only inside a frame to make collision-free third-order witnesses.
After 24 warm consensus sweeps:

- the four sources carry 1,318–2,485 admitted closure triads;
- their individual connected nuclei contain 71–145 frequencies;
- consensus residual falls from 0.0198 rad to 0.000165 rad;
- only 99 frequencies survive independent-source publication;
- the transported image scores 34.40 log dB / 0.893 SSIM against the oracle
  restricted to exactly that sparse support.

The last comparison says the operator transports its admitted evidence
coherently. It does not say the original scene has been recovered: the
support-restricted oracle is itself only a broad, sparse-frequency field. The
remaining blocker is now explicit—closure support and source-chart agreement,
not the phase consensus update or a flattering full-image PSNR.

### Sparse photon-ray transport

`experiments/sparse_ray_transport.py` asks whether registration has to resolve
each dark frame before integration. It does not assign a single shift. Every
detected photon instead votes through its complete admissible motion orbit.
The current belief supplies the exact cyclic Poisson likelihood of each ray,
a tempered posterior retains that uncertainty, and the photon counts are
backprojected through the entire posterior. The continual version consumes
each frame once and stores only the evolving radiance belief and bounded
support:

```sh
.venv/bin/python high_vision/experiments/sparse_ray_transport.py
```

On the fixed 64×64 benchmark, 1,024 frames each receive only 0.08 expected
photons at a white pixel and undergo an unknown cyclic translation within an
8-pixel disk. A separate 256-frame capture supplies held-out Poisson evidence.
The result is:

- unregistered mean: 16.35 dB / 0.177 SSIM;
- oracle-registered mean: 22.18 dB / 0.543 SSIM;
- one-pass soft ray transport: 22.09 dB / 0.677 SSIM;
- eight-pass soft replay control: 22.84 dB / 0.716 SSIM;
- nearly-hard shift posterior: 19.15 dB / 0.434 SSIM;
- injected posterior-noise consensus: 21.27 dB / 0.648 SSIM;
- autocorrelation-only phase retrieval: 11.90 dB / 0.060 SSIM.

The one-pass result also improves held-out marginal Poisson evidence over the
unregistered mean. Its final ordinary-scene posterior still spans about 52
effective shifts per frame, yet its truth-only audit has 1-pixel median and
2-pixel 90th-percentile shift error. Forcing that posterior toward a hard
registration collapses it to roughly one shift and makes recovery worse. The
useful perturbation is therefore epistemic diffusion across admissible rays,
not independent sensor noise or injected random logits.

A night-sky-shaped control adds eight ideal point sources at ten times white
radiance. They reduce the posterior to 11.7 effective shifts, raise its mean
peak probability to 0.396, and recover every shift at the median and 90th
percentile. This suggests a concrete division of labor: sparse, persistent
bright structure can establish the camera-motion gauge while diffuse photons
ride the same transport. These guide stars are an idealized control, not yet
a model of optics, atmospheric blur, subpixel motion, occlusion, or finite
sensor boundaries; those are the next conditions the operator must survive.

### Finite-D Poisson-flow probe

`experiments/finite_d_poisson_flow.py` tests geometry inspired by PPFM/PFGM++.
Here "Poisson flow" means the electrostatic Poisson equation, not the camera's
Poisson count likelihood. The experiment keeps those two mechanisms separate:
the physical count model scores registration rays, while finite-D geometry
changes either their posterior weights or the fusion of posterior-sampled
images.

```sh
.venv/bin/python high_vision/experiments/finite_d_poisson_flow.py
```

Every hyperparameter is selected without consulting Cameraman. A separate
256-frame capture selects the full-count likelihood flow. Complementary
binomial thinning of each observed count supplies two conditionally
independent photon streams for a second bidirectional cross-fit. The clean
image is exposed only afterward for PSNR and SSIM audit.

The literal image-charge construction does not work. Treating 16
posterior-sampled registered images as charges causes finite-D kernels to
select sampling noise and reinforce the initial blur. Its photon witness
selects the broad `D=inf` limit at radius 4, and the bidirectional result reaches
only 19.37 dB / 0.432 SSIM.

Applying the finite-D kernel to shift likelihoods is stable, but finite D
provides no demonstrated advantage on this benchmark. The independent capture
selects the Gaussian limit (`D=inf`) at temperature 2:

- original temperature-4 soft transport: 22.09 dB / 0.677 SSIM, 49.3
  effective final shifts;
- heldout-selected temperature-2 transport: 23.01 dB / 0.603 SSIM, 11.9
  effective final shifts;
- oracle-registered noisy mean: 22.18 dB / 0.543 SSIM.

The held-out improvement over temperature 4 is clear under a paired
per-frame audit. `D=inf` also beats `D=64` and `D=128` at the selected
temperature, although those dimension effects are much smaller. A single
photon-thinning split weakly selects `D=64`, temperature 1.5, but its
cross-fitted reconstruction is only 22.24 dB / 0.555 SSIM; it is not evidence
for a finite-D recovery gain.

The useful result is a support budget rather than an image prior. Roughly
12 live shift hypotheses outperform both the 49-shift diffuse posterior and
the previously observed one-shift collapse. That observation is implemented
directly by the entropy-budget operator below.

### Entropy-budgeted transport

`experiments/entropy_budget_transport.py` replaces global temperature with an
explicit effective-support state:

```text
K = exp(mean_frame H[p(shift | photons)])
```

For each batch, a scalar bisection finds the inverse temperature whose exact
Poisson shift posterior has the requested entropy. This remains on the
likelihood's exponential family; it is an uncertainty projection, not an image
prior or post-filter.

The continual operator does not fix `K`. It binomially thins the current
photons into complementary streams. Each half constructs a posterior at every
candidate budget, and its predictive likelihood is measured using the other
half. Those bidirectional scores accumulate without forgetting. The winning
budget is then applied once to the unsplit batch, so all detected photons reach
the reconstruction and no frame is replayed.

```sh
.venv/bin/python high_vision/experiments/entropy_budget_transport.py
```

On the 64×64, 1,024-frame, 0.08-photon benchmark, support marches
`96 → 64 → 48 → 32 → 24 → 16` and changes only five times:

| Continual operator | PSNR | SSIM |
|---|---:|---:|
| Temperature 4 | 22.09 dB | 0.677 |
| Temperature 2 | 23.01 dB | 0.603 |
| Fixed entropy, photon-witness-selected `K=16` | 22.91 dB | 0.602 |
| **Cumulative-evidence entropy flow** | **23.61 dB** | **0.634** |

Across five independent photon and motion seeds, cumulative entropy beats
temperature 2 in all five trials: mean improvement **+0.257 dB** and
**+0.0209 SSIM**. This is a truth-only repeatability audit; neither metric nor
the clean image feeds the operator.

A preliminary 128×128 run with the same 1,024 frames sharpens the distinction:
temperature 2 reaches 20.34 dB / 0.383 SSIM while cumulative entropy reaches
23.77 dB / 0.532 SSIM and contracts to eight shifts. Capturing the synthetic
stream and running both operators took about 8.5 seconds in the Python reference.
That timing is a throughput check, not yet a camera-latency guarantee.

There is an important distinction between reconstruction and predictive
calibration. A rapidly forgetting controller obtains slightly better
separate-capture likelihood but advances support too eagerly and reconstructs
only 22.98 dB / 0.588 SSIM. Cumulative friction preserves early ambiguity and
lets repeated cross-photon agreement nucleate the gauge before contracting.
Its heldout likelihood is essentially tied with fixed temperature 2 while its
image audit improves materially.

`StreamingEntropyTransport` is the batch-ingest reference. `push(batch)`
consumes a new photon-count batch exactly once and returns the current belief
and diagnostics. `push(batch, discontinuity=True)` atomically drops the old
gauge and initializes a new one, matching camera reconnects, hard moves, and
scene cuts.

#### Native continual controller

The validated operator now has a reusable C++17 implementation in
`high_vision::EntropySupportController`. A registration backend supplies
row-major shift scores for two complementary photon witnesses. The controller:

- performs the exact scalar entropy projection;
- accumulates bidirectional predictive evidence without storing old frames;
- moves at most one adjacent support tier per batch;
- returns a normalized posterior for the unsplit observation;
- exposes the selected and achieved support, inverse temperature, evidence
  margin, batch count, and transition count;
- returns to the broadest support and clears all evidence on `reset()`.

Its persistent memory is only one evidence value per candidate budget. Posterior
storage is bounded by `batch frames × admitted shifts`; there is no capture
history and no iterative replay.

```sh
cmake -S high_vision -B build-high-vision \
  -DHIGH_VISION_BUILD_TESTS=ON \
  -DHIGH_VISION_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-high-vision --parallel
./build-high-vision/high_vision_entropy_benchmark
```

On the local release build, 31 updates of 32 frames × 197 candidate shifts
take about **17 ms per batch** after warmup, including complementary evidence projection
for all nine budgets and the final unsplit projection. At 120 fps, a 32-frame
batch arrives every 267 ms, so the controller consumes about 6.4% of one CPU
core's batch interval. This benchmark deliberately excludes photon-score
generation and image backprojection; those remain the computationally hot
backend.

The native interface is now ready for a BFFT correlation backend, but it is not
silently enabled in OBS. Ordinary 8-bit OBS luma has already passed through a
camera ISP and cannot be claimed to provide exact Poisson thinning. A raw or
high-bit-depth adapter can supply physical photon witnesses; an OBS adapter
will need an explicitly labeled approximate witness construction and its own
validation.

## OBS use

Build and install the plugin as described in `../obs-plugin/README.md`, restart
OBS, and add **BFFT High Vision** to a camera source.

Useful starting points:

- **Synthetic HDR:** support 24, persistence 0.985, registration radius 6,
  local search 2.
- **Night integrator:** use the Night integrator mode, support 60–120, and a
  fixed camera exposure/gain when possible. This mode relaxes scene-cut
  detection because census descriptors themselves become noisy at very low
  signal levels. Weak registration now collapses motion toward the stationary
  gauge without erasing accumulated support, and every unclipped dark frame
  contributes a full observation unit instead of HDR's shadow-suppressed
  weight.
- Use **Reset temporal belief** after a lens/exposure-mode change or a hard
  camera reposition.

OBS does not currently expose exposure time or analog gain on
`obs_source_frame`, so the adapter uses robust relative exposure estimation.
Direct camera or recorded-RAW adapters should populate `FrameMetadata` for a
radiometrically anchored result.

## Adding the inverse-diffusion experiment

Implement `high_vision::ExperimentalStage`, install it with
`Processor::set_experimental_stage`, and select `Mode::experimental`. The stage
may modify the scene-linear belief in place. It receives:

- the current exposure-normalized observation;
- the already transported per-pixel support;
- width, height, timestamps, exposure/gain/black/white metadata, sequence, and
  sensor temperature;
- registration and support diagnostics.

The intended contract is that adding noise, erosion, or a learned prior changes
the belief only. The framework remains responsible for whether and where
evidence should survive.

## Current boundaries

- The core currently processes one scene-linear luminance plane. Multi-plane
  RGB and raw Bayer adapters can be added without changing the temporal
  contract.
- Registration is translational (global plus local). Rotation, scale, rolling
  shutter, and projective camera motion are future registration strategies.
- The OBS path is 8-bit and uses a standard transfer-function approximation.
  Scientific night work should enter through a higher-bit-depth adapter rather
  than quantize through OBS first.
- The OBS Night integrator is the repaired persistent hard-registration path,
  not yet the entropy-budgeted photon posterior. The latter remains behind the
  raw/high-bit score-backend boundary.
- The display mapper is intentionally conservative. It is not the final
  aesthetic HDR operator and does not manufacture values beyond supported
  measurements.
