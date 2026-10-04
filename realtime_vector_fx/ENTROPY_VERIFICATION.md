# Entropy stretch verification — 2026-09-10

## Retained results

| Check | Result |
|---|---|
| Release CMake build on Apple M4 | Passed |
| Existing RVFX core suite | Passed |
| New entropy core suite | Passed |
| AddressSanitizer + UndefinedBehaviorSanitizer on entropy core | Passed, no diagnostics |
| Existing actual OBS/Metal effects and Posterizer Mark IV smoke | Passed |
| New actual OBS/Metal 1920 × 1080 smoke | Passed |
| Identity LUT maximum pixel error | 0 byte codes |
| Alpha preservation maximum error | 0 byte codes |
| CPU/GPU enhanced output agreement | Mean 0.0255 byte codes; maximum 1 |
| Frozen assignment with static input | Bit-exact across repeated frames |
| Zero effect strength | Exact bypass |
| Floating SDR and HDR source-space forwarding/bypass | Passed |

The core suite checks the whitening identity M C Mᵀ = σ² I, singular grayscale
behavior, flat scenes, finite in-gamut LUT values, strictly increasing allocation
curves, expansion of a high-entropy interval beyond identity, contraction of a
low-entropy interval below identity, time-based smoothing at 30/60 fps, the
per-update slew bound, transparent-input rejection, invalid stride handling,
and exclusion of row padding.

The GPU smoke uses a static synthetic image with low-entropy gray ranges,
high-entropy colored material, an alpha-0.5 band, and a transparent band. It
compares the settled GPU profile with an independently applied CPU reference.
It renders 150 timed frames, yielding to the OBS video thread between frames.
UI updates are asynchronous and the controls checks wait for the next video
tick before asserting their effects. The smoke uses fast adaptation settings
(0.05 s, 2 units/s, 30 analysis updates/s) to reach a settled profile quickly;
these are more demanding than the normal 10 Hz analysis cadence.

## Timing and its limits

- **M4 CPU, Release build:** 0.298 ms per 128 × 72 analysis and complete 33³ target
  LUT rebuild, averaged over 100 repetitions in the core test.
- **MacBook Apple A18 Pro, OBS 32.2.1, Metal, 1920 × 1080:** final smoke baseline
  2.321 ms; adaptive source + filter + full output readback mean **9.116 ms**;
  maximum **15.189 ms** over 150 frames.

These GPU timings include a full-resolution staging/readback in the test harness
that the production filter does not perform, but they also use a cheap static
synthetic source. They are a functional performance screen, not an isolated
GPU-kernel benchmark or a universal 1080p60 guarantee. No sustained production
stream, multi-filter scene, thermal stress, Windows, or Linux GPU measurement is
claimed. The Mini's installed OBS was older than the 32.2 target headers, so
CPU work and compilation ran there while final Metal validation ran on the
MacBook's matching OBS runtime. No remote software was changed.

## Implementation corrections covered by verification

The first Metal initialization attempt exposed an OBS RGBA32F texture-upload
trap: its `bytesPerPixel` helper omits 128-bit formats. The final plugin uploads
a 16-bit UNORM LUT while retaining the float profile for temporal interpolation.
This has sub-code precision relative to the 8-bit SDR output. Identity and
CPU/GPU tests cover the final representation and atlas layout.

The filter also forwards its upstream color-space declaration and bypasses
non-SDR sources or non-SDR render targets, so it does not claim HDR support
while changing the declared color space.

## Artifacts

- `output/entropy_stretch/core-tests.log`: retained CTest output and CPU timing.
- `output/entropy_stretch/obs-smoke1080.log`: final actual Metal regression.
- `output/entropy_stretch/existing-fx-smoke.log`: existing filter regression.
- `output/entropy_stretch/obs-smoke1080.png`: synthetic filtered output.
- `output/entropy_stretch/cat-comparison.png`: original and settled default-profile
  reference render using the repository's existing JPEG fixture.
- `realtime_vector_fx/dist/realtime-vector-fx.plugin`: signed development bundle.
- `realtime_vector_fx/dist/realtime-vector-fx-entropy-macos-arm64.zip`: packaged bundle.

The natural-image reference is a visual example, not a denoising or perceptual
quality benchmark. It exhibits the intentional false-color rendering. The
plugin bundle was built and verified without modifying the installed OBS copy.

## Optimization follow-up

The measurements above describe the first installed build. The subsequent
same-output optimization, additional exact-reference regressions, and paired
performance measurements are recorded in [ENTROPY_OPTIMIZATION.md](ENTROPY_OPTIMIZATION.md).
