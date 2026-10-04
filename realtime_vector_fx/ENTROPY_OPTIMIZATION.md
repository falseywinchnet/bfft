# Same-output entropy stretch optimization — 2026-09-10

This revision preserves the filter ID, settings, defaults, RGB transform,
entropy definition, lattice phases and density, LUT resolution, gain limits,
and time-based adaptation law. The shader text is unchanged. No weaker
analysis, reduced precision, approximation to `pow`, or reduced update rate is
used to obtain the speedup.

## Changes

1. Retain sample storage across analysis updates instead of allocating and
   initializing a fresh vector. Reset every reused sample's validity and entropy
   before processing, including transparent and boundary cases.
2. Cache each sample's RGB-family index and compute its family's conditional
   entropy weight once per occupied family, rather than repeating identical
   division and `pow(n, .75)` work for every sample.
3. Precompute entropy logarithm terms for the possible block counts 1..64 once.
   Preserve the original division, multiplication and histogram accumulation
   order instead of changing the entropy estimator.
4. Use ARM64 NEON for independent LUT entries: maximum-distance reduction,
   profile interpolation, and float-to-UNORM16 packing. The ARM float update
   uses the same fused multiply-add as the original compiler-generated update.
   A portable scalar implementation remains available. Covariance accumulation
   is kept in its original order.
5. Compute the next maximum target distance during the current update. Reuse
   that exact maximum on the next frame; invalidate it when a new target is
   built. This avoids a separate full-array scan on most frames.
6. Fuse interpolation and upload packing in one traversal. Upload only if the
   resulting 16-bit texture values changed. Continue the original float
   recurrence even when quantization hides a small change, so sub-code changes
   accumulate and cross an output-code boundary at the same step.
7. Cache OBS effect parameter handles rather than resolving their names on
   every frame.
8. On Metal, defer the staging copy of a rendered lattice until the beginning
   of the next video frame, before that frame's source capture. OBS 32.2.1's
   `stageTextureToBuffer` waits for completion at copy submission; the original
   call therefore waited behind the current frame's capture and lattice draws.
   The revised schedule reads the same previous-frame lattice and does not add
   another frame of analysis latency. Other graphics backends retain their
   original staging schedule.

## Same-output checks

A frozen pre-optimization core and OBS wrapper are retained under
`tests/reference/`. They are test assets and are not linked into the production
plugin. The reference suite compares:

- Exact analysis acceptance, sample counts, entropy, covariance, whitening
  transform and allocation curves.
- Every float in the evolving 33³ LUT across varying dimensions, padding,
  transparency, low variation, gradients, random colors, frame intervals and
  target/settings changes.
- Every packed UNORM16 entry and the decision to skip an upload. An upload may
  be skipped only when the packed table is exactly unchanged.
- Long settling sequences, scene changes, and zero elapsed-time updates.

**255,296,448 float values matched bit-for-bit.** Packed tables also matched
bit-for-bit. Both the ARM SIMD and forced portable-scalar paths pass. The full
comparison also passes AddressSanitizer and UndefinedBehaviorSanitizer.

The actual OBS/Metal comparison uses separate original/optimized modules with
an identical deterministic **test-only** profile clock. All **180 full 1080p
frames** hash identically at the user's saved tuning: decorrelation 0.09,
allocation 0.11, gain cap 16, lattice width 256; other controls use their existing
defaults. The six additional render-submission runs have identical final-frame
hashes. This clock override and phase instrumentation exist only in isolated
benchmark builds; neither is present in the installed plugin.

The standard invariant suite and pre-existing RVFX core suite pass as well.

## Measured cost

CPU numbers below are medians of 31 alternating before/after timings on the
Apple M4, compiled with the same `-O3` settings. Analysis batches contain 50
updates; profile-update batches contain 40 evolving frames.

| Work | Original | Optimized | Reduction |
|---|---:|---:|---:|
| 128 × 72 analysis + target LUT | 309.426 µs | 236.580 µs | 23.5% |
| 256 × 144 analysis + target LUT | 881.805 µs | 592.768 µs | 32.8% |
| Float profile update, 33³ LUT | 75.060 µs | 19.669 µs | 73.8% |

The last row times float-profile updating alone; the integrated OBS path also
fuses packing and omits redundant uploads.

At 1080p on the MacBook's Apple A18 Pro with OBS 32.2.1/Metal, three alternating
before/after runs measured the render thread with the user's saved settings.
Each run has 30 warmup frames and 149 measured frames. These runs submit and
flush GPU work without a full-frame test readback each frame; they read the
last output once for equivalence verification.

| Render-thread statistic, median across runs | Original | Optimized |
|---|---:|---:|
| Median per-frame cost | 1.789 ms | 1.302 ms |
| Mean per-frame cost | 2.623 ms | 2.269 ms |
| 95th percentile | 6.529 ms | 6.364 ms |

That is about **27% less median render-thread time** and **13.5% less mean
render-thread time** by these median-of-run statistics. GPU staging waits are
smaller, but this is not a measurement of shader execution time or a promise of
27% more output FPS. The GPU timer API is unimplemented in this Metal backend.
The output shader is unchanged, so GPU arithmetic/sampling work is unchanged.

The full-readback equivalence screen was noisy and did not show an end-to-end
speedup (16.25 ms before, 18.56 ms after in its initial pair). Repeated full
readback synchronizes every frame and is not part of the production filter.
The submission runs also had scheduling outliers, up to about 81 ms. No claim
of improved worst-case latency or a universal real-time guarantee is made.

## Reproduction and retained results

Build/run the core checks and paired benchmark on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  '/opt/homebrew/bin/cmake -S realtime_vector_fx -B /tmp/rvfx-entropy -DCMAKE_BUILD_TYPE=Release && \
   /opt/homebrew/bin/cmake --build /tmp/rvfx-entropy -j4 && \
   /opt/homebrew/bin/ctest --test-dir /tmp/rvfx-entropy --output-on-failure && \
   /tmp/rvfx-entropy/rvfx_entropy_benchmark > /tmp/entropy_optimization_cpu.json'
```

Copy the JSON immediately using the host selected by `m4host`.
`tools/build_entropy_comparison.py` builds isolated instrumented modules with
matching OBS headers. Run its `driver` with a module path, Metal library path,
and output hash path. Add `--submit-only` for render-thread timings. Run these
modules on the host with matching OBS 32.2.1; the Mini's older installed OBS
cannot load them. No remote OBS installation is changed.

Retained evidence is in `output/entropy_stretch/optimization/`:

- `results.json`: paired CPU and Metal measurements, full run statistics.
- `cpu-second.json`: final CPU timing and exact-comparison count.
- `core-tests.log`, `sanitizer.json`, `scalar.json`: regression evidence.
- `before-0.hashes`, `after-0.hashes`: identical 180-frame sequences.
- `before-submit-*.log`, `after-submit-*.log`: three timing pairs.

Build the production module with `tools/build_obs_macos.sh`. It contains neither
reference implementation nor benchmark clock. Replacing the plugin bundle does
not change the saved scene/filter configuration.

The final uninstrumented production bundle also passes the actual 1080p Metal
smoke (identity, alpha, CPU reference, freeze, bypass, and color-space handling)
and the existing FX/Posterizer smoke. Their retained logs are
`production-smoke.log` and `existing-fx-smoke.log` in the optimization output folder.
