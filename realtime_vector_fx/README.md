# Realtime Vector FX

This directory is the first unified C++ implementation of the repository's
posterizer and topology-first vector tracing ideas, designed around a strict
33.3 ms frame deadline rather than one-shot file export.

## Current pipeline

1. A stratified sample updates a persistent OKLab palette. Population mass is
   tempered with the posterizer's sublinear population exponent, local detail
   receives extra weight, and an EMA preserves palette identities over time.
2. The frame is assigned on a bounded topology lattice. This is the live
   analogue of coarse ownership: its memory and work do not grow with SVG path
   complexity. Exact packed-RGB/YUV source tokens let unchanged cells reuse
   cached OKLab state while still being reassigned against updated priors.
3. Horizontal and vertical ownership changes update dense edge slots. Edge
   position is its stable ID, so unchanged trace segments retain age and phase
   without maps, heap allocation, or global path matching. Adjacent compatible
   slots compile into maximal continuous trace runs before drawing.
4. A persistent shuffled permutation selects a bounded subset of trace runs
   each frame. Every current run is visited before any run repeats, and each
   visit emits a randomized local slice in phosphor, liquid-metal, or embossed
   source-color-sheen mode.
5. A separate persistent particle engine spawns `#82b361` glyphs on trace
   geometry, then evolves them as falling streams, curved trajectories, or a
   mixture. Glyphs and their glowing trails continue independently of the base
   trace-effect choice.
6. The CPU compositor posterizes and draws commands directly into packed RGB,
   NV12, or I420 frames. The public core separates command generation from
   rendering so the OBS GPU filter can posterize the full-resolution texture
   with a shader and stream only low-resolution analysis pixels to the CPU.

## Dedicated posterizer filter

The OBS bundle also registers **Optimal OKLCH Posterizer**, a posterization-only
GPU filter. It does not allocate trace-history textures or a geometry buffer,
and the core skips edge extraction, segment scheduling, glyph simulation, and
all overlay work. Its properties expose 2–64 colors plus the original
posterizer's node separation, lightness/chroma/hue/alpha metric weights, detail
priority, sublinear area exponent, minimum leaf size, split-refinement passes,
sample budget, analysis resolution, and temporal-prior learning rate.

A lightweight GPU-only finish adds optional graphic contours at palette-region
boundaries, restrained luminance-detail ink inside those regions, line reach,
saturation, and contrast. The defaults are intentionally subtle. Setting both
ink sliders to zero and saturation/contrast to `1.0` restores the unadorned
posterized appearance; this finish does not create or schedule vector segments.

The cold palette seed uses the original occupied-space strategy: rarity/detail
importance, local cylindrical OKLCH tangent coordinates, principal and
coordinate split proposals, exact weighted SSE cut searches, deterministic
two-means refinement, and globally gain-prioritized leaf splitting. Display
nodes retain gamut-safe lightness and hue when separation pushes chroma beyond
sRGB. Full-budget temporal centroid updates then keep palette identities stable
from frame to frame.

No allocations occur in the full-lattice, palette-sampling, ownership, or
edge-update inner loops after a resolution and configuration are established.

## Build and measure

The Makefile needs only a C++17 compiler:

```sh
make -C realtime_vector_fx test
make -C realtime_vector_fx benchmark
```

The benchmark defaults to 180 composited 1280x720 frames after 12 warmups and
returns status 2 if the p95 frame time exceeds 33.3 ms. Optional arguments are
`width height frames [nv12] [static]`.

Render a 90-frame visual fixture that cycles through all base effects while
the independent mixed-motion glyph layer persists:

```sh
realtime_vector_fx/build/rvfx_demo /tmp/rvfx_demo_frames
svg_converter/.venv/bin/python realtime_vector_fx/tools/render_demo.py \
  /tmp/rvfx_demo_frames /tmp/rvfx_demo.gif
```

Current M4 Mini results are recorded in [BENCHMARKS.md](BENCHMARKS.md): changing
1080p p95 is 5.534 ms for RGBA and 5.681 ms for direct NV12. Exact cache hits
lower static 1080p to 3.980 ms and 3.142 ms respectively. The synchronized
headless Metal/libobs benchmark measured a 2.903 ms median total filtered frame
and 2.079 ms median incremental 1080p overhead across four runs on the
development A18 Pro host.

CMake builds the same core, tests, and benchmark. Set `RVFX_BUILD_OBS=ON` and
`OBS_SOURCE_DIR` to compile the filter adapters. The CPU fallback supports OBS
RGBA/BGRA/BGRX/NV12/I420 async frames. The GPU filter performs full-resolution
posterization in a graphics effect, uses double-buffered one-frame-delayed
readback only at the bounded trace resolution, and draws persistent traces and
5×7 phosphor glyphs through a reusable dynamic vertex buffer. On macOS the
target also packages and ad-hoc signs
`realtime-vector-fx.plugin`; OBS source headers must match the installed OBS.
`tools/obs_smoke.mm` is a non-installing libobs harness: it initializes OBS's
Metal backend, loads the built module, creates both filters, attaches the GPU
filter to a synthetic moving source, renders eight frames through the real
filter chain, and verifies nonempty staged output. It stays separate from the
portable core build because it requires AppKit and matching generated OBS
headers.

On macOS, `tools/build_obs_macos.sh` reproduces the verified arm64 build,
bundle, and ad-hoc signature in `dist/realtime-vector-fx.plugin`. Set
`OBS_SOURCE_DIR`, `OBS_CONFIG_INCLUDE_DIR`, and `SIMDE_INCLUDE_DIR` to matching
OBS build dependencies. `RVFX_RUN_OBS_SMOKE=1` additionally compiles and runs
the non-installing Metal filter-chain test.

## Source inventory and lineage

- `../posterizer/src/posterizer/core.py` and `oklch.py`: perceptual OKLab/OKLCH
  ownership, detail-weighted/sublinear population allocation, deterministic
  bifurcation, assignment, and component cleanup. This is the optimized
  posterizer (commits `a25e385`, `701e00f`, `98ae5d5`).
- `../svg_converter/src/tlvector/core.py`: topology-first owner lifting,
  parent-locked residual colors, exact oriented raster-square boundary loops,
  simplification, subpixel relaxation, and quadratic SVG compilation. This is
  the standalone optimal/error-bounded SVG maker (commits `157cf3f`,
  `b816221`/`e417296`).
- `../svg_converter_v2/src/tlvector_v2`: compact lattice serialization,
  error-per-byte component merging, SVGZ, and rank-one affine-gradient cell
  painting (through commit `0453e94` locally). It is a second optimized SVG
  engine, not a temporal animation engine.

The earlier SVG animation implementation was recovered from task history as
**SVG Oscilloscope Renderer** v5. Its exact 970-line browser artifact is
preserved at `legacy/svg_oscilloscope_renderer_v5.html`, and its scheduling and
rendering semantics are mapped onto this native engine in
[SOURCE_INVENTORY.md](SOURCE_INVENTORY.md).

## Near-term performance work

- extend exact cell reuse into coarse dirty tiles that can skip unchanged
  edge-run recompilation;
- move palette assignment and edge extraction from bounded CPU analysis into
  compute shaders, retaining the current low-resolution readback as fallback;
- add SIMD-friendly SoA OKLab assignment (NEON first, AVX2 second);
- fit optional smooth curves over compatible trace runs while preserving
  stable constituent IDs;
- add native P010 sampling and HDR-aware palette state;
- profile capture-to-present latency in the OBS application, then add a stable
  installer/notarization path once the live filter behavior is accepted.
