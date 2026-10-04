# Mosaic FX

A deterministic tessellated-mosaic effect, built as its own OBS plugin
(**Mosaic FX**, source id `mosaic_fx_filter`). It consists of a portable C++17
core, a CPU reference renderer, a still/synthetic-motion driver, unit tests, the
OBS filter with its two-pass GPU shader, and an actual libobs/Metal smoke
regression.

## Construction

Every stage is a direct, single-pass construction (no relaxation, Lloyd
iterations or solves):

1. **Observe.** Oklab of the analysis frame (lightly blurred for color
   sampling) and a structure luma. The driver supplies the BFFT Meyer cartoon
   (`bfft_meyer_split` on a power-of-two lattice with reflection padding, as in
   BFFT Cartoon). Tiles therefore follow object outlines, not texture.
2. **Entropy → tile size.** 8×8 color-id entropy with the entropy stretch's
   variance gate is smoothed and thresholded into 1–4 size classes. One global
   scale is chosen so the expected tile count meets `target_tiles`
   (10k–20k intended; measured counts are within about 5%).
3. **Edges.** Structure and chroma gradients, non-maximum suppressed, with
   fragments shorter than `edge_min_length` removed.
4. **Distance/feature transform.** Exact Felzenszwalb–Huttenlocher EDT to the
   edges plus the frame. Level sets of `distance / size` are the contour rows
   (andamento). The direction to the nearest edge, smoothed as a doubled-angle
   field, orients each tile.
5. **Nucleation.** Pixels on row centrelines are taken in row order, with the
   outline row first and raster order within a row. A candidate is accepted
   when no seed lies within `spacing × mean size`. A raster gap fill follows.
   Outline-row tiles are darkened according to edge contrast. Tile jitter, facet
   tilt and twinkle phase are hashed from the tile position, so an identical
   frame always gives an identical mosaic.
6. **Backfill (render).** Each pixel belongs to the tile that is nearest in that
   tile's own oriented L∞ metric. Grout appears outside every tile square and
   near cell bisectors. Shading uses the facet normal plus a bevel; the glint is
   a Blinn highlight from random facet tilts under a slowly sweeping light, with
   per-tile twinkle.
7. **Color.** Tiles are sampled at five points and quantized to the Posterizer
   Mark IV palette (`family_priority` 1 by default), with small per-tile
   lightness/chroma jitter.

### Realtime paths and cost model

- `wants_structure()` compares each 8×8 block's mean luma and two chroma
  differences with the values recorded when the block was last examined. The
  per-channel median shift is removed, so exposure and white-balance changes
  recolor tiles instead of waking geometry. One pass over the frame computes
  it, and the result is cached for the following call.
- Structural testing covers only the woken blocks. It removes each block's
  mean change before testing.
- Geometry runs on the dirty 16-px blocks alone, merged into per-block-row
  runs:
  - entropy, edges, fragment filter and nucleation run on the runs;
  - the distance-transform envelope and orientation run only on rows within
    2 px of a run, and only over that span.
  - A full rebuild is the same code with one full-width run per block row, so
    it remains a pure function of the frame.
- Persistent per-pixel state is 22 bytes: structure, reference, size class,
  edge code, outline weight, frame flag, distance, int16 doubled-angle normal
  and int16 column sites. Oklab, gradients and blur passes are per-run
  scratch. The orientation blurs stream through 5-row and 3-row rings. The
  fragment filter is a flood fill that marks visited pixels in the edge byte.
- Tiles are re-measured only when one of their prefilter blocks woke (or
  after a global shift). They are re-quantized only when that measurement or
  the palette changed.
- `pack()` splits geometry (two RGBA16 texels per tile, plus the cell table)
  from colors (one texel per tile). A color-only change repacks and uploads
  colors alone. `unpack()` decodes what the shader sees, so the CPU renderer
  can verify the GPU.

### Temporal contract

Tiles persist across frames. Each 16×16 block compares the current structure
with the structure its tiles were laid on, not with the previous frame, so
slow drift still accumulates. Blocks above `restructure_threshold` are dilated
by one block and re-nucleated locally against the kept neighbours. More than
`scene_cut_fraction` dirty blocks triggers a full rebuild, which is identical
to a fresh instance. A tile recolors only when its measured Oklab color moves
by more than `recolor_threshold`.

## OBS plugin

`obs/mosaic-filter.cpp` follows the RVFX GPU filter lifecycle:

1. **Render thread:**
   - Capture the source. At the analysis rate, downsample it on the GPU to
     straight (un-premultiplied) RGB with alpha 1, so the CPU never
     un-premultiplies.
   - Read it back one frame late (deferred on Metal) and hand the newest
     frame to the worker without waiting.
   - Adopt published data by buffer swap.
2. **Worker thread:**
   - Run the prefilter. On a wake, run Meyer plus region-limited re-tiling,
     at most **Re-tiling passes per second** (default 6). Woken blocks wait
     until then, while their tiles keep recoloring at the full sample rate.
   - Posterizer Mark IV re-learns the palette while warming up, on every
     fourth wake, and on a slow refresh. A node is adopted only if it moves
     by more than about 5 codes.
   - Geometry and colors are published separately.
3. **GPU, two passes:**
   - *Locate* runs only when geometry changes or the size changes. It
     searches the 3×3 cell neighbourhood (at most 32 tiles per cell; the size
     span is capped at 3×). It writes owner index, grout distance, bevel
     direction and facet tilt to an RGBA16 owner cache.
   - *Draw* runs every frame with three fetches per pixel (source, owner
     cache, tile color). It applies grout coverage, bevel, relief, glint
     sweep, twinkle and mix.
   - The two passes use separate effect objects. On OBS 32.2.1 Metal, running
     both techniques from one effect crashed the following draw on a stale
     texture parameter.

The filter writes a 30-second work profile to the OBS log (`[Mosaic FX] 30 s
profile`). With `MOSAIC_FX_STATS=path` it also writes the profile as JSON when
the filter is destroyed. The filter bypasses itself exactly for floating/HDR
color spaces and at zero mix.

### Build, smoke and install

Build on the Mini with OBS 32.2.1 headers. If `/tmp` has been cleared, clone
them again as in `obs-plugin/README.md`, add
[simde v0.8.2](https://github.com/simd-everywhere/simde) in `/tmp/simde-0.8.2`,
and write a two-line `/tmp/obsconfig.h`:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  '/opt/homebrew/bin/cmake -S mosaic_fx -B /tmp/mosaic_obs_build -DCMAKE_BUILD_TYPE=Release -DMOSAIC_BUILD_OBS=ON && \
   /opt/homebrew/bin/cmake --build /tmp/mosaic_obs_build -j8 && /tmp/mosaic_obs_build/mosaic_tests'
```

The Mini's installed OBS (32.0.4) is older than the headers, so the Metal smoke
runs on the MacBook's OBS 32.2.1. Copy `mosaic-fx.plugin` and `mosaic-fx-smoke`
back, then run:

```sh
./mosaic-fx-smoke mosaic-fx.plugin/Contents/MacOS/mosaic-fx \
  /Applications/OBS.app/Contents/Frameworks/libobs-metal.dylib OUT_DIR 1920 1080
```

The retained logs and frames are in `output/mosaic_fx/obs/`. The verified
bundle is in `mosaic_fx/dist/` (git-ignored). Installing is a manual step:

```sh
cp -R mosaic_fx/dist/mosaic-fx.plugin "$HOME/Library/Application Support/obs-studio/plugins/"
```

## Build and run the core (M4 Mini)

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  '/opt/homebrew/bin/cmake -S mosaic_fx -B /tmp/mosaic_build -DCMAKE_BUILD_TYPE=Release && \
   /opt/homebrew/bin/cmake --build /tmp/mosaic_build -j8 && /tmp/mosaic_build/mosaic_tests'
```

The still driver reads and writes binary PPM. Convert inputs locally with
`.venv-jpeg` Pillow, copy them to the Mini's `/tmp`, and then run:

```sh
/tmp/mosaic_build/mosaic_still in.ppm out.ppm --palette 32 --lattice 512 --diag diag.ppm
/tmp/mosaic_build/mosaic_still in.ppm pan.ppm --palette 32 --frames 6 --shift 3
```

`--shift` pans only the right half to exercise local re-nucleation. Copy
outputs back immediately. The retained stills and their JSON receipts are in
`output/mosaic_fx/`.

## Measured

Timings on this MacBook are unreliable whenever it is under load: an OBS
session with many filters, browsers and swap. During one run, swap use was
9.2 GB on the 8 GB machine. The work counts below do not depend on load, and
the timing rows come from quiet runs.

### OBS/Metal (MacBook A18 Pro, OBS 32.2.1, quiet machine)

| | 1280×720 | 1920×1080 |
|---|---|---|
| Added per steady frame (120-frame throughput) | — | **0.23 ms** (was 0.60, first build 6.4) |
| GPU vs CPU reference on the published pack | mean 0.07 codes, 0 pixels > 24 | mean 0.08 codes, 1 pixel > 24 |
| Far-region pixels changed after local motion | 0 | 0 |

Dynamic 1280×720 run: 10 s of continuous motion plus ±3-code sensor noise, at
30 fps, with default settings (`output/mosaic_fx/obs/dynamic_1280.txt`):

| Count | Value |
|---|---|
| Analysis samples | 120 |
| Wakes | 119 |
| Meyer + re-tile passes | 60 (6 Hz cap) |
| Wakes deferred to colors only | 59 |
| Locate passes | 59 (geometry only) |
| Render-thread CPU per frame | 0.47 ms |
| Render + readback per frame | 3.2 ms (unfiltered 1.6 ms) |
| Meyer per re-tile (4 threads, loaded machine) | 51 ms |

### Core (M4 Mini, cat image, 960-px analysis, 15k tiles)

| Path | First plugin build | Now |
|---|---|---|
| Static frame: prefilter + colors + pack | 2.0 ms | 1.0 ms |
| Structure-unchanged frame | 4.5 ms | 0.5 ms |
| Incremental frame, 5–10% of blocks dirty (excludes Meyer) | ~30 ms (full frame) | 4.9–6.5 ms |
| Full rebuild (excludes Meyer) | 29 ms | 21 ms |
| Persistent core memory per analysis pixel | ~100 B | 22 B |
| Meyer cartoon, 512 lattice | 31 ms | 31 ms (unchanged) |

Operation-level changes:

- **Oklab cube root:** a Halley step (one divide) replaces two Newton steps.
- **Entropy:** a histogram with a c·log c table replaces sorting the ids.
- **Gradients:** NMS runs on squared magnitude, with the sqrt taken only at
  accepted edges.
- **Orientation:** the doubled angle comes from squared offsets, with no sqrt
  or divide. The half angle is solved only at tile centres.
- **Blurs:** separable everywhere.
- **Distance transform:** row-major column-site sweeps.
- **Nucleation:** a counting sort replaces the stable sort, and its exclusion
  mask is bbox-local.
- **Meyer wrapper:** cached resampling maps, with the clamp fused into the
  output pass. The structure hands off by swap.

Receipts: `output/mosaic_fx/cat_bench.json` and `cat_pan.jsonl`.

## Limits and next steps

- The tested plugin build is for macOS (Metal, OBS 32.2.1).
- Under continuous motion, Meyer at the re-tiling rate is most of the worker
  cost. *Re-tiling passes per second* and *Analysis samples per second* bound
  it. A cropped Meyer window around dirty blocks would make that cost scale
  with the moving area, but it is not implemented.
- The edge threshold depends on the lattice: the 512-lattice cartoon gives
  about half the edge pixels of the 1024 one.
- The entropy thresholds are absolute, so a uniformly detailed scene uses
  mostly the small tile sizes.
- The texture/caustic field from the Meyer split is not yet used to guide
  orientation or spacing. It is a candidate cue for relaxing rows around
  objects and reducing clumping.
