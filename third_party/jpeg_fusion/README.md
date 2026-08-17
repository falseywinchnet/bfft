# JPEG ownership fusion

Jpegli is the primary encoder. MozJPEG is a reference implementation for its
block trellis state model; this project does not combine both complete
encoders. Exact source revisions are recorded in `SOURCE_LOCK.json`.
The integration is stored as `jpegli-ownership-fusion.patch` rather than as
vendored upstream trees. `build_backends.sh` clones the locked revisions on
first use, applies the patch idempotently, and builds both tools.

The first fusion boundary is implemented on branch
`codex/ownership-fusion` inside `jpegli/`:

- `jpegli_set_dead_zone_provider` supplies 64 nonnegative threshold offsets
  for every component block;
- both the streaming quantizer and final requantization/PSNR path consume the
  field;
- `cjpegli --dead_zone_field FILE` loads a luma-grid ownership atlas; and
- 4:2:0/4:2:2 chroma blocks receive the mean of their owned luma footprint.
- `ownership_trellis.cc` stores Jpegli's floating DCT in quantized units and
  solves a block-global dynamic program over zig-zag position, zero-run state,
  ZRL, EOB, and every legal magnitude category;
- `--trellis_lambda`, `--ownership_weight`, and `--trellis_edge_weight`
  control the terminal rate, constituent-retention, and edge-frequency costs;
  the encoder reports before/after nonzeros, estimated symbol bits, and the
  terminal objective.
- the encoder tokenizes once, optimizes image-specific Huffman tables, runs
  the terminal trellis against those code lengths, discards the stale tokens,
  and retokenizes the changed coefficient lattice;
- `--quant_luma_tilt` and `--quant_chroma_tilt` redistribute quantizer volume
  across DCT radius. Negative luma tilt preserves the low/mid-frequency
  structure that carries the city illustration's drawn boundaries.

The binary `JLDZ` format is a 16-byte little-endian header (`JLDZ`, version 1,
block width, block height) followed by float32 values with shape
`(3, height, width, 64)` in natural DCT order. Generate one with:

```sh
.venv-jpeg/bin/python -m experiments.manual_jpeg_optimizer make-jldz \
  city_image.jpg /tmp/city.jldz --regions 256 --strength 0.1
```

The region generator inherits the PNG-to-SVG quotient discipline: immutable
parent ownership, globally competing feature-SSE splits, weighted-median
bifurcation, and explicit lineage. At the winning q72 setting,
`city_image.jpg` has 256 leaves containing 50–150 blocks with block-count CV
0.140; the old signature flood produced
11,558 leaves, mostly singleton blocks.

## Trellis seam

MozJPEG's `quantize_trellis` in `jcdctmgr.c` supplies the useful representation
idea: dynamic-programming state over zig-zag position, zero run, EOB, category,
and DC predecessor. The first Jpegli adaptation now implements the AC
run/EOB/category state and adds an ownership shadow price to every zero
transition. It uses the first-pass image-optimized Huffman depths and safely
retokenizes after coefficient changes. Progressive scan partitioning means
its reported bit count remains a terminal rate model rather than an identity
for final file bytes; the measured encoder output closes that outer loop.
The continuous spatial/channel/frequency transport remains the controlling
problem; the trellis is its terminal projection onto legal JPEG coefficients.
Copying MozJPEG's whole quantizer would discard Jpegli's floating DCT,
frequency-dependent matrices, separate chroma matrices, and adaptive
quantization, so it is intentionally not the architecture.

## Current city control

At approximately 217.2 KB, decoded against `city_image.jpg`:

| encoder | bytes | SSIM | PSNR | edge PSNR |
|---|---:|---:|---:|---:|
| TinyPNG control | 217,219 | 0.953066 | 35.665 | 27.891 |
| MozJPEG q76 trellis | 215,693 | 0.957827 | 36.732 | 27.025 |
| stock Jpegli target | 217,413 | 0.963787 | 37.888 | 27.335 |
| Jpegli + balanced field 0.25, edge gate 2 | 217,097 | 0.962708 | 37.778 | 27.371 |
| Jpegli + terminal trellis q72 | **217,166** | **0.966374** | **38.503** | 27.382 |
| Full transport + ownership trellis + tilted quantization | **217,112** | **0.965969** | **38.426** | **28.235** |

The full result is 107 bytes smaller than the TinyPNG control and higher on all
three measured fidelity axes: +0.01290 SSIM, +2.76 dB PSNR, and +0.34 dB Sobel
edge PSNR. Its transport solve has a 5.74e-11 KKT/flow-divergence residual and
separately conserves positive and negative coefficient mass.

The dead-zone hook is proven real: an all-zero field is byte-identical to
stock Jpegli, while a uniform 0.25 AC field changes coefficient decisions and
removes 28 KB at fixed q72. The current heuristic field does not yet beat stock
Jpegli at equal rate, so it is infrastructure and a negative control—not a
claimed win by itself.

Dogfood the fused path from Python with:

```sh
.venv-jpeg/bin/python -m experiments.manual_jpeg_optimizer jpegli-fuse \
  city_image.jpg city_fused.jpg
```

That command's defaults reproduce the measured full city configuration:
q72, 256 balanced regions, transport lambda 0.002, trellis lambda 0.0695,
ownership weight 0.05, edge weight 1, and luma quantization tilt -0.5.
