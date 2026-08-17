# Manual JPEG optimizer

This experiment is both an inspectable JPEG forward model and a measured
rate–distortion search. It writes ordinary, standards-compatible JPEG files;
no custom decoder or hidden region map is required.

## The five visible stages

1. RGB to full-resolution YCbCr.
2. 4:4:4, 4:2:2, or 4:2:0 chroma sampling and reconstruction.
3. Level shift and separable 8×8 DCT (log-magnitude coefficient cascade).
4. Quality-scaled luma/chroma quantization (signed coefficient view).
5. Zig-zag ordering and a blockwise run/category entropy-cost view.

The extra tabs expose the v3-inspired half-scale cartoon, exact residual
texture, connected phase/signature regions, and the pixels sent to libjpeg.

## Structural experiment

The analysis borrows three ideas from the repository rather than importing the
large v3 runtime wholesale:

- **cartoon/texture:** a low-pass cartoon owns the smooth geometry and the exact
  residual is texture;
- **v3 quotient:** block cells are bloom-filled only across compatible texture
  class and chroma-DCT principal-axis signatures;
- **SVG v2 rate distortion:** candidate simplifications are accepted by actual
  encoded bytes and decoded error, not coefficient counts or a rate proxy.

Within each connected region the Cb/Cr texture covariance supplies an aligned
basis. The weak (minor) direction may be smoothed, then the basis is rotated
back before JPEG encoding. A global phase offset lets the search test nearby
directions. This is a distortion-controlled preprocessing operation: an
arbitrary spatial rotation would otherwise alter color or require side
information that baseline JPEG cannot carry.

## CLI

From the repository root, with the optional dependencies installed:

```sh
python3 -m experiments.manual_jpeg_optimizer analyze input.jpg --out stages

python3 -m experiments.manual_jpeg_optimizer optimize input.jpg output.jpg \
  --target-bytes 29200 --report output.json

python3 -m experiments.manual_jpeg_optimizer compare input.jpg tinypng-output.jpg

python3 -m experiments.manual_jpeg_optimizer ownership-relax input.jpg output.jpg \
  --rate-lambda 0 --report output.json

python3 -m experiments.manual_jpeg_optimizer spatial-dct-relax input.jpg output.jpg \
  --transport-lambda 0.009 --frequency-weight 0.3 \
  --chroma-projection 0.35 \
  --luma-mobility 0.15 --cb-mobility 1.75 --cr-mobility 1.75 \
  --report output.json

.venv-jpeg/bin/python -m experiments.manual_jpeg_optimizer jpegli-fuse \
  city_image.jpg city_fused.jpg

.venv-jpeg/bin/python -m experiments.manual_jpeg_optimizer.verify_city_win

python3 -m experiments.manual_jpeg_optimizer gui
```

Add `--exhaustive` for the larger phase/projection sweep. The report contains
the winning parameters and the SSIM/PSNR/edge-PSNR Pareto frontier measured
against the source JPEG's decoded pixels.

## GUI

The Dear PyGui application exposes every manual control, stage/overlay tabs,
the same optimizer used by the CLI, and direct fused-Jpegli save controls for
the ownership atlas, transport witness, terminal trellis, channel allocation,
and quantization tilt. Install with:

```sh
python3 -m pip install -e '.[jpeg-lab]'
```

On a PEP 668 managed Python, create a virtual environment first. The local
checkout used for GUI validation has `.venv-jpeg/` (gitignored):

```sh
python3 -m venv .venv-jpeg
.venv-jpeg/bin/python -m pip install Pillow scipy dearpygui
.venv-jpeg/bin/python -m experiments.manual_jpeg_optimizer gui
```

Size alone is not a quality victory. To compare against TinyPNG fairly, save
its output and run the same decoded-reference metrics against it; the source
file in this experiment is already lossy, so no-reference visual judgment is
still required around whiskers, eye rings, and blanket fibers.

For the city holdout, the default `jpegli-fuse` configuration reproducibly
writes 217,112 bytes versus TinyPNG's 217,219, while also improving the shared
decoded-reference metrics: SSIM 0.965969 versus 0.953066, PSNR 38.426 versus
35.665 dB, and Sobel edge PSNR 28.235 versus 27.891 dB. The verifier fails
unless all four inequalities hold.

## Conserved-constituent ownership relaxation

`ownership-relax` is the chip-inspired route, and is deliberately distinct
from the spectral and SDP marginal diagnostics.  Each non-DC block/mode
three-vector is an owned atom.  A maximum-confidence predecessor forest
transports its sign phase without losing the inverse path.  Unstable cells
bifurcate along their Courant--Fischer principal direction at the exact
weighted-median mass boundary.

At a leaf, an orthogonal Schur--Horn frame decomposes every atom into three
constituents.  Their vector sum is the input atom up to roundoff; no energy is
deleted or shrunk.  The admissible routes are identity quantization or
independent constituent quantization followed by summation.  Bellman recursion
globally minimizes the stated rate--distortion--branch functional over both
routes and every pruning of the generated causal tree.  The JSON report
records the composition residual, ownership and predecessor hashes, selected
leaf counts, channel utilization, and number of changed quantized values.

The word *global* has a precise boundary here: it covers the full relaxation
on the stored causal tree, not all possible trees and not libjpeg's discrete
Huffman symbol problem.  At zero transport pressure, nearest-lattice rounding
proves identity is the global optimum.  Positive pressure reveals the
ownership bifurcation at which a coherently routed subtree becomes preferable.

## Spatial/frequency DCT ownership

`spatial-dct-relax` performs redistribution where JPEG can see it: on the
product graph of block position and non-DC DCT frequency. Each node has
separate Y, Cb, and Cr ownership. Spatial edges connect the same frequency in
adjacent blocks; frequency edges connect adjacent zig-zag positions inside a
block, allowing phase frustration to leave a spatial cell without forcing
luma blur.

Every coefficient is split into positive and negative constituent mass and
the two signs are transported independently. For each sign and channel, the
program globally solves

```text
min_z  1/2 ||z-m||^2 + lambda/2 z^T L z.
```

The unique solution `(I + lambda L)^-1 m` is nonnegative and conserves total
mass because `L 1 = 0`. Its edge-flow divergence equals the coefficient
displacement, so ownership and its inverse accounting are explicit. Y/Cb/Cr
mobility is anisotropic: luma can stay pinned while chroma absorbs
frustration, but all three remain available rather than collapsing into one
carrier. JPEG quantization is applied only after this certified continuous
redistribution and is evaluated by actual bytes and decoded fidelity.
