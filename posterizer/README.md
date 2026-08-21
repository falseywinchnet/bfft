# Posterizer

Posterizer is a raster-image sibling of the converter experiments. It reuses
their deterministic perceptual palette ideas, transparency handling, spatial
component cleanup, and local web workflow, but its artifact is deliberately a
posterized raster—not SVG. PNG inputs produce PNG results; JPEG inputs produce
JPEG results.

The default method builds a tonal palette tree over an edge-aware structural
OKLCH field, then reserves one node for the most useful underrepresented color
family. Unlike ordinary population-weighted quantization, it measures local
detail and tempers the influence of large color populations. A broad flat wall
therefore cannot consume the palette merely because it contains more pixels
than a face, while the reserved anchor prevents complete hue collapse.

## Install

Use the existing converter environment:

```sh
cd svg_converter
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e ../posterizer
```

Posterizer does not depend on converter V2 at runtime.

## Web interface

```sh
cd svg_converter
.venv/bin/posterizer-gui
```

The interface previews the result, shows the selected palette, preserves the
input raster format, and leads its diagnostics with multiscale OKLCH chart
stress, worst supported hue-sector alignment, and collapsed relation energy.
It exposes these primary controls:

- **Colors**: visible palette size.
- **Node separation**: exaggeration of each final child displacement from its
  bifurcation parent; `1.0` is the measured centroid.
- **Detail priority**: extra importance assigned to local OKLab gradients and
  multiscale contrast.
- **Area exponent**: how strongly raw pixel population influences palette
  allocation. `1.0` is ordinary population weighting; values below one give
  rare occupied color cells more representation. The default is `0.65`.
- **Color-family priority**: controls a second, chroma/hue-sensitive proposal
  tree. Its most important missing family contributes one anchor, replacing
  the tonal node whose loss costs least. Set this to `0` for the unmodified
  tonal tree.
- **Structure radius** and **edge threshold**: build local color consensus for
  palette learning. Neighbors cooperate only across soft perceptual
  differences, suppressing noise without softening final labels. Radius `0`
  disables this stage.
- **Texture priority**: amplifies measured local lightness residual before
  final palette assignment. This spends categorical boundaries on real hair,
  facial, fabric, and edge structure rather than adding synthetic dither to
  flat fields. `0` is ordinary nearest-color assignment.
- **Spatial mixing**: opt-in tone synthesis using only the selected display
  palette. Inside stable, low-gradient regions, each pixel may mix its base
  color with one of a few perceptually adjacent nodes. Segmented error
  diffusion realizes the mixture density without carrying error across a
  contour or palette-pair boundary. Start at `0.5`; `0` keeps the flat result.
- **Mix neighbors**: limits the partner search. Three gives useful tonal reach
  while avoiding the conspicuous specks produced by distant color pairs.
- **Lightness/chroma/hue weights**: perceptual allocation preferences.

The server binds only to `127.0.0.1`; images remain local. A fixed-port launch
without opening a browser is:

```sh
.venv/bin/posterizer-gui --port 8767 --no-browser
```

On macOS, `posterizer/launch_gui.command` may be double-clicked.

## Command line

The output suffix chooses and preserves the desired raster format:

```sh
.venv/bin/posterizer input.png output.png \
  --colors 8 \
  --node-separation 1.08 \
  --detail-priority 2 \
  --population-exponent 0.65 \
  --family-priority 1 \
  --structure-radius 2 \
  --structure-threshold 0.065 \
  --texture-priority 0.25 \
  --mixing-strength 0.5 \
  --mixing-neighbors 3 \
  --minimum-island 6 \
  --cleanup-rounds 1 \
  --diagnostics output.json

.venv/bin/posterizer input.jpg output.jpg --colors 8
```

Use `--method inherited` for the inherited palette control.

## Performance

Palette construction uses exact weighted one-dimensional split searches as
initializers, then refines only the distinct candidates. Pixel assignment uses
a matrix form of the same cylindrical OKLCH metric, and connected-component
cleanup is a single equal-neighbor graph pass rather than one full image scan
per palette color. After gamut mapping and 8-bit rounding, display-identical
nodes are collapsed and pixels are assigned again to the actual rendered
palette. Cleanup and diagnostics therefore measure the colors written to the
image.

The edge-aware stage scales with image area and the square of the selected
radius; the default radius `2` visits a 5×5 neighborhood. Palette construction
still runs on at most `sample_limit` structural samples, while assignment and
component cleanup remain linear in full-resolution pixels and local edges.
Spatial mixing adds a deterministic full-resolution serpentine pass when its
strength is above zero.

## Four-color allocation check

On the 400×400 portrait used during development, ordinary population weighting
allocated two of four colors to large background fields. With detail priority
`2` and population exponent `0.65`, the background collapses to one principal
field and the recovered palette distinction moves into facial illumination
and foreground structure.

This intentionally raises ordinary all-pixel MSE: the method no longer claims
that a square metre of blank wall is more important than a pair of eyes. The
diagnostics therefore report both conventional perceptual RMSE and an
importance-weighted perceptual RMSE matching the allocation objective. They
also report high-frequency texture correlation, chroma correlation, mean hue
alignment, and low-pass perceptual RMSE. The last measurement captures the
optical tone reconstructed by spatial mixing; the relational measurements
catch muddy or artificially noisy palettes that RMSE alone incorrectly
rewards.

The primary relational diagnostic treats the source and result as
corresponding charts in perceptual color space. It samples global pixel pairs
and spatial neighbors at radii 1–32 after a 1.5-pixel viewing filter, then
measures normalized distortion of their OKLab distances. Because OKLab is the
Cartesian form of OKLCH, this preserves circular hue geometry without a
singularity at zero chroma. Mean chart stress is reported together with the
worst materially occupied 30-degree source-hue sector so a small but important
color branch cannot disappear inside an excellent average.

See [METHOD.md](METHOD.md) for the distance, weighting, split, and gamut-mapping
details.
