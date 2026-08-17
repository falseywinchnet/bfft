# Posterizer

Posterizer is a raster-image sibling of the converter experiments. It reuses
their deterministic perceptual palette ideas, transparency handling, spatial
component cleanup, and local web workflow, but its artifact is deliberately a
posterized raster—not SVG. PNG inputs produce PNG results; JPEG inputs produce
JPEG results.

The default method builds a binary palette tree over the occupied OKLCH color
volume. Unlike ordinary population-weighted quantization, it also measures
local detail and tempers the influence of large color populations. A broad
flat wall therefore cannot consume the palette merely because it contains
more pixels than a face.

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
input raster format, and exposes these primary controls:

- **Colors**: visible palette size.
- **Node separation**: exaggeration of each final child displacement from its
  bifurcation parent; `1.0` is the measured centroid.
- **Detail priority**: extra importance assigned to local OKLab gradients and
  multiscale contrast.
- **Area exponent**: how strongly raw pixel population influences palette
  allocation. `1.0` is ordinary population weighting; values below one give
  rare occupied color cells more representation. The default is `0.65`.
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
per palette color. The optimization objective is unchanged, and the 4- and
128-color portrait reference outputs remained byte-identical in regression
checks.

As a representative stress check, a 1714×823 photograph at 128 colors completes
in about 3.3 seconds on the project M4 Mini, including saliency analysis,
palette construction, full-resolution assignment, cleanup, and diagnostics.
The 400×400 development portrait completes in about 1.1 seconds at 128 colors.
Exact time varies with image structure and machine.

## Four-color allocation check

On the 400×400 portrait used during development, ordinary population weighting
allocated two of four colors to large background fields. With detail priority
`2` and population exponent `0.65`, the background collapses to one principal
field and the recovered palette distinction moves into facial illumination
and foreground structure.

This intentionally raises ordinary all-pixel MSE: the method no longer claims
that a square metre of blank wall is more important than a pair of eyes. The
diagnostics therefore report both conventional perceptual RMSE and an
importance-weighted perceptual RMSE matching the allocation objective.

See [METHOD.md](METHOD.md) for the distance, weighting, split, and gamut-mapping
details.
