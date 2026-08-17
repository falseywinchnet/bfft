# Posterizer

Posterizer is an experimental sibling of `svg_converter_v2`. It inherits the
compact exact-lattice path compiler, global same-color even-odd paths,
deterministic SVGZ output, transparency handling, and local web workflow. It
does not change converter V1 or V2.

Its purpose differs from the fidelity converter: Posterizer deliberately
constructs a small, expressive color basis. The default palette is a binary
tree over the occupied OKLCH color volume rather than a uniform distribution
around the image mean.

## Method

For every current palette leaf, Posterizer builds local coordinates for
lightness, chroma, circular hue displacement, and alpha. It evaluates several
deterministic initial cuts, refines each with two-means, and measures the
reduction in within-node perceptual distortion. The globally best proposal is
accepted and the process repeats until the requested palette size is reached.

The two child color nodes move to the centroids of their assigned populations.
`node_separation` then scales each final child displacement relative to its
parent:

```text
display child = parent + separation × (occupied child mean − parent)
```

`1.0` is the measured centroid. Values slightly above one create stronger
poster separation. Out-of-gamut nodes are mapped by reducing OKLCH chroma
while retaining lightness and hue.

The cylindrical distance is hue-wrap safe. With unit lightness, chroma, and
hue weights it is algebraically the ordinary OKLab squared distance, while
separate weights allow an artist to allocate more palette capacity to hue,
chroma, or illumination.

## Install

Use the existing converter environment and install all three sibling projects:

```sh
cd svg_converter
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e ../svg_converter_v2
.venv/bin/python -m pip install -e ../posterizer
```

## Web interface

```sh
cd svg_converter
.venv/bin/posterizer-gui
```

The GUI previews the actual compact SVG, shows its palette, and exports PNG,
SVG, and SVGZ. It also retains the inherited mean-palette method as an A/B
control. The server binds only to `127.0.0.1`; inputs remain local.

For a fixed port without opening a browser:

```sh
.venv/bin/posterizer-gui --port 8767 --no-browser
```

On macOS, `posterizer/launch_gui.command` may be double-clicked.

## Command line

```sh
.venv/bin/posterizer input.png output.svg \
  --colors 20 \
  --node-separation 1.08 \
  --minimum-island 6 \
  --cleanup-rounds 1 \
  --also-png \
  --also-svgz \
  --diagnostics output.json
```

Use `--method inherited` for the inherited palette control. Adjust
`--lightness-weight`, `--chroma-weight`, and `--hue-weight` to change which
perceptual distinctions receive palette leaves.

## Initial portrait result

At 20 colors on the 400×400 development portrait, using separation 1.12:

| Method | Perceptual RMSE | RGBA MSE | Loops | SVGZ | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| OKLCH bifurcation | 0.02558 | 44.04 | 1,362 | 26,835 bytes | 4.96 s |
| Inherited palette | 0.02721 | 50.62 | 1,196 | 24,506 bytes | 0.59 s |

The perceptual tree currently spends additional compute to test and refine
several cuts per leaf. This is a quality-first prototype; batching proposals
and histogramming occupied OKLCH cells are the clearest future speedups.
