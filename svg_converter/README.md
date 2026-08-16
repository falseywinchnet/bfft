# Transport-Locked Vectorizer

A standalone PNG-to-SVG research converter built around a topology-first,
hierarchical color basis. It requires Python 3.10+, NumPy, SciPy, and Pillow;
it does not import the surrounding BFFT repository.

## Run

```sh
python -m tlvector input.png output.svg --colors 12 --detail-colors 6
```

From this checkout without installing:

```sh
PYTHONPATH=svg_converter/src \
python3 -m tlvector input.png output.svg
```

Useful controls:

- `--colors`: structural palette/owner budget.
- `--detail-colors`: parent-locked residual child budget.
- `--coarse-side`: maximum dimension of the structural solve.
- `--minimum-region`: minimum residual island area.
- `--simplify`: maximum contour simplification distance in pixels.
- `--curve-tolerance`: maximum accepted quadratic rounding displacement.
- `--seam-overlap`: same-color under-stroke width used to close antialiasing
  cracks between abutting paths; `0` disables it.
- `--no-trim`: preserve fully transparent outer rows and columns.
- `--alpha-mode auto|cutout|preserve`: distinguish raster edge coverage from
  genuinely translucent regions. Auto is appropriate for most logos.

The CLI prints JSON diagnostics. `--diagnostics report.json` saves them.

## Run the web GUI

From the repository root, create the environment once and install the
converter:

```sh
cd svg_converter
python3 -m venv .venv
.venv/bin/pip install -e .
```

Start the local web application:

```sh
.venv/bin/tlvector-gui
```

It opens the interface in the default browser and prints its local URL in the
terminal. Choose a PNG, set the structural palette to as many as 128 colors,
select **Convert**, inspect the rendered SVG, and select **Export SVG**. Press
`Ctrl-C` in the terminal to stop the server.

To use a predictable port without automatically opening a browser:

```sh
.venv/bin/tlvector-gui --port 8765 --no-browser
```

Then visit `http://127.0.0.1:8765/`. On macOS, `launch_gui.command` can also be
double-clicked in Finder; it uses the installed environment when present and
otherwise starts from the source checkout. A no-install terminal launch is:

```sh
PYTHONPATH=src python3 -m tlvector.web_gui
```

The web GUI previews the source and actual rendered SVG and exposes the
structural/detail, curve, subpixel-smoothing, transparency, and seam-overlap
controls. Its checkerboard makes real transparency distinguishable from white
seam artifacts.

The server binds only to `127.0.0.1`; image processing stays local and is not
uploaded. A native Tk alternative remains available as
`PYTHONPATH=src python3 -m tlvector.gui`.

## Performance

The high-color path uses chunked checkerboard assignment instead of allocating
a full height × width × colors cost volume. Residual children are evaluated
only inside their structural parent, and SVG tracing scans each color's
occupied bounds. This makes 128 structural colors practical while preserving
the converter's exact deterministic result.

On the included development machine, the supplied 860×758 portrait at 128
structural + 6 detail colors completes in about 1.33 seconds. A 2× resize
(1720×1516) completes in about 4.04 seconds. Timing varies with contour
complexity as well as pixel count; at larger sizes, SVG contour compilation is
usually the dominant phase.

Run repeatable local trials with:

```sh
python3 tools/benchmark_speed.py input.png --colors 64 128 --detail-colors 6 \
  --repeats 3
```

CLI diagnostics and both GUIs report total conversion time. The JSON report
also separates preprocessing, palette, regularization, detail, color-reduction,
and SVG-compilation timings.

## Reproduce the included study

```sh
python3 tools/make_examples.py
PYTHONPATH=src python3 -m tlvector examples/input/geometric_badge.png \
  examples/output/geometric_badge.svg --colors 8 --detail-colors 4
PYTHONPATH=src python3 -m tlvector examples/input/translucent_ribbons.png \
  examples/output/translucent_ribbons.svg --colors 14 --detail-colors 6
PYTHONPATH=src python3 -m tlvector examples/input/cameraman_source.png \
  examples/output/cameraman_source.svg --colors 12 --detail-colors 6
```

See `METHOD.md` for the algorithmic lineage and the points where this project
deliberately departs from both source systems.
