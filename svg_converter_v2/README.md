# Converter v2

Converter v2 is an experimental sibling of `svg_converter`, not a replacement
for it. Version 1 remains unchanged. V2 explores four compression mechanisms:

1. exact pixel-lattice paths serialized with compact `H`/`V` commands;
2. deterministic gzip-compressed `.svgz` output;
3. component merging ranked by added RGBA error per estimated SVG byte saved;
4. rank-one affine SVG gradients painted inside segmenting-v3 texture cells.

## Install

Install v1 first because v2 deliberately reuses its tested palette and
quality-occupation basis:

```sh
cd svg_converter
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -e ../svg_converter_v2
```

## Compact rate-distortion conversion

```sh
.venv/bin/tlvector-v2 input.png output.svg \
  --colors 128 \
  --split-target-mse 20 \
  --target-mse 30 \
  --merge-maximum-area 32 \
  --also-svgz \
  --diagnostics output-v2.json
```

The first MSE target creates quality headroom. The merger then spends that
headroom on absorbing small connected components, choosing candidates by
error increase per estimated path byte eliminated. It refuses any merge that
would cross the final target. An `.svgz` output may also be named directly.

## Run the web GUI

After performing the installation above, start the local browser application:

```sh
cd svg_converter
.venv/bin/tlvector-v2-gui
```

It opens a V2-specific interface with source and rendered-SVG previews. The
controls expose the structural palette, split headroom, final hard MSE limit,
and error-per-byte merge policy. Both SVG and deterministic SVGZ downloads are
available, and the result reports whether the SVGZ is smaller than the source
PNG. Press `Ctrl-C` in the terminal to stop it.

To use a fixed port without opening a browser automatically:

```sh
.venv/bin/tlvector-v2-gui --port 8766 --no-browser
```

Then visit `http://127.0.0.1:8766/`. The server binds only to localhost and
does not upload the image. On macOS, `svg_converter_v2/launch_gui.command` can
also be double-clicked. A no-install source launch from the repository root
is:

```sh
PYTHONPATH=svg_converter_v2/src:svg_converter/src \
python3 -m tlvector_v2.web_gui
```

## Segmenting-v3 affine-gradient prototype

This optional path requires the enclosing BFFT repository and its native
library. It runs the actual GUI-default canonical-v2/nested-texture segmenter,
fits the best rank-one RGB affine field in every final texture cell, and emits
one SVG linear gradient per non-flat cell:

```sh
BFFT_LIBRARY=build/libbfft.so \
.venv/bin/tlvector-v2-affine input.png affine.svg \
  --also-svgz --diagnostics affine.json
```

The diagnostics compare flat-cell MSE, rank-one affine MSE, and the richer
native v3 reconstruction. This is a prototype: v3's paired one-sided ridges
are not yet representable by a single SVG linear gradient.

The emitted SVG uses `shape-rendering="crispEdges"`. This is intentional: the
cell partition is exact, so independent browser antialiasing at two touching
cell boundaries can otherwise create pale one-pixel seams that are absent
from the model.

## Measured city-image result

On the 1714x823 city illustration used during development, the compact V2
method reached RGBA MSE 29.9902 with 34,479 loops. Its SVG is 1,466,917 bytes,
and the deterministic SVGZ is 410,703 bytes. The prior exact-quality V1 result
used 179,415 loops, 19,816,417 SVG bytes, and 2,914,153 gzip bytes at MSE
29.8358. See [STUDY_RESULTS.md](STUDY_RESULTS.md) for the complete comparison,
commands, timings, and interpretation.
