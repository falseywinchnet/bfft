# Converter v2 study results

## Test image

All full-image figures below use the 1714x823 city illustration supplied for
the MSE-30 study. MSE is measured on the 0-255 RGBA scale. Because the source
is opaque, an RGB-only MSE is multiplied by 3/4 when reported as its RGBA
equivalent.

## Results

| Method | RGBA MSE | Loops / cells | SVG paths | SVG bytes | SVGZ bytes | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 adaptive exact lattice | 29.8358 | 179,415 loops | 146 | 19,816,417 | 2,914,153 | about 21.7 s |
| V2 compact + error/byte merge | 29.9902 | 34,479 loops | 217 | 1,466,917 | 410,703 | 11.21 s |
| V2 v3-cell rank-one affine prototype | 45.7080 | 29,529 cells / 39,421 loops | 29,529 | 9,017,389 | 1,642,890 | 53.25 s |
| Native segmenting-v3 reconstruction | 5.0348 | 29,529 texture cells | not SVG | not applicable | not applicable | included above |

SVGZ sizes use deterministic gzip level 9. For comparison, the V1 gzip size
was measured from the existing SVG with the same compression level; V1 did
not itself expose SVGZ output.

## What changed

The compact method first builds an MSE-19.8969 occupation, then spends the
quality headroom by greedily absorbing small adjacent components. A candidate
is ranked by exact fixed-palette RGBA SSE increase divided by its estimated
path-byte saving. Palette colors are refit after each round and the measured
final image is rejected if it exceeds the requested MSE ceiling.

On the test image it accepted 89 adaptive splits and 152,096 component
merges. The component count fell from 199,697 to 34,295. The serialized path
data uses exact integer-lattice geometry, removes collinear vertices, and
chooses compact absolute or relative `H`, `V`, and `L` commands. Large color
regions are written first, retaining the useful coarse-to-fine loading
behavior observed in the V1 artifact.

The net result at essentially the same error is:

- 92.6% fewer raw SVG bytes;
- 85.9% fewer compressed bytes;
- 80.8% fewer boundary loops;
- about 1.9x faster end-to-end conversion in this run.

## Affine-cell prototype

The affine experiment runs the real GUI-default v3 pipeline: canonical-v2
structural topology, nested texture cells, and the jump-measure Meyer
operator. It fits a least-squares 2D RGB affine field per final texture cell,
then takes the best rank-one factor so that the field is representable by one
SVG `linearGradient`.

Flat colors on those same v3 cells produce RGBA-equivalent MSE 119.8132. One
linear gradient per cell lowers it to 45.7080, a large improvement, but does
not reach 30. The native v3 model reaches 5.0348 because it retains a richer
combination of affine and paired one-sided ridge fields. Reaching MSE 30 with
cell painting therefore needs at least a second spatial term or an SVG-native
approximation to those ridge fields; additional path-serialization tweaks
cannot close that modeling gap.

The affine SVG uses `shape-rendering="crispEdges"` because its cells form a
partition. Without it, browser antialiasing each touching path independently
produced visible pale hairlines even though the underlying regions met
exactly.

## Reproduction

Compact conversion:

```sh
cd svg_converter
.venv/bin/tlvector-v2 input.png output.svg \
  --colors 128 \
  --split-budget 128 \
  --split-target-mse 20 \
  --target-mse 30 \
  --merge-maximum-area 32 \
  --merge-rounds 2 \
  --also-svgz \
  --diagnostics output.json
```

Affine prototype, from the BFFT repository root with a native library that
contains the Meyer and vision translation units:

```sh
BFFT_LIBRARY=build/libbfft-full.so \
PYTHONPATH=svg_converter_v2/src:svg_converter/src \
svg_converter/.venv/bin/tlvector-v2-affine input.png affine.svg \
  --also-svgz \
  --diagnostics affine.json
```
