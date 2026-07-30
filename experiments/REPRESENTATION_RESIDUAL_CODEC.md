# Transport representation plus exact correction experiment

## Question

Can the canonical transport-cell reconstruction act as a deterministic predictor,
with the source recovered exactly from a separately compressed correction?

```text
source = render(geometry) + exact correction
```

The experiment is implemented in
`experiments/representation_residual_codec.py`. Every reported packet has a
decoder, and every combined row is verified byte-for-byte against the source.

## Codecs tested

1. **PNG-style raster prediction**
   chooses None/Sub/Up/Average/Paeth per row and DEFLATEs the result.
2. **Standard optimized PNG**
   provides an external lossless baseline.
3. **Finite 2-D predictor bank**
   chooses order-0, horizontal order-1/2, vertical order-1/2, gradient, or
   Paeth per 32x32 tile and channel.
4. **Cell-ordered correction**
   uses the canonical label map as free side information, walks each cell in a
   deterministic snake order, and chooses order-0/1/2 per cell and channel.
5. **JPEG-style separable DCT**
   uses an 8x8 orthonormal DCT, JPEG luminance/chroma quantization, zigzag,
   DC differencing, zero runs, signed varints, and DEFLATE.
6. **Residual DCT plus exact tail**
   treats the geometric reconstruction as already available, DCT-codes a coarse
   signed residual, and PNG-codes the remaining exact correction.

The DCT packets are lossy by themselves. Rows ending in `+tail_png` or
`+modular_png` are nevertheless exactly lossless because the decoder applies
the correction afterward.

## Results at longest side 384

All sizes below are complete coded sizes except the geometry-base rows, which
explicitly exclude the not-yet-serialized geometric representation `G`.

| Image | Cells | Best direct lossless | Best correction given `G` | Available budget for `G` |
|---|---:|---:|---:|---:|
| Pikachu | 821 | 73,876 B | 79,447 B, cell ordered | **−5,571 B** |
| Cameraman | 4,415 | 112,423 B | 91,985 B, cell ordered | **20,438 B** |
| Coffee | 2,425 | 184,025 B | 181,399 B, residual DCT q25 | **2,626 B** |

The budget is the hard codec criterion:

```text
budget(G) = best_direct_lossless_size - correction_size_given_G
```

Positive correction savings are not yet a complete codec win. The serialized
centers, topology and reconstruction coefficients must all fit inside that
budget.

## What fell out

### The separability hypothesis is real but insufficient

At DCT quality 25:

| Image | Source DCT packet | Support DCT packet | Support saving |
|---|---:|---:|---:|
| Pikachu | 8,410 B | 8,366 B | 0.5% |
| Cameraman | 9,459 B | 8,183 B | 13.5% |
| Coffee | 11,005 B | 9,580 B | 13.0% |

The reconstruction is measurably more DCT-separable on natural images. However,
storing the rendered support as another raster duplicates information. Its
larger exact correction overwhelms the DCT saving, so
`support_dct + correction` loses to direct coding in every tested image.

### Voronoi ownership can help the correction

For Cameraman, raster correction was 106,383 bytes while cell-ordered correction
was 91,985 bytes. Canonical ownership removed 14,398 bytes without transmitting
an ordering or iterating.

This did not generalize uniformly:

- Coffee's cell ordering was slightly worse than residual DCT.
- Astronaut at 256 squared was slightly worse than direct lossless.
- Pikachu remained worse even before charging a single byte for geometry.

Cell ID is therefore valuable context, not the correction's universal primary
coordinate.

### Pikachu is the decisive failure case

Pikachu's source consists of large exact flat colors and crisp contours. Direct
PNG already represents that structure almost ideally. The transport
reconstruction moves or softens contours, so the correction contains paired
edge errors that are harder to code than the original boundaries.

For this class, improving PSNR is not enough. Boundary locations must be
reproduced exactly or represented as an explicit sparse contour-displacement
channel.

### The current viewer state is not yet a portable representation

The pipeline returns centers, labels, a rendered RGB field and reconstruction
diagnostics. It does not return the fitted affine/ridge coefficients required to
render the same support independently. Consequently this experiment measures
the byte budget available to `G`; it does not pretend that `G` currently costs
zero.

Cameraman's 20,438-byte budget is only 4.63 bytes per cell. That is too small for
naively storing two coordinates plus affine/ridge appearance coefficients for
4,415 cells. A viable representation requires strong differential/topological
coding or a much smaller support population.

## Interpretation

The useful architecture is not:

```text
compressed support raster + compressed residual
```

That duplicates information and consistently loses. The plausible architecture
is:

```text
compact deterministic geometry G
  + geometry-conditioned correction
  + optional separable low-frequency residual
  + exact tail
```

The first experiment establishes three concrete facts:

1. the rendered support is more separable on natural images;
2. canonical ownership sometimes reduces correction entropy substantially;
3. present reconstruction geometry is not compact enough, and boundary error is
   the dominant failure on graphic art.

That first result motivated a serializable renderer state with:

- differential site coordinates;
- topology derived rather than stored as a label raster;
- quantized per-cell affine/ridge coefficients;
- a sparse boundary-displacement correction before texture/noise coding;
- predictor choice between raster, cell order and DCT residual by measured byte
  count.

## Complete compact geometry result

The follow-up implementation adds an actual portable geometry packet (`RCGT`):

```text
header
zlib {
    uint16 site_x, site_y
    uint8  mean_r, mean_g, mean_b
} repeated
```

The encoder now performs a finite search over small site counts and three
deterministic subsets of canonical sites:

- largest owned area;
- equal population quantiles in raster order;
- population-weighted farthest-point coverage.

The selected rule is not transmitted: the resulting coordinates already fully
specify the packet. Decode does not store or require the canonical label raster.
It regenerates a Euclidean partition using integer squared distances and
lowest-site-ID tie breaking. The correction packet (`RCEL`) then:

1. derives the same labels;
2. walks pixels in deterministic cell/snake order;
3. selects order-0, order-1, or order-2 per cell and channel;
4. stores signed residual varints followed by DEFLATE.

At longest side 384:

| Image | Best direct | Compact `G` | Exact cell residual | Complete total | Change |
|---|---:|---:|---:|---:|---:|
| Pikachu | 73,876 B | 138 B / 16 mass sites | 84,882 B | 85,020 B | +11,144 B |
| Cameraman | 112,423 B | 108 B / 12 farthest sites | 91,435 B | **91,543 B** | **−20,880 B** |
| Coffee | 184,025 B | 51 B / 4 farthest sites | 198,798 B | 198,849 B | +14,824 B |

Cameraman is therefore a complete lossless win of 18.6%, including geometry.
The saved packet pair independently decodes back to the exact source:

```sh
.venv/bin/python experiments/representation_residual_codec.py \
  --decode-compact \
  experiments/out/representation_residual_codec_selectors_384/camera.rcgt \
  experiments/out/representation_residual_codec_selectors_384/camera.rcel \
  --decode-output camera-decoded.png
```

The surprising part is that the 12-cell base is not a high-quality visual
reconstruction. It is a crude piecewise-constant image. Its value is as a
coordinate system: it groups the source into long locally predictable sequences.
The compression gain comes more from geometry-conditioned ordering than from
small pixel error.

This also gives the correct meta-codec behavior. Select direct lossless for
Pikachu and Coffee and compact geometry for Cameraman. Across these three images,
that finite selection saves 20,880 bytes compared with choosing the best direct
coder independently for every image.

Two richer finite models were also rejected:

- a per-cell two-sided ridge with 16 possible normals, quantized offset and two
  fitted RGB states improved Cameraman only marginally and made Pikachu worse;
- a cell-native 2-D residual predictor bank improved an inferior 64-site
  Cameraman candidate by only 511 bytes, while losing at the overall optimum.

The cheap site subset changed the rate more than either richer appearance model.
For this stream, selecting the coordinate system is the high-leverage operator.

The exhaustive 36-candidate rate sweep is deliberately an offline experiment.
Once the Cameraman winner is fixed, the unoptimized Python path measures about
12 ms for geometry fitting/serialization, 72 ms for `RCEL` encoding, and
142 ms for an exact decode at 384 squared. The weighted farthest scan itself
selects only 12 sites; it is not an image-scale relaxation.

## Rejected contour displacement

Two boundary operators were tested before retaining the compact result:

- splitting exact residual samples into a geometry-derived boundary band and an
  interior stream;
- matching strong support gradients to nearby source gradients and transmitting
  sparse normal shifts, followed by a deterministic local warp.

The boundary band saved only about 0.6 KB on Pikachu and lost on the natural
images. The normal-shift warp increased every correction stream. The reason is
visible in the residual: soft support changes boundary location **and** the
values on both sides. It is not a pure geometric displacement, and a warp
spreads incorrect side colors.

An eventual contour channel must therefore transmit a two-sided edge state
(location plus left/right appearance), not displacement alone.

## Reproduce

```sh
.venv/bin/python experiments/representation_residual_codec.py \
  --gallery pikachu camera coffee \
  --work-side 384 \
  --allocation-side 384 \
  --quality 25 50 75 \
  --output experiments/out/representation_residual_codec_selectors_384
```

Artifacts:

- `experiments/out/representation_residual_codec_selectors_384/rates.csv`
- `experiments/out/representation_residual_codec_selectors_384/summary.json`
- one diagnostic PNG per image
- `*.rcgt` compact geometry packets
- `*.rcel` exact correction packets
- `*-compact.png` views of the actually decoded compact base
