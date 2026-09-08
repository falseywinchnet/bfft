# Canonical glyph and foreground-residual study

## Question

Can scan-specific binary glyph variation be removed from JBIG2, with a shared
eikonal/Meyer cartoon carrying topology and the discarded variation carried by
the MRC foreground texture?

The prototype combines four repository methods:

1. a signed-distance field is the eikonal/embossed glyph coordinate;
2. Fourier-circle energy supplies direction-neutral matching evidence, with a
   low-resolution spatial guard against fusing different glyphs;
3. the jump-measure Meyer split produces a cartoon topology estimate and an
   exact texture complement; and
4. the SVG converter's error-per-byte rule is applied by terminal-encoding
   every candidate representation, including individual glyph clusters.

## Exact ownership construction

Let `M` be the source one-bit mask, `F` the decoded foreground, and `B` the
enlarged decoded background. The source composite is

```text
I = M F + (1 - M) B.
```

Aligned occurrences in a glyph cluster are replaced by one common envelope
`C`. The envelope is the union of the aligned source occurrences, so `C >= M`.
Keeping the source foreground in hidden pixels and setting

```text
F' = I wherever C = 1
```

gives

```text
I = C F' + (1 - C) B
```

exactly before foreground coding. Tests enforce the superset and recomposition
invariants.

A second exact representation avoids recoding `F`. Let `A = C - M`. Paint the
source foreground through `C`, then paint the background again through `A`.
The correction erases false dark envelope pixels and reconstructs the source
composite exactly before codec loss.

## Page 30 measurements

Page 30 has a 1736x2943 mask with 226,663 foreground pixels (4.4365% coverage).
Its source mask and foreground occupy 17,240 and 4,525 bytes, respectively:
21,765 bytes total.

The conservative configuration clustered 123 of 797 eligible components into
21 classes. It added 7,154 opacity pixels and increased coverage to 4.5765%.

| Representation | Mask bytes | Other foreground/correction bytes | Total | Delta |
|---|---:|---:|---:|---:|
| Source MRC pair | 17,240 | 4,525 | 21,765 | - |
| Canonical mask + recoded foreground | 17,387 | 38,326 | 55,713 | +33,948 |
| Canonical mask + exact paper correction | 17,387 | 4,525 + 3,560 + 1,040 | 26,512 | +4,747 |

The direct foreground candidate uses JPEG 2000 rate 400. Its page-level decoded
metrics are ink RMSE 5.04 and full-composite PSNR 43.26 dB. Installing it in the
PDF increases the serialized file by 33,964 bytes, within 16 bytes of the
stream-level prediction.

At 360 dpi, Poppler/PyMuPDF comparison measures:

- full rendered PSNR: 43.28 dB;
- rendered edge PSNR: 40.77 dB;
- dark-pixel IoU: 0.98635;
- extracted text: unchanged.

Visual inspection shows no missing strokes or obvious halos, but this fidelity
costs 2.56 times the source page's mask-plus-foreground budget.

An initially loose spatial gate fused 763 components, doubled coverage to
9.58%, and reduced the generic mask to 13,410 bytes. The quality-passing
foreground then cost 510,872 bytes. Meyer texture gains of 0.75, 0.5, and 0.25
did not materially change JPEG 2000 bytes at matched rates and worsened both
ink and edge error. The unshrunk residual won every quality comparison.

## Terminal cluster screen

Every conservative page-30 glyph class was also tested independently. For
each class the experiment encoded both its canonical mask and its exact paper
correction mask. None won:

- best cluster delta: +204 bytes;
- largest tested cluster delta: +591 bytes;
- one four-member class was already identical and changed zero bytes.

This rules out an accidental bad combination of otherwise profitable local
clusters. Under this envelope representation, every individual exact change
costs more than it saves.

## Dense-page control

Page 100 starts with a 43,012-byte mask and 5,124-byte foreground. Conservative
canonicalization adds 55,069 opacity pixels. Its smallest tested direct pair is
56,916 bytes versus 48,136 bytes for the source, and it fails the configured
edge gate. Increased text density therefore does not reverse the result.

## Cross-page dictionaries

Pages 20-35 contain 16 compatible MRC pages.

Page-local canonical envelopes combined under one `jbig2enc` dictionary are
224,891 bytes larger than their 567,623-byte source mask-plus-foreground
budget. The page-local envelopes are not identical across pages, so most page
streams remain nearly generic.

A true global pass then clustered components across all 16 pages and applied
one common aligned envelope to every occurrence in a global glyph class. With
a strict spatial gate and at least 16 occurrences per class, it formed 83
classes containing 2,213 components. This reduced the loss to 123,904 bytes,
but did not win:

| Global representation component | Bytes |
|---|---:|
| Shared canonical dictionary and pages | 514,635 |
| Generic paper-correction masks | 80,947 |
| Unchanged source foregrounds | 76,652 |
| Repeated low-resolution backgrounds | 19,293 |
| Candidate total | 691,527 |
| Source mask-plus-foreground total | 567,623 |

The global construction helps, but only a small fraction of components enter
strict global classes. Unchanged components continue to dominate the page
streams, while broader classes increase the correction channel.

## Global core and exact XOR refinement

The inverse construction uses a shared glyph **core**, not a superset. Two
core definitions were measured across pages 20-35:

- the aligned intersection, with missing ink encoded as an additive residual;
- the aligned median cartoon, with each occurrence encoded by an exact XOR
  refinement field.

The median form is lossless even when a cluster is imperfect: a bad match can
increase bytes, but it cannot alter the decoded glyph. Flattening the
refinements into page-sized generic masks lost badly because it destroyed the
source bitmap's useful causal neighborhoods. Cropping each exception helped,
but repeated rectangle metadata still dominated.

The final experimental stream instead stores one fixed correction canvas per
glyph class. It separates and independently compresses:

1. page dimensions and median glyph bitmaps;
2. page-grouped, delta-coded occurrence coordinates; and
3. aligned XOR fields in occurrence order.

The first two streams use Deflate and the XOR stream uses BZip2. A standalone
decoder reconstructs each refined glyph before page compositing. Tests cover
both direct serialization and lossless transcoding from the earlier fixed
atlas format.

Terminal frequency selection produced the following 16-page curve before the
final cropped-remainder pass:

| Retained classes | Occurrences | Fixed atlas + remainder | Delta from embedded masks |
|---:|---:|---:|---:|
| 1 | 199 | 493,099 | +2,128 |
| 2 | 360 | 493,915 | +2,944 |
| 4 | 670 | 496,734 | +5,763 |
| 8 | 1,095 | 500,665 | +9,694 |
| 16 | 1,693 | 508,250 | +17,279 |

The same crop and generic-template competition used by the safe PDF optimizer
then reduced the one-class JBIG2 remainder from 488,763 to 487,576 bytes. The
fully framed split atlas is 3,299 bytes:

| Exact one-class hybrid component | Bytes |
|---|---:|
| Cropped/template-selected JBIG2 remainders | 487,576 |
| Split median-glyph/XOR atlas | 3,299 |
| Candidate total | 490,875 |
| Existing embedded masks | 490,971 |
| **Measured delta** | **-96** |

All 16 reconstructed masks are pixel-identical. Adding the second-most
frequent class produces 485,560 remainder bytes plus a 5,544-byte atlas, or
491,104 bytes total: 133 bytes larger than the source. The terminal selector
therefore stops at one class.

This is a representation proof, not yet a portable PDF rewrite. Stock PDF
viewers do not implement the experimental `GCS1` filter, so installing it
would require either a standard-JBIG2 refinement encoder that matches these
bytes or a custom decoder. The experiment deliberately does not label an
unused attachment or a larger compatibility wrapper as a PDF optimization.

## Conclusion

The ownership transform is correct and decoder-compatible, but the tested
superset-envelope realization is not byte-efficient. It moves binary boundary
variation into a continuous plane that JPEG 2000 represents much less cheaply
than JBIG2, or into a second exact mask whose overhead exceeds the first mask's
savings. Simple Meyer texture shrinkage does not change that balance.

The median-core/XOR variant does finally beat the optimized source mask bytes,
but only by 96 bytes across 16 pages (0.0196%). That result validates the
representation and the terminal selector while also showing that generic
JBIG2 was already extremely close to the entropy floor for this scan.

The useful next variants are narrower:

1. map the winning split stream onto standard JBIG2 refinement regions without
   surrendering its class-conditioned coordinate and exception models;
2. terminal-screen individual classes rather than frequency prefixes when a
   larger page sample can amortize the measurement cost;
3. compare BZip2 with a JBIG2 refinement arithmetic coder over the same fixed
   aligned XOR fields; and
4. keep foreground/background optimization for pages whose decoded colour
   planes already contain redundant texture, rather than moving hard bilevel
   edges into those planes.

The PDF object library remains scaffolding. Replacing it with a narrow
byte-preserving parser is appropriate after a stream representation wins; it
cannot improve the codec numbers measured here.
