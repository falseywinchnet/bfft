# JBIG2 and MRC foreground study

## Storage diagnosis

The 468-page scan contains one JBIG2 soft mask and two JPEG 2000 color
planes per page. Across the book:

| Representation | Stored bytes |
|---|---:|
| JBIG2 masks | 12,744,934 |
| JPX foreground planes | 2,434,182 |
| JPX background planes | 1,108,856 |

Every sampled JBIG2 stream contains only a page-information segment followed
by one immediate generic-region segment. No page uses `/JBIG2Globals`.
Re-encoding pages 30 and 100 with the current C `jbig2enc` generic encoder
produced byte-for-byte identical streams. The source therefore already sits
at that encoder's page-local optimum.

## Cross-page symbol experiments

The official C encoder's symbol mode was tested as a proposal generator and
every proposal was decoded before comparison.

- On pages 30 and 100 at the default 0.92 classifier threshold, the combined
  masks grew from 60,353 to 61,548 bytes and changed 2,107 bitmap pixels.
- At threshold 0.97, the pair grew to 63,546 bytes. Page 30 was exact, but page
  100 still changed 57 pixels.
- Across 64 pages at threshold 0.97, symbol output grew from 1,581,107 to
  1,640,304 bytes. Only three pages were pixel-exact; paying for the shared
  dictionary while retaining generic masks for the others lost 13,266 bytes.

The encoder's disabled refinement path was also reconstructed from its older
source logic and repaired enough to encode residuals. At threshold 0.92 the
64-page result grew by 64,332 bytes; 43 pages were exact and the other 21 had
one to three changed pixels. Threshold sweeps from 0.40 through 0.92 were all
larger than the source. This is not a usable production lane.

The upstream documentation agrees with the observed boundary: C `jbig2enc`
symbol coding substitutes classified symbols, while its refinement CLI is
disabled. A newer Rust encoder also describes generic mode as lossless and
its symbol modes as substitution-based; it does not yet implement
refinement-assisted coding.

## Accepted lossless pass

The safe pass keeps generic arithmetic coding but crops the encoded region to
its real ink bounds. It then:

1. decodes the source embedded stream;
2. encodes the cropped one-bit bitmap;
3. patches the JBIG2 page and region coordinates structurally;
4. decodes the proposal;
5. requires pixel-identical bitmap output;
6. accepts only a smaller raw stream.

On the full book, 452 of 468 masks were replaced. Mask bytes fell from
12,744,934 to 12,718,889, a 26,045-byte reduction. Combined with object packing
and verified Flate recompression, the final PDF fell from 18,741,699 to
18,177,839 bytes: 563,860 bytes or 3.01%.

A custom generic-template-1 candidate found another roughly 10 KB, but Poppler
rendering exposed antialiased sample differences on two QA pages despite
`jbig2dec` bitmap equality. That candidate was rejected and is not exposed by
the production CLI.

Seven varied pages (1, 2, 30, 100, 410, 467, and 468) were rendered from both
source and final PDFs at 360 dpi. Their RGB arrays were pixel-identical.

## Lossless refined-symbol hybrid

The disabled symbol-refinement encoder was repaired in two ways. Source
connected components are retained for each page, and refinement template 0 is
now encoded by direct pixel sampling so arbitrary signed reference offsets are
valid. A cropped XOR generic region stores the symmetric difference whenever a
symbol page still differs because of placement or composition edge cases.

This produced stock-`jbig2dec`-verified, pixel-identical output on all 468
pages. It was nevertheless larger: the original generic masks total
12,744,934 bytes, while one shared dictionary plus 468 refined-symbol pages and
their sparse residuals total 13,259,657 bytes. The exact hybrid therefore adds
514,723 bytes (4.04% of mask storage). On a 64-page threshold sweep, 0.92 was
best; thresholds 0.50, 0.65, 0.75, 0.85, and 0.97 all lost by more. Exact
refinement is structurally valid, but it is not a compression win for this
scan and is not integrated into the production optimizer.

Charging a private dictionary independently to every page changes one result:
page 363 falls from 42,902 generic bytes to 39,162 bytes at classifier
threshold 0.94, including its 37,872-byte private globals stream. The
serialized PDF saves 3,350 bytes after indirect-object and decode-parameter
overhead. The other 467 pages retain generic coding. This candidate is exposed
through `--jbig2-refined-symbols` and is always stock-decoder verified.

## Entropy-guided generic regions

Blank-row band splitting did not beat a single cropped generic region on any
representative page. A better proposal model bins a four-neighbor causal
context over 128-pixel cells, locates changes in conditional context
distributions, and terminal-encodes the top 16 change points. The entropy score
is used only for ranking because it substantially overestimates savings by
ignoring arithmetic-context cold starts.

Across all 468 source masks, 56 two-region proposals were smaller by 1,203 raw
bytes. Most isolate narrow left or right edge regimes; a few isolate a top
band. Against the already-cropped production PDF, six pages still win: 423,
425, 428, 431, 459, and 468. Their raw streams save 462 bytes and the serialized
PDF saves 469 bytes. Every proposal was decoded before acceptance.

Combined with the private page-363 dictionary and the earlier crop/Flate/object
pass, the final lossless PDF is 18,174,020 bytes versus the 18,741,699-byte
source: 567,679 bytes or 3.029%. Pages 1, 10, 30, 100, 363, 410, 423, 425,
428, 431, 459, and 468 rendered at 360 dpi with zero differing RGB values.

The other PDF-native one-bit representations were decisively larger over all
468 masks: CCITT Group 4 payloads total 17,106,873 bytes and Flate-compressed
packed bitmaps total 29,498,410 bytes, versus 12,744,934 bytes for the original
generic JBIG2 streams.

## Foreground-color frontier

The exact JBIG2 mask owns glyph geometry; the high-resolution foreground JPX
mostly stores a slowly varying brown ink color. A uniform, lossless JPX plane
can retain the original soft-mask rendering path while collapsing the color
plane to a few hundred bytes.

| Page | Foreground bytes | Uniform JPX | Render PSNR | Edge PSNR | SSIM | OCR similarity |
|---|---:|---:|---:|---:|---:|---:|
| 30 | 4,525 | 440 | 43.04 dB | 34.54 dB | 0.999342 | 0.9879 |
| 100 | 5,124 | 462 | 40.37 dB | 31.77 dB | 0.998843 | 0.9955 |

The whole-book audit measured visible-foreground error only, ignoring pixels
hidden by the mask. With a maximum foreground RMSE of eight 8-bit levels, 78
pages qualify and contain 363,145 source JPX bytes. Their uniform candidates
would be only tens of kilobytes in aggregate. At RMSE 10, 307 pages and
1,530,741 foreground bytes qualify, but that is a more visible tonal change.

This lane is promising but deliberately remains opt-in research. Geometry and
OCR survive because the mask is untouched, but the color plane is lossy in
representation even when the replacement JPX itself is lossless. A production
version should require per-page foreground RMSE, rendered SSIM/edge PSNR, OCR,
and manual spot checks before acceptance.

## Reproduction

The final safe run used an unmodified generic-mode encoder and decoder:

```sh
pdf-repr optimize input.pdf output.pdf \
  --jbig2-encoder /path/to/jbig2 \
  --jbig2-decoder /path/to/jbig2dec \
  --workers 4 --zopfli-iterations 5
```

Primary implementations consulted:

- [agl/jbig2enc](https://github.com/agl/jbig2enc)
- [ArtifexSoftware/jbig2dec](https://github.com/ArtifexSoftware/jbig2dec)
- [LegeApp/jbig2enc-rust](https://github.com/LegeApp/jbig2enc-rust)
