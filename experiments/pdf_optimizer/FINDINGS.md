# Forensic PDF findings

## Task lineage

- `01a00e78-5661-7cd2-93a1-25e3e2d8594a` - **Build DCT JPEG optimizer**
- `01a008b4-f0ef-7ca0-b82d-18d76c37fc76` - **Build novel PNG to SVG converter**

The JPEG work established candidate competition, actual terminal-byte
measurement, block/frequency/channel ownership transport, and a strict
separation between exact transformations and measured distortion. The SVG work
established coarse-owner topology, residual children locked to structural
parents, exact shared interfaces, and a rule that geometric complexity must
earn its bytes.

## Selected PDFs

### Scanned text as images

`Travelling Wave Antennas ... Anna's Archive.pdf` is 18,741,699 bytes and 468
pages. It contains 468 JBIG2 foreground images (12,744,934 stored bytes), 936
JPEG 2000 layers (3,543,038 bytes), and 1,408 Flate streams (1,746,274 bytes).
The JBIG2 pages do not reference shared `/JBIG2Globals` dictionaries. The file
has no object streams and reports itself as not optimized for fast web view.

This is not a naive page-per-JPEG scan. It is already a sophisticated
mixed-raster-content representation: high-resolution bilevel marks plus lower
rate continuous-tone layers. The immediate safe savings are structural and
Flate packing. The ambitious image lane is a lossless, cross-page JBIG2 symbol
dictionary grouped by compatible glyph appearance; it needs a dedicated
encoder and adversarial render/OCR verification because lossy symbol
substitution is unacceptable.

### Native text and vectors

`2608.17897v1.pdf` is 721,400 bytes and 16 pages. It contains no raster image
XObjects. Forty embedded font programs occupy 384,938 stored bytes and 105+
Flate content/font streams occupy most of the remainder. The producer already
uses object streams.

The apparent font fragmentation is mostly legitimate TeX font families and
optical sizes, not repeated copies of one subset. Only a few Latin Modern
families are split multiple times, totaling a few kilobytes. The larger
research target is re-encoding Type1 glyph programs into a more compact shared
representation while preserving charcodes, widths, ToUnicode maps, hinting,
searchability, and rendered pixels. Converting glyphs to page paths would be a
semantic regression.

## First structural baseline

Object-stream generation plus ordinary lossless Flate recompression changed:

- scanned book: 18,741,699 -> 18,428,322 bytes (313,377 bytes, 1.67% smaller);
- native paper: 721,400 -> 728,285 bytes (6,885 bytes larger).

This is why the demo treats the original as a candidate. A rewrite is not an
optimization merely because it uses nominally stronger settings.

## Selective representation results

The final candidate search with eight Zopfli iterations produced:

- scanned book: 18,741,699 -> 18,203,602 bytes, saving 538,097 bytes (2.87%);
- native paper: 721,400 -> 676,850 bytes, saving 44,550 bytes (6.18%).

The book replaced 466 of 1,408 Flate streams and packed small objects. The
paper replaced 132 eligible Flate streams; this includes compact font programs
and content/Form streams without changing their decoded bytes.

Poppler renders of book pages 30, 100, and 468 and paper pages 1, 8, and 16
were pixel-identical between source and output. Pypdf extraction was exactly
equal on every page: 712,439 characters for the scanned book's OCR layer and
66,891 characters for the native paper.
