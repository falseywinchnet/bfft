# PDF representation optimizer demo

This experiment treats a PDF as a graph of reusable representations rather
than as a stack of pages to rasterize again. The safe mode is lossless at the
decoded-stream level and always competes against the original file. If every
rewrite grows, the output is an exact copy of the input.

## Current passes

1. Inventory images, image codecs, font programs, ToUnicode maps, content and
   auxiliary streams, object streams, encryption, linearization, and
   signatures.
2. Garbage-collect resources that are provably unreachable.
3. Generate PDF 1.5 object streams so dictionaries and small objects share a
   compressed representation.
4. Repack individual Flate streams with Zopfli. The optimizer inflates only the
   Flate envelope, keeps the original `/DecodeParms`, requires byte-identical
   inflated data, and accepts a replacement only when its stored bytes shrink.
5. Optionally decode page-local JBIG2 generic regions, crop them to their real
   ink bounds, re-encode, and accept only pixel-identical smaller masks.
6. Save both preserved-object-layout and repacked-object-layout candidates,
   measure real output bytes, and retain the winner among those and the
   untouched source.

JPEG, JPEG 2000, JBIG2, and CCITT streams are passed through exactly in safe
mode unless the caller supplies both JBIG2 tools. That opt-in pass is still
lossless: every proposal is decoded and compared as a one-bit bitmap before it
can replace a stream. JPEG, JPEG 2000, and CCITT are never generationally
re-encoded in safe mode. Signed PDFs are refused unless the caller explicitly
accepts that a rewrite invalidates signatures.

## Install and use

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e 'experiments/pdf_optimizer[test]'

.venv/bin/pdf-repr analyze input.pdf --report analysis.json
.venv/bin/pdf-repr optimize input.pdf output.pdf --report output.json

# Optional verified JBIG2 generic-region crop search
.venv/bin/pdf-repr optimize input.pdf output.pdf \
  --jbig2-encoder /path/to/jbig2 --jbig2-decoder /path/to/jbig2dec

# With the patched research encoder, also compete a private lossless symbol
# dictionary on each page. Its globals bytes are charged to that page.
.venv/bin/pdf-repr optimize input.pdf output.pdf \
  --jbig2-encoder /path/to/patched-jbig2 \
  --jbig2-decoder /path/to/jbig2dec \
  --jbig2-refined-symbols --jbig2-symbol-threshold 0.94 \
  --jbig2-entropy-regions --jbig2-entropy-cell-size 128 \
  --jbig2-entropy-frontier 16
```

The entropy-region lane estimates changes in causal neighborhood statistics,
terminal-encodes the top proposed cuts, stock-decodes every proposal, and keeps
only a smaller pixel-identical two-region stream. It does not trust the entropy
estimate as a byte model.

On the scanned-book benchmark, independent refined symbols select one page and
entropy regions select six more. Starting from the earlier safe output, they
save another 3,819 serialized bytes while twelve changed/control pages remain
pixel-identical at 360 dpi. The final lossless artifact is 18,174,020 bytes,
3.029% below the 18,741,699-byte source.

Large streams are bounded independently in stored and inflated form. Zopfli
work uses a bounded thread window, so the optimizer does not retain every
decoded stream in memory. `--no-zopfli` provides a fast structural-only pass;
`--workers`, `--zopfli-iterations`, and the two stream ceilings tune the slower
archival pass.

On the 149-Flate-stream native-paper control, four workers reduced a five-
iteration run from 15.30 seconds to 9.26 seconds (1.65x) on the MacBook while
producing the same selected bytes.

## Relationship to the image and SVG work

The JPEG optimizer's portable lesson is to treat the terminal encoder as the
measurement authority and move complexity only within a declared
representation. Here the exactly free movement is between equivalent Flate
encodings, object placement, and reachability. DCT image ownership transport is
reported as a possible opt-in lane for `/DCTDecode` images, but is not silently
applied to a PDF.

The SVG converter's portable lesson is topology-locked residual explanation.
A future raster-to-Form-XObject pass should be restricted to flat illustrations
whose exact shared-edge path model is smaller than the raster and whose render
comparison passes. Turning scanned prose into paths would destroy text
semantics and is deliberately not proposed.

The next high-value research lanes are conservative Type1-to-CFF glyph-program
packing, lossless JBIG2 shared dictionaries across compatible page groups, and
measured per-image DCT transport with page-render verification.

## Opt-in foreground/background optimization

The `mrc-optimize` command is a perceptual, separately gated lane for scanned
MRC documents. It never regenerates the one-bit JBIG2 geometry. Instead, it:

1. fits the visible samples of a high-resolution JPX ink-color plane with the
   smallest accepted 1x1, 8-, 16-, or 32-column interpolated RGB field;
2. retains a proposal only when visible-ink RMSE is below the requested limit
   and the encoded field is smaller;
3. recognizes anomalous full-resolution backgrounds only within conservative
   mask-coverage, ghost-energy, resolution, and source-byte bounds; and
4. repairs mask-correlated glyph leakage at the normal background resolution,
   then accepts only a measured JPEG 2000 candidate that passes outside-region
   PSNR and real-byte gates.

Install the additional numerical dependencies and run it after safe mode:

```sh
python3 -m pip install -e 'experiments/pdf_optimizer[mrc]'

pdf-repr mrc-optimize safe-optimized.pdf mrc-optimized.pdf \
  --jbig2-decoder /path/to/jbig2dec \
  --foreground-rmse 8 --background-psnr 44 \
  --report mrc-report.json
```

This command is intentionally opt-in because it changes color samples. Sparse
title pages, rich covers, photographs, multi-image compositions, excessive
mask coverage, and backgrounds with too much correlated scan energy are passed
through. [`MRC_STUDY.md`](MRC_STUDY.md) records the measured book run and the
artifact-driven acceptance gates.

[`JBIG2_STUDY.md`](JBIG2_STUDY.md) records the full mask experiment. The safe
crop pass saved 26,045 JBIG2 bytes and produced pixel-identical 360-dpi Poppler
renders on varied QA pages. Reconstructed refinement plus a cropped XOR
residual made all 468 symbol pages pixel-exact, though one global dictionary
still lost overall. Private per-page dictionaries are now a competing opt-in
candidate because isolated pages can win after their complete globals cost is
charged. The same study identifies a larger opt-in opportunity in the
low-entropy JPX ink color planes while retaining exact JBIG2 geometry.

[`GLYPH_RESIDUAL_STUDY.md`](GLYPH_RESIDUAL_STUDY.md) tests the follow-up
eikonal/Fourier-circle/Meyer glyph construction. Canonical source-superset
envelopes are algebraically exact before coding, but both direct foreground
residuals and exact paper-correction layers are larger on sparse and dense
pages. The inverse median-core construction with instance-local XOR refinement
does reverse it after terminal class selection: a purpose-built split stream
and cropped JBIG2 remainders save 96 bytes across the 16-page sample with
pixel-identical reconstruction. `GCS1` remains an experimental codec rather
than a stock PDF filter, so it is not installed by the safe optimizer.

## Scanned-book JPEG experiment

[`SCAN_JPEG_STUDY.md`](SCAN_JPEG_STUDY.md) records the native-resolution test
against the MRC scan. Its ordinary pages are JPX background/foreground layers
painted through high-resolution JBIG2 masks. A flattened JPEG cannot match the
existing page-stream budget even at quality 1, and needs roughly 4-8x the bytes
before edge and OCR fidelity approach the source. The accompanying
`run_scan_jpeg_study.py` runner inventories the actual page stream budget,
renders at native resolution, sweeps the JPEG frontier, and reports PSNR, AC
SNR, edge PSNR, SSIM, and OCR similarity.
