# Native-resolution scan-to-JPEG study

## Question

Can a page from the 468-page *Travelling Wave Antennas* scan be flattened to a
high-resolution JPEG, passed through the repository's JPEG optimization logic,
and stored at or below the source page's mixed-raster byte budget without
destroying content?

## Source representation

Ordinary pages are 360 dpi and use mixed raster content (MRC):

1. a low-resolution JPEG 2000 (`/JPXDecode`) paper/background layer;
2. a high-resolution JPEG 2000 foreground-color layer;
3. a high-resolution one-bit JBIG2 (`/JBIG2Decode`) soft mask that places text,
   equations, and diagrams;
4. Flate content streams, including the searchable OCR text layer.

Page 30 (text plus diagrams) stores 22,819 image bytes and 2,444 content-stream
bytes, or 25,263 bytes total. Dense-text page 100 stores 49,459 image bytes and
5,822 content-stream bytes, or 55,281 bytes total.

Both pages were rendered at 360 dpi to 1737 x 2943 RGB pixels. The exhaustive
ordinary-JPEG control tested qualities 1-95, 4:2:0/4:2:2/4:4:4 chroma, and both
sequential and progressive output. Metrics are against the source PDF render.
OCR similarity compares Tesseract output from the source render and candidate,
so it measures recognition degradation rather than agreement with the PDF's
existing hidden OCR text.

## Results

| Page | Budget multiple | JPEG bytes | PSNR | AC SNR | Edge PSNR | SSIM | OCR similarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1x | 43,223 (target missed) | 21.26 dB | 0.38 dB | 15.31 dB | 0.91936 | 0.7952 |
| 30 | 2x | 49,749 | 21.43 dB | 0.55 dB | 16.06 dB | 0.92659 | 0.8144 |
| 30 | 4x | 99,446 | 33.91 dB | 13.04 dB | 22.92 dB | 0.97385 | 0.9807 |
| 30 | 8x | 200,527 | 40.15 dB | 19.28 dB | 30.69 dB | 0.99067 | 0.9879 |
| 100 | 1x | 57,533 (target missed) | 20.55 dB | 2.33 dB | 11.41 dB | 0.82629 | 0.3131 |
| 100 | 2x | 107,266 | 24.85 dB | 6.63 dB | 14.83 dB | 0.89455 | 0.9214 |
| 100 | 4x | 216,137 | 31.63 dB | 13.41 dB | 20.16 dB | 0.95102 | 0.9691 |
| 100 | 8x | 437,798 | 36.58 dB | 18.36 dB | 27.12 dB | 0.97982 | 0.9826 |

The conventional SNR includes the large beige DC paper field and therefore
looks generous. AC SNR and edge PSNR better expose what happens to glyphs.

## Repository JPEG optimizer

The full manual JPEG frontier was run on the M4 Mini at 1x and 4x:

- At 1x, it also bottoms out at quality 1 and cannot meet either target.
- At 4x on page 30, its selected 99,446-byte result is byte-for-byte the best
  ordinary progressive JPEG control: no structural transform is selected.
- At 4x on page 100, the optimizer selects 0.5 chroma projection, but the
  difference from the unmodified progressive JPEG is only 18 stored bytes and
  less than 0.00002 dB. It is not a material gain.

This negative result is expected from the domain: the visible information is
almost entirely a one-bit glyph mask. Chroma rotation cannot move monochrome
edge entropy into a cheaper channel, and luma smoothing damages the content it
would need to remove.

## Verdict

A single high-resolution JPEG is not a viable replacement. It cannot even
reach the current page budget at quality 1, and needs roughly 4-8x the stored
bytes before OCR and edge fidelity approach the MRC source.

The result does **not** mean scanned content is incompressible. It means the
source already chose the correct representation family. The next optimizer
must work inside the MRC structure:

1. build lossless cross-page JBIG2 symbol dictionaries or an independently
   verifiable bitmap-glyph atlas;
2. preserve the existing JPX paper and foreground-color layers, which are only
   a few kilobytes per ordinary page;
3. preserve the searchable OCR text layer;
4. compare every page render and OCR extraction before accepting a new mask
   representation.

The JPEG route remains relevant for photographic pages and `/DCTDecode` image
XObjects, not for the book's text masks.

## Reproduction

```sh
.venv-jpeg/bin/python -m experiments.pdf_optimizer.run_scan_jpeg_study \
  '/path/to/Travelling Wave Antennas.pdf' \
  --pages 30,100 --dpi 360 --multiples 1,2,4,8 \
  --out tmp/pdfs/scan-jpeg-study \
  --pdftoppm /path/to/pdftoppm
```

The full repository optimizer runs used:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.manual_jpeg_optimizer optimize \
  tmp/pdfs/scan-jpeg-study/page-30-source.png \
  tmp/pdfs/scan-jpeg-study/page-30-ours-101052.jpg \
  --target-bytes 101052
```
