# MRC foreground/background study

## Artifact diagnosis

The block artifacts in the supplied close-up occur on PDF page 10, in the
sentence containing “proper credit” and “offer most.” The clean comparison is
PDF page 180 around equation (4-50). A 360-dpi source-versus-safe render proved
that page 10 is pixel-identical after the 3% lossless/JBIG2 pass: the optimizer
did not introduce the blocks.

Page 10 stores a 1736x2943, 91,857-byte JPX background, a 21,158-byte JPX ink
plane, and an exact 1736x2943 JBIG2 soft mask. The decoded background itself
contains pale duplicate glyphs. Ordinary pages instead use a roughly 578x980
background under the same 360-dpi mask. The visible artifact is therefore MRC
segmentation leakage, not JPEG ringing and not a JBIG2 failure.

## Representation

The foreground pass keeps the exact soft mask and replaces only eligible ink
color planes with a very small interpolated RGB field. Candidate grids are
terminal-compressed and compete at widths 1, 8, 16, and 32; the smallest grid
whose visible masked samples remain within the configured RMSE limit wins.

The background pass is deliberately narrower. It requires a full-resolution
background of at least 12 KB, between 5% and 18% mask occupancy, mask-correlated
ghost RMSE between 2 and 12, and high outside-mask fidelity after reduction to
the normal 3:1 MRC background scale. A broad low-pass paper estimate removes
glyph leakage without the line-shaped tone bands created by normalized local
inpainting. JPEG 2000 rate candidates compete by actual bytes and an outside-
region PSNR gate.

## Full-book result

Input to this lane was the 18,177,839-byte safe optimizer result. The strict
MRC output is 16,820,813 bytes, saving another 1,357,026 bytes (7.465%). From
the original 18,741,699-byte PDF, the combined saving is 1,920,886 bytes
(10.249%).

The run accepted 308 foreground fields and only page 10's background. The
background became 4,198 bytes from 91,857 bytes; its normal-resolution scale
comparison was 53.90 dB PSNR and its terminal JPX comparison outside repaired
regions was 45.19 dB. The exact JBIG2 mask was retained.

At 240 dpi, representative foreground pages 20, 30, and 100 measured 41.11,
44.62, and 41.73 dB page PSNR with dark-shape IoU of 0.9916, 0.9933, and
0.9912. Page 10 measured 38.12 dB against the artifact-bearing source and a
dark-shape IoU of 0.9584; the intended difference is removal of the pale
duplicate text. Extracted text stayed identical on every QA page.

Sparse front matter (pages 5-9), a clean ordinary page (180), complex late-book
pages (410 and 467), and the photographic back cover (468) were retained and
rendered pixel-identically to the safe input in the QA sweep. Earlier research
variants that created horizontal paper bands or altered the back cover were
rejected before the final artifact was selected.
