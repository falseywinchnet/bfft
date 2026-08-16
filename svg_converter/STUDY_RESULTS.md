# Initial conversion study

All runs used the converter in this folder and the M4 Mini's system Python.
The examples are committed as PNG inputs, SVG outputs, and JSON reports.

| Input | Structural + detail colors | SVG paths / loops | RGBA MSE | SVG size |
|---|---:|---:|---:|---:|
| `geometric_badge.png` | 8 + 4 | 7 / 12 | 0.00 | 8,312 B |
| `translucent_ribbons.png` | 14 + 6 | 20 / 54 | 136.65 | 26,794 B |
| `cameraman_source.png` | 12 + 6 | 18 / 1,137 | 80.54 | 180,679 B |

The geometric control is exactly representable by the final regional means,
so its palette reconstruction has zero pixel error. The ribbon control keeps
the large translucent trajectories but intentionally posterizes its smooth
background. The natural cameraman control keeps the large silhouette, camera,
tripod, skyline, and tonal field. Its high loop count is useful evidence: a
photographic texture field still produces too many small contour islands even
under a small color budget.

This first study therefore supports the hierarchy for icons, flat artwork, and
stylized illustrations. For photographs, the next useful extension is not more
colors; it is a boundary-complexity pressure that prices each new residual
island against its reduction in perceptual error.

The second contour pass added scale-aware subpixel relaxation and a 0.65 px
same-color seam guard. On the controls this reduced line commands from 26,424
to 17,325, introduced 2,003 accepted quadratic events, and removed the white
antialiasing pinholes visible around dark cameraman regions.

## Portsmouth archive findings

`PortsmouthProject.zip` was extracted under a fresh `/tmp` directory and read
without adding it to the import path. The active outline route uses ordered
contours attached to transported medial structure, cyclic phase alignment,
monotone correspondence, topology review, periodic Makima masters, and
error-bounded quadratic compilation. The archive also documents rejected
filled-distance and dense-growth routes that preserved nominal topology while
losing construction or producing optical defects.

Only the portable method lessons were retained here. No font source, source
font, generated outline, or code was copied into this project.

## High-color optimization study

The original high-resolution refinement materialized a dense pixel × color
cost volume and repeatedly scanned the full image for every color and residual
parent. At 860×758×128, the cost matrix alone is about 637 MiB as float64,
before temporary feature differences and neighbor penalties.

The optimized route preserves the same update rule but evaluates checkerboard
pixels in 16,384-pixel chunks, expresses perceptual distance through the
weighted norm identity, groups residual pixels by structural parent, and
traces contours inside occupied bounding boxes. A dense-reference regression
test verifies that chunking does not change regularized labels.

On the supplied portrait, the optimized 128 + 6 run takes 1.29–1.35 seconds at
860×758. A 2× 1720×1516 control takes 4.04 seconds. The 64-color profiled run
fell from 11.41 seconds to 1.59 seconds (7.2× overall; 8.4× inside the core
conversion). The generated SVGs remained byte-identical across the
optimization passes.

## Error-bounded dense-illustration study

The 1714×823 city illustration exposed a representation failure rather than a
parameter failure. The topology route at 128 + 6 colors measured 353.83 RGBA
MSE. Assigning the same 128-color basis at full resolution reduced that to
35.99, showing that the coarse-owner lift—not palette capacity—dominated the
error. For comparison, a conventional unconstrained 256-color median-cut
quantizer measured 43.75 RGB MSE (32.82 when averaged as RGBA with exact
alpha), so palette tuning alone could not reliably satisfy an RGBA target of
30.

Adaptive quality occupation reached 29.8358 RGBA MSE with 128 structural
colors and 18 accepted residual splits; the configured split budget was 32.
The final complete conversion took 20.11 seconds, of which 18.94 seconds was
exact SVG contour compilation. The result contains 146 emitted color paths,
179,415 closed subpaths, and 19,816,417 bytes. Browser raster inspection
confirmed that crisp exact-edge compilation removed the white pinholes seen
when micro-contours were independently relaxed and curved. The browser tool's
JPEG-80 capture was pixel-identical to putting the 29.8358-MSE region model
through the same JPEG encoder, establishing that the pre-compression browser
render and measured model agree at native resolution.
