# SAD-aligned v3 benchmark

This folder measures the v3 segmenting representation against the evaluation
protocol used by [Soft Anisotropic Diagrams (SAD)][sad-project]. It does not
run SAD or claim that v3 already has a production codec.

## Dataset choice

SAD reports four corpora:

| corpus | images | SAD setting | practical status here |
|---|---:|---|---|
| Image-GS | 45 | 0.2 and 0.5 BPP | The paper says its assets came from Adobe Stock and Poly Haven; it is not distributed as one simple public corpus. |
| Kodak | 24 | 50,000 sites, about 16 BPP | **Default.** About 15 MiB total and every image is 768×512. |
| DIV2K validation | 100 | 0.5 and 2.0 BPP | Official HR validation ZIP is 448,993,893 bytes. Useful later for a true fast/full split. |
| CLIC validation | 41 | 0.5 and 2.0 BPP | Larger and less convenient than Kodak for the first pass. |

Kodak is therefore the smallest complete, directly reproducible SAD suite.
Because the viewer's fast limit is 768 pixels, both v3 modes consume the same
native Kodak pixels. The output explicitly marks this as `native_equivalent`;
it is a useful quality/rate test but not evidence of the speed advantage from
downsampling. `download_div2k_sample.py` uses HTTP byte ranges to extract one
official high-resolution validation image without making the 428 MiB archive
an implicit prerequisite.

## Rate accounting

SAD's BPP is itself a parameter-space proxy: 16 bytes per site, divided by
image pixels. Its packed site has two 15-bit positions, 32-bit RGB, 16-bit
temperature, 16-bit radius, 16-bit anisotropy direction, and 16-bit
log-anisotropy.

V3 currently renders and discards the fitted coefficients, so it cannot yet
emit a truthful codec file size. The runner reports three separate quantities:

1. `sad_layered_site_proxy_bpp`: SAD's 16-byte rule applied to both v3 layers.
2. `estimated_stream_bpp`: a declared fp16 parameter layout plus a real,
   reversible zlib estimate for the canonical texture topology and its
   structural-parent map. See `rate_model.py` for every field.
3. `reconstruction_png_bpp`: the actual lossless PNG size of the 8-bit
   reconstruction. This is a sanity control, not v3 representation size.

This distinction must remain visible in any plot or paper table.

## Metrics and timing

The runner records:

- PSNR and SSIM in linear RGB, matching SAD's stated metric color space;
- v3's existing sRGB PSNR;
- native-resolution readout metrics for a downsampled fast result;
- structural and texture cell counts, split/merge counts;
- every internal v3 timing and end-to-end wall time;
- the exact config, revision, Python, NumPy, OS, and processor metadata.

LPIPS is omitted from the lightweight default because this repository does not
otherwise require PyTorch or the LPIPS model weights.

## Run

From the repository root:

```bash
.venv/bin/python benchmarks/sad_v3/download_kodak.py
.venv/bin/python benchmarks/sad_v3/run.py --mode both

.venv/bin/python benchmarks/sad_v3/download_div2k_sample.py
.venv/bin/python benchmarks/sad_v3/run.py --dataset div2k --mode both
```

For a quick validation:

```bash
.venv/bin/python benchmarks/sad_v3/run.py --mode fast --limit 2
```

Generated data and results are ignored by Git. The human-readable result is
`benchmarks/sad_v3/results/kodak/summary.md`; per-image measurements are in
`records.jsonl`.

To rebuild summary files without rerunning the image fits:

```bash
.venv/bin/python benchmarks/sad_v3/run.py --summarize-only
```

[sad-project]: https://luckyiyi.github.io/SAD/
[sad-paper]: https://arxiv.org/abs/2604.21984
[sad-code]: https://github.com/LuckyIYI/SAD
[kodak]: https://r0k.us/graphics/kodak/
[div2k]: https://data.vision.ee.ethz.ch/cvl/DIV2K/
