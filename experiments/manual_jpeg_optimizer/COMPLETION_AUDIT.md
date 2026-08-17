# JPEG ownership-transport completion audit

This audit distinguishes the requested continuous ownership optimum from an
unrequested claim of exhaustive optimality over every entropy-coded JPEG.

| Requirement | Evidence | Status |
|---|---|---|
| Manual JPEG decomposition with five visible stages | `core.py::analyze_five_stages` emits YCbCr, chroma sampling, DCT cascade, quantization, and zig-zag entropy views; CLI `analyze` and GUI tabs render them. | Proven |
| Cartoon/texture, v3-style labels, and SVG-derived regions | `core.py::preprocess` exposes cartoon, exact texture residual, phase labels, and aligned reconstruction. `balanced_regions.py` implements immutable lineage, globally competing exact SSE split gains, and weighted-median bifurcation. | Proven |
| Spatial ownership in Y/Cb/Cr and block/frequency | `spatial_dct_transport.py` transports positive and negative coefficient mass separately on the product graph. The city report records 4,143,226 edges, including 1,372,990 frequency edges. | Proven |
| Global optimum for the stated continuous transport | The fixed-graph objective is strictly convex; the city solve reports KKT and flow-divergence residual 5.74e-11, positive-mass error 2.91e-10, and negative-mass error 8.15e-10. | Proven for the continuous fixed-graph problem |
| Balanced city regions | Winning report: 256 regions, 50–150 blocks, size CV 0.140 and mass CV 0.103. The previous signature method produced 11,558 mostly singleton regions. | Proven |
| Direct Jpegli DCT/quantization/dead-zone control | The vendored Jpegli branch accepts a 3×H×W×64 JLDZ field in streaming quantization and final requantization. A zero field was byte-identical to stock; nonzero fields changed coefficient decisions. | Proven |
| MozJPEG-informed trellis without replacing Jpegli | `ownership_trellis.cc` implements a fresh AC dynamic program over position, zero run, ZRL/EOB, and all legal magnitude categories. Jpegli's float DCT, quant matrices, adaptive quantization, chroma tables, and bitstream remain authoritative. | Proven |
| Image-specific trellis representation | Jpegli tokenizes and optimizes Huffman codes once, trellises using those depths, resets stale token state, retokenizes, and reports before/after nonzeros, modeled bits, and terminal objective. | Proven |
| Quantization-table optimization | Independent luma/chroma log-volume frequency tilts are applied while constructing Jpegli quant tables. The winning luma tilt is -0.5. | Proven |
| CLI and GUI, including save controls | `jpegli-fuse` reproduces the full pipeline and JSON report. DearPyGui exposes transport, atlas, trellis, channel, quantization controls and “Encode + save fused JPEG.” | Proven |
| Beat TinyPNG city holdout at equal measured quality | `verify_city_win.py` confirms 217,112 < 217,219 bytes, SSIM 0.965969 > 0.953066, PSNR 38.426 > 35.665, and edge PSNR 28.235 > 27.891. | Proven |
| Standards-compatible output | The fused result decodes with Jpegli to 1714×823 PNG and Pillow; no side information or custom decoder is required. | Proven |

The balanced planar partition is not claimed to be the unique optimum over all
possible partitions, and the complete JPEG bitstream is not claimed to be the
minimum among all legal JPEGs. Those are outside the user's clarified global
optimality target. The continuous constituent redistribution is globally
solved on its declared graph, and its terminal block projection is globally
solved for the declared trellis rate model.
