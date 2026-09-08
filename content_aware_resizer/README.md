# Content-aware resizing / image retargeting papers

Curated research collection covering the foundational algorithms, the main
classical alternatives, evaluation, learned retargeting, and recent generative
and object-aware systems. PDFs are in [`papers/`](papers/). The collection was
assembled and link-checked on 2026-08-28.

## Suggested reading order

1. **Avidan & Shamir (2007), Seam Carving** — the foundational discrete
   content-aware resizing operator.
2. **Rubinstein, Shamir & Avidan (2008), Improved Seam Carving** — forward
   energy, graph cuts, and temporally coherent video retargeting.
3. **Wang et al. (2008), Optimized Scale-and-Stretch** — the canonical
   continuous mesh-warping alternative.
4. **Rubinstein, Shamir & Avidan (2009), Multi-operator Media Retargeting** —
   combines carving, scaling, and cropping instead of betting on one operator.
5. **Rubinstein et al. (2010), A Comparative Study of Image Retargeting** —
   the RetargetMe perceptual benchmark and the best orientation to failure
   modes of classical methods.
6. **Cho et al. (2017), Weakly- and Self-Supervised Learning...** — the first
   major end-to-end deep content-aware retargeting system.
7. **Shocher et al. (2019), InGAN** — single-image internal learning for
   retargeting the patch distribution of a natural image.
8. **Tan et al. (2019), Cycle-IR** — unsupervised cyclic consistency for direct
   image retargeting.
9. **Shen et al. (2024), Prune and Repaint** — semantic pruning plus
   diffusion-based repair for arbitrary ratios.
10. **Liao et al. (2025), Object-IR** and **Xu et al. (2026), HALO** — current
    object-consistent mesh warping and human-aligned layered transformations.

## Contents by theme

### Foundations and classical operators

| Year | Paper | Why it matters |
|---:|---|---|
| 2007 | Avidan & Shamir — *Seam Carving for Content-Aware Image Resizing* | Introduced seam removal/insertion and multi-size images. |
| 2007 | Wolf, Guttmann & Cohen-Or — *Non-homogeneous Content-driven Video-retargeting* | Early real-time saliency-, motion-, and object-aware nonuniform video warp. |
| 2008 | Rubinstein, Shamir & Avidan — *Improved Seam Carving for Video Retargeting* | Forward energy and seam manifolds in space-time. |
| 2008 | Wang et al. — *Optimized Scale-and-Stretch for Image Resizing* | Influential grid deformation formulation for feature-preserving warping. |
| 2009 | Krähenbühl et al. — *A System for Retargeting of Streaming Video* | Production-oriented, temporally coherent, interactive pixel-accurate video warp. |
| 2009 | Karni, Freedman & Gotsman — *Energy-Based Image Deformation* | Generalized deformation energy with controllable legal transformations. |
| 2009 | Rubinstein, Shamir & Avidan — *Multi-operator Media Retargeting* | Dynamic programming over combinations of resizing operators. |
| 2014 | Zhang et al. — *Image Retargeting by Content-Aware Synthesis* | Patch/synthesis view that complements deletion and continuous warping. |

### Evaluation and surveys

| Year | Paper | Why it matters |
|---:|---|---|
| 2010 | Rubinstein et al. — *A Comparative Study of Image Retargeting* | First broad perceptual study; introduced the standard RetargetMe benchmark. |
| 2010 | Vaquero et al. — *A Survey of Image Retargeting Techniques* | Compact map of cropping, carving, warping, and multi-operator work. |
| 2020 | Garg & Negi — *A Survey on Content Aware Image Resizing Methods* | Later survey covering importance maps, operators, evaluation, and open issues. |

### Learned, internal, and generative methods

| Year | Paper | Why it matters |
|---:|---|---|
| 2017 | Cho et al. — *Weakly- and Self-Supervised Learning for Content-Aware Deep Image Retargeting* | Learns attention and a differentiable shift map without retargeted ground truth. |
| 2018 | Lin, Zhou & Chen — *DeepIR* | Deep semantic structure plus coarse-to-fine nearest-neighbor-field synthesis. |
| 2019 | Shocher et al. — *InGAN* | Zero-shot, single-image GAN that preserves internal patch statistics. |
| 2019 | Tan et al. — *Cycle-IR* | Cyclic perceptual coherence avoids explicit annotations. |
| 2020 | Mastan & Raman — *DCIL* | Contextual internal learning for training-data-independent retargeting. |
| 2022 | *Fast Hybrid Image Retargeting* | Efficient hybrid operator design balancing cropping, scaling, and carving. |
| 2023 | *Supervised Deep Learning for Content-Aware Image Retargeting with Fourier Convolutions* | Global-receptive-field Fourier convolution approach. |
| 2024 | Shen et al. — *Prune and Repaint* | Semantic seam pruning with adaptive diffusion in/outpainting. |
| 2025 | Liao et al. — *Object-IR* | Self-supervised object-consistent learned mesh deformation. |
| 2026 | Xu et al. — *HALO* | Human-aligned end-to-end retargeting with layered transformations. |

## Important companion papers not mirrored locally

These belong in any serious bibliography, but their stable public endpoints
were unavailable or returned non-PDF content during collection. Links below
point to authoritative metadata or a readable public copy.

- Pritch, Kav-Venaki & Peleg (2009), **Shift-Map Image Editing** — global
  discrete graph-labeling formulation that generalizes seam carving.
  DOI: <https://doi.org/10.1109/ICCV.2009.5459159>
- Chen et al. (2010), **Content-Aware Image Resizing by Quadratic Programming**
  — convex mesh optimization with foldover, saliency, and line constraints.
  DOI: <https://doi.org/10.1109/CVPRW.2010.5543281>
- Asheghi et al. (2022), **A Comprehensive Review on Content-Aware Image
  Retargeting: From Classical to State-of-the-art Methods** — the most useful
  recent full taxonomy and RIQA review. DOI:
  <https://doi.org/10.1016/j.sigpro.2022.108496>
- Mei et al. (2021), **Deep Supervised Image Retargeting** — MRGAN and a
  pseudo-ground-truth multi-operator dataset. DOI:
  <https://doi.org/10.1109/ICME51207.2021.9428129>

## Integrity

- `metadata/SHA256SUMS` contains a checksum for every local PDF.
- `metadata/pdf_validation.tsv` records MIME identification, page count, and
  byte size.
- Filenames begin with publication year and use stable ASCII slugs.

The PDFs are author manuscripts, open-access proceedings copies, institutional
repository copies, or arXiv versions. Copyright remains with the respective
authors and publishers.
