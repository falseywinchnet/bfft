# FMMT with certified support birth

This checkpoint folds the PIT-R10 manufactured-coarse-support rule into FMMT itself.
There is no post-denoising PIT correction.

Run:

```bash
python fmmt_certified.py denoise input.png output.png --diagnostics output.json
```

Ablate the new support law while keeping the same file/code path:

```bash
python fmmt_certified.py denoise input.png plain.png --plain-fmmt
```

The implementation is grayscale 2-D and depends on NumPy, SciPy, Numba and Pillow.
The built-in benchmark additionally uses scikit-image.

Key change: FMMT's robust bootstrap is provisional. Independent observation lanes
must certify coarse curvature or fine ancestry before coarse bootstrap structure may
become hereditary eikonal geometry. Unsupported coarse support is allowed a finite,
conservative flux evolution. The same evidence lowers eikonal resistance across
unsupported contrast. The subsequent FMMT signal/noise measure transport and
observation update are unchanged.

See `RESEARCH_REPORT.md` for equations, ablations and measured results.

