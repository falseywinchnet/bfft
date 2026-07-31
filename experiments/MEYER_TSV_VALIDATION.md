# Meyer versus TSV-weighted G-norm: validation scope

Primary paper: Roy Y. He, Martin Huska, and Hao Liu, *Image Decomposition
with G-norm Weighted by Total Symmetric Variation* (SSVM 2025),
[arXiv:2503.22560](https://arxiv.org/abs/2503.22560).

The paper minimizes

```text
TV(u) + lambda ||eta v||_G,  u + v = f
eta(x) = kappa + TSV(f, x).
```

TSV integrates signed directional variation over a symmetric support before
taking its magnitude. Periodic texture variations cancel in an interior;
isolated contours and transitions between texture regimes do not. Therefore
`eta` is low in texture interiors and high at regional boundaries, making
boundary allocation to `v` more expensive.

BFFT does **not** implement this functional. Its hot split is the unweighted
Gilles--Osher alternation

```text
u <- ROF(f - v, lambda)
v <- (f - u) - ROF(f - u, 1 / mu)
```

with scalar `lambda` and `mu`. There is no spatial `eta(x)`. Any comparable
contour rejection must emerge from competition between TV (cheap coherent
contours) and the ordinary G-ball (cheap rapid oscillation), not from the
paper's non-local symmetry weight.

`meyer_tsv_validation.py` tests that emergent behavior rather than copying
the paper's iterative solver. It provides:

1. a clean additive scene with isolated object contours and tapered
   oscillatory texture interiors;
2. the repository's established cartoon plus two-texture ground-truth rig;
3. a four-direction TSV diagnostic following equations (19)--(20); and
4. one native Meyer trace scored at passes 1, 2, 4, 8, 16, 32, and 64.

It also reports the existing segmenter's frozen support geometry separately.
That stage is not a weighted G-norm: it turns the Meyer/cartoon/glass fields
into an amplitude-normalized tensor, collapses the tangent eigenvalue of a
coherent contour, and uses the tensor determinant as population density.
This can reproduce the *allocation* goal of TSV by making contours into long,
low-density supports even when the raw Meyer texture still contains them.

The principal metrics are interior texture gain, excess texture at true
contours, cartoon edge-gradient retention, and an allocation AUC comparing
true texture interiors against contour error. This directly tests the v3
one-pass setting as well as convergence behavior.

TSV is scale dependent. The validation carriers have periods of 8 and 13
pixels, so its default long-axis variance is `12`; the paper explicitly
varies this parameter with texture scale. Using an example value whose
support is shorter than the carrier makes TSV itself fail the known-truth
control and is not a valid comparison.

Run:

```sh
python experiments/meyer_tsv_validation.py
```

## Result

The result is mixed, and the distinction matters:

- **Core Meyer is not TSV-equivalent.** On the clean symmetric-support rig,
  one pass captures `0.545` of the true interior carrier gain and has `0.618`
  contour-excess RMS in units of true texture RMS. At 64 passes, carrier gain
  reaches `1.000`, but contour excess remains `0.642`. The visible contour
  doublets therefore do not disappear at convergence.
- **TSV identifies the missing discriminator.** The top 10% of TSV pixels
  contain `48.6%` of one-pass texture error and `81.5%` of 64-pass texture
  error on the clean rig. On the established two-texture rig the corresponding
  figures are `54.7%` and `68.8%`. This is much stronger than chance and shows
  that high-TSV support localizes the residual failure of the ordinary split.
- **The segmenter already has an adjacent geometric analogue.** On the clean
  rig, its frozen support population measure is only `0.130` as large on
  contours as in texture interiors, and a random texture-interior pixel has
  greater measure than a random contour pixel with AUC `0.992`. This comes
  from coherent-tangent eigenvalue collapse and determinant density, not from
  weighting Meyer's norm. On the overlapping two-texture rig the AUC falls to
  `0.616`, so this downstream geometry is useful but not universally
  equivalent to TSV either.

The practical conclusion is not to replace the fast Meyer solver with the
paper's 2000-step splitting scheme. TSV is valuable here as a validation and
possibly a frozen support field: it pinpoints contour leakage that the core
G-ball cannot distinguish, while our existing tensor geometry already solves
much of the same allocation problem in the clean case.
