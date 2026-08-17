# Results: LELU-only omni-inducement sweep

The M4 CPU completed 390 runs: thirteen forms, five tasks, widths 16 and 36,
and three deterministic seeds, each trained for 600 AdamW steps. The sum of
measured training time inside the runs was 726.4 seconds; evaluation, dense
shape measurement, rendering, and mirror I/O are additional. All raw histories
and per-bin class recalls are in `results_full/results.json`.

## A. Learning speed

At width 36 over the four structured tasks (excluding the chance-level
hypercube parity stress test), the best mean validation learning-curve areas
were:

| form | learning AUC | final balanced accuracy |
|---|---:|---:|
| soft Eikonal pool | 0.892 | 0.989 |
| learned-subspace Gram | 0.890 | 0.984 |
| living Fourier circle | 0.858 | 0.969 |
| hypersphere atlas | 0.855 | 0.970 |
| Banach/Eikonal sieve | 0.807 | 0.979 |

Soft Eikonal wins the speed aggregate, narrowly. Its checkerboard run reached
95% at a mean of 425 steps; on the high-rank N-D spiral it reached 95% at 108
steps. Learned-subspace Gram is close enough that seed and budget sensitivity
matter.

## B. Parameter economy

Living Fourier circle is the clear efficiency result. At width 36 it averaged
only 601 trainable parameters across the structured suite while reaching 0.969
mean balanced accuracy. At width 16 it used 249 parameters on the 2-D spiral
and 473 on 16-D inputs; it was the smallest form above 95% validation accuracy
on the 2-D, low-rank N-D, and high-rank N-D spirals.

This is not omni-inducement. The static circle control did not learn the same
problems, so observation-conditioned filtering matters, but the living circle
still benefits from its aligned circular construction. On checkerboard its
width-36 validation accuracy was 0.873, below the more flexible forms.

## C. Tail class retention

Minimum per-class recall, measured consecutively from the training frontier,
changes the conclusion. No form retained even the first 80%-recall bin on:

- the ordinary 2-D spiral;
- the 2-D checkerboard;
- the rank-2 spiral embedded in 16 dimensions.

The high-rank harmonic N-D spiral was different:

| form | mean min-class recall over tail | consecutive bins ≥80% | per-seed survival |
|---|---:|---:|---|
| learned-subspace Gram | 0.808 | 6.67 / 10 | 0, 10, 10 |
| jet transport | 0.732 | 6.67 / 10 | 0, 10, 10 |
| soft Eikonal pool | **0.828** | 2.67 / 10 | 0, 1, 7 |
| hypersphere atlas | 0.657 | 2.00 / 10 | 0, 5, 1 |
| matrix exponential | 0.626 | 1.67 / 10 | 0, 5, 0 |
| living Fourier circle | 0.720 | 0 / 10 | 0, 0, 0 |

Thus soft Eikonal has the largest whole-tail per-class area, while
learned-subspace Gram is the best contiguous-retention form and uses less than
half the parameters of jet transport. The seed vectors are important: this is
promising but not reliable induction yet.

## MSE versus shape

The dense 3-D surface and region diagnostics validate the user's warning. On
the 2-D spiral, several models reached essentially perfect validation and
near-zero validation probability MSE while their full-region surface had only
two to five connected components versus fourteen in the true solution. The
width-36 learned-subspace model, for example, averaged 1.000 validation
balanced accuracy but only 0.262 boundary F1. All models failed the immediate
tail frontier.

The 3-D SVG atlases plot class score as height and color the wireframe by error.
Their axes are Cartesian: x increases right, y increases upward, and the front
edge is y-min. The decision atlas carries the same explicit orientation and is
covered by the non-mirroring regression test.

## Negative control that matters

The rotated eight-way hypercube checker stayed at chance for every form under
this budget. The best width-36 mean validation accuracy was 0.524. Nothing in
this sweep is an arbitrary-structure learner; the benchmark now has a problem
that prevents that claim from being made by interpolation on friendly tasks.
