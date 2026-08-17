# Omni-inducement benchmark

This project compares whether a layer learns a problem quickly, economically,
and with genuine class retention beyond the observed region. Every learned
neural nonlinearity in the benchmark is `LELU`; there is no GELU fallback.

## Families

The registry contains thirteen distinct forms: exact linear, dense LELU,
static and living Fourier circles, observation-derived metric graph,
learned-subspace Gram witnesses, hypersphere-atlas transport, soft Eikonal
direction pooling, derivative-trained jet transport, a real matrix-exponential
flow, associative Newton shells, the continuous Banach/Eikonal sieve, and
projective role transport. The implementations are benchmark adapters rather
than imports of old scripts: they accept arbitrary input dimension and share
one trainer. The M-layer notebook contributed only the double-spiral problem
setup; none of its model or optimizer code was used.

## Problems and ranks

- `spiral_2d`: 50% inner double spiral observed, ten consecutive outer bins.
- `checkerboard_2d`: inner square observed, ten outward square shells.
- `nd_spiral_low_rank`: a rank-2 spiral randomly embedded in 16 dimensions.
- `nd_spiral_high_rank`: eight harmonic spiral planes, randomly rotated in 16-D.
- `hypercube_checker`: eight-way parity structure in a rotated 16-D cube.

All held-out bins are exactly class balanced. That prevents whole-tail accuracy
from hiding class collapse.

## Metrics

Learning speed is reported as validation learning-curve area and steps to 85%
and 95% balanced accuracy. Economy is the exact trainable parameter count at
two widths. Tail behavior includes minimum per-class recall in every bin,
frontier recall, retention AUC, and consecutive bins above 80% minimum recall.

For 2-D tasks, dense-grid probability MSE is reported beside boundary F1 and
connected-component count error. The 3-D SVG surfaces show learned class score
as height and color each row by deviation from the true surface. This makes a
low-MSE but topologically wrong fit visible.

The decision renderer explicitly reverses raster rows before placing them in
Cartesian coordinates. `x` increases right and `y` increases upward; the old
mirroring failure has a regression test.

## M4 CPU commands

Smoke test:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/omni_inducement_benchmark/test_benchmark.py
```

Balanced full sweep:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/omni_inducement_benchmark/run_benchmark.py \
  --out /tmp/omni_inducement_full --widths 16,36 --seeds 3 --steps 600
```

Because `/tmp` is remote, copy the directory back immediately after completion.
