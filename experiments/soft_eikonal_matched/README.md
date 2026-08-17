# Parameter-matched soft Eikonal study

This benchmark contains only two trained forms:

1. `soft_eikonal`, using the LELU-based soft metric direction pool;
2. `ordinary_mlp`, an encode-expand-LELU-contract-decode MLP with the exact
   same number of active trainable parameters.

The MLP keeps the same latent width and spends the soft model's extra budget on
a wider dense expansion. A tiny active affine residual consumes the indivisible
remainder—below 0.8% of the budget—so matching uses no dead padding. The legacy
globally affine diagnostic remains tested in `models.py`, but is not one of the
two benchmarked forms.

The attached older demos were used only for problem definitions: periodic
wells, ripple, ring SDF, N-D periodic sums, hyperchecker, and the complex 3-D
spiral. Their models, optimizers, GELU layers, and plotting code were not used.

Additional problems are two moons, pinwheel, XOR quadrants, sinusoidal bounds,
radial stripes, Swiss cheese holes, Lorenz lobes, the previous spiral and
checkerboard tails, and high-dimensional periodic/hyperchecker tests.

The 1-D continuation suite contains the user-supplied multi-scale target
(smooth trend, localized bumps, medium oscillation, and damped high-frequency
ripple), a chirp, localized smooth steps, and a mixed Fourier signal. Each is
trained on `[-3,3]` and evaluated densely through `[-5,5]`, with ten paired
left/right tail bins.

Besides speed, fit, and tail metrics, the benchmark measures:

- input-to-input variation of the learned Jacobian;
- allocation entropy and variation in both soft-Eikonal layers;
- correction-to-base magnitude;
- performance after replacing allocations with uniform weights;
- performance after mismatching each observation with another observation's
  allocation;
- performance with both Eikonal corrections removed.

`analyze_results.py` turns the raw run records into a direct paired table. It
keeps interpolation fit separate from held-out continuation, since combining
those two numbers conceals the main failure mode. `run_visual_probes.py`
re-trains a fixed seed and exports dense, correctly oriented fields for the
double spiral, checkerboard, periodic wells, and all four 1-D functions.

The completed three-seed findings and their interpretation are in
[`RESULTS.md`](RESULTS.md).

Run on the M4 Mini CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_matched/test_matched.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_matched/run_benchmark.py \
  --out /tmp/soft_eikonal_vs_mlp --widths 16,36 --seeds 3 --steps 800
```
