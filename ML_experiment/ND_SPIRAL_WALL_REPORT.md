# N-D spiral wall screen

## Scope

This screen deliberately tests only `nd_spiral_high_rank`: sixteen observed
coordinates, eight coupled harmonic planes, two antipodal branches, an inner
observed region, and an equally large unseen outer region.  Every model sees
one 16-vector at a time.  Radius, phase, turn index, the generator's rotation,
neighbors, and tail examples are unavailable during training.

The screen uses LELU throughout; no model uses GELU.  Each configuration trains
for 500 steps on the M4 CPU.  The three runs change both the data seed and the
unknown 16-D orthogonal rotation.

## Result

The experiment found a much simpler solution than self-context or continuous
frame transport:

| mechanism | learned scalars | fixed atlas scalars | observed validation | unseen mean | unseen worst |
|---|---:|---:|---:|---:|---:|
| Shallow real odd-cubic sketch | 2,437 | 0 | 1.000 | **1.000** | **1.000** |
| Shallow learned bispectrum | 4,837 | 0 | 1.000 | **1.000** | **1.000** |
| Fixed random bispectrum + linear readout | 770 | 18,432 | 1.000 | **1.000** | **1.000** |
| Learned deep bispectrum | 22,492 | 0 | 1.000 | 0.993 | 0.986 |
| Generic odd-cubic layer network | 18,652 | 0 | 1.000 | 0.975 | 0.970 |
| Dynamic subspace conduits | 34,382 | 0 | 1.000 | 0.940 | 0.931 |
| Tied moving-frame recurrence | 5,571 | 0 | 1.000 | 0.935 | 0.857 |
| Input-conditioned Cayley flow | 2,210 | 0 | 1.000 | 0.845 | 0.788 |
| Continuous frame flow (AdamW) | 4,454 | 3,472 | 1.000 | 0.792 | 0.732 |
| Relational self-context | 5,006 | 3,456 | 1.000 | 0.703 | 0.555 |
| Even quadratic parity control | 6,530 | 0 | 0.496 | 0.502 | 0.502 |
| Ordinary LELU MLP | 2,834 | 0 | 0.651 | 0.435 | 0.394 |

Several perfectly fitted but structurally wrong mechanisms extrapolate below
chance: midpoint Hessian/coset response (0.284 mean), soft affine hypotheses
(0.381), and living hidden graphs (0.432).  Their observed fit is not evidence
of the intended rule.

## Why the third-order family wins

For this generator, the second branch is exactly the negative of the first:

`x_1(u) = -x_0(u)`.

An even feature map therefore satisfies `phi(-x) = phi(x)` and cannot identify
the class.  The quadratic control's chance result verifies this directly.

An odd cubic feature such as

`(a^T x) (b^T x) (c^T x)`

changes sign under the branch swap.  The eight coordinates are coupled
harmonics of the same phase, so third-order cross-products contain stable phase
relations.  A wide fixed random bank spans enough of that cubic tensor space
for a 770-parameter linear readout to find the invariant.  That model is not
storage-free: its frozen random atlas contains 18,432 coefficients.  It needs
neither an explicit frequency nor a learned observation projection, while the
2,437-scalar shallow real cubic obtains the same result without a fixed atlas.

The complex form makes the connection clearer: products of the form
`z_i z_j conjugate(z_k)` preserve relative phase while remaining odd under the
global antipodal swap.  The learned and fixed bispectral forms both work, but
the real shallow cubic proves complex arithmetic is not essential.

## Interpretation

This is a valid and very strong solution to the current generator, but it also
reveals that the benchmark is less generic than it looked.  It primarily tests
whether the architecture makes the correct polynomial parity/order cheaply
available.  CFF and self-context were laboriously approximating a separator
that a random odd third-order lift exposes immediately.

The result does not establish omni-inducement.  A responsible next experiment
is a family of N-D continuation generators that independently vary branch
symmetry and algebraic order: antipodal, reflected, translated, locally
warped, and non-polynomial.  The model must not be told which family generated
the observation.  That would distinguish a generally useful learned
inducement mechanism from a lucky match between cubic parity and this task.

## Reproduction

Run the model/gradient invariant suite and a complete rotation on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m unittest ML_experiment.test_nd_spiral_wall

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/run_nd_spiral_wall.py \
  --out /tmp/nd_spiral_wall --steps 500 --width 24 --seed 0
```

Copy the remote output back, then build its image gallery locally with:

```sh
python3 ML_experiment/build_nd_spiral_wall.py PATH_TO_COPIED_RESULTS
```
