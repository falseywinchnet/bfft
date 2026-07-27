# Causal transport pre-port tightening

This round audits Claude's six-port review against the current
`causal_density` pipeline. The review predates the continuous first-arrival
front, so its owner/runner findings remain useful for the legacy control but
are not assumptions of the canonical method.

## Decisions by boundary

| Boundary | Decision | Reason |
| --- | --- | --- |
| Frozen Meyer geometry | Accepted | `Q' = alpha Q + beta I` reproduces the angle/eigenvector rebuild to `3.4e-16` relative and removes `atan2/cos/sin`. The metric trace percentile is now measured once per frozen resolution. |
| Density population | Tightened | Per-pixel multiplicity bookkeeping is vectorized. The deterministic local phase and population law are unchanged; no selection or population search was introduced. |
| Metric reduction | Retained | The Gauss/Lagrange reduction already gives the required local obtuse superbase. |
| Cyclic local stencil | Rewritten | Six fixed vectors are ordered with half-plane/cross-product comparisons. No image-wide `atan2` or `argsort` remains. |
| Reverse incidence | Rewritten | A two-pass counting construction builds the CSR graph in linear time and removes duplicate cardinal incidences. At 256–512px it stores 68–71% of the former raw ten incidences per pixel. |
| Hopf--Lax update | Rewritten | The interior simplex minimum is analytic. Eighteen bisection rounds are replaced by the closed-form stationary point and checked against a high-accuracy convex search. |
| First-arrival queue | Rewritten | One decrease-key entry exists per unaccepted pixel. Stale entries, capacity growth, and stale pops are impossible. The measured front performs about 1.8–2.1 key updates per pixel while peak live occupancy is only 5–38%, depending on the scene. |
| Reverse characteristic step | Tightened | The approximate 2x2 curvature is shifted analytically to positive definite. Converged and non-descent directions do not trigger a remarch. Half-inradius containment, every-germ survival, and exact action decrease remain mandatory. |
| Hard affine readout | Corrected | The basis is centered and scaled per cell, giving a mean plus closed 2x2 solve. The physical image-gradient ridge transforms by `1/r_i^2`; carrying the same numeric ridge into the scaled basis was rejected because it changes the model. |
| Ridge ladder | Tightened | The invariant affine score is reused, the selected final score is not recomputed, and a later ridge cannot overwrite an earlier better rung. |

## Important correction to the earlier PORT 07 proposal

The coordinate change

```text
u = (x - cx) / r
```

maps an image-space gradient coefficient `b` to a local coefficient
`beta = b r`. Therefore

```text
lambda n b^2 = lambda n beta^2 / r^2.
```

Using `lambda n` in both bases is not invariant. It changes the prior from a
penalty on physical image gradient to a penalty on variation across each
cell. That version sharpened RGB but seriously worsened the cartoon objective
on 256px cameraman. The implemented `lambda n / r^2` form keeps the
well-conditioned local solve, removes the image-origin coupling, and retains
the established physical smoothing.

On identical causal partitions, the corrected fit agrees with or slightly
improves the established combined objective at medium and large resolution:

| Image | Cells | Established objective | Conditioned objective |
| --- | ---: | ---: | ---: |
| Cameraman 256 | 1,681 | 0.002711438 | 0.002711353 |
| Cameraman 512 | 7,887 | 0.001645010 | 0.001645009 |
| Pikachu 256 | 413 | 0.005104211 | 0.005103870 |
| Pikachu 475 | 730 | 0.004094883 | 0.004094397 |

At 128px, cameraman and Pikachu differ by about `+0.1%` objective while coins
and astronaut are equal or better. This is retained as an honest small-scale
negative rather than hidden by a candidate selector.

## Timing

Isolated warm first-arrival time on cameraman:

| Resolution | Before | Tightened |
| --- | ---: | ---: |
| 256x256 | about 49 ms | about 35 ms |
| 512x512 | about 301 ms | about 253–274 ms |

Warm end-to-end medians on the same machine:

| Image | Previous round | Tightened |
| --- | ---: | ---: |
| Cameraman 256 | 0.477 s | 0.306 s |
| Pikachu 256 | 0.572 s | 0.265 s |
| Cameraman 512 | 2.089 s | 1.18 s |
| Pikachu 475 | 1.704 s | 1.09 s |

The first call can be slower because Numba loads or compiles kernels. Native
port targets should compare warm kernels and report initialization separately.

## Legacy review disposition

- The four-plane symmetric edge stack, direct tensor quadratic, dead floor,
  and runner-band pruning remain valid improvements for
  `legacy_bifurcation`.
- Runner pruning and fused soft moments do not apply to the canonical front:
  it has no runner state or soft ownership.
- The generalized-instability wrapper is now vectorized exactly. Its
  `CQ`/`QC` direction discrepancy is deliberately not changed because it
  affects only the legacy split model and still needs a quality A/B.
- Balanced refill and recursive residual flow received no unverified
  optimization. They remain controls, not native canonical port targets.

## Reproducibility

```sh
PYTHONPATH=viewer:experiments .venv/bin/python tests/port_needed_test.py
PYTHONPATH=viewer:experiments .venv/bin/python experiments/port_math_review.py
python viewer/profile_segmenting_veroni.py --gallery camera \
  --max-side 512 --allocation-side 512
```
