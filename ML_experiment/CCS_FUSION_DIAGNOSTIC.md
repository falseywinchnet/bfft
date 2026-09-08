# CCS fusion diagnostic

## Question

Were the curvature-response results already using the structural acquisition,
gradient-zonotope, intrinsic-support, and odd-factor ideas developed in the
preceding rounds?

No. The 24-task CCS battery was an empirical-training comparison of ordinary
self-context and a one-scalar curvature-response graft. This follow-up fused
the mechanisms explicitly and used low-rank N-D spiral as the falsifier.

## What was fused

- Positive, signed, and magnitude-authorized curvature response.
- Curvature transported into the allocator's chart-selection simplex rather
  than added only to hidden activation state.
- Empirical, curvature-weighted, and observed-graph Hermite candidate pools.
- AdamW-transition covariance zonotopes selected by held-out witnesses.
- Rotation-invariant intrinsic-support geometry. A decisive empirical
  eigengap selects a support subspace; otherwise the full ambient geometry is
  retained.
- Intrinsic-radial candidate-gradient bands, so covariance axes represent how
  the update changes along a low-rank curve rather than minibatch noise.
- Deep odd cubic, learned bispectral, moving-frame, and Cayley recurrent
  controls at 1,000 optimization steps.

No method receives task phase, radius, test labels, Fourier coordinates, or an
analytic target.

## Main findings

### CCS is an acquisition accelerator, not a stable continuation law

On seed 0, CCS improved both N-D spirals. Across three seeds, that apparent
high-rank win reversed: self-context averaged about `.808` held-out / `.755`
tail while CCS averaged about `.706` / `.623`. A signed zero-initialized gain,
which begins as the exact self-context function, removed the arbitrary
positive orientation but also removed the favorable acceleration.

### The former classification atlas erased low-rank geometry

Low-rank spiral has two covariance eigenvalues near `.06`, followed by fourteen
near `.00004` (about a 1,375-fold cliff). Coordinatewise standardization made
those nuisance dimensions unit scale. The corrected atlas uses covariance
support and detects rank 2; high-rank spiral retains all 16 dimensions.

Correcting this error improves the legitimacy of the experiment but Hermite
mixing still does not improve continuation. Therefore support denoising and
local interpolation are not the missing law.

### The faithful radial-zonotope fusion still does not solve the spiral

At width 38 and 500 steps on low-rank N-D spiral:

| Method | Held-out | Tail | Learning AUC | Seconds |
|---|---:|---:|---:|---:|
| Self-context | .405 | .371 | .951 | 4.3 |
| Radial-band zonotope, cold | .430 | .419 | .548 | 21.7 |
| Radial-band zonotope, warm | .403 | .387 | .566 | 21.5 |
| CCS | **.459** | **.435** | **.962** | 7.4 |
| CCS + radial zonotope, cold | .397 | .297 | .571 | 37.8 |
| CCS + radial zonotope, warm | .393 | .359 | .614 | 39.1 |

The cold self-context zonotope has a small endpoint/tail gain, but its collapsed
learning AUC and fivefold cost make it a non-result. CCS and the optimizer
transport interfere rather than compose.

### More generic factor order does not create transport

Every 1,000-step low-rank control reached `1.0` observed validation accuracy.
Their held-out accuracies remained `.397`–`.439`: self-context `.397`, CFF
`.417`, dynamic subspace `.417`, deep odd cubic `.413`, learned bispectrum
`.439`, moving frame `.411`, and Cayley flow `.414`.

Tail-bin accuracy repeatedly changes from nearly zero to nearly one and back.
The models learn a spatial partition that the continuing spiral rotates
through; they do not learn the derivative law that rotates the partition.

## Interpretation

The earlier high-rank winners already contained a useful fusion: the learned
odd cone, operator sphere, and nested operator combined generic factor
expansion with tangent/curvature state and reached roughly `.96`–`.98` on the
seed-0 high-rank spiral. High rank supplies repeated phase relations in the
observation. Low rank withholds them. None of the present pointwise forward
maps reconstructs that missing relation from the observation set.

The next mechanism should therefore change the information topology, not add
another pointwise shell: estimate a reusable connection from relations between
observed samples, retain it as dataset/episode state, and transport a query
through that state. Sparse sine is the scalar falsifier and low-rank spiral is
its projective analogue. This is closer to a set-conditioned recurrent
connection than to another MLP nonlinearity.

## Reproducible artifacts

- `hermite_zonotope_pivotal.py`: fused acquisition and optimizer runner.
- `run_gradient_zonotope_battery.py`: intrinsic-radius witness geometry.
- `response_enhanced.py`: selection-curvature self-context layer.
- `run_nd_spiral_wall.py`: low/high-rank wall switch and exact probes.
- `test_hermite_zonotope_pivotal.py`: support, transport, and gradient tests.
- `results_ccs_fusion_screen/`, `results_ccs_nd_fusion_confirm/`,
  `results_ccs_signed_screen/`, `results_selection_curvature_screen/`,
  `results_intrinsic_hermite_lowrank/`, `results_intrinsic_zonotope_lowrank/`,
  `results_radial_band_zonotope_lowrank/`, and
  `results_nd_spiral_lowrank_factor_wall/`: raw results and probes.
