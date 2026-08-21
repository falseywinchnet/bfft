# Continuous dense flow to positive exposure transport

## One field, no motion-family decision

For registered observations `A` and `B`, the estimated sampling field `u(p)`
is defined by

```text
B(p + u(p)) ~= A(p).
```

Translation, rotation, scale, shear, affine motion, and smooth local
deformation are shapes of this same two-component field. They are not candidate
models and there is no winning class. `dense_estimation.py` estimates forward
and reverse fields on one Gaussian pyramid. At a fixed warp, the increment
minimizes a robust linearized photometric defect plus one image-metric flow
action:

```text
sum_p rho(Ix du + Iy dv + It)
  + alpha^2 sum_(p,q) g_pq |(u+du)(p) - (u+du)(q)|^2.
```

`g_pq` is a positive Eikonal conductance induced by image contrast. All graph
edges remain present; strong contrast lengthens their metric rather than
selecting or deleting a direction. Each normal equation is applied as a
matrix-free positive operator and solved by conjugate gradients. A short
line search accepts only energy-decreasing updates. Pyramid state persists
between levels.

## Reverse-cycle evidence and confidence transport

The reverse field `v` supplies the pointwise cycle defect

```text
c(p) = |u(p) + v(p + u(p))|.
```

Photometric closure and cycle closure form connection evidence. Local texture
is absolute measurement support, not merely a relative weight. Instead of
setting unsupported flat regions to zero motion or granting them full motion,
the connection authority `a` is the metric-harmonic extension

```text
(S + lambda L_g + epsilon I) a = S c_connection,
0 <= a <= 1.
```

Here `S` is the continuous texture-support field and `L_g` is the same
contrast-weighted Laplacian. Thus measured connection is transported only
through short image-metric paths. The authority remains continuous and never
chooses a motion family.

## Flow becomes the existing exposure object

Two observations determine relative motion but not a common absolute warp.
The implementation declares that gauge and uses the symmetric pair midpoint:

```text
d_0(p) = -a(p) u(p) / 2,
d_1(p) = +a(p) u(p) / 2.
```

If the exposure-to-frame interval ratio is `tau`, positive atoms at normalized
exposure times `t_j` carry

```text
r_j(p) = t_j tau a(p) u(p),    t_j in [-1/2, 1/2].
```

These are ordinary `SpatialExposureField` instances. Rotation and dense flow
therefore use `solve_spatial_field_consensus`, the same barycentric-first
coordinate pullback, matched positive gather/scatter operator, discrepancy
stop, fold gate, and uncertainty transport. Estimation changes the field; it
does not select a reconstruction algorithm.

Flow uncertainty combines reverse-cycle disagreement and the amount by which
connection authority is withheld. It is converted to radiometric sensitivity
through the observed image gradient, then combined with cross-observation and
backtransported residual terms. This remains a sensitivity diagnostic, not a
calibrated interval.

## Measured control and boundary

`dense_estimation_results/results.json` uses six 96 by 96 sources, read noise
sigma 0.002, and a field containing translation, affine shear, cross-axis
motion, smooth local deformation, and finite exposure mixing simultaneously.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Best capture | 21.493 dB | 0.6084 |
| Unregistered average | 23.497 dB | 0.6428 |
| Estimated dense consensus | 33.462 dB | 0.9702 |
| Known-field consensus oracle | 35.179 dB | 0.9805 |

Mean/q90/maximum endpoint error is 0.259/0.678/1.332 pixels. Every source gains
at least 8.18 dB over the unregistered average. Identical observations with a
common unknown warp and exposure are left unchanged exactly.

The first algorithmic optimization adds a machine-precision identity-gauge
fast path and a coupled per-pixel Jacobi preconditioner for the unchanged dense
normal equation. The next adds both first-derivative closure equations to the
same field at the finest physical scale, removes rejected substep searches,
and reduces the preconditioned CG ceiling from 80 to 60. M4 battery time falls
from 3.27 to 2.70 seconds while mean PSNR rises from 33.448 to 33.461 dB and
endpoint error falls from 0.262 to 0.259 pixels. A final finest-scale robust
flow action permits motion discontinuities without changing the coarse basin;
the M4 battery remains 2.76 seconds at 33.462 dB and 0.9702 SSIM. Tapered hair
trades -0.024 dB PSNR for +0.00014 SSIM; the other aggregate and per-source changes are retained
in the JSON ledgers. The unpreconditioned and brightness-only artifacts remain
as `results_v1_unpreconditioned.json` and
`results_v2_preconditioned_brightness.json`; the pre-robust ledger is
`results_v3_derivative_constraints.json`.

This is a smooth single-layer flow control. Positive multi-view ownership and
joint fold coverage are now implemented separately in
`VISIBILITY_OWNERSHIP.md`, but multiple motion sheets and depth-separated
latent appearance remain open, as do saturation, rolling shutter, and
real-camera validation. An individual fold aborts a single-observation inverse;
a multi-view solve may continue through the unchanged direct operators only
when their adjoints jointly cover the latent domain. Native promotion is
intentionally deferred until the multi-sheet representation stabilizes.
