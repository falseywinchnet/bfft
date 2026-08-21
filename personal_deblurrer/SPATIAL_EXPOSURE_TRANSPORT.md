# Spatial positive-exposure transport

## One law for distortion and mixing

At sensor position `p`, a spatial positive exposure field carries displacement
atoms `d_j(p)` and normalized positive weights `w_j(p)`:

```text
y(p) = sum_j w_j(p) x(reflect(p - d_j(p))).
```

This is one formation object, not a blur-family catalogue:

- one atom is deterministic optical or camera warp;
- multiple constant atoms reduce exactly to a global positive PSF;
- multiple spatial atoms describe a varying exposure trajectory.

`SpatialReflectedExposureOperator` performs bilinear gather through this field.
Its adjoint scatters through the identical four reflected neighbors and closes
the inner product. The global-kernel lift agrees with SciPy reflect convolution
to floating-point precision.

## Analytical barycentric-first factorization

The deterministic displacement is the exposure barycenter

```text
m(p) = sum_j w_j(p) d_j(p),
r_j(p) = d_j(p) - m(p).
```

The sensor-to-latent map is `q = p - m(p)`. For an invertible field, its inverse
is found continuously by the fixed point `p = q + m(p)`. The observation,
residual atoms, and weights are all pulled through this same inverse coordinate
map. In latent coordinates the formation is

```text
y_pull(q) = sum_j w_j(p(q)) x(q - r_j(p(q))).
```

Thus deterministic warp is removed first and the centered exposure is refined
second, without testing whether an image belongs to a warp or mixing class.
Zero barycentric flow makes the pullback identity. Zero residual variance makes
the centered operator identity and triggers the common numerical-discrepancy
stop at zero passes.

## Geometry ledger

The field records barycentric-flow RMS, centered-mixing RMS, and the determinant
of the sensor-to-latent Jacobian `I - grad(m)`. Nonpositive determinant is a
fold/visibility failure, not permission to hallucinate an inverse. This is now
an executable single-observation gate: any nonzero fold region stops the
coordinate inverse and all centered-exposure passes, preserves the observation
exactly, and places the fold mask plus adjoint-coordinate disagreement in the
sensitivity image. In a multi-observation solve, however, an individual fold
only disables the pullback preconditioner. The unchanged positive normal
equation proceeds through the original operators when their adjoints provide
joint latent coverage. `VISIBILITY_OWNERSHIP.md` records that distinction.
Multiple motion sheets and depth-separated latent appearance remain future
state variables.

## Current controls

`shear_path_exposure` composes a spatial shear with centered horizontal
exposure. `rotational_exposure` constructs actual radius-dependent camera
rotation atoms; its centered and deterministic limits use the same operator.
The six-source battery is `run_spatial_benchmark.py` and its selected artifact
is `spatial_results/results.json`.

The current Python implementation is intentionally the representation oracle.
Native promotion comes only after the spatial recovery battery, adjoint,
boundary, coordinate-convergence, and fold gates pass.

Those gates now pass for ABI v6. The native layer evaluates only the immutable
bilinear source indices and spatial coefficients. Its batch operation carries
distinct images and exact per-plan contribution counts; it does not pool
sheets or choose one. Python retains the field, coordinate inversion,
Jacobian/fold audit, discrepancy stop, and uncertainty law. The Dear PyGui
generator exposes **Rotational exposure** inside the same
**Blur image** versus **Use as-is for deblurring** workflow; it adds no
registration or solver-selection button. Pair A/B reconstruction uses the
continuous dense field, not a rotation-versus-affine-versus-local selector.

## Multi-observation estimation

`spatial_estimation.py` implements the first camera-consensus estimator on the
continuous rotational manifold. Every observation pair supplies forward and
reverse robust registration evidence. One weighted least-squares cycle solves
all relative mean angles with the central observation as an explicit gauge;
neighboring angular velocities and the camera duty cycle generate exposure
extents. The reconstruction then pulls every observation and centered field
into the reference coordinates and performs one normalized positive descent
on a shared latent image.

This construction follows the measurement lessons of Portz et al. (blur is a
function of flow), Kim and Lee (bidirectional flow generates pixel-wise blur),
and Kim, Nah, and Lee (flow and latent restoration require a joint model). It
does not copy their optimizers. If the observations carry no relative motion,
the common rotation/exposure remains a gauge and the estimator abstains rather
than sharpening identical captures.

The rotational estimator is retained as a bounded headless positive control.
`dense_estimation.py` is the active GUI estimator. It replaces the rotational
manifold by one continuous 2-D field and still calls the same
`solve_spatial_field_consensus` inverse. The GUI operates on the explicitly
chosen Pair A and Pair B and never infers registration from equal image
dimensions. Synthetic pairs are required to share one immutable source truth;
truth-coordinate metrics are shown only when the known synthetic barycenters
also close around the estimator's symmetric midpoint gauge.
