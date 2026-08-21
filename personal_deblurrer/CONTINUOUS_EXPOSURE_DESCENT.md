# Continuous positive-exposure descent

## One formation object

After deterministic centroid transport is removed, every supported global
blur is represented by one centered positive exposure measure

```text
y(p) = A_mu x(p) + n(p)
     = sum_j mu_j x(reflect(p - d_j)) + n(p),
mu_j >= 0,  sum_j mu_j = 1.
```

Gaussian, disk, line, and curved-path names remain synthetic-generation
conveniences. They do not select reconstruction algorithms. The original PSF
atoms `(d_j,mu_j)` always define the same reflected gather operator `A_mu` and
the same matched scatter adjoint `A_mu*`.

## Shared descent

The conservative center-mixing basin and the exact exposure refinement both
use normalized positive transport:

```text
x_(k+1) = x_k * A_mu*(clip(y / A_mu x_k)) / A_mu*1.
```

The exact refinement begins from the stable positive/Fourier-gated basin. Its
correction is projected by the measured Fourier coverage of `mu`, so exact
spatial consistency cannot invent coefficients at exposure nulls. Two fitted
path endpoints seed sensitivity branches through the identical operator; they
measure gauge uncertainty and do not replace the accepted basin.

Positivity alone does not eliminate faint null-space echoes. The current
descent therefore also enforces a local fixed-point law. Every constant field
is unchanged by every positive unit-mass exposure operator, so a reconstruction
correction has zero authority wherever the observation has zero variance over
the operator's actual transport reach. The continuous authority is

```text
a_constancy(p) = Var_mu[y](p) / (Var_mu[y](p) + 0.004^2).
```

It is applied in both the center basin and exact path refinement. This is not
an edge detector or a line/Gaussian switch: the same transported first and
second moments define it for every positive measure. Those two moments are
batched through one operator call.

The current Python execution plan is also shared: one Eikonal chart, one
reflected index plan, and one `A_mu*1` normalization are reused by the main and
both endpoint branches. Flat gather plus weighted occupancy scatter preserve
the verified adjoint while removing repeated planning work.

An optional versioned C ABI executes the identical flat gather/scatter plan.
It owns no estimation or stopping decisions: Python supplies immutable
`int64` reflected indices and positive `float64` weights, and remains the
representation oracle and automatic fallback. Native promotion requires
gray/RGB forward parity, adjoint parity, inner-product closure, DC mass
normalization, and end-to-end metric parity.

The padded center-mixing basin has a separate immutable circular plan. It
constructs the kernel OTF once, carries each terminal prediction into the next
iteration, and represents real image transport on the nonredundant Hermitian
half-spectrum. Thus `N` positive passes require `N+1` forward and `N` adjoint
evaluations, rather than `2N+1` forward evaluations and a fresh kernel FFT at
every call. This is an execution change only; frozen basin images agree with
the original complex implementation within `4.3e-15`.

## A constraint coefficient, not a line branch

Differentiated uniform exposure supplies a useful residue-class recurrence
when the exposure measure is straight and directional. It is an auxiliary
correction `r_line - x`, not an independently selected reconstruction. Its
authority is one continuous product:

```text
a_line = a_trust a_descent a_residual a_anisotropy
         a_tangent a_extent (0.025 + 0.225 cos^2(2 theta)).
```

The factors measure, respectively, operator trust, already-spent descent
action, unresolved structured residual, covariance anisotropy, fitted tangent
coherence, path extent, and raster alignment. Center clouds drive anisotropy
coherence continuously toward zero. Curved paths drive tangent coherence
toward zero. Noise-consistent observations drive residual demand toward zero.
Straight supported paths retain finite authority. Omitting recurrence work
below `1e-6` maximum pixel authority is numerical dead-code elimination; it
does not alter the continuous reconstruction law. In the ringing battery this
withholds random-path recurrence calls at `1e-9`--`1e-8` authority while all
straight-line calls remain active at `0.0058`--`0.0344`.

The common exact reflected descent follows the weighted line correction for
every nonidentity positive operator, including Gaussian and disk measures.

## One observation-derived stopping law

Each descent stage compares its forward residual with a robust read-noise
estimate obtained from the residual's 3 by 3 high-pass component. With

```text
rho = RMS(y - A_mu x) / robust_read_sigma(y - A_mu x),
```

descent stops when `rho <= 1.1`. This is a finite discrepancy principle, not
a blur-family pass schedule. Low-noise structured residuals can spend the
maximum action budget; noisy observations stop before positive transport fits
their noise. The same residual excess supplies the continuous line-demand
factor `1 - exp(-max(rho - 1.1,0))`.

## Trust and scope

Known synthetic operators may use the exact exposure plan. A provisional
single-image kernel estimate remains at zero high-gain path authority: absolute
translation is a gauge, and one phase estimate is not yet a trusted exposure
operator. Multi-observation consensus is the intended promotion path.

The next blind continuation should optimize the latent image and optical
transport measure together. A principal identity mass, sparse displaced ghost
masses, diffuse center mass, and a spatial aberration chart are one forward
object; coherent residual phase changes the chart while incoherent residual is
carried as mixing/noise uncertainty. This is discovery of an aberrated lens
operator during descent, not preclassification of blur.

The current exactness claim covers global positive exposure with half-sample
reflection. Spatially varying flow, projective motion, rolling shutter,
occlusion, saturation, depth ownership, and optical turbulence require a new
forward/adjoint plan with visibility and Jacobian terms; they must not be
smuggled into a blur-family selector.
