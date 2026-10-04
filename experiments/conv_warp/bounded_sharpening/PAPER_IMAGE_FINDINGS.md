# Quintic concentration on the paper's technical fields

14 September 2026. The tested concentration proposal is rejected as a general
image refinement. It preserves the stated range and first-derivative constraints
but introduces visible grid texture and increases analytic error on all four
selected paper fields. The failure is present before display interpolation and
remains after target pixel-area integration.

## Exact source selection

The fields are the four cases selected by `_two_dimensional_fidelity` in
`experiments/conv_synthetic_evidence.py` for `technical_synthetics.pdf`, included
in the fused paper's “Technical synthetic fields” figure. Each has 17×17 nodal
source samples and a 513×513 endpoint-aligned point display:

- Edge: angle 37.5 degrees, offset +0.25/16.
- Carrier: angle 37.5 degrees, phase pi/2.
- Curved: radial transition, offset +0.25/16.
- Crossing: phase pi/2.

The original analytic functions and source sampling are retained. These fields
are point-sampled analytic images; they are not antialiased pixel-coverage
rasters. Their exact analytic reference allows an independent fidelity check.
The original paper CONV* point results and Lanczos-3 are retained alongside the
canonical Float64 joint-atlas baseline, so changes of construction remain visible.

## Declared proposal and bounds

The proposal is the preceding study's curvature-weighted tensor quintic current
concentration. Unit strength tends toward the tensor corner profile with each
one-dimensional control row `(a,a,a,b,b,b)`. Source vertices stay fixed. Existing
support-range and joint-current constraints remain active. First-derivative
Bernstein coefficients are certified on half-cells, using caps 1, 2, and 4 times
the base atlas's largest absolute whole-cell first-derivative coefficient. A
fourfold proposal is also tested with cap 2. Bounds use source-index coordinates;
all cases have the same grid spacing.

No source-cell moment or higher derivative-trace equalities are enabled in this
screen. First-derivative certification does not prohibit oscillatory structure
inside the allowed value and derivative ranges.

## Results

Full-domain 33×33 pixel-area MSE against the analytic pixel averages, divided by
the corresponding joint-baseline MSE:

| Field | Unit proposal, cap 2 | Fourfold proposal, cap 2 |
|---|---:|---:|
| Slanted edge | 1.0451 | 1.0453 |
| Carrier | 16.1850 | 151.7247 |
| Curved edge | 1.1300 | 1.1723 |
| Crossing | 5.4478 | 32.7361 |

These are error ratios, not percentages of visible image corruption. The carrier
and crossing errors start small; their absolute errors are retained in JSON.
The comparison plate independently exposes the resulting distortion. Unit
concentration alters smooth ridges; the larger proposal produces conspicuous
cell-scale grid texture. These changes occur despite exact source-vertex
recovery and zero sampled global source-range excursion in every joint variant.

Along the declared outward horizontal normal of the curved edge, the 10–90%
width decreases from 0.093077 to 0.087895 normalized domain units, a 5.57%
contraction. Its half-level displacement from the analytic edge grows from
0.004138 to 0.006216 domain units, or from 0.06621 to 0.09946 source pixels.
The transition is steeper but less correctly located. The source-atlas mean
changes by +0.00036747. Along the slanted-edge normal the width instead grows
from 0.092651 to 0.095229, and the half-level error increases from 0.002945 to
0.004913. Concentration is not a uniform sharpening operation after admission.

Cap 2 and cap 4 produce identical unit-strength fields in all four cases. Cap 1
changes the carrier by at most 0.00471734 in value and leaves the other three
unchanged. The cap-2 first-derivative limit is therefore not the binding
restriction for these four proposals. Tightening an inactive global derivative
allowance does not select the correct local phase or waveform.

The tensor target concentrates transitions around cell midpoints. The observed
grid alignment is consistent with this imposed phase; it is not a browser
scaling artifact. The screen motivates a proposal tied to the existing feature
phase and orientation, together with moment and derivative-trace constraints
where required. It does not validate such a replacement. The existing paper
operator and live browser kernel are unchanged.

Lanczos's pixel-area source-range excursions are 0.01697 (edge), 0.01729
(carrier), 0.02874 (curved), and 0.00223 (crossing), in normalized [0,1] units.
The bounded variants avoid those excursions while still producing their own
in-range shape errors. Range excursion alone is not a complete ringing or
fidelity measurement.

## Numerical verification and artifacts

Five additional focused tests pass alongside the eight geometry tests:
independent analytic evaluation reproduces the paper fields; batched polynomial
evaluation matches the existing point evaluator; exact atlas integration
reproduces constants and linear fields; and reference quadrature refines within
1e-11. The largest observed order-8 to order-12 analytic pixel-integral difference
is 6.90e-12. Lanczos integral differences are below 1e-15. Atlas pixel integrals
use order-four Gauss quadrature split at source knots and are exact for the
quintic tensor patches up to floating-point arithmetic.

Point scores retain the paper's 33×33 held-out mask for comparison. JSON also
contains full-domain MSE. The table above uses full-domain pixel-area MSE. Area
displays use 65×65 integrated pixels enlarged with nearest-pixel inspection.
Point displays use 513×513 values. All images share [0,1] display limits;
unclipped values supply metrics. Signed error images share the fixed ±0.10
scale. The original factor CONV* is included in the point comparison; no
substitute area operator is presented under its name.

Run from the BFFT root with the existing numerical Python environment:

```sh
python -m unittest experiments.conv_warp.bounded_sharpening.test_geometry \
  experiments.conv_warp.bounded_sharpening.test_paper_images -v
python -m experiments.conv_warp.bounded_sharpening.paper_images \
  output/support_geometry/conv_bounded_sharpening/paper-images
python -m experiments.conv_warp.bounded_sharpening.paper_report
```

The retained output directory contains `index.html`, `comparison.png`,
`comparison.pdf`, `errors.png`, all separate images, `images.npz`, and
`results.json` with source parameters and hashes. The HTML supports field and
method selection, point/area/error modes, and synchronized zoom and scrolling.
All computations used the existing local release-validation Python environment.
The Mini was inspected first; its available runtimes lacked required packages.
No software was installed, and no production release was changed.
