# Carrier baseline and pixel-area audit

The original comparison viewer coupled the synthesis functional to its display
resolution: point images were 513×513, area images 65×65, and both were enlarged
to the preview width with nearest-pixel rendering. Switching modes therefore
also changed raster resolution by nearly eightfold per axis. Analytic truth
itself visibly shows the coarse pixel grid under that display.

The viewer now has an independent 65/513 output-size selector. Both point and
area use the selected size, as do their corresponding analytic references.
Metrics describe the displayed resolution. Error is a separate switch. Every
method uses the same pixel display; an unavailable paper-CONV* area result is
shown as unavailable instead of substituting a point image on one side.

## Matched-resolution results

The unchanged joint baseline was evaluated at the same point and area grids.

| Side | Point MSE against analytic points | Area MSE against analytic areas | RMS change from point to area |
|---|---:|---:|---:|
| 65 | 3.383819e-6 | 3.232580e-6 | 4.772133e-4 |
| 129 | 3.472329e-6 | 3.428628e-6 | 1.284051e-4 |
| 513 | 3.523576e-6 | 3.520605e-6 | 1.048470e-5 |

At 65×65 the analytic field's own point-to-area RMS change is 4.218780e-4.
The change in reconstruction error caused by the functional switch has RMS
1.884218e-4. Integration makes a small change and slightly reduces mean-squared
error in all three comparisons. This does not establish that every local error
decreases. Both point and area baseline images retain the reconstruction's
pre-existing shape error.

Order-four and order-eight Gauss integration of the atlas, each split at source
knots, agree within 4.45e-16. These rules integrate the tensor-quintic patches
exactly up to floating-point arithmetic. A new independent randomized
full-degree test checks the same identity. This audit uses no experimental
concentration at any stage.

## Baseline construction stages

At 65×65 the raw collocation proposal has point MSE 1.558008e-6 and area MSE
1.450999e-6. Applying the shared support-range clipping increases them to
3.383819e-6 and 3.232580e-6. The subsequent joint-current admission produces
identical point and area errors, with no further field change in this carrier.
The same stage attribution holds on the 129 and 513 grids.

The source and analytic ranges show why a sample-range requirement excludes
part of the known carrier:

| Field | Minimum | Maximum |
|---|---:|---:|
| 17×17 source samples | 0.19622166 | 0.81435976 |
| Analytic field on 513×513 grid | 0.17463069 | 0.81714526 |
| Raw collocation field on 513×513 grid | 0.17629916 | 0.81680783 |

The true trough is 0.02159097 below the smallest source sample. Enforcing the
source-sample range cannot recover that intersample trough, even when the raw
proposal is close to it. Range certification and fidelity to this analytic
field therefore conflict under the selected sample-range constraint. The
resulting baseline distortion is present before target integration.

The coarse preview grid and this baseline distortion are separate effects.
The previous blanket attribution of visible grid structure to the experimental
concentration was too broad. This audit identifies a baseline limitation and a
viewer confound; it does not change the production WASM constructor or establish
its exact numerical behavior on this field.

## Reproduction

From BFFT with the existing numerical Python environment:

```sh
python -m experiments.conv_warp.bounded_sharpening.carrier_integral_audit
python -m experiments.conv_warp.bounded_sharpening.resolution_assets
python -m experiments.conv_warp.bounded_sharpening.paper_report
python -m unittest experiments.conv_warp.bounded_sharpening.test_geometry \
  experiments.conv_warp.bounded_sharpening.test_paper_images -v
```

The output directory `output/support_geometry/conv_bounded_sharpening/paper-images`
contains `carrier-integral-audit.json`, `carrier-integral-audit.png`, independent
point and area PNGs for each method and resolution, and
`resolution-metrics.json`. Original paper-image experiment results remain
available in `results.json`; the revised viewer uses the new resolution-specific
metrics. The live comparison remains local at port 8899.
