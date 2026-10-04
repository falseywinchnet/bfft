# M4 results: direct boundary measure is promising; image-only contraction fails

23 September 2026. The investigation ran on the selected **Apple M4 Mac mini**.
All results are retained under `output/support_geometry/conv_fast_aa/`.
The target remains a general method with FXAA-class cost and 4× MSAA-class quality.
That general target is **not yet achieved**. There is a concrete restricted
success, a rejected image-only candidate, and a demonstrated visibility obstacle.

## Decision

1. **Reject the first-jet tangent filters as an MSAA-quality replacement.** They
   improve several still-image edge cases, but miss thin geometry, alter already
   covered input, and retain large phase errors. Higher-order CONV interpolation
   cannot supply missing subpixel phase simply by being integrated exactly.
2. **Retain the boundary-measure renderer prototype for further development.**
   For the tested constant-color triangles and one triangulated surface, exact
   coverage reaches output-rounding accuracy at FXAA-class or lower GPU cost.
   Its conservative support and edge arithmetic are part of the measured draw.
3. **Do not promote it as general 3-D antialiasing.** The coincident-overlap
   counterexample fails. Per-triangle exact area is insufficient: one needs the
   area of the visible surface partition and its color measure. This remains
   unresolved, along with textured/varying shading and full moving-scene tests.
4. **Do not claim a new field-wide invention.** Analytic coverage and directional
   postfilters have substantial prior art. The README gives the derivation,
   precise relation to CONV, and primary references.

## Actual Metal renderer comparison

Final run: `geometry_final/geometry_timing.json`; 21 rotating-order observations,
eight draws per command buffer, RGBA8Unorm, warmed runtime-compiled pipelines.
Every entry includes a clear and the complete instanced draw; FXAA includes its
postprocess, and MSAA includes hardware resolve. Geometry generation/upload,
shader compilation, and display presentation are excluded for all methods.

Median GPU milliseconds:

| Resolution / scene | Triangles | Point | Point + FXAA 3.11 | Hardware 4× MSAA | Boundary sum |
|---|---:|---:|---:|---:|---:|
| 1920×1080 / sparse | 464 | 0.0645 | 0.2729 | 0.0682 | **0.0716** |
| 1920×1080 / dense | 7,973 | 0.0738 | 0.4044 | 0.0818 | **0.3887** |
| 3840×2160 / sparse | 1,947 | 0.3279 | 1.1626 | 0.3439 | **0.3535** |
| 3840×2160 / dense | 32,026 | 0.3670 | 2.0114 | 0.3712 | **1.5218** |

Sparse-scene boundary timing is about 3.3–3.8× faster than point-plus-FXAA here.
The 1080p dense difference is only 3.9%, so treat that as similar cost rather than
a robust speed advantage. Earlier dense 4K runs put FXAA around 1.64 ms and the
boundary path around 1.51 ms; the larger final-run gap is not a universal speedup.
P95 data is retained, including scheduling outliers in several baselines.

Hardware MSAA is the fastest antialiased draw on these extremely simple M4
scenes. The result does not support the assumption that MSAA is intrinsically
expensive relative to FXAA on this hardware. These tests have trivial shading,
no depth complexity, and no textured materials.

The original bounding-rectangle implementation took 0.152/0.490 ms at 1080p
(sparse/dense). Hoisting the existing primitive algebra into the vertex stage
reduced that to 0.095/0.479 ms. A conservative centroid-dilated triangle support
then reduced it to 0.072/0.383 ms in its first run. The final support chooses the
smaller of that conservative triangle and the bounding rectangle, with exactly
the same pixel functional. Earlier runs are preserved as `geometry_v1`, `v2`,
and `v3`; they are development timings, not independently randomized ablations.

## Coverage quality and the composition failure

Quality readbacks use 512×512 targets and a separate CPU polygon-clipping oracle.
These are actual Metal triangles and actual hardware MSAA, rather than the
four-sample synthetic control in the image-only battery.

| Scene | FXAA boundary MSE | Hardware 4× MSAA boundary MSE | Boundary sum boundary MSE | Boundary sum maximum error |
|---|---:|---:|---:|---:|
| 49 separated triangles | 0.0131291 | 0.00847872 | **0.00000125150** | 0.00196052 |
| 961 separated triangles | 0.0206157 | 0.00832756 | **0.00000124952** | 0.00196083 |
| 225 subpixel slivers | 0.00248111 | 0.00634454 | **0.00000129386** | 0.00196057 |
| 225 grid-aligned triangles | 0.166918 | 0.00000384468 | **0.00000384468** | 0.00196078 |
| 450 triangles forming shared-edge squares | 0.0314796 | 0.00973219 | **0.000000031989** | 0.000786145 |
| 450 coincident triangle copies | 0.0158398 | 0.00850024 | **0.0611446 — FAIL** | **0.499996** |

The first four rows reach approximately half an 8-bit level in maximum error.
The shared-edge row also passes the one-byte gate. Boundary MSE selects pixels
with fractional reference coverage; therefore the whole-image maximum is also
reported, so an interior mesh seam cannot be hidden by the boundary mask.

Ordinary alpha blending of the exact per-triangle areas leaves a 0.25098 maximum
seam error in the shared-edge squares. Additive area accumulation removes that
seam in the tested mesh. However, two coincident opaque triangles cover the same
visible region only once. Alpha-over then errs by up to 0.25289 and additive
coverage by almost 0.5. The failure is in visibility/composition, not accuracy of
the individual triangle integral.

The CPU reference mass agrees with triangle area to about 1.5e-11 on the first
two scenes. The M4 reports four sample positions
`(0.375,0.125), (0.875,0.375), (0.125,0.625), (0.625,0.875)` in pixel coordinates.
Geometry quality artifacts and test logs are in `geometry_final/`.

## Image-only screen: 456 cases

Final image-only results: `v3/quality.json`. The tests vary angle and eight
subpixel phases, include thin strips, disk/ring/corner geometry, high-frequency
signals, already-covered input, and equal-luminance color edges. Straight-edge
and strip area references are analytic. Other references use a 32×32 midpoint
grid; their absolute values are numerical references, not exact geometry proofs.

Mean error over each case's declared region, then averaged over the family:

| Family | Point | FXAA 3.11 | Tangent box | Quintic-weight tangent | RGB tensor tangent | Four coverage samples |
|---|---:|---:|---:|---:|---:|---:|
| Straight edges | 0.020210 | 0.005982 | 0.006461 | 0.007474 | 0.006461 | **0.002787** |
| 0.35-pixel strips | 0.044925 | 0.014452 | 0.045621 | 0.045659 | 0.045621 | **0.006863** |
| 0.75-pixel strips | 0.034348 | 0.027355 | 0.024465 | 0.021587 | 0.024465 | **0.005057** |
| Thin ring | 0.004472 | 0.003236 | 0.002838 | 0.002667 | 0.002838 | **0.000529** |
| Equal-luminance color edge | 0.006588 | 0.006588 | 0.006588 | 0.006588 | **0.000941** | 0.001406 |
| Already covered input | **0** | 0.008095 | 0.007301 | 0.004755 | 0.007301 | 0.002380 |

The four-sample value in the last row is a second filtering operation, not a
recommended baseline: the intended result there is the unchanged input.
The RGB tensor fixes the chromatic-edge blind spot and beats the four-sample
control on that particular edge family. It does not fix the other failures.

Phase error is measured as successive-frame change of `(output-reference)` at
1/8-pixel motion, not raw frame difference. For 0.75-pixel strips, box-tangent
error-change MSE is 0.02710, versus FXAA 0.01421 and four samples 0.01001.
For the thin ring it is 0.002182, versus 0.000880 and 0.000811. Better still-image
error on these families therefore does not establish better motion quality.

The original admitted CONV box reference is evaluated at the fixed .375 phase,
with both factor orders, and is retained separately. Its subset aggregates must
not be compared as if they covered all eight phases. It is not evidence that
the full joint atlas runs at FXAA cost.

## Image-only cost and numerical verification

Final RGBA8Unorm M4 GPU kernel times, milliseconds:

| Resolution / input | FXAA | Tangent box | RGB tensor tangent |
|---|---:|---:|---:|
| 1080p / flat | 0.1206 | 0.1833 | 0.2561 |
| 1080p / sparse stripes | 0.1277 | 0.1819 | 0.2995 |
| 1080p / dense noise | 0.3581 | 0.2219 | 0.4704 |
| 4K / flat | 0.6506 | 0.8804 | 1.0324 |
| 4K / sparse stripes | 0.6524 | 0.8728 | 1.1987 |
| 4K / dense noise | 1.8336 | 0.9584 | 1.8821 |

These are complete filter dispatches; there is no precomputed CONV field whose
construction is hidden. Luma is packed in the uploaded source alpha for FXAA and
the scalar tangent variants, as an engine could supply it from a preceding pass.
This is not the same workload as the complete primitive renderer table.

The final float32-input NumPy/Metal comparisons pass:

- Box contraction maximum difference: 9.93e-8.
- Quintic-weight contraction maximum difference: 9.71e-8.
- RGB tensor contraction maximum difference: 2.96e-6.
- All outputs finite; copy error below 3e-8.
- Ten focused mathematical tests pass on the M4, including independent Gaussian
  quadrature, range/affine/symmetry checks, polygon mass, shared-edge composition,
  subpixel slivers, and conservative support containment.

v1/v2 numerical comparison receipts are deliberately retained. Comparing to
unquantized source/luma and normalizing arbitrarily small jets caused large
direction discrepancies. Paired differences, a relative float32 zero rule, and
an identical-input reference resolve the comparison in v3. The v1 RGBA32Float
timings were bandwidth dominated; they are not used to claim final performance.

## Next construction required by the original objective

The useful representation must carry **visible color-area**, not independent
triangle opacities. Shared interior boundaries cancel/add correctly within one
surface; occluded overlap must not be counted twice. The next meaningful test is
a retained visible-boundary representation with explicit source provenance and
a complete rendering cost, including any visibility work. Merely supplying
oracle-visible boundaries would not solve the problem.

No solver, extra raster feature classifier, adaptive source subdivision, or
per-pixel active-state search has been added to CONV. The existing CONV source,
native backend, and production applications are unchanged by this experiment.
