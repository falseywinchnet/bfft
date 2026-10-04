# Fine-line coverage and the joint CONV warp

14 September 2026. Diagnostic against the unchanged production WASM kernel.

The muted thin line is not an unavoidable consequence of suppressing ringing.
The current warp reconstructs source raster values as point values and bounds
its continuous controls by those samples. An antialiased raster of a narrow
line already contains coverage averages. Interpolating those averages as points
spreads the line, and integrating the resulting profile adds another averaging
operation. Exact target integration preserves the integral of that reconstructed
field; it does not make its source-cell averages equal the input raster.

## The actual optical-test border

The source generator sets a one-pixel stroke at `0.045 * 513 = 23.085` canvas
coordinates, with stroke RGB `(51,87,103)` over `(233,229,207)`. A Chrome readback
at row 256 gives:

| Column | RGB |
|---|---|
| 21 | 233,229,207 |
| 22 | 158,170,164 |
| 23 | 126,146,146 |
| 24 | 233,229,207 |

The red-channel deficits relative to the intended stroke contrast are 75/182
and 107/182: approximately 41.2% and 58.8%. No fully covered border sample
exists on this straight segment. The geometric coverages before byte rounding
are 41.5% and 58.5%. In this border's local source support, the admitted range
therefore prevents reconstruction of the original stroke colour between samples.

## Measured stages

`study.mjs` runs the unchanged source constructor on 12 vertical line cases:
widths 0.5, 1 and 2 source pixels, each at four subpixel positions. The input is
exact box coverage stored in Float32 and constant along the second axis. It
reads the actual native controls before support-range clipping, after clipping,
and after joint-current admission. Each case is evaluated at four output
spacings and four output phases, for 192 output profiles.

For the one-pixel line whose two samples are 0.415 and 0.585 in normalized
contrast, the control-stage measurements are:

| Stage | Peak contrast | Integrated contrast |
|---|---:|---:|
| Source raster | 0.585000 | 1.000000 |
| Factor/collocation proposal | 0.620216 | 1.049306 |
| After support-range clipping | 0.585000 | 1.029215 |
| After joint-current admission | 0.585000 | 1.029215 |

The final joint-current stage makes no material change in these constant-axis
cases. Range clipping removes the intersample peak, while the resulting field
still has greater total contrast than the original line. Its reduced peak is
therefore accompanied by broadening, rather than simply loss of contrast mass.
The larger two-dimensional admission may matter on other geometries; this stage
attribution is specific to the measured lines.

At unchanged output spacing and alignment, point synthesis returns the source
peak 0.585000; pixel-area integration returns 0.521924. Direct integration of
the original line returns 0.585000. The roughly 10.8% additional peak attenuation
in the area result is not required by the target pixel's actual coverage.

## A finite coverage-preserving witness

Consider background and foreground colours with known unit normalized contrast.
Two adjacent source pixel cells have contrast averages `a` and `b`, with both
strictly between zero and one. If one connected line crosses their shared
boundary `h = i + 1/2` and lies in those cells, its edges are directly recovered:

\[
 L=h-a,\qquad R=h+b.
\]

Define the monotone quintic step

\[
 S(t)=\begin{cases}
 0,&t\le0,\\
 10t^3-15t^4+6t^5,&0<t<1,\\
 1,&t\ge1.
 \end{cases}
\]

For positive transition width `e`, set

\[
 d_e(x)=S\!\left(\frac{x-L}{e}+\frac12\right)
       -S\!\left(\frac{x-R}{e}+\frac12\right).
\]

The finite bound

\[
 0<e\le\min(a+b,2a,2b,2(1-a),2(1-b))
\]

keeps the two transitions disjoint and inside their respective source cells.
Since `S(1-t)=1-S(t)`, smoothing each transition symmetrically about its edge
preserves both source-cell integrals. Thus their averages remain exactly `a`
and `b`, and the total contrast integral remains `a+b`. Also,
`S'(t)=30t^2(1-t)^2` in the transition, so `d_e` has one rise and one fall,
no additional extrema, and values in `[0,1]`. It is C2 throughout. The formula
uses direct algebra and finite polynomial integration, without a solver.

The profile has additional subpixel knots and a latent foreground/background
range. It is not feasible under the current nodal-cardinality and sample-minimum
constraints, nor is it a replacement admission on the same fixed control grid.
It is an explicit demonstration that smoothness, preserved coverage and a strong
thin line can coexist without ringing under a coverage-based source model.

For `a=0.415`, `b=0.585`, the illustrated member takes `e=0.25`. This is a
specific witness from the admissible family, not an edge width inferred uniquely
from the two averages. The target grids below have zero phase offset:

| Output spacing in source pixels | Exact original-line peak | Current CONV area | Coverage-preserving quintic |
|---|---:|---:|---:|
| 1 | 0.585000 | 0.521924 | 0.585000 |
| 0.5 | 1.000000 | 0.561168 | 1.000000 |
| 2 | 0.457500 | 0.365102 | 0.457332 |
| 4 | 0.250000 | 0.257304 | 0.250000 |

For a genuine small target coverage `f`, the area-averaged colour is
`(1-f) * background + f * foreground`. Its contrast reduction is necessary.
The last row illustrates both that necessary reduction and the current field's
small excess integrated contrast. Increasing all thin-line peaks would not
correct this operator consistently.

## What is identified, and what remains to be constructed

The central source condition for coverage data is

\[
 |C_i|^{-1}\int_{C_i}U(p)\,dp=Y_i,
\]

whereas the current atlas imposes `U(p_i)=Y_i`. The existing target integrator
can integrate either declared representation, but cannot repair a mismatched
source reconstruction. A coverage-based extension needs source-cell mass and
subpixel geometry alongside its current and range rules.

A general uploaded raster does not necessarily identify its latent stroke colour,
width or position. For example, a half-pixel line moved wholly within one source
cell has unchanged cell averages but different finer-grid coverage. Unknown
foreground contrast introduces an additional width/contrast ambiguity. The
closed-form two-cell result above uses an identified background, foreground and
connected line. Extending it to measured oblique or curved structures requires
joint geometric evidence and shared source-cell constraints; the present witness
does not certify that general extension.

The next construction to test is a finite, source-coverage-consistent geometric
profile with shared edges and explicit admissibility, transported once through
the projective map. Its tests should measure source-cell recovery, line contrast
mass, phase-dependent width and peak, extra extrema, and target-coverage error.
Simply weakening sample bounds or increasing filter support does not establish
these properties.

## Reproduction and checks

From the BFFT root:

```sh
node experiments/conv_warp/line_coverage/study.mjs \
  /Users/ultimussecundai/paymenottowork/web/instruments/conv-perspective
node experiments/conv_warp/line_coverage/check_profile.mjs
python3 experiments/conv_warp/line_coverage/plot.py
```

The complete measurements are in
`output/support_geometry/conv_line_coverage/results.json`; the prototype sweep
is in `profile.json`, and `line-coverage.png`/`.pdf` show the matched profiles.
The results retain the production WASM and source-construction SHA256 hashes.

Independent one-dimensional polynomial-exact quadrature agrees with native
2-D pixel integration within `6.7e-16` on the measured probes. Target partitions
recover the reconstructed-field mass within `1.4e-8`, and source nodes recover
the input within `2.2e-8` (Float32 storage). Across 108 coverage/transition-width
combinations, the C2 witness preserves source-cell means within `1.5e-15` and
total contrast within `4.7e-15`; sampled derivative signs use a `1e-12` roundoff
threshold, while the no-extra-extrema property follows analytically above.

No production website code, kernel, or paper was changed for this diagnostic.
