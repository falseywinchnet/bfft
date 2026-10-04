# Geometric projective quadrature

Let a source-cell triangle be T, and let its CONV patch be

    U(x,y) = Σᵢⱼ Pᵢⱼ Bᵢ⁵(x) Bⱼ⁵(y).

After changing variables from the output pixel to the source plane, the
36 footprint weights are

    Mᵢⱼ(T) = ∫T Bᵢ⁵(x) Bⱼ⁵(y) C / |ℓ(x,y)|³ dx dy,

where ℓ is the affine denominator of the forward homography and C contains
its absolute determinant and the source-coordinate normalization. A valid
footprint has a denominator of one sign. Its extrema occur at vertices.
Writing c = (ℓmin + ℓmax)/2 gives

    r(T) = (ℓmax − ℓmin) / (2 |c|).

This single dimensionless geometric quantity selects a positive-interior
Xiao–Gimbutas triangle rule. The selected rule evaluates the rational density
C/|ℓ|³ directly at its nodes. There is no density-series evaluation, channel
amplitude scan, tolerance test, or embedded-rule comparison in the renderer.

| Maximum r | Polynomial degree of rule | Nodes |
|---:|---:|---:|
| 0 | 10 | 25 |
| 0.000009 | 11 | 28 |
| 0.00036 | 12 | 33 |
| 0.0023 | 13 | 37 |
| 0.0074 | 14 | 42 |
| 0.0159 | 15 | 49 |
| 0.0276 | 16 | 55 |

If r exceeds the final entry, split the triangle at its three edge midpoints
and apply the same geometric rule to the four children. For a nonsingular
homography, ℓ is bounded away from zero on the compact footprint, while the
variation of ℓ on a child decreases with its diameter. The subdivision therefore
terminates. This partitions an integration domain; the admitted surface and
its controls are fixed. Axis-aligned affine rectangles retain their exact
separable integrated Bernstein weights.

## Operator accuracy

The finite lookup table is validated once. No part of this derivation is
evaluated while selecting a rendering rule.

Put z = ℓ/c − 1. The expansion

    (1+z)⁻³ = Σₖ₌₀ᴷ (−1)ᵏ (k+1)(k+2) zᵏ/2 + Rᴷ(z)

has the uniform remainder bound, for |z| ≤ r < 1,

    |Rᴷ(z)| ≤ Tᴷ(r)
    Tᴷ(r) = rᴷ⁺¹/2 [r(1+r)/(1−r)³
                       + (2K+5)r/(1−r)²
                       + (K+2)(K+3)/(1−r)].

Each tensor Bernstein basis function has total degree at most ten. A rule
exact through degree 10+K integrates its product with the displayed polynomial
exactly. The rule has positive weights, and the 36 basis functions are
nonnegative and sum to one throughout the source cell. Consequently the
sum of the absolute errors of all 36 weights satisfies

    Σᵢⱼ |Qᵢⱼ − Mᵢⱼ| ≤ 2 |T| C |c|⁻³ Tᴷ(r).

The exact Jacobian mass m(T) is at least
|T| C |c|⁻³ (1+r)⁻³. Thus

    Σᵢⱼ |Qᵢⱼ − Mᵢⱼ| / m(T) ≤ 2 (1+r)³ Tᴷ(r).

Every table entry satisfies this inequality with right side below 10⁻⁹.
The inequality is an operator statement: it applies to every assignment of
the patch controls, including a control pattern chosen to maximize the weight
defect. It is independent of source content and subpixel phase. Summing
triangles preserves the same mass-relative bound.

The renderer retains exact target-polygon coverage and a common reference
value R for each output pixel, accumulating (Pᵢⱼ−R) against the weights.
For each channel the absolute error in its pixel average is therefore at most
10⁻⁹ times the largest absolute residual control, times pixel coverage,
in exact arithmetic. Constant fields retain exact target coverage through
this reference term. Floating-point coordinate, quadrature-table, accumulation
and source-storage errors are checked separately by the retained tests.

## Reproduction

`baseline-area-kernel.c` freezes the former production integrator. Its rebuilt
binary hash is b1c90245331dace5a2e3032143f032520668482bd9569cd074698fc56c5d3052.
`geometry-area-kernel.c` contains the replacement; `build.py` adds study-only
counters and a 36-weight probe. `triangle-rules.json` records the pinned rule
source and license; `COPYING.xiao-gimbutas` accompanies the distributed table.

- `python3 research/conv-area-acceleration/build.py`
- `node research/conv-area-acceleration/check-geometry.mjs results.json`
- `node research/conv-area-acceleration/check-geometry-bank.mjs banks.json`
- `python verify-geometry-banks.py --bfft /path/to/bfft --input banks.json --output check.json`
- `VARIANTS=baseline,compact,geometry-plan node --expose-gc research/conv-area-acceleration/profile.mjs timing.json --large`

The bank verifier requires mpmath and uses the BFFT addendum's algebraic
rational/logarithmic integral at 220 decimal digits. It checks every lookup
threshold from both sides, skew and thin triangles, and denominator ratios
up to 1999:1. The image test checks smooth, alternating, noise, constant,
impulse and near-transparent inputs under six maps and three output sizes.
It compares pixel values with the former integrator tightened to 10⁻¹²,
compares encoded frames with the former 10⁻⁶ render, and verifies that all
six input images produce identical rule and subdivision counts.

## Recorded validation and performance

The final instrumented kernel is
`b408fdee117f4e04a6ca1c665cbba19c1fb800e22566988966ddda83f421ec97`.
`results/geometry-check.json` contains 108 image/warp/size cases and 8,640
sampled pixel comparisons. The largest premultiplied-channel difference from
the tightened reference is 1.325606291402437 × 10⁻¹³. All image kinds produce
identical geometry counters. Changing the API tolerance from 10⁻¹² to 0.1
leaves values identical.

Of 1,170,000 encoded channel bytes, 22 differ from the former 10⁻⁶ renderer,
each by one byte level. All occur in the deliberately near-transparent input
whose source alpha is zero or 1/255. The other 102 complete frames, including
all opaque tests and the ordinary variable-alpha noise tests, are identical.

`results/geometry-bank-check.json` compares 57 complete 36-weight banks with
the analytic oracle. The largest relative L1 weight error is
8.692335112030978 × 10⁻¹⁰, on the very thin triangle at denominator ratio
1999:1. This includes Float64 geometry, subdivision and accumulation error.
All weights remain nonnegative, and all seven rules are exercised.

Median timings on the M4 Mini, Node v25.2.1, five warm runs:

| Source | Output | Map | Former kernel | Geometry only | Speedup |
|---|---|---|---:|---:|---:|
| 257 × 257 | 257 × 257 | Identity | 37.62 ms | 35.21 ms | 1.07× |
| 257 × 257 | 257 × 257 | Affine skew | 125.77 ms | 99.68 ms | 1.26× |
| 257 × 257 | 257 × 257 | Perspective | 180.18 ms | 124.46 ms | 1.45× |
| 257 × 257 | 257 × 257 | Strong perspective | 180.38 ms | 136.10 ms | 1.33× |
| 257 × 257 | 65 × 65 | Perspective | 82.68 ms | 55.97 ms | 1.48× |
| 257 × 257 | 513 × 513 | Perspective | 371.10 ms | 261.26 ms | 1.42× |
| 1536 × 1024 | 1536 × 1024 | Perspective | 4481.92 ms | 3010.07 ms | 1.49× |

All seven benchmark output hashes are identical to the former renderer.
Timings include integration, point-bilinear comparison, encoding and output
copies, with the source already prepared. The larger case retains 9,451,560
triangles and reduces quadrature locations from 460,373,248 to 308,823,225.
The earlier compact-rule candidate, which retains the amplitude/tail selector,
took 2993.05 ms in that case. Geometry-only selection therefore obtains nearly
the same measured speed with a fixed content-independent operator budget.

The initial and final receipts are `geometry-profile-initial.json` and
`geometry-profile-final.json`. The final arithmetic reuses the triangle's
three denominator values to evaluate ℓ in barycentric coordinates, avoiding
two source-coordinate divisions per quadrature node. Affine triangles reuse
one constant density. The production instrument adopts this final kernel and
reports “Geometry-selected quadrature” in place of a per-frame tail estimate.
Its numerical, source-construction and display regression suite passes.
The local browser completes a 513 × 513 optical plate → 512 × 512 area warp
and exposes the new status with no console warnings or errors.


## Canonical publication

The geometry-selected positive quadrature is the canonical numerical projective
area construction. Its complete proof is included in the CONV* warp addendum
and the composed CONV paper under “Geometric compilation of projective banks.”
The exact projective edge compilation supplies the analytic reference.
