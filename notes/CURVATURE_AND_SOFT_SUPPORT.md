# Curvature-limited population and owner-free soft support

Date: 2026-07-27

## The Pikachu cheek diagnosis

The red cheek and outlined paw distinguish two failures that ordinary edge
energy conflates.

On the 475 px control, the frozen tensor at the cheek boundary had:

- median coherence 0.997;
- median predicted aspect ratio 25:1;
- median tangent-director curvature 0.047 radians/pixel;
- median curvature sagitta ratio 36.5;
- 7.7 cells of raw support mass over the measured boundary band.

Five final hard cells contained pixels from both the red patch and its yellow
annulus. The paw received 64.0 cells of support mass and 110 touching cells
because its corners and crossing strokes give the tensor a second eigenvalue.
The smooth closed cheek is locally rank one everywhere, so determinant density
mistook it for one indefinitely straight support.

If a tensor predicts tangent and normal semi-spans

    a = 1 / sqrt(lambda_low), b = 1 / sqrt(lambda_high),

a contour of curvature kappa departs from its tangent by approximately
`kappa*a^2/2`. Requiring this sagitta to remain within the normal span gives
the population correction

    factor = sqrt(max(1, kappa*a^2/(2*b))).

The director derivative is computed in doubled-angle form. It is invariant to
the sign of the tensor eigenvector and requires no angle unwrapping.

Measured Pikachu controls:

| allocation | cells | PSNR | cheek-band RGB MSE |
| --- | ---: | ---: | ---: |
| straight support law | 730 | 25.250 | 0.03967 |
| curvature law, fixed count | 728 | 25.145 | 0.02245 |
| straight law, 1199-cell control | 1192 | 26.994 | 0.02683 |
| curvature law, natural count | 1194 | **27.784** | **0.01400** |

Thus curvature placement halves the cheek error at matched population and
adds 0.79 dB over merely spending the same larger count. At the old count it
trades 0.10 dB globally to cut the specific curved-boundary error by 43%.

Across 256 px controls, natural curvature population improved RGB PSNR on all
five images: camera +2.68 dB, Chelsea +0.43, coins +0.46, astronaut +1.53,
and Pikachu +1.58. Camera exposed the companion failure: fine cells improved
RGB and texture but their hard identity leaked into the cartoon score.

## Soft support without deletion or a runner

Let the hard first-arrival cell indicators be the initial support weights:

    w_i(0, x) = 1[x belongs to C_i].

Evolve every indicator under one shared transport-gated heat operator:

    w_i(t) = exp(t L_g) w_i(0).

`L_g` preserves a constant field, therefore

    sum_i w_i(t, x) = 1

for every pixel and time. This is a partition of unity without storing a
pixels-by-sites matrix. Diffusing a rendered site-colour field computes
`sum_i w_i c_i` directly. The anisotropic heat kernels are the elongated,
overlapping shapes seen in SAD-style site-ID views.

Each edge conductance is:

    target-colour agreement / BFFT transport action squared.

Consequently, an unsupported boundary in a homogeneous region conducts and
loses identity. A real red/yellow or object/background edge blocks passage.
No pair is ranked, no site is selected for merging, and no site is deleted.
The optional readout is accepted only when the measured RGB + one-stage
cartoon + one-stage texture objective does not increase.

At 256 px with curvature population and 16 simultaneous heat steps:

| image | accepted | PSNR | cartoon MSE | texture MSE | objective |
| --- | --- | ---: | ---: | ---: | ---: |
| camera | yes | 29.755 -> 29.832 | 2.624e-3 -> 6.730e-4 | 3.250e-4 -> 3.147e-4 | 4.007e-3 -> 2.027e-3 |
| Chelsea | yes | 31.631 -> 31.767 | 7.114e-4 -> 3.675e-4 | 2.310e-4 -> 2.226e-4 | 1.629e-3 -> 1.256e-3 |
| coins | **no** | unchanged | unchanged | unchanged | unchanged |
| astronaut | yes | 27.496 -> 27.612 | essentially unchanged | 5.182e-4 -> 5.027e-4 | 2.349e-3 -> 2.287e-3 |
| Pikachu | yes | 25.772 -> 25.852 | essentially unchanged | 7.043e-4 -> 6.904e-4 | 3.559e-3 -> 3.497e-3 |

Coins is the useful refusal: its fine isotropic texture does not support this
boundary relaxation, and the measured objective keeps the hard result.

## Port boundary

- `port_needed/density_population.py` contains the doubled-angle curvature
  law and local population emission.
- `port_needed/soft_support_diffusion.py` contains conductance construction,
  convex heat steps, and the diagnostic reduction.
- `port_needed/pipeline.py` keeps both mechanisms optional and objective-gated.
- `viewer/segmenting_veroni_app.py` exposes hard/soft site IDs, hard/final
  reconstruction, curvature factor, and conductance independently.

## First native port

Both tight loops are optional C ABI kernels in `include/bfft/vision.h` and
`src/vision.cpp`; an older installed library automatically retains the NumPy
reference.

Measured on the 475x475 Pikachu control:

| kernel | NumPy | native C++ | speedup | maximum discrepancy |
| --- | ---: | ---: | ---: | ---: |
| curvature population | 7.49 ms | 2.97 ms | 2.52x | 0 in all float32 image fields |
| 16-pass, 3-channel soft support | 315.49 ms | 46.46 ms | 6.79x | 5.0e-15 |

The soft kernel gathers all eight incident exchanges, the denominator, and
all channels in one pixel pass. It does not materialize the eight NumPy
scatter temporaries per iteration. Native scalar-reference tests cover
constant-director density and preservation of the partition sum.
