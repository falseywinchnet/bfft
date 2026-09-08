# Fixed source and reconstruction orders after CONV Version 1.0

This experiment studies fixed configurations of the same CONV construction.
Version 1.0 is the six-source/five-current quintic construction with the exact
rolling ledger-cost recurrence. The baseline calls its original two orders
and Q1 blend; the unfinished terminal-fusion experiment is not enabled here.
There is no runtime selector, repeated interpolation, fitted coefficient
family, or added model catalogue.

## Parameters that change together

For an even source support T=2r+2 and an odd polynomial degree D=2q+1:

- Each source node supplies derivatives through order q, estimated directly
  from a centered 2r+1-point Lagrange polynomial in the interior.
- Adjacent nodal stencils have a union of T source samples per interval.
- Endpoint derivatives determine D+1 Bernstein controls through the explicit
  Hermite identities. Their D differences are the interval currents.
- The original causal sign ledger acts on these D currents per interval.
  A coarse transition retains the same neighborhood of two source intervals:
  2D raw currents and at most 2D-1 possible boundary positions. Version 1.0's
  rolling cost recurrence applies unchanged to this expanded index set.
- The same signed-current mass fibre uses D coordinates. The finite threshold
  formula and mass correction are retained. A failed finite threshold search
  produces a diagnostic nonfinite result; this family does not use the old
  exceptional bisection fallback.
- Synthesis uses degree-D Bernstein tails. Basin reduction uses (D+1)/2-point
  Gauss integration, exact for a degree-D profile before floating rounding,
  with the integration weights compiled once per segment as in Version 1.0.
- The Q1 blend uses the same first derivative estimates as that configuration.

Main sequence: (T,D)=(4,3),(6,5),(8,7). Fixed cross-checks (4,5),(6,3),(8,5)
separate the source-support effect from the reconstruction-degree effect.
A (6,5) generated control is measured alongside the frozen native Version 1.0
so implementation overhead is visible rather than misreported as mathematics.

## Coefficients, without numerical fitting

For source nodes j, construct each Lagrange basis polynomial explicitly as
products of linear factors. The coefficient of t^l in its expansion about
node i, multiplied by l!, gives the derivative weight. With endpoint jets
J_l, the left Bernstein controls are

    b_k = sum_{l=0}^k binom(k,l) (D-l)!/D! J_l,  k=0,...,q.

The right controls use (-1)^l on the corresponding right jets. The raw
currents are a_k=b_{k+1}-b_k. All bank coefficients are generated as exact
rational numbers and emitted as compile-time C constants. There is no linear
system solve or frequency-sample optimization in the construction.

Boundary closures follow Version 1.0's pattern: shrink centered support as
an endpoint approaches, and use a shifted l+2-point closure when the centered
stencil cannot supply derivative order l. For T=6,D=5 this reproduces the
original boundary formulas algebraically. Evaluating those formulas as an
FIR can reassociate floating arithmetic, so the generated control is checked
numerically against the original rather than declared bit-identical.

Polynomial *degree* and polynomial *reproduction order* are different. In the
interior this construction reproduces at least degree min(D,2r). In particular,
the main 4/3, 6/5, 8/7 sequence reproduces degrees 2,4,6 at arbitrary phases;
its symmetric half-phase has additional cancellation. No global degree-five
reproduction claim is inferred merely from a quintic representation.

## Cost accounting

Let N be source-line length, C its lane/channel count, and M output length.
For fixed T,D every configuration remains linear in image size.

| Configuration | Source T | Nodal samples | Jet order | Controls | Currents D | Raw FIR multiply-adds / interval / lane | Terminal multiply-adds / query / lane | Gauss nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4c3 | 4 | 3 | 1 | 4 | 3 | 12 | 3 | 2 |
| 4c5 | 4 | 3 | 2 | 6 | 5 | 20 | 5 | 3 |
| 6c3 | 6 | 5 | 1 | 4 | 3 | 18 | 3 | 2 |
| V1 / 6c5 | 6 | 5 | 2 | 6 | 5 | 30 | 5 | 3 |
| 8c5 | 8 | 7 | 2 | 6 | 5 | 40 | 5 | 3 |
| 8c7 | 8 | 7 | 3 | 8 | 7 | 56 | 7 | 4 |

These are direct bank counts before compiler simplification of zero weights.
Profile storage is D(N-1)C floats. The ledger has O(DN) ordinary cost after
sign propagation; its near-tie fallback costs O(D^2) per transition. The
finite projection costs O(D^2) per interval: the fixed sorting networks use
3,9,21 comparisons for D=3,5,7 respectively, and mixed-sign threshold checking
has at most D+1 regions. Synthesis costs O(DMC). The two-dimensional operator
retains both original factor orders. Actual native time includes allocations,
transposes, geometry, both admissions, basin work, and the final blend.

## Measurements

The upper passband/stopband transition is measured in two complementary ways:

1. Exact raw-bank eight-phase coherent response, from frequency zero to twice
   source Nyquist. Its 90%-to-10% width, midpoint and gain at 0.9 Nyquist are
   reported. This is a linear-proposal response, not a transfer function for
   nonlinear admitted CONV.
2. Actual admitted carriers at two source phases, plus oblique 2-D carriers
   at five orientations. Fundamental amplitude and generated residual energy
   are measured with explicit scalar moment formulas. Radial frequency in 2-D
   is measured against axis Nyquist; the Cartesian Nyquist boundary therefore
   depends on angle and is not automatically the radial frequency 1.

Also report step/strip excursions and original-size 8x round trips of camera,
text, brick, coins, grass, and moon. The source assets are used at native size;
nondivisible dimensions use floor division by eight. Synthesis-only tests
share Version 1.0's coarse input; full round trips use each variant's own
basin reduction. Images and metrics are unclipped before scoring.

The original production implementation remains unchanged by this study.
The source frozen in `v1_native.c` includes an unused terminal-fusion API;
none of the operators in this study calls it.

## Reproduction

Preferred repository commands on the Mini:

```
python3 -m unittest experiments.conv_fixed_orders.test_orders -v
python3 -m experiments.conv_fixed_orders.run_study --out /tmp/conv_fixed_orders_results4 --repeats 31
```

Set CONV_NATIVE_THREADS=4, OPENBLAS_NUM_THREADS=1 and
VECLIB_MAXIMUM_THREADS=1. Use m4build and copy the /tmp results back immediately.
The first run encountered a shared-mirror rsync failure and therefore uses an
isolated authoritative-file copy at /tmp/conv_fixed_orders_workspace on the
same Mini. No remote source is edited and no software is installed.
