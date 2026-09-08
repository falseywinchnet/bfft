# CONV Version 1.0 and fixed order variants

September 7, 2026. The requested study is implemented: change the fixed source
support and its derivative coefficients, change the downstream Bernstein and
current counts, and measure the resulting transition and cost. There is no
new runtime selection mechanism. None of these variants replaces Version 1.0
by default.

## Main results

M4 Mini, native C, four workers, 31 alternating-order warmed timing repeats.
Cost includes the entire cameraman 512 -> 64 -> 512 pipeline. Width is the
raw proposal's eight-phase coherent 90%-to-10% upper transition, in source
Nyquist units. Lower width is tighter. Positive MSE improvement is better.

| Fixed source samples | Bernstein controls | Current coefficients | Reconstruction | Transition width | Change from V1 | Full runtime / V1 | Cameraman MSE improvement |
|---|---:|---:|---|---:|---:|---:|---:|
| 4 | 4 | 3 | cubic | 0.84688 | 22.78% wider | 0.664x | -1.452% |
| 4 | 6 | 5 | quintic | 0.85930 | 24.58% wider | 1.019x | -0.800% |
| 6 | 4 | 3 | cubic | 0.69775 | 1.16% wider | 0.671x | -0.582% |
| **6 — Version 1.0** | **6** | **5** | **quintic** | **0.68978** | **baseline** | **1.000x** | **baseline** |
| 8 | 6 | 5 | quintic | 0.59255 | 14.10% tighter | 1.008x | +0.229% |
| 8 | 8 | 7 | septic | 0.59399 | 13.89% tighter | 1.346x | +0.397% |

The generated six-source/quintic control has the exact rational Version 1.0
bank. Its measured complete time is 1.000x and its synthesis time is 0.999x,
so the final implementation comparison has a well-matched control.

A separate one-worker gate confirms the broad cost separation:

| Variant | Four-worker full time / V1 | One-worker full time / V1 |
|---|---:|---:|
| 4 source / 3 current | 0.664x | 0.634x |
| 4 source / 5 current | 1.019x | 1.029x |
| 6 source / 3 current | 0.671x | 0.633x |
| 6 source / 5 current control | 1.000x | 0.992x |
| 8 source / 5 current | 1.008x | 0.994x |
| 8 source / 7 current | 1.346x | 1.361x |

The eight-source/quintic cost is effectively unchanged within this timing
scale; do not interpret its one-worker 0.6% apparent advantage as an established
speedup. The eight-source/septic costs about 35%, within the earlier 50% extra
budget on this workload. These are measured costs, not universal guarantees.

## What the tradeoff actually is

**Eight source / five current:** most of the transition tightening comes from
widening the source jet. It retains the existing downstream quintic/current
count and runs at approximately Version 1.0's cost. It improves four of six
natural-image MSEs but worsens text and brick. Its maximum thin-strip excursion
in the fixed orientation/phase battery rises from 0.27009 to 0.31298.

**Eight source / seven current:** increasing downstream degree gives a little
more near-edge gain (0.68351 versus 0.67597 at 0.9 Nyquist), but does not further
narrow the 90%-to-10% width. It gives somewhat better natural-image MSE than
the eight-source/quintic in this battery; text and brick remain slightly worse
than V1. The extra projection/current work explains its higher runtime. Its
maximum thin-strip excursion is 0.31317, essentially the same stress-test
penalty as the eight-source/quintic.

**Six source / three current:** substantially cheaper, with almost the same
raw 90%-to-10% width as V1. That single width number does not make it equivalent:
its gain at 0.9 Nyquist falls from 0.65538 to 0.63372, and all six natural-image
MSEs worsen by 0.36%-0.82%. This is a speed/fidelity setting, not a free removal
of the extra current coordinates.

**Four source / three current:** similarly cheap, with a broader transition
and lower stress-test excursion (0.19238). **Four source / five current** spends
roughly the original cost without recovering the source information lost by
the narrower derivative stencil. Raising downstream degree by itself is not
an effective substitute for source support in this family.

The main conclusion is a concrete family tradeoff: support width largely
controls this transition tightening; downstream current count strongly affects
cost and modifies the fidelity/shape of the response. No global optimality
claim follows from this finite census.

## Natural-image results

Full 8x basin reduction followed by 8x reconstruction, original image sizes,
without clipping before measurement. Entries are percentage MSE improvement
relative to Version 1.0; negative values mean higher error.

| Variant | Camera | Text | Brick | Coins | Grass | Moon |
|---|---:|---:|---:|---:|---:|---:|
| 4/3 | -1.452 | -0.401 | -0.555 | -2.433 | -0.667 | -2.632 |
| 4/5 | -0.800 | +0.273 | +0.525 | -1.830 | -0.239 | -1.910 |
| 6/3 | -0.582 | -0.506 | -0.818 | -0.689 | -0.363 | -0.763 |
| 8/5 | +0.229 | -0.241 | -0.378 | +0.640 | +0.040 | +0.662 |
| 8/7 | +0.397 | -0.038 | -0.075 | +0.830 | +0.161 | +0.866 |

The first number here is source samples and the second is current coefficients.
Cameraman's maximum excursion beyond [0,1] changes from 0.01012 (V1) to
0.01541 (8/5) and 0.01755 (8/7). These small error improvements therefore do
not establish a replacement preserving Version 1.0's ringing behavior.

## Actual nonlinear and 2-D checks

The retained census contains 1,022 one-dimensional admitted-carrier records,
420 two-dimensional carrier records, 210 step/strip records, and 42 natural
image records, each with synthesis-only and full-roundtrip metrics.
The exact raw-bank transition is not presented as a universal frequency
response for the nonlinear admitted operator.

At radial frequency 0.95 times axis Nyquist, averaged across the five tested
orientations and two phases, actual 2-D fundamental gains are:

- Version 1.0: 0.69360, carrier MSE 0.0171021.
- 4/3: 0.62548, carrier MSE 0.0207941.
- 8/5: 0.72492, carrier MSE 0.0147762.
- 8/7: 0.73159, carrier MSE 0.0148690.

Thus the tightening is visible in actual 2-D carrier reconstruction as well
as the linear proposal. Higher fundamental amplitude alone does not imply
lower total error: residual harmonics and phase matter too. Radial frequency
one is not the Cartesian Nyquist boundary at every orientation.
The exact-Nyquist spike in the two-phase 1-D plot reflects source-phase
ambiguity of real carriers at Nyquist; it is not a sharp passband feature.

## Validation and implementation limits

The tests cover exact bank identity for the six/five control, exact interval
mass, cardinality, finite admitted output, affine reproduction, declared raw
polynomial reproduction, and basin integration against analytic integrals.
They also compare the generated control with the original native operator.
Boundary FIR reassociation can change a few floating-point bits; no bitwise
claim is made between different configurations or the generated control and
V1. No nonfinite result occurred in the full census.

Coefficient derivation is cached setup work. The initial timing screen
incorrectly repeated fixed rational derivative construction in the geometry
path; it is retained only as a diagnostic. The final run caches those fixed
coefficients and uses a constant interior FIR path, matching V1's compilation
structure. Final results are in `conv_fixed_orders_final4/`, with the separate
one-worker record `conv_fixed_orders_final1.json`.

The matching source/control counts 4/4,6/6,8/8 denote polynomial degrees 3,5,7.
They do not automatically imply reproduction of every polynomial of that
degree: centered nodal derivative accuracy also limits reproduction. See
README.md for the exact construction, boundary closures, operation counts,
and the distinction between representation degree and reproduction order.

## Artifacts

- `../../output/support_geometry/conv_fixed_orders/conv_fixed_orders_final4/results.json`
- `../../output/support_geometry/conv_fixed_orders/conv_fixed_orders_final4/coefficients.json`
- `../../output/support_geometry/conv_fixed_orders/conv_fixed_orders_final4/comparison.png`
- `../../output/support_geometry/conv_fixed_orders/conv_fixed_orders_final4/cameraman.png`
- `../../output/support_geometry/conv_fixed_orders/conv_fixed_orders_final1.json`
