# Exact-budget ordinary MLP versus soft Eikonal

## Corrected result

The corrected baseline is a separately trained, ordinary nonlinear MLP:

```text
input → encode → dense expansion → LELU → dense contraction → decode → output
```

The latent width equals the soft model's width. The expansion is enlarged until
the trainable-parameter budget is exhausted: 66 units at width 16 and 104 at
width 36. A tiny active affine residual consumes the indivisible remainder—at
most 18 scalars (0.80%) at width 16 and 4 scalars (0.052%) at width 36. Thus
every pair has exactly equal trainable parameter counts without giving the MLP
dead padding.

This comparison substantially changes the conclusion. At width 36, soft
Eikonal beats the ordinary MLP by more than 0.01 on 10 of 19 tasks, not 17 of
19. Its mean held-out fit advantage is 0.069, not 0.320. The large surviving
wins are ripple (+0.453), radial stripes (+0.286), the user multi-scale target
(+0.166), checkerboard (+0.117), Fourier mix (+0.096), and chirp (+0.083).

The ordinary MLP erases the apparent Eikonal advantage on two moons, pinwheel,
XOR, sinusoidal bounds, the ring SDF, Lorenz lobes, and almost all of periodic
wells. That is exactly why the nonlinear baseline was necessary.

All values average three seeds and 800 optimization steps on the M4 Mini CPU.
“Fit” is held-out validation inside the training support. “Test” is the explicit
outer test for spiral, checkerboard, complex spiral, and the four 1-D tasks; for
the other tasks it is an independent in-support test.

| task | params each | MLP fit | soft fit | Δ fit | MLP test | soft test |
|---|---:|---:|---:|---:|---:|---:|
| spiral | 7,814 | 0.983 | 1.000 | +0.017 | 0.337 | 0.354 |
| checkerboard | 7,814 | 0.840 | 0.957 | +0.117 | 0.507 | 0.445 |
| two_moons | 7,814 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| pinwheel | 7,925 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| xor_quads | 7,814 | 0.998 | 0.997 | -0.001 | 0.998 | 0.997 |
| sinusoid_bounds | 7,814 | 0.987 | 0.989 | +0.001 | 0.987 | 0.989 |
| radial_stripes | 7,814 | 0.519 | 0.805 | +0.286 | 0.519 | 0.805 |
| swiss_cheese | 7,814 | 0.966 | 0.987 | +0.021 | 0.966 | 0.987 |
| lorenz_lobes | 7,814 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| periodic_wells | 7,777 | 0.990 | 0.995 | +0.005 | 0.990 | 0.995 |
| ripple | 7,777 | 0.507 | 0.960 | +0.453 | 0.507 | 0.960 |
| ring_sdf | 7,777 | 0.999 | 1.000 | +0.001 | 0.999 | 1.000 |
| complex_spiral_3d | 7,815 | 0.946 | 0.985 | +0.038 | 0.002 | 0.020 |
| periodic_nd | 7,993 | 0.581 | 0.589 | +0.008 | 0.581 | 0.589 |
| hyperchecker | 8,102 | 0.523 | 0.518 | -0.005 | 0.523 | 0.518 |
| multiscale_1d | 7,741 | 0.814 | 0.980 | +0.166 | 0.140 | 0.278 |
| chirp_1d | 7,741 | 0.901 | 0.984 | +0.083 | 0.024 | 0.307 |
| localized_steps_1d | 7,741 | 0.968 | 1.000 | +0.032 | 0.901 | 0.974 |
| fourier_mix_1d | 7,741 | 0.894 | 0.991 | +0.096 | 0.171 | 0.412 |

## What survives the fair baseline

The strongest evidence is not simply that soft Eikonal wins. It is *where* it
wins. A 104-unit dense LELU expansion still remains near chance on ripple and
radial stripes, while the soft pool reaches 0.960 and 0.805. These are precisely
the problems requiring multiple locally oriented or oscillatory views. The
ordinary MLP can represent them in principle, but the parameter-matched
optimization does not discover them in 800 steps.

The learned allocation is causally involved. At width 36, assigning each
observation another observation's allocation drops soft validation score by
approximately 0.78 on ripple, 0.45 on checkerboard, 0.44 on Swiss cheese, 0.55
on the multi-scale signal, and 0.50 on Fourier mix. Hyperchecker has essentially
zero mismatch cost and remains at chance. This is the cleanest distinction
between useful conditioned transport and nominal architectural complexity.

Increasing baseline capacity narrows the average soft advantage: +0.091 at
width 16 versus +0.069 at width 36. Soft wins 13 tasks at the lower width and
10 at the higher width. That suggests a real parameter-efficiency advantage,
but only on the subset whose geometry matches what the conditioned pool can
discover.

## Learning speed and continuation

Soft Eikonal has higher learning-curve area on 16 of 19 width-36 tasks, but is
6.85 times slower in CPU wall time. Therefore it is faster in optimization
steps, not faster computationally.

Relative continuation looks better than it did against the affine control:
soft wins 6 of 7 outer-support comparisons. Absolute continuation remains poor
on five of those six:

- spiral: 0.354 versus 0.337, with zero tail-class survival for both;
- complex spiral: 0.020 versus 0.002;
- multi-scale 1-D: 0.278 versus 0.140;
- chirp: 0.307 versus 0.024;
- Fourier mix: 0.412 versus 0.171;
- localized steps: 0.974 versus 0.901.

Only localized steps constitutes strong continuation. Checkerboard is the
counterexample: soft fits the inner region better (0.957 versus 0.840) but
generalizes worse outside it (0.445 versus 0.507). This says the allocation can
learn a powerful local atlas while transporting the wrong global continuation.

## Interpretation

The corrected result is encouraging but narrower. Soft Eikonal is not merely a
complicated substitute for a nonlinear MLP: its advantages survive exact
budget matching on ripple, radial structure, checkerboard, and multi-scale 1-D
signals. Yet it is also not omni-inducement. The learned sieve selects useful
local structure without identifying which derivative continuation is valid
outside the observed support.

The next controlled experiment should add a transport-consistency term along
short secants and compare it against an ordinary MLP given the same additional
training objective and scalar budget. The success condition is improvement on
checkerboard and oscillatory tails without installing explicit periodicity.

Raw records are in `results_mlp_full/results.json`; paired aggregates are in
`results_mlp_full/task_comparison.csv`; dense fixed-probe fields and 1-D curves
are in `results_mlp_full/visual_probes.json`.
