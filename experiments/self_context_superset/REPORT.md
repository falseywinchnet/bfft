# Self-context Eikonal superset report

## Verdict

Self-context remains the baseline. None of the five additions dominates it on
fit, continuation, learning speed, and compute simultaneously. Two additions
are nevertheless real:

1. **Hard allocation is the acquisition-speed winner.** It improves
   learning-curve area on 21 of 22 problems, with 11 improvements larger than
   .005 and no losses that large. It costs no parameters and essentially no
   additional CPU time.
2. **Allocation-chart curvature is a continuation specialist.** It has the
   highest mean held-out score (.707) and tail score (.688), including a large
   high-rank N-D spiral gain, but is 2.76× slower and damages radial stripes and
   Fourier continuation. It is evidence for a useful derivative bias, not a
   new universal default.

The exact-budget ordinary LELU MLP remains essential. Averaged across all 22
problems, self-context improves held-out score from .610 to .704 and learning
AUC from .744 to .857, while the MLP is about 11.4× faster per training run.

## Protocol

- **22 problems:** the previous 19-task catalog plus rank-2 and rank-16 spirals
  embedded in 16 dimensions and a rotated 16-D hypercube checker.
- **Seven exact-budget models:** ordinary LELU MLP, self-context, and five
  parameter-identical self-context modifications.
- Confirmation: width 36, 600 AdamW steps, batch 256, two paired seeds, M4 Mini
  CPU only: **308 fits**.
- Preliminary width-16 screen: 154 fits. Total benchmark: **462 fits**.
- Parameter count varies only with task dimensions: 7,741–8,318 parameters per
  model; every comparison within a task is exact.
- Classification score is mean class recall. Regression score is
  `1 / (1 + normalized MSE)`. Tail score measures explicitly withheld support
  when present. No GELU and no task-specific basis were used.

## Overall comparison

| Model | Validation | Held-out | Tail | Learning AUC | CPU s/run | Meaningful AUC wins/losses vs self-context |
|---|---:|---:|---:|---:|---:|---:|
| ordinary LELU MLP | 0.826 | 0.610 | 0.600 | 0.744 | 0.40 | 1 / 15 |
| self-context baseline | 0.922 | 0.704 | 0.680 | 0.857 | 4.55 | — |
| hard gate | 0.922 | 0.702 | 0.681 | 0.866 | 4.55 | 11 / 0 |
| iterated context | 0.921 | 0.698 | 0.682 | 0.858 | 6.48 | 3 / 0 |
| uncertainty gate | 0.918 | 0.693 | 0.675 | 0.856 | 4.64 | 1 / 4 |
| output secant | 0.916 | 0.698 | 0.676 | 0.858 | 4.62 | 5 / 2 |
| chart curvature | 0.911 | 0.707 | 0.688 | 0.853 | 12.55 | 1 / 4 |

## All 22 problems

“Best addition” excludes the MLP and unchanged self-context. Tail is that
addition's score on explicitly withheld support, or its ordinary held-out score
when the problem has no separate tail.

| Problem | MLP | Self-context | Best addition | Addition score | Addition tail |
|---|---:|---:|---|---:|---:|
| spiral | 0.326 | 0.336 | hard gate | 0.368 | 0.302 |
| checkerboard | 0.505 | 0.439 | output secant | 0.476 | 0.303 |
| two_moons | 0.999 | 1.000 | hard gate | 1.000 | 1.000 |
| pinwheel | 1.000 | 1.000 | hard gate | 1.000 | 1.000 |
| nd_spiral_low_rank | 0.383 | 0.407 | chart curvature | 0.409 | 0.379 |
| nd_spiral_high_rank | 0.467 | 0.860 | chart curvature | 0.977 | 0.957 |
| hypercube_checker | 0.502 | 0.504 | chart curvature | 0.506 | 0.473 |
| xor_quads | 0.996 | 0.996 | iterated context | 0.997 | 0.997 |
| sinusoid_bounds | 0.984 | 0.985 | uncertainty gate | 0.986 | 0.986 |
| radial_stripes | 0.518 | 0.837 | hard gate | 0.845 | 0.845 |
| swiss_cheese | 0.957 | 0.982 | uncertainty gate | 0.984 | 0.984 |
| lorenz_lobes | 1.000 | 1.000 | iterated context | 1.000 | 1.000 |
| periodic_wells | 0.951 | 0.990 | hard gate | 0.993 | 0.993 |
| ripple | 0.508 | 0.949 | hard gate | 0.959 | 0.959 |
| ring_sdf | 0.999 | 1.000 | uncertainty gate | 1.000 | 1.000 |
| complex_spiral_3d | 0.001 | 0.019 | chart curvature | 0.021 | 0.036 |
| periodic_nd | 0.568 | 0.578 | output secant | 0.581 | 0.581 |
| hyperchecker | 0.525 | 0.524 | uncertainty gate | 0.525 | 0.525 |
| multiscale_1d | 0.129 | 0.283 | chart curvature | 0.389 | 0.382 |
| chirp_1d | 0.022 | 0.317 | output secant | 0.449 | 0.285 |
| localized_steps_1d | 0.880 | 0.963 | output secant | 0.998 | 0.995 |
| fourier_mix_1d | 0.201 | 0.515 | chart curvature | 0.400 | 0.289 |

## What each experiment says

### Hard allocation

Temperature .55 makes the already conditioned chart commit sooner. It raises
mean learning AUC by .0084 and wins AUC on 21/22 problems without extra compute.
It is the only addition supported as a routine training default. Its warning is
continuation: it improves multiscale 1-D tail by .079 but reduces Fourier-mix
tail by .107 and spiral tail by .042.

### Second context refinement

A second anchored fixed-point refinement is not generally better. Mean fit and
learning speed are almost unchanged while runtime rises 43%. It helps chirp
tail by .099 and localized steps by .033, but hurts radial stripes, high-rank
spiral, and Fourier mix. One private guess is usually enough; repeated internal
reinterpretation can overcommit to the observed chart.

### Uncertainty-gated context

Scaling the contextual guess by allocation entropy does not improve the
baseline. It has small localized wins—especially localized steps—but loses on
radial stripes and multiscale continuation. The successful contextual signal is
not simply “more help when uncertain.” Low entropy can itself be informative.

### Output secants

Explicitly supervising `f(x_i)-f(x_j)` specializes the model toward derivative
continuation. It strongly helps chirp and localized steps, but damages radial
stripes and Fourier mix. Random global pairings impose the wrong relational
neighborhood on some geometries; the next secant experiment should select pairs
in the learned chart rather than input-space batches.

### Allocation-chart curvature

Penalizing the second finite difference of allocation weights is the most
interesting non-default result. Relative to self-context, tail score changes by:

- checkerboard **+.086** (still not true global continuation);
- high-rank 16-D spiral **+.156**;
- multiscale 1-D **+.051**;
- chirp **+.046**;
- localized steps **+.053**;
- radial stripes **−.186**;
- Fourier mix **−.063**.

This is a real trade: flatter transport charts preserve some outward flows but
erase structures whose correct local frame must turn sharply.

## Rank experiment: the most revealing result

The low-rank 16-D spiral remains poor: self-context .407 and the best addition
.409 outside the observed half. The high-rank spiral reaches .860 with
self-context and **.977 with chart curvature**.

That is not evidence that high rank is intrinsically easier. The high-rank
construction presents eight harmonic planes in every observation, so the input
already contains redundant phase derivatives. The low-rank embedding contains
only one rotated spiral plane. The contrast supports the user's warning about
models that “know before they know”: chart regularity can exploit supplied
relational evidence, but it cannot conjure absent continuation evidence.

## The hard boundaries remain

- Checkerboard: ordinary MLP .505, self-context .439, best addition .476.
- Ordinary spiral: best addition .369 and no strict tail-survival bin.
- Low-rank N-D spiral: best addition .409.
- Rotated hypercube checker and 10-D hyperchecker remain essentially chance.
- Complex 3-D spiral continuation remains near zero despite excellent inner
  validation.

So this is not omni-inducement. The promising result is narrower: self-context
is a highly efficient acquisition prior; hard gating accelerates it; and chart
curvature can improve derivative transport when the observation actually
contains a coherent transport direction.

## Artifacts

- `results_confirm/results.json`: all 308 confirmation runs and learning curves.
- `results_confirm/summary.json`: paired-seed aggregates and deltas.
- `results_screen/results.json`: 154-run preliminary screen.
- `results_confirm/probes.json`: fitted fields, tails, and N-D projections.
- `run_benchmark.py`, `run_probes.py`, and `test_superset.py`: reproducible M4
  CPU entry points.
