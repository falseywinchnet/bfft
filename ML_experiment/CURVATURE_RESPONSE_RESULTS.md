# Continuous curvature response

## Question

The sparse-sine experiments showed that Hermite acquisition and an operator
sphere can cooperate even though neither is a generally better model.  This
study asks which competence transfers back into self-context on radial stripes,
the two-class spiral, and the complex 3-D spiral.

The experiment never supplies polar coordinates, frequency, curve parameter,
analytic truth, or test points.  Its supervised geometric atlas is made only
from observed pairs.

## Separating the mechanisms

Hermite and operator behavior were initially conflated:

1. Hermite acquisition replaces empirical point density with local geometric
   support and transports a target jet through observed gaps.
2. The operator sphere gives the hidden response an odd component plus sampled
   tangent and even-curvature components.
3. Optimizer-transported zonotope descent offers empirical, structure-weighted,
   and Hermite gradients as candidate AdamW transitions and lets held-out
   observations choose a continuous mixture.

The pivotal screen confirmed that these are distinct.  On radial stripes at
300 accepted updates, self-context scored `0.528`, self-context plus Hermite
`0.581`, operator alone `0.535`, and operator plus Hermite `0.648`.  The
operator's normalized curvature-coordinate energy rose from `0.008` to `0.073`
under Hermite acquisition.  The interaction therefore teaches the operator to
use its even response; it is not just extra capacity.

## Failed backport and correction

The first lightweight backport selected the principal eigenvector of the local
frame and sampled one antithetic pair.  It was brittle and produced a NaN
trajectory on complex spiral.  The reason is structural: eigenvectors have a
sign ambiguity and switch at eigenvalue crossings.  That selector inserted
faces into a mechanism intended to vary continuously.

The corrected layer uses a fixed dense mixture of every selected frame ray.
Because the frame itself changes continuously with allocation weights, the
probe direction also changes continuously.  It measures

```text
curvature = allocate(x + delta) + allocate(x - delta) - 2 allocate(x)
```

matches its RMS to the current chart response, and adds it through one learned
positive scalar before the ordinary shared LELU/down/output path.  It adds one
parameter to self-context and no alternate decoder.

## Pivotal confirmation

Three seeds, width 38, 500 accepted updates:

| Task / method | Mean score | Mean tail | Mean seconds |
|---|---:|---:|---:|
| Radial · self-context | 0.629 | — | 4.4 |
| Radial · full operator + Hermite | 0.860 | — | 23.0 |
| Radial · continuous curvature + Hermite | **0.874** | — | 14.6 |
| Radial · curvature, equal-batch Hermite mix | 0.763 | — | 7.9 |
| Radial · self-context, 1,000-step sample control | **0.879** | — | 8.7 |
| Spiral · self-context | 0.352 | 0.303 | 5.4 |
| Spiral · continuous curvature, empirical | **0.380** | 0.311 | 7.4 |
| Spiral · self-context zonotope, warm | 0.362 | **0.358** | 21.8 |
| Complex 3-D · self-context | 0.035 | 0.061 | 4.4 |
| Complex 3-D · continuous curvature, empirical | **0.054** | **0.081** | 7.6 |
| Complex 3-D · curvature, equal-batch Hermite mix | 0.052 | **0.086** | 7.6 |
| Complex 3-D · self-context zonotope, warm | 0.040 | 0.062 | 27.6 |

The radial operator/Hermite result learns much more per optimizer update, but
ordinary self-context still wins the strict sample-and-wall-time control.  The
continuous backport is therefore an acquisition-speed and representational win,
not a universal compute Pareto win.

The zonotope method helps stability on spiral and complex spiral, but three
candidate gradients and functional AdamW previews make it substantially more
expensive than simply training self-context longer.  It remains a diagnostic
optimizer, not the default optimizer.

## Complete 24-task battery

The empirical continuous-curvature layer was run beside self-context with the
same initialization, width, batches, and 500 accepted updates.  It adds exactly
one parameter and costs `1.74x` wall time.

- Mean held-out score: `+0.00853`; wins 13 of 24 tasks.
- Mean tail score: `+0.00959`; wins 12 of 24 tasks.
- Mean learning AUC: `+0.01600`; wins 19 of 24 tasks.
- Large wins: radial stripes (`+0.347`), multiscale 1-D (`+0.102`), ripple
  (`+0.037`), checkerboard (`+0.025`), and high-rank spiral tail (`+0.070`).
- Clear losses: sparse sine, localized steps, Fourier mix, polynomial chirp,
  ordinary chirp, and low-rank N-D spiral.

The layer is consequently integrated as
`self_context_curvature_response`, a switchable specialist built around the
relational self-context parent.  It does not replace `self_context`.

## Interpretation

Hermite acquisition succeeds when missing geometric measure prevents a useful
second difference from cohering.  The operator succeeds when it can retain that
even response as state.  The transferable competence is not an operator sphere
or Hermite interpolation by itself: it is a continuous, orientation-free even
response returned to the same self-context decoder.

Radial stripes are especially favorable because repeated level-set curvature
is the target structure.  Complex spiral benefits because the 3-D trajectory's
local bending becomes representable without choosing a global periodic
generator.  Piecewise and high-frequency 1-D targets often lose because the
same even-response bias smooths or redirects focal transitions.
