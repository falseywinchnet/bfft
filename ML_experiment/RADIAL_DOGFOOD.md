# Radial self-context dogfood

## Fixed experiment

Every run uses only `radial_stripes`, seed 0, width 24, 500 AdamW steps,
batch 256, learning rate `3e-3`, and the M4 CPU. The target is never exposed to
the layer as a radius, neighborhood relation, periodic feature, auxiliary loss,
or topology score. Trainable parameter count remains unchanged across all
mechanism variants.

The topology-first success condition is a correct central disk followed by the
six radial class transitions visible over radius `[0, 1.5]`. Validation score
alone is not considered sufficient.

## Result

The winning construction is `self_context_stiefel_flow_curvature`:

| Mechanism | Validation | Radial crossings | Profile accuracy | Angular agreement | Center class-0 probability |
|---|---:|---:|---:|---:|---:|
| Self-context | .7465 | 3 / 6 | .6390 | .9209 | .4400 |
| Raw curvature state | .7156 | 6 / 6 | .9129 | .7686 | .6797 |
| Tight frame + orthogonal curvature | .7930 | 6 / 6 | .8838 | .8267 | .6097 |
| **Continuous frame flow + orthogonal curvature** | **.8884** | **6 / 6** | **.9544** | **.9184** | **.9637** |

The fitted decision field visibly contains the central disk and four coherent
annular bands. Residual errors are small seams near the outer field rather than
a missing central regime or a favorable radial average over torn predictions.

## Mechanism

Random views supplied coverage but no correspondence between their rank rays.
Consequently, a continuous change in allocation coefficients did not imply a
continuous lifted frame. Curvature recovered the correct transition radii but
turned those gauge inconsistencies into positional tears.

The successful atlas transports one common orthonormal rank-4 frame through the
whole hidden space:

```text
K = Q blockdiag(k_j J) Q^T,       K^T = -K
U(theta) = exp(theta K) U(0),     theta in [0, 2 pi)
```

The integer rotation frequencies make the path close exactly, while the random
orthogonal mixing spreads it across every hidden dimension. Twelve stored views
sample this continuous Stiefel flow. The symmetric curvature shell is then
orthonormalized inside its learned subspace before its even second difference is
integrated into self-context.

This does not encode radial coordinates or periodic labels. It supplies the
missing geometric fact that adjacent interpretations are frames in one bundle,
not unrelated arrays whose ray indices happen to match.

## Rejections that located the result

- Separating chart selection from transported values removes useful coupling.
- Odd and even integrated rays do not recover the center reliably.
- Full curvature finds all transition radii but tears annuli angularly.
- Magnitude bounding removes both the tear and the useful inner transitions.
- Arithmetic or log-geodesic shell allocation reaches at most four crossings.
- Projecting curvature onto self-context or the chart point loses transitions.
- Whitening random shell rays improves the radial profile but not gauge seams.
- Direct Laplacian, Richardson, factor-energy, and heat-mean output updates do
  not cohere the field.
- A simple continuous frame cycle is coherent but spans too little hidden
  space; a full-space skew flow is required.
- Smoothing allocation probabilities around the cycle blurs distinct chart
  states. Sampling the same flow at 24 views also optimizes worse at 500 steps.

The graphical report is `radial_dogfood.html`; raw per-run fields and histories
are under `results_radial_dogfood/`.
