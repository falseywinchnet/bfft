# Eikonal ray transport study

## Question

Ordinary self-context can reach strong solutions but often acquires them more
slowly than curvature self-context. Which part of curvature supplies that
advantage, and can self-context expose the same differential information
without inheriting the shell's extrapolation failures?

## Formal distinction

For fixed primitive maps `U_d`, ordinary self-context lifts the allocation as

```text
C(x) = mean_r sum_d w_d(x) U_d^T U_d x
v0   = rms_normalize(C(x), x)
z1   = x + alpha v0
```

The second allocation sees `z1`, but the original allocation and its change are
not retained as separate state. The transport variants make that change
observable without adding trainable parameters.

The best construction is the centered odd response along the allocator's
single integrated eikonal transport ray:

```text
v+ = normalize(C(x + alpha v0))
v- = normalize(C(x - alpha v0))
odd = bound((v+ - v-) / 2, scale=x)
z2 = x + alpha * (v0 + odd / 2)
```

This is a signed first directional derivative along the relation already
assembled by the allocator. It uses four chart observations per layer: the
authentic point, two centered probes, and the corrected endpoint.

## Controlled variants

All variants have identical trainable parameter counts.

- Ordinary self-context: two chart observations.
- Picard iteration: replace the first context by a second anchored context.
- Heun transport: average the first context and its forward update.
- Integrated-ray odd/even: centered derivative along the allocated context.
- Eikonal basis-ray odd: resolve the odd derivative across all rank basis rays
  and contract it covariantly along the transport direction.
- Full curvature: eleven observations and full-RMS even-shell injection.
- Bounded curvature: preserve raw curvature magnitude.
- Detached curvature: identical forward values to full curvature, but no shell
  gradient path.

The benchmark records trainable parameters, chart observations, wall time,
gradient-group norms, allocation transition, context agreement, curvature
authority, learning AUC, held-out score, tail score, and Jacobian variability.

## Four-task mechanism screen

Two paired seeds, width 24, 500 steps, M4 CPU:

| Variant | Chart points | Mean held-out | Mean learning AUC | sec / fit |
|---|---:|---:|---:|---:|
| Self-context | 2 | .5335 | .8237 | 4.7 |
| Picard ×2 | 3 | .5129 | .8219 | 6.4 |
| Heun | 3 | .5157 | .8265 | 6.5 |
| Integrated ray, odd | 4 | **.5570** | .8271 | 8.5 |
| Integrated ray, even | 4 | .5149 | .8261 | 8.6 |
| Basis rays, odd | 11 | .5112 | .8234 | 16.5 |
| Full curvature | 11 | .5172 | **.8423** | 14.3 |
| Bounded curvature | 11 | .5366 | .8289 | 14.3 |
| Detached curvature | 11 | .4939 | .7848 | 9.0 |

The integrated odd ray is the only construction that improves both mean
endpoint and mean acquisition while using substantially fewer observations
than the shell.

The forward/backward curvature ablation is task-dependent. Detaching the shell
collapses radial and multiscale learning, proving that the shell's gradient
route is essential to its focal acquisition. Yet detached curvature is the
best high-rank N-D extrapolator. There, the forward shell state is useful while
its backward route appears to pull optimization toward a worse continuation.

## Eleven-task promotion

Integrated-ray odd versus ordinary self-context, two paired seeds:

| Task | Self-context | Integrated odd | Delta | Tail delta | AUC delta |
|---|---:|---:|---:|---:|---:|
| Checkerboard | .4425 | .4453 | +.0028 | +.0337 | -.0048 |
| Chirp | .3268 | .2676 | -.0592 | -.0480 | +.0036 |
| Fourier mixture | .3174 | .3876 | +.0702 | +.0340 | -.0103 |
| Localized steps | .9420 | .9955 | +.0535 | +.0963 | +.0013 |
| Multiscale | .4597 | .4490 | -.0107 | -.0112 | +.0009 |
| N-D spiral, high rank | .5930 | .6048 | +.0118 | +.0017 | +.0008 |
| N-D spiral, low rank | .3895 | .3955 | +.0060 | +.0182 | +.0015 |
| Radial stripes | .7638 | .7868 | +.0230 | — | +.0222 |
| Ripple | .9403 | .9367 | -.0036 | — | -.0230 |
| Spiral | .3383 | .3393 | +.0010 | -.0067 | -.0057 |
| Swiss cheese | .9819 | .9808 | -.0011 | — | +.0003 |

Across the suite, integrated-ray odd wins 7 of 11 tasks, raises mean held-out
score from `.5905` to `.5990`, and raises mean available tail score from
`.4198` to `.4345`. Mean learning AUC changes from `.8577` to `.8565`, so it
does not reproduce curvature's broad speed advantage. It does reproduce the
radial speed advantage without the radial endpoint loss.

## What the basis-ray clue established

The phrase “eikonal basis rays” suggests sampling every learned rank direction.
That exact construction was tested. It is not the answer: it is unstable on
radial, weak on multiscale, neutral on high-rank spiral, and a Fourier
specialist. Eleven local observations alone therefore do not explain the gain.

The important state is the integrated ray after the allocation has already
combined its basis views. Decomposing that relation back into rank rays loses
the relational integration. The winning odd probe differentiates the assembled
transport rather than the ingredients from which it was assembled.

## Conclusion

Self-context was hiding its own derivative. Making the centered odd transport
explicit is a small genuine improvement: better aggregate endpoint and tail
behavior, fewer observations than curvature, and no new parameters. It is not
yet a universal acquisition accelerator. The remaining problem is to obtain
curvature's helpful gradient conditioning without allowing that gradient route
to overwrite the integrated transport state.

The graphical results and seed-0 fitted fields are in `transport_study.html`.
