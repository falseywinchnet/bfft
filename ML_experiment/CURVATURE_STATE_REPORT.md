# Curvature as state: factor, curvature-context, and nested self-context

## Result

Ordinary self-context remains the best general model in this round. None of the
second-order constructions replaces it across the problem suite.

That negative aggregate result contains two useful positive results:

1. **Factor curvature is a real specialist.** It improves held-out score by
   `+.233` on the high-rank N-D spiral and `+.165` on the Fourier mixture. The
   same construction loses `-.215` on radial stripes and hurts multiscale and
   localized continuation. It recognizes distributed curvature energy, but it
   does not preserve focal geometry.
2. **Curvature self-context changes acquisition more reliably than final
   extrapolation.** It improves learning AUC on 9 of 11 tasks, with 8
   meaningful wins, while reducing mean final held-out score by `.023`. It is a
   better route into many fits, not yet a better destination.
3. **Nested self-context is the most balanced extension.** It is essentially
   neutral on aggregate tail score and has honest wins on chirp (`+.039`),
   localized steps (`+.041`), and Fourier mixture (`+.115`). It still damages
   radial stripes (`-.108`) and multiscale continuation (`-.160`).

The evidence therefore supports a narrower statement than “curvature solves
self-context”: the original context contains useful first-order relational
state; a second context or an even shell can expose additional structure, but
that state must be admitted conditionally. Applying it everywhere is the source
of the brittleness.

## What was tested

All models use LELU and have exactly the same number of trainable parameters
within a task. The control is the ordinary nonlinear
encode-expand-LELU-contract-decode MLP, not an affine-only network. No model
receives a task identity, task-aligned Fourier features, unseen-region labels,
neighboring samples, or target derivatives.

The self-context layer already computes an input-conditioned allocation over a
fixed primitive frame. These experiments reuse that learned chart without
adding parameters:

- **Laplacian shell:** mean symmetric finite difference of the allocation field.
- **Factor curvature:** form the shell-factor Gram operator `KᵀK` and apply it
  to the current pooled response. This retains directional curvature energy
  without depending on shell ordering.
- **Richardson shell:** combine inner and outer shells to suppress the leading
  finite-difference error.
- **Curvature self-context:** lift the even shell response back into activation
  coordinates, add it as a second contextual observation, then recompute the
  allocation.
- **Nested self-context:** present the first normalized contextual displacement
  to the same allocator as the outer system's observation. Its response becomes
  a context of the context, entering at the next power of the context strength.

The last construction is genuinely nested. It is not the earlier experiment
that simply ran the same context refinement twice while remaining anchored to
the original activation.

## Protocol

- M4 Mini CPU; Torch CPU only.
- 11 diagnostic tasks: spiral, checkerboard, low- and high-rank N-D spirals,
  radial stripes, Swiss cheese, ripple, multiscale 1-D, chirp, localized steps,
  and Fourier mixture.
- Width 24, batch 256, 500 steps, two paired seeds.
- 110 confirmation fits: ordinary MLP, self-context, factor curvature,
  curvature self-context, and nested self-context.
- Held-out score measures performance outside the observed support; tail score
  measures the farthest region or minority retention; learning AUC measures
  acquisition across training rather than the final checkpoint alone.
- A preceding one-seed, width-16 screen also tested the Laplacian and Richardson
  shells. Richardson was expensive and brittle; neither survived into the
  paired confirmation.

## Confirmation summary

| Model | Held-out | Δ vs self | Tail | Δ vs self | Learning AUC | Δ vs self | sec / fit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary MLP | .440 | -.150 | .430 | -.119 | .665 | -.193 | .27 |
| Self-context | **.590** | — | **.550** | — | .858 | — | 3.95 |
| Factor curvature | .580 | -.010 | .533 | -.016 | .849 | -.009 | 10.40 |
| Curvature self-context | .568 | -.023 | .535 | -.014 | **.872** | **+.014** | 12.63 |
| Nested self-context | .583 | -.007 | **.550** | +.000 | .857 | -.001 | 6.41 |

## Held-out score by problem

| Problem | MLP | Self-context | Factor | Curvature context | Nested |
|---|---:|---:|---:|---:|---:|
| Spiral | .359 | .338 | .346 | .336 | .349 |
| Checkerboard | .513 | .442 | .457 | .462 | .451 |
| N-D spiral, low rank | .402 | .390 | .398 | .375 | .389 |
| N-D spiral, high rank | .472 | .593 | **.826** | .743 | .584 |
| Radial stripes | .525 | **.764** | .549 | .758 | .656 |
| Swiss cheese | .947 | .982 | .976 | **.983** | .980 |
| Ripple | .508 | **.940** | .849 | .916 | .923 |
| Multiscale 1-D | .126 | **.460** | .334 | .300 | .299 |
| Chirp | .025 | .327 | .300 | .212 | **.366** |
| Localized steps | .841 | .942 | .868 | .890 | **.983** |
| Fourier mixture | .123 | .317 | **.482** | .268 | .433 |

## Interpretation

### Factor curvature is not general curvature understanding

The `KᵀK` state is invariant to the ordering and rotation of the sampled shell
factors. That is attractive when evidence is distributed across many harmonic
planes, exactly the condition in the high-rank spiral and Fourier mixture. It
also turns out to be destructive when the task demands the retention of one
focal center or a compact local event. The radial result is the cleanest
counterexample: the factor operation detects curvature energy but partially
forgets *where that energy is centered*.

### Curvature self-context is an acquisition signal

Promoting the symmetric shell response into activation state makes optimization
find useful in-field representations earlier. Its learning-AUC gain is much
more consistent than its final extrapolation. Normalizing and injecting that
state at full contextual strength appears to overstate weak or ambiguous
curvature after the observed region ends. The state is informative, but its
authority is not calibrated.

### Self-context of self-context is plausible, but cannot be unconditional

Nested self-context is the cleanest answer to “what lives one level outside
self-context?” The first context is a learned relational displacement; the
outer pass asks the same chart what structure that displacement itself has. Its
wins on chirp, localized steps, and Fourier mixtures suggest that this can
capture a local law about how a learned relation changes.

The radial and multiscale failures show why the nesting cannot simply be always
on. In some tasks the first context is already the sufficient statistic. A
second interpretation compounds an error or replaces focal detail with a
broad-scale continuation.

## Next principled experiment

The next step should not be another menu of hand-set gates. It should test one
specific hypothesis: **the baseline allocation can measure whether its outer
state is self-consistent before allowing that state to act.**

A useful criterion should be endogenous and continuous. For example, compare
the direction selected at the authentic activation with the direction selected
after the first contextual displacement. Their projective agreement can assign
the second-order state *confidence*, not task identity. High agreement would
allow nested or factor state; disagreement would leave ordinary self-context
nearly unchanged. This directly targets the observed split between distributed
harmonic structure and focal geometry without supplying prior knowledge of
which task is present.

## Files

- `models.py`: parameter-free shell and nested-state implementations.
- `variants.py`: the seven matched curvature-state variants.
- `results_curvature_screen/`: width-16 screen.
- `results_curvature_confirm/`: two-seed confirmation and fitted probes.
- `curvature_state.html`: accessible interactive comparison.
- `build_curvature_state.py`: reproducible visualization builder.
