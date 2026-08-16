# Continuous frame flow: full matched benchmark

## Experiment

The radial dogfood winner is promoted without modification and compared with
the two controls on the complete 22-problem suite:

1. an ordinary encode-expand-LELU-contract-decode MLP;
2. self-context;
3. self-context with the continuous full-space Stiefel frame flow and the
   orthogonal curvature shell.

Every model has 4,069 trainable parameters at width 24. Every run uses 500
AdamW steps, batch 256, learning rate `3e-3`, two paired seeds, and CPU Torch
on the M4 Mini. The benchmark contains 132 fits. A separate seed-0 visual pass
retrained all 66 task/model pairs with the same budget.

## Aggregate result

| Model | Validation | Held-out score | Tail score | Learning AUC | Seconds / fit |
|---|---:|---:|---:|---:|---:|
| Vanilla MLP | .7851 | .5947 | .5772 | .7226 | .30 |
| Self-context | .9152 | .6865 | .6656 | .8437 | 4.64 |
| **Continuous frame flow** | **.9177** | **.7007** | **.6709** | **.8533** | 15.39 |

Relative to self-context, continuous frame flow gains `.01423` held-out score,
`.00525` tail score, and `.00965` learning AUC. It wins acquisition on 15 of 22
tasks and held-out endpoint on 11 of 22, but costs 3.31 times as much wall time.
The endpoint gain is concentrated rather than universal.

## Where the gain lives

| Task | Held-out delta | Tail delta | Learning-AUC delta | Interpretation |
|---|---:|---:|---:|---|
| N-D spiral, high rank | **+.2260** | **+.2052** | +.0027 | decisive structure recovery |
| Fourier mix 1-D | +.1618 | +.1356 | +.0047 | large but seed-unstable |
| Chirp 1-D | +.0756 | +.0507 | +.0194 | reliable changing-frequency gain |
| Radial stripes | +.0259 | +.0259 | **+.0546** | topology exists, optimizer is unstable |
| Checkerboard | +.0240 | +.0680 | +.0276 | better tails and acquisition |
| Localized steps 1-D | **-.1816** | **-.1890** | +.0048 | faster in-field fit, worse continuation |
| Periodic N-D | -.0106 | -.0106 | -.0053 | self-context remains better |
| Ripple | -.0078 | -.0078 | -.0063 | self-context remains better |

The high-rank N-D result is the cleanest success. The target is distributed
over eight hidden harmonic planes and the transported atlas is not aligned to
those planes. The gain therefore cannot be explained by a radial feature or a
task-specific Fourier coordinate supplied to the model.

The radial mean hides the most important failure. Seed 0 repeats the dogfood
win (`.8884` versus `.7465`), while seed 1 reverses it (`.6910` versus `.7811`).
Continuous flow has demonstrated the representational route to the central
disk and four rings, but not reliable acquisition of that route.

Likewise, almost perfect observed-region scores on localized steps do not
protect the continuation: flow reaches `.9999` mean validation but loses badly
outside the observed interval. The atlas is not a substitute for the correct
transport law.

## What the vision transport pipeline teaches us

The image segmenter and cartoon/texture decomposer impose four constraints that
the current neural layer only partially satisfies.

### 1. Measure geometry, then transport it

The segmenter measures a spatial metric from the unchanged source, carries
owners and predecessor structure through that metric, and only then fits the
local fields. It does not independently rediscover a coordinate frame at every
destination. The continuous Stiefel flow fixes exactly one part of this problem:
adjacent atlas views are now frames in one bundle rather than unrelated arrays.

The curvature shell still calls the allocator independently at every displaced
sample. Thus its metric is not frozen along the shell path. The closest direct
translation from the vision system is a **source-conditioned shell**: measure
the allocation metric once at the center, transport that metric to the shell
locations, and allow only the projected evidence and costs to change there.
This is both more causal and potentially much cheaper.

### 2. Diffuse only where conductance permits

The segmentation pipeline never treats smoothing as intrinsically good.
Cartoon boundaries are barriers, texture routes are parent-restricted, and
soft weights preserve a partition of unity. This explains why unconditional
view smoothing failed in radial dogfood: it blurred genuinely distinct chart
states. A principled atlas diffusion must be gated by learned agreement between
neighboring transported frames, keep weights on the simplex, and permit zero
flow across an inferred boundary.

### 3. Keep structural state and residual surplus distinct

The Meyer split carries a persistent cartoon component and a complementary
texture component; recomposition also carries model residual unchanged. For
the neural layer, the analogous design is not two duplicate full branches. It
is one slowly transported structural state plus a local residual surplus that
can correct it without rewriting the frame field. That may preserve the
localized-step and ripple advantages of self-context while retaining the
high-rank gains of continuous flow.

### 4. Hierarchy should restrict reach, not multiply patches

Segmenting v3 rejected a literal product of cartoon and texture IDs because it
manufactured tens of thousands of fragments. Its successful nesting assigns
each texture germ to one parent, transports only among siblings during
construction, and later discards the parent identity. A neural nested chart
should therefore constrain the second transport by the first chart's inferred
conductance, not concatenate a second independent chart state. The earlier
nested-self-context experiment did the latter and consequently supplied width
without a transport law.

## Current conclusion

Continuous frame flow is the new experimental baseline because its advantage
is strongest exactly where a coherent changing frame ought to matter: high-rank
distributed structure, chirp, checkerboard tails, and early radial acquisition.
It is not yet the production mechanism. The evidence says that frame
correspondence is necessary but insufficient; the next missing object is a
transport permission field.

The next two principled ablations are therefore:

1. **Frozen-center metric shell** — reuse one measured allocation metric over
   all shell samples and compare quality plus runtime.
2. **Conductance-gated frame diffusion** — diffuse simplex weights only between
   adjacent views whose transported evidence agrees, with an exact no-flow
   option.

Neither ablation adds task coordinates, target-derived structure, auxiliary
labels, or trainable parameters.

The complete graphical report is `continuous_frame_full.html`. Raw benchmark,
paired-seed summary, and full visual probes are in
`results_continuous_frame_full/`.

