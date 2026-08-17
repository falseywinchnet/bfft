# Continuous frame flow: optimizer, capacity, and speed

## Result

Continuous frame flow remains the strongest general mechanism in this branch,
but its three proposed refinements do different jobs. Muon improves held-out
continuation, width improves acquisition inside observed support, and the
two-probe shell buys real speed with a task-dependent approximation cost.

The confirmation contains 84 M4 CPU fits: seven diagnostic problems, six named
configurations, two paired seeds, and 500 steps. The ordinary MLP, self-context,
reference frame, Muon frame, and fast frame all use the same trainable parameter
budget at width 24 (4,069–4,454 parameters depending on task I/O). Only the
capacity experiment increases width, to 6,389–6,902 parameters.

| Configuration | Mean score | Mean tail | Learning AUC | Time / fit | Paired interpretation |
|---|---:|---:|---:|---:|---|
| Ordinary LELU MLP | 0.3120 | 0.2686 | 0.6613 | 0.26 s | Parameter-matched functional control |
| Self-context | 0.4727 | 0.3999 | 0.8432 | 3.96 s | First-order chart reinterpretation |
| Continuous frame flow (AdamW) | 0.4925 | 0.4004 | 0.8697 | 14.52 s | Full four-ray curvature transport |
| Continuous frame flow (Muon) | **0.5378** | **0.4404** | 0.8266 | 14.93 s | +0.0453 score, +0.0400 tail, −0.0431 AUC |
| Continuous frame flow (width 32) | 0.4579 | 0.3735 | **0.8855** | 15.12 s | +0.0158 AUC on all 7 tasks, −0.0346 score |
| Continuous frame flow (two-probe) | 0.4634 | 0.3943 | 0.8642 | **12.12 s** | 16.5% faster, −0.0061 mean tail |

Regression score is `1 / (1 + normalized MSE)`; classification score is mean
class recall. Averages are diagnostic, not claims that unlike tasks have one
natural scalar utility.

## Where each form wins

| Problem | Best held-out form | Score | What it diagnoses |
|---|---|---:|---|
| Radial stripes | Continuous frame flow (Muon) | 0.8663 | Muon removes much of the AdamW run's seed brittleness |
| High-rank N-D spiral | Continuous frame flow (two-probe) | 0.8848 | Dense mixed probes can regularize a high-rank tangent trace |
| Multiscale 1-D | Self-context | 0.4597 | First-order local focality remains valuable |
| Chirp 1-D | Continuous frame flow (width 32) | 0.4649 | Extra capacity helps a smoothly changing local frequency |
| Polynomial drifted chirp | Continuous frame flow (Muon) | 0.3251 | Orthogonalized hidden updates improve continuation strongly |
| Localized steps | Self-context | 0.9420 | Additional frame state can damage piecewise focal continuation |
| Complex 3-D spiral | Continuous frame flow (Muon) | 0.0597 | Absolute extrapolation remains difficult; Muon is best by a wide relative margin |

The polynomial chirp and 3-D spiral are now permanent benchmark tasks. Their
plots expose the observed boundary explicitly. The 3-D view contains both the
observed segment and the unseen continuation; it does not infer the scene from
future coordinates.

## Mechanistic reading

Muon is not merely “a better AdamW.” It replaces hidden matrix-gradient
magnitude structure with an approximately semi-orthogonal update, while edge
maps, biases, and scalar parameters remain on AdamW. In this experiment that
reduces the optimizer's preference for a few already-large matrix directions.
The result is worse average in-field acquisition AUC but better continuation on
four of seven tasks, especially polynomial drift and the 3-D spiral. This is
consistent with the hypothesis that the chart contains useful dimensional
signals that ordinary backprop can under-utilize; it is not yet proof of that
causal account. The split follows the [official Muon
implementation](https://github.com/KellerJordan/Muon) and current [PyTorch Muon
API](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).

Width 32 is almost the converse. It improves acquisition AUC on every task but
reduces average held-out score and tail retention. More representational room
lets the optimizer fit the observed chart sooner; it does not force the chart
to select the continuation-compatible state. Capacity is therefore a knob,
not the new default.

The fast mode contracts two fixed dense directions through the full rank-four
tangent shell. It is a deterministic Hutchinson-like trace estimate: no named
task axis is removed and no labels enter the probe. It is excellent on the
high-rank spiral, but loses focal information on some 1-D tasks. It should be
exposed as an explicit speed/variance option.

Finally, 10% of the reference runtime was accidental measurement overhead.
Eigenvalue, Jensen–Shannon, and duplicate base-path calculations now run only
when diagnostics are requested. On the identical radial/high-rank timing
screen, width-24 reference time fell from 9.41 s to 8.45 s and two-probe time
from 7.90 s to 7.06 s, with no function change.

## Standalone use

`continuous_frame_flow.py` depends only on PyTorch and contains `LELU`, the
reference network, fast/frozen shell options, and the hybrid CPU Muon helper.

```python
from ML_experiment.continuous_frame_flow import (
    ContinuousFrameFlow,
    MuonWithAuxAdamW,
)

model = ContinuousFrameFlow(input_dim=2, output_dim=2, width=24)
optimizer = MuonWithAuxAdamW(model, lr=3e-3)

# Optional speed mode: the same trainable degrees of freedom, two dense probes.
fast_model = ContinuousFrameFlow(2, 2, width=24, fast=True)
```

The standalone implementation is tested for numerical equality against the
research implementation after state transfer, not merely for matching output
shape.

## Reproduce

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/test_experiment.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/run_frame_refinement.py \
  --out /tmp/frame_refinement_full --seeds 2 --steps 500
```

The raw confirmation, fitted probes, paired summary, speed screens, and
optimizer/capacity screens are retained under `results_frame_*`.
