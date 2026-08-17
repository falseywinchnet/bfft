# Relational response enhancement battery

## Question

Does giving the allocator its authentic chart coordinates, transported chart
coordinates, their displacement, the unreduced metric action, and
coordinate-wise alignment produce a cheaper substitute for Continuous Frame
Flow (CFF)? Does the same response enrichment plus a second LELU improve CFF?

## Matched protocol

- 23 tasks from `ML_experiment.tasks.TASK_BUILDERS`
- width 38, batch 256, 500 AdamW steps
- two paired seeds
- validation checkpointing uses only the observed region
- M4 CPU, eight Torch threads
- training-kernel time and median inference time for 256 examples measured
  separately
- held-out score, tail retention, learning AUC, Jacobian variability, allocator
  entropy, and fitted-function probes retained

The four models use the same encode → expansion → LELU → contraction → decode
topology. They differ only in the two structured layers:

1. original self-context;
2. relational SCL;
3. baseline Continuous Frame Flow;
4. relational CFF with a two-LELU response head.

## Overall result

| model | mean params | train seconds/task | inference ms/256 | learning AUC | held-out | tail | allocation entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original self-context | 8,616 | 2.45 | 1.92 | 0.852 | 0.672 | 0.647 | 0.757 |
| Relational SCL | 9,168 | 4.03 | 3.35 | 0.856 | 0.657 | 0.630 | 0.522 |
| Baseline CFF | 8,616 | 8.96 | 6.32 | 0.864 | 0.676 | 0.661 | 0.840 |
| Relational deep CFF | 9,480 | 17.36 | 12.76 | 0.859 | 0.658 | 0.636 | 0.723 |

Relational SCL is 2.23× faster to train and 1.89× faster at inference than
baseline CFF. It remains within one held-out score point of CFF on 18 of 23
tasks, despite using 6.4% more trainable parameters. Its mean held-out score is
1.9 points lower, driven by a small set of large structured failures.

The response-enhanced deep CFF is dominated overall. It is 1.94× slower to
train and 2.02× slower at inference than baseline CFF while reducing held-out
score, tail retention, and learning AUC. More response depth is not the missing
CFF ingredient.

## Where relational SCL works

- Multiscale 1-D: 0.324 versus CFF 0.179 and original self-context 0.258.
- High-rank N-D spiral: 0.802 versus CFF 0.784, although original
  self-context averaged 0.841 with very high seed variance.
- Chirp: 0.292 versus CFF 0.248, with a 0.025 learning-AUC gain over original
  self-context; original self-context still retained the best extrapolation.
- Spiral, low-rank N-D spiral, sinusoidal bounds, and most easy/local problems:
  approximately CFF-quality behavior at much lower compute cost.

## Where CFF remains structurally distinct

- Radial stripes: CFF 0.859 versus relational SCL 0.566. This is the clearest
  repeatable CFF-only win.
- Polynomial-drifted chirp: CFF 0.301 versus relational SCL 0.128.
- Ripple and localized steps retain smaller CFF advantages.
- Fourier mix is highly seed-sensitive; relational SCL averages 0.338 versus
  CFF 0.487. The fitted curves show off-support chart selection producing
  unstable continuation.

## Diagnostic interpretation

The enriched response does what the motivating ablation suggested: it makes
the allocator itself a stronger nonlinear learner. Relational SCL has much
lower normalized allocation entropy (0.522 versus CFF's 0.840) and much higher
mean maximum chart ownership. That sharper acquisition often accelerates
in-range fitting, but it can prematurely commit to one chart and lose the
broad mixture needed for distributed or extrapolative structure.

CFF's useful behavior is therefore not reproduced by adding more information
or depth to its response. Its radial advantage coincides with a broad frame
distribution while symmetric curvature probes integrate across that
distribution. The relational deep CFF narrows the allocation and largely
destroys the radial win.

The next principled experiment is not another response MLP. It is to retain the
relational SCL response while explicitly controlling chart concentration—for
example, a continuous entropy floor or concentration-dependent blending that
preserves competing charts without supervising which chart should win.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_response_enhanced_battery \
  --out /tmp/response_enhanced_full_20260816 \
  --width 38 --seeds 2 --steps 500 --batch 256 --eval-every 25 --grid 71
```

Raw results and probes are in `ML_experiment/results_response_enhanced/`.
