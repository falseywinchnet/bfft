# Eight-turn spiral: capacity versus optimization

## Experiment

This sweep isolates three explanations for the eight-turn breakdown on the
fixed-pitch dual spiral:

1. **optimization only** — width 38 at 500, 1,000, and 2,000 AdamW steps;
2. **capacity only** — widths 38, 54, and 76 at 500 steps;
3. **joint scaling** — width 54 at 1,000 steps and width 76 at 2,000 steps.

Vanilla LELU MLP, self-context, and continuous frame flow are exactly
parameter-matched within each configuration. Results average four paired seeds.
The width-38/500 baseline is reused from the evidence-horizon experiment; the
sweep adds 72 fitted models.

## Mean results

| Width | Steps | Parameters | Model | Observed | Withheld | Learning AUC |
|---:|---:|---:|---|---:|---:|---:|
| 38 | 500 | 8,542 | Vanilla | .522 | .503 | .506 |
| 38 | 500 | 8,542 | Self-context | .529 | .503 | .511 |
| 38 | 500 | 8,542 | Frame flow | .574 | .506 | .520 |
| 38 | 1,000 | 8,542 | Vanilla | .531 | .502 | .509 |
| 38 | 1,000 | 8,542 | Self-context | .585 | .506 | .528 |
| 38 | 1,000 | 8,542 | Frame flow | .681 | .504 | .569 |
| 38 | 2,000 | 8,542 | Vanilla | .538 | .503 | .513 |
| 38 | 2,000 | 8,542 | Self-context | .739 | .505 | .583 |
| **38** | **2,000** | **8,542** | **Frame flow** | **.802** | **.508** | **.655** |
| 54 | 500 | 15,518 | Vanilla | .532 | .502 | .506 |
| 54 | 500 | 15,518 | Self-context | .531 | .504 | .510 |
| 54 | 500 | 15,518 | Frame flow | .549 | .506 | .517 |
| 76 | 500 | 28,454 | Vanilla | .528 | .505 | .506 |
| 76 | 500 | 28,454 | Self-context | .535 | .504 | .511 |
| 76 | 500 | 28,454 | Frame flow | .536 | .504 | .511 |
| 54 | 1,000 | 15,518 | Vanilla | .536 | .505 | .509 |
| 54 | 1,000 | 15,518 | Self-context | .615 | .505 | .533 |
| 54 | 1,000 | 15,518 | Frame flow | .639 | .502 | .555 |
| 76 | 2,000 | 28,454 | Vanilla | .532 | .505 | .509 |
| 76 | 2,000 | 28,454 | Self-context | .656 | .506 | .557 |
| 76 | 2,000 | 28,454 | Frame flow | .653 | .504 | .560 |

## Diagnosis

The breakdown is primarily an optimization/effective-capacity interaction, not
a shortage of raw parameters.

At fixed width 38, extra steps produce a monotone structural gain. Continuous
frame flow rises from `.574` to `.681` to `.802` observed accuracy; self-context
rises from `.529` to `.585` to `.739`. Vanilla remains near chance. Thus the
contextual architectures possess a trainable route toward the eight-turn
partition, and frame transport makes that route substantially easier.

Raw width alone does not help. At 500 steps, increasing the matched budget from
8,542 to 15,518 and 28,454 parameters leaves every model near chance. Joint
width/step scaling also underperforms the smaller width-38/2,000 system. AdamW
cannot efficiently organize the additional degrees of freedom at this budget;
the larger search space reduces effective rather than nominal capacity.

No configuration extrapolates. Every withheld score lies between `.502` and
`.508` in the aggregate. Even when frame flow partially recovers the eight-turn
observed geometry, it still learns a finite fitted partition rather than the
radial-to-angular generator.

The graphical report is `spiral_capacity_sweep.html`; raw fits, averaged
metrics, and fitted decision fields are in `results_spiral_capacity_sweep/`.
