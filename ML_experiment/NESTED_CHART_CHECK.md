# Matched nested-chart check

## Question

Does a chart that interprets the change in another chart's allocation benefit
from emerging jointly with ordinary self-context, or from being activated only
after a self-context representation has formed?

## Construction

The first self-context step produces allocations `w0` and `w1`. Their centered
log ratio is the tangent displacement on the allocation simplex. That
displacement is lifted through the allocated primitive views and presented to
the same continuous atlas. The resulting outer chart modifies `w1` in log
coordinates. It changes selection, not activation position, and adds no
parameters.

## Protocol

- Radial stripes and multiscale 1-D only.
- Width 24, batch 256, 500 total steps, two paired seeds.
- Scratch nested chart: active from step 1.
- Staged nested chart: ordinary self-context for steps 1–250, nested chart for
  steps 251–500, with the same optimizer state and total step count.
- Ordinary and curvature self-context references.
- M4 CPU, LELU, identical parameter counts.

## Result

| Task | Configuration | Validation | Held-out | Tail | Learning AUC |
|---|---|---:|---:|---:|---:|
| Radial | Self-context | **.7638** | **.7638** | **.7638** | .5406 |
| Radial | Curvature context | .7575 | .7575 | .7575 | **.5903** |
| Radial | Nested chart, scratch | .7179 | .7179 | .7179 | .5475 |
| Radial | Nested chart, staged | .7444 | .7444 | .7444 | .5574 |
| Multiscale | Self-context | .9677 | **.4597** | **.4747** | .9020 |
| Multiscale | Curvature context | **.9873** | .3004 | .3946 | **.9188** |
| Multiscale | Nested chart, scratch | .9724 | .4422 | .4306 | .9028 |
| Multiscale | Nested chart, staged | .9697 | .3761 | .4033 | .9023 |

## Interpretation

The 300-step run was undertrained: every radial fit was still improving at its
endpoint, and its decision fields did not yet resemble the target. At 500
steps, radial self-context rises from `.5426` to `.7638`, so the corrected
figures are qualitatively and quantitatively different.

Ordinary self-context now has the best final held-out result on both tasks.
Nested-from-scratch stays close on multiscale (`-.0174`) but gives up `.0459`
on radial. Staging recovers some radial performance yet damages multiscale
continuation, so the run does not support either form of nesting as an
improvement.

Curvature self-context still learns fastest on both tasks, but the apparent
acquisition advantage does not survive as extrapolation: it finishes slightly
behind on radial and far behind on multiscale held-out and tail scores. That is
the useful signal here—curvature helps the observed-domain fit before the same
state distorts continuation.

This remains a narrow two-task diagnostic, but it is no longer the knowingly
underconverged comparison reported in the first draft.

The paired learning curves, radial decision fields, and multiscale continuation
fits are in `nested_chart_check.html`.
