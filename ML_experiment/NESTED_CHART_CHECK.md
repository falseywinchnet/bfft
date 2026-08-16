# Rapid nested-chart check

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
- Width 24, batch 256, 300 total steps, two paired seeds.
- Scratch nested chart: active from step 1.
- Staged nested chart: ordinary self-context for steps 1–150, nested chart for
  steps 151–300, with the same optimizer state and total step count.
- Ordinary and curvature self-context references.
- M4 CPU, LELU, identical parameter counts.

## Result

| Task | Configuration | Validation | Held-out | Tail | Learning AUC |
|---|---|---:|---:|---:|---:|
| Radial | Self-context | .5426 | .5426 | .5426 | .4984 |
| Radial | Curvature context | **.5974** | **.5974** | **.5974** | **.5151** |
| Radial | Nested chart, scratch | .5468 | .5468 | .5468 | .4992 |
| Radial | Nested chart, staged | .5392 | .5392 | .5392 | .4976 |
| Multiscale | Self-context | .9554 | **.3325** | .3344 | .8632 |
| Multiscale | Curvature context | **.9699** | .2692 | **.3495** | **.8798** |
| Multiscale | Nested chart, scratch | .9550 | .3247 | .3322 | .8630 |
| Multiscale | Nested chart, staged | .9544 | .3191 | .3310 | .8633 |

## Interpretation

Staging provides no advantage. Scratch nesting beats staging on both held-out
scores (`+.0076` radial, `+.0056` multiscale), supporting the hypothesis that
the outer interpretation must co-emerge with the first chart rather than being
attached after the fact.

The effect is small. Scratch nested charts remain close to ordinary
self-context: `+.0042` on radial and `-.0078` on multiscale. The current
log-simplex transition is stable and inexpensive relative to the shell, but it
does not yet expose a strong additional signal.

Curvature self-context remains the clear rapid-acquisition winner on both tasks
(`+.0167` and `+.0166` learning AUC). On multiscale it repeats the central
tradeoff: better in-field acquisition and tail average, but worse overall
held-out continuation.

This is a deliberately short, underconverged diagnostic. It establishes that
warm staging is not the missing ingredient; it does not establish that nested
charts have no value at longer horizons or in higher-rank problems.
