# `optimizer.py` on Ripple

## Question

Ripple is a useful stability falsifier because AdamW normally learns it well
while full Muon is slow and can leave a badly learned seed. Does the adaptive
optimizer inherit Muon's failure, or does its residual observation keep the
update away from inappropriate full polar geometry?

The comparison uses literal implementation names:

- **AdamW**: ordinary PyTorch AdamW;
- **Muon**: ordinary Muon on hidden matrices with AdamW auxiliaries;
- **Matrix Transport**: the fixed transported-momentum plus longitudinal
  closure baseline;
- **`optimizer.py`**: `ResidualPolarTransport`, which begins from Matrix
  Transport and continuously mixes norm-matched polar shape according to the
  observed residual entropy rank.

## Protocol

Five paired seeds use the complete width-24 self-context model, exact
self-context backpropagation, batch size 256, learning rate `0.003`, 750
optimizer updates, and evaluation every five updates. Every arm receives the
same initialization and minibatch stream within a seed.

`optimizer.py` uses its documented `attach_model(model)` path. Its state comes
from each linear module's training activation/output-cotangent relation. It
does not receive the task's analytic Ripple formula, a validation loss, a
held-out residual, an iteration schedule, or a task label.

## Result

| optimizer | mean acquisition AUC | median steps to `.80` | steps to `.90` | mean held-out MSE | worst-seed held-out MSE | Jacobian variability | mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | `0.7713` | 345 | 5/5, median 525 | `0.04940` | `0.09981` | `7.92` | `4.88` |
| Muon | `0.7014` | 465 | 4/5, median 553 | `0.05334` | `0.12493` | `6.09` | `5.20` |
| Matrix Transport | **`0.8033`** | **265** | 4/5, median **345** | `0.04192` | `0.11123` | `8.33` | `6.56` |
| **`optimizer.py`** | `0.7905` | 270 | **5/5**, median 370 | **`0.03195`** | **`0.05966`** | **`3.82`** | `10.13` |

There are no NaNs or explicit numerical failures in any arm at the standing
rate. Muon's failure is optimization failure rather than floating-point
failure: it has the lowest acquisition AUC, misses `.90` in one seed, and ends
that seed at held-out MSE `0.12493`. Matrix Transport is the fastest average
acquirer but also misses `.90` in one seed and has held-out MSE `0.11123` there.

`optimizer.py` reaches `.90` in every seed, has the best mean and worst-seed
held-out error, and cuts average Jacobian variability by more than half versus
both AdamW and fixed Matrix Transport. It gives up 25 median updates at the
`.90` threshold relative to the four successful Matrix Transport seeds, but
removes that arm's bad seed. This is a stability/acquisition trade rather than
a cosmetic endpoint win.

The observer currently makes the reference implementation about 55% slower
than Matrix Transport in wall time. That cost is not hidden in the iteration
result. The fixed `observer_dim=32` sketch still runs on every backward pass;
observation cadence and reuse are the next engineering optimization.

## What the controller actually does

The new optimizer does **not** decide that Ripple needs zero polar geometry.
It also does not approach full Muon. Across all five seeds:

| update | median residual entropy rank | median polar weight |
|---:|---:|---:|
| 1 | `0.555` | `0.044` |
| 25 | `0.564` | `0.397` |
| 100 | `0.568` | `0.424` |
| 300 | `0.590` | `0.414` |
| 500 | `0.592` | `0.412` |
| 750 | `0.582` | `0.415` |

Ripple therefore holds the optimizer near a stable mixed update: roughly 60%
Matrix Transport and 40% norm-matched polar shaping. In the earlier anisotropic
operator tail, polar weight rose to about `0.8`. The controller is separating
the two geometries rather than merely turning Muon on after a delay.

The state is also live rather than monotone. At the five-update observation
cadence, polar weight decreases on 67--78 of the 150 intervals in every seed.
When a minibatch or layer exposes broader residual support, the controller
returns weight to Matrix Transport; when a narrow residual persists, it earns
some polar shaping again.

## Conclusion

Ripple passes the intended stability test. The adaptive optimizer survives the
problem where Muon's polar decision is a poor global default, learns faster
than AdamW, and is more robust than fixed Matrix Transport. The mechanism is
not "never use polar." It is "do not commit to polar": retain a reversible,
state-conditioned mixture whose position depends on the unresolved operator.

This is evidence for a stable optimizer idea, not yet a universal claim. The
next decisive check is the unseen battery with the same file, default settings,
automatic hooks, and no problem-specific retuning.

## Observation-site ablation

The complete self-context model contains ten `nn.Linear` modules. Observing all
ten is not mathematically necessary:

- the scalar output and the two scalar response heads cannot express residual
  anisotropy because they have only one singular value;
- consecutive affine views before a nonlinearity largely repeat the same
  residual relation in different coordinates;
- the external LELU between the `up` and `down` self-context blocks creates the
  important new tangent geometry.

The observer was therefore restricted to the two matrices that immediately
consume the LELU representation:

    model.down.base
    model.down.metric

Every other matrix remained on unmodified Matrix Transport. No learning rate,
optimizer parameter, seed, batch, or training protocol changed.

| observer scope | observed matrices | mean AUC | median `.90` | mean held-out MSE | worst seed | Jacobian variability | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| every linear | 10 | `0.7905` | **370** | **`0.03195`** | `0.05966` | **`3.82`** | `10.13` |
| after the main nonlinearity | **2** | **`0.7949`** | 415 | `0.03257` | **`0.04501`** | `6.63` | **`6.92`** |

Two observation sites remove 80% of the hooks and 32% of total fit time. The
result is only 5.5% slower than fixed Matrix Transport (`6.56 s`), slightly
improves acquisition AUC, preserves the mean held-out result, and improves the
worst seed. The trade is slower arrival at `.90` and less Jacobian smoothing
than the all-layer observer.

The reported layer-mean polar weight is `0.139` because eight of ten matrices
remain at exactly zero. The two observed matrices settle near `0.69` polar
weight, while the rest of the model stays on Matrix Transport. This is the more
precise form of the stability mechanism: strong polar geometry may be locally
correct after a nonlinearity even when allowing the entire model to become
Muon-shaped is wrong.

Observation is therefore mandatory only where an adaptive polar decision is
desired and a multidimensional residual relation exists. It is not required
for the underlying optimizer step. An unobserved matrix begins and remains on
Matrix Transport unless it is explicitly supplied a global rank. The next
engineering step is to add observation cadence, because both selected ranks
stabilize quickly and do not need a fresh regression on every backward pass.
