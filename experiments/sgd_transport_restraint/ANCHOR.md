# Anchor: transported lead--lag momentum with a dynamic trust controller

Date: 2026-08-29. Status: mechanism result confirmed on a 432-fit paired battery.

The complete recurrence derivation, Euclidean/Bregman comparison, scalar
stability calculation, and falsifiable mechanism assays are recorded in
`ML_experiment/ANCHOR_FORMAL_DECOMPOSITION.md`.

## Standing optimizer

Anchor is the deterministic two-timescale core recovered from the earlier Wolf
experiment, renamed for its slow carried state.  With `c=exp(-1)` and
`a=1-c`, it parallel-transports the slow state into the current signature frame
before forming the fast request:

```text
s_bar  = T[r -> q](s_t)
f_t    = a s_bar + c g_t
s_next = a s_bar + c f_t
v_t    = f_t - tau P_(u-r) f_t
```

Only `v_t` reaches the parameters.  The stored state is transported but never
restrained.  The projection cannot amplify the fast proposal.

## Dynamic learning rate with a fixed ceiling

The nominal learning rate remains `eta_max=1.0`.  For each local parameter
cell, define its proposed relative displacement

```text
b_t   = max(||p_t||, 1)
rho_t = eta_max ||v_t|| / b_t
```

and a target multiplier for trust radius `delta=0.02`:

```text
h_t = min(1, delta / rho_t).
```

The active multiplier brakes immediately but recovers slowly:

```text
lambda_t = h_t                                      if h_t < lambda_(t-1)
lambda_t = lambda_(t-1) + gamma(h_t-lambda_(t-1))  otherwise
gamma = 0.05
```

The applied update is `-eta_max lambda_t v_t`.  In both branches
`lambda_t <= h_t`, therefore

```text
eta_max lambda_t ||v_t|| / b_t <= delta.
```

This is a cheap local certificate: no loss history, extra forward pass,
per-example gradient, global scheduler, or task-specific learning rate is
used.  A troubled cell brakes without spending another cell's stability
budget.  The unit floor lets zero-initialized biases begin moving.

## Does it preserve the speed result?

Yes.  On the original fixed ill-conditioned MLP, both fixed and dynamic Anchor
first reach SGD's final test floor at transition 75.  The dynamic controller
reduces the maximum aggregate relative step from 2.10% to 1.21%; its 95th
percentile falls from 1.04% to 0.85%.  It removes later pressure without
slowing the launch.

## Six-task contrast screen

The screen used six structurally different m-layer tasks, ordinary and
self-context models, width 24, seed 0, 200 transitions, shared clipping at 5,
and a fixed Anchor ceiling of `1.0`.  Dynamic Anchor was paired against fixed
Anchor from the identical initialization and minibatch schedule.

Across the 12 pairs:

- learning AUC improved in 11/12;
- best validation score improved in 10/12;
- mean learning AUC rose from 0.3411 to 0.5664;
- mean validation score rose from 0.5062 to 0.6231;
- held-out/extrapolation score improved in only 5/12.

The clearest stabilization was periodic wells:

| model | fixed Anchor | dynamic Anchor | SGD | AdamW |
|---|---:|---:|---:|---:|
| ordinary MLP | 0.1401 | **0.5575** | 0.5118 | 0.5154 |
| self-context | 0.5149 | **0.8368** | 0.5135 | 0.9454 |

The controller therefore converts an overdriven LR-1.0 trajectory into useful
acquisition, while retaining the earlier 75-transition case.

## Boundary: stability is not continuation

On the complex 3-D spiral, dynamic Anchor improved validation acquisition from
0.5009 to 0.6813 for the ordinary MLP and from 0.4935 to 0.7989 for
self-context.  Its extrapolation score nevertheless fell to 0.0009 and 0.0147.
Vanilla SGD retained the best extrapolation score in this short screen.

This is not a numerical explosion: it is rapid acquisition of an unhealthy
continuation.  A learning-rate controller cannot by itself identify the
generator outside observed support.

An optional 2x row-RMS update-concentration cap improved the two self-context
regressions but damaged their ordinary-MLP counterparts.  It remains an
ablation (`anchor_isotropic`), not part of standing Anchor.

## Full AdamW/SGD/Anchor battery

The completed battery contains 24 tasks, ordinary and self-context networks,
three paired seeds, 500 updates, and fixed optimizer settings: AdamW 0.003,
SGD 0.03, and standing Anchor 1.0.  All 432 fits completed without numerical
failure.

Anchor is a stable specialist, not the unfiltered winner.  Mean acquisition
AUC is 0.7280, between SGD at 0.6343 and AdamW at 0.7712.  Under the declared
winner rule--positive mean AUC margin over the stronger baseline plus paired
wins on at least two of three seeds--Anchor wins 9 of 48 task/model pairs.
Seven wins are ordinary MLPs and two are self-context.  The median speed ratio
among those winners is 1.25 baseline updates per Anchor update.  Five of the
nine acquisition winners also improve held-out score.

The complete protocol, winner table, aggregate result, and largest losses are
in `ML_experiment/ANCHOR_BATTERY_REPORT.md`.  Raw and derived data are frozen
under `ML_experiment/results_anchor_battery/`.
