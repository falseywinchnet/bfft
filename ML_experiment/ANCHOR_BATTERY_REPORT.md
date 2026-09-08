# Anchor 1.0: full paired optimizer battery

Date: 2026-08-29  
Status: completed, 432/432 fits; no numerical failures

## Question

The earlier ill-conditioned MLP assay showed that transported lead--lag
momentum with directional restraint and a local trust controller could reach
SGD's final floor in 75 updates rather than 2,999.  That result established a
mechanism worth testing; it did not establish a generally superior optimizer.

This battery asks a narrower and more reproducible question:

> With Anchor held at its standing `lr=1.0`, does it acquire useful validation
> behavior in fewer updates than fixed AdamW and SGD baselines across the
> existing m-layer problem suite?

The target is iteration efficiency.  Wall-clock time, final validation, and
held-out continuation are reported, but they are not substituted for the
primary acquisition quantity.

## Standing optimizer

For each row-local parameter cell, let `u` be the current normalized gradient,
`r` the carried signature, and `q = normalize(r + u)` when the midpoint is
defined.  Let `s` be the slow request.  With `c = exp(-1)` and `a = 1-c`, one
step is

```text
s_bar  = T[r -> q](s)
f      = a s_bar + c g
s_next = a s_bar + c f
d      = u - r
tau    = (1 - dot(u, r)) / 2
v      = f - tau P_d(f)
```

`T` is the minimum spherical rotation.  At an antipodal collapse there is no
unique shortest transport, so Anchor resets the affected slow cell instead of
choosing an arbitrary turn.  Only `v` is applied; `s_next` retains the
unrestrained transported request.  Since the restraint multiplies the
departure-axis component by `1-tau` and leaves its orthogonal complement
unchanged, `||v|| <= ||f||` for `tau in [0,1]`.

The nominal learning-rate ceiling remains `eta_max=1.0`.  A row-local
multiplier certifies the relative step:

```text
b      = max(||p||, 1)
h      = min(1, 0.02 b / (eta_max ||v||))
lambda = h                                      when h < lambda_previous
lambda = lambda_previous + 0.05(h-lambda_previous) otherwise
update = -eta_max lambda v
```

Thus `eta_max lambda ||v|| / b <= 0.02` at every step.  Braking is immediate;
recovery moves only 5% toward the newly admissible multiplier.  There is no
loss lookahead, second moment, per-example gradient, or task scheduler.

## Protocol

The test matrix contains 24 tasks, two parameter-matched model variants, three
paired seeds, and three optimizers: 432 fits in total.

| Dimension | Setting |
|---|---|
| Tasks | 24 classification and regression generators |
| Models | ordinary MLP; self-context |
| Seeds | 0, 1, 2 |
| Width | 24 |
| Training budget | 500 parameter updates |
| Batch | 256 |
| Validation cadence | step 1, then every 5 updates |
| Gradient clipping | shared global norm cap of 5 |
| Weight decay | `1e-4` for all three optimizers |
| AdamW | `lr=0.003`, PyTorch defaults |
| SGD | `lr=0.03`, no momentum |
| Anchor | `lr=1.0`, trust radius `0.02`, recovery `0.05` |

Within a task, architecture, and seed, initialization and minibatch schedule
are identical across optimizers.  No learning-rate sweep was performed.  This
is a standing-configuration mechanism assay, not a claim that each baseline
was tuned to its per-task optimum.

Classification score is mean class recall.  Regression score is
`1/(1+MSE)`.  The primary metric is trapezoidal area under the sampled
validation-score curve divided by the 500-update budget.  A robust Anchor win
requires both:

1. positive mean acquisition-AUC margin over the stronger of AdamW and SGD;
2. a per-seed AUC win over both baselines on at least two of three paired
   seeds.

The speed ratio is computed per seed against the higher best-validation score
attained by either baseline.  It is
`baseline updates / Anchor updates`, so values above one favor Anchor.  The
ratio is reported only for seeds on which Anchor reaches that target.
Held-out score is evaluated after restoring the checkpoint with the best
validation score; it is not included in the acquisition winner rule.

## Aggregate result

| Optimizer | Runs | Mean acquisition AUC | Mean best validation | Mean held-out | Mean tail score | Mean fit time | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 144 | **0.7712** | **0.8401** | **0.6222** | **0.6033** | 3.017 s | 0 |
| Anchor | 144 | 0.7280 | 0.8108 | 0.5990 | 0.5840 | 4.750 s | 0 |
| SGD | 144 | 0.6343 | 0.6874 | 0.5746 | 0.5399 | **2.911 s** | 0 |

The honest aggregate conclusion is therefore not “Anchor wins the battery.”
Anchor is a stable second: its mean acquisition AUC exceeds SGD by 0.0937 and
trails AdamW by 0.0431.  It has no numerical failures at a nominal ceiling
roughly 333 times the AdamW learning rate and 33 times the SGD learning rate,
but the row-local transport and trust bookkeeping make this Python reference
implementation about 1.58 times as expensive per fit as AdamW on CPU.

## Robust acquisition winners

Anchor satisfies the paired winner rule on 9 of 48 task × model pairs.  Seven
are ordinary MLPs and two are self-context models.

| Task | Model | AUC margin | Paired AUC wins | Validation margin | Held-out margin | Target reached | Median speed ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| complex spiral 3-D | ordinary | +0.02786 | 3/3 | -0.01005 | -0.06051 | 1/3 | 1.250 |
| pinwheel | ordinary | +0.01545 | 3/3 | +0.00050 | +0.00050 | 3/3 | 1.880 |
| swiss cheese | ordinary | +0.01108 | 3/3 | -0.00027 | -0.00027 | 2/3 | 1.165 |
| sparse sine 1-D | ordinary | +0.00679 | 3/3 | +0.01463 | +0.00906 | 3/3 | 1.702 |
| pinwheel | self-context | +0.00560 | 3/3 | 0.00000 | 0.00000 | 3/3 | 1.217 |
| periodic wells | ordinary | +0.00464 | 2/3 | +0.02696 | +0.02696 | 2/3 | 1.273 |
| periodic N-D | self-context | +0.00402 | 3/3 | +0.01390 | +0.01390 | 3/3 | 1.119 |
| two moons | ordinary | +0.00367 | 3/3 | 0.00000 | 0.00000 | 3/3 | 1.148 |
| sinusoid bounds | ordinary | +0.00222 | 2/3 | +0.00421 | +0.00421 | 2/3 | 1.562 |

The median AUC margin among these winners is 0.00560.  The median reported
speed ratio is 1.25.  Five of nine winners also have a positive held-out
margin; two are tied; two are negative.

Pinwheel ordinary and sparse-sine ordinary are the cleanest complete wins:
Anchor wins AUC on all seeds, reaches the baseline target on all seeds, has a
positive endpoint and held-out margin, and uses materially fewer updates.
Periodic wells ordinary and periodic N-D self-context are smaller but coherent
wins.  Complex spiral ordinary is different: the AUC gain is the largest in
the battery, yet Anchor reaches the baseline's final validation target on only
one seed and loses held-out continuation.  It is evidence of fast early
acquisition, not a claim of a better finished model.

## Where it loses

The largest negative AUC margins are not hidden by the winner view:

| Task | Model | Anchor AUC margin | Validation margin | Held-out margin |
|---|---|---:|---:|---:|
| ring SDF | ordinary | -0.2880 | -0.0294 | -0.0294 |
| complex spiral 3-D | self-context | -0.1784 | -0.1293 | -0.0403 |
| localized steps 1-D | ordinary | -0.1504 | -0.0361 | -0.0144 |
| checkerboard | self-context | -0.1389 | -0.0389 | -0.0530 |
| polynomial drifted chirp 1-D | self-context | -0.1346 | -0.0877 | -0.1613 |

This split is informative.  The same complex-spatial generator that benefits
under the ordinary MLP strongly rejects Anchor under self-context.  Likewise,
seven of nine robust wins occur in the ordinary architecture.  The optimizer
is therefore interacting with representation dynamics; restraint is not a
task-independent acceleration theorem.  In particular, a cell-local relative
step certificate controls magnitude but cannot decide whether the transported
direction is statistically healthy for a self-context allocation mechanism.

## What the battery establishes

1. **Stability at the standing ceiling.** Anchor completes every fit at
   `lr=1.0`; the local controller tames the deliberately large nominal rate.
2. **Real iteration wins exist outside the original assay.** Nine paired
   problem/model combinations meet a predeclared majority-seed AUC rule, and
   every available median speed ratio among them exceeds one.
3. **The win is structured.** Ordinary MLPs account for seven of nine wins;
   self-context can either benefit or suffer sharply.
4. **Anchor is not an aggregate AdamW replacement.** AdamW remains ahead in
   mean AUC, best validation, held-out score, tail score, and CPU time.
5. **Acquisition and continuation must remain separate.** Four of nine AUC
   winners do not improve held-out score; the largest AUC winner has the
   clearest continuation warning.

## Reproduction and artifacts

The frozen raw result is `results_anchor_battery/results.json`; the derived
paired analysis is `results_anchor_battery/summary.json`.  Recompute the
summary and browser evidence with:

```sh
python3 -m ML_experiment.analyze_anchor_battery \
  ML_experiment/results_anchor_battery/results.json \
  --out ML_experiment/results_anchor_battery/summary.json \
  --web-out ../paymenottowork/web/ideas/anchor-optimizer/evidence.json
```

The optimizer used for every Anchor fit is `ML_experiment/anchor.py`.  The
public exhibit copies that file byte-for-byte and plots mean validation curves
with the full three-seed range rather than a smoothed or selected seed.
