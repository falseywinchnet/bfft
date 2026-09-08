# Optimizer geometry battery: Anchor, restrained memory, Muon, and Transport-Lepton

Date: 2026-08-29  
Status: complete; 864 neural fits plus 72 operator-witness runs; zero numerical failures

## Result

No single optimizer dominates the complete battery.

AdamW remains the strongest general acquisition optimizer at the frozen
standing configurations.  It has the highest mean validation-score AUC and
the highest mean best-validation score.  Muon, Transport-Lepton, Anchor, and
restrained-momentum Anchor form a remarkably tight second tier in mean AUC,
but their task-level behavior is not interchangeable.

The two new scientific results are:

1. **Restraining stored momentum is a real mechanism.**  Restrained-momentum
   Anchor wins 77 of 144 direct paired-seed AUC comparisons against Anchor and
   becomes the robust six-arm winner on eight task × architecture pairs,
   versus one for standing Anchor.  It does not raise the grand mean: several
   large self-context losses offset its more frequent small wins.
2. **The exact Muon theorem does not automatically transfer to practical
   Muon.**  Exact polar descent solves the anisotropic operator witness in one
   population update or four rank-deficient block updates.  Practical
   five-step Newton--Schulz Muon with momentum is best at the standing
   `lr=0.003` terminal objective, but does not fully recover the quiet operator
   modes in 500 updates.  At a unit LR it oscillates rather than terminating.

The second result identifies a concrete target for the next optimizer: retain
Muon's matrix-frame decision while giving its carried momentum a restraint or
local settling certificate near the floor.

![Six-arm optimizer battery summary](results_optimizer_shape_battery/summary.png)

The complete 48-panel validation and training-loss atlas is
`results_optimizer_shape_battery/atlas.html`.  It includes the operator
population/block trajectories under both standing and operator-native scales.

## 1. The six standing arms

The neural battery freezes one configuration per optimizer.  No task-specific
LR selection is performed.

| Optimizer | Standing LR | State and update geometry |
|---|---:|---|
| AdamW | 0.003 | coordinate first and second moments; decoupled decay |
| SGD | 0.03 | raw Euclidean gradient; no momentum |
| Anchor | 1.0 ceiling | transported rowwise lead--lag request; departure restraint on the applied request; 2% local relative-step certificate |
| Restrained-momentum Anchor | 1.0 ceiling | standing Anchor plus the same live departure contraction on the stored slow state |
| Muon | 0.003 | five Newton--Schulz steps on hidden matrix momentum; AdamW on edge matrices, biases, and gains |
| Transport-Lepton | 0.003 | transported smooth Bregman matrix state on hidden matrices; AdamW auxiliaries |

The hybrid routing for Muon and Transport-Lepton is deliberate and follows
their intended matrix-optimizer use: hidden two-dimensional weights receive
the matrix method, while parameters for which a matrix polar operation is not
defined remain on AdamW.  Anchor, restrained Anchor, SGD, and AdamW act on all
parameters directly.

All arms receive the same initialization, minibatch sequence, global gradient
norm cap of five, and `1e-4` neural weight decay inside a paired
task/architecture/seed cell.

## 2. Restrained-momentum Anchor

Standing Anchor transports the slow state `s`, constructs a faster request,
and then restrains only the applied request:

```text
s_bar    = T(s),
f        = a s_bar + c g,
s_raw+   = a s_bar + c f,
R        = I - alpha tau e e^T,
v        = R f,
s+       = s_raw+.                                      (1)
```

Here `c=exp(-1)`, `a=1-c`, `e` is the live normalized departure axis, and

```text
tau = (1-<u,r>)/2 = sin^2(theta/2).                     (2)
```

The new arm changes exactly one line:

```text
s+ = R s_raw+.                                          (3)
```

Thus the same hidden rank-one geometry shapes both the action and memory.  Its
spectrum is

```text
1                 on e^perp,
1-alpha tau       on e.                                 (4)
```

At a small or uncertain turn, `tau` is small and momentum retains the gambit.
At a confident reversal, rejected departure energy cannot be stored silently
and returned by slow momentum on a later update.  This is matrix/row geometry,
not elementwise momentum suppression.

The implementation records

```text
memory_ratio = ||R s_raw+|| / ||s_raw+|| <= 1.          (5)
```

and preserves standing Anchor's applied-request and relative-step invariants.
Mechanism tests verify that it contracts a live departure component and is
exactly identical to Anchor in a constant frame.

## 3. Neural protocol

The complete matrix is:

- 24 existing classification and regression tasks;
- ordinary budget-matched LELU MLP and self-context network;
- paired seeds 0, 1, and 2;
- width 24;
- 500 accepted parameter updates;
- batch size 256;
- validation at step 1 and every five updates;
- six frozen optimizer arms;
- 864 fits total.

There were no numerical failures.

The primary quantity is normalized trapezoidal area under the sampled
validation-score trajectory.  This measures acquisition per update.  Best
validation, held-out continuation from the best-validation checkpoint, and
fit time remain separate.

A robust task × architecture winner must have both:

1. the highest mean AUC among all six arms;
2. a per-seed AUC win against all five alternatives on at least two of three
   seeds.

This winner rule detects repeatable iteration wins.  It does not certify that
the problem was learned; chance-level null problems are identified explicitly
below.

## 4. Aggregate neural result

| Optimizer | Runs | Mean acquisition AUC | Mean best validation | Mean held-out | Mean fit time | Failures |
|---|---:|---:|---:|---:|---:|---:|
| **AdamW** | 144 | **0.77116** | **0.84013** | 0.62222 | 1.726 s | 0 |
| Muon | 144 | 0.72961 | 0.81378 | **0.62463** | 1.876 s | 0 |
| Transport-Lepton | 144 | 0.72958 | 0.80597 | 0.61012 | 1.931 s | 0 |
| Anchor | 144 | 0.72802 | 0.81076 | 0.59897 | 2.647 s | 0 |
| Restrained-momentum Anchor | 144 | 0.72714 | 0.81291 | 0.59595 | 2.695 s | 0 |
| SGD | 144 | 0.63428 | 0.68738 | 0.57463 | **1.659 s** | 0 |

AdamW's mean AUC lead over Muon is `0.04155`.  The four geometry-aware arms
are separated by less than `0.00247` mean AUC, yet they win different tasks.
Muon's mean held-out score exceeds AdamW by `0.00241`; this is small but paired
across a wide suite and establishes that the AUC ranking is not also the
continuation ranking.

Architecture split:

| Optimizer | Ordinary AUC | Ordinary held-out | Self-context AUC | Self-context held-out |
|---|---:|---:|---:|---:|
| AdamW | **0.70821** | 0.57275 | **0.83411** | **0.67169** |
| SGD | 0.63478 | 0.56887 | 0.63377 | 0.58038 |
| Anchor | 0.68322 | 0.58143 | 0.77283 | 0.61650 |
| Restrained-momentum Anchor | 0.68400 | 0.58159 | 0.77028 | 0.61032 |
| Muon | 0.67484 | **0.58660** | 0.78437 | 0.66266 |
| Transport-Lepton | 0.67606 | 0.57607 | 0.78310 | 0.64417 |

Restraining slow memory is slightly favorable on the ordinary architecture
but unfavorable in aggregate on self-context.  Muon is the closest challenger
to AdamW on self-context continuation.

## 5. Robust winner distribution

Of the 48 task × architecture pairs, 45 have a majority-seed robust winner.
The other three have a mean winner but only one seed agrees.

| Optimizer | Robust winner pairs |
|---|---:|
| AdamW | **32** |
| Restrained-momentum Anchor | **8** |
| SGD | 2 |
| Muon | 2 |
| Anchor | 1 |
| Transport-Lepton | 0 |

Counts alone overstate some wins.  SGD's two wins are ordinary radial stripes
and ordinary hypercube checker, where scores remain near chance.  Muon's
ordinary hyperchecker win is also essentially a null-problem ordering, and
its ordinary ripple win has an AUC margin of only `0.00013`.  Those results are
retained but are not evidence of meaningful problem solution.

The eight robust restrained-momentum Anchor winners are:

| Task | Architecture | AUC margin over runner-up | Seed wins | Mean held-out |
|---|---|---:|---:|---:|
| Periodic wells | ordinary | **+0.01399** | 2/3 | 0.7492 |
| Swiss cheese | ordinary | +0.00549 | 2/3 | 0.9577 |
| Periodic N-D | self-context | +0.00300 | 3/3 | 0.5990 |
| Sinusoid bounds | ordinary | +0.00232 | 2/3 | 0.9777 |
| Pinwheel | ordinary | +0.00048 | 3/3 | 0.9997 |
| Sparse sine | ordinary | +0.00044 | 2/3 | 0.5109 |
| Pinwheel | self-context | +0.00027 | 3/3 | 1.0000 |
| Two moons | ordinary | +0.00019 | 2/3 | 0.9997 |

Periodic wells is the strongest complete new-arm win.  Sparse sine remains a
weakly learned problem and should be interpreted only as a stable ordering.

Standing Anchor's single six-arm win is ordinary complex spiral, with an AUC
margin of `+0.01232`; its held-out score remains only about `0.0022`, so this is
fast interpolation rather than successful continuation.

## 6. Does restraining memory improve Anchor?

Across the 144 direct paired-seed comparisons:

```text
Restrained-memory Anchor beats Anchor: 77 / 144
median AUC delta:                       +0.000139
mean AUC delta:                         -0.000880
```

The sign discrepancy is informative.  A majority of changes are small and
positive, while a smaller set of losses is larger.

Largest mean AUC improvements over Anchor:

| Task | Architecture | AUC delta | Held-out delta |
|---|---|---:|---:|
| Ring SDF | ordinary | **+0.05964** | -0.00041 |
| Complex spiral 3-D | self-context | +0.02385 | -0.01355 |
| Ripple | self-context | +0.02138 | **+0.04603** |
| Polynomial drifted chirp | self-context | +0.01530 | +0.02312 |
| Periodic wells | ordinary | +0.01399 | +0.04187 |

Largest losses include self-context periodic wells (`-0.03293`), self-context
ring SDF (`-0.03276`), both multiscale models (about `-0.028`), and
self-context localized steps (`-0.02255`).

This falsifies the simple claim that momentum should never retain a restrained
component.  Sometimes retaining the gambit is valuable even after a live
turn.  What survives is a narrower claim:

> A turn-conditioned contraction of stored departure energy can eliminate
> correction debt on a structured subset of problems, but turn energy alone
> is not a sufficient confidence estimate for every moving representation.

The next refinement should learn or estimate the confidence of the restraint,
not increase its unconditional strength.

## 7. Muon versus Transport-Lepton

The grand means are nearly identical:

```text
Muon AUC:                0.729605
Transport-Lepton AUC:   0.729581
difference:             -0.000024
```

Yet Transport-Lepton beats Muon on 88 of 144 paired-seed AUC comparisons, with
a median delta of `+0.000867`.  Its typical change is beneficial, but its
negative tail is larger.  Muon finishes with better mean validation,
held-out, and tail-defined scores.

The architecture split clarifies the balance:

- ordinary: Transport-Lepton leads Muon AUC by `0.00123`;
- self-context: Muon leads by `0.00127`;
- self-context held-out: Muon leads by `0.01848`.

Transport-Lepton therefore remains evidence that transport helps the smooth
Bregman state on many paired trajectories.  It is not yet a robust six-arm
winner because AdamW or an Anchor arm is usually still ahead on the same
problem.

## 8. Exact and practical Muon on the special problem

The anisotropic operator problem is

```text
F(W) = 1/2 tr((W-Q) Sigma (W-Q)^T),               (6)
```

with dense orthogonal `Q`, independently rotated `Sigma`, dimension 32, and
condition number `10^4`.

### Exact primitive

At `W_0=0`,

```text
G_0 = -Q Sigma,
polar(G_0) = -Q.                                  (7)
```

Exact Muon at unit operator step reaches `Q` in one population update.  Under
four mutually orthogonal rank-8 covariance blocks, supported exact polar
descent recovers one complete block per update and reaches `Q` on update four.
This remains the algebraic Muon-primacy theorem.

### Practical standing configuration

The six practical arms were run for 500 updates on three independently rotated
instances.  Under the same standing LRs as the neural battery:

| Optimizer | Population final loss | Population operator error | Block final loss | Block operator error |
|---|---:|---:|---:|---:|
| **Muon** | **9.66e-5** | **0.6071** | **7.33e-5** | **0.5102** |
| Anchor | 8.19e-4 | 0.9526 | 2.64e-3 | 0.9796 |
| Restrained-momentum Anchor | 8.19e-4 | 0.9526 | 2.68e-3 | 0.9788 |
| Transport-Lepton | 1.50e-3 | 0.9176 | 3.67e-3 | 0.9533 |
| AdamW | 2.20e-3 | 1.2858 | 6.51e-3 | 1.1614 |
| SGD | 2.87e-2 | 0.9985 | 1.15e-1 | 0.9996 |

Practical Muon is decisively best in terminal objective under the frozen
standing protocol and is the only arm to reach `10^-4` block loss, at median
update 476.  However, no practical arm reaches 1% operator error.  A small
covariance-weighted objective can still hide badly recovered quiet modes.

### Why unit practical Muon does not terminate

The operator-native control sets practical Muon and Transport-Lepton to unit
matrix LR, SGD to its declared stable `1.9`, and AdamW to `0.01`.  Practical
Muon does not reproduce equation (7): its five Newton--Schulz iterations are
only an approximate polar map, especially on the weakest singular modes, and
its `0.95` momentum continues proposing a nonvanishing normalized matrix step
after the target is approached.

At 500 updates, unit-LR practical Muon has mean relative loss `0.388` on the
population problem and `3.89` on the block stream.  Unit-LR
Transport-Lepton also fails to settle.  These are finite oscillatory or
divergent trajectories rather than NaN/Inf failures; unit scale is not a
stable practical configuration for either carried matrix state.

The special problem has therefore done more than display a favorable Muon
curve.  It separates three objects that must not be conflated:

1. exact polar steepest descent;
2. a finite Newton--Schulz approximation;
3. a momentum optimizer that repeatedly applies the approximate normalized
   direction.

Muon's primacy belongs exactly to the first object.  Making the practical
optimizer inherit it requires braking or state restraint near completion.

## 9. Selected iteration readings

Absolute `0.80` and `0.90` validation thresholds are meaningful only on tasks
that reach them.  Selected medians across three seeds:

| Task/model | Optimizer | to 0.80 | to 0.90 |
|---|---|---:|---:|
| Pinwheel ordinary | AdamW | 15 | 95 |
|  | Anchor | 15 | **55** |
|  | Restrained-momentum Anchor | 15 | **55** |
|  | Muon | 35 | 210 |
|  | Transport-Lepton | 35 | 225 |
| Localized steps self-context | AdamW | 35 | 50 |
|  | Anchor | 30 | 50 |
|  | Restrained-momentum Anchor | **25** | 50 |
|  | Muon | 105 | 130 |
|  | Transport-Lepton | 90 | 115 |
| Ripple self-context | AdamW | **275** | **335** |
|  | Anchor | 432.5 | not reached |
|  | Restrained-momentum Anchor | **397.5** | not reached |
|  | Muon | 420 | 455 |

These readings support the AUC interpretation: restraint can save iterations
on specific tasks, but AdamW remains the stronger broad default.

## 10. Conclusions

1. **AdamW remains the aggregate winner.**  None of the new geometry-aware
   methods displaces it on broad acquisition efficiency at these standing
   configurations.
2. **Restraining momentum is worth keeping.**  It creates eight repeatable
   task-level wins and a positive median paired effect against Anchor.  Its
   confidence rule must be improved because the negative tail erases the grand
   mean gain.
3. **Muon is the best continuation method in the aggregate.**  Its slight
   held-out lead over AdamW and strong operator-witness terminal loss justify
   further work even though its AUC is lower.
4. **Transport-Lepton gives frequent small acquisition gains over Muon, but
   not robust battery wins.**  Transport is active; its tail behavior remains
   insufficiently restrained.
5. **The exact Muon theorem survives, and practical Muon exposes the next
   problem.**  Polar geometry supplies the right matrix decision; it does not
   supply automatic settling for approximate, momentum-driven updates.

The next experiment should put a turn- and residual-aware restraint directly
inside Muon/Lepton matrix momentum, then test it first on continuously rotating
operator blocks.  That problem interpolates cleanly between exact polar
completion and the overlapping stochastic frames where transport is needed.

## Reproduction and artifacts

Raw and derived evidence:

- `results_optimizer_shape_battery/results.json`: all 864 neural fits;
- `results_optimizer_shape_battery/operator_results.json`: 72 special-problem
  runs and exact certificates;
- `results_optimizer_shape_battery/summary.json`: paired analysis;
- `results_optimizer_shape_battery/summary.png`: compact six-panel figure;
- `results_optimizer_shape_battery/atlas.html`: every validation/loss curve;
- `run_optimizer_operator_battery.py`: operator assay;
- `analyze_optimizer_shape_battery.py`: frozen paired analysis;
- `build_optimizer_shape_atlas.py`: standalone all-problem viewer.

Neural run:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_benchmark \
  --out /tmp/optimizer_shape_battery \
  --variants ordinary_mlp,self_context --widths 24 --seeds 3 \
  --steps 500 --batch 256 --eval-every 5 \
  --optimizers adamw,sgd,anchor,anchor_restrained_momentum,muon,lepton_transport \
  --optimizer-lrs adamw=0.003,sgd=0.03,anchor=1.0,anchor_restrained_momentum=1.0,muon=0.003,lepton_transport=0.003
```

Operator run:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_optimizer_operator_battery \
  --out /tmp/optimizer_operator_battery.json \
  --dimension 32 --condition 10000 --steps 500 \
  --batch-size 8 --seeds 3 --seed-base 731
```
