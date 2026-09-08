# Euclidean Matrix Transport

## Result first

Euclidean Matrix Transport passes the proposed Ripple gate against AdamW.

Five paired seeds, identical self-context models, initial parameters,
minibatches, gradient clipping, declared learning rate, and weight decay give:

| Measure | AdamW | Matrix Transport | AdamW at equal 4× ceiling |
|---|---:|---:|---:|
| Mean score-AUC | 0.77133 | **0.78532** | 0.72914 |
| Paired AUC wins | 2 / 5 | **3 / 5** | — |
| Median steps to 0.80 | 345 | **285** | 370 |
| Median steps to 0.90 | 525 | **430** | 510 |
| Mean best score | 0.95357 | **0.96378** | 0.94314 |
| Mean M4 time | **4.87 s** | 6.20 s | 4.87 s |

All fifteen final/control runs completed without numerical failure. Matrix
Transport wins mean learning trajectory, median threshold time, and mean best
fit. Cached matrix roots reduce its Ripple time by approximately 7% while
slightly improving the observed AUC. It remains approximately 27% slower than
AdamW per wall-clock Ripple run in this small CPU implementation.

This is evidence for an optimizer, not yet evidence for a universal optimizer.
Ripple influenced the sequence of rejected constructions, so it was excluded
from the frozen unseen battery described in Section 7.

## 1. What transport can mean in Euclidean descent

The parameter space is flat. Its Levi-Civita parallel transport is the
identity, so merely rotating a vector because gradients changed is not
intrinsic Euclidean transport. A nontrivial rotation requires a second object:
an observed moving frame.

For each parameter tensor, the optimizer constructs that frame from the
current gradient after a coordinate-free metric action. The identity remains
the prior connection. A measured correspondence between consecutive frames
may earn a fraction of the minimum rotation; it never replaces Euclidean
identity automatically.

That distinction was decisive in the ablation sequence.

## 2. Non-elementwise second moment

For a matrix parameter with gradient \(G_t\in\mathbb R^{r\times c}\), maintain
left and right covariance factors:

\[
L_t=\beta_2L_{t-1}+(1-\beta_2)\frac{G_tG_t^\top}{c},
\qquad
R_t=\beta_2R_{t-1}+(1-\beta_2)\frac{G_t^\top G_t}{r}.
\]

The live metric request is

\[
P_t=\widehat L_t^{-1/4}G_t\widehat R_t^{-1/4}.
\]

This is the inverse square-root action of the Kronecker metric
\(R_t\otimes L_t\). Under independent orthogonal coordinate changes

\[
G\mapsto U^\top GV,
\]

the request transforms as

\[
P\mapsto U^\top PV.
\]

The optimizer therefore has no privileged scalar coordinate system. Vectors
and scalars use their unique tensorwise isotropic RMS. At the first full-rank
square-matrix step, \(P_t\) has the same RMS convention as Adam's sign step,
but retains a polar matrix shape instead of becoming an elementwise sign map.

## 3. Confidence-weighted frame transport

Flatten and normalize the live request only to define its frame:

\[
h_t=\frac{\operatorname{vec}(P_t)}{\lVert P_t\rVert},
\qquad
q_t=\langle h_{t-1},h_t\rangle.
\]

Let \(\mathcal T_t\) be the minimum spherical rotation taking \(h_{t-1}\) to
\(h_t\). Full transport is not trusted blindly. Its confidence is

\[
\rho_t=[q_t]_+.
\]

First transport the old momentum and split it into its live axial and
transverse parts:

\[
\mathcal T_t m_{t-1}=a_th_t+z_t,
\qquad z_t\perp h_t.
\]

The transverse component is uncertain from only two frame observations, so it
is restrained by the same confidence. The connection read is

\[
\widetilde m_{t-1}
=(1-\rho_t)m_{t-1}
+\rho_t(a_th_t+\rho_tz_t).
\]

When frames agree, the old request is transported. When they are orthogonal or
contradictory, the rule returns continuously to ordinary Euclidean identity
transport. This is the local deformation: observed geometry must earn the
right to rotate memory.

The momentum recurrence is then

\[
m_t=\beta_1\widetilde m_{t-1}+(1-\beta_1)P_t.
\]

Any final request with negative raw-gradient inner product is minimally
projected onto the Euclidean descent half-space

\[
\langle u_t,G_t\rangle\ge 0.
\]

Momentum may propose a gambit transverse to current descent; it cannot reverse
the live decision.

## 4. Coherence-controlled scalar step

The signed frame coherence controls one scalar gain:

\[
\log s_{t+1}
=\Pi_{[\log .25,\log 4]}
\left(\log s_t+.02\,\overline q_t\right).
\]

The applied learning rate is \(\eta s_t\). This is not an elementwise adaptive
rate and does not inspect the loss or validation set. Repeated directional
agreement grows the step; contradiction shrinks it. All five final runs began
at \(s_0=1\) and ended near the declared ceiling of four.

The necessary falsification was AdamW run directly at \(4\eta=0.012\). Its
mean AUC fell to 0.72914. Matrix Transport's advantage is therefore not
explained by merely giving both methods the final scalar step from iteration
one. The observed coherence determines when the larger step becomes admissible.

## 5. Ablation path

| Construction | Seeds | Mean AUC | Finding |
|---|---:|---:|---|
| Row-scalar RMS + full transport | 5 | 0.65092 | Direction preservation without a shaped metric is insufficient. |
| Matrix metric + unconditional transport | 3 | 0.73914 | Correct metric; noisy frames are falsely treated as a connection. |
| Matrix metric + confidence transport | 3 | 0.75327 | Identity fallback restores useful momentum averaging. |
| Add signed coherence gain | 5 | **0.78443** | Beats AdamW's 0.77133 gate. |

The unconditional version recorded mean turn near 0.72, corresponding to a
mean consecutive cosine near −0.44. It was rotating old momentum to agree with
strongly contradictory minibatch evidence. That diagnostic motivated the
identity fallback; it was not a tuned numerical search.

## 6. Certified root reuse

The covariance factors are updated at every optimizer step, but their inverse
fourth roots need not be recomputed when the factors have barely moved. Let

\[
d_t=\max\left\{
\frac{\|\widehat L_t-\widehat L_\tau\|_F}{\|\widehat L_\tau\|_F},
\frac{\|\widehat R_t-\widehat R_\tau\|_F}{\|\widehat R_\tau\|_F}
\right\},
\]

where \(\tau\) is the last root refresh. The cached roots are reused while
\(d_t<0.05\), with an unconditional refresh after ten steps. This is a
task-blind numerical approximation to the same Kronecker metric: it reads only
relative covariance drift and has a hard error-horizon certificate.

On Ripple, only 23.8% of steps recompute roots, mean root age is 3.19 steps,
M4 time falls from 6.65 s to 6.20 s, and mean AUC changes from 0.78443 to
0.78532. Across the unseen battery, evaluation snapshots record a 25.2%
refresh fraction and mean age 3.23. The remaining 1.35× cost over AdamW comes
from the full matrix actions and transport state; optimizing eigendecomposition
alone now has diminishing end-to-end leverage.

## 7. Frozen unseen battery

After the cache was fixed, the optimizer and its learning rate were frozen.
It was tested against AdamW on 23 non-Ripple problems, three paired seeds each,
using the same width-24 self-context M-layer, exact self-context backward pass,
minibatch order, declared learning rate 0.003, and 500-update budget.

| Measure | Frozen result |
|---|---:|
| Numerical failures | 0 / 138 runs |
| Mean acquisition-AUC delta | **+0.00404** |
| Paired AUC wins | **42 / 69** |
| Task-mean AUC wins | **13 / 23** |
| Task-clustered AUC bootstrap interval | [−0.0007, +0.0092] |
| Mean held-out score delta | +0.00039 |
| Mean tail-score delta, applicable tasks | −0.00426 |
| Mean M4 runtime, AdamW → Matrix | 2.61 s → 3.53 s |

The mechanism therefore generalizes more clearly as an acquisition-speed
effect than as a generalization effect. Its strongest new result is the
high-rank 16-D spiral: mean held-out score rises from 0.7040 to 0.8898 and AUC
rises by 0.0282. Radial stripes and chirp continuation are clear counterexamples
to a universal win. The clustered interval crosses zero, so the 23-task result
is promising evidence rather than a population-level conclusion.

## 8. Invariants tested

Eight focused M4 tests establish:

1. first-step matrix RMS matches the declared Adam-scale convention without an
   elementwise sign map;
2. the matrix metric and complete optimizer step are equivariant to independent
   orthogonal row/column coordinate changes;
3. the matrix request cannot reverse the current raw gradient;
4. cached roots obey the declared hard maximum staleness;
5. the scalar-row predecessor has unit first-step row RMS without changing
   gradient shape;
6. that predecessor is rotation equivariant inside its blocks;
7. its request also cannot reverse the live gradient; and
8. its transport state remains finite at an antipodal frame transition.

## 9. Reproduction

```sh
python3 -m unittest \
  ML_experiment.test_euclidean_transport \
  ML_experiment.test_matrix_transport -v

python3 ML_experiment/run_benchmark.py \
  --out /tmp/matrix_transport_ripple \
  --tasks ripple --variants self_context --widths 24 \
  --seeds 5 --steps 750 --batch 256 --lr 0.003 --eval-every 5 \
  --optimizers adamw,matrix_transport \
  --context-backward-mode exact --context-gradient-mode blended
```

The implementation is `MatrixTransport` in `optimizers.py`. Complete histories
are in `results_matrix_transport_ripple_cached/results.json` and
`results_matrix_transport_unseen_battery/results.json`; the equal-ceiling
control is in `results_adamw_ripple_equal_ceiling/results.json`.
