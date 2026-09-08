# AdamW versus source-aware AdamW on Ripple

## Question

Can AdamW handle a self-context layer more cleanly if the optimizer preserves
the two causal gradient sources exposed by source-aware backpropagation?

The forward computation is identical in both arms. Ordinary AdamW receives
the exact live self-context gradient. The experimental arm receives

\[
g_{\mathrm{live}} = g_f + g_c,
\]

where \(g_f\) is the cotangent with the self-context chart held fixed and
\(g_c\) is the additional cotangent caused by moving that chart.

## Experimental rule

The experimental AdamW makes \(g_f\) the owner of persistent optimizer state:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_f,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_f^2.
\]

The current chart feedback is read through the same diagonal metric but does
not write either recurrence:

\[
u_f=\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
\qquad
u_c=\frac{g_c}{\sqrt{\hat v_t}+\epsilon}.
\]

The chart request is capped as one parameter-tensor object after
preconditioning,

\[
\alpha=\min\left(1,\frac{\lVert u_f\rVert}{\lVert u_c\rVert}\right),
\]

and the combined request is minimally projected into the half-space

\[
\langle u,g_f\rangle\geq 0.
\]

This prevents momentum plus chart feedback from reversing the current
fixed-chart decision. Decoupled weight decay is unchanged. Global gradient
clipping applies the same coefficient to the saved source channels, so the
experimental optimizer cannot bypass the shared norm-5 bound.

## Paired protocol

- Task: Ripple regression.
- Model: self-context M-layer, width 24.
- Five paired seeds.
- 750 steps, batch 256, evaluation every five steps.
- AdamW learning rate 0.003, weight decay \(10^{-4}\), default PyTorch betas.
- Identical initialization and minibatch sequence within every seed.
- No learning-rate search, validation-conditioned update, selected-seed
  exclusion, or task-specific optimizer branch.
- Executed on the nearby M4 Mac mini.

All ten runs completed without numerical failure.

## Result

| Measure | AdamW | Source-aware AdamW |
|---|---:|---:|
| Mean score-AUC | **0.7713** | 0.7456 |
| Mean best validation score | **0.9536** | 0.9302 |
| Median steps to 0.80 | **345** | 385 |
| Median steps to 0.90 | 525 | **515** |
| Seeds reaching 0.95 | **4 / 5** | 1 / 5 |
| Mean final score | **0.9450** | 0.9285 |

Ordinary AdamW wins score-AUC on four of five seeds. The experimental arm has
a small median advantage to 0.90, but that isolated crossing does not survive
the integrated trajectory or high-accuracy threshold. It learns stably and
does not explode; it simply learns less quickly overall and stops at a worse
floor within the budget.

## Interpretation

The source separation is not inert. Late in training, the raw chart gradient
is typically only a few percent of the fixed-chart gradient, while its request
after Adam's diagonal metric can be a much larger fraction of the fixed
request. The post-metric restraint and occasional anti-reversal projection
therefore act on a real amplification mechanism.

The negative result identifies the overreach: chart motion is not merely
coordinate noise. On Ripple it contains repeated, useful information. Ordinary
AdamW mixes the two sources before its moment recurrences:

\[
(g_f+g_c)^2=g_f^2+2g_fg_c+g_c^2.
\]

The cross term is an implicit, elementwise source covariance. The experimental
rule avoids its pathologies by deleting it entirely, but also deletes its
signal. Source awareness is therefore useful as an interface, not yet as this
temporal policy.

The next principled candidate is a non-elementwise two-source metric. For each
parameter tensor, retain a small Gram state

\[
C_t=\beta C_{t-1}+(1-\beta)
\begin{bmatrix}
\langle g_f,g_f\rangle & \langle g_f,g_c\rangle\\
\langle g_c,g_f\rangle & \langle g_c,g_c\rangle
\end{bmatrix}.
\]

This preserves source identity and estimates whether chart motion is
repeatedly compatible with the fixed decision without granting every scalar
coordinate an independent sign-normalized authority. It directly occupies the
missing ground between AdamW's indiscriminate elementwise mixture and the
current experiment's complete denial of chart memory.

## Artifacts

- `adamw_self_context_ripple.html`: complete loss curves, validation curves,
  diagnostics, fitted surfaces, error surfaces, and seed table.
- `adamw_self_context_ripple.png`: compact static loss/score/surface figure.
- `results_adamw_self_context_ripple/results.json`: complete histories and
  fitted grids.
- `run_adamw_self_context_ripple.py`: paired runner.
- `test_source_aware_adamw.py`: optimizer invariants.
