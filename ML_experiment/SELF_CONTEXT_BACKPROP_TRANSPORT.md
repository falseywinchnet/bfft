# Self-context backpropagation and transport

## Result

The self-context contribution to the parameter gradient is separately
observable, and its normalization can amplify chart-rotating derivatives. The
forward-identical cure has two parts: make the context-normalization Jacobian
locally nonexpansive, then capture frozen-chart and chart-motion parameter
cotangents before the optimizer blends them. A tensor-level small-gain
certificate prevents chart motion from reversing the fixed-chart decision to
first order without elementwise clipping.

This cure improves Anchor on the paired Ripple and high-rank N-D spiral screen,
but the complete battery does not crown a transported optimizer. Across 24
problems and 288 runs, AdamW is the robust learning-speed and mean held-out
winner. Transported AdamW has a real continuation advantage on the high-rank
spiral and improves AdamW on the Standing Population operator witness, but it
is slower and less universal.

The two mechanisms therefore occur at different locations:

1. self-context normalization can amplify the context-mediated cotangent
   during backpropagation;
2. Transport-Lepton can subsequently discard the magnitude of the resulting
   live parameter gradient and apply a non-vanishing spectral request.

Both require restraint. Neither task detection nor validation feedback is
needed to identify them.

## Exact gradient decomposition

For one self-context layer, write

\[
z_\theta(x)=x+\gamma N(c_\theta(x)),
\qquad
y_\theta(x)=f_\theta(z_\theta(x)).
\]

The parameter gradient splits into

\[
\nabla_\theta L
=
\underbrace{\left.\partial_\theta L(f_\theta(z))\right|_{z=z_\theta(x)}}_
{g_{\mathrm{direct}}}
+
\underbrace{
\gamma (D_\theta c_\theta(x))^T
(DN(c_\theta(x)))^T
(D_z f_\theta(z))^T\nabla_y L
}_{g_{\mathrm{context}}}.
\]

The forward function is evaluated identically in every control. The direct
gradient is obtained by returning the same normalized context value while
zeroing only its incoming cotangent. Consequently

\[
g_{\mathrm{context}}
=g_{\mathrm{exact}}-g_{\mathrm{detached}}
\]

without changing data, predictions, loss values, parameters, or task
geometry.

## Why normalization matters

The implementation normalizes raw context to the authentic activation RMS:

\[
N(c)=r\frac{c}{\operatorname{rms}(c)}.
\]

Ignoring the numerical floor, its Jacobian is

\[
DN(c)=
\frac{r}{\operatorname{rms}(c)}
\left(I-\hat c\hat c^T\right).
\]

Radial changes are removed, while tangential changes that rotate the selected
chart are multiplied by

\[
\frac{r}{\operatorname{rms}(c)}.
\]

On the measured high-rank spiral trajectories, the raw context was commonly
only `4%` to `8%` of the activation RMS. The normalization can therefore give
the chart-rotating backward channel approximately `12x` to `25x` local gain.

This matters especially to a direction-memory optimizer: chart feedback can
rotate while the fitting residual shrinks, yet normalization presents that
rotation as a full-scale directional event.

## Nonexpansive backward certificate

The derived restraint leaves the forward proposal unchanged and scales only
the context-mediated cotangent by

\[
\alpha(c,r)=
\min\left(1,\frac{\operatorname{rms}(c)}{r}\right).
\]

Therefore

\[
\|\alpha DN(c)\|_2\le 1.
\]

This is not a fitted threshold. One is the boundary between an expansive and
nonexpansive local map. The rule does not inspect the task name, target,
validation set, training loss, optimizer identity, or iteration number.

## Measured decomposition

The table reports the complete-model context-feedback norm relative to the
direct-gradient norm on the fixed high-rank spiral probe batch.

| Optimizer | Seed | Step 200 | Step 300 | Step 500 |
|---|---:|---:|---:|---:|
| AdamW | 0 | 0.43 | 0.38 | 0.39 |
| AdamW | 1 | 0.35 | 0.35 | 0.39 |
| Transport-Lepton | 0 | 0.40 | 2.18 | 2.02 |
| Transport-Lepton | 1 | 0.79 | 0.34 | 0.41 |
| Guarded Transport-Lepton | 0 | 0.34 | 3.65 | 2.63 |
| Guarded Transport-Lepton | 1 | 0.82 | 0.35 | 0.28 |

In the failing seed, the self-context feedback becomes two to four times the
direct term. At step 500 its cosine with the direct term is `0.72` for
unguarded transport and `0.64` for guarded transport. This is not merely a
large cancelling residual; it is a coherent chart-mediated request capable of
dominating the direct fitting gradient.

## Training control

All runs use the same self-context forward function, width `24`, learning rate
`0.003`, `500` steps, and two seeds.

| Optimizer | Backpropagation | Seed 0 final probe loss | Seed 1 final probe loss | Two-seed mean |
|---|---|---:|---:|---:|
| AdamW | exact | 2.060e-5 | 3.215e-5 | 2.638e-5 |
| AdamW | nonexpansive context | 2.907e-5 | 3.342e-5 | 3.125e-5 |
| Transport-Lepton | exact | 6.985e-10 | 0 | — |
| Transport-Lepton | nonexpansive context | 6.985e-9 | 0 | — |
| Guarded Transport-Lepton | exact | 1.028e-6 | 9.311e-7 | 9.798e-7 |
| Guarded Transport-Lepton | nonexpansive context | 6.657e-7 | 7.278e-7 | 6.967e-7 |

The nonexpansive channel improves the guarded transport mean floor by about
`28.9%`. In the violent unguarded seed, the largest observed gradient norm
drops from `97.5` to `8.19`.

The unguarded result must not be called stable merely because its final probe
was sampled at a floor. Its pre-update batch loss and gradient still jump, and
its update-to-gradient ratio still reaches approximately `3.6e4`. The
optimizer can leave the floor again because its spectral request does not
vanish with the live gradient.

## Capturing chart motion before the optimizer blends it

The local nonexpansive rule identifies the problematic autograd edge, but an
ordinary optimizer receives only the sum in `parameter.grad`. Source identity
is then irrecoverable. The training path now evaluates the same deterministic
forward value twice:

\[
g_{\mathrm{fixed}}
=\operatorname{backward}_{\mathrm{detached\ chart}} L,
\qquad
g_{\mathrm{live}}
=\operatorname{backward}_{\mathrm{nonexpansive\ chart}} L,
\]

and exposes

\[
g_{\mathrm{chart}}=g_{\mathrm{live}}-g_{\mathrm{fixed}}
\]

on every parameter before the optimizer step. `g_fixed` is the derivative of
the loss with respect to the parameters while holding the self-generated chart
value fixed. `g_chart` is the additional cotangent caused by allowing the chart
itself to move. This is a source decomposition, not a second objective or a
change to the data.

A model-wide norm certificate was tested first and rejected. Across the named
screen, the full-model ratio

\[
\|g_{\mathrm{chart}}\|/\|g_{\mathrm{fixed}}\|
\]

never exceeded `0.12`, so its scale remained exactly one. Large output-head and
direct gradients concealed the affected self-context tensors. Score changes in
that control were floating-point trajectory divergence, not evidence that the
global certificate had acted.

The operative certificate is therefore applied once per parameter tensor:

\[
\alpha_p=
\begin{cases}
\min\left(1,
\dfrac{\|g_{\mathrm{fixed},p}\|}
{\|g_{\mathrm{chart},p}\|}\right),
&\|g_{\mathrm{fixed},p}\|>0,\\
1,&\|g_{\mathrm{fixed},p}\|=0,
\end{cases}
\]

\[
g_{\mathrm{step},p}
=g_{\mathrm{fixed},p}+\alpha_p g_{\mathrm{chart},p}.
\]

The scale is shared by every coordinate of a tensor. It preserves the complete
non-elementwise shape of the chart feedback and is equivariant to orthogonal
changes of coordinates within that tensor. A parameter used only to construct
the chart remains learnable. For every shared tensor, Cauchy--Schwarz gives

\[
\langle g_{\mathrm{step},p},g_{\mathrm{fixed},p}\rangle
\ge
\|g_{\mathrm{fixed},p}\|^2
-\|g_{\mathrm{chart},p}^{\mathrm{applied}}\|
 \|g_{\mathrm{fixed},p}\|
\ge 0.
\]

Thus chart motion may bend or cancel the fixed-chart decision, but cannot make
that tensor ascend the fixed-chart loss to first order.

The tensor certificate is active even when the global ratio is small. On the
paired Ripple and high-rank N-D spiral screen, the minimum live tensor scale
ranged from `0.040` to `0.238`, while the mean across tensors remained about
`0.95` to `0.97`. The intervention is sparse in parameter groups rather than
ubiquitous.

For Anchor, held-out Ripple score improved from a two-seed mean of `0.770` to
`0.813`; high-rank N-D spiral held-out score improved from `0.489` to `0.518`.
Transport-Lepton was numerically unchanged on the same pairs. Transported
AdamW remained seed-sensitive on spiral continuation, showing that a correct
backpropagation certificate does not by itself solve every optimizer's memory
geometry.

## Complete 24-problem battery

The final comparison uses only the self-context architecture, width `24`,
`500` steps, batch size `256`, three paired seeds, and the same tensor-level
source-aware backpropagation for every optimizer. Learning rates are fixed by
optimizer across the entire battery. All `288` runs completed without a
non-finite loss, gradient, parameter, or evaluation.

| Optimizer | Mean learning AUC | AUC wins | Mean held-out score | Held-out wins including ties | Seconds per run |
|---|---:|---:|---:|---:|---:|
| Anchor | 0.7679 | 2 | 0.6110 | 5 | 7.25 |
| Transport-Lepton | 0.7738 | 0 | 0.6364 | 8 | 6.11 |
| AdamW | **0.8275** | **20** | **0.6565** | **13** | **5.69** |
| Transported AdamW | 0.7762 | 2 | 0.6397 | 4 | 10.97 |

Plain AdamW is the robust battery winner. The experiment does not support
replacing it with Anchor, Transport-Lepton, or Transported AdamW as a universal
self-context optimizer. Transported AdamW retains the second-best task-mean
held-out score, but takes approximately `1.93x` AdamW's wall time and wins far
fewer learning curves.

The aggregate does not erase geometric exceptions. On the high-rank N-D spiral
all four methods achieve validation score `1.0`, while three-seed held-out
scores are:

| Optimizer | Learning AUC | Held-out | Tail |
|---|---:|---:|---:|
| Anchor | 0.8324 | 0.4660 | 0.4568 |
| Transport-Lepton | 0.8541 | 0.6205 | 0.5702 |
| AdamW | **0.9223** | 0.7303 | 0.6318 |
| Transported AdamW | 0.9198 | **0.8258** | **0.7660** |

Transported AdamW does not learn the validation problem faster than AdamW, but
it preserves substantially more continuation geometry after validation has
saturated. Conversely, many one-dimensional continuation and ordinary fitting
problems favor AdamW. The atlas exposes both facts with paired loss,
validation, held-out, tail, and fitted-function views.

## Interpretation

The proposed backpropagation restraint is supported as a mechanism:

- it isolates the precise self-context feedback term;
- its bound follows from the normalization Jacobian;
- it preserves the forward function exactly;
- it materially improves transport once the optimizer also has a live-gradient
  certificate.

It is not sufficient by itself. Stable self-context transport needs a
two-level contract:

\[
\text{context backward gain}\le 1,
\qquad
\|\Delta\theta_t\|\le \rho\|g_t\|.
\]

The first prevents the model's moving chart from expanding a tangential
cotangent. The second prevents the optimizer's transported memory from
overriding convergence after the live gradient has vanished.

## Reproducibility

- Implementation: `ML_experiment/models.py`
- Decomposition runner: `ML_experiment/run_context_gradient_decomposition.py`
- Structural tests: `ML_experiment/test_context_backward_restraint.py`
- Source-aware capture: `ML_experiment/context_backprop.py`
- Source-aware tests: `ML_experiment/test_context_backprop_split.py`
- Tensor-certificate screen: `ML_experiment/results_cured_four_arm_tensor/results.json`
- Full battery: `ML_experiment/results_cured_four_arm_full/results.json`
- Atlas: `ML_experiment/cured_optimizer_atlas.html`
- Gradient decomposition: `ML_experiment/results_context_gradient_high_rank.json`
- Backpropagation training screen: `ML_experiment/results_context_backward_screen.json`
