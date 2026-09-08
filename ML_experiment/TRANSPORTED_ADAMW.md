# Transported AdamW

## Question

Ordinary AdamW stores an elementwise second moment. For a matrix row gradient
\(g\in\mathbb R^d\), its state is

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t).
\]

The diagonal operation is tied to the current coordinate axes. Under a right
orthogonal change of basis \(g\mapsto gQ\), in general

\[
(gQ)\odot(gQ)\ne(g\odot g)Q.
\]

AdamW therefore cannot carry the same second-moment object through a rotating
gradient frame. This matters in a self-context network because the model
changes the chart used to produce later gradients while the optimizer retains
history measured in earlier charts.

The experiment asks whether AdamW can retain a non-elementwise second moment
and explicitly transport both moments when the live row-gradient frame turns,
without becoming Muon, Shampoo, or a task-conditioned optimizer.

## State and transport

Each matrix row is one optimizer cell. A vector parameter is one cell. For a
live cell, define the gradient signature

\[
s_t=\frac{g_t}{\|g_t\|}.
\]

Consecutive signatures determine the unique minimum rotation \(R_t\) on the
non-antipodal plane that carries \(s_{t-1}\) to \(s_t\) and is the identity on
their common orthogonal complement. Before incorporating the new gradient, the
stored moments are transported covariantly:

\[
m_{t-1}^{\rightarrow}=m_{t-1}R_t,
\qquad
V_{t-1}^{\rightarrow}=R_t^T V_{t-1}R_t.
\]

For an antipodal turn the shortest rotation is not unique, so the cell state is
reset rather than inventing an orientation. A row whose earlier gradients were
zero has no prior frame; its first live signature initializes the frame without
transport.

## Non-elementwise Adam recurrence

The transported state enters an ordinary exponential recurrence, except the
second moment is a full covariance inside the cell:

\[
m_t=\beta_1m_{t-1}^{\rightarrow}+(1-\beta_1)g_t,
\]

\[
V_t=\beta_2V_{t-1}^{\rightarrow}+(1-\beta_2)g_t^Tg_t.
\]

After bias correction, the descent request is

\[
u_t=(\sqrt{\widehat V_t}+\epsilon I)^{-1}\widehat m_t.
\]

The implementation evaluates the symmetric inverse square root with an
eigendecomposition. Eigenmodes below the numerical support threshold are
removed rather than amplified through \(\epsilon^{-1}\). No row interacts with
another row.

On the first deterministic step, \(V=gg^T\), so the supported request has row
norm one. AdamW's elementwise sign request has per-coordinate RMS one and row
norm \(\sqrt d\). Transported AdamW multiplies by \(\sqrt d\) solely to retain
AdamW's declared learning-rate convention:

\[
\|u_{1,\mathrm{transported}}\|=\sqrt d.
\]

This factor depends only on cell dimension, not gradients, loss, task identity,
or observed performance.

## Covariance property

For any shared right orthogonal basis change \(Q\), the state transforms as

\[
g_t' = g_tQ,
\qquad
m_t'=m_tQ,
\qquad
V_t'=Q^TV_tQ.
\]

Functional calculus gives

\[
(\sqrt{V_t'}+\epsilon I)^{-1}
=Q^T(\sqrt{V_t}+\epsilon I)^{-1}Q,
\]

and therefore

\[
u_t'=u_tQ.
\]

The optimizer is right-orthogonally equivariant within every row cell. The
structural test applies the same sequence of randomly rotated gradients to two
optimizers and verifies that every resulting parameter matrix differs only by
the declared rotation.

## Why this is not Muon

Muon applies a matrix polar map to a complete two-dimensional gradient. Its
left and right singular directions couple rows and columns, producing a
semi-orthogonal request. Transported AdamW never couples rows. Its first step
normalizes each row independently, and its later second moment estimates a
within-row covariance. A structural test confirms that its first request is
not the Muon polar factor.

The exact second-moment oracle explains the boundary. A right-factor covariance
for the whole matrix yields a polar request on the first supported step. A
row-local full covariance does not: it retains off-diagonal geometry inside a
row while refusing the cross-row geometry that defines Muon.

## Standing Population witness

The Standing Population problem exposes one dense rotated linear operator with
condition number \(10^4\). Three seeds, dimension `32`, and `500` steps are
used. The learning rates are fixed before observing results.

| Optimizer | Final relative objective | Final operator error | Steps to objective \(10^{-2}\) |
|---|---:|---:|---:|
| AdamW | 0.002205 | 1.2858 | 185 |
| Transported AdamW | 0.001979 | 0.9826 | 166 |
| Anchor | 0.000819 | 0.9526 | 73 |
| Transport-Lepton | 0.001500 | 0.9176 | 370 |
| Muon | 0.000097 | 0.6071 | 368 |

Transported AdamW improves both AdamW's objective and hidden-operator recovery.
It does not beat Muon on the challenge constructed to expose Muon's matrix
polar geometry. That negative boundary is important: row covariance preserves
more operator structure than an elementwise denominator, but it is not a
substitute for full matrix coupling.

## Interaction with source-aware self-context backpropagation

The model-side cure exposes, for each parameter, a frozen-chart gradient and a
restrained chart-motion gradient:

\[
g_{\mathrm{step},p}=g_{\mathrm{fixed},p}
+\alpha_p g_{\mathrm{chart},p}.
\]

Transported AdamW currently receives their certified sum. Both source channels
are attached to the parameter before `step()`, so a later experiment can
transport separate fixed-chart and chart-motion moments. The current battery
does not claim that decomposition: it tests whether replacing elementwise
history with transported row covariance is already useful under the same
source-aware backpropagation supplied to every optimizer.

## Cost and limits

For row width \(d\), second-moment storage is \(O(d^2)\) per row and the
eigendecomposition is \(O(d^3)\) per step. At width `24` the named neural screen
takes roughly twice AdamW's wall time. This is a research implementation, not a
claim of production efficiency.

The optimizer is geometry-universal but not geometry-complete. It assumes the
chosen parameter row is a meaningful cell, does not model covariance between
rows, and can remain seed-sensitive on continuation outside the validation
support. Those limits are displayed in the atlas rather than hidden by an
optimizer-selection rule.

## Reproducibility

- Implementation: `ML_experiment/optimizers.py` (`TransportedAdamW`)
- Structural tests: `ML_experiment/test_transported_adamw.py`
- Second-moment oracle: `ML_experiment/second_moment_oracle.py`
- Oracle evidence: `ML_experiment/results_second_moment_oracle.json`
- Operator screen: `ML_experiment/results_transported_adam_operator_rms.json`
