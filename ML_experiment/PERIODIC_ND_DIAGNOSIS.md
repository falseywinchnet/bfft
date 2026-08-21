# Periodic N-D diagnosis: a commuting-curvature problem

## Result

The original self-context family is not failing because it lacks parameters or
because the target is inherently difficult to represent. It is using the wrong
differential organization.

The task is

\[
f(x)=\frac{1}{d}\sum_{i=1}^{d}
\left[\cos(k_i x_i)+0.3\sin(2x_i)\right],
\quad k=(1,2,3,4,5,1,2,3).
\]

Consequently, every Hessian is diagonal in the same frame:

\[
H_f(x)=\operatorname{diag}(h_1(x_1),\ldots,h_d(x_d)),
\qquad [H_f(x),H_f(x')]=0.
\]

The target is high-rank, but its curvature directions do not move. It is a
flat collection of independent scalar charts, not a transported manifold.

## What the existing methods learn

Partial-dependence probes expose a sharp spectral cutoff. At 500 steps:

- self-context and learned cone recover the frequency-1 axes;
- they recover much of the frequency-2 axes;
- their R² is approximately zero on every frequency-3, frequency-4, and
  frequency-5 coordinate;
- increasing training to 2,000 steps barely changes self-context or the cone.

The ordinary 26k-parameter MLP eventually acquires parts of frequency 3, but
still leaves frequencies 4 and 5 nearly flat. Its mean held-out R² at 2,000
steps is 0.592. Self-context remains at 0.327.

The differential measurements explain the hard plateau:

| Model | Mean R², 2k | Mixed-Hessian ratio | Hessian commutator ratio |
|---|---:|---:|---:|
| Self-context | 0.327 | 0.816 | 0.603 |
| Learned cone | 0.299 | 0.659 | 0.616 |
| Direct operator sphere | 0.293 | 0.706 | 0.570 |
| Dense 26k MLP | 0.592 | 0.089 | 0.190 |
| Learned orthogonal scalar charts | 0.827 | 0.024 | 0.000 |
| Identity-preserving scalar charts | **0.994** | **0.006** | **0.000** |

Self-context and CFF continually reinterpret the observation through a changing
local allocation. That flexibility was beneficial on radial transport, but on
this task it creates cross-coordinate curvature that does not exist in the
truth. The higher harmonics are weak independent signals, each contributing
only a fraction of the total variance, and their gradients are absorbed into a
moving mixed frame.

## The successful intervention

The scalar chart bank contains no periodic nonlinearity and no Fourier
features. For frame rays \(q_r\), it computes

\[
z_r=q_r^T x,
\qquad
g_r(z_r)=a_r^T\operatorname{LELU}
\left(M_r\operatorname{LELU}(s_r z_r+b_r)+c_r\right),
\qquad
\hat y=\sum_r g_r(z_r).
\]

This is an ordinary collection of trainable one-dimensional LELU functions.
Its useful structural restriction is that nonlinear responses remain attached
to scalar rays instead of being mixed before their curvature is learned.

| Model | Parameters | R², 500 | Mean R², 2k | CPU time, 2k |
|---|---:|---:|---:|---:|
| Self-context | 9,283 | 0.310 | 0.327 | 15.24 s |
| Learned cone | 26,615 | 0.343 | 0.299 | 16.43 s |
| Direct operator sphere | 26,618 | 0.319 | 0.293 | 38.26 s |
| Axis chart, 16 units | **2,561** | 0.782 | 0.977 | 1.15 s |
| Axis chart, 32 units | 9,217 | **0.858** | 0.990 | 1.70 s |
| Identity-preserving learned frame | 5,441 | 0.784 | **0.994** | 1.97 s |
| Self-context + commuting charts | 14,725 | 0.751 | 0.991 | 18.24 s |

The self-context graft proves that this is compatible with the existing model:
retaining self-context and adding a positive commuting-chart residual raises
mean R² from 0.327 to 0.991 across the paired runs.

## What remains unresolved

The identity-preserving frame begins from the observed coordinate basis. It is
fully learnable, but the initialization gives it the correct frame on this
particular task. That is a useful default when input coordinates are meaningful,
not a proof of omni-inducement.

Starting from a random orthogonal frame is much more honest. It already gives a
large improvement, but its acquisition is unstable:

- ordinary frame learning: mean R² 0.827, range 0.623–0.960;
- five-times frame learning rate: mean R² 0.891, range 0.758–0.958;
- direct differentiable QR: mean R² 0.616, range 0.607–0.625.

Thus the representational question is solved for this task; the remaining
research question is how to acquire the commuting frame reliably. A promising
next mechanism is to use the operator sphere as a controller that promotes a
fixed/commuting chart path when the observed curvature repeatedly aligns,
without replacing the ordinary self-context state. That would turn the sphere's
task-sensitive selection into a connection choice rather than another output
mixture.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/periodic_nd_study.py \
  --out /tmp/periodic_nd_study --steps 2000 --seeds 3

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m unittest ML_experiment.test_periodic_nd_study
```
