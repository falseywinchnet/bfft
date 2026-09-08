# What the Meyer/Bregman acceleration actually transfers to Matrix Transport

Date: 2026-08-30. Status: first formal translation and paired Ripple test.

Code: `ML_experiment/optimizers.py`, `MatrixTransport`.

Evidence:

- `results_matrix_transport_longitudinal_fusion_ripple_full/results.json`
- `results_matrix_transport_longitudinal_fusion_ripple_certified/results.json`

## 1. The iteration reductions in Meyer are three different results

It is misleading to attribute the complete speedup to Bregman geometry.
The measured reductions came from three separable operations.

First, the Gilles two-projector algorithm was being solved with tightly
converged inner ROF subproblems even though the outer target moved after each
one. Carrying the Split-Bregman state and interleaving one inner sweep per
outer pass reduced the synthetic rig from 21,071 to 810 inner sweeps. This is
the 26-fold algorithmic reduction. It removes work spent converging to a
soon-obsolete inner target; it is not momentum.

Second, eliminating the texture block identifies the exact static composite

    min_u TV(u) + (lambda/2) dist^2(f-u, G_mu).

The original alternation is ISTA on this objective at its maximal stable
step. FISTA is then legitimate because every extrapolated point belongs to
one fixed objective and the texture transport is recomputed at that point.
With accurate inner proximal solves, FISTA reached ISTA's 90-iteration knee
in about 20 iterations. With the production one-sweep inexact prox, however,
FISTA and ISTA were identical: extrapolation amplified the inner error and
spent the theoretical gain.

Third, the static ROF solver admits a one-shot Hodge consistency closure. If
`u` is the current primal and `p` the feasible dual flux, write

    p = p_L + p_T,
    p_L = grad(phi),
    div(p_T) = 0.

The quadratic term sees only `div(p)`. Its exact dual mismatch is

    r = c(u-g) - div(p).

One Poisson solve gives the minimum-energy longitudinal correction

    delta_p = grad Delta^{-1} r.

The algorithm adds `delta_p`, keeps the learned transverse route `p_T`, hits
the unit disk once, reads the induced primal, and admits a one-sided Taylor
step only when the original ROF objective decreases. It then re-seats both
primal and dual Split-Bregman state. This operation can replace two or three
early sweeps. The same research also established its boundary: applying the
closure inside every moving Meyer outer pass gives only 5--10% lower error
while roughly doubling time.

## 2. The exact and missing correspondences

Matrix Transport already contains the honest analogues of the first two
mechanisms:

| Meyer/ROF object | Matrix Transport object |
|---|---|
| warm `(d,b)` or dual-flux state | persistent matrix covariances and transported momentum |
| one inner sweep before the target moves | one optimizer update per newly observed gradient |
| exact screened-Poisson block solve | two-sided covariance root action `L^-1/4 G R^-1/4` |
| fresh reduced-composite residual | live matrix-whitened request mixed into the Nesterov read |
| do not carry momentum across an unaccounted moving frame | transport memory into the current frame and restrain its uncertain transverse part |

The Hodge correspondence is only partial. Let the live whitened request be

    P_t = p h,       ||h|| = 1,

and decompose bias-corrected transported momentum in its current frame as

    m_t = a h + z,   <z,h> = 0.

The successful Nesterov Matrix Transport read is

    u_N = beta m_t + (1-beta) P_t
        = [beta a + (1-beta)p] h + beta z.

The literal rank-one Hodge translation closes the longitudinal discrepancy
and retains the transverse route:

    u_H = P_t + beta z
        = u_N + beta(p-a)h.

This construction is orthogonally equivariant, preserves the first step and
the coherent fixed point, leaves transverse momentum exactly unchanged, and
prevents stale momentum from attenuating or reversing the live axial
decision.

But the analogy stops at the word "longitudinal." ROF has a known linear
operator, an exact KKT mismatch, a Poisson inverse, a pointwise dual-feasible
projection, and a deterministic objective admission test. A stochastic
neural gradient supplies none of those. `span(P_t)` is merely one currently
observed line; it is not the known range of a fixed differential operator.
Closing it exactly asserts curvature information that has not been observed.

## 3. The attempted objective-free admission certificate

To imitate the one-sided Meyer admission without another loss evaluation or
task-specific rule, the second arm earned closure from consecutive live-frame
coherence:

    gamma_t = max(<h_(t-1), h_t>, 0),

    u_C = u_N + gamma_t beta(p-a)h.

Contradictory frames therefore fall back exactly to `u_N`; a stable frame
earns the full longitudinal drop. Unlike the previously rejected adaptive
Nesterov arm, this certificate never changes transverse momentum.

## 4. Paired Ripple result

All arms used the same self-context model, width 24, batch 256, learning rate
0.003, 750 updates, five paired seeds, and evaluations every 25 updates. A
missed threshold is charged as update 775. No arm inspected the loss, selected
a task branch, or changed its declared learning rate.

| read | mean learning AUC | mean capped steps to 0.90 | 0.90 reached | mean final score |
|---|---:|---:|---:|---:|
| Nesterov Matrix Transport | **0.8210** | **370** | 5/5 | **0.9692** |
| full longitudinal closure | 0.8053 | 470 | 5/5 | 0.9485 |
| coherence-certified closure | 0.8045 | 450 | 4/5 | 0.9599 |

Full closure improved the early curve on one seed and lost on four. The
coherence-certified arm lost learning AUC on all five paired seeds. Its mean
certificate was only 0.24--0.29 and its admitted correction about 0.20--0.25
live-request norms, yet that was already enough to harm the moving problem.
Consecutive frames were nearly orthogonal on average: minibatch sampling and
self-context both change the observed geometry.

The negative is mechanistically specific. It does not say transverse-route
momentum is wrong: both `u_N` and `u_H` retain the identical `beta z`. It says
that a single live stochastic frame is not an exact longitudinal consistency
equation. Promoting its axial component from the Nesterov 10% live injection
to roughly 30% erased useful averaging.

## 5. What survives as the Euclidean design law

The valid translation is:

1. Accumulate memory only after the non-elementwise matrix metric has shaped
   the gradient.
2. Carry that memory only through an explicitly observed frame change.
3. Update once per fresh objective observation; do not over-solve a local
   model that the next minibatch or self-context state will replace.
4. Read a fresh live residual through the warm state. The fixed Nesterov read
   is the successful instance presently measured.
5. Preserve transverse momentum because it can encode where a curved valley
   is going, but restrain it where the frame observation is contradictory.
6. Do not call a rank-one live gradient an exact Hodge block. Analytical
   closure requires an identifiable operator or additional curvature
   evidence.

This explains why the current Nesterov Matrix Transport improvement is real
and why pushing its live component harder is not the next move. The remaining
untranslated Meyer result is true reduced-objective FISTA: evaluate the
gradient at an extrapolated parameter point and rebuild the matrix transport
there. That requires a lookahead-aware training interface (or optimizer
closure), not another algebraic mixture of the already-computed gradient. It
is the next mathematically clean experiment; it should be tested with a
gradient-based restart and charged one gradient evaluation per iteration.

## 6. Muon operator witness: closure works when the frame is truly static

The certified arm was next tested on the deliberately Muon-favorable operator
objective

    F(W) = 1/2 tr((W-Q) Sigma (W-Q)^T),

with orthogonal `Q`, condition-`10^4` covariance, dimension 32, three seeds,
and 500 updates. Under the population gradient this problem supplies the
exact structure missing from Ripple: the operator and its longitudinal frame
are fixed. The exact polar oracle reaches relative loss around `8e-30` in one
step.

At the common declared learning rate 0.003 on the population gradient:

| optimizer | steps to `1e-2` | steps to `1e-4` | final relative loss |
|---|---:|---:|---:|
| AdamW | 185 | not reached | `2.20e-3` |
| practical Muon, five NS steps | 368 | 498 capped mean | `9.66e-5` |
| Matrix Transport | 268 | not reached | `1.37e-4` |
| Nesterov Matrix Transport | 160 | 333 | `3.41e-5` |
| coherence-certified longitudinal fusion | **56** | **97** | **`3.23e-7`** |

This reverses the Ripple result for the predicted reason. Consecutive
population frames now describe one fixed operator, so the longitudinal
observation is trustworthy and the closure removes real momentum lag rather
than stochastic averaging.

At native operator scale, Matrix arms use the derived `1/sqrt(32)` rate
because their first request has RMS one; practical Muon uses rate 1. Both
Matrix Transport and longitudinal fusion cross `1e-2` in one update. Fusion
crosses `1e-4` at update 50. Practical Muon's first-step residual is `3.28e-2`
and it subsequently oscillates. This does not defeat Muon's algebraic claim:
the exact polar oracle is still the one-step solution. It distinguishes that
oracle from the production five-step Newton--Schulz approximation with
continued normalized momentum updates. Matrix Transport is using accurate
eigendecompositions on this small witness, so the iteration comparison does
not imply equal wall cost at large matrix size.

The rank-8 block stream supplies the necessary negative control. At common
rate 0.003, Muon reaches `1e-2` in 328 updates and `1e-4` in about 478;
longitudinal fusion never reaches `1e-2` and ends at `1.29e-1`. At native
scale, fusion cuts below `1e-2` at update 8 and reaches a best `2.18e-4` near
update 16, but regresses to `7.68e-2` by update 500. Individual blocks are
rank deficient and rotate the visible support, so no block supplies the
global longitudinal equation.

The combined result is stronger than a universal leaderboard:

- longitudinal closure is an exceptional accelerator when the observed
  matrix frame is a persistent global operator;
- transported Nesterov is safer on moving dense frames;
- Muon's normalized polar update assembles changing rank-deficient supports
  better at its standing rate;
- the missing robustness variable is not task identity but whether the live
  observation spans a stable global operator or only a moving partial view.

## 7. The same operator through self-context

The operator was then embedded as supervised learning rather than exposed as
the optimized parameter:

    x ~ N(0, Sigma),
    y = Q x.

A complete width-32 self-context network had to represent the 16-dimensional
orthogonal map through its embedding, nonlinear activation, two allocation
charts, self-context backpropagation, and output layer. AdamW, practical Muon,
Nesterov Matrix Transport, and certified longitudinal fusion shared the same
initialization, examples, declared rate 0.003, and 500-update budget. Three
seeds were run with both a fixed 1024-example population batch and moving
128-example minibatches.

| optimizer | population steps to `1e-2` | minibatch steps to `1e-2` | population final MSE | minibatch final MSE |
|---|---:|---:|---:|---:|
| AdamW | 200 | 213 | `2.31e-3` | `2.78e-3` |
| Muon | 143 | 153 | **`2.00e-4`** | **`2.73e-4`** |
| Nesterov Matrix Transport | **40** | 67 | `5.45e-4` | `6.69e-4` |
| longitudinal fusion | 43 | **57** | `5.41e-4` | `6.17e-4` |

Self-context therefore changes the outcome without erasing the transport
advantage. The transported arms reach the useful `1e-2` regime roughly three
times sooner than Muon. Fusion is the quickest minibatch arm and is stable
through 500 steps. Muon subsequently crosses beneath them and learns a more
accurate local Jacobian: mean zero-point Jacobian error is `0.18/0.28` for
Muon on population/minibatch versus `0.37/0.41` for fusion.

This exposes two phases rather than one winner. Matrix transport is the rapid
basin-entry geometry; Muon's polar normalization is the better long-tail
operator fitter once the self-context representation has organized itself.
The next principled construction is consequently a state-driven transition
between these geometries, but its trigger must be intrinsic evidence such as
closure exhaustion or stabilized operator support—not task identity, a held
out score, or a hand-selected iteration number.

## 8. The loss crossover is an endogenous state transition

The first Muon/Fusion loss crossover occurs at updates `230, 275, 260` on the
three population runs and `290, 315, 300` on the minibatch runs. Aligning
optimizer state to those six events reveals an initially compelling signature
inside Muon. Its pre-polar request changes as follows:

| relative update | participation-rank fraction | request/gradient cosine | request log-condition |
|---:|---:|---:|---:|
| -100 | `0.158` | `0.502` | `2.73` |
| -50 | `0.066` | `0.761` | `3.06` |
| -25 | `0.056` | `0.790` | `3.23` |
| 0 | `0.040` | `0.853` | `3.42` |
| +25 | `0.034` | `0.849` | `3.53` |

The quantity previously logged under `stable_rank_fraction` is more exactly
the inverse participation rank,

    r_IPR(S) = (sum_i sigma_i^2)^2 / sum_i sigma_i^4,

normalized by the smaller matrix dimension. A rank/alignment rule

    r_IPR / d < 0.05  and  cos(request, gradient) > 0.85

locates every observed crossover within 20 updates early to five updates late.
It says that Muon begins winning when its accumulated request has collapsed to
one or two persistent singular directions. Muon's polar factor then stops the
largest residual singular value from monopolizing the update and gives the
remaining supported directions comparable authority.

That rule is retrospective, not deployable. The same Muon EMA was therefore
carried as a read-only shadow state while Fusion continued to move the model.
At the crossover the shadow request has median participation-rank fraction
`0.136`, not `0.040`, and median gradient alignment `0.715`, with one population
run as low as `0.017`. Fusion's own request remains near `0.245` participation
rank and `0.792` polar cosine. Its metric condition, cached-root drift, polar
defect, frame confidence, and learning-rate gain contain no reproducible event.

The optimizer therefore creates part of the state later offered as evidence
for choosing it. Muon's action produces the narrow residual spectrum seen on
the Muon trajectory; Fusion does not passively approach the same state. This is
a controlled dynamical transition, not a landscape phase boundary that can be
read independently of the update law. A valid governor must observe unresolved
problem geometry on the active trajectory or maintain a genuine mixed
trajectory from the beginning.

## 9. Residual anisotropy supplies the missing governor

The scalar loss does not say which directions remain unresolved. The raw
output residual is also misleading in this problem because the input
covariance has condition `10^4`. The appropriate operator-level residual is
the training-data regression

    E_t = f_theta(X) - Y,
    B_t = (X^T X + lambda I)^-1 X^T E_t.

`B_t` removes input covariance and estimates the unresolved linear operator,
or equivalently the population-averaged residual Jacobian for this task. Its
scale-free entropy rank is

    p_i = sigma_i(B_t)^2 / sum_j sigma_j(B_t)^2,
    r_ent(B_t) = exp(-sum_i p_i log p_i) / d.

One means equal energy in all singular directions; values near `1/d` mean a
one-direction residual. On Fusion's own trajectory the median entropy rank
falls monotonically:

| relative update | residual-operator entropy rank |
|---:|---:|
| -150 | `0.313` |
| -100 | `0.265` |
| -50 | `0.248` |
| -25 | `0.242` |
| 0 | `0.236` |
| +25 | `0.228` |

This is observable on the trajectory being controlled. It should govern a
homotopy, not trip a brittle switch. Let `F_t` be the longitudinal-fusion
request, let `P_t = polar(F_t)` be rescaled to the same Frobenius norm, and let

    w_t = 0.9 w_(t-1) + 0.1 (1 - r_ent(B_t)),
    U_t = norm(F_t) * ((1-w_t) Fhat_t + w_t Phat_t)
          / norm((1-w_t) Fhat_t + w_t Phat_t).

The interpolation changes singular-value shape while preserving Fusion's
request norm. There is no iteration threshold, loss threshold, task identity,
or holdout measurement. Broad residuals retain Fusion geometry; concentrated
residuals continuously earn polar equalization.

This is deliberately a reversible state-feedback law, not a one-way phase
switch. If a new batch, layer, context chart, or task change makes the residual
operator broad again, `r_ent` rises and `w_t` decays back toward Fusion. An
optimizer committed permanently to Muon after its first narrow tail would lose
that option. The reversible form can use polar geometry locally and then return
to transported Euclidean geometry when the problem ceases to be Muon-shaped.

Rank alone should not receive final authority. A narrow spectrum can be a real
persistent tail or a transient minibatch accident. Let

    C_t = B_t^T B_t / trace(B_t^T B_t)

be the unit-trace residual spectral density, transported into the current
frame when necessary, and define persistence by its normalized overlap with
the previous density. A mature target is therefore

    a_t = (1 - r_ent(B_t)) * persistence(C_(t-1), C_t),

with slow admission and faster release. Isotropy drives the first factor to
zero; an incoherently rotating low-rank residual drives the second to zero.
Only a narrow, persistent residual earns strong polar equalization. This
generalizes the same restraint principle: uncertain geometry falls back to
Fusion rather than being amplified.

The first three-seed experiment is positive:

| optimizer | population `3e-3` | minibatch `3e-3` | population final MSE | minibatch final MSE | minibatch Jacobian error |
|---|---:|---:|---:|---:|---:|
| Muon | 190 | 212 | `2.89e-4` | **`4.61e-4`** | `0.372` |
| longitudinal fusion | 92 | 120 | `2.92e-3`* | `7.44e-4` | `0.437` |
| residual-polar governor | **83** | **103** | `4.48e-4` | `5.33e-4` | **`0.359`** |

`*` Fusion's population mean includes one late blow-up after reaching
`7.67e-4`; the residual-polar arm removes that failure. It preserves the
38-update population `1e-2` entry time, improves Fusion's useful milestones,
pushes Muon's eventual crossover roughly 30–50 updates later, and substantially
closes the long-tail gap. Pure Muon still owns the 400-update population floor,
so this is evidence for the governing variable rather than a finished
optimizer.

The remaining interface problem matters. For a linear module with batch
activations `A` and backpropagated cotangents `Delta`, ordinary backprop exposes
only the contraction

    G = Delta^T A.

The local residual operator's spectrum generally cannot be reconstructed from
`G` alone. The general, non-task-conditioned implementation should capture
`(A, Delta)` before contraction and maintain a small randomized or factored
sketch of

    (A^T A + lambda I)^-1 A^T Delta.

At the output this reduces to `B_t`; inside self-context it becomes a local
chart-aware residual observer. This explains why every optimizer-only scalar
failed while the covariance-corrected residual succeeded: the relevant
anisotropy lives in the backprop relation that produces the gradient, not in a
single already-contracted gradient tensor.
