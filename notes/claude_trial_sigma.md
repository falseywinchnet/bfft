# Sigma round: exact algebra on the coupled system, and descent on the graph

Files: `viewer/claude_trial_sigma.py`, `experiments/claude_trial_sigma_run.py`.
Nothing in the validated model was edited.

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/claude_trial_sigma_run.py --images pikachu \
        --max-side 256 --cells 2400 --steps 18

## The two premises

**The coupled solve already builds the graph the spin experiment needed.**
Its design matrix has exactly two nonzero cell blocks per pixel — owner and
runner-up — weighted by the partition of unity. Its normal matrix

    G = A^T A + Lambda

is therefore the renderer's own adjacency graph, with entries equal to the
measured inner products of the atoms. The earlier signed graph was built from
distance, and rounding preserves the objective it is handed. `G` is the
objective that experiment wanted, and it costs nothing extra to have.

**A shortest path is a minimum of linear functions of the edge weights.** By
the envelope theorem its derivative is the indicator of the achieving path,
so the loss sensitivity accumulated backwards over the Dijkstra predecessor
forest is the *exact* gradient of reconstruction error with respect to every
edge of the metric — one linear pass, no perturbation, no adjoint solve, no
iteration inside the gradient. Descent on the graph, not on the flow through
it.

Both premises were falsified before any score was trusted, in `--check`:

| claim | test | result |
|---|---|---|
| the predecessor walk is the production walk | owner agreement, distance gap | `1.000`, `0.0` |
| LSMR reaches the exact solve | relative coefficient gap at 160 iterations | `5.3e-3` |
| the envelope gradient is the real gradient | six probes against central differences | agreement `0.97`–`1.02` |

## What happened

Pikachu, 256 px, 2,400 cells, coupled overlap 4 / 16. Objective is
`rgb + cartoon + texture` MSE, lower better. `struct` is the removable share
of the remaining residual — the diffuseness of the error, made a number.

| trial | cells | PSNR | objective | struct | solve |
|---|---:|---:|---:|---:|---:|
| coupled_control (LSMR) | 2400 | 29.08 | 1.721e-3 | 0.0028 | 0.58 s |
| direct_normal | 2400 | 29.17 | 1.685e-3 | 0.0027 | 0.09 s |
| graph_random *(control)* | 2400 | 29.17 | 1.685e-3 | 0.0027 | 1.87 s |
| leverage_exchange | 2400 | 29.27 | 1.648e-3 | 0.0028 | 0.51 s |
| ridge_random *(control)* | 2400 | 29.71 | 1.493e-3 | 0.0025 | 0.19 s |
| graph_features (7 numbers) | 2400 | 30.04 | 1.387e-3 | 0.0025 | 1.98 s |
| ridge_geometric *(control)* | 2400 | 30.13 | 1.350e-3 | 0.0027 | 0.19 s |
| graph_descent (free field) | 2400 | 30.75 | 1.185e-3 | 0.0028 | 3.10 s |
| ridge_enriched | 2400 | 31.46 | 9.992e-4 | **0.0015** | 0.30 s |
| **weight_descent** | 2400 | **31.50** | **9.964e-4** | 0.0033 | 3.73 s |
| matched_budget *(control)* | 3733 | 31.63 | 9.614e-4 | 0.0033 | 0.12 s |
| weight_ridge | 2400 | 33.24 | 6.662e-4 | 0.0017 | 2.34 s |
| weight_matched *(control)* | 3733 | 34.75 | 4.753e-4 | 0.0035 | 2.84 s |

At the raised cell count the same weight descent is worth even more:
`matched_budget` 31.63 dB becomes `weight_matched` 34.75 dB, +3.12 dB for the
same 3,733 cells and the same 9 numbers each.

Generality, 128 px / 700 cells, PSNR and objective:

| image | direct | weight_descent | graph_descent | ridge_enriched | matched_budget |
|---|---|---|---|---|---|
| camera | 29.11 / 4.20e-3 | 30.54 / 3.16e-3 | 30.08 / 4.04e-3 | 31.43 / **1.86e-3** | 31.58 / 3.91e-3 |
| chelsea | 32.76 / 9.61e-4 | 33.68 / 8.60e-4 | 33.25 / 8.40e-4 | **35.28** / **4.42e-4** | 34.88 / 5.61e-4 |
| coins | 28.17 / 2.38e-3 | 29.95 / 1.40e-3 | 29.15 / 1.65e-3 | **30.39** / **1.28e-3** | 29.54 / 2.50e-3 |
| astronaut | 24.74 / 4.40e-3 | 25.91 / 3.37e-3 | 25.22 / 3.96e-3 | **27.46** / **2.37e-3** | 27.24 / 2.50e-3 |

## Findings

**1. The reach was wrong, not the metric.** The free per-pixel barrier field
(65,536 parameters) gains +1.6 dB. One additive weight per site (2,400
parameters) gains +2.33 dB — more, from 27× fewer degrees of freedom, on
every image tested. The free field was spending its capacity inefficiently
approximating a per-site reach correction. The gradient for that restriction
is not a new derivation: an additive site weight shifts every distance in
that site's whole subtree by the same amount, so the forest sum collapses
into two `bincount` calls.

This lands on the additive weights of a geodesic power diagram — the object
semi-discrete optimal transport solves for prescribed masses. Here it is
solved against *measured reconstruction error* instead. The step schedule is
still a normalized gradient with halving, and the trace was still rising at
step 15, so this number is a floor, not a limit. A damped Newton step on the
weights is the obvious lever.

**2. Random metric change costs what gradient metric change gains.** Every
random draw at the same span lost 2.4 dB, so `graph_random` never left its
starting point. The loss surface around the BFFT metric is steep and narrow,
which is why the metric felt validated: almost every direction is worse.

**3. The metric is not repairable by retuning its ingredients.** Projecting
the same exact gradient onto seven BFFT feature maps — cartoon edge, texture
activity, entropy, coherence, demand, flow, constant — recovers 30.04 dB,
about half the free field's gain and well short of the site weights. The
learned coefficients wanted more barrier everywhere (cartoon edge +0.35,
texture activity +0.31, texture demand +0.31, coherence −0.05). That is a
sharper statement than "the constants need tuning": no reweighting of the
existing ingredients reaches where the site weights go.

**4. Enrichment is real, and its value is entirely in the measurement.**
One bounded `tanh` ridge column per cell, with axis and offset from the
residual's own cumulative sign statistics, beats the matched-number control
on the objective on all four images at 700 cells. The controls separate
cleanly:

| axis source | Pikachu PSNR | Pikachu objective |
|---|---|---|
| random through the site | 29.71 | 1.493e-3 |
| cartoon edge normal | 30.13 | 1.350e-3 |
| measured from the residual | **31.46** | **9.992e-4** |

The measured axis sits 34.1° from the cartoon edge normal on average
(24.4° at strong edges; 52.7% within 30°, against 33% for chance). So the
contour normal is correlated with the answer and is not the answer, which is
the earlier geometry-only dipole result restated with a number attached. The
reason is mechanical: the affine plane has already absorbed the ramp across
the contour, so what survives is not oriented along the contour normal.

Boundedness is why this composes where the quadratic patch did not.
Partition-of-unity theory attributes overlap stability to the local Lebesgue
constant, and `|tanh| <= 1` keeps it finite; a quadratic's does not.

**5. Geometry and function space are not independent faults.** Composing
them is sub-additive: on Pikachu, weight descent is worth +2.33 dB and the
ridge alone +2.29 dB, but together +4.07 dB, not +4.62 dB. The ridge's
marginal value falls from +2.29 to +1.74 once the boundaries are right,
because part of what it was buying was compensation for a boundary in the
wrong place. Against the matched-number control after weight descent the
ridge wins the objective on 2 of 4 images and loses PSNR on 3 of 4. As a
route to lower error it is now a wash; the reason to keep it is (6).

**6. Adding cells and adding the right basis function fail differently.**
This is the one invariant that held in every configuration. At matched
transmitted numbers, enrichment leaves a residual roughly twice as diffuse as
extra cells do — Pikachu 0.0017 against 0.0035, camera 0.0024 against 0.0041,
chelsea 0.0045 against 0.0077, coins 0.0030 against 0.0068, astronaut 0.0038
against 0.0072. More cells buys error reduction and leaves the error shaped.
The right basis buys error reduction and whitens it. If diffuse error is the
goal, the two are not interchangeable even where their MSE is.

**7. Factoring beats iterating, slightly and for free.** The direct
factorization of `G` reaches 29.17 dB against LSMR's 29.08 at 160 iterations,
in 0.09 s against 0.58 s. 7,200 unknowns, 127k nonzeros, one `splu` in
symmetric mode.

**8. Exact deletion pricing did not pay yet.** `leverage_exchange` — retire
by exact constrained-deletion cost, re-seed by exact Schur-complement gain —
returns +0.10 dB on Pikachu and +0.03 dB on Cameraman. The algebra is right
and the mechanism is small. The likely reason is that after growth under the
expected-gain currency there are few genuinely worthless cells to find, which
is itself a compliment to the current allocator.

## What this round did not do

The scale field and the site weights are derived from the image and are in
the same accounting category as the BFFT metric, the site angles, and the
density map that the model already uses. None of it is decoder-reproducible.
If this ever becomes a codec rather than an analysis representation, all of
it is payload and the accounting changes for every one of these results,
including the existing ones.

The ridge axis and offset *are* counted, at 2 numbers per cell, which is why
`matched_budget` raises the cell count by 14/9 rather than 12/9.

---

# Alpha round (sibling session, 2026-07-26): the slack meter

Files: `viewer/claude_trial_alpha_normal.py`, `claude_trial_alpha_persistence.py`,
`claude_trial_alpha_market.py`, `claude_trial_alpha_run.py`.  Written
independently of the sigma round above and merged into this note afterwards.
Nothing in the validated model was edited.

    PYTHONPATH=.:viewer .venv/bin/python viewer/claude_trial_alpha_run.py \
        --max-side 128 --cells 700 --images pikachu camera coins grass chelsea

## Independent confirmation of two sigma findings

Reached separately, with a separate implementation, on a separate config
(`RGB + decomposition gain`, 256 px / 2,400 cells):

| claim | sigma | alpha |
|---|---|---|
| factoring beats iterating | 29.17 vs 29.08 dB, 0.09 s vs 0.58 s | 28.83 vs 28.76 dB, 0.12 s vs 0.55 s |
| normal matrix size | 7,200 unknowns, 127k nonzeros | 127,026 nonzeros, mean degree 4.88 |
| exact deletion pricing is a small lever on its own | `leverage_exchange` +0.10 dB | `deaths` −0.07 dB |

Two implementations landing on the same nonzero count is a real check on both.
The mean degree 4.88 is the useful shape fact: the renderer's graph is
essentially planar, which is why `splu` in symmetric mode is so cheap and why
nested dissection is the right long-run ordering story.

## What is new: the deletion price is a scene-level slack meter

Sigma finding 7 says exact deletion pricing "did not pay yet", and guesses the
reason is that few worthless cells exist after growth under the expected-gain
currency.  That guess is correct and can be turned into a measurement, because
the price answers the question directly.  Cells removable within 1% of the
fitted objective, 128 px / 700 cells:

| image | removable cells | cheapest cell, as % of objective |
|---|---:|---:|
| pikachu | 101 of 700 | 0.0000 |
| camera | 1 of 700 | 0.907 |
| coins | 0 of 700 | 2.213 |
| grass | 0 of 700 | 2.220 |
| chelsea | 0 of 700 | 7.475 |

This is the shape of the whole allocation log in one table.  Every idea in it
that imposed a scene-independent focus quota helped clean-geometry scenes and
hurt textured ones, and the reason is that **only clean-geometry scenes have
idle budget to reallocate**.  Pikachu carries 14% slack; Chelsea's cheapest
cell already costs 7.5% of the objective, so there is nothing to sell at any
threshold.  The log's sharpest open sentence — "a focus map can identify
attention without specifying how much budget that attention deserves" — has an
answer that is exact, per scene, and correctly returns zero.

Two supporting pieces:

**An O(n) exact upper bound makes pricing everything affordable.**  The exact
price is `c_i^T (H_ii - H_ij H_jj^-1 H_ji) c_i`.  The subtracted coupling term
is PSD, so `c_i^T H_ii c_i` bounds every cell's price at once with no solves —
a cell whose bound is small cannot be expensive.  Shortlist by the bound, then
solve exactly on the shortlist.  Median bound/exact ratio 1.06, so the bound is
tight as well as sound.  This is the cheap half of Erisman–Tinney; the gap
between bound and exact price is precisely the graph's contribution.

**Price against everything the cell is asked to explain.**  Pricing the cartoon
field alone reports texture-carrying cells as idle, because they are idle in
the field being priced and nowhere else.  The first market built this way lost
0.3 dB.  Summing the cartoon price and `precision^2` times the texture price
turned the same mechanism into a gain.  Worth flagging for `leverage_exchange`.

## The market that follows, and why it generalizes

Budget-neutral exchange: price all cells (bound-shortlisted), sell the cheapest
prefix whose cumulative price stays under a fraction of the objective, re-buy
the same number of sites from the pressure field, re-assign, re-fit.  The sale
price is an upper bound *by construction* — it holds the partition fixed, while
a real removal lets neighbours re-own the freed pixels — so an affordable sale
is genuinely affordable.

Pikachu, 256 px, 2,400 cells, sale threshold 0.2% of the objective per round:

| | cells | PSNR | RGB MSE |
|---|---:|---:|---:|
| control (growth + 3 exchanges) | 2400 | 28.832 | 1.309e-3 |
| + 8 market rounds | 2400 | **29.046** | **1.246e-3** |

+0.21 dB at an identical cell count, 28 s, no change to metric, assignment,
renderer or basis.  Trace: 28.607, 28.903, 28.930, 29.002, 29.025, 29.028,
29.034, 29.046 — the first round always dips, because a sale is exact and a
purchase is a heuristic, and the run keeps the best measured state.

The property worth more than the number: **on the guard set the market does
nothing at all**.  At 700 cells no cell on camera, coins, grass or chelsea is
cheap enough to sell, so it reports "nothing affordable" and returns the input
unchanged.  This is the first thing tried in this line that is scene-adaptive
without a scene-dependent knob — not because it was tuned to be safe, but
because the quantity it thresholds is the thing that actually varies.

## Negative result: persistence-ranked births

0-dimensional persistence of the superlevel filtration of the pressure field
(one descending sort, one union-find pass, no smoothing and no scale
parameter) as a replacement for the robust cap, the Gaussian blur, the
exclusion disk and the per-cell quota.  The elder rule separates a one-pixel
spike from a broad under-resolved region by construction, which is what all
four of those defences approximate by hand.

Measured: **+0.20 dB** Pikachu at 128 px / 700, **−0.49 dB** camera,
−0.06 coins, −0.01 chelsea, +0.01 grass, and **−0.03 dB** Pikachu at
256 px / 2,400.  Scene-dependent and negative at the working budget, so it is
not adopted.  The likely mechanism, worth one more experiment: global
persistence ranking discards the per-cell quota, which is what spreads births
across the image; ranking *within* each cell while keeping the quota would test
the topology claim without losing the spread.  `persistent_sites` is written
and cheap, so this is a small experiment rather than a project.

## Standing on sigma's open lever

Sigma finding 1 lands on the additive weights of a geodesic power diagram,
solved against measured error, and notes the trace was still rising at step 15.
There is an exact theorem for the adjacent problem worth knowing before adding
a Newton step by hand: for any prescribed set of cell capacities, the power
diagram realizing them exists, is **unique**, and is the maximizer of a concave
function (Aurenhammer–Hoffmann–Aronov 1998; Kitagawa–Mérigot–Thibert for the
globally convergent Newton form, PDF in `papers/`).  No local minima, unlike
the Lloyd iteration it would replace.  That gives the weight descent a
principled target for what the weights *should* equalize, and connects it to
the budget question above: the multiplier sets the capacity, the theorem
supplies the unique geometry realizing it.
