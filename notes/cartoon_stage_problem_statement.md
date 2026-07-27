# The problem the cartoon stage's solver has to solve

Measured by `experiments/cartoon_stage_difficulty.py`. This document states
the problem exactly, says what every method we have used has in common, shows
which part is actually slow, and names the one class that attacks the part
that is.

## 1. The problem, exactly

`run_passes` alternates two Rudin-Osher-Fatemi solves. With `lam = 0.05` and
`mu = 40` as shipped:

| | fidelity `c` | Bregman `eta` | shrink threshold `theta = 1/eta` |
|---|---:|---:|---:|
| u-step (cartoon) | `lam` = 0.05 | `2*lam` = 0.10 | 10 |
| v-step (texture) | `1/mu` = 0.025 | `10/mu` = 0.25 | 4 |

Each solve is

```
    u* = argmin_u   sum_x || (grad u)(x) ||_2   +   (c/2) sum_x (u(x) - g(x))^2
```

with `grad` the forward difference and, currently, periodic boundaries. The
u-step takes `g = u + w`, the v-step takes `g = f - u`, so `g` moves every
pass: this is a *sequence* of related ROF problems, not one.

## 2. The structure that constrains any solver

1. **Strongly convex** in `u`, modulus `c`. The minimizer is unique and a
   linear rate is available in principle.
2. **Nonsmooth exactly where the answer lives.** `|| grad u ||_2` is not
   differentiable where `grad u = 0`, which is every flat zone of a cartoon.
   The nondifferentiable set is not a nuisance at the boundary of the
   problem; it is the object being computed.
3. **The nonsmooth term is composed with a linear operator.** `prox` of
   `TV` has no closed form, so every method splits `d = grad u` and pays for
   the splitting.
4. **The quadratic part is exactly solvable.** `grad^T grad = -Laplacian` is
   diagonalized by the DFT, so `(c - eta*Lap)^-1` costs one transform pair —
   or, after `include/bfft/meyer_facr_plan.md`, one transform and one sweep.
   **This part is not the difficulty and never was.**
5. **The conditioning is bad in the direction that matters.** With
   `||Lap|| <= 8`, the ratio `eta*||grad||^2 / c` is 16 for the u-step and
   **80** for the v-step. The fidelity that supplies the strong convexity is
   small precisely because we want heavy smoothing.
6. **The solution is combinatorial.** `u*` is piecewise constant. It is
   determined by its jump set `S = {x : grad u*(x) != 0}` together with the
   dual direction field on `S`. Conditional on those, the optimality system

   ```
   c(u - g) = div p,   |p| <= 1,   p = grad u*/|grad u*| on S
   ```

   is **linear**. A method that knew `S` would finish immediately.

## 3. What every method we have used has in common

Chambolle's projection, Chambolle-Pock, FISTA, and the shipped split Bregman
are all first-order operator splittings. They differ in constants and in
which variable carries the memory. They are identical in kind: each one
accesses the combinatorial object of point 6 **only through pointwise
thresholding**, one shrink at a time. None of them can represent "this region
is flat and its level is `t`" as a decision; they can only converge toward it.

That is the shared blind spot, and it is why swapping among them moves the
constant and not the exponent.

## 4. Which part is actually slow — measured

Cameraman, 512², tracking the jump set (top 5% of `|grad u|` in the converged
answer) and the spectrum of the remaining error against the 64-pass answer:

| pass | relative error | Jaccard of jump set with final | low-frequency share of error |
|---:|---:|---:|---:|
| 1 | 4.18e-2 | 0.539 | 0.279 |
| 8 | 2.38e-2 | 0.732 | 0.362 |
| 16 | 1.53e-2 | 0.829 | 0.507 |
| **24 (shipped)** | 1.16e-2 | **0.882** | 0.560 |
| 32 | 8.65e-3 | 0.918 | 0.614 |
| 48 | 4.10e-3 | 0.965 | 0.666 |

The jump set is not found early and then refined. **It crawls monotonically
for the entire run**, reaching 0.98 only at pass 55. On the synthetic
cartoon-plus-texture field it is worse: Jaccard sits at 0.05 through pass 16 —
the edge set the solver believes in for the first quarter of the run is nearly
disjoint from the right one — and only reaches 0.68 by pass 48.

Two things are true at once, and they are separable:

- **Error density is about 6x higher on the jump set** than off it (the
  on-set share is 0.40-0.56 of the norm over 5% of the pixels, against 0.22
  for white error). The edges are the hardest part per pixel.
- **The bulk of the remaining error is in the interiors and is increasingly
  low frequency** — the low-frequency share rises monotonically from 0.28 to
  0.67 (0.03 to 0.85 on the synthetic). Late in the run what is left is the
  *levels of the flat zones*, not the location of the edges.

## 5. What that rules in and out

**Rules out: semismooth Newton alone.** Superlinear convergence on the active
set is worth nothing when the active set is 0.88 correct at pass 24 — Newton
would converge quickly onto the wrong set. It is only viable as a *finisher*
after a splitting method has localized the edges, which is a hybrid nobody has
tried and which the table above says would have to start no earlier than pass
~40 to be safe.

**Rules out: cascadic initialization as the main lever.** Already measured at
~19%. It gives a better starting point for a process whose problem is its
rate, not its start.

**Rules in, partially: nonlinear multigrid (FAS).** The rising low-frequency
share is the textbook signature, and coarse-grid *correction* every cycle is a
different thing from the coarse-grid *initialization* already measured. This
attacks the interiors. It does nothing for the jump set.

**Rules in, fully: methods that never iterate on either quantity.** Only one
family qualifies.

## 6. The class that was not considered

**Coarea decomposition into exact binary problems.**

The total variation satisfies the coarea formula

```
TV(u) = integral over t of Per({u > t}) dt
```

and for ROF the minimizer's level sets are *themselves* minimizers:

```
{u* > t} = argmin_S  Per(S) + c * integral over S of (t - g) dx
```

Each of those is a binary labelling with submodular pairwise costs, so each is
solved **exactly by a minimum cut**, and the level sets are nested in `t`, so
a divide-and-conquer over the range of `g` gives the whole solution — the
Chambolle-Darbon parametric-maximum-flow construction.

Why it is the right shape for the two measured difficulties:

- The jump set is **not discovered iteratively**. It is the boundary of a
  min-cut, produced exactly, at every level, in one pass.
- The plateau levels are **not propagated iteratively**. The level of a flat
  zone is precisely the `t` at which that region changes side. The
  low-frequency tail of §4 does not exist as a phenomenon.
- The levels are independent given the nesting, so it is **parallel over `t`**
  in a way no splitting method is parallel.
- The input is 8-bit. The range of `g` is bounded and the number of distinct
  binary problems is naturally small.

**The honest costs.** Min-cut is exact for *anisotropic* TV — the `l1` of the
gradient components. The kernel uses isotropic `sqrt(tx^2 + ty^2)`. Extra
neighbour directions recover the Euclidean perimeter to within a few percent
(the standard metrication treatment), so this buys exactness in a slightly
different functional, not in ours. That is a real substitution and has to be
scored against the pipeline's own objective, not waved through. Max-flow is
also super-linear in the worst case and heavy at 4K, so this is a different
tradeoff, not a free win: it trades "cheap iterations that never quite finish"
for "expensive solves that finish exactly."

**A second, cheaper candidate in the same spirit.** Exact 1-D TV by taut
string / Condat runs in `O(N)` per line, and 2-D follows by Douglas-Rachford
over rows and columns. Each subproblem is then *exact* rather than a shrink
step, so the jump set along each line is decided rather than approached. The
FACR rework already puts the data in row-and-column sweep order, so the layout
is in place. This is the low-risk member of the family.

## 7. The falsification test — run, and it did not falsify

`experiments/cartoon_stage_tautstring.py`. One question: **does an
exactly-solved subproblem find the jump set in fewer passes than a shrink
does?**

Made apples-to-apples. Exact 1-D TV solves the *anisotropic* problem, so
split Bregman was rerun with a componentwise shrink, and — after a first run
was thrown out for it — with matching boundary conditions: the taut string
treats each line with free ends, so the Bregman solve was moved from a
periodic FFT to a DCT/Neumann one. `div(grad(u))` matches its DCT symbol to
**4.0e-16**, so both methods minimize the identical functional. The 1-D
solver is certified before use: worst relative duality gap **1.5e-14** and
worst dual infeasibility **3.8e-14** over 200 random problems, and it matches
an independent reference solver to 5e-15 on the solution itself.

* Method A: split Bregman, anisotropic shrink — the kernel's algorithm.
* Method B: Douglas-Rachford alternating exact 1-D TV over rows and columns.

**Cameraman, 512², iterations to reach a given jump-set agreement:**

| Jaccard | split Bregman | Douglas-Rachford | |
|---|---:|---:|---|
| 0.90 | 11 | 4 | 2.8x fewer |
| 0.95 | 17 | 5 | 3.4x fewer |
| 0.98 | 27 | 7 | **3.9x fewer** |

**Synthetic cartoon + texture, 512²:** split Bregman **never reaches Jaccard
0.90 in 64 iterations** (0.880 at 64). Douglas-Rachford reaches 0.90 at
iteration 29, 0.95 at 45, 0.98 at 61.

Relative error at equal iteration counts is 4.6x to 6.6x lower for
Douglas-Rachford (cameraman at 64: 2.9e-4 against 1.9e-3), and it reaches the
lower objective on both fields.

**And it is cheaper per iteration, not merely fewer.** 3 ms against 7 on
cameraman, 8 against 11 on the synthetic — because it contains no transform
at all, only O(N) sweeps. The two effects compound.

### What this does and does not establish

It establishes the mechanism claimed in §3: a method that *decides* its
subproblems reaches the combinatorial object several times faster than one
that only thresholds toward it. The jump-set curve moved sharply left, which
was the stated falsification criterion.

It does not yet establish that this should replace the stage, because:

- **It is anisotropic TV.** The functional differs from the shipped
  isotropic one, and the size of that substitution has not been measured
  against the pipeline's own objective. That is the next experiment and it is
  the one that decides.
- The taut string's per-line cost is content-dependent — the texture-heavy
  synthetic cost 2.7x more per iteration than cameraman — because the inner
  scan backtracks on short segments. A proper O(N) implementation flattens
  this; mine does not.
- Method A here is the NumPy reimplementation, verified equivalent to the
  kernel's algorithm in `experiments/cartoon_state_algebra.py`, not the C
  kernel. **The iteration-count comparison is the robust claim; the wall-clock
  comparison is indicative only.**

FAS multigrid on the interiors is no longer the fallback it was in §5 — but it
remains the untested lever for the low-frequency half, and nothing here
addresses that half directly.

## 8. What the substitution costs the pipeline

`experiments/cartoon_stage_isotropy_cost.py`. The substitution is made at the
module boundary — `bfft.meyer_split` and `bfft.rof` are swapped while the
model is built, so the metric, the edge and texture fields, `base_lab` and the
allocation pressure all see the substituted cartoon. `bfft.meyer_channels` is
**not** patched: it is the scoring operator and is applied to target and
reconstruction alike in every arm. Both arms grow to the same budget under the
same currency and finish with the same coupled solve.

A fourth arm isolates a confound I would otherwise have taken credit for: the
taut-string route uses free line ends while the kernel wraps, so an isotropic
Neumann arm separates the boundary change from the anisotropy.

| arm | Pikachu | | Chelsea | |
|---|---:|---:|---:|---:|
| | PSNR | objective | PSNR | objective |
| isotropic, periodic (shipped) | 26.36 | 3.192e-3 | 32.50 | 8.833e-4 |
| isotropic, **Neumann** | 26.04 | 3.411e-3 **(+6.9%)** | 32.49 | 8.879e-4 (+0.5%) |
| anisotropic, 24 passes | 26.66 | 2.968e-3 (−7.0%) | 32.59 | 8.771e-4 (−0.7%) |
| anisotropic, 8 passes | **26.78** | **2.893e-3 (−9.4%)** | 32.49 | 7.662e-4 (−13.3%) |

**The gain is the anisotropy, not the boundary.** The Neumann arm is
neutral-to-worse on its own — measurably worse on Pikachu — so the anisotropic
arm is winning against a *handicap*, not inheriting a boundary fix.

### This corrects §5 of `include/bfft/meyer_facr_plan.md`

That plan describes the Neumann variant as "better, and a behaviour change",
on the grounds that it removes defect 7 of `viewer/TRANSPORT_CELL_MATH.md`.
Measured against the pipeline's own objective, removing the wrap makes things
**worse** — 6.9% on Pikachu. It is still the mathematically cleaner boundary
condition and still worth having behind a flag, but it must not be described
as an improvement and must not be defaulted on. The FACR speed case rests
entirely on the padding and the transform count, and is untouched by this.

### Two things not to lean on

- **Chelsea, anisotropic 8 passes, is anomalous — audited in
  `experiments/cartoon_metric_audit.py` and confirmed an artifact.**
  `native_components` sets `scale = 255/(max-min)` per channel, an
  extreme-value statistic, **separately** for the target and the
  reconstruction, so the two are stretched by different affine maps before
  being decomposed and dividing each by its own scale afterwards does not
  undo it. Measured on Chelsea:

  | arm | scale drift | shipped | commensurable |
  |---|---:|---:|---:|
  | isotropic, periodic | 8.46% | 1.213e-4 | 3.051e-6 |
  | anisotropic, 24 | 5.79% | 1.295e-4 | 2.643e-6 |
  | anisotropic, 8 | 0.50% | 3.021e-6 | 2.596e-6 |

  All three agree within 17% once commensurable. The 8-pass arm merely
  happened to match the target's dynamic range, so its score collapsed onto
  the true value while the others stayed inflated ~40x. **Chelsea is neutral,
  not a win.** Pikachu's drift is comparable across arms so it is not
  differentially biased, and its commensurable cartoon term improves
  1.178e-4 → 9.839e-5 → 9.388e-5.

  This is a defect in the allocation objective, not in this experiment. Any
  `cartoon_mse` or `texture_mse` comparison between arms of differing dynamic
  range is contaminated, including earlier entries in
  `experiments/ALLOCATION_RESEARCH_LOG.md` and the sigma round.
  `rgb_mse` and PSNR are unaffected.
- **Fewer passes beat more passes in every arm.** Eight anisotropic passes
  beat twenty-four on both images. That is consistent with §7 — Douglas-
  Rachford converges roughly 4x faster per iteration, so 8 of its passes are
  worth ~32 Bregman ones — but it also means the pipeline objective is *not
  monotone in cartoon convergence*. A more exactly converged cartoon is a
  slightly worse input. That is worth understanding before anyone tunes
  `passes` on the strength of it.

## 9. Isotropic routes, after the picture rejected anisotropy

The visual A/B (`experiments/cartoon_visual_ab.py`) killed 4-neighbour
anisotropic TV: it staircases visibly on curved contours, and it gets worse
with convergence — axis-aligned edge energy rises 60.0% → 61.8% → 62.1% at 8,
24, 48 passes against a 51.8% target, so it is the functional, not the solver.
The isotropic functional has to stay. Two isotropic routes were then tested.

### 9a. Flat-zone coarse correction — dead, both variants

`experiments/cartoon_zone_snap.py`. The flat zones are a coarse grid that
respects edges exactly, unlike a geometric coarsening, so restricting the
correction to them should fix the plateau levels globally in one step.

**Replacement** (u constant on each zone) is rejected everywhere, by up to
16x: a ROF solution is not piecewise constant on its zones, and flattening
destroys the smooth shading the fidelity term wants.

**Additive correction** (u + sum_k delta_k on zone k, a proper multigrid-style
correction solved as a shifted fused lasso on the zone graph, accepted only
when the true isotropic objective falls) is accepted — and worth **exactly one
Bregman pass**, at 50x the cost, decaying to rejection by pass 24.

The diagnosis is worth keeping: the zones are derived from the current
iterate, whose jump set is only 88% correct at pass 24. **Correcting levels
inside wrong zones cannot help.** The interiors were never the lever; §4's
low-frequency signal was a symptom of the jump set, not a second problem. Any
scheme that fixes levels rather than boundaries is answering the wrong
question here.

### 9b. Isotropy is a discretization, not a binary

`experiments/cartoon_crofton.py`. The shipped isotropic TV is a 4-point
forward-difference discretization, and that is *also* grid-biased. The
Cauchy-Crofton construction gives a third point on the axis: for lattice
directions `e_k` with lengths `l_k` and angular spacings `dtheta_k`,

    TV(u) ~= sum_k w_k * sum_{lines along e_k} |du|,   w_k = dtheta_k/(2 l_k)

converges to true Euclidean TV as the direction set grows. **Every term is a
1-D total variation along disjoint lattice lines, so every subproblem is
still exactly solvable by the taut string.** The mechanism that beat split
Bregman 2.8-3.9x survives; only the number of directions it decides along
changes.

Axis-aligned edge energy on Pikachu at 475², all against the *same* object —
a plain ROF solve at `c = 0.05`, not the TGFD cartoon:

| | within 10° of an axis | excess over target |
|---|---:|---:|
| target | 51.8% | — |
| shipped isotropic ROF | 53.9% | +2.1 |
| Crofton, 2 directions (= anisotropic) | 54.4% | +2.6 |
| Crofton, 4 directions | 53.7% | +1.9 |
| Crofton, 8 directions | **53.4%** | **+1.5** |

Two-direction Crofton is worse than the shipped isotropic, which is exactly
what the picture said. Four and eight are **better** — eight cuts the excess
bias by 29%. All four solutions sit within 0.7% of each other as functions.

### What this does not yet establish

- **The margin is small at ROF level.** +1.5 against +2.1, on one image, on
  one statistic.
- **The decisive test has not been run.** A first version of this table
  compared Crofton ROF solves against the *TGFD cartoon* and showed a much
  larger margin (+5.3 for shipped). That was apples-to-oranges and the number
  above replaces it — but it exposed something worth chasing: the TGFD
  alternation **amplifies** grid bias, 53.9% at ROF level against 57.1% for
  the cartoon it produces. The Crofton construction therefore has to be tested
  *inside* the alternation, where the bias it is fixing is twice as large.
- PPXA convergence at 40 iterations is unverified, and it needs `K+1` sweeps
  per iteration against Douglas-Rachford's 2, so per-iteration cost grows with
  the direction count.

### Status

Quality: demonstrated on two images at one configuration. Speed: **not
demonstrated end to end.** The anisotropic split runs 8-9 ms against 3-4 ms
for the shipped C 24-pass split at 128², because it is NumPy and Numba
against optimized C. The iteration-count advantage of §7 is the reason to
expect a compiled version to win, and that remains a prediction.

