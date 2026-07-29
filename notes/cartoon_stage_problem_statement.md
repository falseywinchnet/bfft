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

## 10. Exact-isotropic radial/angular factorization

The Crofton fork is excluded here: fixed spatial directions recover a
metrication method rather than preserving the pointwise Euclidean norm. The
experiment in `experiments/cartoon_radial_direction.py` instead factors the
exact isotropic dual flux as

    p_i = rho_i n_i,       0 <= rho_i <= 1,       |n_i| = 1.

For the continuous directions `n` captured from an early isotropic Split
Bregman pass, it solves the radial box-QP

    min_rho  1/2 ||g + div(rho n)/c||^2.

No direction is quantized and the feasible set remains the Euclidean unit
disk. An independent Neumann FGP reference was solved to relative duality gap
`2.0e-8` on Cameraman and `3.6e-8` on the synthetic field.

### Direction settles before support

On final-edge pixels, the early dual directions are substantially more stable
than the binary jump set:

| field | pass | jump-set agreement | directions within 10 degrees |
|---|---:|---:|---:|
| Cameraman | 8 | 0.893 | 99.0% |
| synthetic | 8 | 0.583 | 75.4% |
| synthetic | 24 | 0.712 | 95.4% |

This invalidates the strongest version of the earlier objection to Newton:
the angular geometry can be useful well before the active set is complete.

### The radial solve is a predictor, not a primal replacement

The fixed-direction solve improves Cameraman's jump-set agreement sharply
(pass 2: `0.52 -> 0.88`; pass 8: `0.87 -> 0.96`) and improves the feasible
dual bound by `1.6-2.9x`. But its primal objective is worse than the
originating Bregman iterate after the first few passes. Frozen directions
select plausible boundaries while violating angular primal-dual consistency
elsewhere.

Used as a feasible dual warm start for an unrestricted isotropic FGP
corrector, it reduces the steps to a one-percent relative gap:

| field | captured pass | raw Bregman dual | radial predictor |
|---|---:|---:|---:|
| Cameraman | 4 | 41 | 24 |
| Cameraman | 8 | 31 | 13 |
| Cameraman | 12 | 21 | 7 |
| synthetic | 8 | 76 | 67 |
| synthetic | 16 | 54 | 45 |
| synthetic | 24 | 34 | 26 |

The mechanism is real but content-dependent: decisive on Cameraman, modest on
the texture-heavy synthetic field.

### Falsification result

The generic radial QP needs `249-926` L-BFGS-B iterations at 256 squared, so
this implementation is not competitive with Split Bregman. RK on the
projected flow is not the missing operation either; it would repeatedly pay
for the same activation events. The remaining viable version is narrower:

1. use a few isotropic Bregman passes only as a direction predictor;
2. solve radial complementarity with a genuine box-QP active-set or multilevel
   method rather than generic first-order optimization;
3. update directions by a tangent Newton model, applying each step by an exact
   unit-circle rotation;
4. accept against the exact isotropic primal-dual gap.

The BFFT half-angle tree may bound angular sectors or implement the final
rotation, but a fixed-depth leaf must not define the feasible disk; that would
return to metrication.

### Segmenting-front transplant: local analytic acceptance is insufficient

`segmenting_veroni_app.py` does not optimize its first-arrival front to a
tolerance. On fixed topology it computes a closed-form local stationary point,
accepts it in causal order, and never revisits it. The same construction
applies to the radial QP because one `rho_i` changes only three entries of
`div(rho*n)`. Its exact clipped coordinate minimizer is

    rho_i <- clip(rho_i - (a_i dot u) / (a_i dot a_i), 0, 1),

where `a_i` is that three-entry divergence column. A low-to-high pass updates
the residual immediately, so every accepted coordinate sees all causal
predecessors. The warm compiled pass costs about `0.7-1.5 ms` at 256 squared.

It is not decisive. On Cameraman it advances pass 2 to the objective reached
at Bregman pass 4 and pass 4 to pass 5, but from pass 8 onward it is neutral or
harmful. On the synthetic field it advances pass 4 to pass 7 and pass 8 to
pass 9, then falls behind. It barely changes the unrestricted correction
count. Local exactness is still local.

The oracle explains why. At Cameraman 128 squared from pass-8 directions,
`38.3%` of optimal radial variables are capacity-saturated, only `1.3%` are
zero, and `60.3%` are interior. The answer is not a binary capacity mask.
Interior radial flux must accumulate continuously across whole supports.

The segmenting primitive that matches this structure is therefore not the
local Newton step but **reverse residual transport**:

1. build a causal forest following the continuous dual direction;
2. fit a tentative plateau level by the closed-form regional mean;
3. reverse-accumulate `c*(u-g)` through the forest, which gives every required
   interior flux exactly by subtree summation;
4. where accumulated flux first exceeds unit capacity, saturate that event
   and bifurcate the support;
5. repeat the same reduction recursively on the emitted subtrees.

This is the radial analogue of direct density population plus predecessor-tree
refill. It is a finite capacity-event construction, not a box-QP solver. Its
remaining mathematical obligation is to define the characteristic forest and
its edge capacities so that its discrete divergence is the existing
forward-difference divergence; otherwise it would silently substitute a
directional graph functional.

## 11. Finite characteristic-tree solve

`experiments/cartoon_characteristic_tree.py` implements the conditional tree
problem explicitly:

    min_u  (c/2) sum_v (u_v-g_v)^2
           + sum_(v,w in T) a_vw |u_v-u_w|.

For a rooted tree, the derivative message before clipping is

    H_v(x) = c(x-g_v) + sum_(w child of v) clip(H_w(x), -a_vw, a_vw).

Every message is continuous, monotone, and piecewise linear. Its two capacity
events are the analytical crossings `H_v(x)=-a_vw` and `H_v(x)=a_vw`.
A bottom-up meld of slope events, one root crossing, and the top-down rule

    u_w = clip(u_v, crossing_w^-, crossing_w^+)

therefore solve the fixed-tree problem exactly without a tolerance or
convergence loop. A reverse subtree sum independently certifies its KKT
conditions; measured violations are `7e-13` to `1e-11`.

This does **not** declare tree-TV to be isotropic TV. It is a finite proposal,
and it is accepted only when it lowers the original forward-difference
Euclidean-TV objective.

Two tree constructors were tested at 128 squared:

- a direction-weighted 8-neighbor minimum spanning tree;
- a linear-work causal raster tree, where each pixel chooses the best-aligned
  already-accepted neighbor.

From the raw image gradient, the cheaper causal tree reaches approximately
Bregman pass 3 on both Cameraman and the texture-heavy synthetic field. Its
original-objective deficit falls from `132741` to `36784` on Cameraman and
from `199878` to `23141` on synthetic. Starting from pass-2 directions, the
MST reaches pass 3 on Cameraman and pass 4 on synthetic, so paying for the
direction warm-up buys little.

The global capacity normalization is image-dependent. An oracle sweep prefers
about `0.85x` on Cameraman and `1.5x` on synthetic. On a fixed tree active
set, however, every fused plateau is affine in that multiplier:

    d u_C / d s = u_C - mean(g_C).

The experiment uses this identity for one one-sided Taylor drop in the true
isotropic objective, including the exact absolute-value kink at zero spatial
gradients. It predicts `0.878x` on Cameraman and improves the deficit from
`37377` to `36784`. On synthetic it correctly refuses a local move: reaching
the `1.5x` oracle requires crossing capacity events, not a higher-order local
integrator.

The initial Python timings were `~60 ms` construction and `~180-260 ms`
treap solve. Both causal construction and exact message passing were then
compiled. The compiled result is bit-identical to the Python reference and
costs about `0.68 + 7.5 ms` at 128 squared. At 256 squared it costs
`2.9 + 65 ms`, while a warmed optimized Bregman pass costs `5.6 ms`.

This closes the cost question for the present construction. The finite tree
solve replaces about three Bregman passes at 128 squared but costs about
sixteen; at 256 squared it costs about twelve passes before any isotropic
correction. It remains a useful exact oracle and proves that reverse
capacity-event transport can be implemented without optimization, but the
single-tree surrogate is too weak per unit work to become the cartoon-stage
descent. More engineering cannot bridge the required `4-5x` gain in useful
objective progress; the next analytical construction must preserve more than
one incident flux constraint per pixel rather than compressing the grid to one
tree edge.

## 12. Fourier/Hodge transport closure

`experiments/cartoon_fourier_transport.py` keeps both incident flux
components and asks which part of their motion Fourier space can complete
analytically.

With a frozen Split-Bregman projection branch, the longitudinal increment of
each Fourier mode follows the exact scalar recurrence

    r(omega) = eta L(omega) / (c + eta L(omega)),

where `L` is the positive symbol of the periodic negative Laplacian. The
remaining geometric tail is therefore

    D_k(omega) r/(1-r) = D_k(omega) eta L(omega)/c.

This tail is phase coherent in measurement: after pass 4, `94-100%` of
increment power lies in shells whose consecutive complex increments have
coherence above `0.85`. Nevertheless, an objective-safe Taylor drop along
the frozen-symbol limit advances only one Bregman pass. Frequency phase is
not what changes. The spatial unit-disk branch is.

### The missing field is the divergence-free route

The periodic Hodge split is

    p = p_L + p_T,
    p_L = grad phi,
    div p_T = 0.

The ROF quadratic sees only `div p = div p_L`, so `p_L` is diagonal in
Fourier space. But `p_L` alone is not pointwise feasible. At the converged
128-squared reference:

| field | transverse flux energy | pixels with `|p_L|>1` | `max |p_L|` |
|---|---:|---:|---:|
| Cameraman | 30.6% | 17.4% | 2.51 |
| synthetic | 56.2% | 8.1% | 2.09 |

Thus the nominally invisible transverse flux carries essential spatial
information: it routes the longitudinal load so the sum stays in the unit
disk. Its cosine agreement with the final transverse field grows gradually,
for example `0.51 -> 0.64 -> 0.77 -> 0.87` at Cameraman passes
`2,4,8,16`. A Fourier-only extrapolator necessarily moves along the correct
frequency phase but through the wrong local capacity branches.

### One-shot consistency closure

The Hodge decomposition does yield a useful finite operation. Given the
current primal `u`, feasible dual flux `p`, and desired divergence
`q=c(u-g)`, compute

    delta_p = grad Delta^{-1}(q - div p).

This is one scalar Poisson solve (one FFT pair). Adding `delta_p` changes
only the longitudinal flux, so it preserves the current transverse route.
Then perform exactly one pointwise disk projection,

    p_hat = projection_unit_disk(p + delta_p),
    u_dual = g + div(p_hat)/c,

and take one one-sided quadratic Taylor drop from `u` toward `u_dual`.
The candidate is evaluated in the original isotropic objective and rejected
if it does not decrease it. There is no convergence loop.

Measured equivalent-pass advances:

| size | field | pass 2 | pass 4 | pass 8 |
|---|---|---:|---:|---:|
| 128 | Cameraman | 4 | 7 | 10 |
| 128 | synthetic | 5 | 7 | 11 |
| 256 | Cameraman | 4 | 6 | 9 |
| 256 | synthetic | 4 | 7 | 11 |

The NumPy prototype costs `0.69 ms` at 128 squared and `2.87 ms` at
256 squared, versus periodic Bregman sweep costs of `0.31 ms` and
`1.61 ms`, respectively. At 256 squared the closure costs `1.78` sweeps
and often replaces `2-3`, crossing the possibility-of-cheaper gate.
At 512 squared the measured ratio remains `1.92` sweeps. The internal C
kernel already owns the Laplacian symbol, divergence stream, vector flux,
and FFT workspaces, so a fused implementation should avoid most prototype
allocation.

This is not a terminal ROF solve. It is an analytical consistency drop,
best used once after `2-8` ordinary sweeps. Applying it to an isolated
static ROF subproblem is objective-safe; the distinct moving-frame use is
tested below.

That moving-frame test is negative. On the 128-squared Meyer rig, applying
the closure to the cartoon subproblem after every outer sweep lowers
error-to-reference by only `5-10%` at a given outer count while approximately
doubling wall time; closing both subproblems adds essentially no accuracy and
triples time. For example at 32 outers, baseline error/time is
`2.41e-2 / 22.3 ms`, cartoon-closure is `2.16e-2 / 47.1 ms`, and closing
both is `2.16e-2 / 75.3 ms`. The Fourier/Hodge drop therefore belongs to a
static low-budget ROF solve, not inside every moving Meyer pass. This matches
the earlier reduced-composite result: spending extra work to solve a
soon-to-move inner objective is waste.

## 13. The active-capacity Fourier coupling

`experiments/cartoon_fourier_active_coupling.py` isolates the obstruction
left by §12. Let `D` be periodic divergence and

    P_T = I - D* (D D*)^-1 D

the Fourier-diagonal transverse projector. For a fixed overloaded set, let
`N` sample the current outward normal component at those pixels. The exact
linear fixed-normal coupling is the Schur matrix

    S = N P_T N*.

Given a divergence-feasible preflux `p0`, the formal capacity correction is

    delta p = P_T N* S^+ (1 - |p0|).

The Green tensor of `P_T` builds `S` explicitly, so the experiment can
separate mathematical structure from iterative-solver behavior.

### What the Schur block looks like

At pass 4 and 32 squared:

| field | active | components | rank | condition | rank for 99% energy |
|---|---:|---:|---:|---:|---:|
| Cameraman | 362 (35.4%) | 8 | 362/362 | `1.4e3` | 279 |
| synthetic | 216 (21.1%) | 7 | 216/216 | `2.8e3` | 177 |

It is not low rank. It is, however, spatially concentrated: `98.6-99.0%`
of Frobenius energy is within one pixel and `99.6-99.7%` within two.
Interactions between distinct overload components still carry `3-9%` of
the norm, and the largest component contains most active pixels.

The locality does not create a causal ordering. In a strongest-first finite
capacity-event sweep, `89-93%` of already accepted pixels are overloaded
again by later exact Green responses. A radius-2 truncation breaks
divergence conservation and performs worse. The symmetric transverse
projector is reciprocal, unlike a first-arrival predecessor graph.

### Why freezing directions also fails

The normal-hyperplane Schur solve satisfies its linear equations to
`1e-12`, but its weak modes permit enormous tangential motion: flux norms
reach `42-50`, and `66-88%` of pixels remain overloaded. Enforcing the full
two-component condition `p_i=n_i` removes that tangent escape but makes the
system rank deficient:

| field | vector constraints | rank |
|---|---:|---:|
| Cameraman | 724 | 535 |
| synthetic | 432 | 353 |

The direction residual remains order one. Early disk directions cannot all
coexist with the requested divergence; active directions must rotate
together with the capacity set.

A diagonal self-coupling inverse plus one analytical Rayleigh gain is stable
and sometimes lowers the Hodge objective slightly, but it does not advance
another equivalent pass and costs another transverse projection. It is not
a speed result.

### The mask is not a small Fourier convolution

The final possible diagonalization is spectral sparsity of the active tangent
tensor `M(x)`. It is also absent generically. At 128 squared, the largest
1024 Fourier coefficients retain only `57%` of Cameraman tensor energy and
`59%` of synthetic energy; a centered radius-16 square retains about `50%`.
The bandwidth grows with image size because the spatial mask has edges.

The conclusion is precise: Fourier eliminates the longitudinal block, but
the remaining generic disk-mask coupling is full-rank, spectrally broad,
reciprocal, and nonlinear in its angles. It admits a direct closure only for
special masks (constant, one-dimensional, or genuinely band-limited).
For ordinary images, closing all inner passes requires either repeated
active-angle updates or a global masked elliptic solve. The one-shot Hodge
drop of §12 is therefore the maximal analytical Fourier reduction found
without reintroducing an iterative solver.

## 14. Native one-shot engineering result

The §12 closure is now implemented inside the native periodic ROF engine as
an opt-in path. It reuses the plan's spectral symbol and FFT workspaces,
allocates three additional image planes only when requested, and leaves the
ordinary ROF, moving Meyer alternation, ladder, FACR, and Neumann paths
unchanged.

The state transition after an accepted proposal is important. The projected
flux `p_hat` is installed as `b = p_hat/eta`, the accelerated primal is
installed as `u`, and the reflected field is rebuilt as
`d-b = grad(u)-b`. Ordinary Split Bregman then continues from a consistent
state toward the original target. A rejected proposal mutates none of those
live fields; the constant-image rejection test is bit-identical to plain ROF.

An independent NumPy construction and the native output after the closure
agree to relative maximum error `7.4e-16`. The compiled tests also establish:

- strict objective and reference-error improvement at an early fixed budget;
- agreement with the ordinary high-precision target after continued sweeps;
- bit-identical output for one and four worker lanes;
- exact no-op behavior on rejection;
- correct early-stop diagnostics when tolerance fires before the closure;
- explicit rejection on FACR and Neumann plans.

The performance result is narrower than the mathematical result. On the
current native benchmark, the closure costs about two ordinary sweeps: it
contains one FFT pair plus the spatial projection, exact Taylor reduction,
objective check, and state re-seat. At 128 squared it advances the early
trajectory enough to reduce both objective and reference error at eight
sweeps. On a deterministic 512-squared field, a pass-4 proposal is accepted
but is too weak to repay its cost, and continued pass-8 output can be worse
than the ordinary pass-8 trajectory. At `1e-5` stopping tolerances the two
paths reach the same accuracy regime with essentially the same ordinary
sweep count, leaving the closure overhead exposed.

Therefore the accelerator is shipped as an explicit `hodge_after` choice,
not silently baked into every ROF call. This is the serious boundary:

1. the one-shot is analytically valid, objective-safe at insertion, and
   high-precision target preserving;
2. it can be a low-budget accelerator on favorable fields;
3. it is not a universal wall-time accelerator, and no present content-free
   insertion rule makes it one;
4. integration into a production hot path should be gated by
   `examples/meyer_hodge_benchmark.cpp` on that workload.

The implementation keeps the useful primitive available without converting
a conditional numerical win into a default performance regression.

## 15. Finite active-angle terms

`experiments/cartoon_fourier_angle_terms.py` tests whether the remaining
capacity coupling is a short analytical series rather than a call for an
iterative masked solver.

Let `p0` have the requested divergence `q=c(u-g)`. On the current overloaded
set write `p0=m n`, let `N` and `T` sample normal and tangent components, and
let

    S = N P_T N*,       R = T P_T N*.

For fixed-normal transverse sources, `r=S lambda` and `t=R lambda`. Exact
unit capacity requires

    m + r = sqrt(1-t^2).

Introducing a formal overload scale and
`lambda=sum_k epsilon^k lambda_k` produces genuine finite curvature terms:

    S lambda_1 = 1-m
    S lambda_2 = -t_1^2/2
    S lambda_3 = -t_1 t_2
    S lambda_4 = -t_1 t_3 - t_2^2/2 - t_1^4/8.

These are not Runge-Kutta stages or a tolerance loop. They are the explicit
quadratic, cubic, and quartic coefficients of the active-angle equation.

### Early terms are outside their convergence radius

At pass 4 and 32 squared, the exact first Schur term satisfies its normal
equation to `3e-13--6e-13`, but its tangent response is already beyond the
real square-root radius:

| field | active pixels with `|t_1|>=1` | median `|t_1|` | max `|t_1|` |
|---|---:|---:|---:|
| Cameraman | 76.8% | 2.69 | 34.5 |
| synthetic | 54.6% | 1.20 | 29.2 |

The additional coefficients therefore explode. On Cameraman their norms are
`5.1e3, 3.8e5, 6.1e7, 1.2e10`; the active capacity RMS grows from `7.1` to
`1.9e7`. Synthetic behaves the same way. Spectrally truncating enough weak
Schur modes to keep `max |t|<1` leaves `99.2-99.6%` relative residual in the
normal equation. The weak modes are simultaneously what closes capacity and
what rotates the field beyond the local series chart.

### Late terms converge locally but change the mask

At synthetic pass 24, the first tangent response has maximum `0.438`, so the
series is valid. Orders one through four reduce the original active-set
capacity RMS from `9.6e-3` to `2.3e-3`, and the coefficient norms decrease
`17.7 -> 3.1 -> 1.0 -> 0.46`. Yet `26.5%` of all pixels remain overloaded
and the maximum norm grows to `1.89`: satisfying the original mask births a
different mask. At that stage every objective-checked primal proposal is
rejected anyway; the ordinary solver is already beyond the useful Hodge
window.

### Exact angle refresh is stronger and still saturates

As a bound on any truncated local expansion, the experiment also composes
the two exact nonlinear projections a fixed number of times:

    p_(k+1) = P_disk(P_div=q(p_k)).

Every term refreshes all angles with the exact disk map and costs one
additional Poisson FFT pair. At pass 4:

| field/size | term-1 equivalent pass | term-8 equivalent pass | extra objective gain |
|---|---:|---:|---:|
| Cameraman 32 | 8 | 8 | 3.9% of first drop |
| synthetic 32 | 7 | 7 | 6.9% of first drop |
| Cameraman 128 | 7 | 7 | 7.3% of first drop |
| synthetic 128 | 7 | 7 | 0.2% of first drop |

No case gains another equivalent pass. Each added term has essentially the
same transform structure as the native one-shot, already measured near two
ordinary sweeps, so the cost/effectiveness ratio worsens immediately.

### The fixed-current constraints do not intersect

The decisive diagnostic solves the convex oracle

    min_|p_i|<=1  1/2 ||div(p)-q||^2.

Projected-gradient KKT fixed-point errors are `1e-8--6e-8`, but the minimum
relative residual at pass 4 remains nonzero:

| field/size | minimum relative divergence residual |
|---|---:|
| Cameraman 32 | 0.280 |
| synthetic 32 | 0.265 |
| Cameraman 128 | 0.253 |
| synthetic 128 | 0.101 |

Thus no active-angle formula, regardless of how many terms it contains, can
produce a unit-disk flux with the current requested divergence. The current
primal must move at the same time. This is why the one-shot's final
objective-checked primal segment is essential and why “closing the remaining
inner passes” is not a missing higher-order identity.

The finite-term conclusion is therefore:

1. the additional analytical terms can be written explicitly;
2. in the useful early window they lie outside their convergence radius;
3. once they converge, active-set births remain and the proposal has no
   objective descent left;
4. exact angle refreshes add only marginal progress at one FFT pair apiece;
5. further coupled updates would be an unrolled nonlinear solver, not a
   finite analytical closure.

## 16. Hodge-motion-Hodge cycling

The negative moving-frame result in §12 applied a closure on every outer
pass indefinitely. That conflated two questions:

1. can target motion restore a negative Hodge direction?
2. should the closure continue after that restored direction becomes weak?

`experiments/cartoon_hodge_motion_cycle.py` separates them. It preserves the
native Meyer penalties and warm states,

    eta_u = 2 lambda,       eta_w = 10/mu,

and re-seats an accepted Hodge state exactly as the native static path does:
`b=p_hat/eta`, `d=grad(u_hat)`. The opposite ROF step then moves the target
before the next shot.

### Motion really does restore the direction

For additive cartoon-side cycles at 128 squared, a shot after outer pass 4 is
followed by the texture-survivor update, which changes the next cartoon
target. Consecutive shots remain accepted:

- Cameraman pass-4 through pass-7 alphas:
  `0.51, 0.78, 0.59, 0.38`;
- synthetic pass-4 through pass-7 alphas:
  `0.75, 1.00, 0.99, 0.90`.

The later shots have nonzero measured target motion and independently lower
their current ROF objectives. Thus the proposed cycle is not equivalent to
repeating Hodge against a frozen target. Motion genuinely reacquires the
longitudinal inconsistency.

### The winning schedule is a finite burst

Perpetual schedules still lose: alpha and objective gain collapse after the
early window, while every attempted closure pays its transform cost.
Cost-grid search selects:

1. run two to four ordinary Meyer passes;
2. for the next four to six passes, perform the ordinary cartoon sweep and
   one Hodge closure before the texture motion;
3. stop Hodge entirely and continue the ordinary alternation.

Replacing the cartoon sweep with Hodge, closing only the texture-survivor,
closing both sides, and alternating sides were all dominated. The ordinary
cartoon sweep supplies a useful local branch update; Hodge then takes the
larger longitudinal drop; the texture step moves the target for the next
cycle.

### Equal-cost result

One ordinary Split-Bregman sweep is one cost unit and the native Hodge shot
is charged conservatively as two. Error is the joint
`sqrt(||u-u_ref||^2+||v-v_ref||^2)/||f||` against a 4096-outer baseline
reference for the 128-squared grid and 2048-outer references for the scale
sweep.

At 128 squared:

| sweep-equivalent budget | Cameraman cycle/baseline | synthetic cycle/baseline |
|---:|---:|---:|
| 32 | 0.976 | 0.929 |
| 64 | 0.938 | 0.895 |
| 128 | 0.907 | 0.839 |
| 256 | 0.894 | 0.798 |

The cycle pays for three to six Hodge shots by giving up the same number of
ordinary outer passes and still finishes closer to the common target.

The effect is scale- and content-dependent but persists:

| size | Cameraman ratio at budget 128 / 256 | synthetic ratio at budget 128 / 256 |
|---:|---:|---:|
| 64 | 0.771 / 0.742 | 0.654 / 0.550 |
| 128 | 0.907 / 0.894 | 0.839 / 0.798 |
| 256 | 0.931 / 0.903 | 0.886 / 0.879 |

It also survives a higher cost charge. With each Hodge shot priced at `2.5`
sweeps, the 128-squared ratios at budgets 128/256 are `0.928/0.900` on
Cameraman and `0.859/0.808` on synthetic. At price `3.0`, the short
Cameraman budget can lose, but the longer budgets remain positive.

### Precision boundary

The burst does not change the fixed point. After its final shot, execution is
the unmodified warm Meyer alternation, and the cycle/baseline state difference
shrinks under continued passes. On Cameraman the early advantage is eventually
consumed at extreme precision by the six outer passes displaced by Hodge; on
synthetic the transported advantage persists much longer. This is an
accelerated trajectory to the same split, not a new decomposition.

The resulting native candidate is narrow and testable: an opt-in
cartoon-side burst described by `(start_pass, shot_count)`, initially
`(4,4)` for conservative budgets or `(2,6)` for aggressive ones. Native
integration must additionally refresh `u_spec` after an accepted drop before
the texture solve; that extra forward transform is why the `2.5`-sweep cost
sensitivity matters. No production integration is justified for an
unbounded cadence.
