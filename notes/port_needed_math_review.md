# port_needed: formal math review

One port at a time.  For each: what the kernel actually computes, whether a
better closed form or a provable shortcut exists, and the measured result.
Every claim below is checked by `experiments/port_math_review.py` against the
reference implementation on real geometry, and is written here only after that
script prints `EXACT` or a measured bound.

    PYTHONPATH=.:viewer:experiments .venv/bin/python experiments/port_math_review.py

Nothing in the shipped path was edited.  These are findings for the native
port, and each one is independently adoptable.

Status: **01, 02, 03, 04, 05, 07 reviewed.**  06 and 08 structurally read but not yet verified.

---

## PORT 01 — `frozen_meyer_geometry` : the tensor rebuild needs no angle

### What it computes

After the raw structure tensor `Q` is formed, `single_decomposition_geometry`
(lines 261–277) reshapes its spectrum so that a coherent contour stays
expensive across its normal and becomes cheap along its tangent.  It does this
by an explicit eigendecomposition:

    trace = qxx + qyy
    disc  = hypot(qxx - qyy, 2 qxy)
    high  = max(0.5(trace + disc), floor)
    low0  = max(0.5(trace - disc), floor)
    low   = floor + (1 - (1 - f) * disc/trace) * (low0 - floor)
    angle = 0.5 * atan2(2 qxy, qxx - qyy)      # <- transcendental
    n = (cos angle, sin angle),  t = (-n_y, n_x)
    Q' = high * n n^T + low * t t^T

That is `atan2`, `cos` and `sin` on every pixel, plus the outer products.

### The shortcut

The eigenprojectors of a symmetric 2x2 matrix are rational in the matrix
itself.  With `h0 = 0.5(trace + disc)` and `l0 = 0.5(trace - disc)` the exact
eigenvalues,

    n n^T = (Q - l0 I) / disc          t t^T = (h0 I - Q) / disc

Substituting into the rebuild and collecting terms:

    Q' = alpha Q + beta I
    alpha = (high - low) / disc
    beta  = (low * h0 - high * l0) / disc

The whole reshape is **one axpy on the tensor**.  No angle, no trigonometry, no
eigenvector ever formed.  The degenerate branch is clean: as `disc -> 0` the
pixel is isotropic, `coherence -> 0` forces `low -> l0`, and the map becomes
`Q' = Q`, so guarding `disc < eps` with `(alpha, beta) = (1, 0)` is not a
special case bolted on but the actual limit.

Note this is the same identity that makes `_unstable_direction` (lines 338–368)
work without a solver, applied one level up.  The codebase already knows the
trick; PORT 01 just does not use it.

### Measured

| | |
|---|---|
| agreement with the eigenvector rebuild | **exact**, 3.4e-16 relative |
| speed, 192x256 | 1.33–1.66 ms -> 0.32–0.36 ms, **3.7–5.1x** |

Three transcendentals per pixel become four flops.  For a native port this also
removes the only libm dependency in the kernel, which matters more for SIMD
than the flop count does: `atan2`/`cos`/`sin` are the reason this loop cannot
be vectorised as written.

### Second finding: hoist the metric scale

`_edge_cost_stack` recomputes `percentile(qxx + qyy, 90)` on every call, but
that quantity depends only on the frozen geometry, never on `metric_strength`.
`metric_strength` is a viewer slider.  So every slider movement currently
re-partitions the whole image to recover a constant.  It belongs in PORT 01's
output dict.  Measured: 14% of the stencil's runtime.

---

## PORT 02 — `anisotropic_edge_cost` : half the stack is redundant

### What it computes

Eight direction planes of `sqrt(d^T A_d d)` where `A_d` is the average of
`M = I + c Q` at the two endpoints, `c = strength * horizon^2 / scale`.

### Three findings, all exact

**1. The stack is exactly symmetric, so half of it is duplicated.**
`cost[k][p]` averages `M` over `{p, p+d}`; `cost[opp k][p+d]` averages `M` over
`{p+d, p}` and its quadratic form is invariant under `d -> -d` because every
term is quadratic in the components.  Measured max difference over all eight
planes: **0.0, bit-identical**.  A native port stores four planes — N, W, NW,
NE — and reads the opposite direction at an offset pixel.  Compute and memory
both halve: 1.6 MB -> 0.8 MB at 192x256, and 32 MB -> 16 MB at 1024x1024, on a
kernel the port notes describe as bandwidth-bound.

**2. The `max(..., 1e-8)` floor is dead code.**  `Q` is built as
`high n n^T + low t t^T` with both eigenvalues floored at `frequency_floor > 0`,
so `Q` is positive definite, so `M = I + cQ >= I` for `c >= 0`, so
`d^T A_d d >= |d|^2 >= 1` for all eight unit steps.  Measured minimum eigenvalue
of `Q` is 4.7e-4 and the measured minimum cost is 2.227 — the floor is three
orders of magnitude from ever binding.  Removing it lets the port use an
unguarded `sqrt` or `rsqrt`.

**3. `mxx`, `mxy`, `myy` never need to exist.**  Because `M = I + cQ` is affine
in `Q` with a *scalar* `c`, the quadratic form expands to

    cost = sqrt( |d|^2 + c * (dx^2 avg(qxx) + 2 dx dy avg(qxy) + dy^2 avg(qyy)) )

so the kernel reads `qxx, qxy, qyy` directly and applies one fused multiply-add.
Measured difference from the reference: **0.0, bit-identical**.  This removes
three image-sized temporaries and three passes over memory.

Combined, PORT 02 should be: four planes instead of eight, no `M` temporaries,
no floor, and the scale arriving precomputed from PORT 01.

---

## PORT 03 — `two_label_transport` : the runner-up wave is mostly discarded

This is the dominant port, so it got the most attention.

### What it computes

A monotone two-label Dijkstra on the 8-neighbour metric, returning owner,
runner-up, both distances, and the predecessor forest.  Bucket queue (Dial),
lazy deletion, `s_field`-scaled edges.

### Finding 1 — the second label only matters in a thin band, provably

Every consumer of the runner-up weights it by

    owner_weight = expit(clip(gap / T, 0, 40)),   gap = d2 - d1
    runner_weight = 1 - owner_weight

`expit(40) = 1 - 4.2e-18`, which in float64 rounds to **exactly 1.0**, so
`runner_weight` is **exactly 0.0** wherever `gap > 40 T`.  Every runner
contribution outside that band is multiplied by zero.

So the second-label wave can be stopped at `tau = 40 T`, and this is exact
rather than approximate, because a pruned wave can never re-enter the band:

> For any edge `p -> q` of weight `s`, `d1(q) <= d1(p) + s` since `d1` is the
> distance to the nearest site.  A second-label continuation arrives at `q`
> with value `d2(p) + s`, so its gap there is
> `d2(p) + s - d1(q) >= d2(p) - d1(p)`.
> The gap along a continuation is non-decreasing, so once a wave exceeds `tau`
> it exceeds `tau` at every node downstream. ∎

Pruned pixels end with `runner = -1`, which the consumers already handle as
`valid = runner >= 0` — the same state an unreached runner would produce.

**Verified bit-identical**: owner map equal, `d1` equal to 0.0, and every
consumer-visible quantity (`owner_weight`, `runner_weight`, the safe runner
index where its weight is nonzero) equal to 0.0 difference, at every softness
and cell count tested.

The saving is large because the temperature anneals.  `T = softness * horizon`
with softness geometric from 0.20 to 0.0025, so at the end `tau = 4.6` — about
two pixels of geodesic distance, against a metric whose minimum edge cost is
2.23.  Fraction of pixels with a consumer-visible runner at softness 0.0025:
**4.4% at 64 cells, 8.3% at 256, 14.4% at 1024**.  The other 85–95% of the
second-label walk is computed and then multiplied by zero.

### Finding 2 — the scale field is identically one

Line 156 of the walk computes `base_costs[d, y, x] * 0.5 * (s_field[p] +
s_field[q])`, but both shipped entry points (`walk_two_labels`,
`hard_partition_with_forest`) pass `scale = np.ones(...)`.  That is a scattered
gather to `s_field[q]` plus two flops on the hottest edge in the program, to
multiply by one.  A specialised unit-scale kernel removes it.  The general path
must stay for the sigma round's learned barrier field, so this is a second
entry point, not a deletion.

### Measured, both findings together

Full 24-round annealing schedule, 192x256, cells growing 8 -> 4096:

| round | softness | tau | reference | pruned | |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2000 | 368.6 | 12.8 ms | 7.9 ms | 1.62x |
| 8 | 0.0436 | 80.3 | 15.2 ms | 10.9 ms | 1.39x |
| 16 | 0.0095 | 17.5 | 18.9 ms | 10.9 ms | 1.74x |
| 23 | 0.0025 | 4.6 | 44.5 ms | 20.1 ms | 2.21x |
| **total** | | | **439 ms** | **266 ms** | **1.65x** |

Every round bit-exact on owner and `d1`.

### Finding 3 — the queue stores more than it needs

The reference appends a new entry on every improvement and never removes the
superseded one, relying on the pop-time validity check to skip it.  Measured
queue pushes per pixel: **2.87–3.04**, against a hard bound of **2** for a
queue that relocates the existing `(pixel, label)` entry instead of appending
beside it.  With a doubly-linked bucket list, relocation is O(1), the capacity
becomes exactly `2 * npix` and never reallocates — the reference starts at
`4 * npix + 256` and doubles, copying four arrays inside the compiled kernel
when it does.  About a third of pops are currently stale entries.

### Finding 4 — the bucket modulo

`slot % buckets` and `current % buckets` are integer divisions on the hot path
because `buckets = span + 2` is arbitrary.  Rounding `buckets` up to a power of
two turns both into a mask.  Free; only worth mentioning because it is in the
innermost loop.

### Not a finding: the walk itself

The two-label monotone propagation is correct and is the right algorithm.  No
sweep-based or hierarchical replacement is proposed: fast-sweeping variants
lose exactness on an anisotropic metric with this much contrast, and the port
notes require the refresh to solve the current transport state exactly.  The
gains above are all from removing work that is provably discarded, not from
changing the method.

---

## PORT 04 — `soft_transport_moments` : two sweeps should be one

### What it computes

Twenty `bincount` reductions producing mass, barycentre, the centred second
moments, a transport RMS, and the mass-weighted metric tensor, for the
two-label Gibbs plan.

### The sequential barrier

The reference computes `cx, cy` first, then forms `dx_owner = x - cx[owner]`
and `dx_runner = x - cx[runner]` and reduces again.  That is a hard barrier:
the second sweep cannot start until the first has finished for every cell.
For the intended native form — "one parallel pixel pass into per-thread cell
blocks" — a barrier in the middle is the expensive part, not the flops.

### The shortcut

Every contribution to cell `k` is centred at that same cell's `cx[k]` (the
owner term reduces into `owner` and is centred at `cx[owner]`; the runner term
reduces into `runner` and is centred at `cx[runner]`).  So each cell's
accumulation is an ordinary weighted covariance, and obeys the shifted-data
identity for **any** per-cell reference `r`:

    cxx[k] = S_xx[k]/S_0[k] - (cx[k] - r[k])^2,   S_xx = sum w (x - r[k])^2

Choose `r` = the site position, which is known *before* the pass.  One sweep,
same gather count, no barrier.

**Do not choose `r = 0`.**  Measured: the origin-centred form loses the
covariance to 8.7e-4 relative error — thirteen of sixteen digits — because the
cells are small and far from the origin.  This is the classic catastrophic
cancellation in `E[x^2] - E[x]^2`, and a native port that reaches for raw
moments as the obvious fusion will hit it silently.  The site shift costs
nothing and removes it entirely.

### Measured

| | |
|---|---|
| mass | exact, 0.0 |
| barycentre | 2.5e-15 relative |
| covariance, site-shifted one pass | 6.6e-15 relative |
| covariance, origin-centred | 8.7e-4 relative — **unusable** |
| time, 512 cells | 5.4–5.7 ms -> 3.4 ms |

### Composition with PORT 03

After the PORT 03 prune only **10.4%** of pixels carry a nonzero runner weight.
The runner half of all twenty reductions is then a predicated add over that
minority rather than a full second pass over the image.  The two ports
reinforce each other; they should be ported together.

---

## PORT 05 — `metric_instability` : correct kernel, Python loop, and a
## direction that does not match its own docstring

### Finding 1 — the wrapper

`measure_instability` calls the scalar `_unstable_direction` in a Python `for`
loop over cells, with `safety_cells` defaulting to 32768.  The kernel is pure
elementwise arithmetic; the vectorised form agrees to **1 ulp** (eigenvalues
bit-identical, eigenvector within 2.2e-16) and is **65x faster** at 8192 cells
(12.5 ms -> 0.19 ms).  No math changes.  This is the cheapest win in the queue
and it is pure wrapper.

### Finding 2 — CQ or QC

The docstring says "largest eigenpair of `C Q`; eigenvalues match
`Q^(1/2) C Q^(1/2)`".  The eigenvalue half is right: `CQ`, `QC` and
`Q^(1/2) C Q^(1/2)` are all similar, so they share a spectrum.

The eigenvectors do not transport the same way.  If `A u = lambda u` with
`A = Q^(1/2) C Q^(1/2)`, then

    QC (Q^(1/2) u) = lambda (Q^(1/2) u)        so QC has eigenvector Q^(+1/2) u
    CQ (Q^(-1/2) u) = lambda (Q^(-1/2) u)      so CQ has eigenvector Q^(-1/2) u

The code builds `CQ` (verified against the source: `a = cxx qxx + cxy qxy` is
`(CQ)_00`) and returns its eigenvector.  The variational problem whose
spectrum is that of `Q^(1/2) C Q^(1/2)` is

    maximise v^T C v subject to v^T Q^(-1) v = 1

whose maximiser is the eigenvector of `QC`, not of `CQ`.  The two differ by a
factor of `Q`, which for a coherent contour is exactly where the tensor is most
anisotropic — precisely the cells the instability test exists to find.

This is flagged, not fixed.  It may be deliberate: the returned direction is
used to place two child branches, and `CQ`'s eigenvector is the natural
"direction of shape mismatch in image space" while `QC`'s is the same object
pushed through the metric.  But the docstring asserts a variational
characterisation that the returned vector does not satisfy, so one of the two
should change.  Deciding it needs an A/B on split quality through the full
bifurcation loop, which is a separate experiment.

---

## PORT 07 — `hard_region_fit` : the affine basis is in the wrong frame

### What it computes

Per hard region, a 3x3 normal system for an affine Lab jet in the basis
`[1, x, y]` with `x, y` **global image coordinates** in `[-0.5, 0.5]`, plus a
fixed ridge `diag(1e-7, 1e-5, 1e-5) * count`.

### Findings

**1. Assemble in the cell's own frame.**  An affine fit is invariant under a
change of basis, so `[1, (x-cx)/s, (y-cy)/s]` fits the same function.  What
changes is the matrix being inverted.  Shifting makes the first row and column
vanish identically — `sum(x - cx) = 0` by definition of the barycentre,
measured exactly 0.0 — so the 3x3 decouples into a 1x1 (the mean) and a 2x2
(the slopes) with a closed-form inverse, and no LU is needed per cell.

**2. Shifting alone does not condition it; scaling does.**  Measured
condition numbers over 600 regions:

| basis | median | worst |
|---|---:|---:|
| image origin (shipped) | 1.67e4 | 2.42e5 |
| shift only | 1.35e4 | 2.05e5 |
| shift + scale | **4.00** | **4.34e1** |

**4179x** better median conditioning.  My first hypothesis — that the offset
was the problem — was wrong, and the measurement said so: what dominates is
the disparity between the count entry (`n`) and the second-moment entries
(`n r^2` with `r` a small fraction of the image).  Shift fixes the structure,
scale fixes the conditioning, and both are free.

**3. The shipped ridge is position-dependent, which is a modelling artefact.**
The ridge adds `1e-5 * count` to a diagonal entry that is `sum(x^2)` in the
image-origin basis.  That sum is small for cells near the image centre and
large for cells at the periphery, so the ridge's strength *relative to what it
damps* varies by **1651x** across cells: p1 4.2e-5, median 1.6e-4, p99 6.9e-2.
Cells near the centre have their fitted slopes damped by up to 7% while
peripheral cells are essentially unregularised — a bias with no modelling
justification, purely an artefact of where the coordinate origin was put.  In
the shift+scale basis the same constant spreads only **12x**, because it is
then relative to a diagonal that is `n` by construction.

**4. Four of nine assembly reductions are redundant.**  The normal matrix is
symmetric and its `(0,0)` entry is the pixel count, which line 1604 already
computes separately.  Five `bincount` calls suffice instead of nine.

---

## PORT 06 and PORT 08 — not yet reviewed

`balanced_refill` (`_balanced_branch_histogram`, ~140 lines) and
`reverse_residual_flow` (~300 lines) have been read structurally but no claim
about them is verified, so nothing is recorded here.  PORT 06 is the one-shot
histogram replacement for fourteen global bisection passes and is the more
interesting of the two: a mass-balanced split along a direction is a weighted
median, and a weighted median over a fixed bin count has an exact one-pass
form, so the question is whether the current two image passes plus a local
scan can become one.  PORT 08 accumulates over the predecessor forest, which
is where the sigma round's Danskin gradient also lives; they should be
reviewed together.
