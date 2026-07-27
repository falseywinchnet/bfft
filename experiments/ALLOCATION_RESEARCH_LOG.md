# BFFT cell-allocation research log

Control: the recovered robust allocator checkpoint—uniform blue-noise
initialization, integrated robust residual, clearance reward, edge-crowding
penalty, and measured weak-site exchanges.

The recursive residual is computed for **one stage only**. It is placement
metadata, never a source of reconstructed pixel values.

## Protocol

- Images: Cameraman, Coins, Grass, Chelsea.
- Main screen: 96 px maximum side, 96 initial cells, 420 final cells.
- Validation screen: 128 px maximum side, 700 final cells.
- Three measured exchanges after reaching the cell ceiling.
- Measurements, all lower-is-better:
  - sRGB MSE;
  - one-stage BFFT cartoon MSE;
  - one-stage BFFT texture MSE.
- Reproduction:

  `PYTHONPATH=.:viewer python experiments/allocation_research.py`

## Idea ledger

### Uniform robust control — retained

This remains the strongest single general-purpose allocator. The experiments
below do not justify changing its nucleation or refinement pressure.

### Continuous one-stage residual nucleation — rejected

Every initial blue-noise choice was weighted by residual focus.

- Slight Cameraman gain at one scale.
- Worse on Coins, Grass, and Chelsea.
- Cartoon MSE often rose by an order of magnitude.

**Why:** focus is not missing-mass density. Continuous weighting removes the
uniform spatial coat and spends too many initial degrees of freedom in regions
that later robust-error refinement already discovers.

### Residual focus quota, 25% and 10% — rejected

Uniform foundation plus a fixed fraction of residual-focused anchors.

- 25% helped Grass and Chelsea at 96 px but hurt Cameraman and Coins.
- Reducing the quota to 10% did not make it general.

**Why:** a fixed quota is unrelated to the concentration, topology, or
reducibility of the residual. A focus map can identify attention without
specifying how much budget that attention deserves.

### Alternating residual/uniform nucleation — rejected

Every fourth initial decision used residual focus; the others repaired
coverage.

- Strong Chelsea gain.
- Worse on the other three images.

**Why:** temporal interleaving does not repair the underlying budget problem;
the sequence is still imposing a scene-independent focus quota.

### Bounded local yield toward residual mass — rejected

Start uniformly, then move each site no more than half a spacing toward
residual mass inside its own Euclidean cell.

- Worse on all four images.

**Why:** even small centroid motion changes the downstream geodesic partition.
The residual is a focus indicator, but its centroid is not necessarily the
best generator for an affine approximant.

### Unit-vector spin lift and Gaussian rounding — rejected, informative

The initial focus candidates formed a signed graph: distant candidates were
treated as complementary and close candidates as redundant. Signs were lifted
to unit vectors with the supplied construction, Gaussian-hyperplane rounded,
and truncated to a fixed-size subset.

- At 64 px Cameraman: 27.37 → 27.93 dB.
- At 96 px: worse RGB quality on all four images.
- The rounded subset beat the graph's greedy objective on 3/4 images while
  still losing reconstruction quality.

**Why:** the relaxation and rounding worked on the objective supplied to
them. The signed graph was wrong: geometric diversity is not equivalent to
complementary *reducible approximation error*. A future spin formulation
needs pairwise entries estimated from actual joint refits, not distance.

This agrees with the role of vector relaxations and hyperplane rounding in
[Goemans–Williamson](https://math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf):
rounding preserves a graph objective; it cannot correct a misspecified graph.

### Mild decomposition-discrepancy pressure — rejected

The current composition's single-stage cartoon/texture discrepancy received a
small refinement weight.

- Worse aggregate RGB, cartoon, and texture error.

**Why:** persistent decomposition mismatch is not necessarily reducible by an
additional site. This repeats the original edge-lock lesson in a richer
feature space.

### Bounded quadratic cell surfaces — rejected

The exact control sites were retained; affine planes were replaced with
bounded quadratic local color surfaces.

- Lost 0.8–2.6 dB on all images.

**Why:** a quadratic fitted inside one owned region extrapolates poorly when
the partition of unity evaluates it across a neighboring region. Bounding
stops explosions but introduces biased surfaces. Affine patches are unusually
stable under overlap.

### Hard ownership, sharp overlap — rejected

Both reduce quality broadly. Cell transitions need a partition of unity.

### Wider overlap — promising but scene-dependent

Softness 4 improved RGB and texture error on Coins, Grass, and Chelsea, but
slightly worsened Cameraman RGB error. Softness 6 was a useful middle branch.

This is consistent with centroidal Voronoi approximation and partition-of-
unity intuition: smooth overlap helps where neighboring affine predictions
agree, but can mix across true discontinuities. See
[Du, Faber, and Gunzburger](https://epubs.siam.org/doi/10.1137/S0036144599352836)
for the resource-allocation interpretation of centroidal Voronoi cells.

### Three-branch overlap objective — supported candidate

Run softness 4, 6, and 10; choose the branch minimizing

`RGB MSE + cartoon MSE + texture MSE`.

At 128 px / 700 cells:

| Image | Control objective | Selected objective | Change | Selected softness |
|---|---:|---:|---:|---:|
| Cameraman | 0.0026229 | 0.0022459 | −14.4% | 6 |
| Coins | 0.0027654 | 0.0024701 | −10.7% | 4 |
| Grass | 0.0083984 | 0.0083984 | 0.0% | 10 |
| Chelsea | 0.0012157 | 0.0009318 | −23.4% | 6 |

The baseline is one branch, so this search cannot worsen the stated combined
objective. It is not yet a realtime proposal: it is evidence that overlap is
a scene-level latent parameter and should be optimized from measured
decomposition error rather than guessed from a local edge heuristic.

## Current understanding

1. One-stage residual is useful as a **focus display**, but no tested direct
   conversion from focus to nucleation budget is general.
2. The robust allocator already extracts most useful placement information
   from measured reconstruction error.
3. Decomposition discrepancy is valuable as an **evaluation objective**, but
   directly turning it into spatial pressure confuses persistent with
   reducible error.
4. The present supported improvement is objective-selected overlap.
5. The next mathematically clean spin experiment should build its signed
   matrix from measured pairwise refit gains:

   `a_ij = sign(ΔJ(i,j) − ΔJ(i) − ΔJ(j))`.

   Positive entries would then mean actual complementary reduction of the
   three-term objective; negative entries would mean actual redundancy.

## 2026-07-26: reducibility and coupled-fit round

Pikachu (`~/Downloads/25.png`) was added as the primary clean-geometry
control. The guard set remains Cameraman, Coins, Grass, and Chelsea.

### Expected affine gain — retained

Refinement pressure now estimates the closed-form gain of a
Gaussian-windowed constant-plus-gradient correction at the current cell
spacing:

`gain = mean(residual)^2 + sigma^2 * |gradient(residual)|^2`.

It replaces residual magnitude as the allocation currency while retaining
the proven clearance reward and repeated-edge penalty.

At 128 px / 700 cells:

| Image | Robust control | Expected gain |
|---|---:|---:|
| Pikachu | 21.92 dB | **23.88 dB** |
| Cameraman | 27.79 dB | **28.55 dB** |
| Coins | 27.36 dB | **27.38 dB** |
| Grass | 23.29 dB | **23.38 dB** |
| Chelsea | 31.76 dB | **31.80 dB** |

At 256 px / 2,400 cells, Pikachu reaches 26.33 dB before the coupled fit.

**Why it works:** it estimates error the model can remove. A large
high-frequency residual and a large affine-correctable residual no longer
have the same price.

### Reliability-weighted overlap — rejected

Neighbor predictions were blended by inverse OLS predictive variance,
including leverage.

- Lost quality on every guard.
- Cameraman fell from 27.79 to 26.19 dB in the first screen.

**Why:** neighboring cells across a real boundary are not noisy unbiased
estimates of one latent surface. Reliability weighting is correct for
combining estimates of the same signal, but here it confidently mixes
different signals.

### Decomposition expected gain — complementary, not default

Expected gain from signed single-stage cartoon and texture discrepancies was
added to RGB/OKLab expected gain.

- It wins the combined three-error objective on Coins and Chelsea at
  128 px / 700 cells.
- It loses to RGB expected gain on Pikachu, Cameraman, and Grass.
- At 256 px / 2,400 Pikachu cells it reaches 26.80 dB and improves all three
  reported errors over RGB-only expected gain in that run.

A smooth budget schedule was also tested and rejected: the right mixture is
scene- and budget-dependent, not a universal function of cell count. The
viewer therefore exposes both currencies rather than silently blending them.

### Cell-level reducibility gate — rejected

Each cell was weighted by:

`integrated expected gain / integrated residual energy`.

Grass improved only 23.38 → 23.40 dB, while structured controls regressed.

**Why:** the ratio correctly identifies locally affine residual, but the
representation must still spend cells on non-affine texture to improve the
pixel and texture objectives. "Irreducible by one affine correction" does
not mean "safe to ignore."

### Robust-to-expected switch schedules — rejected

Because expected gain loses to robust integrated error on the very small
96 px / 420-cell Pikachu screen, switches after 25% and 50% of the refinement
budget were tested.

- The 25% switch slightly wins Cameraman, Grass, and Chelsea at 128/700.
- It loses substantially on Pikachu and Coins.
- The 50% switch also fails to dominate either endpoint.

**Why:** path dependence matters. Early robust placements cannot be
recovered merely by changing the later pressure field. The right answer is
measured branch selection or a real split/merge market, not a universal
clock.

### Absolute-Hessian support metric — rejected in this geometry

The texture structure-tensor part of the metric was replaced by the absolute
Hessian of the BFFT cartoon plus flow support.

- Reduced cartoon error on Coins.
- Lowered RGB PSNR on every guard.
- Pikachu fell from 23.88 to 22.59 dB.

**Why:** curvature is model-matched for affine approximation, but the
existing BFFT metric also performs border ownership. Replacing its
first-order texture geometry loses useful barriers. A future Hessian term
should affect density/equidistribution without replacing transport topology.

### Geometry-nominated dipole splits — rejected

At a contour-crossing residual, two children were proposed on opposite sides
of the BFFT normal. Both retaining and replacing the parent were tested.

- Slight Chelsea benefit.
- Broad regressions; Pikachu fell to about 22.6 dB.

**Why:** a plausible pair is not necessarily worth two units of budget.
Dipoles require a measured pair gain and merge price, not a geometric
trigger alone.

### Joint partition-of-unity solve — retained

The previous renderer fitted every cell independently to its hard territory,
then blended it with a neighbor it had never been optimized alongside. With
sites and ownership fixed, the new solve constructs the actual sparse
two-cell partition-of-unity design matrix and solves all affine planes
jointly.

At 128 px / 700 cells:

| Image | Expected-gain local fit | Joint fit |
|---|---:|---:|
| Pikachu | 23.88 dB | **26.26 dB** |
| Cameraman | 28.55 dB | **29.44 dB** |
| Coins | 27.38 dB | **27.76 dB** |
| Grass | 23.38 dB | **23.44 dB** |
| Chelsea | 31.80 dB | **32.30 dB** |

This isolates a major failure: allocation was not the only bottleneck.
Independent fitting was inconsistent with the renderer.

### Multiscale coupled solve — retained

Cartoon and texture are jointly solved on the same sites but with genuinely
different partitions: broad overlap for cartoon (softness 4) and sharp
overlap for texture (softness 16).

- Pikachu at 128 px / 700: **26.41 dB**.
- Pikachu at 256 px / 1,200: **25.86 dB**.
- Pikachu at 256 px / 2,400: **28.34 dB**, RGB MSE `1.466e-3`,
  cartoon MSE `2.614e-5`, texture MSE `5.235e-4`.

The multiscale form wins all three Pikachu measurements over both the local
fit and the single-partition coupled fit. Grass and Chelsea still prefer the
single-partition joint solve, so the two support widths remain a visible
research choice rather than a claim of universal optimality.

### Acceleration

The exact two-label geodesic assignment was moved from the Python heap loop
to a cached Numba implementation with a Python fallback.

- 192 px / 1,200-cell expected-gain run: about 10.9 s → 2.5–2.8 s.
- Reconstruction and PSNR were unchanged in the comparison.
- The full regression passes through the compiled path.

The speedup makes high-budget controls and branch falsification practical;
it does not alter the model.

## Updated understanding

1. Residual memory remains a one-stage focus diagnostic, not allocation
   currency.
2. Expected removable affine gain is the strongest general allocator tested.
3. BFFT geometry is most useful as transport support and border topology;
   model-order geometry should supplement it, not replace it.
4. The largest quality gain comes from making fitting agree with the
   partition-of-unity renderer.
5. Cartoon and texture benefit from different overlap scales, but the best
   scale remains scene-dependent.
6. Pair births and merge deaths must be compared in one measured objective
   before either is safe.
