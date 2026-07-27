# Resource-transport cell rework

Run the current validated experiment with:

```bash
python experiments/resource_transport_cells.py ~/Downloads/25.png \
  --max-side 256 --cells 180 --iterations 30
```

The defaults enable consumption-only germination, conserved residual credit,
adaptive diffuse/crystalline support, learned concentration, and
metabolically gated BFFT shape conductivity.  Every accepted mechanism has a
corresponding `--no-*` or mode control in `--help`.

## Why the present method fails abruptly

The canonical viewer performs four conceptually global acts:

1. assign every pixel an owner and runner-up by a two-label geodesic walk;
2. fit local affine models after that discrete assignment;
3. construct a global allocation-pressure field and rank cells/pixels;
4. introduce sites in batches, then repeat the changed global problem.

At high resolution, initialization pays for BFFT geometry and a full graph
walk before a useful image exists.  More importantly, each birth changes a
discrete partition everywhere.  A difficult image can therefore cross an
ownership chamber and change qualitatively rather than degrade gracefully.
Deletion and merge passes are attempted repairs to that primitive.

## Proposed primitive

There are no owners.  Cell `i` has a smooth compact activity field
`phi_i(x)`, a local color model `p_i(x)`, a center, an area, and a symmetric
shape tensor.  The reconstruction is

```
Z(x)    = epsilon + sum_i phi_i(x)
Ihat(x) = (epsilon p0 + sum_i phi_i(x) p_i(x)) / Z(x)
```

The denominator is not a referee.  It is local occupancy.  Reconstruction
error is the shared resource

```
N(x) = robust_error(I(x), Ihat(x)) / Z(x).
```

Cell `i` absorbs `phi_i N`.  Adding support lowers the resource available to
every overlapping cell.  Competition is therefore starvation of a conserved
local demand, not a rank, deletion decision, or pixel label.

Each round consists only of local splats and reductions:

- local block-Jacobi correction of the cell's color plane;
- movement toward the first moment of absorbed resource;
- area growth or contraction from uptake per unit area;
- shape evolution from the second moment of absorbed flux.

A starved site becomes small but remains a valid state and can regrow.  Hard
deletion has no role.

## The glass state as target-directed conductivity

Let `F` be the cached BFFT transport/glass state and
`v = grad(F) / |grad(F)|`.  Let `E` be the current reconstruction demand.
The flux available to cell growth is

```
D_F = I + kappa c_F v v^T
J   = D_F grad(E)
```

where `c_F` is confidence in the local glass direction.  The flow field does
not itself demand elongation.  It only makes transport cheaper in its own
direction.  If that direction does not carry residual downhill toward the
target, the cell absorbs nothing through it.  Repeated fast directional
uptake increases the principal ratio of the cell shape tensor.

This makes anisotropy a memory of useful transport rather than a shape chosen
from a static edge map.

## Scaling target

Use compact supports whose average overlap is bounded.  Raster work is then

```
sum_i support_area_i = overlap * image_pixels,
```

independent of site count to first order.  No site-pair matrix, global
factorization, pixel sorting, top-k candidate set, or all-pairs search is
required.  A future nucleation field should also be local: residual energy
accumulates, diffuses, and crosses a germination threshold under occupancy
inhibition.  Sites appear simultaneously wherever the local dynamics permit,
not because a controller purchased a fixed batch.

## First isolated control

`experiments/resource_transport_cells.py` implements the fixed-population
core.  It deliberately postpones germination until the support and uptake
dynamics are validated.  Its controls are:

- PSNR and visual character on simple and hard images;
- visits per pixel as cell count and resolution change;
- smooth occupancy without uncovered holes;
- emergence of anisotropy with and without glass conductivity;
- stable, diffuse error rather than discrete ownership failure.

## Experiment ledger

### Rejected: additive glass-to-shape force

The first implementation let target-positive glass alignment add a direct
log-axis contrast.  On Pikachu at 128px/180 cells/12 rounds it reached
16.12 dB, while the identical system with that glass term disabled reached
19.53 dB.  The additive guide was able to distort a support beyond the
descent requested by the reconstruction.

The retained interpretation is narrower: glass may be a positive-definite
preconditioner of a nonzero measured receiver gradient.  It may not add a
growth or anisotropy vector of its own.

The strength also matters.  A conductivity gain of 3.0 still made the finite
local step overshoot and reached only 15.84 dB.  A weak gain of 0.25 reached
19.65 dB, slightly above the 19.53 dB unguided control.  Glass is therefore a
perturbative geometric prior in the present parameterization, not the primary
force.

That small Pikachu win did not transfer.  At 128px/180 cells/12 rounds,
unguided versus a 0.25 discrepancy-guided field measured:

| image | unguided | discrepancy guide |
| --- | ---: | ---: |
| Pikachu | 19.53 dB | 19.65 dB |
| camera | 23.56 dB | 23.26 dB |
| astronaut | 20.48 dB | 20.40 dB |

At that stage the default remained `glass_mode="off"`.  Both fixed-target and
target-minus-current center guides remain reproducible controls; the evidence
does not support enabling either for center motion.

That conclusion applies to **center motion**.  It was retested after residual
conservation and remained non-transferring.

### Accepted: glass as metabolically gated shape conductivity

The useful form is narrower.  For a cell axis `u`, the cached target glass
tensor supplies the nonnegative scalar

```
a_u = u^T (integral positive_usefulness * coherence * v v^T) u.
```

The exact receiver-gradient request `delta log(axis)` is multiplied by
`1 + k a_u`.  Glass therefore cannot create elongation, reverse a descent
direction, or pull a center.  It only lowers resistance along an axis already
requested by the target-directed support gradient.

A fixed conductivity continued to exert leverage after a cell's original
resource was exhausted.  The final form gates `k` by the square root of the
cell's current uptake density divided by the uptake density at which its
shape memory formed.  This is the same metabolic rule that prevented stale
directional memory from deforming an exhausted cell.

At 50 rounds, shape conductivity 0.5 versus no glass improved every control
and every combined structural objective:

| image | PSNR off | PSNR shape glass | objective off | objective shape glass |
| --- | ---: | ---: | ---: | ---: |
| Pikachu | 27.64 | 27.68 | 2.400e-3 | 2.366e-3 |
| camera | 33.14 | 33.31 | 7.287e-4 | 7.212e-4 |
| astronaut | 26.25 | 26.40 | 3.078e-3 | 2.978e-3 |

This is enabled in the experiment default.  Center conductivity remains zero.

## Local germination experiment

The optional germination field is a reaction/diffusion process over the
remaining resource:

```
resource = error / occupancy
drive    = diffuse(resource)
birth    = decay * birth + drive / (1 + local_site_inhibitor)
```

Every location updates simultaneously.  A germ fires when its accumulated
activation crosses a fixed threshold and exceeds the local lateral-inhibition
envelope.  This is implemented by image stencils, not a candidate list:

- no sorting or ranking;
- no top-k or requested batch size;
- no ownership lookup;
- no site deletion;
- no scan over existing sites for a nearest neighbour.

Firing consumes the activator in a diffused neighbourhood.  New cells start
small, underneath existing support, and must grow by reducing measured
reconstruction error.  Existing support enters only as a secreted inhibitor
and as depletion of the shared error resource.

The initial birth prototype sampled the target at the germ center.  On HD
photographs this produced dark or bright beads along high-demand contours
until the newborn plane equilibrated.  The corrected birth is initially
invisible: it inherits the current reconstruction value and its local spatial
gradient.  It therefore spawns underneath the mixture and may paint on top
only after a subsequent receiver-gradient update demonstrates useful error
reduction.

### Directional growth memory

The exact local derivative supplies two log-axis growth requests per cell.
An optional exponentially decayed memory integrates those requests.  A stable
direction accumulates and increases anisotropy; inconsistent requests cancel.
Area does not receive this momentum, so memory cannot inflate a cell merely
because residual persists.  This directly tests the hypothesis that repeated
fast useful transport, rather than a static edge measurement, should create a
long narrow support.

Undamped memory improved early rounds but over-elongated some cells after
their original demand had disappeared.  The revised term is metabolically
gated by each cell's own uptake relative to the uptake present when its memory
formed.  Exhausted cells retain their geometry but cannot spend stale memory
to increase anisotropy.

The long-run multi-image control still did not justify enabling this second
memory state by default.  The cell's current shape already stores successful
directional growth; added momentum is retained only as an experimental
control.

### Adaptive support hardness

Each compact support may optionally learn its own falloff exponent.  The
derivative is exact:

```
phi = (1 - q)^p
d phi / d log(p) = phi * p * log(1 - q).
```

This gives a cell an independent boundary-permeability degree of freedom
without introducing a polygon or owner.  It sharpens only when changing its
normalized overlap reduces receiver error; broad low-detail regions may
remain soft.

Because the raw integral is `pi*a*b/(p+1)`, the implementation multiplies the
support by `(p+1)/(p0+1)`.  This holds integrated support mass fixed: hardness
cannot masquerade as area growth or starvation.  The exact derivative includes
the corresponding `p/(p+1)` normalization term.

The unnormalized version was a misleading partial success: it improved
Pikachu but degraded camera and astronaut because changing boundary hardness
also changed the total resource a cell could absorb.  After mass
normalization, learning hardness improved every 20-round control:

| image | fixed exponent | learned exponent |
| --- | ---: | ---: |
| Pikachu | 23.45 dB | 23.97 dB |
| camera | 24.98 dB | 25.15 dB |
| astronaut | 22.09 dB | 22.31 dB |

The retained learning rate is 0.10.  Faster learning helped the two more
geometric controls but slightly reduced camera's best score.

### Conserved residual credit

The first color update contained a duplication error.  If `K` comparable
supports overlap, each cell solved a local least-squares problem against the
entire residual.  All `K` cells could therefore purchase the same correction
simultaneously.  This is not resource competition.

Let

```
w_i(x) = phi_i(x) / Z(x)
c_i(x) = phi_i(x)^2 / (epsilon^2 + sum_j phi_j(x)^2).
```

The physical effect of a cell's coefficient update remains `w_i`, while its
right to spend the residual is `c_i`.  The color normal matrix is unchanged
and the right hand side becomes

```
H_i = integral w_i^2 b b^T
g_i = integral w_i c_i b residual.
```

Thus identical overlapping cells collectively request one correction:
`sum_i c_i = 1`.  There is still no owner, runner-up, ordering, or pair graph.
The extra denominator is accumulated in the same local splat pass as
occupancy.

At 128 px, 180 initial cells, local germination, learned hardness, and 20
rounds, conserved color credit changed:

| image | duplicate credit | conserved credit |
| --- | ---: | ---: |
| Pikachu | 23.97 dB | 24.76 dB |
| camera | 25.15 dB | 28.10 dB |
| astronaut | 22.31 dB | 23.01 dB |

Camera's alternating score became monotone.  At 50 rounds the conserved
system continued rising to 27.47, 29.90, and 24.50 dB respectively.

#### Rejected: applying the same credit to geometry

Partitioning the center/shape gradient by `c_i` reduced the 20-round scores
to 18.90, 26.19, and 21.34 dB.  Color corrections add and can duplicate the
same purchase.  Geometry is different: every nearby support must perceive
the common boundary field for deformation to propagate.  The exact geometric
receiver gradient remains unpartitioned.

### Germination by consumption alone

The first germination control used both resource depletion and a separate
site-secreted inhibitor.  The explicit inhibitor was redundant.  Occupancy
already lowers `error / Z`; it is the existing cell's consumption of the
available energy.

Removing the extra inhibitor improved all three 30-round controls.  Lowering
the fixed activation threshold from 3.25 to 2.0 allowed persistent
high-frequency residual to nucleate fine cells without a budget or ranked
candidate set.  Camera rose from 28.84 dB / 270 cells to 32.28 dB / 415 cells;
astronaut rose from 23.69 / 270 to 25.21 / 382.  Pikachu remained effectively
flat near 26.3 dB because its broad clean regions did not sustain additional
germs.

The structural objective agrees with pixel PSNR.  Comparing the pre-credit
system to conserved credit, learned hardness, and consumption-only
germination after 30 rounds:

| image | objective before | objective after | cartoon MSE after | texture MSE after |
| --- | ---: | ---: | ---: | ---: |
| Pikachu | 4.722e-3 | 3.344e-3 | 3.567e-4 | 6.345e-4 |
| camera | 4.358e-3 | 9.543e-4 | 1.626e-4 | 2.008e-4 |
| astronaut | 7.242e-3 | 3.897e-3 | 6.931e-5 | 8.154e-4 |

The target decomposition is cached exactly once.  Candidate reconstruction
scores pay only for their own single-stage BFFT decomposition.

### HD scaling check

On the 1440x960 photo, 180 initial sites required 0.57 seconds.  Ten
consumption-only germination rounds grew naturally to 1,381 cells and reached
24.58 dB.  A round rose from 2.10 to 2.83 seconds while visits per pixel rose
only from 12.45 to 14.72.  The site population increased 7.7x without a
corresponding work explosion: compact total support area remains the dominant
cost.

## Continuous diffuse/crystalline support

The power kernel was initially described as a hardness family.  That was
incorrect: increasing `p` in `(1-q)^p` concentrates mass at the center but
does not make a clean interface.  A mass-normalized logistic ellipse was
added as the true boundary-temperature control.  Making every cell logistic
was not a transferable answer: it helped the textured astronaut, cost 2–3x
more support visits, and hurt the simpler images at fixed population.

The accepted form lifts cell kind into a continuous state:

```
phi_i = (1-alpha_i) phi_power_i + alpha_i phi_logistic_i.
```

Both components have the same integrated mass.  `alpha_i` is represented by
a logit and receives the exact local receiver derivative

```
d phi_i / d logit(alpha_i)
    = alpha_i (1-alpha_i) (phi_logistic_i - phi_power_i).
```

There is no type classifier or discrete birth species.  Every cell begins at
5% crystallinity and locally earns or rejects a hard interface.  After 30
rounds, mean crystallinity was only 2–5%, while a sparse subset reached about
25%.  Nevertheless the mixture improved all controls:

| image | diffuse power | learned mixture | combined objective |
| --- | ---: | ---: | ---: |
| Pikachu | 26.31 dB | 26.93 dB | 2.899e-3 |
| camera | 32.49 dB | 34.80 dB | 5.703e-4 |
| astronaut | 25.33 dB | 28.56 dB | 1.805e-3 |

An amplitude-dependent compact cutoff reduced mixture work from roughly
34–46 to 21–31 visits per pixel with no measurable loss.  A nearly diffuse
cell does not raster the full logistic tail.

## Removing the hidden population budget

The first germination implementation recomputed the reference cell area from
the current population.  This kept total occupancy almost fixed, so a birth
could never exhaust the signal that caused it.  It was a hidden global budget.

The current form fixes an intrinsic reference area at initialization.  Cells
may shrink to 0.5% of that area through local descent.  New germs begin at
15% of the original coarse radius and are initially invisible.  Their small
intrinsic size recovers the useful fine layering that global population
rescaling had accidentally supplied, without consulting population count.
At activation threshold 0.25:

| image | 30-round PSNR | cells | final visits/pixel |
| --- | ---: | ---: | ---: |
| Pikachu 128 | 28.65 dB | 458 | 21.47 |
| camera 128 | 37.03 dB | 1,079 | 31.99 |
| astronaut 128 | 31.60 dB | 1,526 | 32.46 |
| Pikachu 256 (60 rounds) | 29.11 dB | 856 | 18.09 |

Births fall from hundreds to zero or one per round as occupancy consumes the
available resource.  No site count, batch size, sorting, or deletion enters
the law.
