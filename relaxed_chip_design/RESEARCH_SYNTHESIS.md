# What must live in the transport

## Result

The blue/orange reversal is plausibly a real circular phase-cut effect, and
there is an exact non-enumerative way to remove that ambiguity.  It is not,
however, the missing cell decoder.

The current local support coupling preserves mass and a weak direction bit.
It does not preserve the relative cell-to-row chart needed to turn that
population-level phase into competitive ownership.  The next representation
should therefore be a **connection-lifted, perfectly reconstructing transport
hierarchy**:

```text
mass flow + relative phase/restriction maps + stored lifting detail
```

Mass says how much crosses a boundary.  The connection says how neighboring
identity charts correspond.  The detail record makes coarsening invertible.
None of the three can reconstruct either of the other two after it has been
discarded.

## Experiment outcome: orientation was the missing transported component

The first implementation round tested the proposed channels independently on
both frozen circuits.  The decisive result came from vector diffusion maps,
not from an absolute synchronized row chart.

The former conditioner diffused scalar confidence over the net graph and then
projected it onto each cell's original capacity direction.  The successful
law diffuses the admitted `U(1)` direction itself, while every cell retains its
own bounded capacity-orbit radius.  It therefore transports orientation but
never a neighbor's destination or step length.  The existing residual-gated
backward row graft and support-CDF inverse are unchanged.

| Design | Former selected HPWL (um) | Vector connection HPWL (um) | Guarded wall | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| `gcd` | 7,280.1875 | **7,280.1875** | 0.79 s | 154,244 KiB |
| `wb_dma_top` | 40,411.0225 | **39,932.2000** | 2.01 s | 130,640 KiB |

WB improves by 478.8225 um and GCD is byte-identical.  A second WB run is also
byte-identical, and independent DEF parsing confirms 39,932.2000 um.  On WB,
82.78% of cells receive a redirected orientation and 62.16% gain confidence
from the graph; added working state is 298,992 bytes.

The falsifications refine the original architecture:

- exact aggregate circular transport is real and raises WB within-two-row
  coverage from 68.19% to 70.29%, but its best complete result is 40,436.6275;
- a perfectly reconstructing dyadic parity lift loses phase quality, proving
  that row parity is not the fine ownership detail;
- deterministic degree-2 and degree-4 far-field features greatly reduce
  kernel error but do not improve the complete WB law;
- a cell-to-one-net-center connection has excellent relative consistency yet
  halves exact-row recovery, because HPWL is an interval rather than a meeting
  point;
- connection-resultant conductance gating removes a weak direction that the
  successful vector law needs.

The next implementation round added the **oriented HPWL interval boundary** to
the successful vector state. Full reflection changed about 61% of nets and
destroyed too much common orientation. Requiring positive-confidence movable
witnesses on both lower and upper faces admits only about 10% of nets, restores
GCD exactly, and exposes a stable odd normal channel on WB. Projecting that
channel immediately back into a two-vector still loses 398.8075 um against the
selected law.

Thus `phase + signed boundary moment` must be a true four-component stalk:

```text
(even_x, even_y, odd_x, odd_y)
```

The completed late-synthesis round shows that carrying four components is
necessary but not sufficient. Applying odd incidence independently per cell
moves three WB identities toward the exact row phase, yet violates aggregate
segment capacity and loses through horizontal repacking. The winning
restriction acts on `log(T/R)` and KL-projects the result onto the original
cell and segment marginals before the CDF inverse. It reaches 39,904.3975 um
on WB while preserving GCD byte-for-byte.

Thus a sheaf stalk must enter the coupling as a **conservative tangent**:

```text
sum_s delta(i,s) = 0
sum_i mass_i delta(i,s) = 0
```

This is now the fixed interpretation of experiment C below. A future
multiscale lift must restrict and prolong both connection state and its
zero-marginal constraint; early projection, an unbalanced late correction, or
a tuned even/odd blend is disqualified.

## The exact part of the observed phase reversal

For two equal-mass histograms on an ordered cycle, let

```text
Delta[k] = cumulative(reference - transported)[k].
```

For geodesic `L1` cost, Delon--Rabin--Gousseau reduce circular transport to

```text
min_alpha sum_k interval_weight[k] * abs(Delta[k] - alpha).
```

The exact `alpha` is a weighted median of `Delta`.  The corrected signed flux
is `Delta - alpha`.  Prefix construction plus weighted selection is `O(R)`
for `R` rows; sorting would be `O(R log R)`.  No rotation, row destination, or
candidate DEF is scored.

This is almost exactly the shape seen in the transport microscope: one sign
above the apparent cut, the opposite sign below it, and a crossing around the
middle.  It gives us a direct test for whether the horizontal band at row 40
is an arbitrary cut through a circular cumulative flow.

But `alpha` is one global circulation constant.  The existing per-cell audit
already shows that this cannot supply ownership:

- the local signed moment has correlation `0.0026` and `-0.0851` with the
  required displacement on the two passes;
- when the exact row is in support it has positive gain only about 52% of the
  time;
- exact-row CDF recovery remains about 17%.

The oracle-sorted image can therefore expose a collective phase boundary even
when no individual local record contains the coordinate used to sort it.
Circular correction should condition and diagnose the phase field.  It must
not be used as a post-hoc row decoder.

## What the BFFT vision methods already taught us

The useful commonality is not that these methods process images.  It is that
they transport a small achieving structure rather than asking a decoder to
reconstruct one.

### Segmenter v3: phase is integrated from relations

`experiments/segmenting_v3.py` obtains local wave covectors from paired
one-sided full-band correlations, retains the most confident relational
edges, and unrolls phase through a spanning forest.  It does not enumerate
carrier/object decompositions.  The chip analogue is to measure reliable
relative row phase on pre-quotient circuit relations and integrate those
relations once.

### Night vision: diffuse a circle-valued chart

`high_vision/experiments/realistic_orbit_bootstrap.py` fuses parent votes as
unit phasors during one outward march.  This is connection diffusion: what is
propagated is an orientation relative to neighboring charts, not a blurred
scalar label.  `connection_phase_seed` in
`high_vision/experiments/closure_transport_consensus.py` makes the same point
spectrally.  Its later repeated consensus is not carried into this proposal.

The low-SNR multi-reference alignment paper in the night-vision corpus gives a
warning rather than an algorithm: even when Fourier-phase information exists
in the measurements, an iterative alignment method can organize noise around
its current template.  Its two convergence regimes are further reason to
reject EM/SGD as the row-chart learner.

### FlowCells: the inverse is a stored causal graph

`paper/flowcells/main.tex` stores first and second parents, a barycentric
fraction, the accepted covector, and acceptance order.  A single reverse pass
then returns each pixel's mass to its source.  It avoids both a dense
pixel-by-site coupling and per-pixel path tracing.  For placement, a sparse
predecessor/restriction record is the missing object between coarse row
competition and local support.

### Super-resolution and Meyer preconditioning: coarsen operators, not truth

`experiments/V3_SUPER_RESOLUTION.md` keeps owner-respecting backprojections;
its proposed additional octave is a fixed polynomial of an owner-masked
Laplace--Beltrami operator.  `experiments/MEYER_PRECONDITIONING_RESEARCH.md`
uses one finite spectral power, one Hodge lift, one deterministic transverse
route, and one projection.  It represents a discontinuity as an oriented BV
bond measure: support, normal, and signed jump amplitude, rather than a scalar
halo.

Together these say that dimension may be sparsified, but an interface must
retain both its orientation and its inverse relation.  This is exactly what a
small sheaf stalk plus lifting detail supplies.

### YOND: false support erased early never reaches the fine stage

The YOND source in `tmp/pdfs/night_vision_research/yond_source/YOND.tex`
states the failure concretely: when texture leaks into the flat-region mask,
coarse noise is overestimated, the coarse image is blurred, and the fine
correction has severely limited capacity to repair it.  This matches the chip
audit.  Once competitive identity is averaged out in the initial quotient,
unrelaxation has nothing from which to recover it.

## Proposed representation

Associate each active cell or coarse cluster `v` with a constant-width stalk,
initially no larger than

```text
x_v = [mass, Re(row_phase), Im(row_phase), signed_boundary_moment].
```

An edge or net incidence `e = (u, v)` carries restriction/connection maps that
say how the two local charts compare.  Purely invertible relations may use a
`U(1)` phase.  Capacity projections and many-to-one net summaries should use
general sheaf restrictions because they need not be rotations.

The phase evidence must be constructed **before** the local support quotient.
The current unrestricted capacity displacement and HPWL subgradient are
already two circular witnesses.  The new experiment should pair their
opposite/adjacent responses across fixed row scales, as segmenter v3 pairs
one-sided bands, and retain a relation only when those witnesses agree.  The
exact reference placement is never an input; it is used only after the run to
test whether the relation was identifiable.

For an invertible phase edge, write

```text
g_uv = unit(relative phase from u to v)
H_uv = w_uv * g_uv.
```

One connection-Laplacian factorization/eigenspace synchronization produces a
global chart.  Cycle holonomy measures contradiction in the input evidence.
If relations are projections rather than rotations, solve the corresponding
small-stalk sheaf Laplacian instead.  This is one sparse algebraic solve, not
a walk through DEF alternatives.

## Reversible multiscale transport

The hierarchy should use four finite operations:

1. **Analyze.** Restrict mass and connection state to a coarse separator.
2. **Eliminate.** Schur/Kron-reduce interiors so the coarse operator preserves
   boundary response.
3. **Sparsify once.** Use connection effective resistance as importance and
   BFFT's deterministic low-discrepancy quantization to retain a fixed edge
   budget.  Do not greedily test edge removals.
4. **Store detail.** For prediction `P`, retain
   `d_fine = x_fine - P x_coarse` together with the achieving
   predecessor/restriction relation.

Synthesis reverses the stored lifting operations:

```text
x_fine = P x_coarse + d_fine.
```

The final local CDF inverse remains unchanged.  It receives one prolonged
identity chart and emits one target.  Capacity/tree channels may use exact
subtree imbalance or two-pass generalized-distributive-law messages, but the
identity chart is a separate connection-bearing channel.  A scalar tree
Wasserstein flow is not a substitute for it.

For stalk width `d`, a bounded-degree hierarchy has state
`O(C d + E d^2)` and linear lifting traffic.  The phase-cut correction is
`O(R)`.  The dominant new work is one sparse connection/sheaf solve; there is
no `cells x sites` allocation and no factor proportional to the number of
candidate rows or DEFs.

## First experiment: learn whether phase exists before the quotient

This experiment is deliberately diagnostic before it is an optimizer.

1. From the existing transported/reference row histograms, compute `Delta`,
   its weighted median `alpha`, and the corrected circular flux
   `Delta - alpha` on both WB passes.  Overlay the inferred cut on the current
   gain/loss microscope.
2. Before local-support compression, form pairwise relative-phase observations
   on net/incidence edges from the unrestricted capacity and HPWL witnesses.
   Use fixed dyadic row bands, paired one-sided responses, and deterministic
   confidence thresholds.  Do not read exact rows here.
3. Run one `U(1)` connection solve.  Record edge residual, cycle holonomy,
   factorization time, peak RSS, and stalk/edge bytes.
4. Only after the solve, compare predicted **relative** phase with relative
   exact rows.  This is the identifiability gate.  Absolute phase is gauge and
   should not be scored until the circular median fixes the cut.
5. If the gate passes, prolong the chart through one two-level lifting record,
   run the unchanged support transport/unrelaxation once, and measure HPWL,
   exact-row recovery, completion time, and peak RSS on both GCD and WB.

Use four fixed ablations:

```text
A  mass + scalar circular cut
B  mass + U(1) connection phase
C  mass + sheaf stalk (phase + signed boundary moment)
D  C + one reversible coarse/fine lifting level
```

The exact target may be used to report relative-phase correlation and
exact-row recovery, never to construct `g_uv`, choose an ablation, stop a
solve, or select a DEF.

### Stop before optimizing if the signal is absent

Synchronization enforces consistency; it cannot invent a relation.  If the
pre-quotient edge observations have near-zero correlation with oracle relative
row offsets, or if holonomy remains indistinguishable from a shuffled-edge
control, stop.  That result would mean the required identity never entered
even the unrestricted transport, and the next change belongs in DEF
conditioning rather than in the inverse.

### Success criterion

The representation is promising only if one fixed law:

- improves or preserves both GCD and WB;
- increases relative-phase consistency before hard emission;
- allocates no dense cell-by-site object;
- uses one sparse solve and one synthesis, with no candidate-placement score;
- reports wall-clock completion time, peak RSS, and explicit state bytes.

## Methods deliberately excluded

- EM, SGD, or repeated template alignment from low-SNR MRA;
- repeated Bregman or consensus refinement;
- Lloyd simplification plus a new OT optimization at each scale;
- greedy contraction, greedy sampling-set design, or iterative edge deletion;
- tree-sliced averaging over many random trees;
- full-row permutation synchronization and Hungarian rounding;
- candidate rows, alternative DEF emission, or HPWL-based stopping;
- any decoder that treats the oracle-sorted phase microscope as transported
  per-cell state.

The paper-by-paper evidence and source links are in
[`papers/README.md`](papers/README.md).
