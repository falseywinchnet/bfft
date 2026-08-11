# Design Notes

## The inverse is part of the transport

Let `R_i` be the reference mass on cell `i`'s bounded physical support and
`T_i` the transported mass on that same ordered support. The legal anchor is
not reduced to a segment label. Its within-segment rank-feature phase is
retained as `a_i`.

If the anchor occupies support position `k`, its reference mass coordinate is

```text
q_i = sum(R_i[:k]) + a_i R_i[k].
```

The transported target is the first position whose transported cumulative
mass reaches `q_i`:

```text
target_i = inf {j : sum(T_i[:j+1]) >= q_i}.
```

This is a conjugation of the reference CDF by the transported CDF. It keeps
cell identity, uses the coupling's own ordering, and produces one target. No
candidate objective is evaluated after transport.

## Two-way relaxation

The forward map compresses a physical circuit into a positive, low-rank
measure over a small active segment support. The inverse is not a generic
rounding pass: it returns the individual mass phase through the changed
measure. Horizontal position remains a gauge, changing only as required by
capacity-safe packing after the target segment is known.

The failed feedback experiment is instructive. Decoding and then rebuilding a
new site-wise HPWL pressure field did not continue descent. That rebuild loses
the coupled measure. A genuine next iteration must evolve or re-relax the
existing coupling while preserving its identity coordinate.

## Initial chart and far-field continuation

The fixed support is a local chart even when its center comes from a global
quotient.  A weak initial DEF can therefore choose the wrong integral row
basin before the identity-preserving inverse begins.  The current conditioner
uses two pre-readout witnesses—the unrestricted capacity displacement and the
exact HPWL subgradient—and admits only their common circular phase.  Net-graph
transport propagates confidence, while every cell stays on its own capacity
orbit.  Reapplying the quotient continues that same soft measure until its
support residual is small; emitted DEF quality is never a stop signal.

A spherical initial-chart walk and x/y axis ablation show that the remaining
high-density gap is almost entirely row phase.  This is not a smooth global
low mode: a rank-eight physical fit explains less WB displacement than GCD,
and a four-pass capacity/HPWL graph Krylov fit also fails on WB.  The hard row
assignment is cell-specific competitive information.  A row marginal loses
ownership; a monotone occupancy pass loses identity.

Accordingly, the next representation should retain a compact achieving
structure for global row competition—analogous to the predecessor/transport
records used by the image algorithms—then carry identity backward through
that structure.  It should not append a candidate-row search to the decoder.

An input-space heat erosion provides a useful negative result.  Multiplying
net forces by `1 - exp(-span^2 / (4 sigma^2))` preserved long nets and
attenuated short nets before global placement, but doubled WB's mean distance
from the exact target row chart and worsened final HPWL by 19.75%.  Thus the
far field is not independently recoverable after local support is discarded.
The short constraints are the phase boundary through which global ownership
is learned.

The corresponding constructive rule is multiscale but support-preserving:
coarsen or diffuse messages and gauges, never the incidence measure.  A future
DEF conditioner should keep every local net at unit physical authority while
transporting a separate coarse residual or dual correction through a
hierarchy.  Retaining the achieving path through that hierarchy would let the
inverse carry cell identity back down without evaluating alternative rows.

## Representation and cost

Use these symbols:

- `C`: movable cells;
- `S`: legal sites;
- `R`: positive feature rank;
- `D`: active support width (currently at most 6).

The reference phase evaluator costs `O(SR + CR)` and uses a temporary prefix
for one physical segment. The CDF inverse costs `O(C D log D)` in this clear
reference implementation; with fixed `D`, it is linear in cell count. The
stored inverse state is `C` scalar phases plus `C` target edges, not a dense
`C x S` coupling.

The production code can remove the tiny per-cell sort by storing active slots
in physical order, making the inverse `O(CD)`. A native C++ kernel should use
structure-of-arrays storage, fixed-width support records, compact segment
indices where safe, and caller-owned workspaces. None of those representation
changes should alter the CDF law.

## Invariants

- Every cell's active support contains its legal anchor exactly once.
- Reference and transported rows have positive mass.
- Support order is physical `(segment_y, segment_x)` order.
- The same inverse law is used at low and high utilization.
- Legalization enforces capacity but does not search for a better objective.
- Wall-clock completion time is reported alongside quality and memory.
- Continuation stops on soft transported-support residual, never final HPWL.
- Oracle basin walks are diagnostics and are never reported as algorithms.
