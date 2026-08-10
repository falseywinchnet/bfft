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
