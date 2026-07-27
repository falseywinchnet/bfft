# Native vision port queue

These files are the algorithm boundaries used by the canonical segmenting
viewer.  Each has one input/output contract and can be replaced independently
by a C++ implementation:

1. `frozen_meyer_geometry.py` — one optimized Meyer/ROF support measurement.
2. `anisotropic_edge_cost.py` — fixed eight-neighbour metric stencil.
3. `two_label_transport.py` — exact monotone two-label transport refresh.
4. `soft_transport_moments.py` — fused soft mass and covariance reduction.
5. `metric_instability.py` — closed-form per-cell 2x2 support test.
6. `balanced_refill.py` — simultaneous fixed-pass mass-balanced refill.
7. `hard_region_fit.py` — independent per-cell affine/ridge readout.

`pipeline.py` is orchestration, not a port target.
`allocation_flow.py` fixes the supported structural policy and composes ports
02–06; it contains no additional numerical kernel.

The guiding performance rule is structural: no candidate enumeration, top-k,
site deletion, or all-pairs cell work.  The only image-wide operations are
stencils, monotone propagation, and a fixed number of reductions.  Exact
topology refresh is permitted because it directly solves the current
transport state; later incremental replacement must preserve the same result.
