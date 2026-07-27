# Native vision port queue

These files are the algorithm boundaries used by the canonical segmenting
viewer. Each has one input/output contract and can be replaced independently
by a C++ implementation.

The canonical causal-density path is:

1. `frozen_meyer_geometry.py` — one optimized Meyer/ROF support measurement.
2. `density_population.py` — local parallel quantization of the tensor-implied
   population, with no population search.
3. `metric_reduced_stencil.py` — obtuse unimodular stencil reduction of the
   measured metric.
4. `continuous_eikonal_transport.py` — continuous-source, same-label
   Hopf--Lax first arrival and its causal parent DAG.
5. `first_arrival_site_force.py` — reverse characteristic force, local Newton
   surrogate, half-inradius trust region, and exact action-decrease remarch.
6. `hard_region_fit.py` — independent per-cell affine/ridge readout.

The following remain supported experimental controls:

- `anisotropic_edge_cost.py`, `two_label_transport.py`,
  `soft_transport_moments.py`, `metric_instability.py`, and
  `balanced_refill.py` implement the older simultaneous bifurcation path.
- `reverse_residual_flow.py` implements predecessor-tree residual refill.
- `residual_pressure_transport.py` implements conserved-population soft power
  transport under decomposition-residual pressure.

`pipeline.py` is orchestration, not a port target. `allocation_flow.py`
composes the legacy allocation controls and contains no additional numerical
kernel.

The guiding performance rule is structural: no candidate enumeration, top-k,
site deletion, offspring, or all-pairs cell work. The only image-wide
operations are stencils, monotone propagation, and fixed reductions. Exact
topology refresh is permitted because it directly solves the current
transport state; later incremental replacement must preserve the same result.
