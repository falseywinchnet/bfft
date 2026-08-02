# Native vision port queue

These files are the algorithm boundaries used by the canonical segmenting
viewer. Each has one input/output contract and can be replaced independently
by a C++ implementation.

The canonical causal-density path is:

0. `fast_image_ops.py` — Gaussian tensor smoothing, axis-specific resize
   prefiltering, pixel-centred bilinear resize, Sobel derivatives, and cross
   dilation. These bounded-support DSP kernels are native C++ through the
   `bfft_vision_*_f64` image ABI; the Numba implementations are compatibility
   references for older installed libraries.
1. `frozen_meyer_geometry.py` — one optimized Meyer/ROF support measurement.
2. `density_population.py` — curvature-limited tensor population and local
   parallel quantization, with no population search. The curvature kernel is
   now native C++ through `bfft_vision_curvature_population_f32`. The frozen
   geometry also carries a cross-scale local-null confidence: only weak,
   nonpersistent activity is attenuated, so smooth camera sky fuses without
   suppressing strong isotropic texture.
3. `metric_reduced_stencil.py` — obtuse unimodular stencil reduction of the
   measured metric.
4. `continuous_eikonal_transport.py` — continuous-source, same-label
   Hopf--Lax first arrival and its causal parent DAG. Its local simplex
   minimizer is closed form, reverse incidence is linear and duplicate-free,
   and the heap has one decrease-key entry per unaccepted pixel. The exact
   walk and all parent/covector bookkeeping are now native C++ through
   `bfft_vision_fast_march_first_label`. A separate fourth-order
   unchanged-target jump tensor can add finite action across decisive
   photometric interfaces without increasing population.
5. `first_arrival_site_force.py` — reverse characteristic force, local Newton
   surrogate with an analytic positive-definite projection, half-inradius
   trust region, and exact action-decrease remarch.
6. `hard_region_fit.py` — centered/radius-scaled per-cell affine/ridge
   readout with a physical image-gradient regularizer. Its repeated reductions,
   fixed-small-system elimination, and rendering are native C++ through
   `bfft_vision_hard_affine_fit` and `bfft_vision_hard_basis_refit`.
7. `fractional_interface_coverage.py` — objective-gated subpixel
   rasterization of local front collisions. It uses accepted action and the
   incident edge cost, not a propagated runner-up field.
8. `soft_support_diffusion.py` — objective-gated owner-free partition of unity
   over the finished hard geometry. Its repeated convex heat step is now
   native C++ through `bfft_vision_soft_support_diffuse`; conductance
   construction remains the Python executable specification.

The following remain supported experimental controls:

- `anisotropic_edge_cost.py`, `two_label_transport.py`,
  `soft_transport_moments.py`, `metric_instability.py`, and
  `balanced_refill.py` implement the older simultaneous bifurcation path. The
  exact first-owner and owner/runner Dial walks in `two_label_transport.py`
  are native C++; their Numba implementations remain reference fallbacks in
  `monotone_bucket_transport.py`.
- `reverse_residual_flow.py` implements predecessor-tree residual refill.
- `residual_pressure_transport.py` implements conserved-population soft power
  transport under decomposition-residual pressure.

`pipeline.py` is orchestration, not a port target. `allocation_flow.py`
composes the legacy allocation controls and contains no additional numerical
kernel.

The decisions and before/after measurements from the Python tightening round
are recorded in
[`../notes/causal_port_tightening.md`](../notes/causal_port_tightening.md).
Curvature and soft-support derivations, controls, and native benchmarks are
recorded in
[`../notes/CURVATURE_AND_SOFT_SUPPORT.md`](../notes/CURVATURE_AND_SOFT_SUPPORT.md).
The subsequent front/fit/channel measurements are recorded in
[`../notes/NATIVE_SEGMENTING_VIEWER_ROUND.md`](../notes/NATIVE_SEGMENTING_VIEWER_ROUND.md).
The local-null, jump-action, and fractional-interface controls and rejected
forms are recorded in
[`../notes/NULL_JUMP_INTERFACE_REFINEMENT.md`](../notes/NULL_JUMP_INTERFACE_REFINEMENT.md).

The guiding performance rule is structural: no candidate enumeration, top-k,
site deletion, offspring, or all-pairs cell work. The only image-wide
operations are stencils, monotone propagation, and fixed reductions. Exact
topology refresh is permitted because it directly solves the current
transport state; later incremental replacement must preserve the same result.
