# Why acceleration failed, and where it works: the reduced-composite theory

Date: 2026-07-21, late.  Code: `experiments/meyer_accel.py` (E1-E4).
Prior record: `notes/meyer_bregman_ladder.md` (the two-block measurements),
shipped kernel `src/detail/meyer_kernel.hpp` (incumbent = warm 1-sweep
alternation).

## 1. The identification (E1: confirmed to 0.0)

The Gilles/A2BC alternation is exactly ISTA.  Eliminating v in closed form
(infimal convolution over the rank-deficient coupling, which only sees
u+v) leaves the single-variable static composite

    min_u  J(u) + S(u),    S(u) = (lam/2) dist^2(f-u, G_mu),
    grad S(u) = -lam * w(u),   w(u) = ROF(f-u, 1/mu),   L = lam.

Step identity: f - v_n = u_n + w(u_n), so the alternation's update
u_{n+1} = ROF(f-v_n, lam) IS the proximal-gradient step
prox_{J/lam}(u_n - (1/lam) grad S(u_n)) at the maximal stable step 1/L.
E1: the two trajectories agree to 0.0 (same operations, by the proof).

This explains every prior observation in one stroke:

- The slow geometric transfer (sliver 1/lam per pass) is ISTA on an
  ill-conditioned composite; the flat modes are low-frequency content --
  the measured swirl difference field.
- The primal heavy-ball failure was momentum on the TWO-BLOCK form, whose
  subproblem objectives move every pass -- the changing transport frame.
  The reduced objective is STATIC.  Momentum on u in the reduced form is
  textbook-sound: FISTA, same two ROF solves per iteration, transport
  re-derived fresh at the extrapolated point, nothing stale carried.
- Anderson remains correctly dismissed for the two-block map; in the
  reduced form its role is taken by FISTA with restart, which needs no
  secant model of anything.

Why the two-lattice STFT composition converges in ~1 iteration and this
cannot: the STFT constraint families are transversal at the solution
(measured pin strengths = local angles); the seed buys the basin and the
angle buys the rate.  The Meyer composite is convex but DEGENERATE: S
vanishes on an entire convex set and J is not strictly convex, so the
minimizer set is flat along exactly the modes that transfer last.  No
seed fixes a zero angle; only acceleration (or a different objective)
changes the rate law.

## 2. Where acceleration works (E2, E3, E3b: measured)

Exact-prox regime (128^2, FGP-500 inner, err to cached reference, floor
2.4e-3):

| iters | ISTA | FISTA | FISTA+restart |
|---|---|---|---|
| 10 | 1.6e-2 | 8.7e-3 | 1.1e-2 |
| 20 | 1.1e-2 | **3.0e-3** | 3.1e-3 |
| 90 | 3.2e-3 (plateau) | -- | -- |

**FISTA reaches ISTA's 90-iteration plateau in ~20 iterations (4.5x) at
identical per-iteration cost.**  Endgame honesty: past its knee FISTA
drifts to a 5.0e-3 plateau -- the degenerate minimizer set again;
different algorithms select different minimizers, and err-to-one-
reference is not a Lyapunov function there.  Restart holds the knee
longer.  The knee, not the tail, is the object of interest.

Inexact production regime (256^2, warm SB inner, err vs sweeps):

- 1-sweep inner solves (the shipped kernel's regime): FISTA == ISTA at
  every budget.  Momentum's gain is fully consumed by prox-error
  amplification (accelerated methods compound inexactness ~k*eps; plain
  descent does not).  **The incumbent is optimal in its regime; the
  shipped kernel stands.**
- Accurate inner solves, deep targets: the crossover appears and grows
  with depth.  At 800 sweeps, k=16 inner: 5.35e-3 vs incumbent 6.74e-3;
  at 3200 sweeps, k=32 inner: **2.78e-3 vs 3.74e-3** -- about 2x fewer
  sweeps at equal deep error.  Rule measured: acceleration pays exactly
  when inner accuracy can support the extrapolation, i.e. for
  high-fidelity decompositions, not for the sub-second display regime.

Production consequence (not yet implemented): a "high-fidelity mode" in
the C kernel = FISTA outer (extrapolate u, feed f-y to the w-solver) with
k~16 warm sweeps per operator and gradient restart.  Trivial addition:
one extra plane (u_prev), one scalar sequence, no new transforms.

## 3. The explicit-flux form loses (E4: measured negative)

Meyer's ball is exactly a flux box: v = div q, |q|_inf <= mu.
Substituting makes the transport a standing primal variable in a STATIC
pointwise constraint -- "blocking in the transport" in its purest form:

    min_{u, |q|<=mu} J(u) + (lam/2)||f - u - div q||^2

solved by Condat-Vu (p dual to grad u; every step a gradient, a clip, or
a div/grad; no inner solves, no FFT, 0.65 ms/iter at 256^2).  Measured:
2.2e-2 after 4000 iterations -- ~50x behind the incumbent per unit
progress.  Diagnosis: static geometry does not buy speed, because the
incumbent's SB sweep contains the EXACT spectral solve of the coupled
linear system -- a full Laplacian preconditioner applied every sweep --
while PDHG takes bare gradient steps into the same lam-weak coupling.
Preconditioning PDHG's u-block collapses it back into the reduced-ISTA
family.  Conclusion: transport-implicit-with-exact-linear-algebra beats
transport-explicit-with-cheap-steps in this problem class; the flux
formulation's value here is conceptual (it is why the reduced gradient is
a ROF residual), not computational.

## 4. Standing answers to the driving question

- Why does acceleration fail?  It was applied to a moving frame.  The
  two-block alternation re-poses each subproblem every pass; any method
  with memory (heavy ball, carried FISTA state, Anderson) extrapolates a
  flow that no longer exists.
- The form that does not fail: eliminate the degenerate block.  The
  composite J(u) + (lam/2)dist^2(f-u, G_mu) is static; its gradient is
  the texture-side ROF residual already computed every pass; FISTA on it
  is sound and measured at 4.5x (exact) / ~2x (deep inexact).
- The limit that remains: degeneracy of the minimizer set (S flat on a
  convex set, J piecewise-linear).  This is not a solver defect; it is
  the model's own gauge freedom, the same class of fact as the coverage
  law -- no operator upgrade manufactures curvature that the objective
  does not possess.  Selection within the flat set is a modeling choice
  (cf. the staircase/ramp finding: iteration count IS the regularizer).

## 5. The pass itself is a projected triangular Fourier map

Date: 2026-07-29.  Native implementation:
`src/detail/meyer_kernel.hpp`.

Write

    R_c(g) = argmin_x TV(x) + (c/2)||x-g||^2.

The exact Gilles/Meyer pass, with `w = f-u-v`, is

    u+ = R_lam(u+w)
    w+ = R_(1/mu)(f-u+)
    v+ = f-u+-w+.

Thus its outer algebra is a lower-triangular composition of two TV
resolvents.  Once an exact pass has established
`w = R_(1/mu)(f-u)`, it also reduces to the one-variable map

    u+ = R_lam(u + R_(1/mu)(f-u)).

There is no general scalar Fourier multiplier for `R_c`: isotropic TV is
the projection onto the divergence of a pointwise Euclidean-ball field.
The projection is local in pixels while the screened Poisson resolvent is
local in Fourier coordinates, and the two do not commute.

The production pass uses one warm reduced Split-Bregman/ADMM step for
each resolvent.  Let `G` be the periodic gradient, let

    P_tau(t) = projection of t onto {|t| <= tau}
    rho_tau(t) = t - 2 P_tau(t),

and let `D_u`, `D_v` be the Fourier transforms of
`div(rho_tau(t_u))`, `div(rho_tau(t_v))`.  With

    L(k) = 4 sin^2(k_x/2) + 4 sin^2(k_y/2)
    A(k) = 1/(1 + 2L(k))
    B(k) = 1/(1 + 10L(k)),

one complete native Meyer sweep is exactly

    q_u = U + W - 2 D_u
    q_v = F - 10 D_v

    [ U+ ]   [   A    0 ] [ q_u ]
    [ W+ ] = [ -BA    B ] [ q_v ]

    (u+, w+) = inverse_Fourier(U+, W+)
    t_u+ = G u+ + P_(1/(2 lam))(t_u)
    t_v+ = G w+ + P_(mu/10)(t_v).

The two screened filters are universal: `lam` and `mu` cancel from the
Fourier multipliers because the shipped penalties use
`eta_u=2 lam`, `eta_v=10/mu`.  The parameters remain only in the two
ball radii.

This exposes an exact scheduling improvement.  Both reflected
divergences depend only on the state at the beginning of the pass, and
`W+` needs `U+` only spectrally.  Therefore both forward transforms and
both spectral solves can finish before either inverse transform.  The
native kernel now:

- prepares both reflected-divergence spectra before either inverse;
- performs one fused lower-triangular spectral traversal;
- performs both inverse transforms before a paired projected-dual update.

The reordering agrees with the conventional full Split-Bregman state to
the existing `2e-12` test tolerance for both the full spectral and FACR
paths.  Alternating same-machine A/B runs of the 24-pass benchmark showed
a noisy but repeatable constant-factor gain, about 1--9% for the tested
spectral sizes and up to about 6% for tested FACR/non-power-of-two
cases.  It costs one additional spectrum scratch plane.

The next structural optimization is a two-field transform primitive.
At this point the two forward transforms are independent, as are the two
inverse transforms.  A paired real-FFT transpose/column stage (or the
standard two-real-fields-in-one-complex-FFT packing) can attack transform
cost directly.  A one-shot algebraic collapse beyond this is blocked by
the isotropic ball projection; freezing its mask and directions produces
a variable-coefficient semismooth Newton system, not another diagonal
Fourier filter.

## 6. Isotropic projection: lower bound and useful recomposition

Date: 2026-07-29.  Experiments:
`experiments/meyer_isotropic_projection_benchmark.cpp` and
`experiments/meyer_isotropic_projection_population.cpp`.

The Euclidean-ball projection has the unique rotation-equivariant radial
form

    P_tau(t) = alpha(|t|^2) t
    alpha(s) = min(1, tau/sqrt(s)).

For arbitrary Cartesian input, an exact implementation must determine on
which side of `|t|=tau` the point lies and, outside, recover its direction
by a reciprocal norm.  This is not a weakness of Split Bregman: it is the
metric projection itself.  Fourier recomposition cannot remove it because
a pointwise radial map becomes a global mode convolution in frequency
coordinates.

The Apple-arm64 projection microbenchmark (2^20 mixed interior,
boundary-near, and exterior vectors, median of seven trials) measured:

| kernel | ns/pixel | max error |
|---|---:|---:|
| two-plane copy, memory floor | 0.384 | n/a |
| incumbent exact branchless projection | 0.493 | 0 |
| squared-radius branch | 0.506 | 8.9e-16 |
| float reciprocal-sqrt seed + 1 Newton step | 0.648 | 2.9e-14 |
| float seed + 2 Newton steps | 0.779 | 1.3e-15 |
| projection with radius supplied for free | 0.478 | 1.8e-15 |

The incumbent is already only about 0.11 ns/pixel above the streaming
copy floor.  Even supplying the radius for free saves only 0.015 ns/pixel
in this loop.  The vectorized hardware square root/divide is cheaper than
branching or reconstructing a high-precision reciprocal square root.

The active set does not rescue a general branch specialization.  Measured
inside-ball fractions ranged from nearly 1.0 for the smooth sine input to
roughly 0.18--0.89 across the two fields for blocks+oscillation and white
noise.  Rebuilding or streaming an active-index list costs another pass,
and the dual update can move points across the boundary every iteration.

Two exact reuse attempts were tested:

1. Preserve `P_tau(t)` in place while forming `div(t-2P_tau(t))`, then
   finish the later update as `t+ = grad(x+) + P_tau(t)`.  This removes a
   duplicate norm evaluation but needs predecessor-row snapshots for a
   race-free parallel traversal.  End-to-end time was unchanged; the pass
   is dominated by streaming and transforms.  The added state machinery
   was reverted.
2. Test `|t|^2 <= tau^2` before evaluating the square root.  This was
   equally accurate but did not improve the full pass and was also
   reverted.

The useful recomposition is instead to consume the projection where the
next data movement begins.  The spectral path now row-streams
`div(t-2P_tau(t))` directly into the row FFT stage.  It caches the previous
reflected y-row within each eight-row panel, evaluates each current vector
once, and never materializes the scalar divergence plane.  Controlled
alternating A/B medians against the already-triangular kernel improved
about 2--4% at 256, 512, and 1024 square for 24 passes, with existing
full-state equivalence tests still passing.

Storing polar state does not avoid the lower bound.  Projection becomes a
radius clamp, but the next update is a Cartesian vector addition
`grad(x)+P_tau(t)`; updating its radius and direction requires the norm
again, and updating the angle additionally needs atan2 or an equivalent
rotation.  Quantizing directions or replacing the disk by a polygon makes
this cheaper only by changing the model to an anisotropic ball.

Consequently the exact high-precision avenues left are dataflow changes:

- pair the two independent reflected-divergence FFTs and the two inverse
  FFTs;
- fuse the two fields into one interleaved transpose/FFT pipeline;
- use two-real-signals-in-one-complex-FFT packing if it beats the native
  paired real plans.

Approximate avenues are active-mask hysteresis, quantized directions, or
low-accuracy reciprocal square root.  Each either requires certification
to remain exact or deliberately gives up rotational isotropy/precision.

## 7. Porting the fused projection across shapes and APIs

Date: 2026-07-29.

The spectral kernel is not square-specific: every rectangular input for
which both axes have native power-of-two transforms already uses the same
fused reflected-divergence row FFT.  The actual uncovered pathway was
FACR, which transforms one axis and solves the other with cyclic or
Neumann Thomas sweeps.

FACR needs an orientation-aware implementation:

- If `sweep_height` is true, each transformed line is an image row.
  Reflected vectors, divergence, and the one-axis FFT can be row-streamed
  exactly as in the full spectral kernel.  The preceding reflected y-row
  is cached across each panel, so the scalar divergence plane disappears.
- If `sweep_height` is false, transformed lines are image columns.  Direct
  fusion makes all four reflected-dual inputs strided.  An eight-column
  local gather was implemented and tested, but its scratch write/read did
  not repay the strided projection traffic.  It regressed representative
  512x300 cases and was reverted.  This orientation deliberately retains
  the row-major paired divergence materialization followed by the column
  transform.

This asymmetry is architectural, not mathematical: the disk and operator
remain isotropic, but row-major memory is not.  Controlled medium-size
measurements showed roughly 5--7% improvement for row-swept 300x512 and
288x512 FACR cases.  Column-swept cases retain the incumbent path rather
than accepting a regression.  Reduced/full equivalence is now tested for
periodic FACR and for both Neumann sweep orientations.

The full `decompose()` API had another removable split.  Its outer Meyer
stage still carried conventional `b` and `d-b` planes even though the
subsequent three texture rungs use an independent solver state.  Both
spectral and FACR decomposition now call the reduced triangular outer
kernel and hand its `u`, `w`, `v`, and spectra directly to the unchanged
ladder.  This:

- ports the triangular solve and fused projection to decomposition;
- avoids allocating four image-sized outer `d-b` planes;
- preserves the independent rung/Hodge implementation.

With one rung sweep, so the outer cost remains visible, alternating A/B
medians improved approximately 3--6% for 256x256, 300x512, and 512x300.
At the default deep rung budget the percentage is naturally diluted by
the independent ladder work.

The remaining pathways are the generic standalone ROF solve and the three
deep texture rungs.  They can adopt the reduced reflected state too, but
their tolerance checks and optional Hodge correction currently consume
the conventional feasible flux.  A safe port needs an explicit bridge

    t  ->  P_tau(t)  ->  feasible flux

at Hodge/checkpoint boundaries.  That is a separate state-layout change,
not a direct substitution in the hot Meyer pass.
