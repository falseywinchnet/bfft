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
