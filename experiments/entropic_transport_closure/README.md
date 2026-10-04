# Entropic transport closure

The active direction was reset by the user on September 30: transporting or
preconditioning the potential with BFGS repeats the wrong-object mistake from
Meyer. The potential-descent and response-preconditioning studies below are
historical rejected directions, not the active construction. The unfinished
amortization prototype was removed from active source and retained only in
`results/theory/rejected_solver_amortization_20260930.tar.gz`.

[PRIMITIVE_TRANSPORT.md](PRIMITIVE_TRANSPORT.md) returns to the distinction
between the fixed problem relation, carried state, and the live nonlinear
request. Its immediate experiment carries measured kernel actions while
rebuilding both full reciprocal normalizations. It targets the ordinary
finite trajectory, and contains no BFGS or potential optimizer.

This experiment studies whether the evolving transport rule itself can be
carried and advanced in positive-kernel Sinkhorn iteration.

* [Formal analysis](FORMAL_ANALYSIS.md): exact coupled conditional evolution,
  moving metric, remainder bounds, and an exact projective lift.
* [Applied Sinkhorn results](APPLICATION.md): implemented finite models,
  matched marginal-accuracy timings, negative results, and reproduction.
* [Finite-energy theory](FINITE_ENERGY_THEORY.md): finite remainder bounds,
  gap-uniform second-jet transport, powered energy sums, and the separate
  observability/acquisition obstruction, with a retained spectral audit.
* [Objection tests and Gaussian controls](CLAUDE_OBJECTIONS.md): cluster size,
  approximate rule observability, history prediction, radius and partial cost
  screens, and exact continuous Gaussian covariance closure at fixed scale.
* [Directional limit on the failed scalar model](DIRECTIONAL_LIMIT.md): optimized
  directional-capture bounds, actual-excitation weighting, and an attained
  three-anchor affine-fit obstruction on the six original failed cases.

Example general-kernel call (NumPy arrays K,a,b):

```python
from experiments.entropic_transport_closure.sinkhorn import solve

result = solve(K, a, b, method="quadratic2", tolerance=1e-9)
assert result["converged"]
# The returned coupling is represented by diag(u) K diag(v).
u, v = result["u"], result["v"]
```

`ordinary`, `linear`, and `quadratic` select the controls described in the
application report. `solve_blocks` is for a supplied exact proportional-block
representation only. The general nonlinear model is an approximation with
a whole-state finite-horizon bound and an actual marginal check; it is not
an exact general lift or a fully stabilized production Sinkhorn library.

The higher-state follow-up in [CONTEXT_DESCENT.md](CONTEXT_DESCENT.md) builds an
exact conditional-response lift with a semidual descent identity and tests a
known L-BFGS response-memory realization against scalar, frozen-context, and
lean ordinary controls on the six original problems. It changes the trajectory;
it does not supply a certified long-horizon skip. The evolving scalar control
is fastest in this small retained timing battery.

[ENCLOSED_CONTINUATION.md](ENCLOSED_CONTINUATION.md) implements the
zonotope-inspired continuation follow-up: a measured chart response plus a
proved nonlinear gradient enclosure, with geometric compression preserving
inclusion. Its admitted internal steps use no full-kernel queries. All six
original problems converge, but complete cost loses to the scalar controller;
the exact-feature control and an additional n=1024 case retain that comparison.
The report distinguishes the exact-arithmetic proof from its floating-point
audit and from a certified skip of ordinary Sinkhorn dynamics.

[STRONG_FRAMES.md](STRONG_FRAMES.md) investigates numerical frame conditioning,
retained-direction coverage, and controlled Gaussianity/direction sweeps. The
new stable carriers repair accuracy but do not improve the hardest-case cost.

[MULTIPLICATIVE_ALGEBRA.md](MULTIPLICATIVE_ALGEBRA.md) implements mixed-product
request-family reuse. Finite binary families close cheaply in product counts;
the learned 64-column algebra fails to reduce difficult actual trajectory cost.
