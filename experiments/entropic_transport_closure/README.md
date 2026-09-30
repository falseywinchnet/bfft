# Entropic transport closure

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
