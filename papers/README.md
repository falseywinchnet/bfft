# Foundational references for the sigma round

None of these are about images, vision, or segmentation. Each one addresses a
need this work has *underneath* its subject matter — a need that survives if
you delete every mention of pixels from the problem statement.

The needs, stated without the subject:

1. Solve a least-squares system over overlapping supports **exactly**, not to
   a tolerance.
2. Know what each degree of freedom is worth **without refitting anything**.
3. Choose `k` atoms out of many with a guarantee, in one pass.
4. Differentiate the output of a combinatorial minimum **exactly**.
5. Know the correct pairwise affinity on a graph, rather than supplying one.

## Downloaded

| File | Need | Why it is here |
|---|---|---|
| `spielman_srivastava_effective_resistances.pdf` | 5, 3 | The correct sampling measure on a graph is effective resistance, which for a design matrix *is* statistical leverage. This is the theorem that says the earlier spin/max-cut framing was the wrong objective **class**: the right relaxation of "which atoms, and are they redundant" is spectral approximation, not cut maximization. Rounding a cut objective cannot repair that; changing the objective can. |
| `drineas_mahoney_leverage_coherence.pdf` | 2, 3 | Leverage scores computed fast. Gives the practical route from "I have the design matrix" to "I know what every column is worth" without an eigendecomposition. |
| `deshpande_rademacher_volume_sampling.pdf` | 3 | Volume sampling: a one-pass, provably `(k+1)`-competitive answer to column subset selection. The honest replacement for greedy placement, and it explains why greedy is usually nearly right. |
| `halko_martinsson_tropp_randomized_decompositions.pdf` | 1, 2 | The standard reference for replacing an iterative solve with a randomized factorization, and for the probing estimators used to shortlist deletion costs. |

Already present in this folder from the previous round, and relevant to the
same needs: `spielman2004_nearly_linear_laplacian.pdf`,
`kyng2016_approximate_gaussian_elimination.pdf` (need 1, direct solvers for
graph-structured systems), `merigot2011_multiscale_optimal_transport.pdf` and
`kitagawa2016_newton_semidiscrete_ot.pdf` (need 4, second-order methods for
power-diagram weights), `chazal2011_persistence_clustering_tomato.pdf` and
`bauer2019_ripser_persistence.pdf`.

## Cited, not downloaded

These are the load-bearing ones for what the sigma round actually implements.
All are paywalled or predate arXiv; the results are used directly.

- **Danskin, J. M. (1966), "The theory of max-min, with applications",
  SIAM J. Appl. Math. 14(4).** Need 4. The derivative of a minimum over a
  compact set is the derivative at the achieving argument. A geodesic
  distance is a minimum of linear functions of the edge weights, so its
  derivative is the indicator of the achieving path — exactly, in one pass,
  with no perturbation and no adjoint system. This is the whole content of
  `graph_descent` and it is why that gradient is not an approximation.

- **Melenk, J. M. and Babuška, I. (1996), "The partition of unity finite
  element method: Basic theory and applications", Comput. Methods Appl.
  Mech. Engrg. 139.** Need 1. The renderer here *is* a partition-of-unity
  method. The theory says local spaces may be enriched with non-polynomial
  functions and the global space inherits the approximation order, and it
  says the stability constant is governed by the local Lebesgue constant.
  That is the reason the earlier bounded-quadratic patch lost 0.8–2.6 dB
  while affine patches are stable under overlap, and it is what motivated
  trying a *bounded* discontinuous enrichment instead of a higher-degree
  polynomial.

- **Erisman, A. M. and Tinney, W. F. (1975), "On computing certain elements
  of the inverse of a sparse matrix", Comm. ACM 18(3).** Need 2. Selected
  elements of the inverse are computable on the sparsity pattern of the
  factor. This is the exact form of the deletion price; the sigma round
  approximates it with probing plus an exact shortlist, which is the
  affordable version of the same identity.

- **Allen, D. M. (1974), "The relationship between variable selection and
  data augmentation and a method for prediction", Technometrics 16(1)**, and
  **Golub, G., Heath, M., Wahba, G. (1979), "Generalized cross-validation",
  Technometrics 21(2).** Need 2. Leave-one-out cost in closed form. The
  reason a deletion price needs no refit.

- **Gu, M. and Eisenstat, S. C. (1996), "Efficient algorithms for computing
  a strong rank-revealing QR factorization", SIAM J. Sci. Comput. 17(4).**
  Need 3. One pivoted pass, with a bound.

- **Zador, P. (1982), "Asymptotic quantization error of continuous signals
  and the quantization dimension", IEEE Trans. Inform. Theory 28(2)**, and
  the survey **Gray, R. M. and Neuhoff, D. L. (1998), "Quantization", IEEE
  Trans. Inform. Theory 44(6).** The exchange rate between a local error
  density and a local point density: `lambda ∝ f^(d/(d+2))`, and at the
  optimum every cell carries equal error. This is the missing law behind
  "attention identifies a region but does not say how much budget it
  deserves".

- **d'Azevedo, E. F. and Simpson, R. B. (1991), "On optimal interpolation
  triangle incidences", SIAM J. Sci. Stat. Comput. 12(6).** The optimal
  metric for piecewise-*linear* approximation is the absolute Hessian, not
  the gradient. Recorded here because the previous round rejected a Hessian
  metric empirically; the theorem says what that rejection is evidence
  *against* — it is evidence that the cells are not currently limited by
  affine approximation error.
