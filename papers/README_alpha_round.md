# Foundational reading for the alpha round, indexed by need

Companion to `README.md` (the sigma round).  Same brief — non-iterative
algorithms answering the needs underneath the subject matter, nothing from
vision or segmentation — reached independently, and the two indexes converge
on several entries.  Where they agree, that is worth noticing: **need 2 below
is `README.md`'s need 2, and Erisman–Tinney is cited by both.**

Divergence worth keeping: the sigma round differentiates the geodesic
(Danskin) to learn the graph's *edge weights*.  This round does not touch the
metric at all; it reads the graph off the objective's own Hessian, where the
weights are already exact and require no learning.  Those are complementary,
not competing: one improves where the walls are, the other prices what the
cells are worth once the walls are fixed.

---

## Need 1 — Solve the coupled partition-of-unity fit exactly, not to a tolerance

Each pixel has exactly two owners, so the normal matrix carries one 3x3 block
per adjacent cell pair.  Measured: mean degree 4.88 at 2400 cells — essentially
planar.  Planar sparsity is the classical case where elimination beats
iteration outright, and the *ordering* is what makes it so.

- **George, A. (1973).** Nested dissection of a regular finite element mesh.
  *SIAM J. Numer. Anal.* 10(2), 345–363.
- **Rose, D. J., Tarjan, R. E., Lueker, G. S. (1976).** Algorithmic aspects of
  vertex elimination on graphs. *SIAM J. Computing* 5(2), 266–283.  Perfect
  elimination orderings in linear time; when elimination causes no fill at all.
- `kyng2016_approximate_gaussian_elimination.pdf`, `spielman2004_nearly_linear_laplacian.pdf`
  (shared with the sigma round).

Measured consequence: exact factorization replaced `lsmr` at **+0.07 dB and
4.6x faster** (119 ms vs 550 ms, Pikachu 256 px / 2400 cells).  `lsmr` had not
converged at `maxiter=160`.

## Need 2 — Price a coordinate without refitting

Constraining a solved least-squares problem to a subspace raises the objective
by a closed form in the inverse normal matrix restricted to the constrained
coordinates.  The graph enters *exactly* through a Schur complement, and
dropping that term gives an O(n) upper bound for every coordinate at once —
which is what makes shortlisting sound rather than heuristic.

- **Schur, I. (1917).**  The complement identity
  `[(H^-1)_ii]^-1 = H_ii - H_ij H_jj^-1 H_ji`.  The whole content of
  `claude_trial_alpha_normal.deletion_prices`, and the reason
  `deletion_bounds` is a true bound: the subtracted term is PSD.
- **Erisman, A. M., Tinney, W. F. (1975).** On computing certain elements of
  the inverse of a sparse matrix. *Comm. ACM* 18(3).  Selected inversion on the
  factor's sparsity pattern — the affordable exact form, and the upgrade path
  from this round's shortlist-and-solve.
- **Hager, W. W. (1989).** Updating the inverse of a matrix. *SIAM Review*
  31(2).  The rank-update form, for pricing *additions* rather than removals.
- **Allen, D. M. (1974).** *Technometrics* 16(1).  PRESS; the statistics
  ancestor.

## Need 3 — Decide where a region splits from the field's own structure

The allocator defends against tall narrow spikes with a robust percentile cap,
a Gaussian blur and an exclusion disk.  Persistence separates spike from region
in one sort and one union-find pass, with no scale parameter.

- **Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002).** Topological
  persistence and simplification. *DCG* 28, 511–533.  The elder rule.
- **Carr, H., Snoeyink, J., Axen, U. (2003).** Computing contour trees in all
  dimensions. *Comput. Geom.* 24(2), 75–94.
- **Cohen-Steiner, D., Edelsbrunner, H., Harer, D. (2007).** Stability of
  persistence diagrams. *DCG* 37, 103–120.  Why persistence is safe as a
  currency: 1-Lipschitz in the field, which amplitude is not.
- **Tarjan, R. E. (1975).** *JACM* 22(2), 215–225.
- `chazal2011_persistence_clustering_tomato.pdf`, `bauer2019_ripser_persistence.pdf`.

Measured: +0.20 dB on Pikachu at 128 px / 700, neutral to −0.49 dB on the
guards, and −0.03 dB at 256 px / 2400.  Scene-dependent; **not adopted.**

## Need 4 — Convert a focus map into a budget

The log's sharpest sentence is "a focus map can identify attention without
specifying how much budget that attention deserves."  That is the bit-allocation
problem, solved without iteration in the 1980s: sweep one multiplier, take each
unit's point on the lower convex hull of its own operational rate-distortion
curve, and every unit decides independently at the common slope.

- **Shoham, Y., Gersho, A. (1988).** Efficient bit allocation for an arbitrary
  set of quantizers. *IEEE TASSP* 36(9), 1445–1453.
- **Chou, P. A., Lookabaugh, T., Gray, R. M. (1989).** Optimal pruning with
  applications to tree-structured source coding. *IEEE Trans. IT* 35(2).
- **Riskin, E. A. (1991).** Optimal bit allocation via the generalized BFOS
  algorithm. *IEEE Trans. IT* 37(2), 400–402.  The *entire* optimal budget
  curve for a tree in one bottom-up pass.
- **Breiman, Friedman, Olshen, Stone (1984).** *CART*, ch. 3.  Weakest-link
  pruning.

The sigma round's Zador entry is the continuum limit of the same law; these are
the finite, algorithmic form.

## Need 5 — Let the budget determine the geometry instead of a reach slider

`site_reach` is one global scalar standing in for every cell's additive power
weight.  For any prescribed capacities there is a **unique** power diagram
realizing them, obtained from a concave maximization — no local minima, unlike
the Lloyd iteration it would replace.

- **Aurenhammer, F., Hoffmann, F., Aronov, B. (1998).** Minkowski-type theorems
  and least-squares clustering. *Algorithmica* 20, 61–76.
- `kitagawa2016_newton_semidiscrete_ot.pdf`, `merigot2011_multiscale_optimal_transport.pdf`.

Not yet tried.  This is the strongest untried item in this index.

## Need 6 — Learn the graph rather than round on a supplied one

- **Gower, J. C., Ross, G. J. S. (1969).** Minimum spanning trees and single
  linkage cluster analysis. *Applied Statistics* 18(1), 54–64.  The
  MST / single-linkage / subdominant-ultrametric equivalence: the unique
  maximal ultrametric below a dissimilarity, in one Kruskal pass.
- **Boruvka (1926); Kruskal (1956).**
- The sigma round's `spielman_srivastava_effective_resistances.pdf` belongs
  here too, and states the same conclusion this round reached numerically:
  effective resistance on the design matrix **is** statistical leverage, and
  the deletion price computed here is that quantity wearing different clothes.

## Need 7 — Exact inference on a tree in two passes

Unused so far: the direct factorization was already fast enough.  Kept because
if the cell graph is ever reduced to a tree, the joint fit becomes exact after
one upward and one downward sweep with no factorization at all.

- **Pearl, J. (1988).** *Probabilistic Reasoning in Intelligent Systems*, ch. 4.
- **Rauch, Tung, Striebel (1965).** *AIAA J.* 3(8), 1445–1450.

---

### Downloaded in this round

    bauer2019_ripser_persistence.pdf
    chazal2011_persistence_clustering_tomato.pdf
    kitagawa2016_newton_semidiscrete_ot.pdf
    kyng2016_approximate_gaussian_elimination.pdf
    merigot2011_multiscale_optimal_transport.pdf
    spielman2004_nearly_linear_laplacian.pdf

The remaining citations are paywalled or predate arXiv; their hosts refuse
automated download.  Results are used directly.
