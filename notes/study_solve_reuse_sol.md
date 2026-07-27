# One-shot exact solve study for Sigma's measured co-ownership graph

Date: 2026-07-26

Scope was deliberately narrowed to one system built from the current measured
co-ownership graph. This study does **not** enumerate candidates, reuse
candidate patterns, or assume that a later graph has the same sparsity.

## Conclusion

There is no exact one-shot linear solve for the full reach optimum, even while
the owner/runner graph is held fixed. The current reach system is already the
one-shot exact solution of a **local Gauss-Newton quadratic**. Backtracking and
outer steps are not paying for a weak sparse solver; they are globalizing a
nonlinear model and crossing ownership chambers.

An exact pivoted-direct sideport is still worthwhile for the affine field normal
systems. Eigen's `SimplicialLDLT` reduced factor-plus-three-RHS time by about
1.4x at both measured sizes. A custom 3x3 block LDLT is structurally
well-matched, but its likely gain is a constant factor on a 6--12 ms
factorization, not removal of the Newton trials.

## Why the full reach update is not a linear solve

For a jointly owned pixel, with reach values `rho`, incidence row
`q = e_i - e_j`, fixed distance gap `d`, and fitted cell atoms `f_i`, `f_j`,

```
s(rho) = d + q rho
w(rho) = sigmoid(beta s(rho))
y(rho,c) = w f_i(c_i) + (1-w) f_j(c_j).
```

Even if the affine coefficients `c` are frozen, squared reconstruction error
contains a squared sigmoid. Its Hessian contains both `sigmoid'` squared and a
residual-weighted `sigmoid''` term, and therefore changes with `rho`. The
edge differences also cannot be inverted independently because cycles in the
cell graph impose consistency constraints.

Refitting the affine fields makes the dependence stronger:

```
F(rho) = min_c ||A(rho)c - b||^2 + c^T Lambda c
G(rho) = A(rho)^T A(rho) + Lambda.
```

Both `G` and its inverse depend on `rho`. There is no elimination that turns
`F` into one fixed Laplacian quadratic.

At the current point, the receiver-guided code forms

```
g_GN = E^T z
H_GN = E^T W E + lambda I
delta_rho = -H_GN^-1 g_GN.
```

That last line is already an exact, one-shot direct solve of the local
Gauss-Newton model. It is not the exact minimizer of `F`.

A larger joint normal system can include coefficient relaxation:

```
[ G       C ] [delta_c  ] = -[g_c  ]
[ C^T  H_GN ] [delta_rho]    [g_rho]
```

Solving it is an exact Gauss-Newton step for the joint linearization (or,
equivalently, its Schur complement in `rho`). It still is not an exact step
for the nonlinear objective, so an acceptance evaluation or trust-region
mechanism remains necessary. It also grows the system instead of removing
the nonlinear outer problem.

## Measured costs

Machine: the current Apple host, Python venv with NumPy 2.4.6 and SciPy 1.18.0.
Numbers are best warm times unless a median is shown. The fixtures and fused
normal assembly are from:

- `experiments/sigma_opt/bench_common.py`
- `experiments/sigma_opt/opt_normal_assembly.py`
- camera, 128 px / 700 cells
- Pikachu, 256 px / 2,400 cells

### Scalar reach Laplacian

The reach matrix is `n x n`, not `3n x 3n`.

| cells | graph edges | matrix nnz | gradient + Laplacian build | SuperLU factor | SuperLU solve |
|---:|---:|---:|---:|---:|---:|
| 700 | 1,698 | 4,096 | 6.98 ms | 0.44 ms | 0.018 ms |
| 2,400 | 5,842 | 14,084 | 33.87 ms | 4.36 ms | 0.094 ms |

Residuals were `3.6e-16` relative. At 2,400 cells, constructing the measured
gradient/Laplacian costs about 7.6x the exact direct factor and about 360x the
triangular solve. Replacing this solve cannot remove meaningful iteration
time.

Eigen `SimplicialLDLT` factored the same matrices in 0.195 ms and 1.11 ms,
respectively. This confirms that an SPD C++ solver improves the constant, but
also makes the reach solve even less relevant to end-to-end cost.

### Three-unknown affine normal system

| cells | unknowns | matrix nnz | solver | factor best (median) | solve 3 RHS best (median) | total best |
|---:|---:|---:|---|---:|---:|---:|
| 700 | 2,100 | 36,864 | SuperLU | 2.43 (2.63) ms | 0.174 (0.177) ms | 2.60 ms |
| 700 | 2,100 | 36,864 | Eigen LDLT | 1.49 (1.62) ms | 0.305 (0.319) ms | 1.79 ms |
| 2,400 | 7,200 | 126,756 | SuperLU | 10.71 (11.42) ms | 0.635 (0.726) ms | 11.35 ms |
| 2,400 | 7,200 | 126,756 | Eigen LDLT | 6.88 (7.26) ms | 1.176 (1.217) ms | 8.05 ms |

Eigen relative residuals were `1.5e-15` and `1.9e-15`; SuperLU residuals were
below `3.5e-16`. The isolated C++ benchmark is:

- `experiments/sigma_opt/bench_eigen_one_shot.cpp`

It uses Eigen already present at `/opt/homebrew/include/eigen3` and does a
fresh AMD ordering, factorization, and solve on every repetition. Nothing is
amortized across systems.

## 3x3 block LDLT viability

The affine normal matrix is a block graph with one dense 3x3 diagonal block
per cell and one dense 3x3 off-diagonal block per co-ownership edge. Fill
measurements after minimum-degree ordering:

| cells | scalar `L` nnz | off-diagonal factor blocks | full 3x3 blocks | estimated block-factor flops |
|---:|---:|---:|---:|---:|
| 700 | 61,791 | 6,399 | 100% | 1.49 M |
| 2,400 | 232,281 | 24,209 | 100% | 7.21 M |

SuperLU kept the three scalar unknowns of a cell consecutive for 94.0% and
93.1% of cells. Forcing all cells to remain 3-wide produced **exactly the
same factor nnz** on both fixtures. Thus a C++ block LDLT can use dense 3x3
kernels without a fill penalty on these measured graphs.

This is positive evidence for a compact BSR/block solver, but not evidence
that it removes Newton iteration. Seven million dense-block flops are small;
ordering, symbolic bookkeeping, indirect memory access, and assembly dominate.
Eigen's fresh exact direct factor already captures much of the available gain.

## Elimination and multilevel alternatives

- Sparse LDLT with AMD or nested dissection is the correct exact
  one-shot formulation. Nested dissection may improve scaling for much larger
  planar graphs, but it only changes ordering/fill; it does not solve the
  nonlinear reach objective in one shot.
- AMG and modern Laplacian solvers are approximate or preconditioned iterative
  methods. Driving them to machine precision reintroduces iteration. On the
  measured 2,400-node reach graph, a 1--4 ms direct factor leaves no practical
  opening.
- A custom block LDLT is justified only if profiling after the fused assembly
  and C++ port still shows the affine factorization as a major fraction. The
  measured structure is favorable, but the upper bound is a constant-factor
  win on the field solve.

## Recommendation

1. Port the fused 3x3 finite-element assembly and exact SPD solve together.
   Use a fresh AMD-ordered `SimplicialLDLT` (portable) or CHOLMOD (optional
   high-performance backend). This is exact and one-shot for each measured
   affine field.
2. Keep the reach solve as a scalar sparse direct factor. Do not build AMG,
   Woodbury machinery, or a custom multilevel solver for it.
3. Do not claim that the graph-Newton line search can be replaced by an exact
   one-shot update. Removing its evaluations changes the algorithm to an
   unglobalized local Gauss-Newton step and loses the measured-objective
   acceptance guarantee.
4. If a future profile justifies it, implement a cell-block LDLT over 3x3 BSR.
   The zero fill penalty and 100% dense factor blocks are the relevant proof
   points; benchmark it against Eigen before making it a dependency-free core
   backend.
