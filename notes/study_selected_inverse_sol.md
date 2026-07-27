# Selected inverse and deletion pricing: completed study

Date: 2026-07-26

This finishes the interrupted `experiments/sigma_opt/opt_selected_inverse.py`
study without changing that prototype or any production code.  The short
answer is:

> Use Takahashi selected inversion as the core exact-pricing path whenever the
> direct SPD factor already exists.  It is exact, faster than both current
> pricing implementations, and serves both Sigma's per-cell deletion prices
> and Alpha's adjacent merge prices.  Keep batched factor solves only as a
> fallback for arbitrary nonadjacent inverse blocks.  Do not ship JL probing
> on this path.

## What the pricing operations actually need

Let

```
G = A^T A + Lambda
```

and let `c` be the solution.  For cell `i`, whose local coefficient width is
`w`, constraining its coefficients to zero raises the *regularized* quadratic
objective by

```
p_i = c_i^T [(G^-1)_ii]^-1 c_i.
```

For the three color channels, Sigma contracts the three channel prices with
`LAB_WEIGHTS`; Alpha sums them without those weights.  Thus individual
deletion pricing needs only every `w x w` diagonal block of `G^-1`, not full
inverse columns.

Alpha's merge price for an adjacent pair `(i,j)` additionally needs
`(G^-1)_ij`.  Those pairs are edges of the normal matrix's co-ownership graph.
Every scalar entry of their dense cell block is consequently in the original
matrix pattern, hence also in the filled factor pattern, so selected inversion
returns it exactly.

Selected inversion does **not** provide an arbitrary dense inverse block.
In a sample of 198 random nonedges at 2,400 cells, 196 corresponding scalar
entries were absent from the selected pattern.  Arbitrary nonadjacent merge
queries must therefore retain a batched `factor.solve(E)` fallback.

There is a second scope distinction worth making explicit.  The formula above
is an exact *single-cell* price.  Simultaneously deleting a set `S` costs

```
c_S^T [(G^-1)_SS]^-1 c_S,
```

which is not generally the sum of its individual prices.  Replacing the
current ranker with selected inversion makes every individual decision exact;
it does not by itself certify the cost of a whole exchange batch.

Finally, both Alpha and the Sigma experiment currently compare prices to a
data-residual objective that omits the ridge term, while `G` includes the
ridge.  The price is exact for the regularized objective
`||Ac-y||^2 + c^T Lambda c`.  The ridge is small, but the API and meters should
name or include it correctly.

## Why the Takahashi recurrence fits

With the existing SuperLU settings,

```
permc_spec="MMD_AT_PLUS_A"
diag_pivot_thresh=0
SymmetricMode=True
```

the measured SPD systems have equal row and column permutations and factors
satisfying

```
U = D L^T
```

to roundoff.  The Takahashi/Erisman-Tinney recurrence then computes the entries
of `G^-1` on the lower-triangular pattern of `L` in a backward factor sweep.
All same-cell diagonal blocks are present, as are all original graph-edge
blocks.  No linear solves are issued.

This is a sparse inverse *subset*, not a sparse approximation to the inverse.
Returned entries are exact up to floating-point factorization and recurrence
error.  The work is comparable to sparse numeric factorization rather than
strictly linear in `nnz(L)`; the current scalar Numba recurrence measured about
2.3 times one factorization at the larger tested sizes.  A future supernodal
C++ kernel can use dense block updates, but the scalar implementation is
already decisively faster than the paths it replaces.

SuperLU's symmetric mode prefers diagonal pivots rather than representing a
general contract that factors always have a symmetric triangular shape.
Production code
must validate:

1. `perm_r == perm_c`;
2. `L` is unit lower triangular with a diagonal entry in every column;
3. `U` has nonzero diagonal and agrees with `diag(U) @ L.T` to a
   scale-relative tolerance;
4. every requested block entry is found in the selected pattern.

If any check fails, fall back to batched explicit solves.  In a debug/test
mode, also compare a small random sample against inverse columns returned by
the factor.

## Measured results

All timings are warm best-of-run timings on the current workspace.  The
baseline is Sigma's 48 Hutchinson/Rademacher probes followed by three explicit
solves for each of the cheapest `n/12` cells.

| image / size | cells | unknowns | `nnz(L)` | factor | current Sigma pricing | selected, all cells | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera 128 | 700 | 2,100 | 61,791 | 3.3-4.8 ms | 19.3 ms, 58 exact | 6.8-6.9 ms, 700 exact | 2.8x |
| Pikachu 256 | 2,400 | 7,200 | 232,281 | 11.7 ms | 177.0 ms, 200 exact | 34.8 ms, 2,400 exact | 5.1x |
| Pikachu 384 | 5,400 | 16,200 | 576,594 | 52.8 ms | 1,476.4 ms, 450 exact | 120.4 ms, 5,400 exact | 12.3x |

Accuracy:

- Camera, all 700 blocks versus all explicit inverse columns:
  `1.52e-13` maximum relative block error.
- Pikachu 256, all 2,400 blocks versus all explicit columns:
  `2.35e-12` maximum relative block error and `6.71e-14` maximum relative
  deletion-price error.
- Pikachu 384, 64 sampled cell blocks:
  `1.15e-13` maximum relative block error.
- Enriched width-4 system:
  `1.56e-13` maximum relative block error.

The Alpha path benefits even more because it currently solves a dense batch of
inverse columns.  At Pikachu 256 / 2,400 cells:

| Alpha operation | current | selected inverse |
|---|---:|---:|
| deletion prices | 336.5 ms for 240 shortlisted cells | 30.3 ms for all 2,400 cells |

The all-cell prices agreed with the 240 explicit prices to `1.64e-15`
relative.  For 256 adjacent shortlisted pairs, selected inverse cross-blocks
agreed with the existing inverse columns to `1.02e-12` relative.

The selected values occupy one `float64` array with `nnz(L)` entries.  At the
5,400-cell case this is about 4.6 MB, plus two integer work arrays and the
requested output blocks.  An in-place C++ implementation could overwrite or
reuse factor-pattern storage if lifetime permits.

## What is wrong with the current random shortlist

Hutchinson probing computes the diagonal estimate

```
diag(G^-1) ~= mean(z .* solve(G,z)).
```

It is unbiased, but its variance is controlled by off-diagonal inverse mass.
It can be negative, it does not recover within-cell cross terms, and the code
then applies a second approximation by reading the deletion form as if each
`w x w` block were diagonal.

At Pikachu 256 / 2,400 cells with 48 probes:

- median diagonal relative error: 8.1%;
- 90th percentile: 46.3%;
- maximum: 582%;
- 81 negative diagonal estimates;
- only 83.0% recall for the cheapest 200 cells;
- 93 of the 96 cells actually retired agreed with the exact all-cell ranking.

At camera 128 / 700 cells, only 22 of the 29 actual retirement choices agreed.
The approximation is therefore not merely noisy internally; it changes the
allocator's decisions.

## JL/leverage projection: the correct guarantee and why not to use it here

Define

```
B = [ A ; Lambda^(1/2) ],   M = B G^-1.
```

Then

```
M^T M = G^-1.
```

A random sign projection `R M` therefore produces positive-semidefinite Gram
estimates of each inverse block.  A subspace embedding for each cell's
`w`-dimensional column space gives

```
(1-epsilon) (G^-1)_ii
    <= (RM_i)^T (RM_i)
    <= (1+epsilon) (G^-1)_ii
```

in Loewner order.  With a union bound over cells, the required projection
dimension is `O((w + log(n/delta)) / epsilon^2)`.  Inverting this inequality
also gives a relative multiplicative bound on each resulting deletion price.
That is the useful guarantee.

The interrupted prototype overstates this as relative preservation of every
entry of `G^-1`.  Individual off-diagonal inner products receive an additive
JL error proportional to the two column norms; a near-zero cross entry cannot
have a useful relative guarantee.  The measured off-diagonal relative error
was correspondingly poor.

At 48 projections on Pikachu 256 / 2,400 cells:

- diagonal median relative error: 12.8%;
- 90th percentile: 31.8%;
- maximum: 95.5%;
- full-block shortlist recall: 93.0%;
- time: about 40 ms, already slower than the 35 ms exact selected inversion.

JL improves the bad tail and positivity of the current probe, but `k=48` is
not itself a useful small-epsilon certificate once all cells and a failure
probability are included.  It also needs applications of
`B=[A;sqrt(Lambda)I]`.  Materializing the prototype's
`k x (npix + dimension)` sign matrix is unsuitable for full-resolution HD
(roughly 0.8 GB at 1080p, `k=48`, float64, before other temporaries), and
streaming it still performs `O(k * nnz(A))` work.  This conflicts with the
new fused normal assembly, whose advantage is that it no longer builds or
retains `A`.

JL is the principled fallback only in a different architecture: no accessible
direct factors, but cheap iterative applications of `G^-1` and `B`.  It is not
competitive after the current SuperLU factorization.

## Concrete core integration

Implement one factor-level primitive, independent of image semantics:

```
selected_inverse(
    factor,
    block_width,
    diagonal_blocks=true,
    edge_pairs={}
) -> { diagonal_blocks, edge_blocks, status }
```

Suggested staging:

1. Port the validated scalar recurrence from
   `experiments/sigma_opt/opt_selected_inverse.py` to the C++ sparse solver
   layer, using 64-bit offsets and factor-native scalar/index types.
2. Store selected values on the lower CSC pattern of `L`; build the original
   unknown-to-permuted-unknown map once.
3. Gather all per-cell diagonal blocks and optionally original normal-graph
   edge blocks in the same pass.  Use the pattern already built by fused
   normal assembly to request Alpha's adjacent blocks.
4. Price each SPD block with a tiny fixed-size pivoted elimination, not a
   general pseudoinverse.  Retain a guarded pseudoinverse fallback only for a
   numerically singular block.
5. Sigma: replace `deletion_costs` probing, rough ranking, and shortlist solves
   with all-cell exact prices.
6. Alpha: replace `_inverse_columns` for deletion prices and adjacent
   complementarity.  Keep batched solves for explicitly requested nonedges.
7. Cache selected values with the numeric factor.  Reuse them for every
   channel and every price query until the factor changes; never carry them
   across a Newton line-search geometry whose normal values changed.

For a first integration, keep SuperLU factorization and port only the selected
inverse/gather/tiny-block pricing code.  A later supernodal implementation can
fuse updates with dense `3x3`/`4x4` kernels.  Replacing the factorization or
adding a JL estimator is not required to realize the measured win.

## Sources

- A. M. Erisman and W. F. Tinney, “On Computing Certain Elements of the
  Inverse of a Sparse Matrix,” *Communications of the ACM* 18(3), 1975,
  DOI `10.1145/360680.360704`.
- Lin et al., “SelInv—An Algorithm for Selected Inversion of a Sparse
  Symmetric Matrix,” *ACM TOMS* 37(4), 2011,
  DOI `10.1145/1916461.1916464`.
- Drineas, Magdon-Ismail, Mahoney, and Woodruff, “Fast Approximation of Matrix
  Coherence and Statistical Leverage,” *JMLR* 13, 2012.
- Spielman and Srivastava, “Graph Sparsification by Effective Resistances,”
  *SIAM Journal on Computing* 40(6), 2011.
- SuperLU Users' Guide and SuperLU FAQ, symmetric-mode factorization options.
