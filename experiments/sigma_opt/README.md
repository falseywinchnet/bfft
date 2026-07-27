# Sigma component optimization

Each file takes one component of the sigma round, names the formal target it
is aiming at, and replaces the implementation. Nothing here changes what any
algorithm computes — every file asserts its output against the shipped one
and fails loudly if it drifts.

    PYTHONPATH=.:viewer:experiments .venv/bin/python \
        experiments/sigma_opt/opt_ridge_scan.py

## Results, Pikachu 256 px / 2,400 cells, warm

| component | file | baseline | optimized | speedup | agreement |
|---|---|---:|---:|---:|---|
| gradient accumulation | `opt_tree_accumulate.py` | 10.2 ms | 0.81 ms | **12.6x** | 9.5e-17 |
| per-cell ridge scan | `opt_ridge_scan.py` | 60.7 ms | 10.4 ms | **5.8x** | exactly 0, zero axis/offset disagreements |
| coupled solve, both fields | `opt_normal_assembly.py` | 68.4 ms | 38.4 ms | **1.8x** | normal matrix 1.4e-16 relative, field 2.7e-15 |
| objective evaluation | `opt_score_cache.py` | 47.1 ms | 25.8 ms | **1.8x** | bit-identical |
| two-label geodesic walk | `opt_dijkstra_bucket.py` | 31.1 ms | 19.5 ms | **1.6x** | distance gap exactly 0, zero untied owner mismatches |
| Schur addition gains (96) | `opt_schur_gains.py` | 128.6 ms | 86.9 ms | **1.5x** | 2.0e-14 relative, identical ranking |

Composed on one `receiver_guided_graph` Newton step (7 walks, 7 coupled
solves, 6 scores): **980 ms -> 561 ms, 1.75x**, with every number it produces
unchanged.

A correction to the numbers in `notes/claude_trial_sigma.md`: the 843 ms and
761 ms I first reported for `addition_gains` and `deletion_costs` were cold
first calls. Warm, they are 129 ms and far less. The speedups above are all
warm-against-warm.

## The formal targets

**Monotone priority queue** (`opt_dijkstra_bucket.py`). Dijkstra pops in
nondecreasing key order, so a comparison heap is more structure than the
problem needs. With bucket width no larger than the smallest edge cost, a key
in `[k*delta, (k+1)*delta)` can only relax into a strictly later bucket, so
buckets may be emptied in arbitrary internal order and the answer is exact.
Dial's argument, `O(E log V) -> O(E + range/delta)`. Ten buckets suffice under
the uniform metric, ~55 under the perturbed metric weight descent runs in.
The heap turned out not to be the whole cost: interleaving the eight step
costs to node-major order, so a node's neighbourhood is one contiguous read
instead of eight images apart, is worth as much again.

**The sort was already done** (`opt_tree_accumulate.py`). Because every tree
edge costs at least `delta`, a child always lands in a later bucket than its
parent. Emptying buckets in descending order is therefore a valid reverse
topological order, and the `argsort` over 131,072 states becomes a counting
sort over a few hundred buckets. The direction taken into each node was known
at relaxation time, so the eight-way search per edge becomes one array read.
Both facts are inherited from the queue rather than assumed.

**One sinogram, not sixteen histograms** (`opt_ridge_scan.py`). The scanned
quantity is the cumulative Radon transform of the weighted residual,
restricted to a cell. Every angle reads the same pixel and the same residual,
so pixels belong outermost and angles innermost: 16 image sweeps and 48
temporary allocations collapse to one pass with none, and consecutive pixels
usually share an owner, so the live cell's whole accumulator stays inside
16 KB of cache.

**The algebra was already local** (`opt_schur_gains.py`). A candidate cell is
supported on ~113 pixels but the baseline multiplies the whole `(npix, 3n)`
design against it. `A^T B = A[near]^T C`, and every later contraction against
`cross` pushes through `A[near]`, so `A @ incumbent` is computed once for all
candidates instead of once each. The remaining solves are irreducible, but
they form one right-hand-side block rather than `3C` separate calls.

**The target never changes** (`opt_score_cache.py`). `score` decomposes
`model.rgb` on every call. It is the source image. Memoizing on identity plus
checksum removes 25 ms per call, six times per Newton step. The cache is
tested for staleness, not just for speed.

**The design matrix is an intermediate nobody reads**
(`opt_normal_assembly.py`). Each pixel contributes a rank-one update touching
exactly two cell blocks, so the normal matrix is finite-element assembly:
compute the block pattern once from the (owner, runner) pairs — it is the same
co-ownership graph the receiver-guided Newton step builds its Laplacian on —
then scatter-add per-pixel outer products straight into a fixed CSR data
array. No COO, no sort, no sparse matmul, no design matrix; `A^T b` rides
along in the same pass. The pattern depends only on ownership, and the
cartoon and texture systems share ownership, so it is built once per pair
rather than once per field. The prediction step's six fancy-index gathers
become one pass with no temporaries. After this, `splu` is 10 of the
remaining 19 ms per field — the floor is now genuinely the factorization.

## Deliberately not done

**Symbolic factorization reuse.** The 10 ms `splu` recomputes ordering and
symbolic analysis every call, though the pattern is fixed across a line
search. The fix is CHOLMOD's `analyze`/`cholesky` split via `scikit-sparse`,
which is a new dependency in your venv — flagged rather than installed.

**Selected inversion for `deletion_costs`.** The proper target is the
Takahashi / Erisman–Tinney recurrence: diagonal blocks of `G^-1` on the
factor's own pattern, replacing 48 stochastic probes plus ~600 exact solves
with one backward pass. It is the most satisfying target in the set. It needs
`L`, `U` and both permutations pulled out of SuperLU and a careful
implementation, and the component it serves was worth +0.10 dB. Deferred on
value, not on difficulty.
