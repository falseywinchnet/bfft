# Topology interleaving: the Meyer lesson for transport cells

Date: 2026-07-27

## The shared phenomenon

The input to each TGFD subproblem changes after the other subproblem moves:

\[
g_u^k=f-v^k,\qquad g_v^k=f-u^{k+1}.
\]

A Split Bregman sweep for either subproblem is

\[
u^{k+1}
=
(cI-\eta\Delta)^{-1}
\left[cg^k-\eta\,\operatorname{div}(d^k-b^k)\right],
\]

\[
t^{k+1}=\nabla u^{k+1}+b^k,\qquad
d^{k+1}=\operatorname{shrink}_{1/\eta}(t^{k+1}),\qquad
b^{k+1}=b^k+\nabla u^{k+1}-d^{k+1}.
\]

The shrinkage changes the active gradient set

\[
\mathcal A^{k+1}
=
\{e:\lVert t_e^{k+1}\rVert>1/\eta\}.
\]

Thus the edge topology seen by the next sweep changes. TGFD is fast because
it does **one** coherent sweep against the current \(g\), carries the
Bregman state on the fixed pixel-edge lattice, changes \(g\), and immediately
lets the active set change again. Solving an inner problem to convergence
would polish the topology of an obsolete right-hand side.

The transport-cell system has the same structure in min-plus form. For fixed
BFFT metric \(M\), sites \(p_i\), and additive reaches \(a_i\),

\[
D^k(x)=\min_i\{d_M(p_i^k,x)-a_i^k\}.
\]

The owner, runner, predecessor forest, and co-ownership graph are the active
set of this minimum. A population or reach event changes the source impulses
\((p_i,a_i)\), and therefore changes that active set on the next global walk.
The fixed operator is not the current cell graph. It is geodesic closure on
the fixed pixel substrate.

This identifies the error in frozen local recursion: it performs several
population events while retaining the active set of one obsolete min-plus
right-hand side.

## Measured topology churn

`experiments/wasserstein_allocation_tree.py --trace-topology` now records
three diagnostic quantities without changing allocation (the bookkeeping is
off by default so timing runs remain clean):

- `topology_pixel_change`: after mapping every new child back to its parent,
  the fraction of pixels whose parent owner changed;
- `topology_edge_jaccard`: overlap of the previous co-ownership graph with
  the new graph mapped into previous-parent labels;
- `topology_new_edge_fraction`: fraction of mapped current edges absent from
  the previous graph.

On `Downloads/25.png`, 475 pixels, binary metric limit 8:

| round | sites before -> after | pixels changing parent | graph Jaccard | new edges |
|---:|---:|---:|---:|---:|
| 5 | 16 -> 32 | 27.1% | 0.619 | 18.8% |
| 8 | 127 -> 244 | 21.9% | 0.841 | 8.6% |
| 12 | 813 -> 904 | 17.0% | 0.844 | 9.2% |
| 16 | 1051 -> 1082 | 10.1% | 0.903 | 4.9% |
| 20 | 1162 -> 1178 | 7.5% | 0.904 | 5.2% |
| 24 | 1237 -> 1252 | 6.7% | 0.924 | 3.9% |

Topology change is therefore part of the computation, not a small rendering
correction.

The direct metric-limit-4 experiment is even more diagnostic:

| round | sites before -> after | largest direct branch | pixels changing parent | new edges |
|---:|---:|---:|---:|---:|
| 1 | 1 -> 26 | 26 | unavailable | unavailable |
| 2 | 26 -> 185 | 25 | 0.0% | 0.0% |
| 3 | 185 -> 546 | 14 | 19.0% | 4.4% |
| 4 | 546 -> 946 | 8 | 23.5% | 14.2% |
| 6 | 1158 -> 1249 | 3 | 16.6% | 11.2% |
| 10 | 1428 -> 1469 | 2 | 9.5% | 6.1% |

The first jump predicts roughly five binary levels from a one-site state with
no co-ownership edge at all. The second still collapses to the same original
parent under the comparison. These jumps use the frozen covariance tensor,
but they cannot use topology that has not yet emerged. Their good MSE is
evidence that the tensor is informative; their poorer geometry is evidence
that the missing graph matters.

## The scope law

The Meyer result says precisely what may and may not persist across a changing
objective.

### May persist

- the fixed BFFT pixel metric and edge costs;
- fields stored on the fixed pixel/edge substrate;
- one-stage residual or transport stress at fixed pixel coordinates;
- site state carried through an explicit parent-child map, without treating
  ancestry as ownership.

### Must be refreshed after every population/reach event

- owner and runner;
- the predecessor forest;
- the measured co-ownership graph;
- its Hessian/Laplacian;
- branch covariance and unstable direction;
- any momentum constructed from those changing objects.

This matches the Meyer momentum result: the flux may persist, but momentum
constructed for one changing flow must restart.

## Consequence: interleave events, do not freeze generations

The corrected iteration is

1. apply one exact geodesic closure for the current sites;
2. reduce soft mass and support covariance once;
3. apply one local population/reach event;
4. immediately refresh geodesic topology;
5. carry only fixed-substrate memory into the new topology.

This is not an argument for many slow global walks. It says the acceleration
must preserve the topology refresh rather than skip it.

The useful implementation route is an **incremental min-plus closure**.
Inserting a new source changes

\[
D_{\mathrm{new}}(x)
=
\min\{D_{\mathrm{old}}(x),\,d_M(p_{\mathrm{new}},x)-a_{\mathrm{new}}\}.
\]

With fixed existing sources this is an exact decrease-only shortest-path
update: seed the existing bucket queue at the new source and visit only
pixels whose first or second label improves. Owner/runner topology is allowed
to change immediately, but unaffected regions are not walked again. Several
population events can share one live queue without pretending that their
supports share one frozen graph.

Moving an existing source or increasing its distance is not decrease-only.
The first experiment should therefore separate the effects:

- retain the existing site during insertion;
- insert the new mass-balanced site;
- update topology incrementally;
- postpone centroid migration to a correction sweep, or express reach
  changes as monotone source additions during the event phase.

The final multi-source result remains ancestry-free.

## A topology-aware direct-branch safeguard

There is a cheaper intermediate experiment before the dynamic queue:

\[
\chi_i^k
=
\frac{
 \mu\{x:o^k(x)=i,\,
        \pi(o^{k+1}(x))\ne i\}
}{
 \mu\{x:o^k(x)=i\}
}.
\]

Here \(\pi\) maps a child to its previous parent. \(\chi_i\) is the measured
topology churn of parent \(i\). Direct multibranching is permitted only when
the previous event gave small \(\chi_i\); high-churn cells receive one binary
event and a topology refresh. This uses one stage of support memory, not an
error ranking, candidate search, deletion, or top-K selection.

The prediction is falsifiable:

- it should retain most of the direct method's runtime gain in stable broad
  panels;
- it should recover binary-quality slivers where ownership is still moving;
- if it does neither, topology churn is descriptive but not causal.

## Incremental insertion experiment

`experiments/incremental_topology_interleave.py` implements the exact
decrease-only case. Given a settled two-label solution, it seeds only newly
inserted sources into the existing heap and applies the production relaxation
and tie rules unchanged.

The pruning law is exact. If a new source is not among the two best labels at
pixel \(x\), then it cannot become top-two after crossing \(x\). Let \(s_2\)
be the old second-order distance and let edge \(x\to y\) cost \(w\). Both the
new-source distance and \(s_2\) obey the graph triangle inequality:

\[
d_{\rm new}(y)=d_{\rm new}(x)+w,
\qquad
s_2(y)\le s_2(x)+w.
\]

Therefore \(d_{\rm new}(x)\ge s_2(x)\) implies
\(d_{\rm new}(y)\ge s_2(y)\). A rejected wavefront never needs to be revived.

Measured on 12 randomized batches and seven batches sampled from the
475-pixel Pikachu population trajectory:

- maximum first-distance error against a fresh global walk: exactly 0;
- maximum second-distance error: exactly 0;
- untied owner mismatch: 0;
- runner mismatch with a different distance: 0.

| allocation round | sites | affected pixels | incremental | full walk | speedup |
|---:|---:|---:|---:|---:|---:|
| 3 | 4 -> 8 | 47.7% | 25.3 ms | 87.8 ms | 3.5x |
| 10 | 332 -> 424 | 57.2% | 26.2 ms | 104.5 ms | 4.0x |
| 13 | 590 -> 653 | 28.9% | 11.0 ms | 93.6 ms | 8.5x |
| 17 | 791 -> 822 | 13.5% | 5.0 ms | 98.7 ms | 19.6x |
| 20 | 886 -> 909 | 6.2% | 3.0 ms | 128.8 ms | 43.4x |
| 24 | 986 -> 1009 | 11.5% | 5.0 ms | 132.3 ms | 26.3x |

The global-walk cost stays roughly fixed while the true topology event
becomes local. The current batch rebuild discards that locality.

## Three useful negatives

The exact primitive does not by itself define correct support evolution.

1. **Fixed old point plus one new point.** Starting from one site, retaining
   every established source and inserting only the farther branch produced
   998 cells / 24.45 dB. Late insertions eventually changed almost no pixels.
   The old point prevented the intended replacement from occupying its
   support.
2. **Growing multi-anchor support.** Giving the parent an additional anchor
   at one branch and creating a new label at the other remained exactly
   incremental, but 1,490 labels / 2,979 anchors fell to 22.37 dB. A
   min-distance skeleton can only purchase territory; without retiring old
   influence it over-occupies the domain.
3. **Mature-graph hybrid.** Running the established topology-forming process
   through round 13, then freezing those sites and continuing by exact
   insertion, reached 881 cells / 24.56 dB. The established 24-round control
   reached 1,009 cells / 25.95 dB. Site replacement remains necessary after
   coarse topology has emerged.

These are not failures of incremental topology repair--every incremental
distance still matched a full walk exactly. They falsify monotone-only
support evolution.

## Next primitive: exact local source replacement

The sharper Meyer analogy is that Bregman shrinkage changes topology in both
directions: gradients activate and deactivate. Persistent substrate state
does not mean persistent active influence.

Moving or replacing site set \(S\) should use a dynamic deletion/insertion:

1. mark
   \[
   I=\{x:o(x)\in S\ \text{or}\ r(x)\in S\},
   \]
   the pixels whose stored top-two answer depends on a changed source;
2. invalidate first and second states inside \(I\);
3. seed the boundary of \(I\) with the unaffected owner and runner states
   immediately outside it;
4. seed the new positions of the changed labels;
5. run the identical two-label relaxation only inside \(I\).

Why boundary top-two states suffice: if a third unaffected label cannot beat
the two unaffected labels at the point where its shortest path enters \(I\),
the same triangle-inequality argument shows it cannot beat them deeper along
that path. Every entry edge is seeded, so paths that leave and re-enter are
also covered.

This experiment must first compare source replacement against a complete
walk with zero first/second distance error. Only after that proof should it
replace global walks in allocation.

## Central conclusion

The Meyer analogy does not say that topology can be held fixed. It says the
opposite: make the global operator cheap enough that the right-hand side and
its active topology may change after every useful local event. Preserve the
transport substrate; refresh the graph.
