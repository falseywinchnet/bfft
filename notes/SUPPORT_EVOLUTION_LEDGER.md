# Support evolution ledger

Date: 2026-07-26

## Rejected architecture: top-K candidate expansion

The first probe replaced the terminal owner/runner pair with a dense top-K
partition. It was explicitly belayed and is not a candidate for the live
algorithm. The code is archived at
`experiments/archive/rejected_topk_support_evolution.py`.

The probe is still informative as an ablation of *signals*:

- Diffusing candidate-support scores reduced boundary count and disconnected
  hard-label fragments on all four controls.
- Preserving a measured per-site capacity reduced fragmentation further.
- Residual-covariance, determinant-one anisotropy improved every tested
  top-K control and was substantially stronger than the temperature heuristic.
- Receiver Gauss-Newton bias updates improved all four experimental fits.
- “High temperature at edges” was not a sufficient temperature estimator and
  generally hurt the composite objective.
- A coefficient-only graph-fusion penalty was negative.
- Mild survival pressure did not extinguish a single support. Forcing enough
  pressure to delete supports reduced quality, and refitting after deletion
  reduced it further.

The last three points directly constrain the active owner/runner work:
temperature must be learned from reconstruction response rather than read from
edge strength; merging cannot be imposed on coefficients after geometry; and
hard deletion is only justified after the native support dynamics actually
drive a cell's measured mass to zero.

## Active architecture: exact two-owner support

Keep exactly the current owner and runner at every pixel. Evolve:

1. additive reach on the ownership-response graph;
2. determinant-one per-site shape as a continuous correction to the BFFT
   geodesic distance;
3. independent per-site boundary temperature;
4. a capacity/boundary budget on the same owner/runner mass Jacobian.

Soft merging means two adjacent supports become functionally identical while
one loses competitive mass under this joint solve. Hard deletion is merely the
zero-mass event; it is not a separate greedy fusion pass.

## 2026-07-27 — one-decomposition Wasserstein allocation

The pass-trace interpretation was rejected before scoring. `meyer_trace`
contains intermediate convergence states of one TGFD solve, not a legitimate
scale-time population. Allocation now sees exactly one finished
`meyer_split`; annealing occurs only in the transport plan.

### Runtime diagnosis

The browser prototype is fast because its hot loop is one hard `Int32Array`
owner per pixel plus local Euclidean radius updates. A V8-shaped benchmark at
1800 px measured 2,323 sites, 9.5 ms for the local owner rebuild, and 27.9 ms
for one full pixel-statistics pass. It carries no runner-up, overlapping
support raster, coupled sparse solve, or repeated support sampling.

The fixed-canopy Python prototype at 256 px / 3,707 cells took 6.30 s and
peaked at about 669 MB RSS. Of 4.62 s spent in canopy evolution, 3.88 s was
25 repeated calls to `support_samples`; it ultimately materialized 490,978
compact-support samples. This is the principal reason it blooms at HD.

### Falsified variants

1. **Nested hard allocation tree.** Soft branch probabilities were rounded
   inside the parent's inherited hard territory. Site IDs became irreversible
   stripes across the whole image. At 128 px / 761 cells it reached only
   24.31 dB. Ancestry is therefore not ownership.
2. **Release only at the end.** Replacing the nested labels by one global
   Euclidean wavefront from the same barycentres reduced the 761-cell result
   to 20.87 dB. Global completion is necessary but final release alone cannot
   repair unmigrated sites.
3. **Cartesian instability with geodesic drawing.** Global geodesic
   reassignment plus centroid migration improved 734 cells to 25.08 dB at
   128 px, but at 475 px produced 6,041 cells. The allocator drew cells in the
   transport metric, then used ordinary x/y covariance to decide division; a
   long curved support was consequently misread as broad two-dimensional
   area.
4. **BFFT response as local population evidence.** The nonlocal glass/texture
   field bled 8--16 pixels into the constant white frame. White carried only
   5.1% of support mass but received 1,061 of 6,041 site centres. Gating BFFT
   magnitude by variation in the unchanged source reduced this false local
   evidence, but did not by itself fix the split law.
5. **Transport radius alone.** Transported RMS radius permits the desired
   broad white panels: at a permissive HD threshold it used 65 cells total and
   only two white centres. It was only 13.07 dB because a scalar radius cannot
   demand a thin cross-edge support.

### Supported hybrid

The current experiment, `experiments/wasserstein_allocation_tree.py`, uses:

- one fixed BFFT cartoon/texture/glass geometry;
- a two-nearest soft transport plan with temperature in support-horizon units;
- centroid migration in the shared domain;
- mass-conserving bifurcation into the two branch barycentres;
- transported extent for longitudinal division;
- BFFT precision, scaled by support horizon squared, for transverse cost;
- coherent-eigenvalue suppression so contours remain cheap along their
  tangent;
- one final hard site-ID diagram.

On the original 475x475 Pikachu, the support integral predicts 718 cells.
At a threshold yielding 793 hard cells, the site-ID map contains large clean
white and black panels with narrow contour cells. The plain affine readout is
22.28 dB; one measured bounded `tanh` ridge per cell raises it to 25.66 dB,
two to 26.13 dB, and four to 26.45 dB. At the denser 4,078-cell operating
point the affine readout reaches 29.02 dB.

This separates the remaining problem: the sub-1,000 support geometry is
credible, while the hard affine local function space is not. The next test is
a soft coupled readout or a curved transported ridge coordinate on the same
frozen 793 sites, not another density heuristic.

### Runtime audit of the 793-cell HD default

A profiled 475x475 run measured 0.16 s for the fixed BFFT geometry, 3.99 s
for allocation, and 0.44 s for the enriched fit itself. Diagnostic
decomposition scoring adds about 0.5 s and is not required in a live renderer.
Plotting, font layout, and cold imports account for much of the remaining
standalone wall time.

The dominant reducible allocation work is branch balancing. Eighteen branch
rounds currently use fourteen parallel bisection reductions apiece;
`_balanced_branch_barycentres` costs 1.07 s and the run executes 1,056
`bincount` reductions costing 1.03 s. The split boundary is a weighted median
of the transported branch coordinate. It can therefore use the same fused
histogram/cumulative construction as the optimized residual-ridge scanner:
one pixel pass, one small per-cell scan, no numerical descent.

The soft moments cost another 0.54 s and are likewise finite-element-style
scatter sums suitable for one fused C++ pass. The remaining sequential object
is the global transport walk between population changes. Two routes are now
well defined:

1. retain the causal generations but move their bucketed walk and all moments
   into C++; or
2. emit several allocation-only descendants from one frozen predecessor tree,
   release them into the shared domain, and perform one global correction.

The second route must be judged by final site IDs: allocation ancestry may be
reused, but inherited pixel ownership remains rejected.

### Runtime experiments after the audit

The weighted-median replacement was implemented as a compiled fused histogram
and prefix crossing. It is a useful kernel but not a useful default:

- 64 bins produced 813 cells / 25.41 dB;
- 256 bins produced 811 cells / 25.51 dB;
- exact bisection produced 793 cells / 25.66 dB.

The histogram itself costs only about 0.10 s, versus 1.07 s for exact branch
balancing, but its small placement errors keep one or two leaves unstable and
consume all 24 rounds. The exact method stops after 19 rounds, so the complete
runs are both about 4.2 s. Approximate branching is therefore opt-in.

Direct timing shows the exact HD allocation consists of about 1.92 s of
two-label transport, 1.07 s of branch quantiles, and 0.54 s of soft moments.
The exact monotone bucket queue now retained in
`port_needed/monotone_bucket_transport.py` reduced the walks
to 1.56 s in an unintegrated test and reproduced distances exactly (label
differences were tied sites). This is worthwhile, but cannot alone produce a
sub-second solve: whole transport rounds have to disappear.

A transport pyramid does remove rounds without introducing another target
decomposition. Restricting the one frozen 475-pixel geometry to 192 pixels,
solving there, prolonging 128 normalized sites, and applying five HD
corrections produced 783 cells / 25.69 dB. The HD allocation fell from about
4.2 s to 1.15 s, with about 0.33 s for the coarse allocation. This is the
current supported acceleration path. Four HD corrections fall to about
0.95 s but also to 25.16 dB; the fifth correction carries real information.

### Missing criterion found: metric extent was diagnostic-only

The experiment computed the largest eigenvalue of the transported
cell/support product,

`lambda_max(Q_i^(1/2) C_i Q_i^(1/2))`,

and its unstable eigenvector on every round, but used only scalar transported
RMS radius to decide whether to split. Thus the BFFT support could orient a
branch after another criterion requested it, but could not request density
where a narrow support direction remained unresolved.

On the 793-cell Pikachu result, the final metric-extent p90 was 6.71 while its
maximum was 191.91. Enabling a second, local split condition gives:

- metric limit 16: 1,031 cells / 26.63 dB;
- metric limit 8: 1,252 cells / 27.87 dB, p90 3.42, maximum 26.29.

The matched transport-radius-only control used 1,235 cells / 27.56 dB and
left a maximum of 143.11. The 0.31 dB gain is therefore not merely a larger
population: the support tensor spends it in unresolved directions and reduces
the outlier tail by more than 5x.

At 128 pixels the same limit is inert on camera (218 cells / 25.96 dB) and
grass (130 cells / 21.82 dB), because scalar transport had already brought
their metric maxima below eight. On astronaut it activates only where needed:
85 cells / 19.42 dB becomes 117 cells / 20.15 dB and the maximum falls from
19.05 to 7.82. This is evidence that the criterion is selective rather than a
disguised global density multiplier.

This also supplies a closed-form analogue of SAD's learned aspect parameter.
SAD uses a determinant-one per-site metric

`G_i = exp(a_i) u_i u_i^T + exp(-a_i) (I - u_i u_i^T)`,

learning both `u_i` and `a_i`. Our frozen BFFT tensor supplies the surrogate
metric, while the achieved covariance supplies the discrepancy. The desired
shape condition is

`Q_i^(1/2) C_i Q_i^(1/2) = alpha_i I`.

Its eigenvectors give the corrective axis; its largest eigenvalue prices
missing directional density. The eigenvalue *ratio* is not by itself a safe
split trigger because a genuinely rank-one contour support should have a very
large image-space aspect ratio. The absolute whitened extent is the disciplined
quantity.

### Topology is an interleaved state, not a reusable inner graph

The direct multibranch experiment exposed the same scope law as TGFD's
changing-`g` Bregman sweeps. Owner/runner adjacency is the active set of the
min-plus geodesic closure; a population event changes that active set. On the
475-pixel Pikachu control, mapped parent ownership changes by 10--24% over
the productive middle rounds, while 6--14% of the next co-ownership edges are
new. Even the late binary tail still moves 6--10% of pixels and replaces
4--5% of edges per event.

The direct limit-4 jump emits 26 sites from a one-site state with no
co-ownership edge and 185 sites before a nontrivial parent graph exists.
This explains both its speed and its geometry deficit: it predicts topology
from covariance before topology has emerged. The corrected direction is
Meyer-style interleaving--one local event, immediate topology refresh--made
fast by incremental min-plus updates rather than by freezing several
generations. Derivation and full measurements:
`notes/TOPOLOGY_INTERLEAVE.md`.

The follow-up proved exact incremental insertion: first and second distance
fields agreed bit-for-bit with full recomputation on all randomized and image
events. Late Pikachu batches touched only 6--14% of pixels and refreshed
topology 20--43x faster. Three geometry controls were negative: fixed-point
insertion (998 cells / 24.45 dB), multi-anchor growth (1,490 labels / 22.37
dB), and insertion after a mature round-13 graph (881 cells / 24.56 dB)
all lost to the 1,009-cell / 25.95 dB moving-site control. This localizes the
next requirement: exact dynamic source replacement, not more monotone growth.
