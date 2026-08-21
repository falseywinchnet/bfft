# Truth under distortion: a post-FMMT denoising foundation

## Status and source boundary

FMMT is rejected as a denoising model. It remains executable only as a
falsified historical control. Its acceleration work is representation work on
that control and is not evidence for the estimator.

The three papers discussed here are research sources, not repository
instructions. Their theorems do not combine automatically:

- Gonzalez, Cedeno, and Puig prove enclosure and coverage statements for a
  finite-dimensional **linear state-space model** with a declared zonotopic
  mixture noise law.
- Pereg demonstrates **one-shot supervised restoration** from one paired
  input/output image and relates a recurrent patch encoder to iterative sparse
  coding. The paper does not recover a clean image from one unpaired corrupted
  image.
- Balanov, Huleihel, and Bendory analyze EM for **multi-reference alignment**
  under random circular shifts and Gaussian noise. Their current expanded
  version establishes low-SNR convergence and initialization pathologies; it
  does not present a general image denoiser.

What survives the change of problem is a common epistemic discipline: truth
must remain a set of structurally distinct, observation-consistent hypotheses
until independent evidence can falsify all but one. A denoiser must not turn
unresolved alternatives into their spatial average.

## The Cameraman falsification

The supplied three-panel image is enough to reject the present FMMT ontology.
The corrupted center panel is unpleasant, but the hair contour, face, camera,
hand, coat, and tripod are still directly legible. The FMMT panel lowers
pointwise error while converting those visible relations into wide, blocky
regions.

A diagnostic measured on the displayed 384-square panels gives the FMMT image
MSE `0.00357` and SSIM `0.660` against the clean panel, but only `0.483` of the
clean Sobel magnitude on the strongest 20 percent of clean edges. These are
screenshot measurements, not a raw-image benchmark, but the direction is
unambiguous: FMMT wins by replacing uncertain high-frequency structure with a
low-frequency commitment.

That is not a tuning failure. It follows from the state equation:

1. a smoothed provisional chart is formed from the corrupted observation;
2. that chart determines hereditary support and geodesic communication;
3. empirical intensity measures are transported on the resulting geometry;
4. a posterior mean averages the surviving proposals.

The observation proposes its own geometry through a destructive map, and that
geometry then certifies the same proposal. If two thin hair-edge hypotheses
remain possible, their mean is a thick grey edge that was never a hypothesis.
More support, more iterations, a better stopping rule, or faster C++ cannot
repair this circularity.

## What the three papers say together

### 1. Pereg: the source contains a small internal structural vocabulary

Pereg's useful result is not the particular RNN. It is that patches from one
paired image can provide enough representatives of an image-specific typical
set for a compact local inverse to generalize. The source itself therefore
contains a finite empirical vocabulary of repeated structures; a universal
external prior is not always required.

The paper also gives an unusually clean mismatch warning. A predictor trained
to invert Gaussian blur `sigma_s` and applied at `sigma_t` leaves a calculable
residual blur or over-deblurs. A learned inverse is conditional on its
distortion operator even when it generalizes across image content.

For blind denoising we may borrow only the first claim as a prior-construction
principle:

> Build an image-specific typical set from held-out patch relations, but do
> not pretend that a corrupted/clean training pair exists.

### 2. ZMF: probability between bounded hypotheses, ignorance within them

The Zonotopic Mixture Filter separates two kinds of uncertainty. A mode is
chosen probabilistically; conditional on that mode, the disturbance is merely
known to lie in a bounded zonotope. The data can conclusively falsify a mode
history when its innovation lies outside the predicted set. Passing the test
does not prove that the history is true. Because no within-zonotope density was
declared, the observation supplies no legitimate within-mode likelihood.

This is exactly the distinction that FMMT violates. A local residual histogram
or a transport distance does not become a posterior likelihood because it is
convenient to normalize it.

The second transferable construction is safe reduction. When the number of
hypotheses becomes too large, ZMF merges components by a containing enclosure
and sums their weights. It spends precision, not coverage. For images, the
finite modes should be understood as numerical cells in a continuous
distortion bundle, not as ontological names such as Gaussian, impulse, or
speckle.

### 3. MRA: truth is an orbit, and iterative alignment can remember its seed

In multi-reference alignment the identifiable truth is not a signal `x` but
its orbit under the nuisance group. At low SNR, EM has two different slow
directions: magnitude-like errors are visible to second-order invariants while
phase-like errors require third-order information and contract at the much
slower `exp(-SNR^2 t)` scale. In noise-only population EM, Fourier magnitudes
decay but the initialization phases remain fixed. At finite sample size, EM
can first approach truth and later diverge (the Ghost of Newton).

The 2025 version's mini-batch remedy reduces those artifacts, but the deeper
lesson is not “use a smaller batch.” It is:

> Do not align observations to a current template before the data has
> identified the template's orbit. First accumulate invariant evidence; use
> iterative alignment only as a late refinement, if at all.

The paper's comparison with method of moments is especially important here.
More observations directly improve estimates of invariant moments, whereas
population-level EM slowness is not cured by more data.

## One object: the typical-orbit feasible set

Let `P_i x` be an overlapping clean patch of the unknown image. Let `G` be the
compact geometric nuisance group appropriate to the patch: initially finite
translations, reflections, and quarter-turns. Bounded offset and positive
contrast coordinates form a separate photometric chart. Let `theta` be a
continuous distortion coordinate and let `E_theta` be a bounded residual set.
A computational cell may be a zonotope

\[
Z_k=\{c_k+B_k u:\lVert u\rVert_\infty\leq 1\},
\]

but the physical state is the union of its geometric and photometric nuisance
orbits,

\[
\mathcal T=\bigcup_k\left\{a(gz)+b\mathbf 1:
z\in Z_k,\ g\in G,\ (a,b)\in\mathcal A\right\} .
\]

Here `A` is a bounded photometric coordinate set, not an invariance claim over
arbitrary rescaling. `T` is the image-specific structural typical set. It is estimated from
cross-fitted raw patch relations, never from an FMMT or Gaussian provisional
image. The clean-image feasible set is

\[
\mathcal F(y)=\left\{(x,\theta):
P_i x\in\mathcal T^{(-i)},\quad
P_i(y-x)\in E_{\theta_i},\quad
R_{ij}P_i x=R_{ji}P_j x\ \forall(i,j)
\right\}.
\]

Here `T^(-i)` excludes every observation ancestor used by target patch `i`,
and `R_ij` restricts overlapping patches to their common pixels. The last
equation is not a penalty: overlapping local truths must assemble into one
image exactly.

The estimator state is a set-measure on

\[
(x, g, \theta, k),
\]

not one image. Its update has only three lawful operations:

\[
\mathfrak M^+
=\operatorname{EncloseReduce}\left(
\operatorname{PushForward}(\mathfrak M)
\cap \mathcal F(y)
\right).
\]

1. **Push forward** a hypothesis through a known observation or overlap map.
2. **Intersect/falsify** it with new bounded evidence.
3. **Enclose/merge** it without deleting any still-feasible truth.

Probabilities may weight distortion cells only when their prior law is
actually declared. Inside a bounded cell, ignorance remains ignorance. If the
distortion law is unknown, component weights are capacities or retained
coverage, not posterior probabilities.

## Invariants before alignment

For each orbit family, compare patches first through quantities unchanged by
the nuisance action. For translations, the initial hierarchy is:

\[
m_1=\text{mean},\qquad
m_2(\ell)=\sum_r z_rz_{r+\ell},\qquad
m_3(\ell_1,\ell_2)=\sum_r z_rz_{r+\ell_1}z_{r+\ell_2}.
\]

Second-order structure contracts magnitude uncertainty. Third-order structure
is required before phase or edge placement may collapse. In 2-D these become
local autocorrelation and bispectral/triangle moments, evaluated on oriented
BV bonds or paired one-sided residual traces rather than on a blurred scalar
chart.

This creates a strict order:

```text
raw observation
  -> cross-fitted orbit-invariant moments
  -> bounded typical-set components
  -> observation/overlap intersection
  -> safe enclosure reduction
  -> optional componentwise alignment/refinement
  -> point readout only where the surviving component is narrow
```

There is no global posterior mean. Where several components survive, the
system returns a structural medoid plus an ambiguity field, or multiple
representatives. Averaging incompatible edge locations is forbidden.

## What survives from this repository

| Repository mechanism | Decision | Post-FMMT role |
|---|---|---|
| FMMT provisional chart, support birth, geodesic histogram posterior, entropy inertia | Reject | Historical negative control only. |
| FABADA and repeated diffusion/fixed-point denoisers | Reject | Evidence that reusing correlated smoothed states invents information. |
| V3 paired one-sided full-band correlations and spanning-forest phase integration | Retain conditionally | Raw relational proposal and gauge integration; never a support certificate on corrupted data. |
| V3 predecessor/parent records and exact residual complement | Retain | Observation ancestry, overlap bookkeeping, and conservative reverse restriction. |
| Personal deblurrer path-mixture/closure logic | Retain | Multiple operator hypotheses, exact forward/adjoint checks, and explicit common-gauge ambiguity. |
| Zonotopic mixture logic already studied in the deblurrer | Strengthen | Conclusive falsification, coverage accounting, and enclosure-preserving reduction. |
| Meyer oriented jump/BV bond representation | Retain as coordinates | Represents an edge as support, normal, and signed jump rather than a grey annulus. ROF smoothing is not a truth oracle. |
| Compressor/posterizer exact reconstruction and budget ledgers | Retain as engineering discipline | Every representation loss is explicit and testable; no relevance to the denoising objective itself. |
| FMMT vector recurrence and native image kernels | Reusable machinery | Only after a new estimator passes the mathematical and visual gates. |

The segmenter's eikonal front therefore survives only as a sparse causal data
structure. Eikonal distance is not evidence that two noisy pixels share truth.
Transport survives in its literal sense—push-forward of feasible sets and
ancestry—not as a synonym for smoothing.

## First executable experiment

The next experiment is a **typical-orbit set denoiser**, not an FMMT variant.
Its first checkpoint should be deliberately small and auditable:

1. Work on grayscale 2-D only. Extract overlapping raw patches and construct
   disjoint ancestry folds.
2. Describe each patch by DC, second-order autocorrelation, third-order
   triangle moments, and oriented BV bonds. No prefilter is permitted.
3. Cluster only in invariant space. Each cluster becomes a low-rank zonotope
   in patch-coordinate space plus its nuisance orbit.
4. Sweep a continuous nested disturbance radius as quadrature. Cells are not
   named noise families. Falsify a patch/component pair only when its bounded
   innovation and overlap equations are impossible.
5. Reduce the mixture only by outer enclosure and retained-mass accounting.
6. Assemble globally compatible patch components through exact overlap
   restrictions. Use a component medoid for display; preserve ambiguity rather
   than averaging distinct geometries.
7. Keep EM out of the first checkpoint. A later componentwise refinement may
   use mini-batches, but it must be initialized from invariantly identified
   orbit components and monitored for late divergence.

The first test matrix is more important than speed:

- the supplied Cameraman mixed replacement-plus-uniform observation;
- tapered hair, camera rim, hand, coat edge, and tripod subregions;
- clean-input identity;
- noise-only and phase-scrambled negative controls, which must not reconstruct
  the initialization;
- woven texture, line drawing, geometric interface, text, and repeated natural
  texture;
- corruption mechanisms withheld from any calibration pass;
- patch/orbit/disturbance quadrature refinement.

Promotion requires all of the following at once:

1. materially better strong-edge retention than the FMMT screenshot result;
2. no clean-structure deflation;
3. truth coverage by the reported feasible set on synthetic controls;
4. ambiguity that grows when evidence is removed or corruption is increased;
5. invariance/equivariance under the declared nuisance group;
6. no Einstein-from-noise template reconstruction;
7. no late Ghost-of-Newton regression;
8. stable results under representation refinement without a truth-selected
   patch size, noise branch, stopping time, or quality strength.

Only after these gates should the set intersection, invariant-moment
accumulation, and enclosure reduction be specialized in C++.

The first execution is recorded in `TYPICAL_ORBIT_FIRST_RESULT.md`. Global
patch-centre transplantation failed the clean and structural gates. The
surviving checkpoint is local orbit survival plus an all-scale projective
phase veto on an FMMT cleanup after-pass. It preserves more tripod evidence
but does not yet transport the connected oriented bond needed to reconstruct
the tripod as one structure.

## The new principle

Noise does not turn truth into smoothness. It enlarges the set of truths that
remain compatible with what was observed.

A principled denoiser therefore does not ask, “What smooth image best explains
this sample?” It asks:

> Which image-specific structural orbits remain feasible under a continuous
> family of bounded distortions, and what independent evidence can lawfully
> remove them?

Recovery is the contraction of that feasible orbit set. Denoising is only its
final, optional projection.
