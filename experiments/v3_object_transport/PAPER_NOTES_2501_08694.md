# Bayesian multifractal segmentation: reusable evidence and rejected inference

Source: K. M. León-López, A. Halimi, J.-Y. Tourneret, and H. Wendt,
“Bayesian Multifractal Image Segmentation,”
[arXiv:2501.08694](https://arxiv.org/abs/2501.08694), v2.

## What transfers to V3 object transport

The paper's wavelet leader is a scale-causal local observable. For a dyadic
cell `lambda_(j,n)`, it is the supremum of normalized wavelet-detail magnitude
over the spatial neighborhood `3 lambda_(j,n)` and every finer scale contained
there. This is materially different from averaging independently inferred
segment kernels across resolutions: a persistent irregularity survives into
its parent chart before any label exists.

The paper also separates two content coordinates:

- the centered log-leader chart itself;
- the affine scale law of its mean and variance, whose slope and intercept
  encode regularity and multifractality.

The V3 adaptation computes orthonormal Haar details at every possible dyadic
scale for the target and all three exact fused Meyer constituents. It retains
every raw mean/RMS coordinate and, separately, the closed-form slope,
intercept, and residual of the mean and variance scale laws. All controls are
jointly whitened using every numerical covariance mode. V3 regions and
boundaries are unchanged.

This evidence has the desired semantics on the 256 atlas:

- cup/plate content is strong while cup/table is weaker and the spoon remains
  distinct;
- flag-blue/flag-red is positive while all sampled flag/suit relations are
  near zero or negative;
- repeated coins share material regularity but do not become one instance;
- checkerboard receives little global organization, correctly leaving that to
  the role coordinate;
- the illustrated ear tips differ from the flat black surround even though
  their colors agree.

## What does not transfer

The paper's final inference assumes a chosen class count `K`, a four-neighbor
Potts potential, learned spatial and inter-scale granularity coefficients,
patchwise k-means initialization, 300 Gibbs iterations, and a MAP label
readout. Those are inappropriate for the requested object stack: they impose
labels and homogeneity rather than measuring participation, and they are
iterative. None is implemented here.

The paper's parent/child graph is conceptually useful, but a literal multiplex
of the three independently inferred V3 segmentations is not sufficient. Its
full versus shuffled-alignment controls are too close on coffee and sometimes
favor the null. Exact pixel overlap tells us where scale atoms coincide; it
does not tell us which structural support a highlight should inherit.

## Current interpretation

Wavelet leaders are promoted as a distinct **content regularity chart**, not
as an object kernel. Their visual behavior proves why: the flag and face can
share regularity without sharing an object; coins share material without
sharing instance identity; and checker squares share scale law without the
leader chart supplying board-wide role.

Three typed transports were tested. A directed transition fibre retains the
outside-minus-inside leader change; a complete ordered endpoint fibre retains
both absolute endpoint charts. Ordered endpoints give a genuine resolution-
stable astronaut gain against their shuffled correspondence, but coffee
remains proposal-dominated. Treating leader similarity itself as a dense heat
generator fails more strongly, both directly and after exact Schur conjunction
with the independent boundary-role kernel. Generic material correspondence
diffuses across instances and can strengthen shuffled controls.

The surviving interpretation is stricter: leader content is a participation
coordinate in the complete positive kernel algebra, while structural transport
must remain sparse and incidence-derived. The next problem is to extract
object idempotents from that algebra without a chosen class count, threshold,
or iterative clustering rule.
