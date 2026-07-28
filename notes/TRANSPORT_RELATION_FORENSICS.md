# Transport relation forensics

This experiment asks whether disconnected regions carry enough information in
the one-pass transport representation to recognize a common object role. It
does not merge regions and does not train a classifier.

`experiments/transport_relation_forensics.py` represents each empirical
distribution with deterministic random Fourier features. Euclidean distance
between their weighted means estimates kernel maximum-mean discrepancy. Cell
area and literal interface length are the only aggregation weights.

The viewer exposes independent anchor maps for:

- target colour;
- the complete interior transport state;
- transport action alone;
- the transport metric tensor;
- signed inside/outside boundary transitions in the boundary-normal frame;
- the same boundary relation with target colour restored.

Clicking a part changes only the forensic anchor.

## Pikachu control

At 256 pixels, the top black ear tip is exactly colour-identical to the black
background, yet the full interior transport distribution reverses the
association:

| Anchor: top black tip | tail | body | black background |
| --- | ---: | ---: | ---: |
| colour | 0.371 | 0.302 | **1.000** |
| interior transport state | **0.707** | 0.650 | 0.599 |

One-dimensional distribution ablations show that the reversal is carried by
transport energy, metric trace/orientation, and weakly by null support.
Cartoon state strongly makes the opposite prediction.

At full 475×475 resolution, the robust claim is the disconnected tail/body
association:

| Transport representation | tail→body | tail→black | margin |
| --- | ---: | ---: | ---: |
| action + metric tensor | 0.899 | 0.681 | +0.218 |
| metric trace | 0.904 | 0.671 | +0.233 |

These means were measured over sixteen independent 1024-feature kernel
projections. The sign of the margin was stable.

The ear-tip result becomes weaker at full resolution. Action and metric trace
individually lean toward Pikachu, but their joint empirical distribution does
not yet separate the tip cleanly from black. That is the current hard probe.

## Rejected transform: rotation-quotiented twist spectrum

The normalized traceless metric tensor is a spin-2 field. We histogrammed its
phase, transformed the histogram with an FFT, and compared magnitude spectra,
which exactly removes global rotation. This did not recover object
affiliation:

- tip→tail 0.397;
- tip→body 0.320;
- tip→black 0.566.

Therefore the useful information is not merely anisotropic shape modulo a
single global rotation. Absolute/local phase, action magnitude, or a
higher-order spatial relation is necessary.

## Coffee / table falsification

The `skimage.data.coffee` scene is a stricter counterexample. Evaluation-only
geometric masks were used for cup, plate, and table; they do not enter the
model. At the current object waterline the plate and table have already
collapsed to the same hard part, so a part-level descriptor cannot possibly
recover their distinction.

Below that waterline, sixteen 1024-feature projections gave:

| empirical measure | cup–plate | cup–table | plate–table |
| --- | ---: | ---: | ---: |
| action | 0.167 | 0.403 | 0.368 |
| metric tensor | 0.367 | 0.638 | 0.336 |
| action + metric | 0.368 | 0.606 | 0.328 |

Thus marginal equality is the wrong object. A multi-hop lazy-walk
autocorrelation of the action/metric field also failed: cup–plate cosine
similarity was 0.830, below cup–table at 0.889. Adding normalized
boundary-to-core depth did not reverse the ordering. These are recorded
negatives, not hidden tuning opportunities.

The viewer therefore exposes **anchor-cell** likeness fields on the canonical
connected support fragments, before the object hierarchy. They show whether
the raw information exists locally without allowing the current grouping to
pre-answer the question.

## Phase-preserving graph scattering

The rejected twist spectrum took the magnitude of a global orientation
transform too early. The replacement propagates the complex spin-2 metric
director on the literal support graph and only then takes modulus:

`|P^(2^(j-1)) z - P^(2^j) z|`.

Second-order bands retain the arrangement of those responses at larger
scales. A rigid rotation multiplies the director by one unit complex phase,
which commutes with diffusion and disappears under the final modulus. The
implementation verifies this invariance directly.

At 256 pixels:

| Anchor relation | scattering likeness |
| --- | ---: |
| tail→body | 0.864 |
| tail→black | 0.643 |
| black ear tip→body | 0.741 |
| black ear tip→black | 0.591 |

At full 475×475 resolution, tail→body remains 0.886 versus 0.705 for
tail→black across sixteen independent kernel projections. Ear-tip support is
weak but stable and scale-local: with only the first graph neighborhood it is
0.459 to body versus 0.372 to black; after five dyadic bands it is 0.385
versus 0.361. Pooling scale therefore destroys information. The complete
scale-indexed response must remain visible.

Coffee remains a negative at every scale. Graph scattering registers repeated
transported geometry; it does not explain why two unlike patterns such as a
cup and its plate form a functional group.

## Rejected transform: one-step predictability

Whitening the covariance between a scattering state and one lazy transport
step produces singular values from roughly 0.82 to 1.00. The step preserves
nearly every smooth feature, so predictability is too easy: it weakens the
Pikachu relation and leaves coffee unchanged. A transfer operator must remove
the independent/smooth baseline rather than reward survival.

## Centered edge-relation operator

Let `phi_i` be a fragment's scattering coordinate. Define its area-weighted
covariance `G`, and define `J` as the symmetric joint moment of feature pairs
on literal graph edges. Centering `phi` makes `J` the observed edge-joint
measure minus the independent product contribution. The complete whitened
operator is

`B = G^(-1/2) J G^(-1/2)`.

No modes are selected. The signed embedding retains every eigenpair of `B`;
`|B|` supplies its norm and `sign(B)` supplies positive versus negative
association. Thus a positive relation means two support patterns co-occur
more than their prevalence predicts, while a negative relation means they
co-occur less.

On full 600×400 coffee, across scattering depths one through five:

- cup–plate remains positive: +0.251 to +0.396;
- cup–table remains negative: −0.584 to −0.795;
- plate–table remains negative: −0.861 to −0.930.

The signs survive whitening floors from `1e-4` through `1e-8`, ±15-pixel
evaluation-mask shifts, and ±10% mask dilation. On Pikachu the same operator
gives a positive disconnected tail–body relation and a negative tail–black
relation at both 256 pixels and full resolution. It does not rescue the weak
black ear tip.

This is the first probe here that handles both repeated disconnected geometry
and association between unlike but coupled patterns with one analytical
object. It remains a forensic relation field, not an object partition.

## Higher operator order and the direction limit

Powers of the centered operator encode longer relational chains without
walking candidate paths. They are not the missing semantic layer. On coffee,
orders one through six strengthen cup–plate from +0.261 to +0.856 and drive
both table relations toward −1. On astronaut they simultaneously strengthen
the wrong face–shuttle relation from +0.721 to +0.890 and make face–suit more
negative. Longer association reinforces relational recurrence; it cannot
invent physical-layer identity.

The causal allocation front also cannot supply that identity. Once the frozen
precision tensor `Q` is fixed, its eikonal distance is reversible. The arrows
in its achieving forest point away from arbitrary allocation germs.
Directional collision slack only records where those reversible fronts meet.
Treating either as scene depth would convert allocation gauge into a false
observable.

A contrast-inversion control on astronaut supports the separation. Metric
and action fields remain highly correlated under inversion (`0.95–0.97`),
while signed texture/cartoon fields reverse (`−0.96` approximately). The
transport metric registers geometry but has no intrinsic foreground arrow.
An actual depth layer requires asymmetric evidence such as embedded occlusion
topology, defocus, perspective, or temporal motion. This is an information
boundary, not a missing scalar transform of `Q`.

## Contrast-normalized focus evidence

Defocus is the first genuinely asymmetric image observable added after that
negative. `experiments/transport_focus_forensics.py` performs a calibrated
reblur experiment in linear light. For a Gaussian-blurred ideal step, a
derivative aperture `s` and known added blur `delta` give

`r = g_reblur / g = a / sqrt(a^2 + delta^2)`,

where `a^2 = sigma^2 + s^2`. Hence

`a = delta r / sqrt(1-r^2)`.

The unknown edge contrast cancels. On synthetic optically blurred steps the
implementation recovers input radii 1, 2, and 4 pixels to within 0.11 pixels,
and changing linear edge contrast from 0.15 to 1.0 changes the estimate only
at floating-point precision. Flat fields have exactly zero observation
confidence.

This does **not** make every dense value a defocus measurement. On arbitrary
texture, intrinsic spatial scale and optical blur are not identifiable from
one image without a prior. The implementation therefore:

- measures in linear luminance;
- trusts coherent gradient ridges rather than raw high-frequency energy;
- keeps evidence confidence separate from blur scale;
- treats flat transport fragments as unknown rather than sharply focused;
- pools evidence only inside an existing connected support fragment.

The distinction is visible in the real controls. Weighted Wasserstein
distances between log effective-scale distributions were:

| scene relation | focus distance |
| --- | ---: |
| astronaut face–shuttle | 0.045 |
| astronaut face–suit | 0.155 |
| astronaut face–background | 0.149 |
| coffee cup–plate | 0.137 |
| coffee cup–table | 0.268 |
| coffee plate–table | 0.131 |

Thus focus adds real information, but it falsifies the claim that focus alone
repairs the current association problem. The astronaut face and shuttle are
both sharp. The coffee plate and table are effectively coplanar and similarly
focused. Human grouping in these controls is using more than focal depth.

The more consequential focus observable lives on an interface. Psychophysical
work shows that blur of a common occlusion boundary can resolve the near/far
ambiguity left by unsigned depth-from-focus. The experiment therefore also
measures three quantities independently on every embedded interface arc:

1. the blur scale inside each incident support;
2. the blur scale on their common boundary;
3. which side's interior scale the boundary matches, with a separate
   reliability field.

No ownership threshold or object rule is applied. In two synthetic controls,
a sharp foreground over a blurred background and a blurred foreground over a
sharp background, the common boundary matches the foreground side in both
cases. This gives a physically motivated border-ownership arrow that is not a
gauge of the transport allocation forest.

The viewer exposes raw edge scale, confidence, transport-fragment focus,
anchor focus likeness, boundary-to-side match, and boundary ownership
reliability.

### Promotion into soft support

The first attempted promotion was almost inert. It estimated each side from
pixels in the strict interior of its incident transport fragment. At 256-pixel
Pikachu, 93% of interfaces then had no two-sided focus observation: the cells
were deliberately much smaller than the optical sampling aperture.

The corrected observable is attached to the embedded interface, not to an
individual tessellation cell. Every edgel measures confidence-weighted focus
in two physical half-strips, 6–14 pixels from the common boundary. Those
strips may cross reconstruction-cell seams because focal state belongs to the
imaged surface, not the cell partition. Separately connected arcs aggregate
the samples without sorting or candidate search.

Two earned signals are now active:

1. a reliable difference between the two side distributions, multiplied by
   frozen BFFT boundary confidence, adds interface resistance before the
   maximum-support forest is formed;
2. the boundary-scale match supplies a signed focus border-ownership vote,
   accumulated with independently observed T-junction ownership.

The boundary factor is essential. Without it, different intrinsic texture
frequencies on arbitrary tessellation seams masquerade as optical-depth
changes. At strength `0.75`, that unanchored control over-segmented astronaut
from 130 to 162 parts and coffee from 42 to 71. Once anchored to an observed
BFFT boundary, the same strength instead gives 125 and 40 parts respectively.
This is the distinction between using focus and manufacturing walls from
texture.

At 256-pixel Pikachu the half-strip correction raises nonzero two-sided focus
support from 7% to 47% of embedded interfaces. With the boundary-anchored
default strength `0.50`, the same 13 highpoints survive and 0.36% of soft cell
assignments change. Astronaut coarsens from 130 to 126 parts and coffee from
42 to 41. This is a real change to the support forest rather than a new colour
applied after grouping. The flat graphic changes only minimally, as it
should: it contains essentially no optical depth-of-field.

Missing focus remains the additive identity: zero reliability contributes
zero resistance and zero direction. `Unanchored focus-scale difference`
remains visible as the falsified control.

Relevant references:

- George Mather, *The Use of Image Blur as a Depth Cue*:
  https://doi.org/10.1068/p261147
- Marshall et al., *Occlusion edge blur: a cue to relative visual depth*:
  https://doi.org/10.1364/JOSAA.13.000681
- Tony Lindeberg, *Dense scale selection over space, time and space-time*:
  https://arxiv.org/abs/1709.08603
- Xu, Quan, and Ji, *Estimating Defocus Blur via Rank of Local Patches*:
  https://openaccess.thecvf.com/content_iccv_2017/html/Xu_Estimating_Defocus_Blur_ICCV_2017_paper.html

The subsequent autofocus metric selection, texture scale-space model,
chromatic control, and rejected highpoint-birth experiment are recorded in
`notes/AUTOFOCUS_METRIC_FORENSICS.md`.

## Current interpretation

The BFFT factorization contains two conflicting statements:

- the cartoon/appearance branch can say “this black tip is background”;
- the action/metric branch can retain evidence that it participates in the
  same transported geometry as the yellow object.

This is evidence for a multi-measure representation, not justification for a
hand-written voting rule. The next experiment should preserve spatial
relations inside the metric field instead of collapsing every region to an
unordered distribution.
