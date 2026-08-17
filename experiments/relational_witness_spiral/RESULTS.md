# Result ledger

> **Decision-plot correction.** The shared raster renderer originally placed
> `y_min` at the top while scatter points used Cartesian `y_max` at the top.
> This vertically mirrored decision-region backgrounds relative to their
> points. Numerical predictions, accuracies, tail bins, and survival metrics
> were computed directly from models and are unaffected. The renderer now
> flips the raster explicitly and has an orientation regression test. Current
> Banach-eikonal decision plots were regenerated; older stored decision PNGs
> from the preceding experiments retain the visual-only mirror until rerun.

Both experiments used five seeds, 1,800 training steps, 20 consecutive held-out
bins, and CPU execution on the M4 Mini. `train_fraction=0.50` is a 50% holdout;
`train_fraction=0.30` is a 70% holdout. The primary result is contiguous
survival from the training frontier at 80% accuracy.

## Fixed-view witness result

At the 50% frontier, the dense reference, marginal witness, and relational
witness obtained respectively 3.02%, 8.51%, and 7.28% mean accuracy over the
first five held-out bins. All had zero contiguous survival. At the 70%
holdout, their frontier-five accuracies were 25.95%, 26.98%, and 26.21%, again
with zero survival. Pair/triple witness relations did not beat the marginal
control.

This rejects a geometry-free interpretation of the fixed-view prototype. Its
orthogonal view scaffold is still privileged structure, even if it is less
obviously task-shaped than Fourier circles.

## Learned-subspace result

This follow-up strictly contains an ordinary dense map. All additional
subspaces are learned. The static control is still exactly linear; the dynamic
models generate branch weights from either individual branch energies or the
mutual Gram matrix.

| train fraction | model | parameters | seen validation | frontier-five | whole tail | survival bins |
|---:|---|---:|---:|---:|---:|---:|
| 0.50 | dense reference | 2,498 | 99.97% | 2.92% | 49.79% | 0 |
| 0.50 | static factorized | 4,232 | 99.97% | 5.02% | 49.34% | 0 |
| 0.50 | marginal dynamic | 4,456 | 99.94% | 11.05% +/- 17.66% | 49.28% | 0 |
| 0.50 | Gram dynamic | 4,552 | 99.94% | 5.02% | 49.64% | 0 |
| 0.30 | dense reference | 2,498 | 99.97% | 26.87% | 45.71% | 0 |
| 0.30 | static factorized | 4,232 | 99.97% | 25.86% | 45.26% | 0 |
| 0.30 | marginal dynamic | 4,456 | 99.97% | 25.36% | 45.24% | 0 |
| 0.30 | Gram dynamic | 4,552 | 100.00% | 26.73% | 45.74% | 0 |

The large marginal mean at 50% comes from one degenerate seed that predicts
near 50/50 over almost the entire tail; its variance and zero survival make it
non-evidence. Gram relations add no measurable continuation over the static
factorization. Whole-tail scores near 50% are periodic recrossing artifacts.

## Interpretation

Fourier circles can be the best spiral score while still being the less general
operator family. Radial pooling supplies a continuation rule aligned with this
task, but excludes many dense linear maps. The learned-subspace model avoids
that defect by retaining a full dense base, yet its conditional Gram mechanism
does not infer the missing continuation law from pointwise classification.

The limiting issue appears to be supervision, not matrix parameterization:
training labels on an inner interval do not identify which of many equally
accurate decision boundaries should continue outward. Another pointwise
operator can fit the observed interval but has no evidence that selects the
correct extension. A meaningful next experiment needs relational observations
across an actual domain or several measurements of a shared latent object, not
several algebraic rewrites of one hidden vector.

## Hypersphere-atlas result and correction

The learned-atlas follow-up also fits every observed inner spiral essentially
perfectly and has zero contiguous survival at both holdout frontiers. At the
50% frontier, the dense, identity-atlas, isotropic-atlas, Eikonal-atlas, and
shuffled-atlas models obtain respectively 2.92%, 3.93%, 3.00%, 3.02%, and
3.21% mean accuracy over the first five held-out bins. The Eikonal and shuffled
controls are indistinguishable for the relevant continuation metric.

This run does **not** reject learned transport. The atlas correction in this
prototype is bounded by construction: chart directions are normalized,
coefficients use `tanh`, and the global atlas scale uses `tanh`. With `C`
charts, its correction norm is bounded on the order of `sqrt(C)` independently
of input radius. The full dense residual preserves the ability to represent an
unbounded linear map, but the filter itself cannot learn an unbounded
continuation law.

It also trains the filter directly through pointwise classification. That
objective naturally rewards selecting useful elements of the observed state;
it does not separately reward preservation of the relation that generated
those elements. The experiment therefore identifies two distinct missing
pieces:

1. an unbounded nonlinear carrier built from learned linear maps, so the
   learned path can own range and continuation rather than merely gate a dense
   residual; and
2. an indirect objective that trains the filter through relational prediction
   or transport consistency, while withholding task-specific coordinates,
   phase, Fourier frequencies, and held-out samples.

The plots in `results_hypersphere/` show this failure directly. All decision
regions track the observed inner coils and then collapse immediately outside
the training frontier; later whole-tail accuracy is periodic recrossing, not
survival.

## Integrated jet-transport result

The next follow-up removes the bounded carrier and makes the learned derivative
field operational. Each candidate layer contains a dense map plus an
unsaturated input-conditioned low-rank operator. In the transport variants,
the representation and operator coefficients are produced by three-point
quadrature of learned connection fields rather than by querying a pointwise
coefficient network. Observed nearest-neighbor secants train the input-to-
representation and representation-to-operator derivatives; observed triples
train their composition. No radius, phase, ordering, Fourier feature, or
held-out sample is used.

Five seeds at 1,800 steps produced:

| train fraction | model | parameters | seen validation | first unseen bin | frontier-five | whole tail | survival bins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.50 | dense | 2,498 | 99.97% | 14.76% | 2.95% | 49.75% | 0 |
| 0.50 | direct unbounded jet | 12,712 | 99.97% | 14.92% | 2.98% | 49.38% | 0 |
| 0.50 | secant-regularized jet | 12,712 | 99.97% | 17.20% | 3.44% | 49.33% | 0 |
| 0.50 | integrated jet transport | 12,712 | 99.81% | 35.76% | 7.49% | 48.66% | 0 |
| 0.50 | shuffled transport | 12,712 | 99.97% | 42.36% | 8.69% | 46.51% | 0 |
| 0.30 | dense | 2,498 | 99.97% | 15.04% | 27.02% | 45.81% | 0 |
| 0.30 | direct unbounded jet | 12,712 | 99.97% | 30.44% | 25.00% | 45.30% | 0 |
| 0.30 | secant-regularized jet | 12,712 | 99.97% | 21.04% | 24.66% | 45.05% | 0 |
| 0.30 | integrated jet transport | 12,712 | 99.97% | 46.40% | 23.03% | 44.69% | 0 |
| 0.30 | shuffled transport | 12,712 | 99.97% | 35.96% | 31.81% | 46.80% | 0 |

The integrated connection changes immediate-frontier behavior substantially:
its first-bin mean is more than twice the dense mean at both splits. This is not
evidence of learned continuation. Every candidate fails the first 80%
survival test, bins two through five collapse at the 50% split, and shuffled
transport is at least as competitive on the reported aggregate metrics. One
shuffled 70%-holdout seed also creates a large accidental frontier-five score,
showing why periodic recrossings and averages cannot substitute for contiguous
survival.

This rejects the present **selection rule**, not the derivative carrier. Local
Euclidean nearest-neighbor secants permit many connection fields that explain
the observed jets. Integrating one of those fields makes its hallucinated
continuation persistent and unbounded, but nothing in the current objective
maintains competing hypotheses or updates their probabilities using genuinely
new relational evidence. A faithful next version needs multiple connection
hypotheses and an observed-only predictive likelihood on relations withheld
from fitting; the posterior mixture, rather than one arbitrarily optimized
connection, should determine transport.

## Probabilistic connection-hypothesis result

The probabilistic follow-up trains five independently initialized integrated
connections per run. Each receives a different bootstrap of 80% of the
observed relation triples. The remaining 20% are excluded from relation-loss
gradients and serve as observed-only predictive evidence for checkpointing and
posterior weighting. Three seeds were run for each frontier. A uniform mixture
and a posterior with permuted hypothesis weights are matched controls. The
best-hypothesis oracle uses test data and is diagnostic only.

| train fraction | selection | first unseen bin | frontier-five | whole tail | survival bins |
|---:|---|---:|---:|---:|---:|
| 0.50 | uniform mixture | 32.87% | 6.57% | 48.40% | 0 |
| 0.50 | soft evidence posterior | 35.00% | 7.00% | 47.70% | 0 |
| 0.50 | permuted posterior | 32.47% | 6.49% | 48.88% | 0 |
| 0.50 | evidence MAP | 46.07% | 12.95% | 48.43% | 0 |
| 0.50 | test-only oracle | 52.40% | 14.21% | 47.90% | 0 |
| 0.30 | uniform mixture | 35.67% | 21.88% | 44.42% | 0 |
| 0.30 | soft evidence posterior | 35.00% | 21.84% | 44.42% | 0 |
| 0.30 | permuted posterior | 36.00% | 21.91% | 44.44% | 0 |
| 0.30 | evidence MAP | 32.53% | 23.45% | 44.65% | 0 |
| 0.30 | test-only oracle | 51.47% | 24.08% | 44.91% | 0 |

The wider observed domain produces the first evidence that behaves like the
proposed reinforcement process. At the 50% frontier, held-relation loss and
first-bin accuracy have mean within-seed correlation -0.881: lower predictive
loss generally ranks the better local continuation. Evidence-MAP consequently
raises first-bin accuracy from 32.87% to 46.07%, approaching the 52.40% oracle.
The soft posterior is too diffuse and averages away most of that difference.

This relationship is absent with only 30% of the domain observed. Its mean
within-seed correlation is +0.224, evidence-MAP does not match the oracle in
any seed, and posterior, uniform, and permuted mixtures are indistinguishable.
The relevant additional information is structural extent, not sample count:
both splits contain the same number of observations, but the 50% split exposes
more of the changing derivative law.

The result remains negative for actual extrapolation. Even the test-only oracle
has zero contiguous survival, so posterior selection is not the only missing
piece. The present bootstrap hypotheses mostly differ in how a broad exterior
partition is placed; none maintains the learned turning law through successive
unseen bins. The next hypothesis basis must branch over derivative evolution
itself (connection curvature or higher jet shells), rather than merely over
initialization and samples of the same first-order connection family.

## Associative Newton-shell result

The simplest follow-up removes explicit pairs, triples, graphs, scene encoders,
and hypothesis posteriors. Each context observation contributes one rank-one
write to a shared key/value matrix, `M = mean_i(value_i key_i^T)`. Queries
propagate through one to three nonlinear residual applications of `M`. Products
such as `M^2` and `M^3` contain implicit cross-observation terms without
enumerating them. Training context and query subsets are disjoint. The shuffled
control preserves key and value marginals but permutes complete values against
keys, destroying coherent writes.

| train fraction | model | parameters | seen validation | first unseen bin | frontier-five | whole tail | survival bins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.50 | context-free query | 2,212 | 80.22% | 2.04% | 0.82% | 49.93% | 0 |
| 0.50 | one shell | 2,213 | 98.97% | 44.12% | 11.50% | 47.01% | 0 |
| 0.50 | two shells | 2,214 | 99.19% | 43.52% | 9.78% | 47.57% | 0 |
| 0.50 | three shells | 2,215 | 99.34% | 47.80% | 11.86% | 45.72% | 0 |
| 0.50 | three shells, shuffled writes | 2,215 | 94.16% | 39.32% | 26.28% | 48.23% | 0 |
| 0.30 | context-free query | 2,212 | 99.47% | 10.16% | 26.77% | 45.69% | 0 |
| 0.30 | one shell | 2,213 | 99.91% | 38.72% | 23.34% | 44.78% | 0 |
| 0.30 | two shells | 2,214 | 99.97% | 31.76% | 23.81% | 44.93% | 0 |
| 0.30 | three shells | 2,215 | 99.97% | 46.72% | 26.86% | 45.60% | 0 |
| 0.30 | three shells, shuffled writes | 2,215 | 99.88% | 48.60% | 25.27% | 45.24% | 0 |

The shared medium is unquestionably used: removing it from trained memory
models reduces seen accuracy toward chance, and coherent writes materially
improve the 50%-domain fit over the matched shuffled control. Relative to the
earlier dense model, three shells move the immediate frontier from 14.76% to
47.80% with fewer parameters. That improvement does not survive subsequent
bins. The large shuffled frontier-five number at 50% is caused by predictions
collapsing near 50%, not by continuation.

Superposition solves the pinhole problem, but powers of one global memory are
still powers of one fixed operator. They create higher-order interference among
observations; they do not describe how the medium sampled by a query changes as
the query moves. Shell depth is not monotonically helpful, and no shell count
learns the turning law. Any next minimal mechanism should remain a shared
medium but make it a field sampled by position, rather than adding explicit
relational machinery back in.

## Projective-labeling quotient result

This follow-up processes only an inner context. A generic learned feature map
and the complete context covariance produce eight rank-two subspaces. A point's
normalized projection energies are its soft structural labels; they are not
Voronoi cells or coordinate IDs. Each label pools an affine least-squares law
from its assigned context observations. Query labels are absent, context and
query episodes are disjoint, and there is no learned coordinate-to-class head.

Five seeds at 1,500 steps produced:

| train fraction | model | parameters | seen validation | first unseen bin | frontier-five | whole tail | survival bins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.50 | global pooled affine law | 326 | 93.72% | 1.80% | 0.36% | 49.24% | 0.0 |
| 0.50 | positional cells | 530 | 99.94% | 26.24% | 5.25% | 48.55% | 0.0 |
| 0.50 | diagonal scene operator | 513 | 99.97% | 39.36% | 7.87% | 47.34% | 0.0 |
| 0.50 | shuffled role/law relation | 513 | 91.81% | 2.32% | 0.46% | 49.42% | 0.0 |
| 0.50 | full projective quotient | 513 | 99.94% | 21.64% | 4.33% | 48.95% | 0.0 |
| 0.30 | global pooled affine law | 326 | 99.63% | 99.48% | 38.12% | 46.53% | 1.4 |
| 0.30 | positional cells | 530 | 99.94% | 56.08% | 39.74% | 47.80% | 0.0 |
| 0.30 | diagonal scene operator | 513 | 99.97% | 44.16% | 24.02% | 44.93% | 0.0 |
| 0.30 | shuffled role/law relation | 513 | 99.38% | 73.32% | 23.06% | 44.46% | 0.4 |
| 0.30 | full projective quotient | 513 | 99.97% | 36.04% | 24.36% | 44.87% | 0.0 |

The roles are real rather than decorative: at the 50% frontier, shuffling
which context observations support which role reduces seen validation from
99.94% to 91.81%. The learned-label plot also shows coherent arcs and repeated
structural pieces rather than a random coloring. But this solves labeling, not
continuation. In the decision plot each role's affine law becomes a broad
exterior wedge. Every 50%-observed run fails the second unseen bin, and the
full covariance transport is worse than the diagonal operator on the immediate
frontier.

The sharpest falsification is at the 30% split: the single pooled law survives
1.4 bins on average while every coherent projective quotient survives zero.
Adding labels has fragmented a continuation that the simpler learned feature
space briefly possessed. The mechanism learned *which inner elements belong
together*, but no affine relation among the labels says how one structural role
transitions into the next as the scene extends.

This isolates the missing object more narrowly than the prior failures. The
affine law cannot merely live inside each projective label. Affine structure
must govern transitions **between** projective labels, with the quotient basis
identified only up to relabeling. A next version should therefore learn a
small role-transition operator from inner-scene adjacency and require its
iterates to agree with withheld inner regions. Otherwise a projective atlas is
still only an atlas of pieces, not a transport law.

## Projective role-transition result

This experiment makes the proposed correction operational. The projective
basis is fixed across episodes (and therefore identifiable up to relabeling),
while visible-context adjacency induces a directed transition matrix on its
roles. A learned structural clock orients edges. A query anchors to the visible
context and mixes zero through four transition iterates before the pooled
affine laws act. Training uses randomly oriented half-space occlusions and
occasional random context/query partitions entirely inside the observed scene;
no radius, phase, spiral order, or outer point is supplied.

Three seeds at 1,200 steps produced:

| train fraction | model | parameters | seen validation | first unseen bin | frontier-five | whole tail | survival bins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.50 | direct projective placement | 632 | 99.95% | 17.67% | 3.53% | 48.49% | 0 |
| 0.50 | context anchor, identity transition | 632 | 99.79% | 25.00% | 5.00% | 48.24% | 0 |
| 0.50 | shuffled transition edges | 632 | 99.84% | 32.87% | 6.57% | 48.65% | 0 |
| 0.50 | learned role transition | 632 | 99.74% | 22.33% | 4.47% | 48.28% | 0 |
| 0.30 | direct projective placement | 632 | 100.00% | 56.73% | 24.72% | 45.08% | 0 |
| 0.30 | context anchor, identity transition | 632 | 100.00% | 53.07% | 24.73% | 45.06% | 0 |
| 0.30 | shuffled transition edges | 632 | 100.00% | 51.60% | 24.52% | 45.16% | 0 |
| 0.30 | learned role transition | 632 | 100.00% | 55.60% | 25.44% | 45.31% | 0 |

The transition is neither unused nor uniform. At the 50% split its mean row
entropy is 0.833 versus `log(8)=2.079`, and its mean diagonal mass is 0.664.
Nonetheless it is worse than shuffled edges at the immediate frontier and all
methods fail the second bin. The 30% results are statistically clustered and
again have zero survival.

The large diagonal mass exposed a possible construction error: ordinary point
adjacency counts many edges inside a role, while a quotient graph should retain
only crossings between roles. A matched 600-step diagnostic removed the entire
diagonal and assigned dwell time to the selector's zero-step state. Learned
transport then obtained 14.0% first-bin accuracy, versus 29.2% for identity
anchoring and 24.0% for shuffled crossings, with zero survival. Thus self-loop
dominance is not the explanation.

The remaining obstruction is state aliasing. A projective role says what kind
of piece an observation is, but not how it entered that piece. The same role
can have different successors depending on incoming direction or local affine
frame. Collapsing the inner scene to a first-order role destroys precisely the
derivative information needed for continuation; powers of the resulting
transition cannot recover it. The minimal next state is not another larger
atlas. It is a lifted quotient state `(projective role, incoming relation)`,
with affine transport acting on that relational tangent and role names still
permutation-equivariant.

## Continuous Banach-eikonal sieve result

This experiment implements the proposed interpretation circle as a continuum,
not a categorical expert bank. Sixteen views are quadrature cells on one
periodic operator-valued curve represented by three Fourier harmonics. A
permutation-invariant summary of the labeled inner context creates a smooth,
tanh-bounded periodic potential. A conservative finite-volume eikonal flux
transports probability between neighboring cells on the circle. The resulting
density integrates the operator used inside each of three hidden residual
layers; a separate radial coordinate controls inducement magnitude. The
unbounded hidden carrier is not passed through `tanh`.

Continuity tests verify exact `2*pi` periodicity, local phase continuity,
uniform-quadrature refinement, context permutation invariance, mass
conservation/positivity, and gradients through both density transport and the
nonconstant operator modes. Training uses only inner-scene observations, with
equal parts random half-space occlusion episodes and disjoint random
context/query episodes.

Three seeds at 2,000 steps produced:

| train fraction | density mechanism | parameters | seen validation | first unseen bin | frontier-five | whole tail | survival bins | density concentration | context response TV |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | frozen uniform | 8,711 | 71.77% | 12.20% | 2.61% | 49.51% | 0 | 0.000 | 0.000 |
| 0.50 | direct potential reweighting | 8,711 | 90.26% | 10.53% | 2.44% | 49.65% | 0 | 0.583 | 0.025 |
| 0.50 | broken ring differential | 8,711 | 99.38% | 9.40% | 2.01% | 49.56% | 0 | 0.088 | 0.045 |
| 0.50 | coherent eikonal transport | 8,711 | 88.49% | 8.27% | 1.96% | 49.48% | 0 | 0.136 | 0.062 |
| 0.30 | frozen uniform | 8,711 | 94.27% | 8.53% | 27.55% | 45.88% | 0 | 0.000 | 0.000 |
| 0.30 | direct potential reweighting | 8,711 | 100.00% | 15.13% | 26.11% | 45.30% | 0 | 0.961 | 0.069 |
| 0.30 | broken ring differential | 8,711 | 100.00% | 12.67% | 26.15% | 45.36% | 0 | 0.108 | 0.131 |
| 0.30 | coherent eikonal transport | 8,711 | 100.00% | 16.67% | 25.24% | 45.34% | 0 | 0.245 | 0.099 |

The continuous mechanisms are operational. Direct reweighting becomes highly
concentrated, whereas conservative transport maintains several separated
regions of mass. The radial inducement reaches 0.994 at the 30% split, so the
model is not trapped at the cone origin. Coherent transport also fits all three
30%-observed inner scenes perfectly. None of these facts yields continuation:
every model has zero contiguous survival, and bin two is zero in every run.
Broken adjacency is indistinguishable from coherent differential transport on
the relevant metrics.

The context-response diagnostic exposes the dominant shortcut. It compares
the learned circular density under the left and right halves of the same
observed scene. Coherent transport changes by only 6.2% total variation at the
50% split and 9.9% at the 30% split. The density moves far from uniform but is
mostly a task-wide learned constant, not a filter inferred anew from the
observation. Direct reweighting is even less conditional despite its near-point
mass. The pointwise classification carrier therefore learns the inner spiral
while the context pathway supplies little indispensable information.

This is a negative result for the present objective, not for the continuous
sieve geometry. A continuous circle cannot create observation dependence when
one target function is shared by every episode: ignoring context remains a
valid optimum. The next experiment must remove that optimum through an
observed-only relational objective or multiple counterfactual scene laws. A
mere penalty demanding density movement would manufacture arbitrary motion;
the conditioner must be required to predict a withheld relation that the query
alone cannot determine. Only then can we test whether eikonal motion selects a
useful inductive region rather than becoming another fixed hidden layer.
