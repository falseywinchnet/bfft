# Research checkpoint: from task-aligned filters to acquisition priors

## The motivating mistake

Fourier circles can fit and extrapolate periodic structure extremely well, but
that success is epistemically ambiguous when periodicity is already enforced by
the representation. The same objection applies to a matrix exponential on a
rotational task. A useful general learning layer should not win because its
surface geometry happens to be the answer.

The target is therefore not a universal collection of hand-picked structures.
It is an **acquisition prior**: a low-parameter mechanism that learns how
observations relate, changes its internal interpretation continuously, and lets
the evidence choose a favorable chart.

## Current mechanism

The soft Eikonal layer begins with an ordinary affine map. In parallel, it
projects the authentic activation through several fixed, unprivileged views.
An input-conditioned positive-semidefinite metric scores those views, and a
soft allocation produces a continuous mixture rather than a discrete chart
choice. The allocated projection is returned to the authentic activation as a
small self-context guess. The chart is recomputed from that augmented input,
then used to produce the correction to the affine map.

This is the acquisition-auxiliary direction: the model does not receive an
external derivative or structural label. It creates a private contextual guess
from its own current interpretation of the sample. Backpropagation can refine
that interpretation across many observations.

The construction is still finite and imperfect. Its atlas has a discrete
number of primitive views, although the allocation among them is continuous.
The experiment tests whether that continuous conditionality is useful without
claiming that it is a true structure-complete Banach sieve.

## Epistemic constraints

The benchmark is designed to prevent the model from knowing before it knows:

- every comparison within a task has an identical trainable parameter count;
- the control is a normal dense LELU MLP, not a single affine map;
- no task-specific basis or periodic encoding is supplied;
- training sees only the declared inner support;
- outer regions and explicit tail bins measure continuation separately;
- low-rank and high-rank N-D constructions distinguish missing evidence from
  redundant relational evidence;
- two paired confirmation seeds reduce optimizer-luck comparisons.

## Five hypotheses tested on top of self-context

### 1. Harder allocation

Lowering the allocation temperature asks whether acquisition improves when the
continuous chart commits more quickly. It adds no parameters or extra forward
passes.

### 2. Iterated private context

A second anchored refinement asks whether repeatedly reinterpreting the same
authentic activation improves structural inference. Every iteration remains
anchored to the original activation so context cannot accumulate unchecked
drift.

### 3. Uncertainty-gated acquisition

Allocation entropy scales the injected context. This tests the tempting idea
that the model should ask itself for more structural help when its current
chart is uncertain.

### 4. Output secants

Training also compares differences between randomly paired outputs and target
differences. This asks whether explicit relational supervision helps the model
learn the derivative sense in which one observation should transform into
another.

### 5. Allocation-chart curvature

The loss penalizes a second finite difference of allocation weights under a
small random displacement. This is the closest experiment to Eikonal transport:
it rewards a chart whose local interpretation continues smoothly through the
observation manifold.

## What survived

Self-context itself survives as the baseline. Hard allocation is the best
general acquisition-speed modification: it improves learning-curve area on 21
of 22 problems at the same parameter count and essentially the same runtime.

Chart curvature survives as a conditional research lead. It improves mean
held-out and tail behavior and is exceptional when multiple input planes carry
coherent phase evidence. It also damages radial stripes and Fourier mixtures,
showing that a globally flat chart is the wrong prior when the correct local
frame must turn sharply.

The low-rank/high-rank spiral contrast is the central warning. A high-rank
observation can already expose redundant phase derivatives; regularized chart
transport can exploit them. A low-rank observation that withholds those
relations remains hard. The experiment therefore demonstrates transport of
available relational evidence, not arbitrary structural hallucination.

## Next principled questions

1. Replace random secants with neighborhoods selected by the learned chart.
2. Make curvature strength conditional on estimated local turning, so straight
   flows are preserved without flattening circular or Fourier structure.
3. Measure whether acquisition decisions agree across independently sampled
   local witnesses rather than merely across nearby input coordinates.
4. Test whether learned primitive views can remain unprivileged while reducing
   the finite-atlas limitation.
5. Continue treating ordinary spiral, checker continuation, and low-rank N-D
   spiral as hard falsification cases rather than hiding them in aggregate MSE.
