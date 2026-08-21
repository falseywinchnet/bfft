# Sparse sine: observation measure versus intrinsic geometry

## The probe

`sparse_sine_1d` is deliberately the smallest version of the red-tail failure
in the complex spiral. The target is a phase-offset sine with ten observed
periods on `[0, 1]`, followed by five completely unseen periods on `(1, 1.5]`.
The per-period training counts are:

```text
768, 768, 512, 384, 256, 128, 64, 32, 16, 8
```

The last observed period therefore has 96 times fewer observations than the
first. Points are stratified within each period so this measures acquisition
under thinning, rather than an accidental uncovered gap. Evaluation over the
observed interval is uniformly spaced in coordinate space; the dense head
cannot hide a missed tail.

## Why ordinary training misses the tail

Empirical MSE approximates an integral against the observation measure,

```text
L_empirical = sum_i p_i |f(x_i) - y_i|^2,
```

not against coordinate measure. The final period contributes roughly `1/96`
of the gradient mass of the first period. A pointwise layer also receives no
indication that one sparse point owns a much larger interval than a dense
point. It can lower loss while declining to learn the sparse geometry.

Reweighting or segment balancing alone did not fix this. They increase the
tail's importance, but do not supply the missing relation between neighboring
observations. This is why the balanced and Voronoi variants were unstable or
collapsed despite having a fairer loss.

## The decisive acquisition result

The Hermite experiment below first proved that the sparse observations contain
enough information. A later control found a simpler and more important result:
the pseudo-targets are not necessary.

Assign every observation its one-dimensional Voronoi support, normalize those
cell lengths into a probability distribution, and draw one jittered sample
from every equal-mass quantile of that distribution in each batch. Coordinates
are whitened using the same support measure rather than the empirical measure.
The targets remain the original observed labels—there is no interpolation,
frequency feature, derivative target, or sinusoidal assumption.

This is the stable form of the measure correction. Empirical sampling followed
by inverse-density weights was noisy because a rare sparse-tail sample carried
a very large gradient. Ordinary importance sampling improved one seed but
remained stochastic. Stratification makes every optimizer step see the full
coordinate support and removes those rare gradient spikes.

At 1,000 steps, the operator sphere produced the following three-seed result:

| Seed | Observed R² | Last-three R² | Minimum period R² |
|---:|---:|---:|---:|
| 0 | 0.9989 | 0.9982 | 0.9965 |
| 1 | 0.9968 | 0.9945 | 0.9917 |
| 2 | 0.9962 | 0.9900 | 0.9855 |

CFF also solved all three observed regions (`0.944`, `0.996`, and `0.988`
minimum-period R²), but required about 40 seconds per run versus 25 seconds for
the operator sphere. Plain self-context was fastest at about 9 seconds and
solved two seeds strongly, but one seed retained only `0.646` worst-period R².

## In-context cheating: witnessed branch descent

The next experiment turns a held-out subset into a repeated outer acquisition
signal. It is one model trajectory, not an inference ensemble. At every
accepted update, three copies of the current model and AdamW state take one
step on different support-stratified pools. A common witness fold, excluded
from all three gradients, selects the candidate state that survives. The other
two states are discarded.

The witness atlas divides coordinate support into sixteen equal-support cells
and locally interleaves four folds inside each cell. Consequently every fold
spans the dense head and sparse tail. Candidate pools are isotropic in global
support mass but anisotropic in their local sample phase. The witness remains
fixed for 50 accepted steps before rotating; one-step rotation chases the
current fold, while a permanently fixed fold becomes a privileged pattern.

With 1,000 accepted updates and 3,000 total candidate-gradient evaluations,
resident witnessed descent gave:

| Seed | Observed R² | Last-three R² | Minimum period R² |
|---:|---:|---:|---:|
| 0 | 0.9988 | 0.9981 | 0.9966 |
| 1 | 0.9985 | 0.9969 | 0.9942 |
| 2 | 0.9946 | 0.9874 | 0.9744 |

The mean minimum-period R² is `0.988`, versus `0.859` for ordinary
self-context after the same 1,000 accepted steps. This removes the catastrophic
seed: the worst run improves from `0.646` to `0.974`. However an ordinary
3,000-step self-context run reaches mean minimum-period R² `0.993` in about
25.5 seconds, while the three-branch method reaches `0.988` in about 51.8
seconds. It is therefore a basin-robustness result, not yet a compute Pareto
win.

The negative result is equally sharp. Unseen continuation did not improve and
was sometimes much worse. Repeatedly selecting the signal that explains
structured unseen *samples inside observed support* is sufficient to suppress
local density basins, but it does not identify the recurrence that transports
the curve beyond support. Held-out explanation and extrapolating structure are
not the same criterion.

The zonotopic-mixture construction supplies the right organizational analogy:
retain a bounded bank of candidate histories, use observations to falsify
inconsistent histories, and reduce only after accounting for the information
lost by merging. Zhu et al.'s dimensionality-compression objective suggests
the next efficiency move: represent the candidate update bank by its
low-dimensional second-order response subspace, then explore that continuous
subspace instead of cloning complete models.

The current conclusion is therefore sharper: the missing sparse tail was
primarily a mismatch between empirical sampling measure and geometric support,
with spectral conditioning as a secondary optimizer problem. The operator
sphere is the most reliable carrier tested, but it does not need constructed
Hermite labels once acquisition is support-stratified.

## Earlier geometry-identification experiment

Treat adjacent observations as the endpoints of a local transport cell. From
the nonuniform samples, estimate one-sided/centered tangents `m_i`. Inside an
interval of width `h`, query the cubic Hermite transport

```text
H_i(a) = h00(a)y_i + h10(a)h m_i + h01(a)y_(i+1) + h11(a)h m_(i+1),
```

with intervals sampled in proportion to their width and `a` uniform on
`[0, 1]`. This changes the acquisition measure from “how many points happened
to be recorded here” to “how much coordinate support does this pair witness?”
It supplies local value and tangent coherence, but no sinusoidal basis,
frequency, future point, or periodic label.

The Hermite reconstruction itself reaches greater than `0.999` R² from the
observed pairs for all tested seeds, including each sparse period. The task is
therefore identifiable from the observations; the remaining question is
whether a model and optimizer can absorb that geometry.

## Earlier Hermite result

Representative CPU runs at roughly 26k parameters:

| Model / acquisition | Steps | Observed R² | Last-three R² | Minimum period R² |
|---|---:|---:|---:|---:|
| Self-context / empirical | 1,000 | 0.731 | 0.155 | -0.186 |
| Self-context / Hermite cells | 1,000 | 0.847 | 0.653 | 0.216 |
| CFF / Hermite cells | 1,000 | 0.909 | 0.912 | 0.761 |
| Operator sphere / empirical | 1,000 | 0.585 | poor | -0.074 |
| Operator sphere / linear cells | 1,000 | 0.885 | 0.794 | 0.712 |
| **Operator sphere / Hermite cells** | **975** | **0.967** | **0.962** | **0.942** |
| Ordinary MLP / Hermite cells | 2,000 | 0.188 | poor | -0.156 |

The winning run's ten period R² values are:

```text
0.989, 0.979, 0.972, 0.979, 0.969, 0.948, 0.950, 0.942, 0.952, 0.991
```

Thus the model recovered the entire observed interval, including the period
represented by only eight points. This established identifiability and showed
that an ordinary MLP could not readily absorb the local curvature construction.
The later support-stratified result supersedes the stronger claim that Hermite
pseudo-data itself is necessary.

The outcome is robust but optimization-sensitive. Two of three operator-sphere
seeds crossed the threshold by 1,000 steps; the slow seed reached observed R²
`0.963` and minimum-period R² `0.931` by 2,000 steps.

## What remains unsolved

The five unseen periods are a separate problem. Across the successful
support-stratified seeds, unseen extrapolation remains negative and unstable:
outside support the models race away or flatten, just as the red spiral tail
does. Support-aware acquisition repairs geometry *inside* the convex hull; it
does not infer a recurrence that continues beyond the final cell.

This cleanly separates two failures:

1. sparse observed geometry was being suppressed by the empirical measure;
2. continuation beyond evidence requires a learned connection or recurrent
   transition, not merely better density correction.

For the next self-context formula, the first missing object is now clearly the
observation's uncertainty cell (support radius/measure). A transported local
jet can remain optional rather than mandatory: it improves reconstruction, but
the original labels suffice when batches cover support coherently. A separate
sequential mechanism is still needed to learn how the acquired relations
compose beyond observed support.

The interactive result viewer is `ML_experiment/sparse_sine_geometry.html`.
