# Randomize the defect decision, keeping its candidate updates fixed

This is the corrected experiment requested by the user. It changes only
the accept/reject judgment in the existing defect-gated operator. It adds
no new memory-update stage, candidate solver, or noise sample to a field.
Earlier random scheduling experiments in `decision_noise.py` are separate
controls and are not the basis of the conclusions below.

## Decision law

The existing gate accepts an extra candidate when beta dot e >= 0, where
beta is the first compatibility correction and e the proposed additional
correction. Define the dimensionless confidence

    c = (beta dot e) / (||beta|| ||e||),

with c=0 when the denominator is zero. Replace the threshold zero with
U uniform on [-1,1], accepting iff c >= U. Consequently

    Pr(accept | state) = (1+c)/2.

Parallel nonzero defects are always accepted, antiparallel defects always
rejected, and perpendicular or undefined-direction cases are coin flips.
This is a specified exploratory randomization law, not a uniquely required
probability model or a calibrated uncertainty estimate. Its scale is the
natural cosine range; no trajectory multiplier or fitted temperature is used.

The shared version uses one threshold for the whole pass and both branches.
The local version uses independent thresholds per site and branch. Vector
components at a site share the same decision, preserving disk feasibility.

For candidate values a and b, the random selection has conditional mean
a+pi(b-a) and mean squared deviation pi(1-pi)||b-a||^2. The random variable
is only a selector. Its influence vanishes wherever the two candidates
coincide, including common fixed points. This does not guarantee identical
finite-time images or preservation of spatial symmetry: differing valid
decisions can generate spatial variation even without adding pixel noise.

In a strictly one-dimensional nonzero collinear decision, c is +/-1, so
this randomization cannot change it. The ramp's near-collinear decisions
are a relevant limiting case, rather than a reason to assume noise must
break its existing dynamics.

## Protocol and outcomes

The main validation uses five sources at (.05,40): a 1x128 edge and four
128-square sources. It runs to 1024 passes with random seeds 11,29,47,
checking the better feasible incoming/projected bounds every eight passes
for every method. The deterministic baseline runs once. Update timings
include random generation, normalization, allocations and comparisons;
common diagnostics are excluded. Method order is fixed. These are small
exploratory NumPy measurements, not a native timing promotion gate.

First sampled pass reaching gap/sample <= 1e-3:

| Source | Deterministic gate | Shared threshold, three seeds | Local threshold, three seeds |
|---|---:|---|---|
| Edge | 632 | 632, 632, 632 | 632, 632, 632 |
| Ramp | 160 | 160, 160, 160 | 160, 160, 160 |
| Crossing | 416 | 416, 416, 416 | 416, 416, 416 |
| Noise | 168 | 168, 168, 176 | 176, 176, 176 |
| Carrier | 272 | 272, 272, 272 | 272, 272, 272 |

The median time ratio relative to the deterministic gate, over the 15
case/seed comparisons, was 1.194 for shared thresholds and 1.396 for local
thresholds. These ratios reuse each case's single deterministic timing.
There was no time-to-target improvement in those comparisons. Some tighter
targets have small iteration gains; in the seed-11 noise census, 1e-4 is
reached at 400 passes instead of 408. That does not recover the overhead.

At 1024 passes the local-threshold ramp has gap roughly 1.8e-8 to 2.1e-8,
while the deterministic/shared variants are at rounding scale (~1e-14).
These finite-time results do not establish a persistent asymptotic floor.

## Did the gate actually change?

A separate untimed decision census uses 512 passes and seed 11. A decision
is counted as active when ||e|| > 1e-12, a diagnostic cutoff only. The
algorithm itself has no such cutoff. 'Ambiguous' counts |c| < 1-1e-8;
it includes weak departures from collinearity, not only decisions near zero.

For the low-parameter ramp (.02,20), only 0.0656% of active deterministic
decisions depart from collinearity at that tolerance. Shared threshold noise
flips 0.0348% of active decisions and 0.0915% of proposed correction energy.
It reaches 1e-4 at exactly the same sampled pass, 104.

Local thresholds break repeated-row symmetry. After 512 passes this ramp
has u row-variation RMS 4.23e-6 versus rounding scale for deterministic and
shared thresholds. Its gap is 6.97e-8 versus 5.95e-11 for deterministic.
The evolving spatial variation also makes later decisions less collinear.
This is noise induced through selection, not noise directly added to values.

On crossing, shared and local thresholds flip about 0.82% and 1.77% of
active decisions respectively, accounting for 3.67% and 6.56% of proposed
correction energy. On noise, about 0.68% and 0.64% of decisions change,
accounting for approximately 17.2% of correction energy. Thus the negative
result is not solely an inactive implementation: meaningful choices change
on the genuinely two-dimensional inputs, without a useful speed gain here.

## Checks and interpretation

`NoisyDefectGate` overrides only `admit_defect`; the deterministic operator's
decision was extracted into that method without changing its behavior.
Tests check bitwise equality to the former deterministic path, seeded
reproducibility, the acceptance probability, exact collinear decisions,
unchanged primal computation for a given input, selection of existing
feasible candidates, and preservation of a nonconstant-source fixed point.
All three new and four existing test groups passed on the Mini.

This particular confidence-based randomization did not improve the tested
defect gate. Shared randomness preserves spatial coherence better than
independent site noise; it also barely changes the almost binary ramp
decisions. Randomizing another statistic or noise distribution remains a
different hypothesis. No production gate was replaced.

Files: `noisy_defect_gate.py`, `probe_noisy_gate_decisions.py`,
`test_noisy_defect_gate.py`, `meyer_noisy_defect_gate128.json`, and
`meyer_noisy_gate_census.json`. The default `DefectRelaxation` decision remains
deterministic. Reproduction commands are in the repository AGENTS.md.
