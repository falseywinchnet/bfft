# Strong frames, Gaussianity, and directional growth

The user's follow-up asks whether increasing Gaussianity and the number of
directions smoothly reduce the benefit, and requests investigation of the
strongest frame changes. This remains primitive relation transport, with no
BFGS, objective descent controller, or stationary surrogate solve.

## What fails in a strong frame

The transported input columns are `D Q`, where `D=diag(x_old/x_new)`.
The condition number of this diagonal change itself is

    cond(D) = exp(osc(log(x_old)-log(x_new))).

Augmenting the transported columns with the new constant direction can make
them nearly dependent. Orthogonalization then subtracts almost equal input
and response columns and divides by a small residual norm. Any response
roundoff is divided by the same number. Repeating this operation on already
transformed images compounds the loss. This is a failure of the chosen
numerical representation, not a failure of the exact diagonal identity.

There is a second, separate issue. Even perfectly transported old directions
need not cover the requests produced by the new full reciprocal normalization.
Preserving the algebraic relation does not preserve its usefulness to the
future request family.

## Three attempts to separate and repair those effects

1. `guarded`: carry a componentwise response-error estimate through every
   frame change and orthogonalization. Drop columns whose propagated error is
   too large. Include response uncertainty as well as omitted-input residual
   in the product acceptance test; measure the live product if it fails.
2. `spectral`: use an SVD of the transformed input columns to choose well
   conditioned combinations, with the same uncertainty accounting. This is
   a basis representation operation, not an optimization or potential solve.
3. `atomic`: retain original measured physical input-output pairs. Re-express
   those original measurements in a new frame rather than repeatedly feeding
   an already transformed response into the next transformation. Apply the
   same conditioning and uncertainty gates.

The `reset` control adds the uncertainty bookkeeping to the simpler cache
without attempting whole-basis transport. `cache` is the previous lightweight
cache; `ordinary` is the direct recurrence. All reciprocal requests remain
full vectors. Failed queries, new response measurements, fallback products,
representation transformations, and bookkeeping are charged.

The componentwise numerical estimates use conservative floating-operation
budgets, including `64 max(m,n) eps_machine`; these are engineering estimates,
not outward-rounded interval proofs. Consequently a gate rejection does not
prove that a direction is mathematically unusable. Separate small-case audits
measure actual represented-action errors with full kernel products, outside
performance runs. No audit measurement is fed back into the algorithm.

## Hard-case result

The original n=1024, epsilon=.001 cases run 256 ordinary passes with two seeds
and three complete timing repetitions. Error is the final gauge-free log-state
discrepancy from those ordinary passes, not an error to a solved optimum.

| Method | Seed 0 products | Seed 1 products | Median ms, seed 0 | Median ms, seed 1 | Largest trajectory error |
|---|---:|---:|---:|---:|---:|
| ordinary | 512 | 512 | 73.27 | 60.73 | 0 |
| cache | 442 | 433 | 75.25 | 75.25 | 2e-08 |
| reset | 497 | 493 | 103.03 | 104.50 | 1.44e-08 |
| guarded | 691 | 663 | 305.46 | 268.59 | 4.67e-09 |
| spectral | 614 | 594 | 152.47 | 151.03 | 1.43e-08 |
| atomic | 708 | 711 | 434.54 | 440.33 | 2.58e-09 |

All three repairs remove the large accuracy failure of the earlier raw
whole-basis transfer, whose hard-case errors reached 1.94e-3. They do not
produce a competitive accelerator. The spectral arm is substantially cheaper
than repeated guarded Gram-Schmidt, but still uses more products than ordinary.
The immutable-response arm establishes that merely preventing recursive
re-expression of old images is insufficient; it also loses on complete cost.
The basic reset cache remains the least expensive tested carried form on
these hard problems, and it still loses to ordinary iteration.

Audited n=128 hard trajectories had no measured positive violation of the
new numerical estimate. At frame boundaries, largest represented-action
errors were about 5.8e-15 (guarded) and 2.4e-14 (spectral). Their admitted
uncertainty ceilings approached 1e-9, so the gates are demonstrably
conservative. The experiment does not prove that all the rejected directions
must be rebuilt by every possible representation.

The non-audited n=1024 raw run predates an output-label fix: its
`audit_violation: 0` fields are unexecuted default values, not performed audits.
Current code emits `audit_enabled: false` and a null audit value in that case.
Only `strong_frame_audit` and `strong_frame_spectral` contain those audit
observations. Timing results from audited runs must not be compared with the
non-audited final timing table.

## Gaussianity and number of directions: isolate the claim

Two different experiments are necessary. Gaussian source distributions,
Gaussian conditional responses, and Gaussian-valued input direction fields
are not equivalent properties.

First, an ordinary-Sinkhorn sweep uses empirical point clouds in dimensions
1,2,4,8. The distributions range from single Gaussians to standardized
separated mixtures, with covariance controlled in the mixed coordinate. Two
seeds, two kernel widths, and three mixture separations give 48 cases. The
cost is divided by its median before choosing the kernel width. The probe
also measures conditional skewness/kurtosis and the effective rank of the
actual log-update directions.

That sweep does not support a universal monotone law in ambient dimension or
source Gaussianity. Higher ambient dimension sometimes makes the chosen
normalized kernel easier, and the actual update effective ranks remain only
about 1.2–2.3. Moreover a fixed 256-pass comparison gives excessive credit
for cheap reuse after an easy problem is nearly stationary. A separate
moving-prefix diagnostic stops the benchmark horizon at the ordinary
reference's first 1e-8 marginal residual, capped at 256. That reference is
scoring infrastructure only, not a runtime acquisition oracle. At kernel
width .1, the fraction of products required over that moving prefix averages
.197, .212, .359, .344 across dimensions 1,2,4,8; the misleading full-256
fractions were .195, .181, .099, .076. At width .03 none reaches that reference
threshold in 256 passes, and the dimensional trend is different again.

Second, a controlled primitive-query probe explicitly excites 1,2,4,8,16
modes. Its direction fields interpolate between binary and Gaussian entries:

    w = sqrt(g) Normal(0,1) + sqrt(1-g) Rademacher,
    h(t) = 1.4/sqrt(d) sum_l w_l sin(omega_l t + phase_l),
    x(t) = exp(h(t)).

The frequencies occupy the same [.7,1.3] interval in every dimension; this is
not a sweep that increases frequency range with direction count. The input
field columns are standardized. The carrier sees only each actual full
positive request, not the generating coefficients. K is held fixed. This is
an isolation test of primitive response reuse, NOT a Sinkhorn trajectory or
an end-to-end acceleration benchmark.

With fixed RMS amplitude, the mean product counts out of 256 queries are:

| Modes | Binary fields | 50% Gaussian variance | Gaussian fields |
|---:|---:|---:|---:|
| 1 | 2 | 81.5 | 81.5 |
| 2 | 4 | 222 | 220.5 |
| 4 | 24 | 228 | 231.5 |
| 8 | 228 | 234.5 | 233 |
| 16 | 235 | 235.5 | 242 |

Gaussian fields have larger ranges at equal RMS. Therefore a second 60-case
probe matches every request's log-range to exactly 6 and uses Gaussian
variance fractions 0,.001,.01,.1,.5,1. At two modes the counts are
35,196.5,205.5,209,213.5,203. At four modes they are
182.5,251,251.5,252,252,255. At eight and sixteen modes nearly all 256 products
are required. The equal-range one-mode family has only two normalized
requests, so every Gaussian fraction takes about 5.5 products; it is a
sanity check and not evidence that Gaussianity never matters.

Thus the controlled query experiment supports diminishing reuse as the
multiplicative request family becomes richer. It does NOT establish smooth,
monotone decay in a Gaussianity parameter. At the strict 1e-8 admission
threshold and fixed rank budget 32, a very small continuous perturbation can
move an exactly finite-pattern family out of its cheap representation.

## Why the small family is special

For d binary fields, the coordinate locations have at most 2^d joint sign
patterns. Every `exp(sum_l s_l w_l)` is constant on each such pattern.
The pattern-indicator space is closed under multiplication and division
where the denominator is nonzero. Its fixed K responses can therefore be
reused exactly for this request family. This explains the 2/4/roughly-16
initial response economy without invoking a fitted potential.

Generic continuously valued fields do not share that finite-pattern algebra.
Even one field with n distinct values generates n linearly independent
polynomial evaluations (the Vandermonde fact). This is an exact-algebra
statement, not an impossibility theorem for accurate approximate compression.
For several fields, mixed products appear as exponentials and reciprocal
normalizations evolve. An ordinary small linear span need not be closed
under those operations.

The object to seek is consequently an economical representation of the
multiplicative request family and the fixed operator's action on it. The
current tests identify that closure requirement; they do not yet construct
a cheap general closure for the hardest frames. Gaussian appearance is a
possible correlate of lost pattern structure, not an intrinsic certificate
of transportability or its absence.

## Files and verification

`frame_transport.py`: guarded, spectral, and immutable-response attempts.
`probe_frame_transport.py`: strong-frame comparisons.
`probe_frame_dimension.py`: actual ordinary-trajectory mixture/dimension sweep.
`probe_direction_fan.py`: controlled primitive family, with `--fixed-range`.
`report_frame_transport.py`: the standalone comparison plot.

Five focused tests passed on the Mini, including the previously failing hard
trajectory for all three new representations and the independent represented
response audit. The existing default primitive cache is unchanged. Results
are in `results/theory/strong_frame_*.json`, `frame_dimension_*.json`, and
`frame_direction_fan*.json`. The plot is `frame_direction_fan.png`.
