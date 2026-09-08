# Dynamic factors: first certificate-driven investigation

September 4, 2026. This follows the user's requirement that a factor be
continuously determined by formally reliable statistics that are cheap to
obtain. The first prototype establishes a useful local acceptance contract;
it does **not** yet meet the cost or trajectory-performance requirement.

## What is measured

Let T be the complete six-field warm transport map, z the current state,
h = T(z) - z, and z(alpha) = z + alpha h. Every proposal uses the current
transport direction, including both newly computed driving fields. It does
not mix residuals from different inner problems.

The recovered feasible pair and dual witness in certificate.py give

    Gamma(z) = P(z) - L(z) >= 0,
    P = TV(u) + lambda/2 ||f-u-vF||^2,
    L = <f,q> - ||q||^2/(2 lambda) - mu TV(q),
    q = -eta_u div Pi(tu),
    vF = -(eta_w/cw) div Pi(tw).

P is an upper bound and L a lower bound on the same discrete model optimum.
This measures the recovered feasible pair. The emitted texture f-u-w is not
automatically that feasible texture. This distinction matters when judging
whether a visually improved output is closer to the objective minimum.

## Reliable statistic, heuristic proposal, checked decision

dynamic_factors.py computes Gamma and its exact right directional derivative
along h using projections, spatial differences, and reductions. No extra
transport-map evaluation or FFT is required. The disk projection derivative
uses the tangential exterior formula outside the disk and for outward motion
at its boundary; inward motion uses the interior derivative. At zero spatial
gradient, the right derivative of TV contributes |D h|, rather than zero.
Thus the statistic includes nonsmooth cases that an ordinary gradient formula
would miss.

Write G0 = Gamma(z), G1 = Gamma(T(z)), and s = Gamma'(T(z);h). A quadratic
matching those two values and the right slope at alpha=1 has curvature
c = G0 - G1 + s. If s < 0 and c > 0 it proposes

    alpha = 1 + min(1, max(0, -s/(2c))).

For negative slope and nonpositive curvature it proposes 2; otherwise it
uses 1. The interval [1,2] is a bounded search domain, **not** a proved
stability interval. The quadratic is **not** a majorizer. Exact values and
derivatives do not turn this interpolation into a reliable global model.

One- and two-trial variants evaluate the actual proposed state's certificate.
They accept only if

    Gamma(z(alpha)) <= Gamma(T(z)) + recorded roundoff allowance.

A rejected proposal halves the advance above 1; exhausted trials return the
ordinary endpoint. This establishes a same-input, one-step certificate
comparison. It does not establish monotonic decrease from Gamma(z), primal
energy descent, convergence, or domination of the full ordinary trajectory.
The allowance is an engineering floating-point tolerance, not an interval
arithmetic proof of error bounds.

A third variant caps the proposal at the first positive disk-boundary event,
computed from quadratic ray intersections. The root locates a branch event;
it does not certify descent or an affine exterior map. This cap was tested
because it follows the current projection geometry rather than historical
residual correlation.

## Executed tests and experiments

Four focused tests passed on the M4 Mini: directional behavior at projection
boundaries; generic finite-difference agreement and certificate-value
agreement; a known disk-boundary intersection; and the acceptance contract
over 24 steps at three parameter pairs.

The numerical screen uses five sources (ramp, edge, low-frequency carrier,
crossing, and uniform noise), three (lambda, mu) pairs ((.02,20), (.05,40),
(.1,80)), and 128 passes, at both 64-square and 128-square resolution. Every
method uses the identical NumPy FFT map and four-pass ordinary prefix.
Timing includes selection, projections, rejected trials, and state updates;
diagnostic checkpoints are excluded for every method. These are single-run
screen timings, not randomized native performance benchmarks.

| Size | Method | Lower final gap than ordinary | Median gap / ordinary | Worst gap / ordinary | Median elapsed / ordinary |
|---|---|---:|---:|---:|---:|
| 64 | fixed 1.75 control | 13/15 | 0.231 | 4.169 | 1.098 |
| 64 | one trial | 13/15 | 0.650 | 1.243 | 3.011 |
| 64 | two trials | 14/15 | 0.618 | 2.167 | 3.074 |
| 64 | boundary cap | 12/15 | 0.697 | 2.153 | 3.510 |
| 128 | fixed 1.75 control | 12/15 | 0.486 | 2.197 | 1.081 |
| 128 | one trial | 11/15 | 0.669 | 3.820 | 2.788 |
| 128 | two trials | 12/15 | 0.669 | 3.585 | 2.813 |
| 128 | boundary cap | 12/15 | 0.921 | 3.606 | 3.089 |

The one-/two-trial median selected factors are 1.035/1.168 at size 64 and
both 1.000 at size 128. They range from 1 to 2. The geometry cap has medians
1.0094 and 1.0006: the first event across an entire field often leaves very
little advance. Its weaker larger-resolution progress is evidence against
using the earliest pixel event as the sole global decision statistic.

The 128-square low-penalty ramp is particularly informative: the ordinary
gap per pixel is about 0.000020; the one-trial result is about 0.000077.
The absolute gaps are small, but the relative degradation is real. All
accepted comparisons can hold while the evolving state takes a worse route.
At 128 square, high-penalty edge and low-frequency-carrier cases also expose
failure. Adding a second trial does not remove these limitations.

## Interpretation and next mathematical target

This experiment rejects the proposed selector as a finished acceleration.
It is an instrument for separating three questions: is a statistic correct,
does a decision have a stated guarantee, and does that guarantee buy useful
progress per elapsed time? Only the first two have been established locally.

The central issue is transport memory. A gap sees today's recovered primal
and dual witnesses; a large step also changes the retained fields that build
tomorrow's coupled problem. A locally better witness need not preserve the
most useful future state. That is a concrete reason to investigate a bound
on the **complete-state transport defect** alongside objective progress.

The next useful derivation would bound the nonlinear defect
T(z+alpha h)-T(z)-alpha A h through the actual projection remainders and the
screened operators. Boundary crossing must be handled explicitly: persistence
of a disk branch does not make its exterior derivative constant, and a
uniform quadratic remainder is false at a boundary under the interior
derivative convention. Such a bound would need to justify its factor decision
without another full map and remain inexpensive after every memory traversal
is charged. That is a research target, not a proved rule in this implementation.

There is also an engineering opportunity to reuse certificate terms and fuse
their reductions into the native transport loops. The present NumPy prototype
rebuilds several arrays and computes diagnostics through a general certificate
routine. Its 2.8–3.5x cost does not prove that native statistics must be that
expensive, but it supplies no evidence yet that they can be cheap enough.
Neither optimization should be called a success before a native same-accuracy,
same-time comparison and a broader held-out parameter screen.

## Artifacts and reproduction

Full per-step decisions, proposals, slopes, curvature fits, trial counts,
accepted gaps, and allowances are in meyer_dynamic64.json and
meyer_dynamic128.json. Source is dynamic_factors.py; tests are
test_dynamic_factors.py. Run from the repository through m4build:

    python3 -m unittest experiments.meyer_transport_audit.test_dynamic_factors -v
    python3 experiments/meyer_transport_audit/dynamic_factors.py --size 64 --steps 128 --out /tmp/meyer_dynamic64.json
    python3 experiments/meyer_transport_audit/dynamic_factors.py --size 128 --steps 128 --out /tmp/meyer_dynamic128.json

Copy each /tmp result back before another mirror sync. No production preset
or public API was changed.
