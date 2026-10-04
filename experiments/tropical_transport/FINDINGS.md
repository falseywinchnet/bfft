# Tropical vs ordinary-linear transport of Sinkhorn trajectories

Question: is a slow entropic iteration a Maslov interpolation between a
tropical-linear map (far regime) and an ordinary-linear map (near regime),
so that transport should switch algebra where the two remainder bounds cross?

## Exact facts used (proved in probe.py docstring / derivation)

* Tropical limit T of a Sinkhorn sweep is a min-max map; on a frozen
  argmax/argmin piece it is a selection map y'_j = y_sigma(j) + c_j.
* For every state, F(y)-T(y) lies in [-log n, log m]; F and T are osc
  (Hilbert projective metric) nonexpansive, so osc(F^H y - T^H y) <= H(log n + log m).
  Displacement-independent, unlike the Taylor remainder osc(d)^2/8.

## Result (results/tropical_probe_v1.json; n=m=128, 2 seeds, eps .1-.003)

The hypothesis in its stated form is refuted.

* Tropical transport never describes the trajectory: its error is of the
  same order as the total motion (e.g. eps=.003, k=0, H=64: motion 76,
  tropical error 40). Near the solution it is useless (error 4-96 while
  the true state moves <1e-3); T^H keeps drifting toward its own limit.
* Frozen tropical pieces are worse still: error grows linearly in H
  (nonzero cycle drift of the frozen piece).
* Tropical beats Krylov only in the far, large-step phase at small eps and
  long H (eps=.003,k=0,H=256: 62 vs 467), where both are poor, and that
  phase lasts only tens of sweeps anyway.
* Krylov is extremely accurate once per-sweep motion is small
  (relative error 1e-3..1e-9); its bound is loose by 1e2..1e5 in the far phase.
* Bound crossover roughly marks the empirical far-phase crossover, but
  both bounds are too loose to be predictive.

## Why: slowness and tropicality are anti-correlated

The tropical slack is ~2 log n per sweep in log units. It is negligible only
if per-sweep motion >> 2 log n. But iteration time is spent precisely where
per-sweep motion is SMALL (that is what "slow" means), which is exactly
where the Taylor remainder is tiny. Small eps makes the early phase more
tropical, not the slow phase. The algebra-switching picture conflated
"small eps" with "large steps".

## What this points to instead

Where Krylov errs appreciably is the intermediate phase (eps=.003, k=10:
relative error ~20-40% at H>=16), in which steps are moderate and the local
rule J is still changing. That is Astra's actual question: transporting the
evolution of J itself (alpha recurrence (10), geometry update (5)) across
slow, long horizons, not replacing the algebra of J.

Reproduce (Mini; copy the JSON back immediately):

    /Users/ultimussecundai/.local/bin/m4build -- env OPENBLAS_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 python3 -m experiments.tropical_transport.probe \
      --out /tmp/tropical_probe_v1.json
