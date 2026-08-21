# Shallow odd-cubic baseline on the extant battery

## Protocol

The exact shallow real odd-cubic model that solved the high-rank N-D spiral is
run unchanged on all 23 existing tasks.  The comparison reuses the preserved
width-38 vanilla MLP, self-context, and continuous-frame-flow results rather
than retraining them.  All four columns therefore use the same task generators,
two paired seeds, 500 AdamW steps, batch 256, learning rate `3e-3`, and the same
observed/tail splits.

The reference models contain roughly 8,500–9,100 learned scalars.  The shallow
odd-cubic model varies from 308 learned scalars on scalar regression to 3,837
on the 16-D binary tasks.  It has no frozen atlas.

## Aggregate result

| Model | Validation | Held-out | Tail | Learning AUC | Seconds / fit |
|---|---:|---:|---:|---:|---:|
| Vanilla LELU MLP | .794 | .585 | .575 | .731 | .397 |
| Self-context | .912 | .672 | .647 | .852 | 4.286 |
| Continuous frame flow | **.926** | **.676** | **.661** | **.864** | 16.560 |
| Shallow odd-cubic | .626 | .510 | .507 | .587 | .400 |

It is not a generally superior replacement for self-context or CFF.  It is an
extremely cheap, sharply specialized algebraic baseline.

## The important split

| Task | Vanilla | Self-context | CFF | Odd cubic | Interpretation |
|---|---:|---:|---:|---:|---|
| N-D spiral, high rank | .469 | .841 | .784 | **1.000** | Exact cross-harmonic odd invariant |
| N-D spiral, low rank | .403 | .405 | .401 | .374 | No multi-plane relation to demodulate |
| Two moons | **1.000** | **1.000** | **1.000** | .928 | Strong for 617 scalars, but not a win |
| Pinwheel | **1.000** | **1.000** | **1.000** | .878 | Useful low-cost approximation |
| Lorenz lobes | 1.000 | 1.000 | 1.000 | .995 | Near-complete with 617 scalars |
| Checkerboard | .503 | .456 | .446 | .496 | Every method fails; odd cubic does not solve locality |
| Radial stripes | .525 | .552 | **.859** | .515 | No radial/transport induction |
| Multiscale 1-D | .134 | .258 | .179 | **.358** | Numeric win, but visually a smooth mean-like fit |
| Localized steps 1-D | .833 | .968 | **.989** | .284 | Global cubic cannot preserve localized regimes |

The high-rank/low-rank reversal is the central result.  Both N-D generators
have antipodal labels, but only the high-rank version supplies multiple coupled
harmonics.  A third-order product can cancel phase among those planes and leave
a stable class sign.  On the one-plane spiral, a low-degree polynomial still
has to chase a winding boundary and fails.

This means the new baseline is best understood as a **cheap algebraic
coherence detector**, not a spiral learner and not an omni-inducer.

## What the pictures add

The per-task atlas prevents several misleading metric conclusions:

- The multiscale scalar score is higher, but the odd-cubic curve suppresses the
  focal ripples and mostly follows a broad global trend.
- Its small positive scores on complex 3-D spiral and checkerboard are not
  structural wins; every compared model fails those endpoints.
- Two moons, pinwheel, and Lorenz lobes demonstrate impressive parameter
  efficiency even though the larger reference models retain the endpoint win.
- Periodic, radial, piecewise, and localized tasks expose exactly what a fixed
  third-order algebraic prior cannot induce.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m unittest ML_experiment.test_odd_cubic_battery

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/run_odd_cubic_battery.py \
  --out /tmp/odd_cubic_battery --width 38 --seeds 2 --steps 500
```

After copying the remote output to `ML_experiment/results_odd_cubic_battery`:

```sh
python3 ML_experiment/assemble_odd_cubic_battery.py
python3 ML_experiment/analyze.py ML_experiment/results_odd_cubic_full/results.json
python3 ML_experiment/build_odd_cubic_battery.py
```
