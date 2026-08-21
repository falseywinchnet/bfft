# Continuous operator frame battery

## Construction

The direct operator sphere begins as the learned-cone odd bridge and acquires
two transported derivatives from the self-context allocator's own selected
frame:

```text
u       = self_context_up(e)
q       = weighted allocator frame rays
u_plus  = self_context_up(e + radius q)
u_minus = self_context_up(e - radius q)

tangent   = mean((u_plus - u_minus) / 2)
curvature = mean(u_plus + u_minus - 2 u)
```

Odd relation, tangent, and curvature are RMS-matched. Their coefficients are
normalized onto one unit sphere and the result is added before the ordinary
shared LELU/down response. The initial coordinates are exactly `(1, 0, 0)`, so
the initial function is the learned cone and transport has no independent
scale or output path.

The nested operator chart uses the same probes but does not emit transport:

```text
nested_u = u + 0.5 tanh(s) * transported_curvature
relation = odd_bridge(e, nested_u)
hidden   = LELU(u + alpha * relation)
```

Thus curvature changes what the odd chart factor sees. It cannot directly
predict the target.

## Complete battery

All models use width 38, 500 AdamW steps, the same sampled task, and the same
training seed. Two ordinary LELU MLPs exactly match the trainable-parameter
budgets of relational self-context (~9k) and the cone/operator family (~26k).

| Model | Mean score | Mean tail | Learning AUC | Mean seconds |
|---|---:|---:|---:|---:|
| MLP, self-context budget | .5866 | .5826 | .7315 | .28 |
| MLP, cone budget | .5741 | .5634 | .7312 | .43 |
| Relational self-context | .6506 | .6319 | .8556 | 4.71 |
| Relational CFF | **.6728** | .6264 | .8650 | 19.10 |
| Learned cone | .6711 | **.6587** | .8652 | **5.11** |
| Direct operator sphere | .6606 | .6504 | **.8666** | 11.95 |
| Nested operator chart | .6679 | .6535 | .8657 | 12.05 |

Against self-context:

| Model | Score change | Score wins | Tail change | Tail wins |
|---|---:|---:|---:|---:|
| CFF | +.0222 | 15 / 23 | -.0055 | 13 / 23 |
| Learned cone | +.0205 | 15 / 23 | **+.0268** | 17 / 23 |
| Direct sphere | +.0100 | 14 / 23 | +.0185 | 17 / 23 |
| Nested chart | +.0173 | 15 / 23 | +.0216 | 17 / 23 |

The learned cone remains the Pareto baseline: almost the best mean score, the
best tail behavior, and less than half the operator-frame time. CFF wins mean
score by `.0017`, but is 3.7 times slower and loses mean tail retention.

The larger ordinary MLP is worse than the smaller ordinary MLP on all four
aggregate metrics. Capacity cannot explain the cone/operator results.

## Task-family separation

The full fitted-function atlas makes the separation visible:

- **High-rank N-D spiral:** nested `.9810`, cone `.9785`, direct `.9620`,
  self-context `.7865`, CFF `.6790`, both ordinary MLPs about `.43`.
- **Radial stripes:** CFF `.7236`, direct sphere `.5947`; all other mechanisms
  remain roughly `.52–.57`.
- **Ripple:** nested `.9447`, cone `.9442`, CFF `.9385`, direct `.9285`.
- **Multiscale 1-D:** cone `.3629`; nested is slightly lower in whole-domain
  score (`.3580`) but has the best tail (`.3744`).
- **Chirp:** CFF `.3327`; the operator and cone variants lose to ordinary
  self-context's `.3005` except for CFF.
- **Polynomial-drift chirp:** CFF `.2636` and cone `.2406`; both new operator
  forms regress badly.
- **Fourier mix:** CFF `.4982`, raw odd bridge `.5330` in the earlier screen,
  nested `.4330`, cone `.3908`, direct `.2669`.
- **Complex 3-D spiral:** every model extrapolates poorly; self-context remains
  the least bad at `.0518`.

## What the learned coordinates say

The direct sphere uses task-level curvature selectively:

- radial stripes: 95.7% curvature energy;
- ripple: 40.6%;
- Fourier mix: 13.8% curvature and 23.0% tangent;
- high-rank N-D spiral: 96.3% odd energy;
- localized steps and ring SDF: over 99.5% odd energy.

This is successful operator identification. It is not yet successful operator
composition: rotating toward curvature removes odd energy, so mixed problems
often underperform the unmodified cone.

The nested chart behaves differently. Its tangent coordinate remains almost
unused and its curvature coordinate stays near one. What changes is the signed
transport strength:

- ripple: `-.201`;
- periodic N-D: `-.258`;
- spiral: `-.203`;
- periodic wells: `-.180`;
- multiscale: approximately zero;
- radial stripes: approximately zero.

It learns to **subtract curvature from the chart interpretation** on several
oscillatory tasks. That improves high-rank spiral, ripple, periodic N-D, and
some tails without supplying radial topology. CFF's radial success instead
requires curvature to enter the transported state directly.

## Current conclusion

There are now three empirically distinct operations:

1. the odd cone exposes high-rank angular relations;
2. direct positive curvature transports radial topology;
3. signed nested curvature flattens or stabilizes the chart used to interpret
   oscillatory relations.

A single normalized mixture cannot preserve all three because selecting one
operator removes energy from another. The next design should therefore model a
connection: curvature should transport the odd frame itself while preserving
its norm, rather than replacing odd state or merely perturbing one chart
factor. This suggests parallel transport of the factor directions on their
hypersphere, with the connection learned from antithetic frame probes.

## Reproduction and viewer

```sh
python3 -m unittest ML_experiment.test_odd_context_hybrids

python3 ML_experiment/run_odd_context_battery.py \
  --out /tmp/operator_frame_full_battery \
  --variants ordinary_mlp_self_budget,ordinary_mlp_cone_budget,cff,self_contextual_operator_sphere_global_r2,self_contextual_nested_operator_r2 \
  --width 38 --seeds 1 --steps 500

python3 ML_experiment/assemble_operator_frame_battery.py
python3 ML_experiment/analyze.py \
  ML_experiment/results_operator_frame_battery_merged/results.json
python3 ML_experiment/build_operator_frame_battery.py
```

The generated fitted-function viewer is `ML_experiment/operator_frame_battery.html`.
