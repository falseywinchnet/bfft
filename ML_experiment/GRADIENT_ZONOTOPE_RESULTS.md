# Optimizer-transported gradient zonotope

## Mechanism

At each accepted update, several structured acquisition pools produce gradient
samples `g_i`. Their common component and centered covariance define a tiny
update frame:

```text
g_common = mean_i(g_i)
g_i - g_common = covariance disagreement
```

The important correction is that AdamW is nonlinear and anisotropic. Mixing
raw gradients and then applying AdamW is not equivalent to mixing the candidate
AdamW transitions. The useful geometry is therefore constructed after
optimizer transport:

```text
(theta_i, m_i, v_i) = AdamW-preview(theta, m, v, g_i)
delta_i = theta_i - theta
delta_i = delta_common + U c_i + discarded_i
```

`U` contains the retained covariance directions of the candidate parameter
displacements. A structured witness fold scores the projected candidate
histories. Softmax weights form one continuous barycenter of the retained
parameter states and their first- and second-moment states. Only that state is
committed; inference uses one model, not an ensemble.

With three gradient samples and rank two, the entire empirical disagreement
subspace is represented in two coordinates regardless of the model's roughly
26,000 parameter dimensions.

## Sparse sine gate

The cold transported mixture (`temperature=1e-6`) produced the following
three-seed minimum observed-period R² values after 1,000 accepted updates and
3,000 gradient evaluations:

```text
0.9832, 0.9826, 0.9833
```

This is strikingly uniform. The earlier cloned discrete trajectories obtained
`0.9966, 0.9942, 0.9744`: slightly higher mean performance but a wider spread
and about 51.8 seconds per run. Optimizer-transported reduction takes about
43.1 seconds per run. Ordinary 3,000-step self-context remains faster and
slightly stronger on this task (about 25.5 seconds and mean minimum-period R²
`0.9926`), so the new method is not a compute Pareto win.

Temperature changes the learned continuation discontinuously. On seed 2:

| Temperature | Minimum observed-period R² | Scored extrapolation R² |
|---:|---:|---:|
| `1e-6` | 0.983 | -11.08 |
| `1e-5` | 0.883 | -17.55 |
| `1e-4` | 0.826 | **-0.97** |

The plotting horizon is now twenty unseen periods (`x=1` through `x=3`), four
times the scored extrapolation range. This does not change training or model
selection. It reveals slower oscillations, frequency drift, and divergence
which the five-period scalar score can conflate.

## Twenty-three-task battery

The battery compares ordinary self-context at 400 accepted updates against cold
and warm transported mixtures at 400 accepted updates. Because each transported
update consumes three gradients, a separate ordinary 1,200-step self-context
run is the compute-matched control.

Against the 400-step control, warm transport improves final score on 15 of 23
tasks and both temperatures improve learning AUC on 21 of 23. Against the fair
1,200-gradient control, each fixed temperature wins 11, loses 11, and ties one.
The useful result is selective rather than universal.

Large compute-matched wins for the better transported temperature include:

| Problem | 1,200-gradient self-context | Transported | Difference |
|---|---:|---:|---:|
| Fourier mix 1-D | 0.248 | 0.504 warm | +0.257 |
| Multiscale 1-D | 0.207 | 0.310 cold | +0.103 |
| High-rank N-D spiral | 0.785 | 0.865 cold | +0.080 |
| Spiral | 0.353 | 0.375 warm | +0.022 |
| Checkerboard | 0.451 | 0.471 warm | +0.020 |
| Hyperchecker | 0.516 | 0.527 warm | +0.011 |
| Localized steps | 0.980 | 0.988 warm | +0.008 |

Clear losses include radial stripes, ripple, complex spiral 3-D, periodic N-D,
and polynomial drifted chirp. Radial improves strongly over the 400-step model
but loses to ordinary self-context given the same number of gradients. Periodic
N-D remains unlearned. Complex spiral 3-D also remains structurally unresolved.

The low-rank/high-rank spiral split is especially informative: transported
covariance does not improve the low-rank version, but gives the largest
classification/tail gain on the high-rank version. This supports the idea that
the method retains useful disagreement between acquisition views; it is not a
generic periodic or spiral prior.

## Current conclusion

There is genuine information in the covariance of candidate optimizer
transitions. It can alter which structure self-context acquires, substantially
improve several high-rank or multiscale problems, and expose qualitatively
different long-horizon continuations. It is not yet an optimizer replacement:
wall time is about six times ordinary 400-step training, fixed temperature is
not universally beneficial, and several recurrence-heavy problems remain
untouched.

The next principled step is to infer temperature from the witnessed score gap
and retained covariance geometry, rather than choosing cold or warm per task.
Temporal integration must align consecutive covariance frames before averaging;
averaging branch identities directly was shown to suppress necessary
anisotropy.

The interactive viewer is `ML_experiment/gradient_zonotope_battery.html`.
