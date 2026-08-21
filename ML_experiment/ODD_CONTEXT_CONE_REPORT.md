# Odd relations inside self-context: cone and optimizer diagnostics

## Result

The chart-coupled odd bridge is a real mechanism improvement, not a capacity
effect.  Its strongest broad form is the learned factor cone:

```text
e = embed(x)
u = self_context_up(e)

a = A e / ||A||row^rho_a
b = B e / ||B||row^rho_b
c = C u / ||C||row^rho_c

relation = RMSNorm(a * b * c) * RMS(e)
h = LELU(u + softplus(alpha) * relation)
y = decode(self_context_down(h))
```

The three learned exponents are continuous.  `rho = 0` retains raw factor
magnitudes, `rho = 1` retains only directions on the row-wise hypersphere, and
intermediate values interpolate without a discrete gate.

On the 23-task, width-38, 500-step battery (one matched seed), learned cone
versus relational self-context gives:

| Metric | Learned-cone change |
|---|---:|
| Mean held-out score | +.0205 |
| Held-out score wins | 15 / 23 |
| Mean tail score | +.0268 |
| Tail wins | 17 / 23 |
| Learning AUC | +.0096 |
| Mean runtime | 5.11 s versus 4.71 s |

The large gains are structural rather than uniform:

| Task | Score change | Tail change | Mean learned `rho` |
|---|---:|---:|---:|
| N-D spiral, high rank | +.1920 | +.2877 | .548 |
| Multiscale 1-D | +.1628 | +.0848 | .494 |
| Fourier mix 1-D | +.1342 | +.0894 | .487 |
| Polynomial-drift chirp | +.1004 | +.0461 | .491 |
| N-D spiral, low rank | -.0040 | +.0673 | .554 |
| Chirp 1-D | -.1220 | -.0764 | .479 |
| Radial stripes | -.0478 | — | .500 |
| Complex spiral 3-D | -.0192 | -.0226 | .488 |

The high-rank N-D spiral is the clearest separation.  Fixed unit rays score
`.9985` with `.9973` tail score, learned cone scores `.9785` / `.9577`, raw
factors score `.9290` / `.8847`, and self-context scores `.7865` / `.6700`.

## Why the bridge is different

An output-parallel cubic branch collapses to only 2–4% of parent RMS.  The
parent solves the observed region first and removes the residual's gradient.
The hidden bridge cannot be bypassed in the same way.  It changes the state
before the shared LELU and down chart, and its chart factor depends on the
parent's own interpretation:

```text
J = D LELU [I + alpha d(relation)/du] du/dx
  + D LELU alpha d(relation)/de de/dx
```

Thus the ordinary path and odd relational path must co-adapt.  On the N-D
screen, the full bridge's parent-only observed accuracy is roughly `.60–.69`
while the complete model is perfect in-field and about `.91` unseen.  The
relation is state, not a separately supervised predictor.

The triple product is also a useful explanation of the old square/cubic/norm
experiment.  With tied source factors it becomes `(A e)^2 * C u`: source
energy conditions a signed chart view.  Untying the source factors supplies a
third-order CP atlas with more angular rank.  Normalizing the product and then
restoring authentic source RMS makes the bridge degree one and leaves the
ordinary path responsible for output range.

## Factor geometry

Three-seed N-D factor conditioning separated direction from rigidity:

| Factor geometry | Mean unseen accuracy |
|---|---:|
| Unit rays | .934 |
| Tight frame + RMS | .934 |
| Factor RMS | .916 |
| Raw factors | .911 |
| Hard tight frame | .885 |

Equal row length helps because no ray can win only by scale.  Hard
orthogonality hurts because it removes the continuous, redundant directions
needed by the learned atlas.  The useful manifold is spherical but not rigid.

The learned cone recovers some of both regimes.  Its task-specific movement is
small but coherent: the high-rank N-D spiral moves toward direction
(`rho ~= .55`), while ripple and Fourier-like tasks retain more raw magnitude
(`rho ~= .45–.49`).  Ordinary in-field loss stops moving the exponents once the
observed region is solved, so it cannot be expected to discover the tail-optimal
endpoint.

## Rejected extension: learned radial homogeneity

We tested a continuous radial-degree atlas without changing the initial
function:

```text
relation_j = angular_j * RMS(e)^degree_j
degree_j = 3 sigmoid(theta_j)
degree_j starts at 1
```

Both one global degree and one degree per ray usually reduced continuation.
The optimizer pushed global degree below one on the tasks where continuation
was most valuable:

| Task | Learned degree | Score change versus cone |
|---|---:|---:|
| N-D spiral, high rank | .873 | -.0005 |
| N-D spiral, low rank | .890 | -.0030 |
| Fourier mix | .886 | -.0584 |
| Multiscale | .953 | -.0735 |
| Polynomial-drift chirp | .948 | -.1606 |

The extension is exact at initialization, so this is not a capacity failure.
Backprop finds a lower-degree contraction that improves or preserves observed
loss, then sacrifices unseen amplitude and phase.  More expressivity created a
bad descent direction.  Radial stripes remained near chance, confirming that a
homogeneity exponent is not a substitute for CFF's radial transport operator.

## Rejected extension: independent direction and gain coordinates

We also reparameterized every raw factor row as
`exp(gain) * direction / ||direction||`.  This preserves the raw function class
and initial function while making direction gradients tangent to a sphere.
On the three-rotation 16-D spiral it scored `.9168 +/- .0213`, compared with
raw `.9107 +/- .0198`, learned cone `.9248 +/- .0324`, and fixed unit rays
`.9342 +/- .0257`.

The coordinate system changes basin selection but does not define a better
basin.  Independent gains restore the same unstable scale channel that fixed
unit rays remove.

## Current boundary

The odd bridge adds the missing high-rank angular interaction to self-context.
The cone makes its factor geometry continuously task-adaptive.  Neither
creates radial transport, recurrence, or an evidence horizon:

- radial stripes remain native to continuous frame flow;
- ordinary chirp and the complex 3-D spiral remain better served by the base
  self-context path;
- the high-rank N-D spiral is native to the unit-ray odd relation;
- the learned cone is currently the broadest single compromise.

The next mechanism should therefore not add another freely descending output
degree.  It should make optimizer geometry intrinsic: acquire a continuous
operator frame while preventing scale from becoming a shortcut.  Any blend of
odd chart transport and radial CFF transport must occur inside shared state,
not as an output mixture or a discrete task gate.

## Reproduction

Run all commands on the M4 CPU with the Torch path documented in the repository
`AGENTS.md`.

```sh
python3 -m unittest ML_experiment.test_odd_context_hybrids

python3 ML_experiment/run_odd_context_hybrids.py \
  --out /tmp/odd_context_weight_norm_nd \
  --models self_contextual_angular,self_contextual_full_row_unit,self_contextual_full_learned_cone,self_contextual_full_weight_norm \
  --width 24 --seeds 3 --steps 500

python3 ML_experiment/run_odd_context_battery.py \
  --out /tmp/odd_context_homogeneity_screen \
  --tasks nd_spiral_low_rank,nd_spiral_high_rank,radial_stripes,multiscale_1d,chirp_1d,poly_drifted_chirp_1d,fourier_mix_1d,complex_spiral_3d \
  --variants self_contextual_full_learned_cone_global_degree,self_contextual_full_learned_cone_ray_degrees,self_contextual_full_row_unit_ray_degrees \
  --width 38 --seeds 1 --steps 500
```
