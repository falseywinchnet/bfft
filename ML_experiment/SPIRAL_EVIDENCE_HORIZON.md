# Dual spiral evidence horizon

## Question

Does observing more repeated dual-spiral behavior cause an ordinary LELU MLP,
self-context, or continuous frame flow to infer the continuation law?

The experiment uses one fixed-pitch infinite dual spiral. The underlying target,
radial pitch (`0.45`), transverse noise (`0.018`), samples per turn, model width,
and optimizer budget are invariant. Only the observation cutoff changes:

- 2 visible turns followed by 2 withheld turns;
- 4 visible turns followed by 4 withheld turns;
- 8 visible turns followed by 8 withheld turns.

All learned models have exactly 8,542 parameters. Results average four paired
seeds after 500 AdamW steps on the M4 CPU. Checkpoint selection uses only
observed-region validation accuracy.

## Result

| Visible / withheld turns | Model | Observed accuracy | Withheld accuracy | Last withheld turn | Learning AUC |
|---:|---|---:|---:|---:|---:|
| 2 / 2 | Vanilla LELU MLP | .8714 | .4886 | .4947 | .6486 |
| 2 / 2 | Self-context | .9997 | .4689 | .5100 | .9043 |
| 2 / 2 | Continuous frame flow | 1.0000 | .4811 | .4889 | .9369 |
| 4 / 4 | Vanilla LELU MLP | .5725 | .4996 | .5131 | .5349 |
| 4 / 4 | Self-context | .9563 | .4862 | .5133 | .7359 |
| 4 / 4 | Continuous frame flow | .9889 | .4879 | .5019 | .8108 |
| 8 / 8 | Vanilla LELU MLP | .5221 | .5034 | .4894 | .5057 |
| 8 / 8 | Self-context | .5290 | .5032 | .5014 | .5106 |
| 8 / 8 | Continuous frame flow | .5738 | .5056 | .4983 | .5197 |

More visible turns do not induce extrapolation. At two and four turns the
contextual models learn the observed dual spiral, with frame flow acquiring it
faster, but every withheld-turn bin remains statistically at chance. There is
not even a reliable gain on the first unseen turn.

At eight turns the fixed 8,542-parameter, 500-step systems no longer acquire the
observed geometry reliably. That condition is therefore a capacity/optimization
saturation result, not evidence about continuation. The four-turn condition is
the decisive one: both contextual models fit the observed rule, yet neither
continues it.

## Interpretation

The result separates geometric interpolation from generator inference. The
models can place a highly accurate alternating partition over the observed
spiral without representing the radial-to-angular phase law that creates the
partition. Additional repetitions reinforce the fitted partition but do not
change its ontological status into a reusable transformation.

Continuous frame flow's radial-stripe success is consequently narrower than
general radial-angular reasoning. It is strongly biased toward coherent normal
or shell transport. A dual spiral requires coupling radial displacement to
angular phase and retaining that connection outside the observed support. The
current transported frame does not infer that connection.

Exact global radial symmetry is rare among arbitrary functions: invariance
under the full orthogonal group imposes a large family of constraints. But
distance-like structure is locally common in physical and perceptual problems:
wavefronts, signed-distance fields, prototype distance, isotropic kernels, and
normal coordinates near smooth boundaries. The radial win therefore matters,
but it should be read as evidence for a useful shell/level-set prior rather than
omni-inducement.

The graphical report is `spiral_evidence_horizon.html`; raw results and fitted
decision fields are in `results_spiral_evidence_horizon/`.
