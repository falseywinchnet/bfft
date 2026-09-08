# First projected-signature SGD result

Date: 2026-08-29. Status: mechanism-positive, optimizer claim unproven.

## Design

No learning-rate sweep, momentum, scheduler, clipping, Newton surrogate, or
gradient amplification was used. All arms shared the same initialization,
step size, and transition budget. `SIG-scalar` was norm-matched to the shape
candidate at every transition.

The decisive deterministic assay was the chained Rosenbrock valley at
`lr=0.002`, `alpha=1`, float64. Thresholds were read every 100 transitions.

| dimensions | SGD step to 1e-4 | scalar step to 1e-4 | shape step to 1e-4 | shape step to 1e-8 |
|---:|---:|---:|---:|---:|
| 8 | did not reach | 6,800 | **2,700** | 5,300 |
| 16 | did not reach | 8,600 | **2,900** | 5,300 |
| 32 | did not reach | 11,300 | **3,500** | 5,900 |

After 20,000 transitions, ordinary SGD remained at losses 7.81, 15.63, and
31.47 respectively. Both restrained arms converged. Shape was 2.5--3.2x
faster than norm-matched scalar restraint at the `1e-4` threshold. Its mean
sampled gradient norm ratio was 0.76, 0.60, and 0.61 respectively, so the
effect did not come from a larger instantaneous update.

At the conservative control `lr=0.001`, all three 16-dimensional arms reached
`1e-4` at 10,100 transitions and the signature arm's ratio was 1.000. The
observer therefore becomes inert when the trajectory presents no material
incompatible motion.

## Ordinary MLP assay: iteration compression

The original sibling configuration was retained: 32 inputs, three hidden
GELU layers of width 128, condition number `1e4`, batch 512, 3,000 transitions,
and `lr=0.03` with an identical minibatch schedule.

The target is plain SGD's attained test floor at transition 2,999
(`0.310903`). Evaluation is every 25 transitions; the final transition is
also recorded.

| method | first transition to SGD floor | reduction vs SGD | mean gradient ratio |
|---|---:|---:|---:|
| SGD | 2,999 | -- | 1.0000 |
| SIG-scalar | did not reach | -- | 0.6677 |
| SIG-shape | **2,425** | **19% fewer** | 0.6646 |
| SIG-rotate | **1,925** | **36% fewer** | 0.8659 |

The restraint-only result is controlled: shape reaches the common floor
earlier while equally strong scalar damping never reaches it. `SIG-rotate` is
the separate active-transport branch. It is faster here but fails the
deterministic Rosenbrock stress assay, so it is not promoted to the standing
operator.

## Formal ablation boundary

On the 16-dimensional fixed Rosenbrock assay, the one-cell energy gate remains
the fastest successful law:

| variant | first transition to 1e-4 |
|---|---:|
| energy gate `k=tau` | **2,900** |
| amplitude gate `k=sqrt(tau)` | 4,000 |
| hard projection | 4,800 |
| two fused energy cells | 4,000 |
| four fused energy cells | 4,400 |
| half-strength, quartic, cycle-2, active rotate | did not reach |

See `ANALYSIS.md` for the closed-form geometry and smooth-descent bound.

Correct spherical parallel transport removes the apparent rank penalty:
transported rank 2 is exactly tied with rank 1 on both the deterministic and
MLP hitting-time assays. This is an algebraic completeness result, not an
additional acceleration. The capacity-preserving observer takes 2,500 MLP
transitions, slightly behind the normalized signature's 2,425. Tensor-wide
and row-local cells have identical measured hitting times.

## Negative branch

The earlier same-batch secant response operator projected the current gradient
against `y = g(theta_1; B) - g(theta_0; B)`. On the deterministic
16-dimensional Rosenbrock control at `lr=0.001`, SGD reached `1e-4` in 10,100
transitions and both secant restraint arms took 10,200. That operator is
rejected. The successful object is the carried projected signature, not a
finite-difference response axis.

## Anchor: high-LR transported lead--lag momentum

The earlier Wolf experiment was reduced to its deterministic two-timescale
momentum core and renamed **Anchor**: a slow EMA
state and a faster lead readout.  Its random multiplier, elementwise sign gate,
and `p -= lr*p` fallback were removed.  The transported variant parallel-moves
the slow state into the new signature frame and applies the live BFFT restraint
only to the fast parameter request.

On the same fixed MLP assay, Anchor used `lr=1.0` while the SGD controls retained
`lr=0.03`.  Two times are reported: first contact with the fixed SGD floor, and
the first evaluation after which every remaining evaluation through transition
2,999 stays below it.

| method | first floor hit | sustained floor residence | final test loss |
|---|---:|---:|---:|
| SGD | 2,999 | 2,999 | 0.310903 |
| SIG-shape | 2,425 | 2,425 | 0.310106 |
| Anchor lead--lag | 100 | 700 | 0.303060 |
| Anchor + state transport | 75 | 2,700 | 0.306837 |
| **Anchor + transport + output restraint** | **75** | **225** | **0.287721** |
| Anchor + transport + output/state restraint | 75 | did not sustain | diverged |

The selected composition is therefore 40x faster than SGD to first contact
and 13.3x faster to sustained residence at the identical target.  Transport
alone launches quickly but rings for most of the budget; live output restraint
removes that settling tail.  This is the observed role of restraint in Anchor.

A pressure point at `lr=1.5` touched the floor by transition 50 but did not
sustain it until 2,700; `lr=3.0` eventually diverged.  The standing result is
the `lr=1.0` mechanism, not the transient lower hitting time.

The dynamic trust controller retains the 75-transition first hit while making
the same `lr=1.0` ceiling usable on contrasting m-layer tasks.  See `ANCHOR.md`
for its local displacement certificate and the six-task acquisition versus
continuation boundary.

## Interpretation boundary

The result supports the mechanism claim: restraining transport-incompatible
shape can enlarge useful relaxation while scalar damping cannot explain the
gain. It does not establish broad superiority over tuned SGD, momentum, Adam,
or other optimizers. The next honest test is a predeclared small family of
curved deterministic objectives followed by multiple-seed ordinary ML tasks,
still keeping scalar norm matching and the no-amplification invariant.
