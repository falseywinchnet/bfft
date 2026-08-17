# Curvature self-context versus self-context: 22-problem superset

## Conclusion

Curvature self-context wins this round **as an acquisition mechanism**. Across
the complete 22-problem suite it improves mean learning AUC by `+.0072`, wins
learning AUC on 15 tasks, and has 10 meaningful acquisition wins against 3
meaningful losses.

It does not yet win as a continuation rule. Its mean final held-out score is
`.0112` lower and its mean tail score is `.0172` lower. Those aggregate losses
are concentrated rather than universal: multiscale, chirp, localized steps,
ripple, and one high-dimensional parity seed account for most of the damage.
The positive endpoint cases are structurally informative: high-rank N-D spiral,
checkerboard, and complex 3-D spiral.

This is not a contradiction. Curvature state helps the model discover how the
observed manifold bends, but the current update does not preserve the law that
determines how that bending should continue.

## Matched protocol

- Self-context and curvature self-context only.
- Exactly identical parameter counts within every task.
- Width 24, batch 256, 500 AdamW steps, two paired seeds.
- Complete 22-task suite, 88 benchmark fits.
- A further 44 seed-0 fits produce matched fitted-function views for every
  problem.
- CPU Torch on the M4 Mini; no MPS.
- LELU throughout; no GELU or task-aligned features.

## Aggregate result

| Metric | Self-context | Curvature self-context | Difference |
|---|---:|---:|---:|
| Validation score | .9152 | .9154 | +.0002 |
| Held-out score | **.6865** | .6753 | -.0112 |
| Tail score | **.6656** | .6484 | -.0172 |
| Learning AUC | .8437 | **.8509** | **+.0072** |
| Seconds per fit | **4.34** | 13.86 | 3.20× |

Curvature self-context wins learning AUC on 15/22 tasks, held-out score on
6/22, and tail score on 6/22. Saturated tasks such as moons, pinwheel, Lorenz
lobes, XOR, and ring SDF are effectively ties and should not be used to argue
for either mechanism.

## Per-problem differences

All values are curvature self-context minus self-context, averaged over paired
seeds.

| Problem | Learning AUC | Held-out | Tail |
|---|---:|---:|---:|
| Radial stripes | **+.0497** | -.0063 | -.0063 |
| Chirp 1-D | **+.0300** | -.1152 | -.0804 |
| Spiral | **+.0290** | -.0020 | -.0070 |
| Checkerboard | **+.0225** | **+.0195** | **+.0922** |
| Multiscale 1-D | **+.0168** | -.1592 | -.0801 |
| N-D spiral, low rank | **+.0137** | -.0143 | -.0438 |
| Complex spiral 3-D | **+.0116** | **+.0176** | **+.0210** |
| Localized steps 1-D | **+.0092** | -.0523 | -.0632 |
| Periodic wells | +.0073 | +.0008 | +.0008 |
| N-D spiral, high rank | +.0063 | **+.1498** | **+.0857** |
| Hyperchecker | +.0031 | +.0013 | +.0013 |
| Fourier mixture 1-D | +.0017 | -.0492 | -.0306 |
| Ring SDF | +.0002 | -.0000 | -.0000 |
| Lorenz lobes | +.0001 | -.0000 | -.0000 |
| Two moons | +.0000 | -.0004 | -.0004 |
| Swiss cheese | -.0002 | +.0007 | +.0007 |
| XOR quads | -.0004 | -.0009 | -.0009 |
| Sinusoid bounds | -.0010 | -.0023 | -.0023 |
| Pinwheel | -.0022 | +.0000 | +.0000 |
| Periodic N-D | -.0056 | -.0102 | -.0102 |
| Hypercube checker | -.0114 | -.0002 | **-.2313** |
| Ripple | -.0215 | -.0239 | -.0239 |

## Are high derivative factors hidden by second-order integration?

Yes, although “integration” is slightly too generous a description of the
current implementation.

For an allocation-derived contextual field `F`, the symmetric shell computes

```text
C_h(z; u) = F(z + h u) + F(z - h u) - 2 F(z).
```

Locally this contains

```text
h² D²F(z)[u,u] + h⁴ D⁴F(z)[u,u,u,u] / 12 + ...
```

The shell therefore mixes all even derivative orders. Odd derivative evidence
cancels. In a Fourier direction its response is proportional to
`-4 sin²(ωh/2)`: it resembles `-ω²h²` only at small `ωh`, then saturates and
aliases distinct high frequencies into similar states.

The present curvature-context path collapses this evidence further:

1. directional shell responses are averaged;
2. the result is lifted into activation coordinates;
3. its RMS magnitude is normalized away;
4. the normalized vector is added directly to the activation;
5. allocation is recomputed at the displaced point.

Thus the model receives the *direction of a curvature-induced displacement*,
not a separated derivative state with trustworthy scale. High derivatives can
be hidden by shell aliasing, directional cancellation, and normalization. The
network then composes this displacement through later layers, which acts like
an implicit integration but supplies no boundary condition or law of transport.

This explains the characteristic result:

- faster, better in-field acquisition on radial, chirp, spiral, checkerboard,
  multiscale, and low-rank spiral;
- strong final gains when curvature is distributed and coherent across planes,
  especially the high-rank N-D spiral;
- poor continuation when curvature changes frequency, amplitude, or regime,
  especially chirp, multiscale, ripple, and localized steps.

## The next structural correction

Curvature should not update position directly. A more faithful nested state is
triangular:

```text
observation z  →  first context v  →  curvature/context-change a
                              a updates v; updated v transports z
```

In other words, `a` should modify the learned contextual velocity or connection,
and only the updated context should move the activation. A continuous
projective-agreement measure between `v` and its transported version can control
how much authority the curvature state receives. This retains the acquisition
advantage while testing whether the local derivative law is stable enough to
continue. It adds no task identity and need not add parameters.

## Files

- `results_curvature_superset/`: raw 88-fit benchmark, paired summary, and 44
  fitted probes.
- `curvature_superset.html`: complete interactive comparison.
- `build_curvature_superset.py`: reproducible compaction and report builder.
