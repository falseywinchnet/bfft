# Transported zonotopic-mixture enrichment: first result

## Construction tested

The experiment leaves the CONV extrema selector, exact ITD knot law, and
irregular CONV synthesis unchanged.  Only baseline extraction is averaged over
an auxiliary measure.

The initial measure is represented by nested bounded zonotopes

\[
 Z_m=\langle0,r_mG\rangle,
\]

where the columns of \(G\) are orthogonalized Rademacher generators.  The
radial nodes and weights are the absolute five-point Gauss--Hermite rule:

| \(r_m\) | weight |
|---:|---:|
| 0 | 0.5333333333 |
| 1.3556261800 | 0.4441518440 |
| 2.8569700139 | 0.0225148227 |

These weights sum to one and reproduce the standard-normal second and fourth
radial moments.  Each nonzero generator is evaluated with complementary signs.

At level \(\ell\), every auxiliary generator \(q_{j,\ell}\) is split by the
same nonlinear CONV--ITD operator used for the signal:

\[
 q^B_{j,\ell}=\mathcal B(q_{j,\ell}),\qquad
 q^R_{j,\ell}=q_{j,\ell}-q^B_{j,\ell}.
\]

The signal baseline is the antithetic radial mixture over
\(q^R_{j,\ell}\), and the next auxiliary state is
\(q_{j,\ell+1}=q^B_{j,\ell}\).  This is a columnwise cubature approximation to
the measure pushforwards \((\mathcal B)_\#\mu_\ell\) and
\((I-\mathcal B)_\#\mu_\ell\); it is not yet an exact nonlinear zonotope-image
calculation.

## Full 512-sample synthetic

The record contains a known trend, steady 33-cycle component, quadratic chirp,
localized 91-cycle burst, amplitude-modulated 16-cycle component, and seeded
white noise.

| Method | Mean best component \(|r|\) | Correlation concentration | Trend MSE | Burst localization | Rotations | M4 time |
|---|---:|---:|---:|---:|---:|---:|
| Plain CONV--ITD | 0.53958 | 0.73741 | 0.00524465 | 0.52793 | 5 | 2.18 s |
| Static mixture, \(\alpha=.05\) | 0.59219 | 0.66314 | 0.00422200 | 0.54515 | 10 | 62.92 s |
| Transported mixture, \(\alpha=.05\) | 0.56373 | 0.74143 | 0.00144188 | 0.54455 | 5 | 30.07 s |
| Transported mixture, \(\alpha=.10\) | 0.57336 | 0.74531 | 0.00621717 | 0.56123 | 5 | 30.40 s |

The static mixture raises pairwise correlations but continues supplying the
same fine noise at every level.  It reaches the ten-level cap, rather than the
five rotations of the plain and transported decompositions, and has lower
correlation concentration.  Transporting the measure removes this persistent
fine-scale population.

At \(\alpha=.05\), the transported construction improves the best correlation
of every known generated component and reduces trend MSE by 72.5 percent in
the first eight-generator run.  At \(\alpha=.10\), the trend result is worse
than plain CONV--ITD.  Enrichment is therefore not harmless for arbitrary
amplitude.

Normalizing every transported rotation back to unit RMS also performed poorly:
it prevents the noise measure from becoming coarser and produced trend MSE
`0.010548` at \(\alpha=.10\).  The unnormalized pushforward is the meaningful
transport law in this experiment.

## Cubature convergence

The fixed \(\alpha=.05\) transported arm was rerun with increasing angular
generator counts.

| Generators | Mean best component \(|r|\) | Concentration | Trend MSE | Burst localization | Rotations | Time |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.53193 | 0.71148 | 0.00866399 | 0.53946 | 6 | 18.72 s |
| 8 | 0.56373 | 0.74143 | 0.00144188 | 0.54455 | 5 | 29.78 s |
| 16 | 0.56524 | 0.74540 | 0.00166503 | 0.54355 | 5 | 57.93 s |
| 32 | 0.55588 | 0.73550 | 0.00165064 | 0.54823 | 5 | 114.83 s |

Four generators do not resolve the angular integral.  The 16- and 32-generator
trend estimates agree closely and both reduce MSE by about 68 percent relative
to plain CONV--ITD.  Mean best component correlation remains above the plain
result.  The 32-generator oracle is approximately 53 times slower.

Four independent eight-generator banks all improved mean best component
correlation and burst localization, but one worsened trend MSE and created a
sixth rotation.  A small randomly oriented bank is not a reliable definition
of the measure.

## Present finding

Transporting the auxiliary measure rather than reinjecting static broadband
noise is the productive part of the construction.  On this synthetic, a
sufficiently resolved transported mixture improves component alignment and
substantially improves trend recovery without proliferating decomposition
levels.  The result is not parameter-free, not yet a theorem, and not yet
computationally competitive.

The next mathematical problem is now narrower: replace angular sampling by a
compact deterministic cubature or moment-preserving zonotopic mixture
reduction that approaches the 16--32-generator result at bounded cost.
