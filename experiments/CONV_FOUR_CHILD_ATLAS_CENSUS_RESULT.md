# Four-child moment atlas: expanded analytic census

## Question

The fused four-child atlas recovers detail discarded by a coarse
piecewise-constant raster, but its child cells need not remain inside the
parent sample range.  The census asks where positive target-footprint
integration suppresses that internal excursion, where the recovered detail
improves exact-target fidelity, and where the atlas must remain a scale state
rather than an enlargement operator.

## Design

The source and target values are uniform-cell averages of declared continuous
synthetics.  Plane waves and their affine pullbacks use closed-form sinc
averages; half-plane and strip cells use exact polygon coverage.  The reference
arm is exact positive overlap of the piecewise-constant source cells.  The two
candidate arms apply one or two native four-child atlas levels and then the
same positive overlap.  Thus no competing interpolator supplies truth and the
only changed object is the subcell representation.

The 9,648-case population contains plane waves over seven radial bands,
twelve orientations, and eight phases; crossed waves; half-plane interfaces;
thin strips; three affine maps; ten source/target scale pairs spanning
reduction, identity, and enlargement; and both one-level and two-level atlas
states.  Each record retains MSE, maximum absolute error, signed target gain,
internal child excursion, final source-range excursion, final exact-target
range excursion, parent-mean residual, and factorwise sign-surplus diagnostics.

## Scale separation

The fourfold atlas results are:

| regime | cases | MSE wins | losses | ties | geometric-mean MSE advantage | nonzero final source-range excursion | maximum excursion |
|---|---:|---:|---:|---:|---:|---:|---:|
| reduction | 6,000 | 5,876 | 36 | 88 | 13.03x | 6 | 0.004793 |
| identity | 912 | 0 | 0 | 912 | exact parent means | 0 | below 0.000000070 |
| enlargement | 2,736 | 2,542 | 194 | 0 | 3.285x | 2,436 | 2.57836 |

The result is not one undifferentiated quality claim.  During reduction, the
target footprint averages many atlas children and almost always suppresses
their internal excursion.  At identity scale, parent-mass conservation returns
the source cell means.  During enlargement, small target footprints expose the
child state itself; its range excursion therefore becomes visible and grows as
the target approaches the atlas resolution.

One additional atlas level is usually useful for reduction.  Relative to the
twofold state, the fourfold state lowers MSE in 5,893 of 6,000 reductions,
raises it in 15, and is numerically tied in 92.  Its geometric-mean advantage
over positive source-cell overlap rises from 5.21x to 12.86x.  The same level
also increases the maximum exposed enlargement excursion from 1.35127 to
2.57836, so level depth cannot be selected from MSE alone when a final range
law is required.

## High-frequency result

Every one of the 4,320 reduction plane waves improves in MSE.  The fourfold
atlas geometric-mean improvements are 46.39x below half the limiting Nyquist,
13.93x in the remaining resolvable band, and 6.60x above the target Nyquist.
Median signed gains in those bands improve, respectively, from
0.9736/0.8668/0.7807 for positive source-cell overlap to
0.9975/0.9689/0.9193 for the fourfold atlas.

This supports a precise empirical statement: for the sampled plane-wave
population, the moment atlas transports substantially more of the declared
continuous field's cell-average modulation through reduction.  It does not
establish recovery of arbitrary above-Nyquist content, nor does it turn the
internal child state into a range-contained interpolant.

The non-sinusoidal cases locate the remaining boundary.  All 96 crossed-wave
reductions improve.  Half-plane interfaces have no material losses.  All 36
material reduction losses occur in thin strips; the worst is an axis-aligned
one-target-cell strip for which positive overlap is already exact.  The atlas
changes that exact answer by MSE `2.7582106e-5` while remaining within the
source range.  It has reconstructed subcell curvature where the target
functional required none.

## Enlargement is a different operation

The atlas still improves most enlargement MSEs, but that fact is insufficient
to make it the canonical enlargement operator.  In the resolvable high
plane-wave band it wins all 864 cases, yet every case has nonzero final
source-range excursion.  Above the source-limited Nyquist, it wins 144 and
loses 144 cases.  The worst case is a 25-to-97 axis-aligned plane wave at
1.05 times the source Nyquist: the positive-overlap MSE is `0.2500562`, while
the twofold and fourfold atlas MSEs are `0.3068304` and `0.3155315`; their
source-range excursions are `0.5896353` and `0.9316769`.

Even below half Nyquist, ten 25-to-37 axis-aligned cases lose slightly in MSE
although their signed gains move closer to one.  This is a phase-error example:
more modulation energy is not synonymous with lower pointwise squared error.
Direct admitted CONV therefore remains the enlargement/interpolation operator;
the four-child atlas belongs to conservative analysis, lifting, and
detail-retaining reduction.

## Arithmetic audit

The investigation found two binary32 representation defects which are now
covered by tests.  First, an exactly vanishing coarse face had been classified
by `a+b==0`; SIMD/scalar rounding could make the same analytic cancellation
nonzero.  The implementation now applies the forward-error zero test

\[
 |a+b|\leq\gamma_8(|a|+|b|),\qquad
 \gamma_8=\frac{8\epsilon}{1-8\epsilon}.
\]

Second, NEON lanes and the scalar line-bank tail used different contraction
paths.  Equal transverse lines could differ by one ulp and the second atlas
level could amplify that residue.  Scalar tails and endpoints now use the
same explicit fused multiply-add evaluation, and a two-level constant-line
test requires bitwise equality across the SIMD boundary.

Parent recovery remains within binary32 rounding, but the recursive native
census still observes a maximum factorwise sign surplus of two in rare
near-zero cases.  The exact theorem is unchanged; the measured surplus is a
finite-precision conformance residual and must be reported as such unless a
future implementation carries a certified zero class or higher-precision
state between levels.

## Paper consequence

The theoretical insertion remains small: define the composed four-child map,
state parent-mass and factorwise variation control, and describe its fused
CONV* realization.  The high-frequency gains, the thin-strip counterexample,
the scale split, and the binary32 audit belong only in the experimental insert.
No theorem should say that positive footprint integration always removes
internal range excursion.

The full record is `output/support_geometry/conv_four_child_atlas_census_scale_full.json`.
