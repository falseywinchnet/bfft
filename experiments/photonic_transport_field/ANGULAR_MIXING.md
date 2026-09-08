# Surviving intensity and surviving angular detail

The engine should distinguish attenuation from angular mixing. The former
reduces light; the latter can reduce the information needed to represent light
without removing its surviving energy. Neither geometric bounce count nor
optical phase decoherence alone certifies a scalar radiance closure.

## Physical operator

For a normalized scattering kernel P, the spatially homogeneous angular
collision subproblem is

    dL/ds = -sigma_a L + sigma_s (P L - L).

This excludes streaming, boundaries, emission, and changes of material. In the
complete equation, direction-dependent spatial streaming couples position and
direction again. Absorption and out-scattering attenuate a particular beam;
in-scattering supplies other directions. Extinction of the unscattered beam
is not extinction of all the light.

For a rotationally invariant kernel, spherical harmonics diagonalize P. If its
degree-l eigenvalue is k_l, the direct collision update is

    a_lm(s) = exp(-[sigma_a + sigma_s (1-k_l)] s) a_lm(0).

Normalization gives k_0=1. Thus total angular intensity decays by absorption
alone. Other modes decay according to the actual kernel's angular mixing.
For the Henyey-Greenstein kernel k_l=g^l. This is a controlled analytic example,
not a fitted surface BRDF or a mapping from the renderer's roughness to g.

After n discrete HG scattering events of constant survival rho, the factors
are rho^n g^(l*n). For rho=.95 and n=8, 66.34% of the intensity survives:

| g | Dipole remaining relative to intensity | Degree-8 detail remaining |
|---|---:|---:|
| .5 | .00390625 | 5.42e-20 |
| .9 | .43046721 | .00117902 |
| .99 | .92274469 | .52559649 |
| 1 | 1 | 1 |

A broad mixer can therefore retain considerable intensity while strongly
suppressing detail. A sharply forward kernel retains low-order directionality
much longer. The g=1 limit has no mixing. A mirror also preserves angular
detail, although its deterministic reflection is a different operator from
HG forward scattering. Smooth mixing generally approaches zero asymptotically;
an exact rank-one diffuse operator can remove its angular residual in one event.

At scattering optical depth 8 with no absorption and g=.5, the unscattered beam
is exp(-8)=.00033546 of its input, while the zeroth angular mode remains 1 and
the dipole remains exp(-4)=.01831564. Treating all extinction as energy loss
would discard almost all the light incorrectly in this homogeneous example.

## When an intensity-only state is justified

Write the local field as retained angular modes plus residual r. A finite
observer response W incurs error |<W,r>|. In a common orthonormal angular
basis, Cauchy-Schwarz bounds it by ||W_omitted||_2 ||r_omitted||_2.
Retaining only intensity is justified when this bound fits the observer's
remaining error budget, including all contributions sharing that budget.
The residual must include an independently bounded unrepresented tail.
A delta-direction observer is not L2, so this particular bound does not apply
to it; finite pixel/angular support or a stronger residual norm is required.

The downstream operator belongs in W. Apertures, caustic concentration,
visibility changes, and new light sources can invalidate a bound based only
on the preceding number of rough events. A scalar *spatial field* is also
different from replacing a whole scene region with a single brightness.

The retained architecture suggested by this derivation is surviving spectral
intensity, necessary low angular modes, and a bounded directional residual.
Closed specular components remain directional; sufficiently mixed components
can flow through a compact field. This report establishes the criterion and
checks the homogeneous operator, not a certified full-scene accelerator.

## Existing engine diagnosis

`native/regime_scene_native.cpp::specular_area` integrates area emitters through
a half-vector lobe. `rough_terminal_relation` follows one ideal reflection
direction and weights it by a primitive-edge coverage approximation. The
sealed Metal/Glossy continuation uses that one terminal; it does not propagate
the spread of directions produced by successive rough reflections.

Consequently, rough reflected scene light is not mixed by the same angular
operator as direct area-emitter light. The source populations are mutually
exclusive in this continuation: area emitters are excluded from the delta
branch. Adding both branch weights for the same illumination would be an
invalid energy-conservation test. A physical replacement needs a common
passive material operator and tests under equal incident illumination,
including grazing views, before using its mode decay as an admission rule.

The renderer already transports intensities rather than complex optical
amplitudes. Its geometric angle calculations do not represent optical phase.
Loss of optical coherence therefore cannot itself justify throwing away its
directional state. An independently observed dielectric total-internal-
reflection weighting issue also remains outside this diagnostic.

## Own-domain experiment: isolated, not promoted

`native/spectral_domains.hpp` decomposes the existing optical packet rule into
local source events and child continuations. Each event integrates over its
own wavelength support. The production renderer has no new include or CLI
dispatch; it is identical to the accepted pre-experiment source snapshot.
Only `native/spectral_domain_probe.cpp` includes the experimental header.

On three nearby dielectric rays around the expensive aperture-canyon pixel
(275,324), 297 pointwise comparisons across three bands agree with the original
tracer to a maximum absolute error of 8.89e-16. This verifies the sampled
event decomposition, not all scenes or all wavelengths.

The measured runs were slower on all nine band cases. At the central
pixel's red band, source quadrature increased from 26,670 to 93,157, and time
increased from 4.94 to 19.78 ms. These are focused single-run timings, not a
full-frame performance benchmark. Adaptive local shading work outweighs the
saved repetition on this probe.

Dense 512- and 2048-wavelength midpoint integrations provide independent
comparisons, not certified references. At (275.25,324.25), red, the event
integral is 2.083878 versus dense-2048 2.080692; its accumulated quadrature
estimate is only .0000871. Endpoint/midpoint topology discovery can miss
support and hit its resolution limit, so the reported estimate is not a
global error certificate. That prevents promotion even independently of cost.

Results are saved in `angular_mixing_m4/spectral_domain_probe.json` and
`angular_mixing_m4/angular_mixing.json`. The angular diagnostic checks HG
eigenvalues independently by 512-node Legendre quadrature, conservative
zeroth-mode transport, absorption, semigroup composition, clear/forward
limits, and the finite observer bound.

Run on the selected M4 host through m4build:

```sh
python3 -m unittest experiments.photonic_transport_field.angular_mixing -v
python3 -m experiments.photonic_transport_field.angular_mixing --out /tmp/angular_mixing.json
```

The standalone native probe compiles like `source_expansion_probe.cpp`, linked
with `retained_transport.cpp` and the CONV C object; run it without arguments
and save its JSON stdout. Copy all `/tmp` results back before another sync.

Physical references: [equation of transfer](https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer),
[normalized phase functions and HG](https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions),
and [rough microfacet reflection](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory).
