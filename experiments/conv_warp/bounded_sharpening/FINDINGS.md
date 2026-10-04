# Bounded quintic concentration

14 September 2026.

The finite CONV margin construction can admit a sharpening correction around
an already feasible potential. Its constraints can simultaneously carry range,
current-cone, derivative, trace, and moment requirements. The derivation and
proof are in [BOUNDED_GEOMETRY.md](BOUNDED_GEOMETRY.md). This reuses the existing
joint-admission mechanism with a different feasible base and explicit proposed
current concentration.

The quintic current concentration has a concrete geometric meaning: a monotone
interval's five signed currents move toward the central current, producing the
potential controls `(a,a,a,b,b,b)`. The screened two-dimensional proposal blends
toward the tensor product of these corner profiles. A nodal curvature weight
`W = (kx²+ky²)/(kx²+ky²+gx²+gy²)` is bilinearly interpolated onto shared controls;
`W=0` for a zero denominator. Differences use unit source spacing. The proposal
vanishes on affine data and at every source vertex. It is an axis-based candidate,
not an orientation-independent selection theorem.

## Measured line profile

The source is a 9×9 field constant along the vertical axis. A unit-contrast line
occupying `[4.085,5.085]` supplies exact source pixel averages `0.415` and `0.585`.
The study starts from `finite_joint_control_nets`, using its canonical Float64
joint admission around the native proposal. Unit target pixels are integrated
exactly by four-point Gauss quadrature on each polynomial half-cell.

| Admission of the unit-strength proposal | Integrated peak | Sampled maximum slope / base |
|---|---:|---:|
| Unmodified reference potential | 0.490625 | 1.000000 |
| Range and existing support-current cone | 0.522421 | 1.752761 |
| Above plus derivative cap, whole-cell certificate | 0.499816 | 1.217590 |
| Same derivative cap, half-cell certificate | 0.522421 | 1.752761 |

The derivative allowance in both capped runs is the same: twice the maximum
absolute first-derivative Bernstein coefficient of the base atlas, in unit
source coordinates. The certificate includes both Cartesian derivatives.
Half-cell subdivision refines only the certificate; it neither changes the
potential nor relaxes the allowance. The resulting line's refined derivative
coefficient maximum is 1.758397 times the original base coefficient maximum,
below the declared factor two. The integrated peak improvement is 6.48%.

The proof of the relevant tightening is particularly simple for the basic
quintic step. Its derivative coefficients are `(0,0,5,0,0)`. After one exact
subdivision the left-half coefficients are `(0,0,1.25,1.875,1.875)`, with the
right-half coefficients reversed. The same polynomial is certified against
its true maximum `15/8` instead of the coarse coefficient bound `5`.

These numerical gains describe this reference atlas. The published WASM
constructor has a different storage-tolerance admission path; the earlier
unchanged-WASM study measured an integrated peak of 0.521924 for the analogous
line. This screen does not measure an improvement of the live browser engine.

## General screen and finite checks

Eight source cases—constant, affine, step, thin line, diagonal, crossing,
smooth wave, and seeded noise—were tested at strengths 1, 4, and 16. Four
configurations were recorded: inverse diffusion, quintic concentration,
concentration with a whole-cell derivative certificate, and concentration
with the same allowance and half-cell certificates. Across 96 results, source
vertices were recovered exactly; the largest sampled range excursion was
`3.11e-15`; the most negative constraint margin was `-4.83e-14`, already present
in the reference input. The Float64 diagnostic uses a `1e-14` harmful-motion
deadzone and records nominal margins. It is not a directed-rounding certificate.

Eight focused tests pass. They include 120 exact-rational trials of overlapping
moment-preserving corrections, first-contact maximality of the completion ray,
zero-margin cases, higher mixed derivative identities, shared-edge zero-moment
bundles, exact subdivision, and the sharp quintic slope certificate.

The proposed geometry has mixed effects. At unit strength it improves the thin
line and step, but decreases the sampled maximum slope of the diagonal case by
about 6.4%. The initial inverse-diffusion proposal improved that diagonal while
leaving the thin line unchanged. Strong concentration changes smooth data and
noise. Groupwise admission and floating active-face handling do not supply a
monotone visual sharpness control as proposal strength increases.

The current image screen preserves nodal values and shared potential traces.
It does not impose cell-mass or derivative-trace equalities. The unit-strength
line's total atlas integral changes by `0.00079827`, and the crossing changes by
about `-1.33591`; thus moment preservation must be explicitly included when it
is required. The mathematical note gives finite zero-moment correction bundles
and states the additional requirements for global higher continuity. The tests
verify those bundle identities independently; they are not enabled in this
image proposal screen.

The fixed derivative allowance bounds the admitted field throughout each cell.
It does not establish photometric fidelity, a nonlinear transfer function,
continuity of the complete data-to-atlas map, or a universally preferable
sharpening proposal. No browser or production kernel was changed.

## Reproduction

The Mini's existing Python lacked NumPy, so these small screens ran locally
with the existing release-validation environment; no remote software was
installed. From the BFFT root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/ultimussecundai/.cache/pmnw-release-python/bin/python \
  -m unittest experiments.conv_warp.bounded_sharpening.test_geometry -v

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/ultimussecundai/.cache/pmnw-release-python/bin/python \
  -m experiments.conv_warp.bounded_sharpening.study \
  output/support_geometry/conv_bounded_sharpening/derivative-refined.json \
  --derivative-cap 2 --refine-derivative

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/ultimussecundai/.cache/pmnw-release-python/bin/python \
  -m experiments.conv_warp.bounded_sharpening.plot
```

Omit `--refine-derivative` for the whole-cell certificate. Omit both derivative
flags for range/current-only concentration. Use `--proposal inverse-diffusion`
for the initial proposal. JSON records are under
`output/support_geometry/conv_bounded_sharpening/`; the figure is
`bounded-concentration.png` with a vector PDF alongside it. The mathematical
note, study, plot script, tests, and this record are retained in the research
source directory.
