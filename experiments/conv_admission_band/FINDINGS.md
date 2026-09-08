# CONV admissibility and transition-band refinement

September 5, 2026. Completed native diagnostic study. No production operator
or manuscript change. None of the tested modifications qualifies as the
requested generally sharper, non-ringing replacement.

## Main result

The coefficient certificate is unnecessarily restrictive on some currents,
but relaxing it does not materially tighten the measured transition response.
Across 224 two-dimensional plane-wave enlargement cases, removing admission
entirely changes fitted fundamental gain by at most 4.05e-8. Across 35
plane-wave reduction/enlargement cases, the corresponding maximum is 7.22e-8.
The dominant attenuation is already present before admission in this census.
This is an empirical statement about the declared probes, not a theorem
that admission never matters for high-frequency signals.

There is also unused bandwidth headroom inside the existing admission law:
canonical higher-order derivative stencils improve retained carrier amplitude.
However, they give mixed natural-image results and enlarge excursions on
thin structures. The cost gate passes on the measured cameraman task; the
quality requirement does not.

## Controlled construction

`core.py` makes isolated copies of the current native backend in /tmp and
changes only its fibre admission or, in the second experiment, its interior
derivative bank. All arms retain the actual demo's endpoint alignment,
basin reduction, five-current quintic synthesis, two Cartesian orders, and
nodal tensor blend. The unchanged experimental wrapper is tested for
bit-for-bit equality with `distilled_conv_resize` on both reduction and
enlargement. There is no image classifier, fitted model, retained fine-image
side channel, output clipping, or new iterative solver in the construction.

The variants are:

- `conv`: unchanged native admission.
- `raw`: remove admission, diagnostic only.
- `uniform1`: accept an unchanged raw current when one subdivision level
  proves its assigned uniform sign; otherwise use the original admission.
- `word0`, `word1`, `word2`: retain raw current when its Bernstein sign word
  at depth zero, one, or two is a subsequence of that cell's ledger word.
- `ray1`: additionally recover the maximal admissible fraction of the
  remaining proposal along a straight ray in uniform-sign cells.

Each current is a quartic derivative. De Casteljau subdivision changes only
its coordinates and certificate, not its polynomial, degree, or output grid.
The floating FIR mass discrepancy is repaired in the central coefficient
before the experimental certificate, preserving reversal symmetry. Saved
results are binary32 measurements, not exact-arithmetic certificates.

## Certificate derivation and its limit

On a uniform-sign cell let M_d map the original five coefficients to the
concatenated Bernstein coefficients on the 2^d subintervals. Define

\[
 C_d=\{a:\mathbf1^Ta=\delta,\ s M_da\geq0\}.
\]

Subdivision is a positive coordinate map, so C_0 is contained in C_1,
C_1 in C_2, and each lies inside the true functional sign cone. For mixed
signs, the sign word of the polynomial on a subinterval is a subsequence
of its Bernstein coefficient sign word. Requiring the concatenated word to
be a subsequence of the original cell ledger therefore retains its ordered
variation bound. Concatenation across cells retains the factorwise bound.
This does not establish arbitrary-direction variation control for the
spatially blended 2-D operator.

For a feasible baseline b and raw proposal a, a uniform-sign restoration ray
b+t(a-b) has the explicit endpoint

\[
 t_* = \min\left(1,
 \min_{k:s[M_d(a-b)]_k<0}
 \frac{s[M_db]_k}{-s[M_d(a-b)]_k}\right).
\]

This is a fixed finite minimum, not an optimization loop. It preserves mass
and the same refined certificate. It is maximal only on this ray; it is not
the nearest point in the full enlarged cone.

The naive accept-or-original-project construction has a serious stability
defect. For p_e(u)=(u-1/2)^2+e, e approaching zero from above passes the
depth-one certificate. Approaching from below fails and returns the original
coefficient projection, whose limit differs from p_0. Thus the mapping has
a finite jump at an admissibility boundary. At e=+/-1e-7, the represented
input profiles differ by 2e-7, while the outputs differ by 0.00469298.
The ray construction reduces this example's output difference to 1.99e-7;
that is not a general continuity theorem for its mixed-sign branch.

Consequently, finer certification plus an unchanged fallback is useful as
a diagnostic, but is not a satisfactory final admission law. A replacement
must specify stable behavior at the boundary of the enlarged feasible set.

## What happens on cameraman

The standard 512x512 source is reduced to 64x64 and expanded to 512x512.
The study separately changes synthesis, analysis, and both, to avoid
attributing a different coarse raster to improved interpolation.

On the two source-axis profile banks of the 64x64 image, 1,336 of 8,064
cell currents receive a material admission correction (threshold 1e-6).
Depth-two sign-word certification restores 600 of those currents, yet
recovers only 4.80% of the squared coefficient correction. Uniform depth-one
certification restores 449, recovering 3.34%. Most restored corrections are
small. The ray variant recovers 7.76% of the squared coefficient correction.
These are coefficient diagnostics, not signal-domain error improvements.

| Full 512 -> 64 -> 512 method | MSE | MSE improvement | Maximum excursion beyond source range |
|---|---:|---:|---:|
| Existing CONV | 0.00472576877 | reference | 0.0101242 |
| No admission, diagnostic | 0.00472769388 | -0.04074% | 0.0308761 |
| Uniform depth-one certificate | 0.00472536295 | 0.00859% | 0.0101244 |
| Depth-two sign-word certificate | 0.00472571057 | 0.00123% | 0.00997361 |
| Depth-one ray restoration | 0.00472554505 | 0.00473% | 0.00997232 |
| Sixth-order derivative bank, original admission | 0.00471532 (rounded) | 0.22118% | 0.0154435 |
| Eighth-order derivative bank, original admission | 0.00471337 (rounded) | 0.26236% | 0.0188213 |

The image plates show effectively unchanged detail for certificate-only
refinement. More retained raw current is not the same as recovered detail.

## Where the measured spectral corner comes from

The unadmitted interior half-sample kernel follows directly from the existing
five-current bank and quintic integration:

\[
 h_{1/2}=\frac1{256}(3,-25,150,150,-25,3).
\]

Its phase-corrected amplitude at angular frequency w is

\[
 H_{1/2}(w)=
 \frac{300\cos(w/2)-50\cos(3w/2)+6\cos(5w/2)}{256}.
\]

Thus a substantial part of the roll-off is already an explicit property of
the linear proposal. A downstream admission which leaves feasible proposals
unchanged cannot tighten an already admitted carrier. Merely enlarging its
feasible set cannot change this fact. This does not rule out a different
reconstruction or a different proposal within the same admissibility law.

For the axial 257 -> 33 -> 257 test (exact 8x sample-spacing change), the
retained fundamental amplitudes are:

| Frequency / coarse axial Nyquist | CONV | Admission removed |
|---|---:|---:|
| 0.25 | 0.973995 | 0.973995 |
| 0.50 | 0.878699 | 0.878699 |
| 0.70 | 0.710596 | 0.710596 |
| 0.85 | 0.526529 | 0.526529 |
| 0.95 | 0.421171 | 0.421171 |
| 1.05 | 0.209785 | 0.209785 |
| 1.20 | 0.132378 | 0.132378 |

The response is a measured fundamental gain for this signal family; CONV
is nonlinear and does not have a universal LTI transfer function. The above-
Nyquist probes measure alias/stopband behavior, not recoverable information.
The full census also includes 80 crossed-wave, amplitude-modulated, chirp,
and edge-plus-texture cases, where admission is active but the certificate
changes do not yield a substantial aggregate improvement.

## Testing unused admissible bandwidth

For radius r, use the unique centered derivative stencils exact through
degree 2r. For k>0 their coefficients are

\[
 d_k=\frac{(-1)^{k+1}\binom{2r}{r-k}}{k\binom{2r}{r}},
 \quad d_{-k}=-d_k,\quad
 q_k=q_{-k}=2d_k/k,\quad q_0=-2\sum_{k>0}q_k.
\]

The existing r=2 proposal uses six source taps after joining the two endpoint
jets. Radii three and four use eight and ten taps, respectively, while
retaining exactly five currents. Constants, endpoint mass, and polynomial
derivative moments are verified with rational arithmetic. No parameter is
fitted to the images or spectral probes. The outer closure remains the
existing fourth-order/second-order construction where the larger stencil
does not fit; no global C1/C2 claim is introduced.

At 0.7 axial source Nyquist in the declared enlargement probe, fitted gain
increases from 0.87584 to 0.90954 and 0.93301. At 0.85 it increases from
0.73642 to 0.76340 and 0.78603. Original admission and refined admission give
essentially the same carrier responses. The higher-order proposal is
therefore already admitted: insufficient admissible bandwidth is not the
obstruction in those cases.

Natural-image eighth-order results relative to CONV: camera +0.262%,
text -0.326%, brick -0.608%, coins +0.915%, grass +0.030%, moon +0.993%
MSE improvement. This is not a uniform improvement. Pure sampled steps
remain without range excursion on the tested orientations/phases, but the
worst thin-strip excursion grows from 0.27009 to 0.33902. That excursion
need not add an extremum and must not be conflated with the ordered-current
sign-count theorem. The stronger non-ringing image requirement remains unmet.

## Cost and validation

All native timing runs use the M4 Mini, identical compiler flags and actual
two-order synthesis, including reduction, geometry, allocations, and output.
Runs alternate method order after warmup. Four-worker medians use 21 paired
repetitions; one-worker medians use 31. No compilation time is included.

| Variant | Four-worker ratio | One-worker ratio |
|---|---:|---:|
| Uniform depth-one | 0.962x | 0.958x |
| Sign-word depth-one | 0.873x | 0.864x |
| Sign-word depth-two | 1.174x | 1.225x |
| Depth-one ray | 0.907x | 0.906x |
| Sixth-order bank, original admission | 1.002x | 1.000x |
| Eighth-order bank, original admission | 1.003x | 0.998x |

Some certificate variants avoid the projection on already feasible cells,
which can repay their extra certificate work. This is not a guaranteed
speedup or a cost bound on every size/content. Timings meet the requested
ceiling on the measured task; quality and boundary stability prevent promotion.

Seven tests pass on the Mini: actual-demo equivalence; exact subdivision;
stationary-inflection retention; random fibre mass and sampled functional
topology; line variation and cardinality; exact refined-jet moments; and
native refined-jet agreement with the independent rational bank.

## Interpretation

The useful distinction is between an overly small feasible set and a
proposal which does not approach the useful part of that set. This study
finds both, but the latter dominates the measured carrier transition.
There is also a separate weakness: preserving extrema count does not fix
their amplitude or phase, so a sharper proposal can pass admission while
making thin structures less satisfactory.

A stronger joint value/current law could address that weakness, but it has
not been constructed here. It would need to retain the demonstrated spectral
headroom while controlling the distribution of variation, and remain stable
at the admissibility boundary. Neither deeper certification alone nor wider
derivative stencils establish that operator. These are measured exclusions
and a precise remaining target, not a replacement claim.

## Reproduction and artifacts

Run on the Mini through m4build with CONV_NATIVE_THREADS=4 and
OPENBLAS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1:

```
python3 -m unittest experiments.conv_admission_band.test_refinement
python3 -m experiments.conv_admission_band.run_study --out /tmp/conv_admission_band_full --repeats 21
python3 -m experiments.conv_admission_band.run_jet_headroom --out /tmp/conv_jet_headroom
```

With CONV_NATIVE_THREADS=1:

```
python3 -m experiments.conv_admission_band.boundary_and_timing --out /tmp/conv_admission_boundary.json
```

Copy each /tmp result immediately into
`output/support_geometry/conv_admission_band/`, then render locally with
`.venv-jpeg/bin/python -m experiments.conv_admission_band.report`.
Saved JSON contains per-case records and all timing repetitions; NPZ contains
unclipped image arrays. PNGs use a common display range only.
