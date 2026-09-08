# CONV-scale direct joint modes

**Rejected replacement direction (September 5, 2026).** The subsequent
cameraman 4x reduction/expansion test did not activate this mode catalog.
Its fallback did not improve the actual native two-order CONV demo.
The cost results below compare against a different, single-order Python
baseline; they do not establish the requested budget against the demo.
Keep this implementation as an archived experiment, not an accepted
general 2-D replacement. See the cameraman addendum below and
`CONV_HISTORY_AND_8X_CHALLENGE.md`.

September 5, 2026. Implemented in `experiments.conv_budget_modes.resize`, also
exported as `conv_budget_resize`. This is a complete fixed-integer-scale
resampling entry point: admit a scalar characteristic field when directly
witnessed; otherwise use the existing `convstar_resize` result unchanged.
Multichannel arrays use existing CONV. No solver, optimization iteration,
learned parameter, ground-truth coefficient, or destination-error selection
is used by the operator.

## Outcome and scope

The fixed 576-case end-to-end gate passed the user's 1.5x cost ceiling on the
M4 Mini with the same Python/NumPy CONV backend. Median cost was 1.0062x;
maximum was 1.4643x (rounded upward), on a rejected 33² disk at scale 2.
535/576 cases were within 1.2x. These are paired median measurements,
not a universal execution-time guarantee or a prediction against a different
native implementation. The candidate computes its geometry on every call:
there is no cached-mode discount or exclusion of rejection/fallback work.

The gate contains 32 fields at each of 9², 13², 17², 33², 65², and 129²,
each at integer scales 2, 3, and 6. Exactly 294/576 evaluations use the
characteristic field; 282 retain CONV or an exact one-axis CONV shortcut.
These correspond to 98 admitted source fields out of 192, repeated at three
output scales. This is a global monotone straight/radial mode catalog, not
a local atlas for arbitrary image edges. Ordinary textured or color images
retain existing CONV quality. The result meets the tested runtime target
without establishing broad photographic quality gains.

A separate larger-size check tested the eight original fields at 257² and
513², both at scale 3 (15 paired repetitions): all 16 also stayed within
1.5x, with a maximum of 1.342x on a rejected disk. Straight and curved
smooth-edge modes cost 0.806–0.921x CONV; binary straight edges cost
0.315–0.317x. The new mode admitted six of these 16 evaluations. Together
these are 592 tested evaluations, all within the measured 1.5x ceiling.
The larger check is saved in `budget_large.json`.

The four originally successful 17² -> 49² situations are all retained:

| Situation | CONV total ms | New total ms | Cost ratio | CONV interior MSE | New interior MSE | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| diagonal sigmoid | 0.726 | 0.673 | 0.927x | 7.70663e-05 | 3.73147e-14 | 100.000000% |
| curved sigmoid | 0.868 | 0.674 | 0.776x | 0.00169561 | 3.30211e-07 | 99.980526% |
| diagonal step | 0.626 | 0.425 | 0.679x | 0.0157205 | 0.000351708 | 97.762741% |
| disk | 0.730 | 0.445 | 0.610x | 0.0446923 | 0.0147977 | 66.889799% |

Relative to the old expensive characteristic compiler's reduction in
interior MSE, the smallest retained benefit on these four fields is 99.419%
(the diagonal step). The smooth straight/radial profiles improve further.
This measures retained *error reduction*, not equality of all reconstructed
values or equivalence of catalog coverage.

No meaningful interior or whole-image value-MSE regression appeared in the
576 cases with available analytic truth. The regression screen requires
an increase exceeding both 1% and an absolute MSE floor of 1e-18. One
sampled-gradient tradeoff remains: the 13² disk at scale 3 has 7.82% greater
gradient MSE while its interior value MSE falls 16.74%. Its subpixel circle
boundary is not uniquely identified by binary samples; no truth-dependent
rule was added to suppress this case. General accuracy dominance is not
claimed. Unsupported fields are bitwise identical to the existing result.

Across this gate, maximum source-node residual was 5.33e-15, minimum
normalized current certificate margin was -1.12e-14, and minimum normalized
range margin was -1.12e-16. Every admitted raster had zero measured global
range excursion. Existing CONV excursions remain on fallback cases.
The underlying acceptance tolerance remains 2e-11 in normalized units.

## How the expense was removed

1. **Finite geometric witnesses.** A repeated equal-level displacement, a
   boundary flux computed by fixed five-point Boole quadrature, and the
   angular midpoint of adjacent source-value order differences supply a
   small ridge catalog. Every candidate still faces the complete scalar
   phase ordering and source-coincidence checks. A proposal is not a proof.
2. **Binary frontiers.** For a monotone two-level edge, the last low and
   first high source sample on each row are the relevant separating
   witnesses. Choosing the shorter transverse axis limits their pair count
   to O(HW), rather than O((HW)^2). The finite angular intersection proposes
   the direction; full source ordering and current certificates remain.
3. **Support cones by filters.** In an angular frame whose cut lies outside
   a pointed cone, its angular minimum and maximum are its boundaries.
   Two antipodal cuts suffice in exact arithmetic for widths <= pi; two
   additional quarter-turn cuts handle the original near-pi tolerance.
   Min/max filters aggregate the unchanged 5x5 Q1-cell support. This replaces
   per-cell Python construction and sorting of up to 100 rays. Degenerate
   halfplanes without a strong interior witness are treated conservatively
   as lines. No original cone is enlarged.
4. **Vectorized profile and certification.** Phase coincidences are reduced
   in batches. Positive derivative controls and exact geometric phase
   extrema certify the original support ranges and product currents.
   Exactly flat phase ranges impose no spurious direction requirement.
   A sparse range-maximum table is built only if the cheaper global
   derivative-amplitude bound is insufficient.
5. **Exact representation shortcuts.** Redundant flat samples in a binary
   profile collapse to at most four knots, with the same quintic transition.
   One smoothstep expression then evaluates the complete profile. Exactly
   axial images use one existing CONV pass and replication. Biquadratic
   source data returns directly to CONV. Large general query arrays use one
   Horner accumulator instead of a six-coefficient-per-pixel temporary.
6. **Cheap failure.** Coordinatewise order rejects obvious texture before
   attempting a profile. Corner-centered radial candidates must satisfy
   their necessary diagonal reflection before sorting a radial phase.
   At most one complete current certificate is attempted per source;
   failure returns to CONV, rather than repeating expensive admissions.

With P source pixels, K phase knots and Q destinations, proposal/profile
work is O(P log P), support-cone construction is O(P), and the optional
range-maximum table costs O(K log K). General evaluation costs O(Q log K),
with fixed polynomial degree. Binary evaluation is O(Q). These bounds
remove quadratic all-source-pair storage; they do not imply an asymptotic
constant-factor guarantee against CONV for every possible size.

## The additional mathematical improvement: shared scalar curvature

The first fast version inherited the old zero-curvature phase profile. On
well-resolved edges that profile and an approximate direction could be less
accurate than CONV. Higher-order boundary flux improves the direction
proposal. The accepted scalar profile now carries a shared first derivative
m_i and second derivative c_i at each phase node, obtained by directly
differentiating a five-point nonuniform Newton interpolant. This is fixed
polynomial arithmetic, with no Vandermonde system solve.

For interval width h and secant d, its physical derivative Bernstein controls
are

    [m_L,
     m_L + h c_L/4,
     5d - 2m_L - 2m_R + h(c_R-c_L)/4,
     m_R - h c_R/4,
     m_R].

All five must be nonnegative. The old capped-harmonic, zero-curvature
profile is a constructive feasible baseline. Each node's changes in m and c
are one shared entity. Each derivative-control constraint allocates its
baseline margin over the sum of harmful incident responses. A node retains
the minimum incident allocation. Nonnegative responses cannot worsen the
bound, so the resulting profile is monotone by finite arithmetic. If the
complete proposal is already feasible, it passes unchanged. Shared m and c
retain C2 traces across every phase knot. Radial exterior endpoints retain
zero m and c, so constant extensions remain C2.

This is the economical version of the joint-potential idea:

    F = phi(G)
    grad F = phi'(G) grad G
    Hess F = phi''(G) grad G grad G^T + phi'(G) Hess G.

For squared radial phase, the second term carries the required curvature
companion. The construction does not separately fit or limit x, y, and
mixed derivatives. In a smooth unit-speed Eikonal coordinate T, the same
identity reads Hess F = psi'' nn^T + psi' kappa tau tau^T: profile steepness
and contour bending remain coupled. No numerical Eikonal front is added.

## Verification and reproduction

The mathematical suite checks original cone membership, nonuniform
polynomial reproduction, shared C2 traces, monotone derivative controls,
exact binary compression, actual currents against original normals,
cardinality, axial/color/fallback identity, transpose/reflection/value
covariance, and exclusion of solver calls. Together with the previous
characteristic/finite suites it contains 20 tests.

The benchmark warms both methods, alternates execution order, and records
15 times per method per case. Both include construction and output allocation.
BLAS thread settings are identical: VECLIB_MAXIMUM_THREADS=1 and
OPENBLAS_NUM_THREADS=1. The initial broad run encountered another heavily
loaded Python process on the Mini. It is archived for diagnosis; the final
budget gate was started after that process finished. Ordinary scheduling
noise still applies.

    /Users/ultimussecundai/.local/bin/m4build -- env \
      VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python3 -m unittest experiments.test_conv_budget_modes \
      experiments.test_conv_joint_characteristic_modes experiments.test_conv_joint_finite

    /Users/ultimussecundai/.local/bin/m4build -- env \
      VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python3 -m experiments.validate_conv_budget_modes \
      --sizes 9,13,17,33,65,129 --scales 2,3,6 --repeats 15 \
      --out /tmp/conv_budget_gate.json

Copy the JSON immediately to
`output/support_geometry/conv_joint_characteristic_modes/budget_gate.json`.
The summary is `budget_summary.json`; saved matched visual data and figure
are `budget_visuals.npz` and `budget_profiles.png`. Earlier v1/v2/v3 records
are diagnostic history, not the final operator's measurements.

Usage:

    from experiments.conv_budget_modes import conv_budget_resize
    result = conv_budget_resize(source, scale=3)

The core requires finite float-compatible HxW or HxWxC arrays with at least
five source nodes on each spatial axis, and positive integer enlargement.
This preserves the lattice convention output_length=(input_length-1)*scale+1.
It is not a downsampler, arbitrary-size demo-backend replacement, or proof
of a compatible local atlas for arbitrary natural images.

## Cameraman 4x shrink/expand falsification

The subsequent user-requested 512 -> 128 -> 512 cameraman test exposes the
practical limitation: the characteristic compiler rejects the reduced scalar
image immediately, with zero profile attempts. It contributes no geometric
reconstruction. This test does not establish how an admitted characteristic
mode would ring on a natural image; no such mode was admitted.

It also exposes a baseline distinction: `run_image_demo.py` uses
`distilled_conv_resize`, with basin reduction and tensor-weighted two-order
synthesis. The earlier budget tests used single-order `convstar_resize`.
The literal new approach retains that single-order fallback. Both cameraman
arms use the same demo CONV basin reduction. To return exactly to 512 rather
than 509, the probe evaluates the same Python admitted currents at arbitrary
endpoint-aligned positions; its agreement with the existing integer-scale
wrapper was verified to 1.12e-16 on the nested 509 output.

Demo CONV round-trip MSE is 0.00230958157 (26.36467 dB). The new approach's
single-order fallback gives 0.00231032724 (26.36327 dB), slightly worse.
Unclipped output has 59 versus 65 pixels outside [0,1]; this is only a global
overshoot count, not a complete ringing metric. Float64 candidate versus
native float32 single-order CONV differs by at most 1.04e-5, confirming that
the principal comparison is between fallback constructions. If the new
catalog were attached to the demo's own fallback, it would produce exactly
the demo output here, because admission fails.

Consequently the measured synthetic budget success is not evidence of a
better general photographic resizer or a demonstrated 1.5x cost bound
against the demo's native two-order implementation. The images, unclipped
arrays and metrics are in `output/support_geometry/conv_cameraman_roundtrip/`.
Reproduce with `python3 -m experiments.probe_conv_cameraman_roundtrip` on the
Mini, copy /tmp/conv_cameraman_roundtrip immediately, and render locally with
its --plot-only option.
