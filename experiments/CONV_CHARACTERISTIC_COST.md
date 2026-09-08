# Characteristic-mode cost and Eikonal geometry

Follow-up: `CONV_BUDGET_REPLACEMENT.md` documents the implemented replacement
that removes the all-pairs compiler and passes the measured 1.5x CONV budget.
The expensive compiler timings below are historical diagnostics.

Measured September 5, 2026 on the M4 Mini, same Python/NumPy backend for both methods. These are research implementation timings, not native-kernel predictions. Source generation and output coordinate meshes are excluded. One warmup; median of seven constructions and 31 sampling/CONV calls. All six straight/radial source cases admitted; all three crossing cases rejected. Source cardinality checked below 1e-10. The earlier shorter run is retained as cost_initial.json; final results are cost.json. Host inspection found no competing heavy compute process, though timing noise remains.

| Source | Geometry | Output | CONV total ms | Mode construction ms | Mode sampling ms | First-use / CONV |
|---|---|---|---:|---:|---:|---:|
| 9² | straight | 25² | 0.379 | 10.168 | 0.069 | 27.0× |
| 9² | straight | 49² | 0.435 | 10.168 | 0.195 | 23.8× |
| 9² | radial | 25² | 0.453 | 9.735 | 0.065 | 21.6× |
| 9² | radial | 49² | 0.534 | 9.735 | 0.172 | 18.6× |
| 17² | straight | 49² | 0.843 | 73.611 | 0.239 | 87.6× |
| 17² | straight | 97² | 1.087 | 73.611 | 0.854 | 68.5× |
| 17² | radial | 49² | 0.855 | 45.689 | 0.187 | 53.6× |
| 17² | radial | 97² | 1.157 | 45.689 | 0.683 | 40.1× |
| 33² | straight | 97² | 2.034 | 320.475 | 0.812 | 158.0× |
| 33² | straight | 193² | 2.875 | 320.475 | 3.088 | 112.6× |
| 33² | radial | 97² | 1.929 | 223.071 | 0.729 | 116.0× |
| 33² | radial | 193² | 3.061 | 223.071 | 2.839 | 73.8× |

Rejection costs at 9², 17², and 33² are respectively 4.29, 20.71, and 116.76 ms. A hypothetical try-mode-then-CONV pipeline must ADD the CONV time; this benchmark does not implement that fallback. Accuracy coverage in the preceding experiment remains 4/8 matched and 12/16 validation: timing only admitted edges cannot establish universal cost or utility.

## Where the work goes

Let P=HW source samples, Q=H'W' destinations, K phase knots, and C tested tie-direction candidates. CONV uses fixed-size current banks and admission per Cartesian interval, followed by two separable synthesis passes: O(P+HW'+Q) for the x-then-y path. The current characteristic recognizer allocates P(P-1)/2 pairs, sorts their order angles, and may check each candidate against all strict orders. Thus its pair stage has O(P² log P + CP²) time and O(P²) memory. It is not a scalable full-image recognizer. Pair counts are 3,240 / 41,616 / 592,416 at the measured sizes; 65² would already require 8,923,200 pairs.

Fixed-footprint support cones are also built in Python cell loops, with repeated construction in recognition and certification. The ridge certificate scans intersecting phase pieces for derivative bounds, so that implementation can add O(PK) worst-case work. Radial recognition first pays for a failed ridge attempt. These costs are properties of the prototype, not requirements of the representation.

After construction, the stored field has O(K) controls plus a direction or center. Each query computes G, searches its phase interval (O(log K)), and evaluates one quintic. Current code uses six explicit Bernstein terms and array temporaries. At 33² -> 193² its straight-mode evaluation alone is 3.088 ms versus CONV's complete 2.875 ms. Therefore even cached evaluation is not uniformly faster. Construction amortization only applies to genuinely new queries of the same source; repeated identical output can simply be cached for either method.

## Joint derivatives and Eikonal behavior

For F=phi(G), exact chain differentiation gives

    grad F = phi'(G) grad G
    Hess F = phi''(G) grad G grad G^T + phi'(G) Hess G.

A Euclidean unit-speed distance coordinate T obeys |grad T|=1 away from singularities. With n=grad T and F=psi(T), differentiating that identity gives Hess(T)n=0. In two dimensions, with tangent tau and signed curvature kappa defined by Hess(T)=kappa tau tau^T,

    grad F = psi'(T)n
    Hess F = psi''(T)nn^T + psi'(T)kappa tau tau^T.

Consequently normal steepness and tangential bending are coupled, with no independent tangential-curvature knob. This is the precise relation to jointly admissible derivative modes. A straight phase G=n·x is already Eikonal. Our radial phase G=polarity*r² is not unit speed: |grad G|=2r. Away from the center one can rewrite the SAME field using T=r and psi(T)=phi(polarity*T²). This is a representation identity, not invariance of the profile-construction algorithm under nonlinear coordinate changes. Squared radius also avoids the distance coordinate's singularity at the center.

When phi'=0, current vanishes regardless of grad G. The successful zero-amplitude certificate is therefore naturally a statement about the product current; restricting geometry alone can reject a valid mode. Eikonal normalization fixes coordinate speed, not signal steepness or ringing. Those require the profile's finite admission and the original joint support certificates.

The repository's Eikonal-owner proposal supplies a different but complementary object: regions and causal transport paths under a metric. Its arrival coordinate is not automatically an image intensity phase. A separate compatible phase/covector must be retained. Curved support and collisions can guide where a characteristic chart applies, but do not alone establish cardinality, range preservation, or shared traces at chart boundaries. A numerical Eikonal front would be additional construction work and is not part of this solver-free mode experiment.

A plausible next construction is a finite catalog of directly witnessed phase geometries, with scalar order/profile checks and shared original support certificates. Given a candidate phase, scalar sorting is O(P log P); testing a fixed catalog avoids enumerating all pairs for direction discovery. This is a proposed route, not a measured speedup or a proof of general coverage. Any local atlas must additionally supply a finite joint trace rule; independently blending valid potentials introduces derivative terms from spatially varying weights and can break admission.

For background on the distance-coordinate Eikonal identity, see Per-Olof Persson's Berkeley level-set notes: https://persson.berkeley.edu/math228b/html/levelset_notes.html. The chain-rule and curvature identities above follow directly from the stated equations.

Reproduce with:

    /Users/ultimussecundai/.local/bin/m4build -- python3 -m experiments.benchmark_conv_characteristic_cost

Copy /tmp/conv_characteristic_cost.json immediately to output/support_geometry/conv_joint_characteristic_modes/cost.json.
