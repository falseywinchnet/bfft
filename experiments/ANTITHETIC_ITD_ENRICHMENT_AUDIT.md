# Antithetic CONV--ITD enrichment audit

This audit distinguishes universal identities, conditional differential
statements, implementation invariants, and empirical observations.  Numerical
tests corroborate implementations; they are not substituted for proofs of
universal statements.

## Exact results that survive

### Antithetic cancellation

For every map `B` for which both evaluations exist, the expression

\[
 A_\epsilon(f,q)=\frac{B(f+\epsilon q)+B(f-\epsilon q)}2
\]

is an even function of \(\epsilon\).  Every odd power therefore cancels
algebraically.  The test suite verifies the binomial identity with exact
rational arithmetic through degree nine.

If `B` is four times differentiable at `f`, Taylor's theorem gives

\[
 A_\epsilon(f,q)
 =B(f)+\frac{\epsilon^2}{2}D^2B_f[q,q]+O(\epsilon^4).
\]

This is conditional on differentiability and a fixed active set.  It is not a
Taylor statement at an extrema-creation, extrema-annihilation, or current-face
boundary.  On a smooth random `tanh(Ax)` operator, the measured quadratic
order was `1.9999851`, and the residual after subtracting the exact quadratic
coefficient had order `3.9998734` on the M4 Mini.

### Null calibration

Define

\[
 \Delta_\epsilon(f,q)=A_\epsilon(f,q)-B(f)
 -A_\epsilon(0,q)+B(0).
\]

Then \(\Delta_\epsilon(0,q)=0\) identically.  If `B` is affine,
\(\Delta_\epsilon(f,q)=0\) identically.  Both were checked with exact rational
arithmetic where applicable and with the actual CONV--ITD baseline.  All 12
randomized CONV--ITD trials had exactly zero measured null response.

For the explicit nonlinear operator \(B(x)=x^2\), antithetic averaging of a
noise-only input left a coherent response of norm `0.0320920549`; null
calibration reduced it to exactly zero.  This proves that an even noise-only
ghost can occur and that the stated control removes that particular response.
It does not prove that every signal-conditioned ghost is removed.

### Fixed-clock second-order frame

For a fixed extrema clock \(\tau\), let \(H_\tau=I-K_\tau\), let
\(U=\operatorname{range}H_\tau\), and let \(r=\dim U\).  An orthonormal basis
`Q` of `U` obeys

\[
 Q^TQ=I,\qquad QQ^T=P_U,\qquad
 \frac1rQQ^T=\frac1rP_U.
\]

This is an identity.  Across 40 randomized irregular clocks, the largest
spectral orthogonality error was `2.02e-15`.

Among positive semidefinite covariances supported on an \(r\)-dimensional
space with trace one,

\[
 \lambda_{\min}(C)\leq\frac{\operatorname{tr}C}{r}=\frac1r.
\]

Equality requires all \(r\) eigenvalues to equal \(1/r\), hence
\(C=P_U/r\).  The normalized projector is therefore minimax only for the
specific second-order objective of maximizing the least probed variance under
fixed trace.  It is not a globally optimal nonlinear-noise distribution.

A single probe has rank-one covariance and cannot cover a detail space of
rank greater than one.

### Known covariance

For an externally supplied covariance `R`, the restricted covariance
\(P_URP_U\) is positive semidefinite and can be factored directly.  The audit's
reconstruction error was `2.88e-14` in spectral norm.

### Decomposition closure

If the detail is defined by \(d=f-b\), then \(b+d=f\) algebraically.  Across
the randomized CONV--ITD battery, the largest floating-point closure error was
`1.11e-16`.

## Statements that required repair

### CONV lifting does not preserve frame isotropy

Lifting an orthonormal knot-space frame through admitted irregular CONV did
not preserve its Gram matrix.  Across 40 irregular clocks, the raw lifted Gram
condition number ranged from `1.48` to `4762.90`, with median `38.95`.

Consequently, isotropy can only be claimed in sample space after forming all
lifts and rewhitening their span.  After SVD rewhitening, the largest measured
orthogonality error was `2.67e-15`.

### Second-order tightness is not finite-amplitude orientation neutrality

Two orthonormal frames in two dimensions have the same covariance.  For the
quartic directional response \((e_1^Tq)^4\), the coordinate frame gives mean
`0.5`, while its 45-degree rotation gives `0.25`.  The response differs by a
factor of two despite identical second moments.

Thus a tight frame removes directional preference only from the quadratic
term.  Fourth and higher terms can remain frame dependent.

A reference bounded cubature was constructed by mixing the cross polytope
with all normalized hypercube vertices.  In rank \(r\), give the cross
polytope total weight \(2/(r+2)\) and the hypercube total weight \(r/(r+2)\).
It reproduces the uniform-sphere second and fourth tensors exactly:

\[
 E[q_iq_j]=\frac{\delta_{ij}}r,
\]

\[
 E[q_iq_jq_kq_l]
 =\frac{\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}
 +\delta_{il}\delta_{jk}}{r(r+2)}.
\]

At rank seven, the 142-point rule had second-moment spectral error
`3.91e-16` and fourth-moment maximum error `5.55e-17`.  It is a degree-four
reference, not a complete solution: it is exponential in rank and does not
control degree six and above.

### The sampled-current boundary is not an admitted-extrema boundary

The value

\[
 \epsilon_{\Delta}
 =\min_{\Delta q_i\ne0}\frac{|\Delta f_i|}{|\Delta q_i|}
\]

is exactly the first amplitude at which one of the two antithetic sampled
first-difference sequences can reach zero.  It does not locate a root event of
the admitted piecewise-quartic CONV derivative.

In the 12 actual CONV--ITD stress trials, four showed no admitted-extrema count
change through \(16\epsilon_\Delta\).  Among the remaining eight, the first
observed count change ranged from `1.01` to `8` times
\(\epsilon_\Delta\).  The earlier topology interpretation is rejected.

### Current-fibre projection does not by itself prove an amplitude range

The exact KKT projection was stress-tested on 40 random multichannel banks.
Its largest endpoint-sum error was `4.44e-15`, its signed-halfspace violation
was zero, and its largest idempotence error was `1.78e-15`.

Those are precisely the constraints it enforces.  They do not alone imply
that every intermediate Bernstein control lies between the two endpoints.  A
feasible current sequence

\[
 (1,1,-1,-1,1),\qquad \sum_i c_i=1,
\]

has controls \((0,1,2,1,0,1)\), exceeding the endpoint interval \([0,1]\).
Therefore the phrase "projection restores the range guarantee" is false
without an additional, explicitly defined amplitude polytope.  Projection
does restore the stated signed-current fibre and endpoint conservation.

## Empirical CONV--ITD behavior

Below the sampled-current boundary, the null-calibrated correction of the
actual nonlinear CONV--ITD baseline was approximately quadratic.  Across 12
random multicomponent signals, the median fitted order was `2.000043`; one
trial reached `2.447`, demonstrating active-set sensitivity rather than a
uniform Taylor theorem.

The preferred perturbation amplitude cannot be recovered from symmetry.  In
the quartic response assay, external noise scales
`[0.03, 0.10, 0.25, 0.70]` produced corresponding optimal sigma-point
amplitudes `[0.03, 0.10, 0.25, 0.70]`.  An external noise model or an explicit
optimization criterion is necessary.

## Present conclusion

The defensible construction is narrower than the initial proposal:

1. Complementary signs exactly remove odd perturbation orders.
2. Null calibration exactly removes the noise-only response at zero input.
3. A rewhitened lifted frame is second-order minimax on a fixed detail
   subspace.
4. Finite-amplitude orientation neutrality requires higher-order cubature;
   the tight frame alone is insufficient.
5. No universal finite amplitude or universally optimal noise law has been
   established.
6. Any CONV guarantee must be attached to the exact post-enrichment admissible
   set.  The signed-current fibre supplies endpoint conservation and its sign
   law, not an amplitude range by itself.

The experiments therefore support continued investigation, but not yet the
insertion of a canonical enrichment operator into CONV or ITD.

## Reproduction

```sh
python3 -m unittest experiments.test_antithetic_itd_enrichment
python3 -m experiments.run_antithetic_itd_enrichment_audit
python3 -m experiments.run_antithetic_itd_enrichment_stress --trials 40
```

The preserved M4 Mini outputs are
`experiments/antithetic_itd_enrichment_audit_m4.json` and
`experiments/antithetic_itd_enrichment_stress_m4.json`.
