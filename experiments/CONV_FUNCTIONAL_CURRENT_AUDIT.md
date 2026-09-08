# CONV audit: admit the continuous current

2026-09-04. Research result, with a numerical prototype and an analytic
counterexample. The existing manuscript and production operator are unchanged.

Follow-up: `CONV_COMPATIBLE_CURRENT_FINDINGS.md` records the subsequent global
proposal study and numerical hardening of the shared projection oracle. The
measurements and solver tolerances below describe this first archived run;
the current implementation uses the updated unit-mass solve documented there.

**Finding:** CONV's five signed Bernstein increments are a useful compilation
coordinate, but their signs are an unnecessarily restrictive definition of
admissible slope. Projecting the derivative function onto its actual positive
cone removes a provable reconstruction defect at stationary inflections.
This gives a concrete basis-independent extension of the current representation.
It does not establish general superiority to Lanczos, or even to CONV on all
monotone transitions.

## Audit scope and what the paper already knows

Audited the composed TeX source `output/pdf/conv_paper_composed.tex`, including
the conservative lifting construction, CONV* compilation, and direct joint
warp appendix, against `experiments/convstar.py`, `conv_paper_operator.py`,
`conv_distilled_core.py`, the projection-metric certificate, and the existing
universal-tuning and high-frequency audit work. This is a focused mathematical
and implementation audit, not a rerun of every historical figure or timing.

The manuscript already makes several essential qualifications correctly:

* Lines 134–157 acknowledge exact quartic positivity and distinguish the
  Bernstein fibre from the full monotone-quintic region. Discovering that
  larger region is not a new result of this audit.
* Lines 881–896 state that independently admitted cells need not be C1 or C2.
  However, interpreting their mismatch as a witnessed differential change is
  too strong: the cubic example below creates mismatch in perfectly smooth,
  exactly reconstructed source geometry solely through admission.
* Lines 995–1027 condition fifth-order interior value accuracy on inactive
  admission. The stationary-inflection example falls outside that hypothesis;
  it does not refute the theorem. It exposes a consequential missing case.
* Lines 791–856 explicitly include the spatial blend derivative
  `(1-beta) A' + beta B' + beta' (B-A)`. Convex factor blending does not
  automatically inherit the factorwise variation theorem.
* Lines 1349–1399 correctly distinguish retained-detail inversion from
  zero-detail reconstruction. Exact inversion is not recovery of discarded
  information.
* Lines 2692–2750 of the warp appendix again use positive Bernstein
  coefficients as a sufficient certificate for a continuous directional
  derivative. The same representational restriction occurs in 2-D.

The old `CONV_UNIVERSAL_TUNING_MODEL.md`, section 3.4, already derives the
profile and derivative Gram metrics. The advance tested here is the larger
**functional admissible set**, together with its accuracy consequence;
replacing the Euclidean norm alone is explicitly an ablation.

## An exact slope defect

On one unit cell, take

\[
f(u)=\frac{(u-1/2)^3}{3}+\varepsilon(u-1/2),
\qquad f'(u)=(u-1/2)^2+\varepsilon.
\]

For positive epsilon this is strictly increasing. Interior CONV two-jets
reproduce this cubic exactly. Its five raw current increments are

\[
a=\frac15\left(\frac14+\varepsilon,\ \varepsilon,
 -\frac1{12}+\varepsilon,\ \varepsilon,\ \frac14+\varepsilon\right).
\]

For `epsilon < 1/12`, the middle coefficient is negative even though the
derivative is nonnegative everywhere. Because all sample differences are
positive, CONV applies its positive-coefficient mass fibre.

At epsilon = 1/1000, the cell mass is 253/3000. The original Euclidean
projection returns

\[
c=(253/6000,0,0,0,253/6000).
\]

Its midpoint slope is

\[
5\sum_jc_jB_j^4(1/2)=253/9600
=0.0263541666667,
\]

whereas the exact slope is 0.001. The projection increases it by a factor
of **26.354**, while preserving mass, cardinality, and monotonicity. This is
an error in the distribution of slope that the sign-variation theorem cannot
detect. It is not ringing under the paper's definition.

There is also an order consequence. Set epsilon to zero and sample the fixed
function `f(x)=x^3/3` with the zero halfway between two nodes separated by h.
The raw cubic is exact. Its current coefficients and the nonzero projection
correction both scale as h^3. The projected value error on the straddling
cell therefore has a nonzero O(h^3) leading term. The mathematical operation
is homogeneous, so this is an exact scaling argument, not a regression fit.

The 64-phase numerical check gives:

| h | Original CONV maximum sampled cell error | Functional admission error |
|---:|---:|---:|
| 1 | 4.6899542e-3 | 1.3877788e-17 |
| 1/2 | 5.8624428e-4 | 1.7347235e-18 |
| 1/4 | 7.3280535e-5 | 2.1684043e-19 |
| 1/8 | 9.1600668e-6 | 2.7105054e-20 |
| 1/16 | 1.1450084e-6 | 3.3881318e-21 |

CONV's error decreases by eight per halving. Functional admission preserves
the exact cubic, leaving only floating-point error. The sampled maximum is
not asserted to be the exact continuous maximum.

## The representation and the replacement constraint

Represent the cell by its anchored derivative:

\[
p_i(u)=5\sum_{j=0}^4c_{ij}B_j^4(u),\qquad
U_i(u)=y_i+\int_0^u p_i(t)\,dt.
\]

For a uniform ledger sign s, replace the coefficient fibre by

\[
\mathcal K_{s,\delta}=
\{c:\ \mathbf1^Tc=\delta,\quad s\,p_c(u)\ge0
\text{ for every }u\in[0,1]\}.
\]

This is a nonempty closed convex set whenever s*delta is nonnegative. A
constant current is feasible. The coefficient fibre is a strict subset.
Both sets conserve the same endpoint current, but the new one depends on the
polynomial function rather than its chosen basis coordinates.

Use the derivative fidelity objective

\[
\min_{c\in\mathcal K_{s,\delta}}
\frac12\int_0^1|p_c(u)-p_a(u)|^2\,du
=\frac12(c-a)^TW_1(c-a),
\]

\[
(W_1)_{ij}=\frac{25{4\choose i}{4\choose j}}
{9{8\choose i+j}}.
\]

The Gram matrix is positive definite, so the minimizer is unique. Changing
the polynomial basis changes both the coefficients and Gram matrix and leaves
this optimization problem unchanged. On a physical cell of length h,
the squared physical-derivative error has the additional factor 1/h.

Consequences in exact arithmetic:

1. Endpoint interpolation and integrated cell current are exact.
2. A uniform-sign cell stays monotone and within its endpoint range.
3. Any already monotone raw quartic derivative passes unchanged, including
   the entire cubic-shoulder example.
4. For fixed sign and mass, projection is firmly nonexpansive in the W1 norm.
   If the true current belongs to this polynomial cone, projection cannot
   increase its W1 error relative to the raw proposal.
5. Integrated value error is controlled by derivative error through
   `|integral_0^u (p_c-p_a)| <= sqrt(u) ||p_c-p_a||_L2`.
6. The optimum has no greater **distance to the raw proposal in W1** than
   the optimum over the smaller coefficient fibre. This is not a theorem
   about improvement against unknown ground truth.

The implemented extension changes only uniform-sign cells. Mixed-sign cells
retain the original ledger and original projection exactly. A modified cell
contributes only its assigned sign to the derivative sequence; unchanged
mixed cells retain their original total-positivity bound. Concatenating these
sequences gives the original global factorwise upper bound on sign changes.
The proof uses function signs in the modified cells; their coefficient signs
can no longer be used as the paper's intermediate certificate.

This construction is not globally C1. It removes admission-induced jumps
when a smooth raw profile becomes admissible, but does not impose shared
endpoint derivatives in general. Shared first and second traces can be added
as linear constraints to a joint optimization, at the cost of coupling cells.
That extension was not implemented or benchmarked here.

## Exact cone representation and practical compilation

A quartic p is nonnegative on [0,1] exactly when it can be represented as

\[
p(u)=z_2(u)^TQz_2(u)+u(1-u)z_1(u)^TRz_1(u),
\quad Q\succeq0,\ R\succeq0,
\]

where `z_2=(1,u,u^2)` and `z_1=(1,u)`. Coefficient matching is linear.
Thus the desired cone has a small semidefinite representation. This is
classical interval polynomial positivity, not a new positivity theorem.
See the [MIT Drake sum-of-squares tutorial](https://drake.mit.edu/tutorials/sum_of_squares_optimization.html).

The prototype instead solves a five-variable convex quadratic problem with
constraint exchange. After each solve it finds all real stationary points
of the quartic by solving its cubic derivative, checks those and both
endpoints, and adds the most violated location. It does not merely test a
fixed raster of slope samples.

This remains a floating-point research oracle, not an interval-certified
positivity solver or a bounded-work replacement for CONV*. It normalizes
cell amplitudes, targets a 2e-11 normalized positivity tolerance, and removes
small remaining negativity by a mass-preserving mix with the positive
constant derivative. SLSQP line-search failures are accepted only after a
separate primal feasibility and KKT-residual check. Exchange failure raises
an error. Reflection tests use a 2e-5 coefficient tolerance because nearby
active roots make the numerical solution less accurate than the exact
algebraic invariance. The exact cubic counterexample bypasses optimization.

A cheaper certificate hierarchy could use Bernstein subdivision or degree
elevation on the same polynomial and the same functional Gram objective.
Strictly positive polynomials eventually admit such certificates; interior
zeros require special care. No universal finite certificate depth should be
claimed for all boundary cases. Changing certificate resolution must not
change the represented polynomial or its error metric.

## M4 experiment and the negative results

The complete run used 25 source nodes and 64-fold interpolation. It includes
44 analytic cases: 12 cubic shoulders (four phases, three epsilon values),
16 tanh sigmoids (four phases and four widths), 12 sinusoids (four phases,
frequencies .08/.22/.4 cycles per sample), and four step phases. All methods
receive the same point samples. Analytic truth is used only for evaluation.

Six methods are retained in the JSON: original CONV, coefficient cone with
W1, functional cone with Euclidean coefficient error, functional cone with
W1, raw two-jet, and normalized finite Lanczos-3. Error is evaluated away
from source boundaries. Value and derivative MSE are normalized by the
squared true value range on the declared evaluation region. The table uses
arithmetic means within each family; it is not a natural-image population.

| Family | CONV value NMSE | Functional/W1 value NMSE | Lanczos-3 value NMSE |
|---|---:|---:|---:|
| Cubic shoulder | 1.66515e-8 | 9.82255e-34 | 6.06015e-5 |
| Sigmoid | 7.84095e-4 | 7.87092e-4 | 9.19583e-4 |
| Sine | 4.60435e-3 | 4.60435e-3 | 2.67683e-3 |
| Step | 3.03326e-2 | 3.03326e-2 | 3.31641e-2 |

* All 12 shoulder cases improve. Changing only the coefficient metric reduces
  shoulder slope NMSE from 4.11941e-7 to 2.75985e-7; enlarging the cone reduces
  it to floating-point error, with either objective. This isolates the main
  mechanism as the admissible set, not metric tuning.
* Maximum interior derivative jump over the shoulder cases falls from
  0.0416667 to 8.88e-15. This removes an artifact caused by admission.
* On sigmoids, functional/W1 value NMSE is **0.382% worse** and slope NMSE is
  **0.649% worse** than CONV. Value results are two wins, eight ties, six
  losses using an absolute NMSE tie threshold of 1e-15. A more faithful raw
  slope can be a less faithful true slope when the raw jet is biased.
* The sine and step results are unchanged. These examples do not require the
  newly available positive-polynomial freedom. Lanczos is materially better
  on the sine set. Improving its passband is a separate proposal problem.
* The tested CONV and functional outputs have no additional sampled derivative
  sign transitions. Both have zero range excess on steps. Raw two-jets and
  Lanczos have step range excesses of 0.101566 and 0.117795, respectively.
  Numerical transition counts support, rather than replace, the argument above.
* Different step positions inside the same source interval produce identical
  point samples. Their subpixel positions cannot be inferred exactly from
  these observations alone. No sharp-edge recovery claim follows from the
  positive-cone construction.

## Toward one representation for values, slopes, scales, and warps

The coherent state is an anchor and an integrable current, with each
observation declared as a functional of that state:

\[
U(x)=b+\int_{x_0}^{x}d\mu,\qquad y_a=L_a[U].
\]

For smooth cells, `dmu=p(x) dx`. Point observations are evaluations; pixel
observations are normalized aperture integrals; coarse differences are
integrated currents; retained detail is the residual of the corresponding
observation functional. These are different measurements of one potential.
Cell averages should not silently become point values when changing scale.
The paper already builds matched residual lifting; a full unified model
would impose the declared observation functionals on the same continuous
state throughout, instead of regarding that predictor linkage alone as
measurement equivalence.

In multiple dimensions, use a scalar potential U and its exact differential
dU. Admitting unrelated derivative components independently would lose
integrability. The natural joint law is

\[
L_a[U]=y_a,\qquad \ell_r^T\nabla U(x)\ge0,
\]

with shared potential traces and an integrated gradient-error objective.
For an inverse warp Phi, `d(U composed with Phi)=D Phi^T (dU composed with
Phi)`. This is the existing appendix's transport identity applied to the
functional current itself. The opportunity is to relax conservative
coefficient certificates while preserving the pointwise law and potential
compatibility. Bivariate positivity requires additional work; the exact
small univariate semidefinite formula does not transfer automatically.

Allowing singular current measures would also accommodate actual jumps.
Their support and identifiability require an explicit acquisition model.
Finite-rate-of-innovation sampling supplies established precedent for
reconstructing non-bandlimited piecewise polynomials from suitable measured
moments; it does not justify recovering arbitrary subpixel jumps from the
point samples used in this experiment. See
[Vetterli, Marziliano and Blu (2002)](https://bigwww.epfl.ch/publications/vetterli0201.html).

The defensible result is therefore **an exact repair of a basis-induced
slope/accuracy defect and a common functional representation to extend**.
A universal beyond-Lanczos transform, an improved 2-D image method, and a
fixed-cost implementation remain unestablished. The next discriminating
experiment should couple a better raw proposal to this same cone and test
both high-frequency sinusoids and under-resolved monotone transitions;
otherwise it would repeat the shoulder success without addressing the
measured limitation.

## Reproduction and artifacts

Run from the authoritative repository:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.test_conv_functional_current
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_functional_current --out /tmp/conv_functional_current
```

Copy `/tmp/conv_functional_current/results.json` back immediately using the
host returned by `m4host`. Plot locally with the existing GUI environment:

```sh
.venv-jpeg/bin/python -m experiments.conv_functional_current --plot-only \
  --out output/support_geometry/conv_functional_current
```

Four test methods pass on the M4 system Python, including 40 randomized
positive-cone projection cases, integrated Gram verification, reflection,
cardinality, conserved mass, unchanged mixed cells, affine reproduction,
and the exact shoulder counterexample. The 44-case sweep and five-scale
convergence probe also completed successfully.

Files: `experiments/conv_functional_current.py`,
`experiments/test_conv_functional_current.py`,
`output/support_geometry/conv_functional_current/results.json`, and
`output/support_geometry/conv_functional_current/profiles.png`.

Additional prior art already acknowledged by CONV:
[Ulrich and Watson, quartic positivity](https://epubs.siam.org/doi/10.1137/0915035),
[Lux et al., MQSI](https://doi.org/10.1145/3570157).
These prevent treating monotone-quintic construction itself as a new discovery.
