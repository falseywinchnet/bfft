# Larger exact CONV fusion proofs

September 5, 2026. This note develops the shared-linear-backbone identity into
coefficient-level fusion rules and a finite certificate for reusing admission
across phases. All equalities below concern the real-arithmetic CONV operator.
The new rational oracle verifies algebra; it is not a production backend or a
speed result. The previously promoted rolling ledger remains the native change.

## 1. Operator and scope

Let Y be the source raster, with axes x and y. Let K_d be the existing linear
raw-current bank, D_d the interval difference, A_d the existing ordered-ledger
and signed-fibre admission, and S_d the linear synthesis map

    S_d(Y,c) = I_d Y + B_d c.

Within a unit source interval, I_d Y is its left anchor and B_d c is
sum_{k=0}^4 tau_k(u)c_k, where

    tau_k(u) = sum_{j=k+1}^5 binom(5,j) u^j (1-u)^(5-j).

Admission preserves sum c_k = D_d Y. The raw bank has this identity as well.
Use the existing boundary closure consistently in every occurrence of K.
Define

    T_d Y = S_d(Y,A_d(K_d Y,D_d Y)),
    R_d Y = S_d(Y,K_d Y),
    d_d(Y) = A_d(K_d Y,D_d Y) - K_d Y,
    E_d(Y) = B_d d_d(Y).

These definitions do not introduce a new projection, solver, image model,
threshold, or admissibility relaxation. The target is the existing two orders
F_xy = T_y(T_x Y), F_yx = T_x(T_y Y), blended with the existing source-defined
Q1 coordinate beta. A preceding basin reduction is outside these synthesis
identities; it supplies Y unchanged.

## 2. Mass-null correction has exactly four coordinates

**Theorem.** Each interval correction has the unique representation

    E(u) = sum_{k=0}^3 [tau_k(u)-tau_4(u)] d_k
         = u(1-u) q(u),       degree q <= 3.

**Proof.** Both admitted and raw currents have mass delta, so
`d_4 = -sum_{k=0}^3 d_k`. Substitute this into B d. Every difference of tails
vanishes at u=0 and u=1 and has degree at most five, so it is divisible by
u(1-u). The current-to-polynomial derivative map is injective because the
quartic Bernstein basis is independent. Restricting this map to the
four-dimensional mass-null subspace preserves injectivity. Thus this is an
exact four-coordinate representation, not a low-rank approximation. QED.

The four quotients, in ascending power order, are

    (5, -5,  5, 0),
    (0, 10, -10, 5),
    (0,  0, 10, -5),
    (0,  0,  0, 5).

[These coefficients are checked by the executable tests.]

This removes one redundant correction channel. It does not automatically make
a raw-plus-correction evaluation cheaper than the existing single admitted
five-current evaluation: the raw reconstruction must still be paid for.
Precomputed difference-tail weights need four correction multiply-adds instead
of five, wherever computing a correction separately is already justified.

## 3. Lift transverse linear analysis before phase expansion

**Theorem.** For every transverse linear map L_y, including K_y and D_y,

    L_y S_x(Y,c_x) = I_x L_y Y + B_x L_y c_x.

**Proof.** For a fixed source x-cell and phase u, I_x and the five scalar
weights tau_k(u) act only in x. L_y acts only in y. Distribute L_y across the
sum and commute those scalar weights with it. This is valid for any fixed
c_x, including currents produced by nonlinear source admission. It does not
require A_x itself to be linear. QED.

Consequently the complete input to second-stage admission can be formed from
coarse coefficient fields:

    a_y(u) = I_x K_y Y + B_x K_y c_x,
    delta_y(u) = I_x D_y Y + B_x D_y c_x.

Both are degree-at-most-five polynomial fields in horizontal phase u. This
is an executable fusion of linear analysis and first synthesis: filter the
polynomial coefficients before evaluating phases. The nonlinear second
admission still receives exactly its original a_y(u), delta_y(u).

**What cannot be commuted.** In general

    A_y(a_y(u),delta_y(u))

cannot be reconstructed by applying A_y separately to polynomial coefficients.
Its signs, ledger decisions, and projection faces can change with u. The next
result states exactly when one execution can be reused.

## 4. A fixed admission path compiles to a polynomial

A path records the following existing discrete decisions for one full line:

1. Signs of coarse secants, including identically zero runs and their inherited
   signs; hence all coarse transition knots.
2. Signs of the raw currents used in contradiction costs.
3. The greedy selected boundary at each transition, in causal order, with its
   original permitted interval and first-minimum tie convention.
4. The active coordinates of each signed-fibre projection.

**Theorem.** If these decisions are valid throughout a phase interval, the
admitted current in every second-stage cell is a polynomial of degree at most
five on that interval. No further admission is needed for its phases.

**Proof.** Items 1–3 fix every ledger sign sigma_k. For a nonempty active set
J, the unique Euclidean projection is

    lambda(u) = [sum_{k in J} a_k(u) - delta(u)] / |J|,
    c_k(u) = a_k(u)-lambda(u),  k in J,
    c_k(u) = 0,                k not in J.

It is valid precisely when

    sigma_k [a_k(u)-lambda(u)] >= 0,  k in J,
    sigma_k [a_k(u)-lambda(u)] <= 0,  k not in J.

These are the feasibility and KKT inequalities of the original fibre. Strict
convexity makes the admitted vector unique, even if zero coordinates permit
more than one active-set description. The formula is linear in a and delta,
with division only by the fixed integer |J|. Thus it preserves degree five.
The all-zero projection is a separate degenerate face; it is identically zero
where its original KKT conditions hold. QED.

This is partial evaluation of the existing finite admission program. It is
not a catalogue of straight/radial/edge image models and does not select a new
reconstruction by a quality score.

### A sufficient finite certificate, without finding polynomial roots

Map the proposed phase interval affinely to [0,1]. For a scalar polynomial
p(t)=sum a_k t^k of degree n, its degree-n Bernstein coefficients are

    b_j = sum_{k=0}^j a_k binom(j,k)/binom(n,k).

All b_j >= 0 certify p >= 0 everywhere; all b_j > 0 certify p > 0 everywhere.
These conditions are sufficient, not necessary. They do not change CONV's
admissible cone: they only establish whether an existing path can be reused.

Check every inequality defining the path:

- A nonzero witnessed sign s requires s p > 0. A zero branch is certified only
  for an identically zero polynomial. The current oracle conservatively
  requires strict signs even at interval endpoints.
- Once raw signs are fixed, each ledger cost is a sum of squared quintics,
  hence degree at most ten.
- If the midpoint path selected boundary b*, certify C(b)-C(b*) > 0 for each
  earlier eligible boundary and >= 0 for each later one. This preserves the
  first-minimum tie rule. Eligibility depends on the previous selected
  boundary; induction through the original causal scan certifies the whole
  ledger, not independent unconstrained local choices.
- Certify the degree-five KKT inequalities above for every fibre.

One ordinary path execution at an interval midpoint supplies the witness.
The certificate either establishes it on the whole interval or declines to
compile. A declined certificate must fall back to ordinary phase evaluation;
it is not rejection of the input and not evidence of a mathematical violation.
A fixed, bounded subdivision schedule is possible. No root isolation,
iterative optimization, or global solve is needed. The proof oracle uses the
existing finite breakpoint formula to select a projection face at its midpoint.

The oracle intentionally declines the empty-active-set interval case, except
for an identically flat full line. This loses coverage, not correctness.
It also records raw signs outside transition windows; that makes it more
conservative than a production implementation would need to be.

### Finiteness is not a useful small bound by itself

The exact-real phase behavior has a finite polynomial partition: a finite
program with finitely many polynomial sign/cost tests yields finitely many
semialgebraic intervals and isolated points in one variable. Identically zero
polynomials are handled by the prescribed ties. This observation does not
bound the partition count by a small constant independent of line length.
It does not justify computing all roots or pretending compilation is free.
The inherited-sign scan remains a whole-line operation across long zero runs.

## 4a. Fuse a certified projection face directly into reconstruction

**Theorem.** Fix a valid nonempty active face J at the query. Define

    bar_tau_J(u) = sum_{k in J} tau_k(u) / |J|.

Projection and synthesis together equal

    F(u) = y_i + bar_tau_J(u) delta
                + sum_{k in J} [tau_k(u)-bar_tau_J(u)] a_k.

**Proof.** Substitute c_k=a_k-lambda on J into the original synthesis and
collect the coefficient of lambda. Then substitute the finite-face formula
for lambda from Theorem 4. QED.

Because a=K y and delta=D y, this is the single effective nodal stencil

    W_J(u) = e_i + bar_tau_J(u) D
                 + sum_{k in J} [tau_k(u)-bar_tau_J(u)] K_k.

Its interior support is the same six source nodes as the raw bank. Boundary
stencils use the original closure. No current array or explicit threshold is
needed to evaluate this expression once the face is known. There are only
31 nonempty subsets of five current coordinates. These are the actual
projection faces, not added image models; a runtime lookup would use the
already established face and would not score all thirty-one alternatives.
The empty face gives F=y_i and must be established separately.

For |J|=1 the sum involving raw currents disappears exactly:

    F(u) = y_i + tau_j(u) (y_{i+1}-y_i).

For a general face, the coefficients of a_k sum to zero, so the raw-dependent
term has at most |J|-1 independent coordinates. In particular,

    sum_{k in J}(tau_k-bar_tau) a_k
      = sum_{k in J, k != r}(tau_k-bar_tau)(a_k-a_r)

for any r in J. This gives an exact face-dependent reduction in what must be
transported after certification. The full-five face reduces to raw synthesis
because sum a_k=delta, as it should.

This theorem matters specifically after phase reuse is proved. Discovering
the face still needs the original admission or a valid certificate. Computing
a fresh face for every phase and then replacing a five-current terminal dot
product with a six-node dot product is not an automatic saving.

## 5. The order difference is zero on every source grid line

**Theorem.** For arbitrary data and phases,

    F_xy(i,y) = F_yx(i,y),
    F_xy(x,j) = F_yx(x,j)

at source knots i or j.

**Proof.** At a source x-knot, T_x reproduces the original column exactly.
Doing x first therefore leaves that column as the input to T_y. Doing x last
copies the same column of T_y Y by cardinality. The y argument is identical.
The proof uses cardinality and the identical one-axis rule, not linearity,
separability of the source, or inactive admission. QED.

Therefore the commutator Delta=F_yx-F_xy lives inside original source cells.
The common edges can be computed once; beta is irrelevant where Delta=0.
For a target lattice, however, count actual coincidences, not nominal scale.
With endpoint-aligned n -> m sampling there are gcd(n-1,m-1)+1 coincident
coordinates. For 64 -> 512 that is 8, and the two-dimensional fraction on
source grid lines is

    1 - (1-8/512)^2 = 3.1005859375%.

For truly nested refinement m=s(n-1)+1 the large-image fraction tends to
2/s-1/s^2 (23.4375% at s=8). These are different sampling situations.
Neither fraction is a complete-pipeline speed estimate.

## 6. Conditional fusion of both orders into one cell polynomial

**Theorem.** On a phase rectangle where the second admission paths of both
orders remain valid, F_xy and F_yx are bicquintics: degree at most five in each
coordinate. With the existing bilinear beta, their blend is of bidegree at
most (6,6).

**Proof.** First-stage currents are fixed from Y. Theorem 4 gives second-stage
currents of degree at most five in the first coordinate. Second synthesis is
quintic in the other coordinate. For the reverse order exchange coordinates.
Multiplication by the bilinear beta raises each degree by at most one. QED.

**Stronger whole-cell corollary.** If both orders admit one such polynomial
representation across the entire original source cell, then

    Delta(u,v) = u(1-u)v(1-v) Q(u,v),   bidegree Q <= (3,3).

**Proof.** Theorem 5 makes the bicquintic difference zero on u=0, u=1, v=0,
and v=1. Polynomial division successively removes the four linear factors.
The quotient has at most sixteen coefficients. QED.

The two-order blend can then be expressed as one order plus an interior
bubble times beta Q; beta Q has at most twenty-five coefficients. An
alternative common-edge Coons patch yields the same bubble factorization for
each order's departure from that patch.

**Restriction that matters:** a smaller certified phase rectangle need not
touch all four original cell edges. Its polynomial extension need not vanish
there. Do not use the sixteen-coefficient corollary on arbitrary pieces.
Globally, Delta vanishes on the cell boundary, but it need not be one
bicquintic. Boundary zeros alone do not even establish a bounded quotient by
the product bubble without additional regularity. The work here does not
claim global C2 compatibility or a new joint-potential reconstruction.

## 7. Exact backbone sharing and the remaining dependencies

The Cartesian raw operators commute, including their separable endpoint
closures. Expanding T_d=R_d+E_d gives

    F_xy = R_xy Y + R_y E_x(Y) + E_y(T_x Y),
    F_yx = R_xy Y + R_x E_y(Y) + E_x(T_y Y).

Thus one raw tensor reconstruction suffices algebraically. The exact defect
is the difference of the four transported/terminal correction terms. A zero
correction does not contribute outside its linear synthesis support; this can
justify skipping actually certified zero terms.

But the terminal corrections must be computed or certified from the actual
first-stage results. Source correction masks alone do not establish them.
A sparse correction implementation also must not replace the inherited-sign
scan with a fixed-radius rule; that is a different dependency question.
A tolerance-based declaration that small corrections are zero changes the
operator and is not part of this proof.

## 8. Cost conditions: what would make this a reduction

There are three distinct opportunities, with different break-even tests.

**Four-coordinate residual transport.** Useful if a raw backbone is already
shared or retained. Its dense cost is not lower merely because four is less
than five: original synthesis evaluates admitted currents directly, whereas
backbone-plus-correction has two contributions. Sparse or reused correction
transport must pay for the extra representation.

**Phase-path compilation.** Let m be the number of phases served by one
certificate, L the ordinary per-phase raw-analysis/ledger/projection work,
P the compiled polynomial-current evaluation work, and C all coefficient
construction, witness execution, and certificate work. Holding final spatial
synthesis common, compilation reduces work exactly when

    C + m P < m L, or m > C/(L-P), provided L > P.

Failed certificates add overhead and no saving. For a mixture of attempted
blocks the actual condition is that total savings on accepted blocks exceed
all certification attempts, including failures. A bounded attempt budget can
limit overhead; it cannot manufacture a speed gain.

A degree-five input has six coefficient fields. Moving the raw bank before
phase expansion consequently filters six fields once, instead of one field
per phase. Evaluating five resulting quintics costs up to twenty-five
multiply-adds per phase before final synthesis. Ledger cost certificates
involve degree-ten polynomial arithmetic. At only seven inserted phases per
nested 8x cell, this is not a free reduction. Reusing one certificate across
more phases, repeated queries, or zoom frames is more favorable, provided the
source and its admitted state are unchanged. No timing win is asserted here.

**Shared cell polynomial.** A bicquintic has thirty-six coefficients; the
blended bi-degree-six representation can have forty-nine. Dense two-variable
Horner evaluation at every pixel may exceed the existing two five-current
terminal evaluations. Tensor sweeps can reuse one-dimensional contractions
across rows, but must include coefficient assembly and certification in their
cost. The whole-cell bubble corollary is a representation theorem, not a
proof that sixteen stored numbers outperform the existing dataflow at 8x.

The most direct allocation-only fusion remains streaming terminal blend:
produce an order's terminal row/tile and consume it in the final blend while
its data are hot. That can reduce full-sized temporary storage and traffic,
but does not eliminate either admission or reconstruction and requires a
separate native scheduling benchmark.

## 9. Real arithmetic versus native floating arithmetic

The polynomial oracle uses exact rational arithmetic, including the original
jet closures, finite ledger choices, and signed-fibre projection. It proves
facts about the intended real operator for its rational inputs.

Native float32 raw currents do not necessarily sum to the secant exactly;
reconstructing a fifth correction from the other four can therefore change
bits. Filtering coefficients before evaluating phases also reassociates sums,
and R_y R_x need not equal R_x R_y bit for bit. Rounded branch comparisons
can disagree near ties or active-set boundaries. A native compiler requires
roundoff-enclosed certificates and an explicitly agreed numerical-equivalence
contract, or retention of legacy calculations wherever exact-bit behavior is
required. Rational certificates are not automatically certificates for all
rounded intermediates of the native execution.

The earlier rolling-ledger promotion preserved legacy arithmetic near cost
ties. This larger proof does not extend that bitwise guarantee by assertion.

## 10. Executable evidence and reproduction

`polynomial_proof.py` implements exact polynomial construction, the original
finite admission path, Bernstein path checks, and fixed interval substitution.
`test_polynomial_proof.py` checks the mass-null bubble, transverse analysis,
nontrivial certified projection reuse, rejection of an invalid midpoint-only
path, interval substitution, and exact agreement of both nonlinear orders on
source grid lines while retaining a nonzero interior order difference.

`study_polynomial_proof.py` additionally compares thirty exact-rational source
profiles against the existing independent float64 CONV* implementation. Its
small two-dimensional certificate probe is explicitly a feasibility diagnostic,
not an image-quality benchmark or whole-image acceptance estimate.

Run on the Mini:

```
python3 -m unittest experiments.conv_exact_fusion.test_polynomial_proof -v
python3 -m experiments.conv_exact_fusion.study_polynomial_proof --out /tmp/conv_polynomial_proof.json
```

Use the repository m4build wrapper and copy the JSON into
`output/support_geometry/conv_exact_fusion/` immediately after the run.

### Measured proof-oracle results

All seven exact-rational tests passed on the Mini, including the direct
projection/synthesis stencil and its reduced current-difference form. Thirty independently
checked source profiles agreed with the existing float64 implementation to
maximum absolute current error 1.7763568394002505e-15.

For each 7x7 scene the probe used source x-cells 1 and 3 in each factor order,
first as four whole-cell phase intervals, then as thirty-two predetermined
one-eighth intervals. A certificate covers the complete second-stage line.

| Source | Whole intervals certified | Eighth intervals certified | Certified eighths with nonzero admission correction |
|---|---:|---:|---:|
| Nonseparable integer data | 0 / 4 | 5 / 32 | 2 |
| Smooth nonseparable polynomial | 4 / 4 | 32 / 32 | 0 |
| Cameraman coarse crop [24:31,24:31] | 0 / 4 | 9 / 32 | 4 |

Every accepted interval was checked at four additional rational phases,
including its endpoints, against fresh exact admission. The cameraman input
comes from the retained actual 512 -> 64 basin reduction, but the 7x7 crop
uses its own boundary closure. These figures are small diagnostic coverage
measurements, not whole-image coverage or 8x pipeline timing. Whole-line
certification is deliberately stricter than certification of a dependency
segment. The smooth case proves inexpensive state structure exists; the
nonzero-correction columns establish that reuse is not limited to raw-linear
behavior. At approximately 8x expansion, an eighth-cell phase interval usually contains
only about one output phase. Certifying that small an interval therefore
provides little or no admission amortization at the user's target scale,
even when the certificate succeeds. The natural-data whole-cell attempts
all declined in this tiny probe. Together with certificate overhead, that
prevents promotion of this prototype as a generally faster replacement.
The useful result is the exact fusion law and its explicit reuse condition;
a cheap certificate covering larger intervals or smaller dependency segments
remains necessary for this route to help the 8x workload.
