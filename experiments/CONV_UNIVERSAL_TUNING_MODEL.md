# Universal signal tuning of CONV

## 1. Fixed operator architecture

CONV is the composition

\[
Y\xrightarrow{A_\theta}C
\xrightarrow{P_\theta}a
\xrightarrow{\Pi_{\mathcal C}^{W_\theta}}c
\xrightarrow{S}T,
\]

followed in two dimensions by the fixed convex factor-order blend.  Here

- \(A_\theta\) is conservative scalar analysis;
- \(P_\theta\) is a translation-invariant jet/current proposal;
- \(\Pi_{\mathcal C}^{W_\theta}\) admits that proposal to the same signed,
  endpoint-conservative current fibre \(\mathcal C\);
- \(S\) is the cardinal quintic Bernstein integral.

The parameters are fixed once for the operator.  They are never estimated
from, switched by, or adapted to the presented signal.

## 2. Untunable invariants

The following are definitions or proof obligations, not design controls.

| Object | Reason it is fixed |
|---|---|
| Cardinal endpoint coordinates | Required for interpolation |
| Current sum \(\sum_k c_k=\Delta y_i\) | Endpoint conservation |
| Ordered sign fibre \(\mathcal C\) | Source of the variation theorem |
| Nonnegative factor-order blend | Source of the convex two-order result |
| Constants and affine signals | Reproduction obligations |
| Axis-exchange covariance | No Cartesian direction may be privileged |
| Quintic integration once five currents are retained | Representation identity |
| Boundary closure order | Must meet the declared endpoint accuracy |
| Floating tolerances | Numerical realization of exact comparisons |

Changing any of these changes the theorem or the representation rather than
tuning the signal response.

## 3. Valid universal parameters

### 3.1 Conservative ownership exponent

Let \(a_j\) be the target lattice's affine hat coordinates and define

\[
q_j^{(p)}(x)=\frac{a_j(x)^p}{\sum_k a_k(x)^p},\qquad p>0.
\]

Then

\[
q_j^{(p)}\geq0,\qquad \sum_jq_j^{(p)}=1,
\]

and weighted coarse conservation is exact.  The range is

\[
0<p\leq\infty.
\]

Distinguished values are \(p=1\), ordinary affine ownership; \(p=2\), the
lowest nontrivial symmetric power; and \(p=\infty\), hard Voronoi basins.
This parameter trades smooth ownership and lower generated energy against
passband retention.  It does not change with content.

### 3.2 Factor-order contrast exponent

For measured current energies \(G_{xx},G_{yy}\), define

\[
\beta_q=
\frac{G_{yy}^{q}}{G_{xx}^{q}+G_{yy}^{q}},\qquad q>0.
\]

This obeys

\[
\beta_q(G_{xx},G_{yy})
=1-\beta_q(G_{yy},G_{xx}),
\]

preserves \(\beta_q=1/2\) for isotropic current, and remains in \([0,1]\).
The limits \(q\downarrow0\) and \(q\uparrow\infty\) give symmetric averaging
and hard factor-order selection respectively.  The value \(q=1\) is uniquely
linear in measured quadratic energy.

Across four source/target lattices and the fixed sixteen-case geometric
population, \(q=8\) reduced aggregate geometric-mean MSE by only \(0.0176\%\)
relative to \(q=1\), while the finest lattice preferred \(q<1\).  The measured
effect is not yet stable enough to displace the principled value \(q=1\).

### 3.3 Proposal support and nullspace coefficients

The ordered-current theorem constrains the admitted current \(c\), not the raw
proposal \(a\).  Therefore the translation-invariant proposal may be designed
as a fixed FIR bank

\[
a_{i,k}=\sum_{r=-R_-}^{R_+}h_{k,r}y_{i+r}
\]

provided its coefficients satisfy exact linear constraints for

1. constant annihilation;
2. affine reproduction;
3. the selected polynomial moment order;
4. reversal symmetry;
5. \(\sum_k a_{i,k}=y_{i+1}-y_i\) when exact proposal conservation is desired.

For a fixed support, these conditions define an affine coefficient space

\[
h=h_0+Nz.
\]

Only the nullspace coordinate \(z\) is tunable.  It can be chosen once by a
minimax frequency-response calculation.  Increasing integer support \(R\)
adds work and spectral degrees of freedom but does not weaken topology because
the same projection follows it.

This is the principal route to a better detail/ringing/work point.  It changes
the destination proposed to the theorem, not the theorem.

### 3.4 Projection fidelity metric

The current implementation uses Euclidean projection

\[
\min_{c\in\mathcal C}\frac12\|c-a\|_2^2.
\]

Any fixed SPD metric preserves uniqueness and membership in the identical
admissible fibre:

\[
\min_{c\in\mathcal C}\frac12(c-a)^TW(c-a),\qquad W\succ0.
\]

Reversal symmetry requires the objective to be invariant on the fibre tangent
space \(\mathbf1^Td=0\).  A sufficient full-space condition is

\[
JWJ=W,
\]

where \(J\) reverses the five currents.  Arbitrary diagonal slot weights are
not the most principled form.  The representation itself supplies a canonical
metric.  If

\[
e_c(u)-e_a(u)=\sum_{k=0}^4(c_k-a_k)\tau_k(u),
\]

where \(\tau_k\) are the quintic tail bases, then profile-domain fidelity is

\[
W_0[k,l]=\int_0^1\tau_k(u)\tau_l(u)\,du.
\]

The left-anchored tail Gram is not literally centrosymmetric on all of
\(\mathbb R^5\), but its quadratic form is reversal-invariant on
\(\mathbf1^Td=0\).  That is exactly the subspace occupied by the difference
between two endpoint-conservative current vectors.

A universal Sobolev family is

\[
W_\alpha=W_0+\alpha W_1,
\qquad
W_1[k,l]=\int_0^1\tau_k'(u)\tau_l'(u)\,du,
\qquad \alpha\geq0.
\]

The dimensionless \(\alpha\) trades profile fidelity against current fidelity.
Every value retains the same endpoint and ordered-variation theorem.  The
distinguished value \(\alpha=0\) is exact integrated squared reconstruction
error, rather than an invented slot weighting.

### 3.5 Fixed spectral sharpening before admission

A response sharpening map such as

\[
s_r(\lambda)=
\frac{\lambda^r}{\lambda^r+(1-\lambda)^r},\qquad r\geq1,
\]

is symmetric and fixed, but is a valid CONV parameter only if its proposed
current is subsequently admitted through \(\Pi_{\mathcal C}\).  Spectral
boundedness alone does not imply pointwise variation diminution.  This
parameter remains provisional until that composition is written as an exact
current proposal and its reproduction constraints are verified.

## 4. Parameters that are not valid

The following are excluded from the model:

- parameters selected from edge strength, image class, or local texture;
- learned or inferred per-image support;
- thresholds that switch between interpolators;
- a cutoff estimated from the values being interpolated;
- numerical tolerances treated as visual controls;
- arbitrary anisotropy multipliers unsupported by the measured quadratic
  current tensor.

## 5. Selection problem

Universal tuning is a constrained operator-design problem.  Let

\[
\theta=(p,q,R,z,\alpha,r).
\]

The admissible set is defined analytically by conservation, reproduction,
symmetry, positivity where required, and fixed work.  On that set, report the
nondominated vector

\[
\mathcal E(\theta)=
\left(
E_{\rm pass},E_{\rm stop},G_{+},G_{-},E_{\rm generated},
E_{\rm cycle},C_{\rm work}
\right).
\]

No photographic preference selects \(\theta\).  The final coefficients should
be frozen from a declared signal battery containing impulses, steps, boxes,
sinusoids over frequency and phase, chirps, diagonal and curved interfaces,
crossings, and repeated matched cycles.

## 6. Present evidence

The powered-partition sweep establishes a real monotone control: increasing
\(p\) improves half-cutoff retention and slightly tightens transition while
increasing generated energy.  Hard basins are the sharpness endpoint of that
family.

The factor-order exponent has much smaller leverage.  Its apparent optimum
moves with lattice density, so the current evidence favors keeping \(q=1\).

The untested high-value variables are therefore the moment-constrained FIR
proposal nullspace and the representation-induced projection metric
\(W_\alpha\).  Both can improve fidelity while leaving the admitted set—and
therefore the central topology theorem—unchanged.

The completed joint sweep resolves these two variables.  Nonzero proposal
nullspace optima point in opposite directions for 1-D spectral signals and
2-D transported geometry, so the supported universal value is \(z=0\).
For that proposal, \(\alpha=0\), the exact profile-\(L^2\) metric \(W_0\),
slightly improves aggregate error in both populations while leaving worst
1-D error unchanged.  These are the current universal values.
