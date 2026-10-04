# Slow implies flat: Dirichlet control of Sinkhorn curvature

September 30, 2026. Follows `FINDINGS.md` (tropical hypothesis refuted) and
Astra's `experiments/entropic_transport_closure/FORMAL_ANALYSIS.md`, whose
notation (P, Q, J=QP, c-metric, log-moment remainders) is used throughout.

## 1. Chart picture (classical; stated for orientation)

Modulo gauge, Sinkhorn F has a globally attracting fixed point y* with
spectrum of J* in [0,1) (Poincare domain). By Poincare / Poincare-Dulac,
F is analytically conjugate near y* to Lambda (no resonances) or to a
finite polynomial (not linear) normal form. Extension phi = Lambda^{-k} o
phi o F^k needs Lambda invertible (positive K can be rank deficient, giving
zero eigenvalues) and global injectivity is NOT implied by global
attraction; both are unproved assumptions here. Hence the evolving local rule is a fixed rule
seen through a moving chart: J(y) = Dphi(F y)^{-1} Lambda Dphi(y). Astra's
2x2 Mobius lift is the case where phi is itself a cross-ratio. Frozen-J
Krylov is a first-order chart anchored at y_k instead of y*.

The concern was near-resonance: the chart coefficient of the slowest mode is
beta/(lam-lam^2) ~ beta/delta (forward chart phi o F = Lambda phi), delta = 1-lam. If beta stayed O(1) as
delta -> 0, the linear (Koenigs) domain would shrink like delta and slow
Sinkhorn would be near-parabolic (Fatou coordinates, sublinear phase).
The lemma below shows this cannot happen.

## 2. Second-order term (exact)

From log(R e^h) = Rh + Var_R(h)/2 + O(h^3) applied twice,

    F(y+u) = F(y) + J u + B(u,u) + O(u^3),
    B(u,u) = 1/2 [ Q Var_P(u) - Var_Q(P u) ].                  (B)

Checked against second finite differences at random non-fixed states
(`test_dirichlet.py`) and at y* (relative error <= 7e-5, results/normal_form_v2.json).

## 3. Lemma (Dirichlet control of curvature). At EVERY state y, for all u, w:

    <u,(I-J)u>_c   = E_a[ Var_P(u) ],                              (TV1)
    <u,(J-J^2)u>_c = E_c[ Var_Q(P u) ],                            (TV2)
    |<w, B(u,u)>_c| <= ||w||_inf <u,(I-J)u>_c.                     (D)

Proof. Pi = diag(a)P has column sums c, so <u,u>_c = E_a[P(u^2)] and
<u,Ju>_c = <Pu,Pu>_a (Astra eq. 2); subtracting gives TV1. The same with
Q, whose weighting has row mass c, and QPu gives TV2. Pair (B) with w:
<w,Q Var_P u>_c = <Pw, Var_P u>_a (adjointness), bounded by
||Pw||_inf E_a Var_P u <= ||w||_inf <u,(I-J)u>_c; and |<w,Var_Q(Pu)>_c|
<= ||w||_inf <u,(J-J^2)u>_c <= ||w||_inf <u,(I-J)u>_c since 0 <= J <= I.
Half the sum gives (D). QED.

Interpretation: the curvature in a direction is bounded by that direction's
Dirichlet energy, i.e. by how slowly the iteration moves along it. Astra's
osc(d)^2/8 bound is the energy-blind version.

Measured: max ratio lhs/rhs of (D) over 60 random states/directions 0.25.

## 4. Corollaries

(a) Gap-uniform quadratic scale (not a proved analytic radius). For a c-normalized eigenvector v, lam<1:
|beta| = |<v,B(v,v)>_c| <= 1/2(||Pv||_inf + lam ||v||_inf) delta, so
a* = delta/|beta| >= 1/||v||_inf, independent of delta. The quadratic
self-coefficient cannot create a near-parabolic slow mode. This is a
second-order statement only: it does not give an analytic convergence
radius, and it does not exclude Fatou-like transients from higher-order
or mixed terms.
Measured (n=128, 2 seeds, eps .03-.001, gap .43 -> .0094): a* in 6.4-285,
always >= 1/||v||_inf (0.17-0.63). beta is 15-100x below its bound
(the two variance terms cancel), so (D) is valid but not sharp.

(b) Nonlinearity per unit motion is gap-independent. In a spectral window
with <u,(I-J)u>_c <= delta_w ||u||_c^2, one step moves ~delta_w||u|| and
has second-order defect <= ||w||_inf delta_w ||u||^2: ratio O(||u||),
with no 1/delta. More sweeps do not make the nonlinearity matter more.

(c) Why frozen-anchor Krylov fails in the mid phase (explains FINDINGS).
The anchor Jacobian's slow Rayleigh quotient is lam + 2 beta a + O(a^2):
per step a relative rate error O(delta a ||v||_inf), tiny. But a jump is
only worth taking over H ~ 1/delta, where (1 + 2 beta a/lam)^H ~ exp(O(a)).
The error is O(1) relative whenever a ||v||_inf = O(1). In contrast the
linear model AT y* has error that decays with amplitude and does not
compound. Measured (eps=.001, k=10, H=1024): actual -5.3e-4, linear-at-y*
-1.07e-3, scalar quadratic -7.9e-4, frozen-anchor -1.2e-6.
So the obstacle is not changing geometry per se but freezing it at the
wrong point and exponentiating the discrepancy.

## 5. What transport of transport should carry (proposal, not yet tested)

By (c) the evolution of the rule that matters is, to first order, the drift
of the slow Ritz values with amplitude, d theta/d a = 2 beta, and (D) bounds
it by the gap. Both a (through residual size / (1-theta)) and theta are
observable along the ordinary trajectory, so theta(a) can be extrapolated
to a = 0 and the jump taken with the scalar quadratic slow model
a -> lam* a + beta a^2 instead of the frozen anchor rate. This uses no
fixed-point oracle. Its acquisition cost and multi-mode accuracy (the
mid phase has other-mode amplitude comparable to the slow one) are open.

## 6. Consequences and conjectures

* Consequence: for Sinkhorn most sweeps (all but ~log(a0 ||v||)/delta)
  lie in a regime where dynamics are linear to O(amplitude); we CONJECTURE
  acceleration there is limited by linear theory (Chebyshev/CG-type
  ~ 1/sqrt(delta)); no such ceiling is proved, and it would need
  assumptions on the acquisition model.
  Nonlinear transport can at best shorten the transient.
* Conjecture (EM / Richardson-Lucy): the missing-information principle
  gives the same total-variance structure (I-J = observed/complete
  information), so an analogue of (D) should hold. Not yet proved.
* Conjecture (Meyer): the coupled disk projections have rotating branches
  without a total-variance identity; curvature is then not controlled by
  the gap. This would explain why Krylov transport transfers to Sinkhorn
  and fails on Meyer. Testable by measuring beta/delta on Meyer.

Reproduce (Mini; copy JSON back immediately):

    m4build -- env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 sh -c \
      'python3 -m unittest experiments.tropical_transport.test_dirichlet -v && \
       python3 -m experiments.tropical_transport.normal_form --seeds 2 --out /tmp/normal_form_v2.json'

## 7. Mixed coefficients (joint with Codex, September 30)

Correction: my first hand expansion of the symmetric case was wrong.
Codex's counterexample K=(1/600)[[281,101,218],[101,281,218],[218,218,164]]
(uniform marginals, Ku=.3u, Kw=-.09w, lam_w = lam_u^2) has resonant
coefficient exactly 27/5000 (Fraction arithmetic and finite differences).

General marginals at y*: SVD of diag(a)^-1/2 Pi diag(c)^-1/2 with s >= 0,
right modes v (c-orthonormal), left modes u=Pv/s (a-orthonormal):

    2 <v_k,B(v_i,v_j)>_c = (s_k - s_i s_j)^2 T_c + 2 s_k s_i s_j (T_c - T_a),
    T_c = E_c[v_k v_i v_j],   T_a = E_a[u_k u_i u_j].                (M)

Verified to 2e-12 (`test_mixed_split.py`). Chart coefficient for
phi(u)=u+h(u,u), phi o F = Lambda phi, so h = B/(lam_k - lam_i lam_j)
(sign corrected per Codex; scalar check h = beta/(lam(1-lam))):

    h_kij = (s_k - s_i s_j)/(2(s_k + s_i s_j)) T_c
            + s_k s_i s_j (T_c - T_a)/(s_k^2 - s_i^2 s_j^2).

The first term is bounded by |T_c|/2 for all s>0: resonance-free.
Only left/right mode asymmetry T_c - T_a can meet a small divisor.
Symmetric PSD (u=v): cancels identically. Symmetric with negative
eigenvalues: u=-v on that mode, T_a=-T_c, Codex's counterexample.
Median |T_c-T_a|/max(|T_c|,|T_a|) = 0.21 on random 2-D problems:
asymmetry is generic under unequal marginals. Measured max |h| over the
10 slowest modes (mixed.py): <= 23 on 2-D points, <= 7 on a 1-D grid.
These are coefficientwise; no dimension-free operator bound is proved.

## 8. Single-mode acquisition test: falsified, with diagnosis

`acquisition.py` (results/acquisition_v1.json): at anchors K*{.05,.10,.15},
Arnoldi (depth 12, state c-metric) gives the slowest Ritz pair; signed
coordinate a_hat=<z,r>_c/(theta-1), Kato-Temple error rho^2/(theta1-theta2),
fast leakage of r outside z. Fit theta = lam + 2 beta a on anchors 1-2,
predict anchor 3. y* used for scoring only; anchor placement uses K.

* Signed coordinate: good once leakage is moderate (a_hat within 1-2% of the
  true amplitude at the later anchors; 25-160% off at the earliest).
* Drift resolvable: |theta2-theta1| >> Kato-Temple bound except eps=.001.
* Held-out: FAILS in 6/6; miss is 0.4-4.4x the drift scale. beta_hat has
  the wrong sign in 2/6. The criterion |beta_hat-beta| < delta/a passed in
  6/6, so it is too weak (it accepts wrong signs); replace it.
* Prediction over H=1/(1-lam_hat): 0.2-2% vs frozen 0.4-3% where in band;
  catastrophic for both at eps=.001 (lam_hat wrong).

Diagnosis (`drift_check.py`, exact dense eigenvalues, scoring only):
theta(y_k)-lam* is matched by the ALL-MODE first-order law
2<v1,B(y_k-y*,v1)>_c to 0.4-26%, while the slow-only law 2 beta a1 is up to
10x off with the wrong sign (s1 eps=.01, k=8: +2.6e-2 vs observed -2.6e-3).
The rule's drift is first order in every mode's amplitude, and those decay at
their own rates, so theta is not a function of a1 in the band. The
coefficients gamma_i = 2<v1,B(v_i,v1)>_c are exactly the mixed (1,1,i)
interaction set: single-mode rule transport is not identifiable here.

Second failure: at eps=.001 (s0, k=305) theta-lam* = +4.7e-3 while first
order predicts -5.8e-5. The slowest eigenvalue of J(y_k) exceeds lam*, so
a different mode is transiently slowest (identity swap). "Spectrally
isolated slow subspace" fails inside the band at small eps.

Implications for the joint conjecture:
1. Acquisition must target theta_k as a sum of exponentials
   lam* + sum_i gamma_i a_i(0) lam_i^k (a scalar Prony-type history), or
   the retained set must contain the (1,1,i) couplings; a two-anchor line is
   insufficient. Untested.
2. Any certificate must handle slow-mode identity swaps (track an invariant
   subspace, not the top Ritz vector), or restrict to anchors after the
   swap. Untested.
