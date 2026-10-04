# Finite current concentration under geometric bounds

Let \(P\) denote the shared Bernstein control vector of an admitted CONV
potential. Write its finite admissibility conditions as
\[
 AP\ge b,\qquad EP=e.
\]
The inequalities include the declared source-support intervals and the
Bernstein coefficients of the dual current-cone conditions. The equalities
include source interpolation and any prescribed traces or moments. The
source state, supports, cones, and bounds are fixed during the following
construction.

## Concentration of a quintic current

A scalar quintic interval has the representation
\[
 U(t)=u_0+\sum_{k=0}^4 c_k\tau_k(t),\qquad
 \tau_k(t)=\sum_{j=k+1}^5 B_j^5(t),
 \qquad U'(t)=5\sum_{k=0}^4c_kB_k^4(t).
\]
For a monotone interval of increment \(\delta\), its signed-current fibre is
\(\sum c_k=\delta\), with all currents having the sign of \(\delta\).
The central current \(c^\sharp=(0,0,\delta,0,0)\) yields
\[
 U^\sharp(t)=u_0+\delta S(t),\qquad
 S(t)=10t^3-15t^4+6t^5.
\]
Thus \(c+\theta(c^\sharp-c)\), \(0\le\theta\le1\), preserves the increment
and sign while concentrating the current. Its six potential controls tend
toward \((u_0,u_0,u_0,u_1,u_1,u_1)\). The limiting profile has zero first and
second derivatives at its endpoints and maximum slope \(15|\delta|/8\).

This defines a finite proposal in the existing current geometry. More general
proposals may redistribute current around another phase, alter the source
metric and re-evaluate the finite factor construction, or change the local
knots. Admission is applied to the resulting potential correction. For
moving knots, both potentials must first be represented on a common finite
refinement, with the constraints formed on that refinement.

## Geometric margin admission

Choose a finite collection of proposed control corrections \(d_g\), each
satisfying \(Ed_g=0\). Define
\[
 s_k=(AP-b)_k,\qquad H_k=\sum_g[-(Ad_g)_k]_+,
 \qquad R_k=H_k/s_k.
\]
Set \(R_k=0\) when \(H_k=s_k=0\), and \(R_k=+\infty\) when
\(H_k>0=s_k\). For each correction define
\[
 \alpha_g=\min\left(1,
   \min_{k:(Ad_g)_k<0}\frac{s_k}{H_k}\right),\qquad
 P_0=P+\sum_g\alpha_gd_g,
\]
where an empty minimum imposes no restriction.

**Proposition.** The vector \(P_0\) satisfies \(AP_0\ge b\) and \(EP_0=e\).

**Proof.** For row \(k\), the harmful contributions obey
\[
 \sum_{g:(Ad_g)_k<0}\alpha_g[-(Ad_g)_k]
 \le (s_k/H_k)H_k=s_k
\]
when \(H_k>0\). When \(H_k=0\), every contribution is nonnegative.
Adding the helpful contributions establishes the inequality. Each correction
belongs to the nullspace of \(E\), so the equalities are preserved. ∎

Helpful contributions can leave unused margin. Put
\[
 r=P+\sum_g d_g-P_0,
 \qquad t=\min\left(1,
 \min_{k:(Ar)_k<0}\frac{(AP_0-b)_k}{-(Ar)_k}\right).
\]
Then \(P_*=P_0+tr\) is feasible, and \(t\) is the largest admissible common
step in \([0,1]\) along this remaining direction. The construction uses
finite products, sums, comparisons, and divisions. It requires no iterative
projection or optimization.

For one proposed direction \(D\) and requested strength \(\gamma\), the rule
reduces to
\[
 R=\max_k\frac{\gamma[-(AD)_k]_+}{s_k},\qquad
 P_*=P+\gamma\min(1,R^{-1})D.
\]
The value \(R=1\) is the first contact with an admissibility face. This ratio
compares the actual proposed geometric change with the remaining margin in
the same constraint. It does not require a universal sharpening threshold.
The maximality statement is along the specified ray; the grouped construction
need not maximize a global sharpness functional.

## Derivative bounds and transition width

On a cell with side lengths \(h_x,h_y\), let
\[
 U(u,v)=\sum_{i,j=0}^5 P_{ij}B_i^5(u)B_j^5(v).
\]
For \(0\le a,b\le5\),
\[
 \partial_x^a\partial_y^bU
 =\frac{(5)_a(5)_b}{h_x^ah_y^b}
 \sum_{i=0}^{5-a}\sum_{j=0}^{5-b}
 (\Delta_x^a\Delta_y^bP)_{ij}B_i^{5-a}(u)B_j^{5-b}(v).
\]
Consequently the finite inequalities
\[
 \left|\frac{(5)_a(5)_b}{h_x^ah_y^b}
 (\Delta_x^a\Delta_y^bP)_{ij}\right|\le\Gamma_{ab}
\]
certify \(|\partial_x^a\partial_y^bU|\le\Gamma_{ab}\) throughout the cell.
Each is two rows of \(AP\ge b\), so the same margin law admits simultaneous
range, slope, curvature, and higher mixed-derivative restrictions. The
Bernstein coefficient condition is sufficient and can be conservative.
Bounds must admit the chosen base potential. Selecting smaller bounds
requires a separately established feasible base.

The certificate can be tightened without changing the potential. Let
\(T_\ell\) be the fixed de Casteljau map from a derivative control net to its
controls on subcell \(\ell\). Replacing the whole-cell inequalities by
\(|T_\ell D_{ab}P|\le\Gamma_{ab}\) for every subcell is again a finite set
of linear inequalities. Nonnegative subdivision maps preserve any previous
certificate and can admit fields rejected by a coarse coefficient bound.
For \(S'\), the whole-interval quartic controls are \((0,0,5,0,0)\), whereas
the controls on the left half are \((0,0,5/4,15/8,15/8)\); the right half
is their reversal. One subdivision therefore certifies the exact maximum
\(15/8\), in place of the coarse bound \(5\). It changes neither the source
geometry nor the actual derivative allowance.

For a transition of amplitude \(A\) and physical width \(\varepsilon\),
\[
 U(x)=u_0+A S((x-L)/\varepsilon),\qquad
 \|U^{(r)}\|_\infty=|A|C_r/\varepsilon^r,
\]
where
\[
 (C_1,C_2,C_3,C_4,C_5)
 =\left(\frac{15}{8},\frac{10}{\sqrt3},60,360,720\right).
\]
Thus derivative allowances determine a minimum transition width,
\[
 \varepsilon\ge
 \max_{r:\Gamma_r\text{ prescribed}}
 \left(\frac{|A|C_r}{\Gamma_r}\right)^{1/r}.
\]
The constants describe the polynomial transition interior. Extending the
profile constantly beyond its endpoints gives a globally \(C^2\) function;
its higher derivatives are piecewise derivatives. Global \(C^q\) continuity
requires equality of the appropriate physical derivative traces across every
join. These trace equalities can be included in \(E\). A flat-ended \(C^q\)
transition requires degree at least \(2q+1\): prescribing the two endpoint
values and the first \(q\) zero derivatives at both ends gives \(2q+2\)
conditions. A septic transition therefore supplies flat-ended \(C^3\)
regularity. Higher degree and sharper transitions are separate choices.

## Source-cell moments and shared traces

A tensor-quintic patch has the exact integral
\[
 \int_C U\,dx\,dy=\frac{|C|}{36}\sum_{i,j=0}^5P_{ij}.
\]
Preserving its current integral is the equality \(\sum_{ij}\delta P_{ij}=0\).
Explicit moment-preserving corrections can be formed without solving a
system. An interior correction is made zero-sum by subtracting its mean
from the sixteen interior controls. For each shared edge, place the proposed
change on its four nonvertex controls and place minus one sixteenth of their
sum on every interior control of each incident cell. Each such edge bundle
has zero integral in each incident cell and one shared edge trace. Overlap
between bundles is allowed in the margin proof. Source vertices remain fixed.
Additional derivative-trace equalities require correction bundles that also
satisfy those equalities.

Preserving the integral of the base field preserves its existing moments.
For input samples specified as pixel averages, exact source consistency is
instead the prescribed condition
\(\int_{C_i}U=|C_i|Y_i\). It requires a base satisfying those prescribed
moments. A sample-range bound remains unchanged by sharpening admission: if
all admitted controls are at most \(0.585\), the reconstructed value cannot
reach \(1\). A latent foreground/background coverage construction supplies
different physical bounds and a different source functional.

## Warped-pixel transfer

After concentration and admission, CONV* acts on the new potential by
\[
 B_F[j]=\frac1{|Q_j|}\int_{Q_j}U_*(F(q))\,dq.
\]
The integration geometry and the finite Bernstein moment banks are unchanged
when the source knots and degree are unchanged. Updating the admitted
controls consequently changes the source geometry without replacing target
area integration. Positive normalized integration preserves an admitted
scalar range. Source derivative bounds transport through the actual warp:
\[
 \nabla(U_*\circ F)=J_F^T\nabla U_*,
\]
\[
 D^2(U_*\circ F)=J_F^T(D^2U_*)J_F+
 \sum_a(\partial_aU_*)D^2F_a.
\]
Thus target-space slope and curvature allowances also depend on the warp.
For a fixed affine map, directional target derivative bounds are additional
linear control inequalities. A projective map requires bounds on its varying
Jacobian and higher derivatives over the footprint.

With fixed \(A,b,E,e\), every repeated admitted update remains in the same
feasible set. Uniform control over a refinement family requires derivative
allowances in physical units: the factors \(h_x^{-a}h_y^{-b}\) cannot be
omitted. Feasibility supplies the stated range, derivative, moment, and trace
properties; image fidelity is determined by the proposed geometry and the
source functional.
