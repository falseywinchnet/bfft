# The finite-flow Meyer jump

## 1. Reference computation

The reference is the optimized 64-pass, warm-interleaved Meyer--Bregman
recurrence.  A pass is not a scalar cartoon/texture filter.  Its reduced live
state is

\[
 z=(u,w,t_u^x,t_u^y,t_w^x,t_w^y),
\]

where `u` is the cartoon-side primal field, `w` is the texture-side survivor,
and the four `t` fields are the accumulated split-Bregman gradient states.
The emitted texture and effective cartoon are

\[
 v=f-u-w,\qquad \bar u=f-v=u+w,
\]

so \(f=\bar u+v\) to roundoff at every finite pass.

Let \(D\) be the periodic forward gradient, \(D^*\) its negative divergence,
and

\[
 \Pi_a(t)=\begin{cases}
 t,&|t|\le a,\\
 a,t/|t|,&|t|>a,
 \end{cases}
 \qquad R_a(t)=t-2\Pi_a(t).
\]

With

\[
 c_u=\lambda,\quad \eta_u=2\lambda,\quad a_u=\eta_u^{-1},
 \qquad
 c_w=\mu^{-1},\quad \eta_w=10\mu^{-1},\quad a_w=\eta_w^{-1},
\]

and screened inverse \(S(c,\eta)=(cI+\eta D^*D)^{-1}\), one exact fused
pass \(T_f:z\mapsto z^+\) is

\[
\begin{aligned}
u^+ &=S(c_u,\eta_u)
 \{c_u(u+w)-\eta_uD^*R_{a_u}(t_u)\},\\
w^+ &=S(c_w,\eta_w)
 \{c_w(f-u^+)-\eta_wD^*R_{a_w}(t_w)\},\\
t_u^+&=Du^++\Pi_{a_u}(t_u),\\
t_w^+&=Dw^++\Pi_{a_w}(t_w).
\end{aligned}
\]

These are exactly the equations evaluated by the full-spectral C++ kernel:
the two screened solves are lower triangular in Fourier space, and the two
reflected divergences are streamed before the paired inverse transforms.

## 2. What the old hard jump removes

Write \(H\) for its finite scalar contraction, \(P_K=I-H^K\), and let
\(\mathcal S\) denote its scalar separator.  If \((u_*,v_*)\) is the effective
64-pass split, the two-observation hard construction satisfies the exact
identity

\[
 v_H-v_*
 =\underbrace{P_K(u_*-s_1)}_{\text{signed structural halo}}
 -\underbrace{H^K v_*}_{\text{texture retained in cartoon}}
 +\underbrace{q_\mu}_{\text{final capacity correction}},
\]

where \(s_0=\mathcal S(f)\),
\(s_1=\mathcal S(f-P_K(f-s_0))\), and \(q_\mu\) is the final capacity
projection correction.  The preceding second observation is

\[
 f-P_K(f-s_0)=u_*+H^K v_*+P_K(s_0-u_*).
\]

Thus the two visible faults have one cause: the six-field finite trajectory
was replaced by a finite scalar contraction observed twice.

* For any Fourier carrier with scalar contraction gain \(0<\rho<1\), the
  omitted tail is \(H^Kv_*=\rho^K v_*\ne0\).  It is necessarily left in the
  effective cartoon.
* For a step mismatch \(e=u_*-s_1\),
  \(P_Ke=e-h_K*e\).  A positive normalized smoothing kernel \(h_K\) produces
  the paired positive/negative contour response of a halo.  Its mean is zero,
  so suppressing only one sign cannot repair the construction.

The identity was checked numerically on six independent synthetic controls;
the largest relative identity residual was \(1.4\times10^{-13}\).  The
reproduction script and arrays are
`experiments/meyer_hard_jump_defect_proof.py` and
`experiments/out/meyer_hard_jump_defect_proof/`.

This proof is source-independent.  It does not classify edges, textures,
regions, amplitudes, or frequencies.  It explains why a content admission
rule is the wrong repair: such a rule acts after the two missing terms have
already been created.

## 3. Jumping the actual fused trajectory

After a short ordinary prefix, fix the current state \(z\), define

\[
 r=T_f(z)-z,\qquad A\in\partial T_f(z),
\]

and consider a finite horizon \(m\).  If the pass map were affine on the
visited state chart, then exactly

\[
 T_f^m(z)-z = p_m(A)r,
 \qquad p_m(A)=I+A+\cdots+A^{m-1}.
\]

This is the appropriate object.  The target is a later *finite fused state*,
not the stationary Newton solution \((I-A)^{-1}r\).  The latter was tested
and rejected because it overshoots the desired finite trajectory at
structural transitions.

The only nonsmooth derivative required is already determined by the disk
projection in the reference pass.  For a direction \(h\),

\[
D\Pi_a(t)h=\begin{cases}
h,&|t|<a,\\
\dfrac{a}{|t|}(I-nn^T)h,&|t|>a,
\end{cases}
\qquad n=t/|t|,
\]

and \(DR_a(t)h=h-2D\Pi_a(t)h\).  At the measure-zero boundary
\(|t|=a\), the implementation takes the interior member of the Clarke
generalized derivative.  Differentiating the triangular pass gives

\[
\begin{aligned}
\dot u^+ &=S(c_u,\eta_u)
 \{c_u(\dot u+\dot w)-\eta_uD^*DR_{a_u}(t_u)\dot t_u\},\\
\dot w^+ &=S(c_w,\eta_w)
 \{-c_w\dot u^+-\eta_wD^*DR_{a_w}(t_w)\dot t_w\},\\
\dot t_u^+&=D\dot u^++D\Pi_{a_u}(t_u)\dot t_u,\\
\dot t_w^+&=D\dot w^++D\Pi_{a_w}(t_w)\dot t_w.
\end{aligned}
\]

No image statistic enters these equations.

## 4. Depth-two Krylov compression

The implementation evaluates only

\[
 r,\qquad Ar,\qquad A^2r.
\]

Let \(B=[r,Ar]\), \(G=B^TB\), and \(K=B^TAB\).  The Galerkin matrix in this
nonorthogonal basis is

\[
 \widehat A=G^{-1}K.
\]

Because \(Ar\) is the second basis vector, the first column of
\(\widehat A\) is exactly \((0,1)^T\).  Only the projection of \(A^2r\)
requires a two-by-two Gram solve.  The finite displacement is

\[
 \delta_m=B\,p_m(\widehat A)e_1.
\]

This form avoids normalization and full-field modified Gram--Schmidt passes.
A fixed 64-chunk parallel reduction computes the five required products
\(\langle r,r\rangle\), \(\langle r,Ar\rangle\),
\(\langle Ar,Ar\rangle\), \(\langle r,A^2r\rangle\), and
\(\langle Ar,A^2r\rangle\), with a fixed final summation order.  The result is
bit-identical across thread counts.

## 5. Error statement

The finite-flow jump does not claim that a depth-two polynomial equals 64
nonlinear passes.  Its error has a different and explicit origin from the
hard jump.

Write the semismooth expansion at the current state as

\[
 T_f(z+d)=T_f(z)+Ad+N_z(d).
\]

Let \(d_k=T_f^k(z)-z\), \(p_0=0\),
\(p_{k+1}=r+Ap_k\), and \(e_k=d_k-p_k\).  Then

\[
 e_{k+1}=Ae_k+N_z(d_k),
\]

so, whenever \(\|A\|\le\gamma\) and
\(\|N_z(d)\|\le L\|d\|^2/2\) on the traversed chart,

\[
 \|e_m\|\le {L\over2}
 \sum_{k=0}^{m-1}\gamma^{m-1-k}\|d_k\|^2.
\]

The implemented approximation adds the Krylov projection error

\[
 \|p_m(A)r-Bp_m(\widehat A)e_1\|.
\]

Two ordinary fused passes after each jump refresh the nonlinear projection
branches, thereby shortening the chart over which the remainder is
accumulated.  Repeating jump--refresh blocks replaces one large extrapolation
by several locally relinearized finite displacements.

Consequently the remaining discrepancy from pass 64 is the sum of

1. finite-horizon truncation relative to the selected target,
2. depth-two Krylov projection error,
3. accumulated semismooth chart remainder, and
4. the ordinary fused tail after the last refreshed state.

The old terms \(P_K(u_*-s_1)\) and \(-H^Kv_*\) cannot occur in this
decomposition of the error: the method never forms \(s_1\), \(P_K\), or a
scalar surrogate for the six-field state.  This is the structural repair.

## 6. Fixed schedules and cost

For a prefix \(p\), horizon \(m\), two tangent products, \(s\) ordinary
refresh passes, and \(j\) jumps, the exact operator-pass cost is

\[
 C=p+j(1+2+s).
\]

The two public schedules are fixed before seeing an image:

| schedule | prefix | horizon | refresh | jumps | pass cost |
|---|---:|---:|---:|---:|---:|
| fast | 4 | 18 | 2 | 3 | 19 |
| quality | 4 | 10 | 2 | 5 | 29 |

At 256 square pixels on all six surgical controls, the fast schedule is
closer to fused pass 64 than ordinary fused pass 19, and the quality schedule
is closer than ordinary fused pass 29.  The native M4 Mini measurements at
256 square pixels are 11.521 ms (fast), 18.155 ms (quality), and 34.598 ms
(fused 64), giving 3.00x and 1.91x speedups.  The old hard jump is faster, but
its proven halo and retained-texture terms make it a negative control rather
than a Pareto endpoint.

## 7. Shorter-jump constraint ablation

A post-jump tangent state need not lie exactly on the nonlinear Bregman graph.
Writing `b_u=t_u-Du` and `b_w=t_w-Dw`, exact pass states satisfy
`|b_u|<=a_u` and `|b_w|<=a_w`.  The pointwise retraction

\[
t_u\leftarrow Du+\Pi_{a_u}(t_u-Du),\qquad
t_w\leftarrow Dw+\Pi_{a_w}(t_w-Dw)
\]

restores that feasibility in linear spatial work and no transforms.  This was
tested as the analytic alternative “jump less, constrain, and relinearize”:

1. Removing both ordinary refresh passes without retraction was unstable;
   error rose to 2--8 times the equal-cost ordinary baseline at 128 square.
2. Dual-graph retraction prevented that blow-up, but retraction alone gave up
   most of the acceleration gain.
3. Shorter jumps with one exact refresh appeared superior at 128 square, but
   did not uniformly beat the existing two-refresh schedules at 256 square.
4. Reducing only the fixed horizon redistributed small wins among the six
   controls and did not yield a dominating schedule.

The constraint operator itself is therefore cheap; a useful universal
replacement is not yet cheaper in total.  The evidence indicates that at
least one true-map pass is needed to refresh nonlinear projection branches,
and the extra relinearizations consume approximately the spectral work saved
by deleting refreshes.  This is recorded as a negative ablation rather than a
resolution-selected production change.
