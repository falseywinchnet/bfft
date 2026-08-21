# Joint posterior/residual value–jet contractor

This checkpoint keeps the exchange conservative and contracts a set rather
than selecting a denoised image.  If

\[
    y=p+r, \qquad t=G\alpha, \qquad 0\leq\alpha\leq1,
\]

then every candidate exchange obeys

\[
    p'=p+t, \qquad r'=r-t, \qquad p'+r'=y.
\]

`G` is the exact local `(continuous scale lineage, Selling edge)` generator
family.  Posterior shed and inherited residual ancestry remain distinct in
that family.

## Cross-fitted covariance coordinate

For every horizontal, vertical, and primitive diagonal target edge
`i -> j`, the two parallel edges displaced by `+/-` its transverse covector
are witnesses.  Neither witness shares an endpoint with the target.  In the
value/first-jet chart

\[
    z_f(i,j)=(f_i, f_j-f_i),
\]

the two witnesses determine a tangent `tau`.  The contractor uses only its
unit normal `nu`; amplitude along the witnessed tangent is deliberately free.
The scalar coordinate is

\[
    h_f(i,j)=\nu^T z_f(i,j).
\]

The interval hull of the two witness coordinates and the current target
coordinate gives `[l,u]`.  Current inclusion is structural, not a fitted
margin: it guarantees that zero exchange remains feasible.  Residual and
posterior impose opposite actions on the same transfer:

\[
 l_r \leq H_r(r-G\alpha) \leq u_r,
 \qquad
 l_p \leq H_p(p+G\alpha) \leq u_p.
\]

The resulting state is the constrained zonotope

\[
 \mathcal C=\{G\alpha:\underline\alpha\leq\alpha\leq\overline\alpha,
 b_-\leq [H_rG;H_pG]\alpha\leq b_+\}.
\]

No penalty, noise label, threshold, band, or learned parameter enters this
construction.  Opposite-fold target exclusion is not statistical
independence, however, and the three-point normal interval is not a population
confidence law.

## Measured result

The size-20 audit is in
`joint_value_jet_zonotope_contractor_20.json`.  The coefficient interval hull
barely changes because tens of thousands of local generators can compensate
one another.  The sparse slab intersection itself contracts strongly:

| Scene | Condition | mean constrained support / parent support | full-action rows rejected |
|---|---:|---:|---:|
| cameraman | clean | 0.132 | 0.790 |
| cameraman | mixed | 0.241 | 0.724 |
| tapered hair | clean | 0.046 | 0.855 |
| tapered hair | mixed | 0.243 | 0.715 |
| woven chirps | clean | 0.224 | 0.688 |
| woven chirps | mixed | 0.273 | 0.698 |

Thus an unchanged axis-aligned coefficient box is not evidence of an
unchanged feasible set.  Discarding the sparse inequalities would discard the
result.

The retrospective true exchange satisfies both posterior and residual normal
slabs on only 1.2–8.1% of target edges in the clean controls and 3.1–5.2% in
the mixed controls.  This falsifies the contractor as a replacement for the
support set.  It is instead a sharply defined **noise-consistent stability
component**: actions inside it leave both sides locally explainable by
target-excluded parallel relations.  Large support-forming actions should
usually live outside it.

The state therefore retains four branches:

1. uncontracted identity support explanation;
2. uncontracted positively pushed support explanation;
3. joint-contracted identity noise stability;
4. joint-contracted positively pushed noise stability.

This is a mixture of meanings, not a choice of denoising modes.  The next law
must transport mass between support explanation and noise stability using the
action needed to enter each component.  It must not collapse the mixture to
the coefficient-box shadow, and it must not promote this research state to the
GUI before that action law produces a defensible point readout.
