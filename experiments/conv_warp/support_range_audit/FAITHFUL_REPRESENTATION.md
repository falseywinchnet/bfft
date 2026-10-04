# Fixed algebraic preservation of the order blend

Additional local source analysis, adaptive subdivision, feature classification,
and new active-state searches are excluded from this construction. The input
is two already represented tensor-quintic potentials and the existing bilinear
order coordinate. All coefficient operations have a fixed, source-independent
layout.

Let

\[
A=\sum_{i,j=0}^5 A_{ji}B_i^5(u)B_j^5(v),\qquad
B=\sum_{i,j=0}^5 B_{ji}B_i^5(u)B_j^5(v),
\]

\[
\beta=\sum_{r,s=0}^1 b_{sr}B_r^1(u)B_s^1(v),\qquad
T=(1-\beta)A+\beta B.
\]

Define \(w_{i0}=(6-i)/6\) and \(w_{i1}=i/6\). Terms with indices
outside \(0,\ldots,5\) are omitted. Then

\[
\boxed{
P_{ji}=\sum_{r,s=0}^1 w_{ir}w_{js}
\left[(1-b_{sr})A_{j-s,i-r}+b_{sr}B_{j-s,i-r}\right],
\quad 0\leq i,j\leq6.
}
\]

**Proposition.** The tensor degree-six polynomial with controls \(P\)
equals \(T\) at every point of the cell.

**Proof.** The Bernstein product identity is

\[
B_k^5(t)B_r^1(t)
=\frac{\binom5k\binom1r}{\binom6{k+r}}B_{k+r}^6(t).
\]

For \(i=k+r\), the coefficient ratio is \(w_{ir}\). Applying this
identity on both axes to the two products gives the displayed expression. ∎

The transform uses at most four neighboring coefficients from each order for
each output coefficient. Its weights are fixed rational constants. It does
not estimate derivatives, search a neighborhood, select a filter, or decide
whether a feature is an edge or an extremum.

For \(0\leq b_{sr}\leq1\), the output controls are convex combinations
of the contributing order controls. More strongly, the represented function
is the exact pointwise blend and therefore lies between \(A(u,v)\) and
\(B(u,v)\). If both orders share source vertex values, those values remain
fixed. If each order has shared boundary traces and the bilinear weights
agree on incident edges, the resulting degree-six traces also agree.

All derivatives and all linear pixel functionals of this polynomial are
preserved, including the order-coordinate gradient term
\((B-A)\nabla\beta\). The complete-cell integral is

\[
\int_C T=\frac{|C|}{49}\sum_{i,j=0}^6P_{ji}.
\]

With local quintic interpolation denoted by \(\mathcal I\), the existing
blended collocation proposal is \(\mathcal I[(1-\beta)A+\beta B]\).
The fixed transform retains the degree-six component that this interpolation
generally removes. It uses 49 coefficients per cell instead of 36. A shared
rectangular lattice changes from approximately 25 to 36 stored controls per
source cell away from boundaries. These are arithmetic and storage counts,
not measured execution-time claims.

When the original factor outputs are functions \(f,g\) which have first
been represented as \(A=\mathcal I f\), \(B=\mathcal I g\), the retained
field is

\[
T_{\mathrm{rep}}=(1-\beta)\mathcal I f+\beta\mathcal I g.
\]

Relative to their original blend, its error is exactly

\[
T_{\mathrm{rep}}-[(1-\beta)f+\beta g]
=(1-\beta)(\mathcal I f-f)+\beta(\mathcal I g-g).
\]

Thus the transform adds no blend representation error to the two order
representations. It does not recover an active-state change lost when either
order was interpolated. No additional local analysis is proposed to recover
such changes here.

The current browser instead weights factor samples before its inverse
collocation passes, then clips the joint controls and performs cone admission.
Using the displayed transform would require retaining the unweighted order
representations, compiling their blend, and extending downstream coefficient
and integration operations to degree six. Subsequent clipping or admission
would again change the resulting field; their effects are separate from this
identity. No production pipeline change is made by this reference.

`faithful_blend.py` implements the fixed transform. Its exact-rational checks
cover every monomial in the two quintic input spaces against every monomial
in the bilinear weight space, for both factor contributions: 288 coefficient
identities. Further checks establish identity under equal orders and retention
of the degree-six tensor term and its exact integral. These are algebraic
checks, without image experiments or parameter fitting.
