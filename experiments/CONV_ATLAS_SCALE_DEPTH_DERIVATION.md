# Why atlas depth behaves differently under enlargement and reduction

## 1. Operators

Let $R$ be positive $2\times2$ block averaging, $S$ the admitted zero-detail
four-child synthesis, and $D$ piecewise-constant duplication of one parent
value into its four children. Both prolongations preserve the parent mean:

\[
 RS=I,\qquad RD=I.
\]

For a source cell-average raster $z$, put

\[
 A_0z=z,\qquad A_Jz=S^Jz,
\]

and compare successive depths on the same fine lattice through

\[
 e_{J+1}=A_{J+1}z-D A_Jz.
\]

It follows without approximation that

\[
 Re_{J+1}=0.
\tag{1}
\]

Thus the information introduced by every additional zero-detail level is a
blockwise zero-mean dyadic detail. Equation (1), rather than a frequency
heuristic, is the source of the observed scale asymmetry.

## 2. Exact footprint-cancellation identity

Let $B$ be one target footprint and let $P_B^{J+1}$ denote positive
integration of a level-$(J+1)$ piecewise-constant atlas over $B$:

\[
 P_B^{J+1}u=\frac1{|B|}
 \sum_{Q\in\mathcal Q_{J+1}}|B\cap Q|u_Q.
\]

Partition the children $Q$ by their level-$J$ parent $C$. If $C$ is contained
in $B$, (1) cancels its complete contribution; if $C$ is disjoint from $B$,
it contributes nothing. Consequently

\[
 \boxed{
 P_B^{J+1}A_{J+1}z-P_B^{J+1}D A_Jz
 =\frac1{|B|}
 \sum_{C:\,C\cap\partial B\ne\varnothing}
 \sum_{Q\subset C}|B\cap Q|(e_{J+1})_Q.}
\tag{2}
\]

This identity is exact. An additional atlas level changes a target average
only through parent cells cut by the footprint boundary.

If $U_J(B)$ is the part of $B$ covered by those cut cells, then

\[
 \left|P_B^{J+1}A_{J+1}z-P_B^{J+1}D A_Jz\right|
 \leq \frac{|U_J(B)|}{|B|}\,\|e_{J+1}\|_\infty.
\tag{3}
\]

For an axis-aligned rectangular footprint of side lengths $b_x,b_y$ and a
level-$J$ parent width $\delta_J$,

\[
 \frac{|U_J(B)|}{|B|}
 \leq
 \min\!\left\{1,
 2\delta_J\!\left(\frac1{b_x}+\frac1{b_y}\right)
 \right\}.
\tag{4}
\]

The analogous warped-footprint estimate replaces the rectangular expression
by a boundary-tube area, proportional to
$\delta_J\operatorname{perimeter}(B)/|B|$ under bounded distortion.

For reduction, $B$ is large relative to the source cell. Most dyadic parents
are complete, and (2) cancels their new detail exactly. Refinement therefore
improves the geometric placement of the few cut boundary cells while positive
integration suppresses most internal excursion.

For enlargement, $B$ is comparable to or smaller than a source cell. Its
boundary-cell fraction approaches one, so (3) supplies no attenuation below
$\|e_{J+1}\|_\infty$. The same recursively predicted detail is exposed
directly as visible subcell structure.

## 3. Why the first, twofold atlas is canonical for enlargement

Let $\mathcal T_z$ be the admitted continuous CONV profile constructed from
the observed source raster, and let $Q_J$ return its exact dyadic cell averages
at spacing $2^{-J}$. On every interior source cell, the first atlas level
satisfies

\[
 A_1z=S z=Q_1\mathcal T_z,
\tag{5}
\]

because its half-cell means are evaluated exactly by three-point
Gauss--Legendre integration of the piecewise quintic profile. The two outer
cells use the separately declared one-sided jet closure. Thus, globally,
$A_1$ is the direct half-cell functional of the original source profile and
its fixed boundary closure; no synthesized child raster is used as input.

The next recursive level is instead

\[
 A_2z=S(A_1z)=Q_1\mathcal T_{Q_1\mathcal T_z},
\tag{6}
\]

on its interior, which is not generally equal to

\[
 Q_2\mathcal T_z.
\tag{7}
\]

Equation (6) reconstructs a new profile from already predicted half-cell
means. Its difference from (7) is a refinement-semigroup defect; it has zero
parent mass but is not new measured information. Accordingly, $2\times$ is
the unique nonrecursive atlas depth evaluated directly from the original CONV
profile, with exact interior half-cell integration and the declared outer
closure. Among detail-carrying atlas states it also has the smallest
worst-case stability factor: the existing one-level bound is $3$, whereas the
$J$-level bound grows as $3^J$.

This proves a principled statement for enlargement: if an atlas rather than
direct point synthesis is required, the twofold state is the last depth whose
subcells are obtained directly from the source-defined profile. Deeper atlas
levels require an additional refinement model and are fully exposed by small
output footprints. Direct CONV remains the canonical arbitrary enlargement
operator.

## 4. What can and cannot be proved about fourfold reduction

The fourfold state $A_2z$ is the **minimal** atlas that introduces
within-half-cell variation. In a local Taylor interpretation, $A_1$ carries
the parent mean and first split moment; $A_2$ additionally distinguishes the
two half-cell moments, whose symmetric and antisymmetric combinations carry
the next slope and curvature information. By (2), reduction sees those new
coordinates only where its footprint cuts the twofold lattice. This explains
why $4\times$ gives a large fidelity gain with very little final range
excursion in the measured reductions.

It does not make $4\times$ a universal mathematical optimum. A further level
again adds a zero-mean boundary correction and can improve the footprint
integral. On a stratified 280-case plane-wave check, the eightfold atlas beat
the fourfold atlas in all 56 sampled $25\to9$ reductions and all 56 sampled
$25\to13$ reductions. The geometric-mean fourfold-to-eightfold MSE ratios
were respectively $2.9044$ and $1.9652$. Hence an MSE-only theorem saying
“fourfold is optimal for reduction” would be false.

The defensible result is:

\[
 \boxed{
 \begin{array}{l}
 2\times\text{ is the canonical nonrecursive cell atlas for enlargement};\\
 4\times\text{ is the least-depth curvature-bearing atlas for reduction}.
 \end{array}}
\tag{8}
\]

Calling $4\times$ *optimal* requires an objective that includes work or
storage, not fidelity alone. Each additional two-dimensional level multiplies
the atlas population by four, while (2)--(4) make its useful action a shrinking
boundary correction. A deterministic scale-only stopping rule can therefore
select the smallest $J$ whose certified boundary bound in (3) is below a
declared numerical tolerance. Without such a tolerance or resource term,
there is no finite universal optimal depth: finer conservative atlases may
continue to reduce boundary-integration error.

## 5. Paper wording

The theoretical body may use (1)--(4) as a footprint-cancellation proposition
and (5)--(7) to distinguish direct from recursive atlas depth. The observed
choice of two levels for the reduction implementation belongs in the
implementation/evidence insert. It should be described as the least-depth
curvature-bearing or selected operating point, not as a universal optimum.

The stratified depth record is stored in
`output/support_geometry/conv_atlas_depth_probe.json` and is generated by
`experiments/conv_atlas_depth_probe.py` from the expanded analytic census.
