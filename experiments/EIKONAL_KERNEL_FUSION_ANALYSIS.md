# Exact fusion of the all-source Eikonal admission

## Result

For a general spatially varying SPD metric, the inverse-square all-source
admission does not collapse to a closed scalar Eikonal equation and does not
equal the cell-local CONV* coefficient. Exact fusion is available in special
translation-invariant geometries, or after replacing the inverse-square
distance kernel by the Green kernel of a chosen operator. For the declared
kernel, the general exact object remains a dense nonlocal kernel application.

The local coefficient nevertheless has a stronger interpretation than an
empirical approximation: it is the unique bilinear finite-element interpolant
of the formal admission's exact nodal trace. Its output defect is the product
of the admission interpolation error and the factor-order commutator.

## 1. The transformed Eikonal hierarchy does not close

Write \(A=M^{-1}\), and away from source \(a\) put

\[
u_a=D_a^{-2}.
\]

Since \(\nabla D_a^T A\nabla D_a=1\),

\[
\nabla u_a=-2D_a^{-3}\nabla D_a,
\qquad
\nabla u_a^TA\nabla u_a=4u_a^3.
\]

Thus each inverse-square field is itself a transformed Eikonal solution. For

\[
U=\sum_a u_a,
\qquad
V=\sum_a\eta_a u_a,
\]

one obtains

\[
\|\nabla U\|_A^2
=4\sum_a u_a^3
+8\sum_{a<b}D_a^{-3}D_b^{-3}
  \langle\nabla D_a,\nabla D_b\rangle_A.
\]

The second sum contains the pairwise arrival directions. It is not determined
by \(U\) and \(V\). The same obstruction appears in
\(\langle\nabla U,\nabla V\rangle_A\) and \(\|\nabla V\|_A^2\).

This failure is pointwise. In Euclidean geometry take two sources at equal
distance \(r\) and give both the same label \(c\). Then

\[
U=2r^{-2},\qquad V=2cr^{-2},
\]

independently of the angle between the two arrival covectors, whereas

\[
\|\nabla U\|^2=8r^{-6}(1+\cos\theta).
\]

Opposite and orthogonal arrivals therefore have identical \(U,V\) and
different gradient norm. Hence no autonomous first-order scalar Eikonal law
whose state is only \(U,V\) can reproduce every source configuration. Adding
one vector moment does not terminate the problem: differentiating it exposes
directional second moments and distance Hessians. Exact local propagation
produces a moment hierarchy rather than a two-scalar closure.

## 2. No exact collapse to cell-local weights

At every nonsource point at finite Riemannian distance from all sources,

\[
w_a(x)=\frac{D_a(x)^{-2}}{\sum_bD_b(x)^{-2}}>0.
\]

The bilinear cell weights are zero for every source outside the containing
cell. If

\[
\sum_aw_a(x)\eta_a
=\sum_{p\in\operatorname{cell}(x)}b_p(x)\eta_p
\]

held for every label vector \(\eta\), equality of the two linear functionals
would require equality of every coefficient. Strict positivity outside the
cell contradicts that requirement.

The exact rational counterexample is already present in one-dimensional
Euclidean space. Put sources at \(0,1,2\), query \(x=1/2\), and take labels
\((0,0,1)\). The inverse-square kernels are \((4,4,4/9)\), so

\[
\beta(1/2)=\frac{4/9}{4+4+4/9}=\frac1{19},
\qquad
\beta^*(1/2)=0.
\]

Consequently, eliminating the Eikonal solve while retaining the exact
all-source Shepard functional is impossible by a strictly cell-local rule.

There is also no fixed-rank exact global compression for arbitrary labels.
For fixed target and source sets, \(V=K\eta\), where
\(K_{xa}=d_M(x,a)^{-2}\). Already in a constant metric, distinct targets
chosen sufficiently close to corresponding sources make \(K\) strictly
diagonally dominant and nonsingular. Its determinant is analytic away from
collisions, so the constant-metric kernel matrix is full rank for generic
configurations. Any representation through a fixed number \(r<A\) of label
aggregates has rank at most \(r\) and therefore cannot equal \(K\eta\) for
every \(\eta\). Since the varying-metric class contains the constant case,
the same fixed-rank exact compression cannot exist for the full class.

## 3. Exact fusion that is possible

If \(M\) is constant and the domain is convex,

\[
D_a(x)^2=(x-a)^TM(x-a),
\]

and the two required sums are translations of one kernel,

\[
U=K*\mathbf 1,
\qquad
V=K*\eta,
\qquad
K(z)=(z^TMz)^{-1}.
\]

On a common uniform lattice they are two discrete convolutions. Zero-padded
FFT evaluation computes them without materializing any individual distance
field, subject only to floating-point arithmetic and explicit treatment of
the cardinal singular sites. This is an exact mathematical fusion of the
discrete constant-metric operator. It does not extend to a general varying
metric because \(d_M(x,a)\) is not translation invariant.

An exact two-field PDE fusion would become available if the kernel were a
Green kernel. If \(K_M(\cdot,a)\) satisfied

\[
L_MK_M(\cdot,a)=\delta_a,
\]

then

\[
L_MU=\sum_a\delta_a,
\qquad
L_MV=\sum_a\eta_a\delta_a
\]

would require only two solves. The declared \(d_M^{-2}\) is not the Green
kernel of the two-dimensional Laplace--Beltrami operator: its local
fundamental singularity is logarithmic, while \(d^{-2}\) is the Euclidean
Laplacian singularity associated with four dimensions. Choosing a heat,
resolvent, or other spectral kernel could provide two-field fusion, but it
would define a different admission law.

For the declared varying-metric kernel, hierarchical matrices, kernel FMM,
or grouped propagation can reduce work approximately. Their acceptance
criterion is an approximation tolerance, so they cannot supply a
parameter-free exact replacement.

## 4. What CONV* computes canonically

The formal action partition is cardinal:

\[
\beta(x_p)=\eta_p.
\]

Therefore

\[
\beta^*(x)=\sum_{p\in\operatorname{cell}(x)}b_p(x)\eta_p
\]

is precisely the \(Q_1\) finite-element interpolant \(I_h\beta\) of the
formal admission's known nodal trace. It is the unique rule that is

1. local to the containing cell;
2. linear in its four nodal order coordinates;
3. cardinal at the cell vertices; and
4. affine in each Cartesian coordinate separately.

Indeed, a separately affine function has the basis \(1,x,y,xy\), and its four
vertex values uniquely determine its four coefficients. The resulting basis
functions are the four bilinear barycentric weights. Their nonnegativity
gives \(0\leq\beta^*\leq1\).

The nodal coordinate itself also reduces exactly. With principal eigenpair
\((\lambda_1,n)\), secondary eigenvalue \(\lambda_2\), and
\(\chi=(\lambda_1-\lambda_2)/(\lambda_1+\lambda_2)\),

\[
\eta
=\frac{1-\chi}{2}+\chi n_y^2
=\frac{G_{yy}}{\operatorname{tr}G}.
\]

This is the unique normalized-diagonal rule that is scale invariant,
reflection invariant, maps a pure vertical current to one, maps a pure
horizontal current to zero, and changes to \(1-\eta\) when the axes are
exchanged.

## 5. Formal output control

Let \(A_Y=\mathcal T_{xy}Y\) and \(B_Y=\mathcal T_{yx}Y\). Formal CONV and
CONV* obey the identity

\[
\mathcal T^*Y-\mathcal TY
=(I_h\beta-\beta)(B_Y-A_Y).
\]

This is not merely an empirical explanation. If \(\beta\) is Lipschitz on a
cell of diameter \(h_c\),

\[
|I_h\beta-\beta|\leq h_c\operatorname{Lip}(\beta).
\]

If it is twice differentiable on a rectangular cell with side lengths
\(h_x,h_y\), positivity of the one-dimensional interpolation operators gives

\[
|I_h\beta-\beta|
\leq
\frac{h_x^2}{8}\|\partial_{xx}\beta\|_\infty
+\frac{h_y^2}{8}\|\partial_{yy}\beta\|_\infty.
\]

Multiplication by \(|B_Y-A_Y|\) gives the corresponding CONV output bound.
The approximation is therefore suppressed twice: by refinement of the nodal
admission and by near-commutation of the two admitted factor orders. At source
sites, and for every field on which the factor orders commute, the fast and
formal outputs agree exactly.

## Decision

The harder exact fusion does not exist as a two-scalar local Eikonal solve for
the declared varying-metric inverse-square kernel. The honest alternatives
are:

- retain the formal kernel and use an approximate hierarchical summation;
- change to a Green kernel and obtain two exact aggregate PDE solves; or
- make the local \(Q_1\) admission canonical.

The third choice is the only one that is simultaneously parameter-free,
bounded-work, local, cardinal, axis-covariant, and already used by CONV*.
Its relationship to the Eikonal construction is exact at the nodes and
quantified everywhere by the interpolation--commutator identity. It should be
described as a principled replacement of the all-source kernel, not as its
exact computational fusion.
