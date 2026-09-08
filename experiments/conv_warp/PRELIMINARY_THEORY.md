# Direct CONV warp study

## One inverse map

A single map \(F:q\mapsto p\) sends target coordinates to source coordinates.
The point operator is

\[
  (W_FY)(q)=T_Y(F(q)).
\]

No rotation, skew, anisotropic scale, affine map, or homography is decomposed
into shear passes. A three-shear implementation is the same operator only if
its intermediate sampling maps compose exactly to the expression above. In
general they do not, because admitted CONV synthesis is nonlinear and repeated
sampling changes its state.

The warp induces its geometry rather than adding a new geometric input. If
\(M(p)\) is a source metric and \(J_F(q)\) the inverse-map Jacobian, then

\[
  G_F(q)=J_F(q)^T M(F(q))J_F(q),\qquad
  \bar M_F(q)=\frac{G_F(q)}{\det(G_F(q))^{1/2}}.
\]

Thus the determinant-one tensor carries shape and \(\det(G_F)^{1/4}\) carries
linear scale. Applying Cartesian scale factors inside a separately rotated
eigenframe would incorrectly assume that the two axis systems commute.

## Joint current admission

The Cartesian factor theorem does not control a derivative taken in an
arbitrary source direction. The joint construction represents every source
cell by one bidegree-\((5,5)\) Bernstein potential. Adjacent patches use one
shared control line, so the atlas is \(C^0\) and its current is integrable by
construction.

For cell \(C\), let \({\cal G}_C\) contain the four corner gradients of every
degree-elevated \(Q_1\) cell whose nodes occur in the declared two-jet support.
The witnessed current cone is

\[
  K_C=\operatorname{cone}{\cal G}_C.
\]

This object does not select an owner or a preferred normal. A clean edge
produces a narrow cone, curvature produces a wedge, and mutually crossing
currents can generate the entire plane, in which case no unsupported
directional restriction is imposed.

In two dimensions \(K_C\) has at most three required dual generators
\(\ell_{Cr}\). Cone admission is consequently the finite family

\[
  \ell_{Cr}^{T}\nabla U_C(u,v)\geq0.
\]

After degree elevation, every left-hand side has 36 Bernstein coefficients.
Nonnegativity of those coefficients proves the inequality over the complete
cell. The admissible set is nonempty: the \(Q_1\) gradient is a convex
combination of its four corner gradients, all of which occur in
\({\cal G}_C\), and therefore \(\nabla Q_1\in K_C\) everywhere.

The implemented admission is finite. Start with the global degree-elevated
\(Q_1\) lattice \(B\), write the canonical high-order lattice as

\[
  P=B+\sum_g d_g,
\]

and assign every nonvertex control to exactly one shared-edge or cell-interior
entity \(g\). For an inequality \(a_k^Tz\geq0\), define its baseline margin
and total harmful proposed current by

\[
  m_k=a_k^TB,\qquad
  H_k=\sum_g\max(0,-a_k^Td_g).
\]

Each harmful entity receives the capacity

\[
  \alpha_g\leq\min_{k:a_k^Td_g<0}\min\!\left(1,\frac{m_k}{H_k}\right).
\]

It follows immediately that

\[
  a_k^T\!\left(B+\sum_g\alpha_gd_g\right)
  \geq m_k-\frac{m_k}{H_k}H_k=0.
\]

A final common ray consumes the remaining margin exactly. The rule has no
iteration, tolerance, active-set search, or raster-scan order. Source vertices
remain fixed, shared edges remain single variables, an already feasible
high-order proposal passes unchanged, and affine fields are reproduced.

Amplitude admission uses the same declared support. For each shared control,
intersect the per-channel min--max intervals of every incident cell support,
and clip the proposal to that nonempty interval before current admission. The
baseline control is a convex combination of local samples and belongs to every
incident interval. Since the final control lies on the segment between that
baseline and the clipped proposal, it remains in the intersection. Bernstein
convexity consequently bounds the complete continuous patch—not only its
sampled evaluations—by its admitted support range.

The cone law transports through the warp without a new directional choice. If
\(\gamma\) is a target-space curve, then

\[
 \frac{d}{ds}U(F(\gamma(s)))=
 \bigl(J_F(\gamma(s))\gamma'(s)\bigr)^T\nabla U.
\]

Hence this derivative is nonnegative whenever the pushed-forward tangent lies
in \(K_C^*\). Affine maps transport such directions linearly; homographies
transport them as a spatially varying direction field. This is the precise
sense in which the joint theorem controls rotated, skewed, and perspective-
warped currents.

## Warped-pixel analysis

For a target Voronoi basin \(P_j\), the raster functional is

\[
  (B_FY)_j=\frac1{|P_j|}\int_{P_j}T_Y(F(q))\,dq.
\]

This definition is conservative because the target basins partition the
domain. It is also a positive average of the admitted continuous profile.

For affine \(F\), a target rectangle maps to a source parallelogram. Splitting
that parallelogram at source knot lines leaves convex polygons lying in single
Bernstein patches. Fan triangulation is positive. On each triangle the
integrand has total degree at most ten; the Duffy Jacobian raises one separate
degree to at most eleven. Tensor Gauss order six therefore evaluates the
integral exactly.

For a projective map, source-coordinate integration gives

\[
  (B_FY)_j=\frac1{|P_j|}
  \int_{F(P_j)}T_Y(p)\,\left|\det D F^{-1}(p)\right|\,dp,
\]

where

\[
  \left|\det D F^{-1}(p)\right|
  =\frac{|\det H^{-1}|}{|\ell(p)|^3}.
\]

The exact integral remains the canonical operator. Its implementation uses
the local expansion

\[
  (1+z)^{-3}=\sum_{k=0}^{K}(-1)^k {k+2\choose2}z^k+R_K(z).
\]

The finite sum times the quintic atlas is polynomial and is integrated
exactly by a correspondingly raised Duffy--Gauss rule. If \(|z|\leq r<1\),
the absolute tail is bounded without asymptotics by

\[
 |R_K(z)|\leq
 \sum_{k=K+1}^{\infty}{k+2\choose2}r^k.
\]

That sum is evaluated in closed form. Four-way triangle subdivision is used
only when the resulting certified pixel error exceeds the requested bound.
Constants have zero residual before expansion and are therefore preserved
exactly.

## Present evidence boundary

The finite current-cone atlas is cardinal, \(C^0\), affine-exact,
support-range admitted, and satisfies all imposed directional Bernstein
inequalities to floating roundoff in the current tests. On the 72
curved-interface warp probes at source sides 13, 17, and 25, both wrong-way
normal variation and range excursion are at floating zero. At side 9, range
admission remains exact and mean wrong-way variation is
\(1.94\times10^{-3}\), because the declared support spans competing conic
normals. These measurements select among proved constructions; they are not
themselves the variation proof.

The affine footprint rule is exact for the admitted polynomial atlas. On a
nonconstant projective plane test, an independent 40-point tensor integration
differed by \(1.18\times10^{-11}\), inside the computed
\(5.84\times10^{-10}\) certificate. The complete proposal analysis now uses
the native compiled factor passes on one shared lattice; at source side 25 it
takes about 0.12 s for a scalar field and 0.25 s for three channels in the
current unoptimized Python admission harness.

The vector-valued construction is the direct product of the scalar channel
cones and ranges under one common inverse map. Its guarantee is componentwise;
it does not claim invariance under arbitrary mixing of channel coordinates.
For perspective footprints, the exact integral is the mathematical operator
and the implementation returns a certified approximation. Those are the main
remaining boundaries of the present result.
