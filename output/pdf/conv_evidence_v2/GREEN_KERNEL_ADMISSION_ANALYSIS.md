# Density-screened Green admission

## Definition

Let \((\Omega,M)\) be the two-dimensional Riemannian source domain,
\(A=M^{-1}\), and let

\[
\mu=\sum_a\delta_{s_a},
\qquad
\nu=\sum_a\eta_a\delta_{s_a}.
\]

If \(\rho\) is the source count per unit Riemannian area, define

\[
L_\rho=\rho-\Delta_M,
\qquad
\Delta_M f
=|M|^{-1/2}\operatorname{div}
 \left(|M|^{1/2}M^{-1}\nabla f\right).
\]

CONV uses \(\det M=1\), so
\(\Delta_M f=\operatorname{div}(A\nabla f)\). The proposed aggregate fields
are

\[
L_\rho U=\mu,
\qquad
L_\rho V=\nu,
\qquad
\beta_G=\frac{V}{U},
\]

with the no-flux boundary condition

\[
n^TA\nabla U=n^TA\nabla V=0.
\]

The two equations share one SPD operator. A sparse factorization, spectral
diagonalization in a constant metric, or any exact linear backend can apply
the operator to the two right-hand sides together. No individual source
distance is formed.

## Why the mass term is determined

In two dimensions, \(\Delta_M\) has physical dimension
\(\text{length}^{-2}\). Source density has the same dimension. Under the
coordinate dilation \(x\mapsto cx\),

\[
\rho\mapsto c^{-2}\rho,
\qquad
\Delta_M\mapsto c^{-2}\Delta_M.
\]

Thus \(L_\rho=\rho-\Delta_M\) is scale covariant. On the determinant-one unit
sampling lattice, \(\rho=1\). This selects the screening length
\(\ell=\rho^{-1/2}\) from sampling geometry rather than from the interpolated
values or an adjustable bandwidth.

The Neumann condition is likewise structural. It is the natural variational
boundary condition, introduces no exterior value, and preserves constants.
Dirichlet zero data would create an artificial boundary attractor.

## Positivity and partition of unity

The shifted Neumann operator is coercive:

\[
\langle f,L_\rho f\rangle
=\int_\Omega \left(
 \rho f^2+\nabla f^TA\nabla f
\right)\,d\operatorname{vol}_M>0
\]

for nonzero \(f\). Its Green function \(G_\rho(x,a)\) is strictly positive by
the maximum principle. Therefore

\[
\beta_G(x)
=\frac{\sum_a\eta_aG_\rho(x,s_a)}
       {\sum_aG_\rho(x,s_a)}
\]

is a convex partition:

\[
\min_a\eta_a\leq\beta_G(x)\leq\max_a\eta_a.
\]

If all labels are the same constant, \(V=\eta U\) and
\(\beta_G=\eta\) exactly.

## Cardinal limit

In two dimensions the Green function of any smooth uniformly elliptic
second-order operator has the local form

\[
G_\rho(x,s_a)
=-\frac{1}{2\pi}\log d_M(x,s_a)+O(1).
\]

At \(s_a\), its own singular term dominates every other source term. Hence

\[
\lim_{x\to s_a}\beta_G(x)=\eta_a.
\]

The continuum construction is cardinal by continuous extension. This limit
is logarithmically slow. A naive finite-difference Green diagonal is finite
and therefore is not exactly cardinal at a finite mesh. A cardinal discrete
implementation must either retain the analytic singular basis or impose
\(\beta_G(s_a)=\eta_a\) as a nodal constraint in the equivalent weighted
problem below.

## The induced scalar transport law

Away from the sources, \(L_\rho U=L_\rho V=0\). Substituting
\(V=U\beta_G\) and cancelling the common mass term gives

\[
\Delta_M\beta_G
+2\langle\nabla\log U,\nabla\beta_G\rangle_A=0.
\]

Equivalently,

\[
\boxed{\operatorname{div}(U^2A\nabla\beta_G)=0.}
\]

Thus Green admission is a weighted harmonic transport of the nodal order
coordinate. It minimizes

\[
\mathcal E_U[\beta]
=\frac12\int_\Omega U^2\nabla\beta^TA\nabla\beta\,
 d\operatorname{vol}_M
\]

subject to the cardinal source traces. The same metric controls propagation;
\(U^2\) supplies the source-density and boundary geometry.

This provides two equivalent computational forms:

1. solve the two shared resolvent right-hand sides and form \(V/U\); or
2. solve once for \(U\), then solve the weighted harmonic problem for
   \(\beta\) with exact nodal constraints.

The second form is preferable when discrete cardinality is mandatory.

## Constant-metric kernel

For \(M=I\) on the plane,

\[
G_\rho(r)=\frac1{2\pi}K_0(\sqrt\rho\,r),
\]

where \(K_0\) is the modified Bessel function. It is positive, logarithmically
singular at zero, and exponentially localized beyond the density-determined
length \(\rho^{-1/2}\). This gives an exact convolutional realization on a
uniform infinite or periodic lattice.

The singularity is necessarily weaker than \(r^{-2}\). In two dimensions a
second-order local elliptic inverse has a logarithmic Green singularity.
No positive-order local differential inverse can reproduce the inverse-square
kernel; doing so would require an order-zero or nonlocal operator.

## Formal relation to CONV*

Both admissions have the same exact nodal trace:

\[
\beta_G(s_a)=\beta^*(s_a)=\eta_a.
\]

CONV* is the unique cellwise \(Q_1\) extension of that trace. Green admission
is the unique minimizer of the global weighted Dirichlet energy determined by
\(L_\rho\). They are therefore two canonical continuations of the same nodal
coordinate:

- CONV*: compact, algebraic, bounded work;
- Green: global, metric-covariant, two shared linear solves.

For either continuation \(\widetilde\beta\), the reconstruction relation is

\[
\mathcal T_{\widetilde\beta}Y-\mathcal T_{\beta^*}Y
=(\widetilde\beta-\beta^*)
 (\mathcal T_{yx}Y-\mathcal T_{xy}Y).
\]

The factor-order commutator remains the exact sensitivity measure.

## Discrete anisotropic reference result

The construction was evaluated with the source-derived determinant-one SPD
metric, a monotone Selling-stencil discretization, no-flux boundaries, and
the exactly cardinal weighted-harmonic form.  The same sparse factorization
was used for the two resolvent right-hand sides.  Across the fixed sixteen-case
synthetic population, the truth geometric-mean MSEs were

| source lattice | local \(Q_1\) | raw \(V/U\) | cardinal Green |
|---:|---:|---:|---:|
| \(9\times9\) | \(2.3888571\,10^{-3}\) | \(2.3889134\,10^{-3}\) | \(2.3890219\,10^{-3}\) |
| \(17\times17\) | \(1.8625653\,10^{-4}\) | \(1.8626406\,10^{-4}\) | \(1.8626100\,10^{-4}\) |

The cardinal Green coordinate differed substantially from the local
coordinate--RMS differences \(4.76\,10^{-2}\) and \(3.89\,10^{-2}\)--while
the reconstructed images differed by only \(7.78\,10^{-5}\) and
\(8.50\,10^{-6}\) RMS.  This is direct evidence for the exact sensitivity
identity above: most continuation freedom lies in a direction suppressed by
the factor-order commutator.

The raw finite-grid ratio was not cardinal: its maximum source error was
\(0.542\) and \(0.628\).  Enforcing source traces removed that error exactly,
but did not improve fidelity.  Consequently the continuum logarithmic
singularity must not be silently identified with a finite-grid Green
diagonal.

## Consequence

The Green construction is a mathematically closed two-solve replacement for
the all-source Eikonal family, but it is not an elimination that preserves the
inverse-square admission.  It defines a different global continuation.  The
anisotropic reference shows no accuracy consideration that justifies putting
that global solve in the interpolation operator.

This also sharpens the role of the local rule.  Once the nodal coordinate
\(\eta\) is fixed, \(Q_1\) is the unique extension satisfying cell locality,
linearity in the four nodal values, cardinality, and separate affine exactness.
The Green rule replaces cell locality by minimization of a global weighted
Dirichlet energy.  The experiment changed the continuation substantially and
barely changed the output, so the additional global axiom is unsupported.

The implementation must nevertheless distinguish:

- the continuum Green coordinate;
- a finite-difference ratio \(V/U\), which is not exactly cardinal; and
- the weighted-harmonic finite-element form with enforced source traces.

Only the third is simultaneously a faithful discrete Green coordinate and an
exact interpolator.  It remains useful as a reference destination and as an
exact global variational comparator.  The compact \(Q_1\) coordinate remains
the justified operational form.
