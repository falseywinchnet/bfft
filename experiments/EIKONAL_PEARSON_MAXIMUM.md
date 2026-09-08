# Eikonal Pearson maximum-error measure

Let `f` be the reference, `u` the reconstruction, and `r=u-f` their paired
spatial residual.  At every admitted interior point form the differential

\[
 dr=r_x\,dx+r_y\,dy
\]

and its centered Pearson product-moment tensor

\[
 C_r
 =\frac{1}{|\Omega|}\int_\Omega
 (\nabla r-\overline{\nabla r})
 (\nabla r-\overline{\nabla r})^Tdx
 =
 \begin{bmatrix}
  \operatorname{Var}(r_x)&\operatorname{Cov}(r_x,r_y)\\
  \operatorname{Cov}(r_x,r_y)&\operatorname{Var}(r_y)
 \end{bmatrix}.
\]

The off-diagonal Pearson product moment is essential.  Reporting only
`E(r_x^2)` and `E(r_y^2)` tests the two Cartesian covectors and cannot give
the error along a different orientation.

Every Euclidean unit Eikonal covector `p` defines

\[
 T_p(x)=p^Tx,\qquad |\nabla T_p|=1.
\]

The residual current observed in that basis is `j_p=p^T grad r`, and its mean
square is exactly

\[
 \operatorname{Var}(j_p)=p^TC_rp.
\]

Consequently the maximum over the complete continuum of straight Eikonal
bases is not an angular search:

\[
 \boxed{
 \mathcal E_{\max}^2(r)
 =\max_{\lVert p\rVert=1}p^TC_rp
 =\lambda_{\max}(C_r).}
\]

For two dimensions,

\[
 \lambda_{\max}
 =\frac{P_{xx}+P_{yy}
 +\sqrt{(P_{xx}-P_{yy})^2+4P_{xy}^2}}{2}.
\]

The reported Pearson maximum is `sqrt(lambda_max)`, in value units per source
pixel.  Its eigenvector is the orientation at which residual variation is
largest.
No directional grid, fitted angle, image classifier, or interpolator state is
used.

## Exact properties

Each centered differential contributes the positive-semidefinite matrix

\[
 (\nabla r-\overline{\nabla r})
 (\nabla r-\overline{\nabla r})^T\succeq0.
\]

Thus oppositely signed errors cannot cancel their directional energy.  The
Rayleigh--Ritz theorem proves that the principal eigenvalue is the exact
maximum over all orientations.  In particular,

\[
 \lambda_{\max}(P_r)\ge\max(P_{xx},P_{yy}),
\]

with strict inequality when a nonzero cross moment rotates the principal
error away from both Cartesian axes.

Pearson centering deliberately removes a constant directional ramp as well as
a constant value bias.  The implementation therefore also reports the mean
gradient and the exact total second-moment maximum obtained from
`C_r + mean(grad r) mean(grad r)^T`.  Constant value bias is reported through
the residual mean and residual RMS.  None is silently folded into Pearson
correlation.

For a constant SPD Riemannian metric, the same construction becomes the
largest generalized eigenvalue under the Eikonal constraint
`p^T M^{-1}p=1`.  The present diagnostic uses the isotropic metric so that it
does not favor any reconstruction orientation.

## Qualification tests

The executable tests establish:

1. identical fields have zero maximum;
2. a 37.5-degree plane wave produces a principal error normal within half a
   degree and a maximum strictly larger than either Cartesian diagonal term;
3. horizontal and vertical rotations have equal maxima to numerical
   precision; and
4. a supported oriented error has substantially larger product moment than
   a single pixel of the same amplitude.

For the 129-to-33-to-129 synthetic cycles, maximum directional RMS error is:

| reconstruction | ridge | edge |
|---|---:|---:|
| CONV basin | 0.052414 | 0.028628 |
| straight owner chart | 0.050557 | 0.027575 |
| Lanczos-3 | 0.052061 | 0.028682 |
| exact straight Eikonal fibre | 0.048022 | 0.026567 |

On the complete `text` cycle, basin CONV gives 0.043714, the straight owner
chart 0.043539, and Lanczos-3 0.043861.  The owner chart changes the maximum by
only 0.4 percent, correctly describing it as a marginal rather than a visibly
new result.
