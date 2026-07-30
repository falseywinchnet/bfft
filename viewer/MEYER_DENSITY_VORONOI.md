# Frozen Meyer-density Voronoi amplitude operator

## Constraint

The segmentation stage may perform one frozen image measurement and one
causal first-arrival solve. It may not search for sites, alternate site and
cell updates, split residuals, diffuse supports, or run to convergence.

The two-stage C++ Meyer trace is not being used as an optimization target.
Its first stage is the universal triangular pair of screened Fourier maps.
The second stage is the first feedback of the bounded isotropic dual-ball
projection. It therefore exposes nonlinear support geometry with one local
projection event, before the long Gilles/Meyer convergence trajectory begins.

## Frozen support tensor

Let \(u\), \(v\), and \(\delta\) denote the two-stage cartoon, texture, and
cartoon-side outer defect measured by the shipped C++ kernel. After amplitude
normalization, the support channels are

\[
c_0=u-G_3*u,\qquad
c_1=\sqrt{\omega_v}\,v,\qquad
c_2=\sqrt{\omega_\delta}\,\delta .
\]

Their smoothed structure tensor is

\[
J=G_{1.5}*\sum_k \nabla c_k\nabla c_k^\top.
\]

Reliability gates distinguish transported nonlocal orientation from local
evidence in the unchanged RGB image. After imposing a maximum physical
support floor and collapsing the minor eigenvalue on coherent contours, the
result is the positive-definite inverse-support tensor \(Q(x)\).

The important long-edge rule is the coherent minor-eigenvalue collapse. A
straight contour keeps its normal precision but its tangent precision falls
toward the support floor. It therefore produces long regions rather than a
necklace of nearly round cells.

## Population without a requested cell size

The ellipse

\[
E_x=\{z:z^\top Q(x)z\leq1\}
\]

has area \(\pi/\sqrt{\det Q(x)}\). Its reciprocal area is therefore the
continuous cell density

\[
\rho(x)=\frac{\sqrt{\det Q(x)}}{\pi}.
\]

This is the missing amplitude operator: support size is an algebraic output
of the measured tensor. A fixed low-discrepancy phase quantizes
\(\rho(x)\) into the complete germ population simultaneously. There is no
candidate ranking or exclusion distance.

Director curvature supplies a second analytic correction. If the predicted
tangent and normal semi-spans are \(a\) and \(b\), a contour of curvature
\(\kappa\) has tangent sagitta approximately \(\kappa a^2/2\). The required
population multiplier is

\[
q_\kappa(x)=
\sqrt{\max\left(1,\frac{\kappa(x)a(x)^2}{2b(x)}\right)}.
\]

Thus a straight support remains long, while a sharply turning ear, paw, or
coin boundary receives more germs in one image pass. This is dynamic cell
size without Lloyd motion or births.

## One causal partition

The transport metric is the existing segmentation metric

\[
M(x)=I+\beta Q(x)/q_{90}+\gamma^2 B(x),
\]

where \(B\) is the separately measured photometric jump tensor. Population
and boundary blocking are deliberately separate: a strong edge can prevent
crossing without automatically manufacturing sites.

All germs enter one reduced-basis anisotropic fast march. The output is the
hard Eikonal Voronoi owner map and arrival distance. Cell interfaces provide
the intrinsic knot graph.

## Bounded amplitude readout

For each cell \(V_i\), measure its scalar amplitude once:

\[
a_i=\frac{1}{|V_i|}\sum_{x\in V_i} f(x).
\]

The intrinsic graph update is a convex neighbour average

\[
b_i=(1-\alpha)a_i+
\alpha\frac{\sum_{j\sim i}w_{ij}a_j}{\sum_{j\sim i}w_{ij}}.
\]

The baseline is \(B(x)=b_{\operatorname{owner}(x)}\), and the rotation is
\(R=f-B\). Conditional on the frozen cells this readout is linear, bounded by
the measured cell amplitudes, and has no nonlinear products. Reconstruction
is exact by definition: \(f=B+R\).

## Measured checkpoint

On the original 475 × 475 Pikachu, with the frozen geometry measured at full
resolution and the allocation march restricted to 256 × 256:

- two C++ Meyer/defect stages: about 85–95 ms;
- warm population plus Eikonal march: about 32 ms;
- analytic curvature population: about 850 cells;
- oracle foreground IoU: about 0.973;
- boundary F-score at two pixels: about 0.990;
- exact full-resolution march, when requested: about 135 ms allocation,
  0.979 IoU, and 0.997 boundary F.

The earlier iterative free-site experiment reached a slightly higher oracle
IoU only by spending several remarches and hundreds of milliseconds. It is
not the production direction.
