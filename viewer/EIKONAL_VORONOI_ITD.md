# Rejected: iterative free-site amplitude–Eikonal Voronoi ITD

**Status: rejected on 2026-07-29.** The persistence/extrema initialization was
useful diagnostically, but distortion births, repeated remarches, Lloyd
motion, and heat continuation duplicate quantities already emitted directly
by the frozen Meyer/segmentation geometry. Better images do not excuse an
iterative allocation mechanism here. The replacement is documented in
[`MEYER_DENSITY_VORONOI.md`](MEYER_DENSITY_VORONOI.md).
The rejected implementation has been removed from the live viewer module.

## Why the first prototype had a scale conflict

The first 2-D operator selected extrema with a Euclidean `min_distance` and
then queried nearest sites in the point cloud `(x, y, amplitude)`. That puts a
fixed spatial scale into the support before the image geometry is known.
Pikachu needs sparse cells in the black background and narrow cells around its
ears, paws, and tail; the coins need many cells on textured metal but long
cells across the ground. No single exclusion radius can express both.

Euclidean nearest-neighbour distance also ignores everything between a pixel
and a site. Two endpoints can be close in feature space even when a real edge
separates them.

## Revised metric

For a scalar amplitude \(f:\Omega\to\mathbb R\), embed the image domain as

\[
X(x,y)=\left(x,y,\gamma\, f(x,y)/s\right),
\]

where \(s^2\) is a robust global percentile of
\(\lVert\nabla f\rVert^2\). The pullback of Euclidean length on this surface is

\[
M_f=I+\frac{\gamma^2}{s^2}\nabla f\,\nabla f^\top.
\]

For vector guidance \(g=(g_1,\ldots,g_c)\), used by the RGB viewer,

\[
M_g=I+\frac{\gamma^2}{s^2}
       \sum_c \nabla g_c\,\nabla g_c^\top.
\]

The geodesic action of a path \(p\) is

\[
\mathcal L_M[p]=
\int_0^1\sqrt{\dot p(t)^\top M_g(p(t))\dot p(t)}\,dt,
\qquad
d_M(x,s)=\inf_{p(0)=s,p(1)=x}\mathcal L_M[p].
\]

Thus travel tangent to an isophote remains cheap, while crossing an amplitude
edge is expensive. The cells are the Eikonal Voronoi regions

\[
V_i=\{x:d_M(x,s_i)\le d_M(x,s_j),\ \forall j\}.
\]

The implementation solves the anisotropic Eikonal equation with the existing
FM-LBR-style reduced lattice basis marcher. This avoids the finite-direction
crystalline distance of a four- or eight-neighbour graph.

## Sites without a cell-size parameter

Initial germs are regional `h`-maxima and `h`-minima. Their persistence height
is

\[
h=\max\left(
  \epsilon\,(\operatorname{p95}f-\operatorname{p5}f),
  \kappa\,\widehat\sigma_{\rm residual}
\right),
\]

where the residual noise is estimated by a MAD after Gaussian continuation.
This is an amplitude/topology criterion, not a distance between sites.

The current adaptive step is a penalized facility-location approximation. For
cell \(V_i\), define

\[
D_i=\int_{V_i}\rho(x)\,d_M(x,s_i)^2\,dx,\qquad
\rho(x)=0.05+\eta\,\min(\sqrt{\operatorname{tr}J_g/s^2},4).
\]

Cells whose \(D_i\) is a robust statistical outlier are split at

\[
s_{\rm new}=\arg\max_{x\in V_i}\rho(x)d_M(x,s_i)^2.
\]

The child has no prescribed separation from its parent. Its position is a
consequence of the current partition and distortion. Optional constrained
Lloyd passes relocate a germ to the density barycenter of its own cell, snapped
back inside that cell. Relocation is off by default because the Pikachu
ablation found that persistence extrema plus births preserve thin contours
better than unconstrained centroid motion.

This split rule is the greedy form of minimizing

\[
\mathcal E(S)=
\int_\Omega \rho(x)\min_{s_i\in S}d_M(x,s_i)^2\,dx+\lambda |S|.
\]

A later native version should replace the robust split test by an explicit
birth/death evaluation of \(\Delta\mathcal E\), which would make \(\lambda\)
the only population regularizer.

## Intrinsic knot graph and continuation

Two sites are adjacent only when their Eikonal cells share an image-space
interface. This dual replaces the old spatial Delaunay triangulation in the
ITD knot update:

\[
b_i=(1-\alpha)f(s_i)+
\alpha\frac{\sum_{j\sim i}d(s_i,s_j)^{-1}f(s_j)}
                 {\sum_{j\sim i}d(s_i,s_j)^{-1}}.
\]

Start with the hard cell field \(B_0(x)=b_{\operatorname{owner}(x)}\), then
apply a small number of conservative anisotropic heat steps in \(M_g\):

\[
B_\tau=\exp(\tau\Delta_{M_g})B_0.
\]

The heat operator preserves constants, so this is implicitly a
partition-of-unity continuation. It needs neither a dense pixel-by-site matrix
nor a fixed number of overlapping neighbours. The rotation remains
\(R=f-B_\tau\), hence every multilevel result telescopes exactly:

\[
f=\sum_\ell R_\ell+B_{\rm residual}.
\]

## Pikachu ablation at 256 × 256

The alpha channel is used only as evaluation truth, never by the operator.
Every cell is assigned its majority alpha label after decomposition; this is
an oracle grouping score that measures whether the partition is capable of
representing the object.

| support | cells | foreground IoU | boundary F, 2 px |
|---|---:|---:|---:|
| fixed-spacing lifted Euclidean, best small sweep | 110 | 0.965 | 0.929 |
| free-site RGB Eikonal, compact setting | 207 | 0.972 | 0.985 |
| free-site RGB Eikonal, denser research setting | about 275 | about 0.975 | pending |

The new operator materially improves contour allocation and removes the
coins-versus-Pikachu spacing choice. It does **not** yet constitute perfect
semantic extraction: the nearly black ear tip is visually almost identical to
the black background, and cell grouping is intentionally outside this
unsupervised support operator.

## Research lineage

- Du and Wang, *Anisotropic Centroidal Voronoi Tessellations and Their
  Applications*, SIAM J. Sci. Comput. 2005:
  <https://doi.org/10.1137/S1064827503428527>
- Mirebeau, *Anisotropic Fast-Marching on Cartesian Grids Using Lattice Basis
  Reduction*, SIAM J. Numer. Anal. 2014:
  <https://arxiv.org/abs/1201.1546>
- Wang et al., *Structure-Sensitive Superpixels via Geodesic Distance*,
  IJCV 2013:
  <https://doi.org/10.1007/s11263-012-0588-6>
- Ye et al., *Geodesic Centroidal Voronoi Tessellations: Theories, Algorithms
  and Applications*, 2019:
  <https://arxiv.org/abs/1907.00523>
- Couprie et al., *Power Watersheds: A New Image Segmentation Framework
  Extending Graph Cuts, Random Walker and Optimal Spanning Forest*, ICCV 2009:
  <https://perso.esiee.fr/~coupriec/945paper.pdf>
