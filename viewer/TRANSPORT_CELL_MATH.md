# Transport Cells: Proposed Mathematical Model

## 1. Objective

Let

\[
z:\Omega\subset\mathbb R^2\rightarrow\mathbb R^3
\]

be an image in OKLab. We seek a coarse-to-fine representation

\[
\widehat z_L(x)=\sum_{\ell=0}^{L}\Delta_\ell(x)
\]

whose atoms are compact, image-shaped regions rather than fixed raster
blocks. The intended division of labor is:

- the BFFT cartoon determines coarse objects and barriers;
- the BFFT texture determines how distance, cell shape, and local color
  gradient travel inside those objects;
- blue-noise sampling fills every region without a grid bias;
- residual pursuit decides which regions subdivide;
- a partition of unity guarantees complete coverage without holes.

This is not ordinary anisotropic Voronoi. A fixed ellipse at a centroid sees
only the tensor at that centroid. The desired cell must respond to the tensor
at every point along its route.

### Implemented checkpoint, 2026-07-26

The operative model is a **flat global population**, not the recursive
hierarchy proposed later in this note. Parent and generation arrays are
diagnostic metadata; every site is refitted directly to the full cartoon and
texture targets over the current global partition. This distinction matters
when interpreting an experiment.

The implemented checkpoint separates four measured effects:

1. BFFT cartoon, texture, and one accurate TV outer-map defect define a
   spatial transport metric.
2. Uniform farthest-point blue noise supplies the initial complete coat.
3. Expected removable affine gain spends later sites.
4. A sparse coupled solve fits all cell planes under the partition of unity
   that actually renders them.

One-stage recursive residual memory is retained only as a focus diagnostic.
It does not contribute pixels and has zero allocation weight by default.

The hierarchical subtraction model below remains a proposal. Experiments
that attempted to approximate it with births, species, or frozen territories
performed substantially worse than this flat checkpoint.

## 2. BFFT fields

Write the lightness decomposition as

\[
z_L = u + v,
\]

where \(u\) is the BFFT cartoon and \(v\) is its texture residual. Define the
one-step cartoon-map defect

\[
\delta=u-\operatorname{ROF}(z_L-v,\lambda).
\]

This is the correction from the current TGFD cartoon toward one accurate
Gilles outer-map evaluation. It is not the unrelated flattening residual
\(u-\operatorname{ROF}(u,c)\).

Define a cartoon edge strength and normal

\[
e_c(x)=\|\nabla u(x)\|+\eta_f\|\nabla \delta(x)\|,
\qquad
n_c(x)=\frac{\nabla u+\eta_f\nabla \delta}
{\|\nabla u+\eta_f\nabla \delta\|+\epsilon}.
\]

The texture structure tensor is

\[
J_v(x)=G_\sigma *
\left(\nabla v(x)\nabla v(x)^\mathsf T\right).
\]

Let \(\lambda_1\geq\lambda_2\) be its eigenvalues, \(n_v\) the principal
gradient direction, and \(t_v=R_{\pi/2}n_v\) the tangent direction. Texture
coherence is

\[
\kappa(x)=
\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\epsilon}.
\]

Thus \(t_v\) follows a coherent stroke and \(n_v\) crosses it.

## 3. Texture is the geometry: a spatial metric

The first implementation assigns every site a single ellipse. Replace that
with the spatially varying positive-definite metric

\[
M(x)
= I
+ \beta\,\kappa(x)
\left(1+\xi\sqrt{\widehat\rho(x)}\right)
n_v(x)n_v(x)^\mathsf T
+ \gamma\,\bar e_c(x)\,n_c(x)n_c(x)^\mathsf T.
\]

The texture term makes travel across a coherent texture stroke expensive
while leaving travel along it inexpensive. Density \(\widehat\rho\) amplifies
that pressure only where a direction is already supported by coherence; high
isotropic density cannot invent an orientation. The cartoon term makes
crossing a coarse object boundary expensive. All act locally.

The distance from site \(p\) to point \(x\) is the Riemannian geodesic

\[
d_M(p,x)=
\inf_{\Gamma(0)=p,\ \Gamma(1)=x}
\int_0^1
\sqrt{\dot\Gamma(s)^\mathsf T
M(\Gamma(s))\dot\Gamma(s)}\,ds.
\]

This equation captures “where the cell runs to.” A route can bend along a
curved texture, but it pays heavily to cut across that texture or through a
cartoon boundary.

On the pixel graph, neighboring pixels \(x,y\) use the edge cost

\[
w(x,y)=
\sqrt{(y-x)^\mathsf T
\frac{M(x)+M(y)}{2}(y-x)}.
\]

A multi-source shortest-path solve then produces all site distances and
owners together. Reflecting image boundaries must be used; opposite image
edges are not neighbors.

## 4. Metric blue noise

Let the sampling density at hierarchy level \(\ell\) be

\[
\rho_\ell(x)
=\rho_{\min}
+a\,\bar e_c(x)
+b\,\sqrt{G_\sigma*(v^2)}(x)
+c\,\|R_\ell(x)\|_W,
\]

where \(R_\ell=z-\widehat z_\ell\) is the current OKLab residual.

The local exclusion radius in two dimensions is

\[
r_\ell(x)=q_\ell\,[\rho_\ell(x)]^{-1/2}.
\]

Sites form variable-radius blue noise in the geodesic metric:

\[
d_M(p_i,p_j)
\geq \max(r_\ell(p_i),r_\ell(p_j)).
\]

The initial level samples inside cartoon basins. Later levels sample only
inside the selected parent cell. A deterministic weighted farthest-point
sequence is a useful approximation, but its distance must be \(d_M\), not
ordinary Euclidean distance.

This gives both desired properties:

- edge regions receive more, smaller cells without letting them cross the
  edge;
- flat regions remain completely filled by fewer, larger cells.

## 5. Geodesic power cells and complete coverage

For site \(i\), define a power score

\[
s_i(x)=d_M(p_i,x)^2-r_i^2.
\]

The hard cell is

\[
\Omega_i=\{x:s_i(x)\leq s_j(x)\ \forall j\}.
\]

For rendering and fitting, use a local soft partition over the \(K\) best
sites:

\[
\phi_i(x)=
\frac{\exp[-\tau_i s_i(x)]}
{\sum_{j\in N_K(x)}\exp[-\tau_j s_j(x)]}.
\]

Because \(\sum_i\phi_i(x)=1\), the representation has no unfilled pixels.
At deeper levels, candidates are restricted to the same parent cell; this
makes the hierarchy respect already-discovered objects.

The radius \(r_i\) controls reach, \(\tau_i\) controls boundary softness, and
the spatial metric controls shape. These are separate concepts and should not
be collapsed into one anisotropy slider.

## 6. Transport coordinates and cell color

A Cartesian affine plane still cuts across a curved cell. Instead, let
\(\Gamma_i^x\) be the shortest path from \(p_i\) to \(x\), and let

\[
F(x)=[\,t_v(x)\ \ n_v(x)\,].
\]

Define local transport coordinates

\[
\xi_i(x)=
\int_{\Gamma_i^x}
F(\Gamma(s))^\mathsf T\dot\Gamma(s)\,ds.
\]

The cell model is an OKLab affine patch

\[
a_i(x)=c_i+A_i\xi_i(x),
\qquad
c_i\in\mathbb R^3,\quad A_i\in\mathbb R^{3\times2}.
\]

Because the coordinate frame turns along the path, the patch gradient turns
with the texture instead of merely inheriting one angle from the centroid.

Fit color and gradient jointly:

\[
\min_{c_i,A_i}
\int_\Omega \phi_i(x)
\|T_\ell(x)-c_i-A_i\xi_i(x)\|_W^2\,dx
+\lambda_g\int_\Omega\phi_i(x)
\|\nabla a_i(x)-\nabla T_\ell(x)\|_W^2\,dx
+\lambda_A\|A_i\|_F^2.
\]

\(T_0\) is the coarse cartoon-colored target. At finer levels \(T_\ell\) is
the residual. The final regularizer prevents the severe affine extrapolation
visible in the current Pikachu result.

The level reconstruction is

\[
\Delta_\ell(x)=\sum_i\phi_i(x)a_i(x).
\]

OKLab is preferable for the least-squares system; it is the Cartesian,
hue-wrap-safe form of OKLCH. UI controls and perceptual interpretation can
still be expressed as OKLCH.

### 6.1 Expected removable affine gain

Residual magnitude is not the correct refinement currency. Let

\[
R_k(x)=z_k(x)-\widehat z_k(x)
\]

be one weighted OKLab residual channel and let \(G_\sigma\) be a normalized
Gaussian whose scale follows the current mean cell spacing:

\[
\sigma = c\sqrt{\frac{|\Omega|}{N}}.
\]

Inside a radially symmetric window, the constant and the two centered linear
coordinates are orthogonal. The reduction available to a local
constant-plus-gradient correction is therefore proportional to

\[
\Delta J_k(x)=
\left(G_\sigma*R_k\right)^2
+\sigma^2
\left\|\nabla(G_\sigma*R_k)\right\|^2.
\]

The implemented field sums channels with the OKLab weights:

\[
\Delta J(x)=
\Delta J_L(x)+1.5\Delta J_a(x)+1.5\Delta J_b(x).
\]

The actual placement pressure is

\[
P(x)=
\Delta J(x)
\left[1+\alpha
\sqrt{\frac{d_M(x,\mathcal P)}
{\operatorname{percentile}_{90}d_M}}\right]
\frac{1}{1+\beta e_c(x)}.
\]

The clearance factor preserves space filling. The last term prevents an
already crowded one-pixel contour from purchasing every later site. The key
change is that \(\Delta J\) estimates what the local model can remove; a
large residual that is locally white no longer has the same value as a
large residual with coherent mean or slope.

Signed discrepancies between the target decomposition and a fresh
single-stage decomposition of the current composition can form two
additional gain accounts. They are exposed as a separate allocation
currency because experiments show that no fixed mixture dominates RGB-only
gain on every scene and budget.

### 6.2 Coupled partition-of-unity fit

The earlier implementation solved each cell using only its hard owner set,
then rendered a blend with a neighboring cell that was absent from that
fit. This makes fitting and rendering inconsistent.

With geometry fixed, write the affine basis of site \(i\) at pixel \(x\) as

\[
b_i(x)=[1,q_i(x),r_i(x)]^\mathsf T,
\]

and let \(\phi_i(x)\) be the implemented two-nearest-site partition. For one
color channel, stack all site coefficients in

\[
\theta=[\theta_1^\mathsf T,\ldots,\theta_N^\mathsf T]^\mathsf T.
\]

Each pixel contributes one sparse design row

\[
A_x=
\left[
\phi_1(x)b_1(x)^\mathsf T,\ldots,
\phi_N(x)b_N(x)^\mathsf T
\right],
\]

with at most six nonzero entries because only two cells overlap. The
coefficients are solved together:

\[
\theta^\star=
\arg\min_\theta
\|A\theta-z\|_2^2+
\lambda_s\sum_i\|\theta_{i,\mathrm{slope}}\|_2^2+
\lambda_0\sum_i\theta_{i,0}^2.
\]

This is the exact continuous least-squares solve for the current geometry
and renderer. It does not use a target-dependent blend oracle.

The multiscale version builds two sparse systems on the same sites:

\[
\widehat z(x)=
A_{\tau_c}(x)\theta_c+
A_{\tau_t}(x)\theta_t,
\qquad \tau_c<\tau_t,
\]

where the broad cartoon partition currently uses softness \(4\) and the
sharper texture partition uses softness \(16\). This gives the two BFFT
components genuinely different supports while maintaining complete
coverage in each field.

## 7. Recursive subtraction

Initialize

\[
\widehat z_{-1}=0,\qquad R_{-1}=z.
\]

At level zero, fit cartoon-scale atoms. Then repeatedly:

\[
\Delta_\ell
=\operatorname{FitCells}(R_{\ell-1};
u,v,f,\text{parent partition}),
\]

\[
\widehat z_\ell=\widehat z_{\ell-1}+\Delta_\ell,
\qquad
R_\ell=z-\widehat z_\ell.
\]

For parent cell \(i\), use the residual energy

\[
E_i=
\frac{\int\phi_i(x)\|R_\ell(x)\|_W^2\,dx}
{\int\phi_i(x)\,dx}
\]

to select subdivisions. New metric-blue-noise sites are placed at residual
maxima subject to the exclusion radius and parent barrier. This is the
original subtract-and-recurse idea expressed as a stable hierarchical
matching pursuit.

Children should be constrained to have zero weighted mean correction inside
their parent,

\[
\int \phi_{\text{parent}}(x)\Delta_{\ell+1}(x)\,dx=0,
\]

so texture refinement does not drift the parent cartoon color.

## 8. Centroid motion

If sites are relaxed, use a barrier-aware Riemannian centroid:

\[
p_i^+
=\arg\min_{p\in\Omega_{\operatorname{parent}(i)}}
\int \phi_i(x)\rho_\ell(x)d_M(p,x)^2\,dx.
\]

This is a geodesic Lloyd step. It cannot jump across a cartoon boundary.
Unconstrained Euclidean averaging, as in the first implementation, can.

## 9. What the current prototype gets wrong

The current code is a useful visualization, but its reconstruction is not yet
the model above:

1. It samples with Euclidean weighted farthest points.
2. It reads one tensor value at each centroid and freezes a straight ellipse.
3. It has no cartoon edge-crossing cost or inherited parent barrier.
4. It fits independent Cartesian planes directly to the full image.
5. Those planes can extrapolate across long cells and create bright/dark
   shards.
6. A Euclidean centroid update can move a site through an object boundary.
7. Wrapped finite differences falsely connect opposite image borders.

The Pikachu image makes all seven failures visible. The next solver should
first replace ownership with the geodesic pixel-graph construction. Only
after the cells respect objects should transport-coordinate color fitting and
recursive subtraction be added.

## 10. Minimal next implementation

A disciplined implementation order is:

1. reflect-boundary BFFT tensor fields;
2. multi-source geodesic ownership with cartoon barriers;
3. geodesic weighted blue-noise initialization;
4. parent-contained residual subdivision;
5. bounded constant-color cells as a geometry control;
6. Cartesian affine color as an A/B;
7. transported-coordinate gradient fitting;
8. soft top-\(K\) partition and optional radius/temperature refinement.

Constant-color cells in step 5 are important. If their shapes do not follow
Pikachu cleanly, color-gradient sophistication cannot repair the geometry.
