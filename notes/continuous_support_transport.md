# Continuous support transport

## What the tripod experiment falsified

The original eight-neighbour walk does not merely approximate the desired
transport slowly. It solves a different local geometry.

For a BFFT-derived SPD path metric \(M(x)\), the intended cost of a curve is

\[
L_M(\gamma)=\int_0^1
\sqrt{\dot\gamma(t)^T M(\gamma(t))\dot\gamma(t)}\,dt.
\]

The corresponding arrival field from a site satisfies the anisotropic
eikonal equation

\[
\nabla T(x)^T M(x)^{-1}\nabla T(x)=1.
\]

An edge-Dijkstra discretization replaces the ellipse of possible local
velocities by the convex hull of a finite set of graph edges. Its unit ball
is therefore a polygon (a Wulff crystal). Horizontal, vertical, and exact
diagonal characteristics are cheap; intermediate directions are staircases.
The distortion grows rapidly with tensor anisotropy.

Measured worst angular cost inflation:

| metric condition | 8 edges | radius-2, 16 edges | radius-3, 32 edges | radius-4, 48 edges |
|---:|---:|---:|---:|---:|
| 16 | 1.94x | 1.38x | 1.19x | 1.12x |
| 64 | 3.46x | 2.14x | 1.64x | 1.40x |
| 256 | 6.70x | 3.91x | 2.78x | 2.21x |

This is why the vertical tripod leg developed a long clean support while
oblique legs did not. Adding 48 global directions made the other legs appear,
but it remained both expensive and substantially biased.

## Why infinitely many directions do not require infinitely many edges

The missing operation is a simplex Hopf--Lax update.

For a receiver \(x\), let \(y,z\) be two accepted vertices of a local
triangle. Instead of choosing either graph edge, minimize over the entire
opposing edge:

\[
T(x) =
\min_{\alpha\in[0,1]}
(1-\alpha)T(y)+\alpha T(z)
+\left\|x-\big((1-\alpha)y+\alpha z\big)\right\|_{M(x)}.
\]

The continuous scalar \(\alpha\) supplies every direction inside the cone
spanned by \(x-y\) and \(x-z\). Stencil vectors bound cones; they are not path
directions.

A fixed Cartesian triangulation is still inadequate at high anisotropy:
the minimizing characteristic can fall outside its causal cone, collapsing
the update back to a vertex. In two dimensions, Lagrange/Gauss lattice
reduction constructs an \(M(x)\)-obtuse superbase at each receiver. Its six
signed vectors rotate and stretch with the local tensor, making the
Hopf--Lax triangles causal.

On the 512px cameraman tensor:

- path anisotropy ratio: median 2.17, p90 16.33, p99 49.39, max 77.46;
- reduced superbase: exactly six local vectors;
- required Chebyshev reach: median 1px, p90 4px, p99 16px, max 34px;
- all superbases satisfy zero vector sum, unimodular determinant, and
  pairwise non-positive metric inner products to numerical tolerance.

Thus the tensor chooses a few locally relevant lattice vectors. There is no
global direction catalogue.

## Constant-metric control

`experiments/hopf_lax_rotation_control.py` compares the reduced-basis
Hopf--Lax solve against the exact constant-metric distance.

At metric condition 64, across tangent rotations from 0 to 45 degrees:

- mean relative distance error: 0.48% to 1.32%;
- p95 relative distance error: 1.33% to 4.89%.

This replaces the 3.46x worst cost inflation of the eight-edge graph and the
1.40x inflation still present with 48 global edges.

## Image-level control

`port_needed/continuous_eikonal_transport.py` implements a first-label
multi-source fast march using:

1. one locally reduced superbase per pixel;
2. continuous Hopf--Lax triangle updates;
3. four immediate-neighbour consistency edges as a connectivity floor at
   discontinuous measured tensors.

With the exact same 3,804 sites on cameraman at 512px:

- eight-edge support/readout: 27.45 dB, objective 0.002370;
- continuous support/readout: 27.79 dB, objective 0.002210;
- continuous front solve: about 0.44 seconds after compilation.

All three tripod legs develop aligned supports. This is already faster than
the radius-4 graph control (about 20.3 seconds for allocation and final
refresh in the measured run).

## First arrival, not runner-up ownership

The continuous front is causal. A pixel begins unclaimed and is accepted by
the first source-labelled front that reaches it. Acceptance is irreversible:

\[
T(x)=\min_i \left(a_i+d_M(s_i,x)\right),\qquad
\ell(x)=\arg\min_i \left(a_i+d_M(s_i,x)\right).
\]

The source offset \(a_i\) is an initial action/power potential if one is
needed. The metric distance is accumulated transport energy. There is no
probabilistic ownership and no meaningful runner-up state.

A support stops growing because another advancing front has already
purchased that location at lower action. If the wrong support stops short,
the error lies in its accumulated action, source potential, or local metric,
not in a missing blend with a second-best source.

Cell reductions must consequently use accepted mass, accepted barycenter,
arrival-energy moments, terminal-front energy, and characteristic strain
along accepted paths. The previous owner/runner gap is removed from the
model.

The collision point on a hard interface is nevertheless available at
subpixel precision without evaluating a runner-up. For adjacent accepted
pixels \(p,q\) with different labels and symmetric crossing cost \(c\), linear
arrival along the edge gives

\[
s_{p\rightarrow q}
=\frac{T(q)+c-T(p)}{2c}.
\]

Here \(s=1/2\) is a centered collision, \(s\to1\) means \(p\)'s front almost
purchased \(q\), and \(s\to0\) means it barely retained \(p\). This is both a
smooth interface location and a local terminal-pressure signal. It uses only
the winning arrival on each already accepted side.

The literal interface Hessian is also available without a second arrival.
If \(p_i\) and \(p_j\) are the one-sided arrival covectors, perturbing the
source-energy difference moves their interface by

\[
\delta n = \frac{\delta(w_i-w_j)}{\lVert p_i-p_j\rVert}.
\]

On a square grid, horizontal and vertical interface crossings sample
projected arc length. Summing both families yields the local Crofton weight

\[
h_{ij}(e)=
\frac{\rho_e}
{|p_{i,x}-p_{j,x}|+|p_{i,y}-p_{j,y}|}.
\]

This replaced the earlier, incorrect \(\rho/(2c_e)\) edge-cost surrogate. On
an 81x81 two-source Euclidean control the old conductance was 41.0, while the
covector jump gives 60.41. One undamped global solve reduced accepted-mass CV
from 19.77% to 0.93%; three exact remarches reached 0.046%.

## Site position is a transport-momentum equilibrium

For path-length action, a centroid is the wrong stationary condition. At
fixed transport weights, the envelope theorem gives

\[
\nabla_{s_i}E
=
\int_{C_i}\nabla_{s_i}d_M(s_i,x)\,\rho(x)\,dx.
\]

For a constant metric, with terminal arrival covector

\[
p_i(x)=\frac{M(x-s_i)}{d_M(s_i,x)},
\]

this becomes

\[
\nabla_{s_i}E=-\int_{C_i}p_i(x)\rho(x)\,dx.
\]

Thus a site is balanced when its accepted incoming transport momentum sums
to zero. It is a Riemannian geometric median, not a Euclidean barycenter and
not a PCA axis. No finite direction set is needed. In a varying metric the
terminal covector must be transported backward along its achieving
characteristic; merely summing endpoint covectors is an explicitly marked
local surrogate, not the exact gradient.

The BFFT tensor also already contains the population law and aspect law. If
its eigenvalues are \(\lambda_{\max},\lambda_{\min}\), the unit support
ellipse has

\[
a=\lambda_{\min}^{-1/2},\qquad
b=\lambda_{\max}^{-1/2},\qquad
\text{area}=\frac{\pi}{\sqrt{\det Q}},
\]

so the locally implied cell density and aspect ratio are

\[
\rho_Q=\frac{\sqrt{\det Q}}{\pi},\qquad
\text{aspect}=\sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}.
\]

No aspect-ratio controller is required. The regularized front metric
\(M=I+\beta Q\) inherits the same axes and realizes its aspect continuously
through the eikonal level sets.

## Honest limitations found

The first-label solver is not yet a replacement for the complete allocation
walk.

- Existing sites were equilibrated using the old two-label graph. Replacing
  only the final partition is therefore a mismatched control.
- The current allocator was calibrated using a soft two-label moment
  calculation. Its thresholds cannot simply be reused with first-arrival
  energy.
- FM-LBR assumes a sufficiently continuous metric. Raw BFFT tensors can
  change abruptly. A receiver-local long simplex can cross a topology change
  unless its segment is checked.
- Segment-consistency gating prevents unreachable pixels and invalid long
  jumps, but at 256px camera the continuous hard allocation improves RGB and
  texture while worsening cartoon MSE. This shows that the stopping-energy
  statistic and topology barrier must be recalibrated for causal arrival; it
  does not justify restoring runner-up ownership.
- Coins remains a negative control for a partition-only swap: its largely
  isotropic fine texture does not benefit from changing the final support
  while retaining graph-equilibrated sites.
- Equal support-measure mass is not a valid drop-in target for a population
  inherited from the older centroid/bifurcation allocator. On the 128px
  cameraman control the inherited cells have support-mass CV 0.272. A raw
  equal-mass Newton step crosses too many topology events and collapses the
  fit. Small damped steps lower mass CV but also slightly worsen the image
  objective. Equal resource quanta must be paired with a population emitted
  from that same resource law; it is not a post-hoc repair.

The full inverse Hessian has a second mismatch with the intended physics: in
one algebraic solve it communicates a pressure defect through an arbitrarily
long chain of cells, before any intervening front has moved. A local
simultaneous response follows directly from the same interface Hessian:

\[
\Delta w_i=\frac{m_i^\star-m_i}{2H_{ii}}.
\]

The factor \(1/2\) is exact for one interface because both incident source
energies move simultaneously. Neighbours then respond through the next exact
causal remarch. With a population quantized directly from the BFFT measure,
one such local response improved the 128px Pikachu control from 20.98 to
21.35 dB. On Coins, a smaller topology-preserving response improved the fit,
while the raw half-step could still starve cells. Positive cell mass is
therefore a real admissibility condition, not a numerical afterthought. This
agrees with damped-Newton theory for semi-discrete transport, but the
remaining task here is to express that admissibility as a germ/front energy
rather than a line search.

## Reverse characteristic resource

The continuous fast march now retains its achieving same-label Hopf--Lax
DAG. A reverse pass can transport every accepted pixel's resource toward its
source. When that reverse resource first enters a small shell around the
source, its local arrival covector is the initial-momentum estimate. This is
one \(O(|\Omega|)\) pass over the already causal acceptance order:

1. initialize every pixel with its accepted resource;
2. visit accepted pixels in reverse causal order;
3. split resource between the two achieving simplex parents;
4. stop at the source shell and reduce resource times covector by label.

No pixel traces a separate path. There is no direction histogram, PCA,
runner, candidate set, or sort. On a constant-metric symmetric square, a
two-pixel shell returns zero force to numerical precision; off-centre forces
agree with the direct terminal-covector integral as the shell expands.

The real-image diagnostic strongly rejects centroid motion. Using populations
emitted directly from \(\rho_Q\), the median angle between Euclidean centroid
displacement and reverse characteristic force was:

- cameraman: 49.3 degrees (p90 118.0);
- Pikachu: 71.3 degrees (p90 132.3);
- Coins: 67.6 degrees (p90 150.9).

Even terminal covector sum is not interchangeable with source momentum: the
median/p90 disagreement after reverse transport was 4.2/47.6 degrees on
cameraman, 8.6/49.6 on Pikachu, and 15.1/95.0 on Coins. Thus curved BFFT
geodesics materially change which way a germ should move. The remaining
position problem is the locally admissible step/Hessian, not the force
direction.

An attempted shortcut was rejected: differentiating only the four virtual
source-to-pixel seed links gives a four-direction position force. It is the
exact derivative of that snapped discrete injection, but not the continuum
geometry we want. The reverse characteristic shell removes that artificial
source stencil.

## Correct next implementation

The next allocation uses only the accepted label and its arrival action:

1. march every source-labelled front in one causal heap;
2. accept each pixel exactly once;
3. reduce accepted mass, centroid, tensor, and path-energy statistics by
   label;
4. measure where each front terminated against already accepted substrate;
5. update sites or source potentials from those conserved local reductions;
6. remarch only when the support geometry itself changes.

There is no runner field, ownership temperature, top-k population selection,
PCA, or explicit aspect-ratio parameter. The entire allocation descends under
the continuous BFFT transport energy.

## Primary references

- Jean-Marie Mirebeau, *Anisotropic Fast-Marching on Cartesian Grids Using
  Lattice Basis Reduction*, SIAM Journal on Numerical Analysis 52(4), 2014:
  https://arxiv.org/abs/1201.1546
- Max Budninskiy et al., *Optimal Voronoi Tessellations with Hessian-based
  Anisotropy*, ACM Transactions on Graphics 35(6), 2016:
  https://www.geometry.caltech.edu/pubs/BLdG%2B16.pdf
- Matt Elsey and Selim Esedoğlu, *Threshold Dynamics for Anisotropic Surface
  Energies*, 2016:
  https://dept.math.lsa.umich.edu/~esedoglu/Papers_Preprints/elsey_esedoglu_anisotropy.pdf
