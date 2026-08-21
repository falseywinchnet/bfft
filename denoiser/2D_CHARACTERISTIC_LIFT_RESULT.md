# The first 2-D characteristic lift: useful failure

## Status

The minimal 2-D lift is **not** the fused denoiser and is not exposed in the
GUI. It is a controlled falsification of the simplest extension of the
successful full-scale 1-D law. It reaches an intrinsic covariance equilibrium,
reduces observation error without mean or range collapse, and wins meaningful
subfamilies. It nevertheless loses to integrated FMMT on aggregate MSE, SSIM,
and edge retention. That is enough evidence to reject promotion and precise
enough evidence to identify the next state variable.

The complete 108-case M4 record is `2d_denoiser_battery.json`.

## Executed lift

For each primitive lattice tangent

\[
 d\in\{(0,1),(1,0),(1,1),(1,-1)\},
 \qquad 1\le s\le s_{\max}(d),
\]

three first-jet predictions compete: opposite interpolation and the two
one-sided affine extrapolations. For prediction \(p_{d,s,k}\), path variation
\(v_{d,s,k}\), and path-averaged absolute prediction action
\(a_{d,s,k}\), its conductance is

\[
 g_{d,s,k}(x)=
 \frac{1}{s\,[a_{d,s,k}(x)+v_{d,s,k}(x)]}.
\]

The seed reads out the full directional/scale marginal as

\[
 T[y](x)=
 \frac{\sum_{d,s,k}g_{d,s,k}(x)p_{d,s,k}(x)}
      {\sum_{d,s,k}g_{d,s,k}(x)}.
\]

Residual continuation is admitted only by the 1-D candidate's debiased
covariance equilibrium. With \(r=y-z\) and \(q=T[r]\),

\[
 c=\mathbb E[rq],\qquad
 \nu=\frac{\operatorname{Var}(rq)}{|\Omega|},\qquad
 e_+=(c^2-\nu)_+,
\]

and the transported amount is

\[
 \alpha=
 \min\!\left(1,\frac{c}{\mathbb E[q^2]}\frac{e_+}{c^2}\right).
\]

Thus the run duration is not set. Every battery case stopped when \(e_+=0\):
the mean number of accepted continuations was `3.287`, the maximum was `7`,
and none reached the numerical ceiling.

## Decisive M4 gate

Six structures were crossed with nine corruption conditions and two seeds at
96 square: Cameraman, tapered hair, geometric interfaces, woven chirps, a line
drawing, and multiscale blobs; additive uniform, Gaussian, Laplace,
multiplicative, random replacement, salt-and-pepper, and mixed replacement plus
uniform corruption were all represented.

| Method | MSE | SSIM | variance | central range | edge retention |
|---|---:|---:|---:|---:|---:|
| four-direction theoretical seed | 0.003724 | 0.7044 | 0.8069 | 0.9061 | 0.5305 |
| integrated FMMT checkpoint | **0.003495** | **0.7744** | **0.8329** | **0.9107** | **0.6694** |

The seed won 28 of 108 cases by MSE and 30 by SSIM; FMMT won 53 and 59.
It passed the equilibrium, error-reduction, half-variance, and half-range gates,
but failed both matched-FMMT gates.

The aggregate hides the most useful result:

| Structure | seed MSE / SSIM / edge | FMMT MSE / SSIM / edge |
|---|---:|---:|
| Cameraman | 0.003490 / 0.630 / 0.553 | **0.002009 / 0.778 / 0.773** |
| tapered hair | 0.002515 / 0.630 / 0.392 | **0.001113 / 0.802 / 0.743** |
| geometric interfaces | 0.003113 / 0.639 / 0.469 | **0.001281 / 0.816 / 0.857** |
| woven chirps | **0.001498 / 0.859 / 0.695** | 0.002467 / 0.721 / 0.478 |
| line drawing | **0.009959 / 0.790 / 0.632** | 0.013048 / 0.738 / 0.528 |
| multiscale blobs | 0.001766 / 0.679 / 0.441 | **0.001052 / 0.791 / 0.638** |

The seed also beats FMMT in MSE under each diffuse additive family and under
multiplicative corruption. Under heavy replacement its failure is decisive:
at replacement density 0.25 it scores MSE `0.006781`, SSIM `0.570`, and edge
retention `0.472`, versus FMMT's `0.004584`, `0.828`, and `0.644`.

This is not ordinary mean collapse. Aggregate variance retention is 0.807,
central-range retention is 0.906, and mean bias is -0.00225. The failure is
selective: relations distributed throughout an oscillatory field survive,
while sparse interface ancestry is diluted by an image-global evidence test.

## Rejected repairs

These trials are preserved as negative reasoning, not hidden tuning history:

1. Increasing the angular stencil improved the tapered-hair edge response but
   materially changed the answer between angular resolutions. The result was
   not converging to a continuum tangent integral; it was choosing a richer
   crystalline catalogue.
2. A conductance-weighted median preserved impulsive-replacement edges and
   range, but left diffuse additive noise largely intact. Selecting between a
   mean and median would reintroduce a human noise taxonomy.
3. Interpolating Wasserstein-1 and Wasserstein-2 readouts by local
   participation only moved along the same replacement-versus-additive
   tradeoff; it did not create missing ancestry.
4. Per-characteristic covariance with a weight-derived “effective population”
   continued spuriously. The characteristic paths are dependent, so their
   weights cannot manufacture a sample count. Only causal parentage can say
   which evidence is genuinely independent.

## Diagnosis: the missing coordinate is causal ancestry

The four-direction seed averages predictions first and asks a global question
afterward. A hair edge occupies little area, so its covariance contribution is
small even when its local transported lineage is coherent. Woven chirps and
line drawings repeat their relations over many locations, so the same global
test recognizes them. This explains both the success and the failure with one
mechanism.

Adding edge bands, corruption labels, local thresholds, selected windows, or
more directions would fit the screen without solving the problem. The next
object must instead carry the V3 eikonal solver's causal parent fractions into
the joint signal/residual/jet measure. Effective population then becomes the
mass of distinct transported ancestry reaching a point, not a proxy computed
from local weights.

The next executable form should:

1. emit observation-excluded `(z,r,j)` particles from the continuous tangent
   sphere rather than four named directions;
2. march them under a determinant-one eikonal metric while preserving causal
   parent fractions;
3. parallel-transport jets before comparing predictive laws;
4. compute horizontal Wasserstein volume only over distinct causal ancestry;
5. admit residual transport only when the single causal action decreases;
6. reproduce the diffuse/oscillatory wins without sacrificing hair,
   interfaces, or heavy replacement.

Until those conditions pass together, integrated FMMT remains the image
control and the 2-D theoretical seed remains deliberately absent from the GUI.

## First causal kernel

`causal_ancestry.py` now consumes the actual V3 streams
`parent_first`, `parent_second`, `parent_fraction`, and `acceptance_order`. If
\(A_p\) and \(A_q\) are complete source laws at an accepted parent simplex,
it transports

\[
 A_x=(1-t)A_p+tA_q,
 \qquad
 N_2(x)=\left(\sum_i A_x(i)^2\right)^{-1}.
\]

Because the full source vectors are retained, shared ancestry is counted once;
it cannot be mistaken for two independent paths. Tests feed this kernel the
production continuous-eikonal parent stream and verify causal order, mass
conservation, exact barycentric transport, and overlap-aware participation.

That integration test also locates the next architectural change. Production
V3 permits a simplex only between equal hard owners, so transported owner
ancestry correctly remains one-hot. Denoising population must be measured
before that hard partition: observation-excluded roots need a shared transport
label while their distinct identities remain coordinates of \(A\).
