# Analytic uplift: from certified support rule to transported support state

## Scope

FMMT is the denoiser under study. The repository's audio denoiser, exact
compressor, optimized Meyer descent, and allocation transports solve different
problems. They are examined here only to answer four questions:

1. What object counts as support?
2. How is that support propagated?
3. What conservation or exactness law constrains it?
4. What ends the evolution?

No operator is imported into FMMT merely because it worked elsewhere, and no
cross-domain score is presented as a quality comparison.

## The supplied checkpoint

Let the unchanged observation be \(y\) and the robust FMMT bootstrap be
\(x_0\). The supplied method measures cross-predictive coarse support and
fine-scale ancestry, forms an admitted support \(S\), and diffuses the coarse
bootstrap through

\[
  c^{k+1}_i=c^k_i+
  \Delta t\sum_{j\sim i}\frac{g_i+g_j}{2}(c^k_j-c^k_i),
  \qquad g=a(1-S).
\]

This is already much better than a final-image cleanup: the support decision
occurs before eikonal geometry and both FMMT measures subsequently inherit it.
The remaining analytic defect is that physical evolution is selected by
constants: scales 8 and 12, threshold ramps, authority bands, split scale 1.5,
time step 0.18, and 128 sweeps.

## Current continuous form

### Scale is integrated, not selected

Disjoint observation lanes produce independent scale-space estimates
\(y_{\ell,\sigma}\). At every physical scale \(\sigma\), their normalized
curvature and tangent evidence are separated into reproducibility and relative
amplitude:

\[
  r_\sigma(x)=
  \frac{|\bar\kappa_\sigma(x)|^2}
       {|\bar\kappa_\sigma(x)|^2+\operatorname{Var}_\ell\kappa_{\ell,\sigma}(x)},
\]

\[
  a_\sigma(x)=
  \frac{|\bar\kappa_\sigma(x)|^2}
       {|\bar\kappa_\sigma(x)|^2+
        \mathbb E_x|\bar\kappa_\sigma|^2}.
\]

The same construction is applied to the scale-normalized gradient vector.
Curvature and oriented edge evidence are joined by a smooth union, then
integrated uniformly in \(d\log\sigma\):

\[
  S(x)=1-\exp\left(
    \frac{1}{\log(\sigma_1/\sigma_0)}
    \int_{\sigma_0}^{\sigma_1}
    \log(1-e_\sigma(x))\,d\log\sigma
  \right).
\]

The implementation uses log-quadrature nodes. Their count is numerical
resolution, not a selected physical scale.

### Authority is measured without transition bands

Two dimensionless empirical quantities replace the fixed censor and remote
tail thresholds:

\[
  H(y)=-\frac{\sum_b p_b\log p_b}{\log B},\qquad
  P(r)=\frac{(\mathbb E|r|)^2}{\mathbb E r^2},\quad r=y-x_0.
\]

The witness authority is

\[
  a=\sqrt{H(y)P(r)}.
\]

Low occupied-state entropy weakens censored or degenerate observations;
low participation weakens residuals owned by a few remote atoms. Neither is a
noise-family classifier and neither contains an image-quality threshold.

### Evolution is stopped by transported action

Support controls a conservative mobility

\[
  g(x)=a(1-S(x))^2,
  \qquad \partial_t c=\nabla\cdot(g\nabla c).
\]

The observation supplies only the part of \(y-x_0\) that survives the same
scale continuum. Its unsupported energy defines a finite action budget:

\[
  B_y=\frac a2\int (1-S(x))
  \left|\frac{1}{\log(\sigma_1/\sigma_0)}
  \int G_\sigma*(y-x_0)\,d\log\sigma\right|^2 dx.
\]

The flow stops when its accumulated Dirichlet action reaches \(B_y\). The last
step is shortened to spend that budget exactly. Thus a maximum step count is a
failure guard, not the support horizon. Pairwise flux is antisymmetric, so
mass is conserved, and the CFL-resolved update satisfies a discrete maximum
principle.

The eikonal barrier receives the same support statement in a different role:

\[
  b(x)=1-a(1-S(x)).
\]

After this point the supplied FMMT signal/noise packet transport, additive
observation coupling, posterior mean, and entropy inertia are unchanged.

## What the other repository mechanisms teach about support

These are orthogonal mechanism audits, not a proposed hybrid.

| Work | Its support object | Its propagation/exactness law | What FMMT should learn | What must remain separate |
|---|---|---|---|---|
| `irfft_denoise_ab` | shaped, phase-carrying patches in a 2-D frame representation | exact analysis/synthesis; hard removal of Meyer scale layers | support must include phase/relational state, not magnitude alone | it is a 1-D audio representation experiment with known-noise scoring, not this image posterior |
| representation-residual codec | a small deterministic geometry used as a coordinate system for exact correction | source = decoded geometry + byte-exact residual | useful support should be representable as compact state; another support raster is duplication | coding rate and blind denoising risk are different objectives; the codec must reproduce noise exactly |
| optimized Meyer jump measure | an oriented Hodge jump measure plus routed oscillatory layer | exact cartoon + texture recomposition; fixed-cost spectral realization | edge support should be oriented and should preserve two-sided boundary state | Meyer separates BV/cartoon from G-norm texture; FMMT estimates a latent signal under an empirical residual law |
| continuous eikonal transport | causal first-arrival action in a locally reduced anisotropic metric | simplex Hopf-Lax updates, accepted mass, arrival covectors | the final FMMT graph should lose its polygonal eight-edge Wulff crystal | first-label cell allocation does not supply FMMT's two transported empirical measures |

The compressor's Cameraman result is especially relevant only at the level of
representation: a crude 12-cell geometry can be valuable because it orders
the residual, not because it visually approximates the image. For this
denoiser, support should likewise be judged by the posterior state it enables,
not by whether the support raster itself looks photographic.

## Hair-edge falsification

The current 2-D probe contains a smooth dark cap and three tapering oblique
strands against a low-complexity background. It measures both MSE and the
truth-gradient projection of the reconstruction.

The first agreement-only transport version failed: it reduced MSE while edge
retention fell from 0.410650 to 0.369728. A relative-amplitude correction then
over-smoothed the edge almost completely, exposing a second error: the full
pixel residual is not the support flow's available action. The retained form
uses oriented evidence and only scale-transportable residual action:

| state | MSE | edge retention |
|---|---:|---:|
| noisy observation | 0.0220191 | — |
| provisional chart | 0.00270794 | 0.410650 |
| retained support transport | 0.00270314 | 0.410627 |

The result is conservative and stable, but it does not restore edge response
already lost in the provisional chart. The Cameraman hair problem is therefore
still open.

### Matched full-FMMT result

The same quantized 128-square scene was run through all three FMMT support
paths on the M4 Mini. Everything after support birth is matched:

| FMMT support law | MSE | edge retention | time |
|---|---:|---:|---:|
| no certification | 0.00279218 | 0.436720 | 6.12 s |
| supplied integrated checkpoint | **0.00207545** | 0.436732 | 6.49 s |
| current continuous transport | 0.00271462 | **0.437520** | 6.34 s |

The continuous law is slightly better at retaining the tapered edge than both
matched alternatives, but it has not reproduced the checkpoint's distortion
gain. It is therefore an analytic research branch, not a promoted default.
The exact diagnostics and decoded images are in `results_fmmt_hair/`.

### Supplied Cameraman panel

The archive did not contain the original standalone noisy input. The first
panel of `website_cameraman_compare.png` was therefore recovered losslessly
from the montage crop as `assets/website_noisy_cameraman.png`; a 256-square
copy is used only for exploratory timing and visual comparison. Because the
montage has already resampled the source and no clean truth is supplied, no
distortion claim is made from this run.

At 256 square the supplied checkpoint admits essentially every fine pixel as
ancestry (`mean_fine_ancestry = 0.99999987`), so its support evolution becomes
the identity on this recovered panel. The continuous branch retains a graded
support density (`mean = 0.5463`) and spends a finite action budget, but the
three decoded outputs remain visually very close. This is evidence that the
actual pre-montage noisy file is needed before the Cameraman hair question can
be measured honestly. Artifacts are in `results_website/`.

## Next evolution before C++

1. Replace scalar edge evidence with a transported oriented line measure
   \(m=(\rho,t,\kappa)\): density, tangent, and curvature.
2. Evolve that measure by tangent continuation while charging curvature and
   cross-edge flux, so tapered hair remains hereditary even when its contrast
   weakens.
3. Replace the eight-neighbor FMMT graph with local reduced-basis simplex
   Hopf-Lax propagation, then devise a shared-front full-measure transport
   rather than importing a first-label allocator.
4. Replace packet ratio, local mass, likelihood floor, and entropy clips with
   transported concentration or discrepancy laws.
5. Measure stability under grid refinement and log-scale quadrature refinement.
6. Only then freeze a serializable state and port kernels to C++.

The eventual native representation should carry scale-integrated support,
oriented continuation state, authority, and transport budget—not a saved
support bitmap and not a list of test-selected branches.

## Representation-only acceleration checkpoint

Profiling after the first GUI integration showed that the scalar separable
histogram bootstrap, not the ordered fronts, owned almost all 128-square
runtime. `fmmt_certified.py` now advances the identical recurrence as full
vector packets. The scalar Numba form remains an oracle, and randomized tests
are bit-for-bit equal. Dijkstra anchor batches are selected from a reported
workspace budget; explicit batch comparisons differ only by floating-point
summation order.

On the recovered 256-square Cameraman input, continuous-support FMMT measured
`1.34 s` with adaptive batch 256 versus `1.96 s` with batch 16, with maximum
pixel difference `3.33e-16`. Earlier scalar-bootstrap runs were about
`25–26 s`. This changes representation and scheduling only; the empirical
measures, geometry, likelihood, posterior, and support law are unchanged.

The next representation pass retains that rule and accelerates the active GUI
form further. Front distance is converted to attenuation in place, signal and
residual packets share one matrix multiplication, recurrence transmissions are
reused across sweep order, histogram bins share one compiled box filter, and
the complementary support lanes share their linear Gaussian work. The GUI now
passes its displayed support field into FMMT instead of evaluating it twice.
On the current matched mixed-corruption 256-square record, median kernel time
is `1.205 s`; shared-support GUI latency is `1.215 s` versus `1.330 s` for the
duplicated path, with zero pixel difference. Exact measurements and the
remaining native-front target are in `2D_ACCELERATION.md` and
`2d_acceleration_m4.json`.

## Current fused-law update

The later joint-information experiment no longer uses FMMT support birth as
its candidate law. It transports complete `(z,j,r)` branch density through the
V3 Hopf--Lax parent DAG. The latest positive endpoint derives its collision
order from causal simplex dimension:

\[
\alpha_x=1+\frac{1}{(1-t_x)^2+t_x^2}
\]

for a two-parent update, with exact root and one-parent reductions. This
improves the fixed two-history HJ endpoint on MSE/SSIM/edge/variance in
`51/50/58/56` of the full 60 image cases. It is still below FMMT in aggregate
and is not GUI-promoted.

Four attractive 1-D endpoints were also rejected: bidirectional max-product,
split value/jet W1 descent, a determinant-one joint field action, and equal
left/right logarithmic pooling. Each lowers some derivative errors but
deflates structure. Their common failure shows that the next 1-D state must
transport residual ancestry between distinct observations; same-endpoint
residual comparison cancels the observation through `r=y-z`.
