# Direct compressed observation from eikonal support

## Correction

The virtual observer is not a scene-lensing or deblurring model. Noise is not
an aberration variable. The observer is a finite explanatory instrument: what
its support-derived measurements can represent belongs to the current scene
explanation; what is orthogonal to those measurements remains unresolved.

## First operator

The one-pass operator in `compressed_eikonal_observer_2d.py` is:

1. measure the V3 structural precision/support density once;
2. quantize that measure directly into germs, with no candidate ranking;
3. march their continuous eikonal first-arrival partition in the measured
   precision tensor itself;
4. let each transported cell measure the affine moments `(1,x,y)`;
5. solve every three-coordinate measurement directly;
6. retain both the explained field and the exact unexplained residual.

For the sparse support-indexed basis `B_S`,

\[
e_S=B_S(B_S^TB_S)^+B_S^Ty,
\qquad r_S=y-e_S.
\]

This gives the certificate

\[
B_S^Tr_S=0.
\]

The current observer literally has no remaining measurement that explains
`r_S`. A new outer pass remeasures support on `r_S`; it does not optimize the
old coefficients.

## Dimension-corrected explanation yield

A representation with more degrees of freedom must not be rewarded merely
for capturing more energy. The probe reports

\[
Y_S=
\frac{\|e_S-\bar y\|^2/(\operatorname{rank}B_S-1)}
     {\|y-\bar y\|^2/(N-1)}.
\]

`Y_S=1` is the isotropic per-degree reference. The current stopping probe
accepts a new observation only while `Y_S>1`. This contains no denoising
strength, noise-family label, or fitted variance.

## 32x32 gate

| scene | condition | rank / pixels | first yield | accepted observations | first MSE |
|---|---|---:|---:|---:|---:|
| Cameraman | clean | 0.170 | 5.524 | 3 | 0.00501 |
| Cameraman | mixed replacement + uniform 0.25 | 0.185 | 3.044 | 1 | 0.01732 |
| tapered hair | clean | 0.103 | 8.757 | 3 | 0.00260 |
| tapered hair | mixed replacement + uniform 0.25 | 0.158 | 2.325 | 1 | 0.01279 |
| woven chirps | clean | 0.199 | 3.059 | 1 | 0.00455 |
| woven chirps | mixed replacement + uniform 0.25 | 0.167 | 1.464 | 1 | 0.01093 |
| zero-mean Gaussian null | null | 0.170 | 0.818 | 0 | -- |
| uniform null | null | 0.185 | 1.100 | 1 | -- |

Repeated clean observations recover detail rather than merely smoothing it.
Cameraman edge retention rises from `0.631` after the first explanation to
`0.806` after three; tapered hair rises from `0.571` to `0.757`. Under mixed
noise the global second-pass yield falls below one and stops before those
details are recovered.

## Disjoint cross-measure

The self-selection problem is now separated from the coefficient estimate.
Two checkerboard charts are reconstructed from disjoint original samples.
Chart A emits an eikonal support while chart B measures coefficients on that
support, and then their roles reverse.  For the two projected coordinates
`a_i,b_i` in support cell `i`,

\[
C_i=\langle a_i,b_i\rangle,
\qquad
M_i=\left\|\frac{a_i+b_i}{2}\right\|^2.
\]

The first transported prior solves

\[
(\operatorname{diag}M+L_M)g=[C]_+.
\]

`L_M` is the positive Laplacian of the eikonal-cell interface graph.  Its
conductance is interface length times the harmonic mean of measured-action
density.  This transports statistics, not pixel values.  The matrix is an
M-matrix, so `0 <= g <= 1` without a gain threshold or smoothing strength.

On the mixed 32x32 battery this reduced MSE from `0.01279` to `0.00990` on
tapered hair and from `0.01093` to `0.00799` on woven chirps.  Gaussian-null
false explanation fell from `0.14019` for self-measurement to `0.01792`.
The remaining defect is visible: taking `[C]_+` independently makes chance
positive cells accumulate, and the positive part is analytically jagged.

## Signed phase transport

Cross-chart equality is not the definition of structure.  An alternating
structure can be antisymmetric across the two disjoint charts.  The next
operator therefore uses

\[
m_i=\frac{a_i+b_i}{2},\qquad d_i=\frac{a_i-b_i}{2},
\]
\[
C_i=\langle a_i,b_i\rangle,\qquad
T_i=\|m_i\|^2+\|d_i\|^2,
\]

and transports a signed phase coordinate:

\[
(\operatorname{diag}T+L_T)h=C.
\]

Since `|C_i| <= T_i`, the two-sided maximum principle gives `-1 <= h <= 1`.
There is no positive-part projection.  A smooth cubic readout uses

\[
w_+(h)=\tfrac12 h^2(1+h),\qquad
w_-(h)=\tfrac12 h^2(1-h)
\]

for symmetric and antisymmetric phase.  Both weights are nonnegative and
vanish smoothly when phase is unorganized.

The useful global statistic is the phase-order Rayleigh quotient

\[
\kappa=\frac{\mathbb E[h]^2}{\mathbb E[h^2]}\in[0,1].
\]

It is near one when transported phase agrees and collapses when local phase
cancels.  Contracting the stronger positive-cross prior by `kappa` gives a
conservative scene seed.  This is not a fitted acceptance threshold.

| scene | condition | transported-cross MSE | phase order | ordered-cross MSE |
|---|---|---:|---:|---:|
| Cameraman | clean | 0.00471 | 1.000 | 0.00471 |
| Cameraman | mixed | 0.01789 | 0.803 | 0.02430 |
| tapered hair | clean | 0.00267 | 0.998 | 0.00267 |
| tapered hair | mixed | 0.00990 | 0.485 | 0.01532 |
| woven chirps | clean | 0.00478 | 0.985 | 0.00479 |
| woven chirps | mixed | 0.00799 | 0.453 | 0.00934 |
| Gaussian null | null | 0.01792 | 0.011 | 0.00022 |
| uniform null | null | 0.00274 | 0.028 | 0.00006 |

The distinction is important.  The positive-cross result remains the better
empirical denoiser on mixed damage.  The ordered-cross result is the much more
defensible prior: it gives up unsupported amplitude and almost annihilates
null self-certification.  It is the seed to which a later residual posterior
may transport; it is not yet the final image.

Repeated ordered observations are also qualitatively different from the old
pursuit.  After 32 outer observations, Gaussian-null MSE remains `0.00024`
and uniform-null MSE `0.00013`, instead of rapidly absorbing the null.  Clean
detail continues to return.  A physical terminal law has not yet been proved,
so this repeated form remains a convergence experiment rather than an API.

## Complete parity-covector phase union

The two-chart posterior failed the 96x96 gate because genuine structure in
one region granted authority to raw residual samples elsewhere.  Simply
averaging more phase signs also failed: horizontal, vertical, and diagonal
parity phases are different gauge coordinates, and their signed mean erased
clean woven structure.

The corrected operator uses the complete nonzero covector set of the 2-D
lattice modulo two,

\[
q\in\{(1,0),(0,1),(1,1)\}.
\]

Each covector has its own reciprocal cross prior and signed phase order
`kappa_q`.  The cross priors are barycentered by their order mass.  The phase
sections are joined without identifying their signs:

\[
\kappa_{\cup}=1-\prod_q(1-\kappa_q).
\]

This is a smooth union, not a hard maximum.  Any coherent covector section can
survive, so the clean woven falsification disappears.  On the 32x32 mixed
battery the union prior gives Cameraman/hair/woven MSEs `0.01806`, `0.01069`,
and `0.00864`; Gaussian and uniform nulls remain `0.00031` and `0.000063`.

## Support-remeasured residual posterior

The decisive correction is that original-scene support must never authorize
raw residual noise.  Let the complete-covector observer be `O`.  The system
now performs

\[
s_0=O(y),\qquad r_0=y-s_0,\qquad s_1=O(r_0).
\]

Only the newly observed residual component `s_1` enters the Selling
posterior:

\[
(I+L(s_0,r_0^2))z=s_1,\qquad x=s_0+z.
\]

Thus the screened positive Selling stencil regularizes something that has
already survived a new compressed observation.  It never smooths all of
`r_0` and hopes that noise will disappear.  The resolvent certificate is now
`||z|| <= ||s_1||`.

At 32x32 the posterior gives mixed Cameraman/hair/woven MSEs `0.01707`,
`0.00988`, and `0.00853`.  Gaussian and uniform nulls remain low at `0.00042`
and `0.000067`.

At 96x96 the first union prior has MSE `0.01264`, SSIM `0.460`, and edge
retention `0.397`.  Remeasuring and screening the residual component gives
MSE `0.01210`, SSIM `0.456`, and edge retention `0.414`.  This is a smaller
visual move than the rejected raw-residual posterior, but it improves the
tripod/detail balance without restoring the salt-and-replacement field.

## What is established

- V3 support can act as a sparse explanatory sensor rather than a segmenter.
- Its emitted representation is genuinely compressed on these scenes.
- A direct extraction, exact residual certificate, and repeated-support
  architecture exist without inner coefficient or geometry descent.
- Structured scenes concentrate measurement energy far more strongly than a
  Gaussian null.
- Causally disjoint charts remove most support self-certification.
- The screened eikonal Laplacian transports evidence while preserving a
  maximum principle.
- Signed phase transport supplies a smooth, scale-free order coordinate that
  distinguishes organized structure from phase-disordered nulls.
- The complete parity-covector union retains coherent woven phase without a
  selected direction or hard maximum.
- Remeasuring support on the unexplained field is materially safer than
  granting raw residuals the first scene's authority.

## What is not established

The original support and explanation are measured from the same samples.
Uniform noise producing `Y=1.10` remains the self-selection witness for that
baseline.  The reciprocal charts repair most of it, but their finite
checkerboard interpolation is a quadrature, not a proof of continuous phase
convergence.

The global baseline gate misses sparse supported detail after mixed
corruption.  The new cell-local cross statistic repairs that failure, but its
aggressive positive readout and the conservative phase-ordered readout still
bracket the desired posterior rather than determine it.  This is the
compressed-sensing analogue of the MRA pitfall/remedy: never let a fluctuation
both create its sensing support and certify its own explanation.

The four-sublattice disjoint experiment improved mixed tapered-hair SSIM but
damaged clean woven structure; fixed phase refinement can alias the very
support it is meant to certify.  It is therefore recorded as a falsified
endpoint, not promoted.

The next step is now the terminal law for additional outer observations.  A
fixed number of residual observations would merely hide a denoising-strength
setting.  The terminal state must instead follow from contraction of the
unexplained action or a dimension-corrected null law.  Covector refinement
beyond the complete mod-two fibre also remains a convergence obligation.  The
current result is intentionally not wired into the GUI or called a denoiser.

An ancestry-product probe was run but not promoted.  Multiplying every later
correction by the product of all preceding phase-union orders makes Gaussian
and uniform null corrections decay geometrically to machine zero.  It also
improves mixed Cameraman through roughly 25 observations.  However its
residual phase order then approaches one and the nonlinear support changes
produce small nonmonotone corrections rather than a proved terminal state.
This is evidence for transporting uncertainty about transport itself, but not
yet the required stopping theorem.
