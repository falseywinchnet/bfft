# Finite Meyer state preconditioning

This round uses only analytic sources authored for the experiment. No
photograph, inherited gallery image, or visually inferred truth contributes
to a quality claim.

## What the descent is discovering

The first native pass is linear. With the Bregman fields cold, its cartoon
side is

\[
  \widehat u_1(\omega)=H(\omega)\widehat f(\omega),\qquad
  H(\omega)=\frac{\lambda}{\lambda+2\lambda L(\omega)},
\]

where `L` is the positive symbol of the discrete negative Laplacian. The
first shrink only prepares pass two. Consequently pass one cannot allocate
two equal-frequency signals differently because one is a contour and the
other is an oscillatory interior.

The oracle ablation shows that the missing state is plural:

1. the cartoon ROF projector must discover its active structural flux;
2. the texture projector must discover a bounded transport field whose
   divergence is the texture.

Supplying exact texture before the cartoon solve does not complete the split:
the cold texture-side ROF survivor still absorbs it. Supplying exact cartoon
does not complete the split for the same reason. Both exact projector states
do.

The carrier atlas explains the amplitude and scale failure. A sinusoidal
texture needs transport magnitude proportional to `amplitude / frequency`.
Recovery deteriorates near

\[
  A / |k| \simeq \mu.
\]

That is the model's G-ball capacity, not arbitrary optimizer sluggishness.
Long-wave or high-amplitude authored texture may lie outside the selected
ball; no amount of descent can place all of it in `v` without changing the
model or `mu`.

This agrees with the transport interpretation in Gilles and Osher's
[Bregman implementation of Meyer's G-norm](https://arxiv.org/abs/2410.22777)
and with the BV-G examples in Gilles and Meyer's
[properties of BV-G decomposition models](https://arxiv.org/abs/2411.04456).
Gilles' [multiscale texture separation](https://arxiv.org/abs/2411.00894)
also supports the usefulness of a Littlewood--Paley separation, but the
fixed local band implementation tested here was not the best survivor.

## Falsified shortcuts

- **Structural flux alone:** greatly reduces contour leakage, but leaves the
  linear texture transfer almost unchanged. This is the earlier conditioned
  first pass.
- **Balanced dyadic bands:** restores texture amplitude but interprets the
  positive/negative lobes of edge halos as oscillation.
- **Direct Hodge projection of the source:** the source's low-frequency
  geometry dominates the minimum-energy lift and overloads almost every
  vector before texture can be distinguished.
- **Raw virtual diffusion tail:** recovers texture well but its contour
  leakage can make the output fall outside the G-ball. It is a segmentation
  heuristic, not a valid Meyer-family preconditioner.

## Surviving finite construction

The selected method has no outer descent and no runtime candidate scan.

1. Take `K=8` early linear cartoon steps in one spectral multiplication:

   \[
     u_K^{\rm virtual}=\mathcal F^{-1}(H^K\widehat f).
   \]

2. Build the same four-direction symmetric structural tail as the native
   conditioner. With gate `q`, form a texture seed

   \[
     s=(1-q)^8(f-u_K^{\rm virtual}).
   \]

3. Perform one structurally conditioned cartoon solve on `f-s`.

4. Let `v_tilde=f-u`. Hodge-lift it through one scalar Poisson potential:

   \[
     \phi=\Delta^\dagger(v_{\rm tilde}-\bar v_{\rm tilde}),\qquad
     p_0=\nabla\phi.
   \]

5. Before projection, use the lift's capacity tangent frame to take one
   deterministic divergence-free route. A fractional underloaded reservoir
   gives the overload somewhere to move, and one analytic right-Newton
   coefficient fixes the route magnitude.

   With `n=p0/|p0|`, `t=J n`, demand `d`, and transverse projector `P_T`,

   \[
     q=P_T(dn)=-J P_L(dt),\qquad \operatorname{div}q=0,
     \qquad p_r=p_0+\alpha q.
   \]

   Thus the existing tangent frame generates a stream-function route while
   the cheaper implementation applies the algebraically equivalent
   transverse radial projection.

6. Project every vector once onto the radius-`mu` disk and read out

   \[
     p=\operatorname{clip}_{\mu}(p_r),\qquad
     v=\operatorname{div}p,\qquad u=f-v.
   \]

The final `v` is constructively in the discrete G-ball because the algorithm
exhibits a field `p` with `|p(x)| <= mu` and `v=div(p)`. Recomposition is exact
to floating-point roundoff.

`K=8` is an exponent in one multiplier, not eight image sweeps. In native
code the Hodge lift uses one scalar inverse transform followed by forward
differences, rather than two inverse vector transforms.

## Native synthetic results at 256 x 256

All values below use `lambda=0.05`, `mu=40`, strength `1.5`, virtual power
`8`, and gate power `8`. Texture error and contour leakage are normalized by
the authored texture RMS.

| authored scene | method | texture gain | texture error | contour leakage |
|---|---:|---:|---:|---:|
| symmetric support | pass 1 | 0.545 | 0.455 | 0.618 |
| | finite virtual-transverse | 0.961 | 0.046 | 0.145 |
| | pass 64 | 1.000 | 0.058 | 0.642 |
| multiscale crossing | pass 1 | 0.290 | 0.756 | 0.799 |
| | finite virtual-transverse | 0.846 | 0.193 | 0.456 |
| | pass 64 | 0.931 | 0.096 | 0.716 |
| hard checker support | pass 1 | 0.705 | 0.316 | 0.479 |
| | finite virtual-transverse | 0.982 | 0.047 | 0.106 |
| | pass 64 | 1.000 | 0.043 | 0.498 |
| thin junction + texture | pass 1 | 0.427 | 0.652 | 1.107 |
| | finite virtual-transverse | 0.893 | 0.170 | 0.661 |
| | pass 64 | 0.985 | 0.095 | 1.298 |

Median native time with four worker lanes was about `2.2 ms`, compared with
roughly `22-25 ms` for 64 passes: approximately `10x` faster on these 256-square
sources. The ordinary first pass was about `0.30 ms` and the structural-only
conditioner about `1.20 ms`.

## Crossing-carrier outline and jump-response cancellation

The multiscale rig exposes a defect hidden by aggregate scores. Pixelwise
gating removes the candidate texture throughout a finite contour band. When
a real carrier crosses the circle or dark square, that carrier remains in
the cartoon as an annulus or outline. Pass 64 eventually transports it
through the front, but over-amplifies a coherent diagonal phase stripe.

The experimental `jump-cancelled` arm does not fill the gate or search for a
phase. It estimates a structural jump potential from the existing normal
frame,

\[
  \Delta s=\operatorname{div}\left[
    c(g)\left(1-\frac{\tau^2}{|\nabla f|^2}\right)_+\nabla f
  \right],
  \quad \tau=\frac{1}{2\eta},
  \qquad v_0=(I-H^K)(f-s).
\]

The half-threshold is the symmetric one-sided share of the existing Meyer
threshold. The nonnegative-garrote coefficient subtracts its energy without
the permanent amplitude bias of soft shrinkage. `c(g)` comes from a 256-bin
Otsu partition of the already-computed structural statistic: its lower anchor
is the between-class-variance optimum and its upper anchor is the measured
high-class mean. No truth-fitted confidence endpoints are used.

Subtracting the known virtual-tail response of `s` preserves the carrier's
measured phase. A single feed-forward residualization removes the initially
measured carrier from the accepted contour bonds before rebuilding the jump
once; repeating that map is a genuine nonlinear iteration and is deliberately
not used. The fixed construction is now the native two-product default. When
texture contrast dominates weak object contrast, the high response population
can itself be texture, so the explicit legacy split remains available for
regression and counterexample studies. Some coarse carrier also remains inside
the dark square.

## Discontinuity representation: measure, not annulus

A scalar first-pass Meyer texture is not the discontinuity itself. At pass
zero it is the fixed band-pass response

\[
  K_1 C=(I-H_v)(I-H_u)C,
\]

which renders a step as paired positive and negative lobes--a false annulus
around a closed object. If jumps are declared part of texture, the canonical
object is instead the singular part of the BV derivative,

\[
  D^j C=(C^+-C^-) n\,\mathcal H^1\!\restriction_{J_C}.
\]

This oriented bond measure records support, normal, and signed jump amplitude.
Hodge integration reconstructs its scalar jump potential without inventing a
halo. The multiscale synthetic source now authors its smooth absolutely
continuous composition and jump potential separately. The exact discrete jump
measure integrates back to that potential with maximum error `4.7e-12`.

After one feed-forward residualization, the experimental estimator's
longitudinal measure has relative flux error `0.119`, normal jump gain `0.983`,
and only `0.68%` of its energy off the authored boundary at 256 square. The raw
accepted bonds are not the representation: they contain a small nonintegrable
carrier component, and the longitudinal Hodge projection rejects it. The
one-pass scalar annulus remains in the audit only as a negative control.

The public truth and candidate are nevertheless only two products. With the
undetermined DC mode assigned to cartoon and `H_u` denoting the first Meyer
cartoon resolvent,

\[
  v^\star=(I-H_u)s_{\rm jump}+t_{\rm material},\qquad
  u^\star=C_{\rm smooth}+H_u s_{\rm jump}.
\]

Thus cartoon retains the circle and square with continuous transitions;
texture owns exactly the complementary energy that sharpens those transitions
into discontinuities. The jump measure is only the internal coordinate used to
obtain `s_jump`. On the 256-square two-product audit, ordinary pass one has
texture RMSE `6.34`, the jump-measure candidate `1.60`, and pass 64 `2.31`.
The ordinary arms produce a different transition profile rather than the
declared exact complement.

## Paired one-sided traces

The BV jump amplitude is the difference of two traces evaluated at the same
interface point. For an x-bond, the minimal affine-reproducing estimator is

\[
 [f]_{x+1/2}=\frac32(f_{x+1}-f_x)
              -\frac12(f_{x+2}-f_{x-1}).
\]

The weights are the unique solution that reproduces a unit step and
annihilates an affine field. Adding a third pair and requiring the cubic odd
moment to vanish gives `(5/3, -5/6, 1/6)`. With the authored boundary ridge
provided exactly, adjacent bonds have relative jump-flux error `0.0766`, the
affine trace `0.0395`, and the cubic trace `0.0166`. At fine-carrier crossings
the errors are respectively `0.1787`, `0.1056`, and `0.0448`.

This is not yet installed in the candidate. The current soft support is a
finite band rather than a codimension-one ridge; applying extrapolating traces
throughout that band amplifies carrier extrema. A fixed normal non-maximum
suppression recovers the entire authored ridge but admits about one false bond
for every true bond, so its improved flux estimate does not improve the final
two-product error. The next unresolved variable is therefore boundary-ridge
support, not trace amplitude.

## Honest limitation and next state variable

The tangent route reduces disk-readout distortion, but one transverse
projection is not the exact metric projection onto the G-ball. On the
multiscale crossing it reduces texture error from `0.201` to `0.193`; on the
thin junction source it reduces error from `0.184` to `0.170`. Part of the
remaining coarse-texture loss is correct--the authored truth itself
measurably exceeds `mu=40`--and part remains routing error.

An important negative result is that forcing the correction to follow the
source structure-tensor tangent makes it worse: a tangent-only vector cannot
reduce a normal capacity overload to first order. The useful frame is the
capacity frame of the lifted flux. The source structural gate should decide
*where* rerouting is allowed; it should not prescribe the correction vector.

## Code and reproduction

- Research harness: `experiments/meyer_preconditioning_research.py`
- Transverse-route harness: `experiments/meyer_transverse_route_research.py`
- Native benchmark: `experiments/meyer_native_conditioning_benchmark.py`
- Native API: `bfft_meyer_split_preconditioned`
- Python planned API: `MeyerPlan.split_preconditioned`
- Arbitrary-size front end: `bfft.meyer_split_preconditioned`
- Exact report: `experiments/out/meyer_native_conditioning/results.json`

The native implementation matches an independent NumPy spectral reference
within `3e-4`, is bit-identical at one and four worker lanes, and passes the
full CTest suite plus a direct C++17 `-Wall -Wextra -Wpedantic -Werror`
compile of `src/meyer.cpp`.

## Native promotion

The jump-measure route is implemented directly in `meyer::engine` and is now
the behavior of the existing `bfft_meyer_split`, `MeyerPlan.split`, and
`bfft.meyer_split` entry points on the full spectral solver. No caller change
is required. Its virtual depth is fixed at 8; a supplied legacy `passes`
argument is accepted for source compatibility but does not alter the spectral
default. `bfft_meyer_split_legacy`, `MeyerPlan.split_legacy`, and
`bfft.meyer_split_legacy` retain the former pass-controlled alternation.
Periodic FACR now carries the same oriented jump/Hodge construction using an
O(N) spatial realization of the four-direction structural gate, mean-zero
one-axis Poisson lifts, and a two-pole approximation of the virtual-depth
resolvent. Neumann FACR and the five-product scale ladder remain legacy.

At 512 square, the optimized native default measures about 9 ms on the test
machine, versus about 1.17 s for the 64-pass three-rung decomposition. The
native jump-measure result matches the independent NumPy research operator to
about `2.1e-12`, is bit-identical across tested thread counts, and recomposes
arbitrary-size inputs to about `2.8e-14` maximum absolute error. The complete
native CTest suite, 22 focused research/default tests, the Meyer implementation
audit, and the recomposition-effects audit pass after promotion.
