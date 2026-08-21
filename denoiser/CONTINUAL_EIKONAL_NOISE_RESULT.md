# Continual eikonal noise transport: first checkpoint

## Status

This is the first patch-free fused 2-D recurrence. It is a live theoretical
checkpoint, not a promoted denoiser and not an FMMT repair. It already exposes
a useful edge/noise trade that FMMT hides, but it does not yet remove enough
diffuse or replacement noise.

The implementation is `continual_eikonal_noise_transport_2d.py`. The complete
128 x 128 Cameraman record and panels are under
`continual_eikonal_final_128/`.

## One evolving state

At numerical time `t`, retain

\[
  \mathcal S_t=(x_t,\mu_t,v_t,h_t,M_t,N_t,D_t).
\]

Here `x` is radiance; `mu`, `v`, and `h` are the centre, total mixture
variance, and outer radius of the residual-noise law; and `M` is the SPD
eikonal metric. `N` and `D` are the four full-band phase-correlation
numerators and denominators. The observation identity is exact at every step:

\[
  y=x_t+n_t,\qquad n_t=y-x_t.
\]

There is no noise-class branch, patch search, band, user fidelity constant, or
chosen smoothing duration.

Four paired covectors measure target-excluded noise witnesses:

\[
 \eta_d(p)=y(p)-\frac{x_t(p-d)+x_t(p+d)}2.
\]

Their median, complete dispersion, and outer radius define a bounded mixture
law. Reusing the same image does not divide its variance by an iteration
count. After eikonal transport, old and incoming laws are joined by the law of
total variance:

\[
 \mu^+=(1-\beta)\mu_P+\beta\mu_W,
\]

\[
 v^+=(1-\beta)\{v_P+(\mu_P-\mu^+)^2\}
      +\beta\{v_W+(\mu_W-\mu^+)^2\}.
\]

The radius is the smallest outer interval enclosing both transported sets.
This is the repaired FABADA principle: resolution evolves with the data, but
correlated revisits never masquerade as new independent measurements.

## V3 eikonal basis

Central radiance jets form a structure tensor. Isotropic noise uncertainty is
subtracted and the positive-semidefinite part is retained. That excess gives
the metric

\[
 M_t=I+\frac{[\nabla x_t\nabla x_t^T-\tfrac12v_tI]_+}
 {v_t+\sqrt{\det [\nabla x_t\nabla x_t^T-\tfrac12v_tI]_+}+\epsilon_{\rm mach}}.
\]

The V3 FM-LBR reduction produces an obtuse three-vector superbase at every
pixel. Selling decomposition expresses `inverse(M)` as nonnegative lattice
flux. After edge symmetrization this is a conservative Dirichlet Laplacian
`L_t`; its statistic operator `P_t = I - L_t / max_degree` is nonnegative,
row-stochastic, column-stochastic, and transports centres, second moments, and
radii together.

## Transported full-band phase

V3's phase is measured on the exact post-cartoon residual rather than on the
cartoon. The denoising analogue measures paired one-sided statistics on the
evolving residual centre `mu`, not on radiance. For horizontal, vertical, and
the two diagonal covectors:

\[
 N_d=(\mu-\mu_{-d})(\mu_{+d}-\mu),\qquad
 D_d=\frac{(\mu-\mu_{-d})^2+(\mu_{+d}-\mu)^2}{2}.
\]

`N` and `D` are transported separately through the geometry-only eikonal
Markov operator before forming `C_d=N_d/D_d`. A single integrable wave
covector obeys the unordered cosine-addition identity

\[
 \{C_+,C_-\}=\{C_xC_y+q,C_xC_y-q\},\qquad
 q=\sqrt{(1-C_x^2)(1-C_y^2)}.
\]

Squared transverse distance from that manifold is `Delta_phase`. Its ratio to
correlation energy supplies phase-incoherent noise authority. The first
placement on radiance was rejected: smoothness itself increased its apparent
coherence. Residual placement is the V3-consistent form.

## Back-to-Basics descent and the joint Z contractor

Witness agreement is the continuous authority

\[
 a_t=\frac{\mu_t^2}{\mu_t^2+v_t+\epsilon_{\rm mach}}.
\]

For frozen state statistics, radiance descends

\[
 E_t(x)=\frac12\|x-(y-a_t\mu_t)\|^2+\frac12x^TL_tx.
\]

Since `lambda_max(L_t) <= 2 max_degree(L_t)`, the majorizer step is determined
by the operator itself:

\[
 x^+=\Pi_{[\min y,\max y]}\left(x_t-
 \frac{(x_t-y+a_t\mu_t)+L_tx_t}{1+2\max\deg(L_t)}\right).
\]

Every accepted step measurably lowers this frozen action. It must also reduce
distance of the exact residual to the bounded witness set:

\[
 C_t(x)=\frac1{|\Omega|}\sum_p
 \left[\max\{|y(p)-x(p)-\mu_t(p)|-h_t(p),0\}\right]^2.
\]

Amplitude membership alone is insufficient: clean structure can fit a broad
noise interval. The final contractor adds the scale-dual value/first-jet
energy of coherent structure that the proposal attempts to place in the exact
residual. With `r=y-x`, coherent fraction `gamma`, and directional Dirichlet
density `D(r)`:

\[
 \mathcal C_t(x)=C_t(x)+
 \sqrt{\langle r^2\gamma\rangle\,\langle D(r)\gamma\rangle}.
\]

The geometric mean is the parameter-free Sasaki balance of value and first
jet. Both terms have intensity-squared units, so there is no mixing parameter.
A step must lower `E_t` and `mathcal C_t`. This is a contractor, not a
likelihood: membership never increases credibility, and coherent residual
phase is explicitly charged as structure loss.

## M4 result

Metrics below are MSE / SSIM / strong-edge projected retention / tripod
projected retention. FMMT is a rejected matched control.

| corruption | continual eikonal | FMMT control | accepted steps |
|---|---:|---:|---:|
| clean | 0 / 1 / 1 / 1 | .000598 / .9337 / .857 / .761 | 0 |
| uniform .12 | .002104 / .5990 / .840 / .767 | .002076 / .7009 / .729 / .574 | 1 |
| Gaussian .12 | .005232 / .4148 / .752 / .669 | .003046 / .6497 / .618 / .475 | 2 |
| Laplace .10 | .005799 / .4044 / .778 / .691 | .002698 / .6890 / .678 / .525 | 2 |
| replacement .15 | .008458 / .3956 / .747 / .672 | .002188 / .8755 / .777 / .667 | 1 |
| salt-pepper .15 | .015445 / .2922 / .779 / .699 | .002876 / .8839 / .820 / .720 | 1 |
| mixed .12/.15 | .010703 / .3286 / .729 / .674 | .003160 / .6369 / .660 / .531 | 1 |

The first decisive positive result on this 128-pixel control is exact
clean-image identity: the phase contractor rejects the first proposed motion,
whereas the amplitude-only law incorrectly moved once. This is not a universal
clean-image theorem. At 32--64 pixels, some clean Cameraman, tapered-hair, and
geometric-interface sources still move once; chirps, line drawing, and
multiscale blobs are fixed at 64 pixels. The second positive result is the
additive/mixed structure trade. Under
uniform noise the MSE is close to FMMT while tripod retention rises from
0.574 to 0.767. Under mixed corruption the current output is still visibly
noisy, but the tripod remains a connected object rather than being replaced
by smooth ground. Mixed hair-edge retention is 0.730 versus 0.710 for FMMT.
The negative result is equally clear: the recurrence does
not yet identify enough diffuse noise, and FMMT remains much better on sparse
replacement cleanup.

The matched six-source/nine-corruption gate at 32 pixels gives aggregate
MSE/SSIM/edge retention `0.00814/0.5576/0.6904`, versus
`0.00649/0.7245/0.5607` for FMMT and `0.01880/0.4553/0.9105` for the raw
observation. Thus the phase law's structure/noise trade generalizes, but its
advantage is not uniform: it beats FMMT MSE on line drawings and retains more
edge energy on four of six sources, while FMMT remains better on small
Cameraman and tapered hair. The four-covector discretization is measurably
resolution dependent.

## Rejected shared-basis experiment

Using the evolving Selling superbase itself to generate both noise witnesses
and flux looked more unified but was false. The transport chart manufactured
the law that certified its next move; clean Cameraman continued to contract
its self-generated residual while losing tripod energy. An independent
covector chart also failed when its mere compatibility was treated as positive
permission. These are concrete versions of the MRA confirmation pitfall and
the ZMF warning: survival is not evidence of truth.

The retained separation is principled. Paired covectors observe; the eikonal
operator transports. The next unification must couple them through transported
phase sufficient statistics, not let either role certify itself.

## Performance and native route

At 128 x 128 the phase form takes about 0.27--0.50 seconds after warmup. The phase and
radiance operators currently repeat the same Selling reduction and sparse
COO/CSR construction, so this timing is intentionally unoptimized. Sparse
graph construction and repeated Python allocation dominate. A useful native
syscall should fuse:

1. tensor positive-part and metric reduction;
2. local Selling edge emission, symmetric degree, and matrix-free Laplacian;
3. radiance, centre, second-moment, and radius transport in one raster pass;
4. frozen-action, bounded-set, and coherent-residual reductions;
5. luminance/RGB or Lab joint statistic transport without channelwise graphs.

The algorithm should not be native-optimized yet. The representation is
linear-memory and suitable for C++, but its clean fixed-point behavior remains
resolution/source dependent and it is still too conservative on unknown
diffuse noise.

## Next experiment

The first cell-free phase law now exists. Its four covectors are still a
crystalline seed, and the eikonal statistic graph is redundantly materialized.
The next theoretical experiment is a continuous projective-circle phase law:

- transport correlation numerators/denominators over the same local FM-LBR
  simplex rather than four raster covectors;
- carry uncertainty over the wave covector itself as a bounded mixture;
- require agreement between phase transport and radiance transport without
  letting either chart validate itself;
- test whether that added angular population authorizes more diffuse-noise
  removal while retaining the exact clean fixed point.

The current phase law already distinguishes:

- incoherent residual energy that may enter the noise law;
- coherent oscillatory phase that must remain radiance;
- uncertain transport orientation that remains a bounded mixture rather than
  collapsing to one metric.

The immediate engineering task remains deferred: share one native Selling
graph across phase, statistics, and radiance only after the continuous-circle
law survives the next gate.

## Pure-averaging hierarchy follow-up

The proposed reversal was implemented and tested: positive local averaging is
the radiance primitive, the Selling eikonal Laplacian is only its generator,
and transported FABADA-style statistics define a measure over the entire
smoothing path.  No screened radiance solve is used.  The operator invariants
hold and additive-noise edge retention improves, but the current posterior
path measure under-denoises and moves clean inputs.  It therefore does not
replace this screened checkpoint.  The full equations, matched 128-pixel and
54-case results, and the posterior diagnosis are in
`FABADA_EIKONAL_AVERAGING_RESULT.md`.

That follow-up now has a second stage.  The blurred-depth barycenter was
replaced by a transported zero/noise residual mixture whose posterior mean is
an operator-split source term on the positive averaging flow.  It improves the
uniform/Gaussian structure-error trade, while falsifying global scalar packet
mass as a likelihood and exposing sparse edge-flux allocation as the next
problem.  See `TRANSPORTED_RESIDUAL_POSTERIOR_RESULT.md`.
