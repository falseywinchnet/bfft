# Pure averaging under the evolving eikonal metric

## Question

The screened phase-eikonal estimator made the screened equation the radiance
update.  This experiment reverses that hierarchy.  The primitive operation is
nearest-neighbour averaging on the current metric graph; the eikonal
Laplacian is its generator, and FABADA-style transported statistics define a
posterior measure over smoothing depth.  No screened radiance solve occurs.

The question was whether that cleaner order gives a better denoiser.

## Operator

At iteration `t`, the evolving structure/noise metric is reduced into a
positive Selling graph with symmetric conductances `c_ij` and Laplacian
`L_t`.  The metric-local averaging step is

\[
 A_t=I-\Delta t_tL_t,\qquad
 \Delta t_t={1\over 2\max_i\sum_j c_{ij}},\qquad
 u_{t+1}=A_tu_t.
\]

The Gershgorin bound makes `A_t` nonnegative, symmetric, and doubly
stochastic.  Consequently constants are fixed, the observed range is
preserved, and the frozen-metric Dirichlet action cannot increase.  “Nearest
neighbour” here means neighbours in the locally Selling-reduced eikonal
lattice, not necessarily only the four adjacent raster pixels.

The same `A_t` transports the noise centre, total second moment, outer radius,
and residual phase sufficient statistics.  A fresh directional witness is
fused over the exact unit-rate interval fraction

\[
 q_t=1-e^{-\Delta t_t}.
\]

The local continuation mass is contracted by bounded-set violation and the
coherent value/jet action.  Its present posterior density is

\[
 \rho_t(p)=s_t(p)
 {\mu_t(p)^2\over \mu_t(p)^2+v_t(p)}\,
 \eta_t(p),
\]

where `s` is surviving depth mass and `eta` is residual phase-noise authority.
The output is the pointwise trapezoidal barycenter of the averaging path plus
an explicit no-transport atom:

\[
 \hat x(p)=
 {m_0(p)y(p)+\int \rho_t(p)u_t(p)\,dt
  \over m_0(p)+\int\rho_t(p)\,dt},
 \qquad m_0=(1-\rho_0)_+.
\]

Every accepted readout must also lower the existing joint amplitude/phase
contractor.  Numerical iteration count therefore does not serve as a denoise
setting.

## Result

It does **not** work better as a general denoiser with this posterior density.
It is a sharper, more conservative structural path, but it leaves too much
diffuse noise and does not retain universal clean identity.

The focused 128-pixel Cameraman metrics are MSE / SSIM / strong-edge retention
/ tripod retention:

| corruption | pure averaging | screened phase-eikonal | FMMT control |
|---|---:|---:|---:|
| clean | .000046 / .9970 / .950 / .937 | 0 / 1 / 1 / 1 | .000598 / .9337 / .857 / .761 |
| uniform .12 | .003159 / .4987 / .917 / .894 | .002104 / .5990 / .840 / .767 | .002076 / .7009 / .729 / .574 |
| Gaussian .12 | .007644 / .3535 / .840 / .812 | .005232 / .4148 / .752 / .669 | .003046 / .6497 / .618 / .475 |
| Laplace .10 | .008429 / .3467 / .851 / .812 | .005799 / .4044 / .778 / .691 | .002698 / .6890 / .678 / .525 |
| replacement .15 | .009315 / .3841 / .710 / .658 | .008458 / .3956 / .747 / .672 | .002188 / .8755 / .777 / .667 |
| salt-pepper .15 | .016614 / .2859 / .744 / .679 | .015445 / .2922 / .779 / .699 | .002876 / .8839 / .820 / .720 |
| mixed .12/.15 | .012381 / .3062 / .698 / .662 | .010703 / .3286 / .729 / .674 | .003160 / .6369 / .660 / .531 |

The pure path's additive-noise edge advantage is real: under uniform noise it
retains `0.917` of strong-edge projection and `0.894` of tripod projection,
versus `0.840/0.767` for the screened form.  The corresponding image and
MSE show why edge retention alone cannot certify recovery: visible noise is
being retained along with structure.

On the six-source, nine-corruption, 32-pixel gate:

| method | MSE | SSIM | variance ratio | central-range ratio | edge retention |
|---|---:|---:|---:|---:|---:|
| pure averaging | .00999 | .5362 | 1.0446 | 1.0259 | .7349 |
| screened phase-eikonal | .00814 | .5576 | .9535 | .9870 | .6904 |
| FMMT control | .00649 | .7245 | .7632 | .8801 | .5607 |
| observation | .01880 | .4553 | 1.5532 | 1.2306 | .9105 |

Pure averaging beats screened MSE in 10 of 54 cases and SSIM in 11.  Its most
interesting regime is woven chirps: aggregate noisy SSIM/edge retention are
`.6019/.6811`, versus `.5885/.5768` for screened transport and
`.5514/.3570` for FMMT.  This supports the operator as a representation of
structure-preserving evolution, not as the current final estimator.

## Diagnosis

The failed part is not positivity, locality, or descent.  Those invariants all
hold.  The failed part is the inference from transported uncertainty to
posterior smoothing depth:

- multiplying survival, centre agreement, and phase-noise authority makes
  the depth density collapse before diffuse noise is removed;
- the no-transport complement protects clean structure but is not strong
  enough to make every clean source an exact fixed point;
- a barycenter of radiance-only averages cannot express the estimated noise
  centre correction that helped the screened estimator;
- high edge retention can mean either recovered geometry or preserved noise,
  so it cannot authorize continuation by itself.

The screened solve now has a clearer interpretation: it is not necessarily
the primitive smoothing mechanic.  It may be an effective posterior
resolvent that performs the noise-centre correction missing from a naive path
barycenter.

The first such correction has now been implemented.  Rather than weighting
blurred endpoints, it transports a complete zero/noise residual mixture and
uses its mean as a source term along the positive averaging flow.  It improves
the additive-noise trade but does not yet solve sparse spatial allocation; see
`TRANSPORTED_RESIDUAL_POSTERIOR_RESULT.md`.

## Decision

Retain the pure averaging implementation as a falsifiable operator oracle and
texture-preserving branch.  Do not promote it to the Dear PyGui interface and
do not native-optimize it yet.  The next experiment should keep `A_t` as the
primitive but derive the path measure and readout jointly from residual-law
risk, including a transported noise-centre correction.  That new posterior
must recover exact clean identity and beat the screened form on MSE without
giving back its edge advantage.

Reproducible artifacts:

- `fabada_eikonal_compare_128/results.json` and its focused panels;
- `fabada_eikonal_six_source_32.json` for the 54-case gate;
- `test_continual_fabada_eikonal_2d.py` for positivity, conservation,
  Dirichlet descent, constants, and non-screened trajectory-readout checks.
