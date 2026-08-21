# Zero-component transport simmer

## Result

The complete residual moment is real but visually modest when it is used only
inside the HJ branch metric. On the displayed 128-pixel mixed-corruption
Cameraman gate, the centered and complete HJ outputs differ by RMS `.01426`
and by at most `.09652`; 64.4% of pixels differ by more than one 8-bit level.
Their displacement from the observation is larger (RMS `.06406` and `.07578`
respectively), but the complete-moment correction is only a reweighting of an
already-similar reconstruction. The user's “looks the same” diagnosis was
therefore substantially correct.

The 36-case, six-source, phase-two confirmation is even clearer: centered and
complete HJ observation RMS are `.12387` and `.12397`, with aggregate truth
MSE `.008030` and `.007937`. Complete moments improve the branch law, but do
not create a new visual regime.

An explicit zero/nonzero residual component creates a genuinely different
readout. Its best small-grid form is a direct terminal mixture: infer component
probability once, collide the nonzero HJ branch fibre once, and do not collide
the already-terminal component odds a second time. It improves the 18-case
16-pixel HJ screen and preserves considerably more edge energy, but it fails a
64-pixel mixed-corruption Cameraman gate by leaving granular residual noise.
It is not promoted.

The scale failure exposes the next missing state. The current `root_mass`
labels causal histories but its conditional residual laws are nearly
identical: measured between-root residual variance is generally `1e-8` to
`1e-35`, versus within-root variance around `1e-2` to `1e-1`. Uncertainty about
the transport map has already been marginalized away. A correct next law must
transport distinct maps/histories in the state before forming branch or
component marginals.

## Observation displacement

For estimate `x` and observation `y`, every gate now reports

\[
 d_{\rm obs}^2={1\over |\Omega|}\sum_i(x_i-y_i)^2,
 \qquad d_{\rm obs}=\sqrt{d_{\rm obs}^2},
\]

plus maximum displacement and the fraction moved by more than `1/255`. This
distinguishes a visually inert correction from a substantive reconstruction;
it is not a quality objective.

On the 128-pixel mixed Cameraman measurement:

| form | truth MSE | observation MSE | observation RMS |
|---|---:|---:|---:|
| observation | .027886 | 0 | 0 |
| centered-residual HJ | .013122 | .004103 | .06406 |
| complete-residual HJ | .011359 | .005742 | .07578 |
| FMMT control | .003160 | .022131 | .14877 |

## Explicit component law

Let the terminal branch fibre contain signal/residual atoms `(z_k,r_k)`, Haar
reference `h_k`, order-one HJ density `q_k`, and causal-simplex collision order
`c`. The numerical component is

\[
 \nu\in\{0,1\},\qquad
 \nu=0:r=0,z=y,\qquad \nu=1:(z,r)=(z_k,r_k).
\]

For residual mean `mu` and variance `v`, the tested coherent and complete
evidence laws are

\[
 \pi_{\rm mean}={\mu^2\over\mu^2+v},\qquad
 \pi_{\rm complete}={\mu^2+v\over\mu^2+2v}.
\]

Machine precision is used only for the zero denominator. The component base
measure is symmetric; `1/2` is representation mass for each hypothesis, not a
noise prior or fitted quality parameter.

The first experiment collided component odds and branch density together. It
was overconfident because `pi` was already a terminal component statistic. The
retained experimental readout instead uses

\[
 q_k^{(c)}={h_k\exp(c\log(q_k/h_k))\over
                 \sum_jh_j\exp(c\log(q_j/h_j))},
 \qquad
 \hat x=(1-\pi)y+\pi\sum_k q_k^{(c)}z_k.
\]

Thus component probability is used exactly once and branch collision is used
exactly once.

## Small-grid result

Across Cameraman, tapered hair, and woven chirps under clean, Gaussian,
replacement, mixed, Poisson, and row-correlated signal-dependent conditions:

| form | MSE | SSIM | variance ratio | edge | observation RMS |
|---|---:|---:|---:|---:|---:|
| complete-residual HJ | .007046 | .6325 | .8177 | .5247 | .1203 |
| recollided complete component | .006693 | .6546 | .8365 | .5584 | .1123 |
| direct complete terminal mixture | .006634 | .6576 | .9339 | .6477 | .0891 |
| FMMT control | **.005666** | **.6857** | .8030 | .5656 | .1116 |

The direct terminal mixture beats FMMT on Gaussian MSE (`.00376` versus
`.00392`) and on row-correlated SSIM/edge (`.5445/.6991` versus
`.5368/.5736`). It trails FMMT overall and retains too much corruption under
replacement, mixed, and Poisson damage.

## The decisive 64-pixel gate

The larger mixed-corruption Cameraman result reverses the tiny-grid optimism:

| form | truth MSE | SSIM | edge | observation RMS |
|---|---:|---:|---:|---:|
| FMMT control | **.00407** | **.6691** | **.6227** | .1816 |
| complete-residual HJ | .00656 | .5321 | .4714 | .1714 |
| direct complete terminal mixture | .01026 | .3895 | .5806 | .1246 |

The terminal mixture is not “the same”; it is substantially closer to the
observation and visibly leaves granular noise. The comparison is in
`terminal_component_visual64/terminal_component_comparison.png` and the raw
8-bit images and measurements are retained beside it.

Applying FMMT after the terminal mixture improves MSE to `.00468` and SSIM to
`.6751`, but lowers edge retention to `.5427` and remains worse than direct
FMMT. Repeated FMMT is a revealing historical control: on this one gate MSE is
best after two passes (`.00389`), SSIM peaks around four to five, and edge
retention falls monotonically. Selecting a pass count would merely reintroduce
an unknown stopping setting, so it is not a theoretical answer.

## Rejected probability coordinates

- **Order-one mean agreement.** Preserves clean structure and edges, but
  under-denoises zero-mean diffuse corruption.
- **Complete order-one agreement.** Improves every noisy small-grid condition,
  but its terminal recollision double-counts evidence.
- **Self-consistent terminal mean.** Helps sparse replacement but under-removes
  Poisson noise.
- **Marginal transport Hellinger contrast.** Local-versus-causal branch
  contrast is only `.01-.02` and is larger on clean Cameraman than diffuse
  corruptions. Marginal rearrangement is not epistemic transport uncertainty.
- **Observation cavity mean.** Dividing out the current likelihood is the
  correct target-free construction, but mean agreement is too conservative:
  aggregate MSE `.01068`, edge `.7803`, observation RMS `.0630`.
- **Root-resolved moment split.** The formula correctly keeps between-history
  variance out of nuisance evidence, but the existing state has already erased
  that variance, so it cannot enact the theory.

## Next state

Do not add a strength control or select among these modes. Transport the joint
law

\[
 (\mathcal T,b,z,j,r,a,\nu)
\]

where `T` is the local transport map/history itself. Root-conditional branch
and residual laws must remain distinct. Independent causal histories may then
contract the zero/nonzero component action; disagreement remains uncertainty
about transport and prevents structure removal. Only after that contraction
may the map, branch, and component coordinates be marginalized. This is the
required lift above a terminal mixture and above a fixed-point denoiser.

## Artifacts

- `zero_residual_component_2d.py`
- `test_zero_residual_component_2d.py`
- `probe_zero_residual_component_2d.py`
- `probe_terminal_component_visual_2d.py`
- `zero_component_terminal_18_phase1.json`
- `complete_moment_lineage_36_phase2_displacement.json`
- `terminal_component_visual64/`
