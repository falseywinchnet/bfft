# Meyer–Bregman transport: novelty, convergence, and acceleration audit

Audit began September 4, 2026. Runtime experiments remain isolated. The
manuscript was subsequently revised, and the native kernel now documents
the verified state contracts; this audit has not changed the production
update rule or public API.

## Current direction

The current integration decision is [confirmed pre-refresh results](PRE_REFRESH_INTEGRATION.md).
The implementation already contains the established transport savings.
The additional exact identities are retained as contracts and regression
oracles; no refresh, gating, or interval scheme is promoted. The sequence
below records earlier investigations.

The latest work follows the user's instruction to treat transport and its
changing geometry as inseparable and investigate carrying more state.
[TRANSPORT_OF_TRANSPORT.md](TRANSPORT_OF_TRANSPORT.md) gives the exact
finite-increment lift and records two unsuccessful compressed implementations.
The older intermediate-state diagnostic below is supporting groundwork.


The latest investigation follows the user's correction to inspect the
intermediate state itself. See [INTERMEDIATE_STATE.md](INTERMEDIATE_STATE.md)
for the exact within-pass identities, hidden vector-field information,
nonlinear remainder localization, and consistent versus obstructed local
patterns. The figure is intermediate_summary.png. Earlier terminal and
factor studies below are historical investigations, not the current direction.


The user has ended the factor investigation. See [TERMINAL_GEOMETRY.md](TERMINAL_GEOMETRY.md)
for the subsequent terminal transport formulation, direct capacity certificate,
and experiment. Earlier factor records below remain historical evidence.
The paper was revised in the subsequent September turn; its clean timing
record is meyer_clean256.json. The original audit text below predates those
changes.

## Findings

The transport construction is the substantive insight: preserve the coupled
primal/Bregman state, including the information needed to reconstruct each
changing driving field, and advance that state together. The development
record attributes the intuition to transport, following the earlier
superresolution work. The word *Krylov* describes the small numerical
approximation used in the final finite-flow implementation; it is not a
claim about the origin of the idea.

The implementation has real strengths: exact global screened solves, warm
interleaving, reduced state, retained spectra, a fused triangular spectral
pair, and a finite displacement computed on the actual nonlinear state.
The manuscript's historical cold NumPy/SciPy comparison cannot isolate an
algorithmic speedup from native implementation and different stopping rules.
The native 19/29 versus 64 timings are speed/approximation tradeoffs, not
equal-accuracy time-to-solution measurements.

A new fixed current-state step, `z <- z + 1.75 (T_f(z)-z)`, improves progress
without carrying past-update history or constructing a frozen derivative.
After fusion into the native spectral pass, 32 total passes give 59–65%
less texture error to the published pass-64 reference than the quality
schedule on all four 256-square audit sources, at comparable elapsed time.
The recovered feasible objective gaps also improve. This is an encouraging
experimental option, not a universal default: the parameter/source stress
screen has counterexamples, and clean isolated timing remains necessary.

## What makes the existing approach fast

1. **Global transport is solved, not propagated locally.** Each screened
   inverse `(c I + eta D*D)^-1` handles the spatial linear coupling globally.
   A cheap local-gradient method may need many more iterations to transfer
   the same material. The earlier repository study reports precisely that
   failure for the explicit-flux primal-dual alternative.
2. **The state carries the changing subproblem.** The retained
   `(u,w,tux,tuy,twx,twy)` determines both current driving fields. Restarting
   nested ROF solves discards useful work; extrapolating stale subproblem
   updates can misrepresent the new driving field.
3. **The spectral implementation eliminates redundant work.** It keeps
   source/primal spectra, streams reflected divergences, and solves the
   two primal spectra as a lower-triangular pair. There are two screened
   subproblems per ordinary pass, each with a forward and inverse transform.
4. **The finite jump approximates several coupled steps at once.** With
   `r=T(z)-z` and local derivative `A`, the affine finite displacement is
   `r + Ar + ... + A^(m-1)r`. The depth-two representation compresses the
   directions of this movement into a two-dimensional calculation. Ordinary
   refreshes then rebuild the nonlinear projection geometry.

The transport origin and the numerical classification are compatible. A
method can be discovered through a different line of reasoning while using
established algebra in its final implementation.

## Novelty assessment and relevant prior work

This is a scoped literature audit, not an exhaustive priority or patent
search. I did not establish an earlier implementation of this exact
six-field, finite-horizon, depth-two, fused Meyer construction.

| Ingredient | Assessment |
|---|---|
| Meyer BV–G model and bounded-flux interpretation | Established. |
| Bregman realization through coupled ROF problems | Established Gilles–Osher work. |
| Warm starts, Fourier diagonalization, and spectral screened solves | Established techniques; the exact fusion and state economy are implementation contributions. |
| Krylov compression of a matrix function acting on a vector | Established numerical machinery. The particular function here is a finite geometric polynomial. |
| Accelerating an enlarged or reduced splitting state | Close prior art exists in Anderson/ADMM/Douglas–Rachford work. Those methods are not identical to this finite transport jump. |
| Exact scalar-jump defect decomposition | Useful model-specific diagnosis, obtained by elementary substitution. It does not by itself prove that another representation can never produce halos. |
| This coupled finite-time transport realization and native schedule | Plausibly distinctive combination; deserves comparison to the closest methods and a narrower novelty claim than a new general acceleration principle. |
| The new factor-1.75 step in this audit | Classical full-state overrelaxation applied and fused to this transport. Not claimed as a new numerical method. |

Relevant primary sources:

- [Gilles and Osher, UCLA CAM 11-73](https://ww3.math.ucla.edu/camreport/cam11-73.pdf).
  The [author's publication list](https://jegilles.sdsu.edu/publications_en.html)
  dates the work to **2011**. The manuscript cites its 2024 arXiv upload;
  historical discussion should acknowledge the earlier report.
- [Fu, Zhang, Boyd, Anderson Accelerated Douglas–Rachford Splitting](https://arxiv.org/abs/1908.11482), submitted 2019, journal 2020.
- [Ouyang et al., Anderson Acceleration for Nonconvex ADMM Based on Douglas–Rachford Splitting](https://arxiv.org/abs/2006.14539), 2020: especially relevant to which state is retained or reduced before acceleration.
- [De Sterck, He, Krzysik, Anderson Acceleration as a Krylov Method](https://arxiv.org/abs/2109.14181), 2021: clarifies the relation between history extrapolation and Krylov residual polynomials.
- [Jia and Lv, A Posteriori Error Estimates of Krylov Subspace Approximations to Matrix Functions](https://arxiv.org/abs/1307.7219), 2013/2015: relevant to estimating matrix-function projection error rather than fixing depth without an error indicator.

These references classify the implemented operations. They do not establish
that the user's transport intuition was derived from those sources.

## Mathematical and evidentiary issues in the manuscript

### Adjoint sign and initialization

The manuscript defines `D*` as the adjoint of the periodic forward gradient,
but prints `-eta D*R(t)` in the primal update. The code uses
`-eta div(R(t))`, and `div=-D*`. Therefore the correctly adjoint-written
numerator has **`+eta D*R(t)`**. The tangent equations and spectral proof
must use the same convention. The denominator `c I + eta D*D` is correct.
The audit checks the adjoint identity and the tangent by finite differences.

The code also has a special source-driven first pass. Applying the printed
ordinary map directly to a zero six-field state does not give that pass:
its first primal update would be zero. The source-only initialization needs
an explicit equation, as already present in the numerical oracle.

### The fixed-chart remainder is conditional

The conditional estimate assuming `||N_z(d)|| <= L ||d||²/2` is algebraically
valid. Semismoothness alone does not establish it for one derivative frozen
at the starting point across projection-boundary crossings. At radius one,
take `t=(1,0)`, the implementation's interior derivative `D Pi(t)=I`, and
outward `h=(epsilon,0)`. Then

`Pi(t+h) - Pi(t) - h = -h`.

The remainder is first order. An executable test verifies this for epsilon
from `1e-2` to `1e-6`. Even without crossing the inside/outside boundary,
the exterior disk projection changes its tangential derivative as the
direction and magnitude change. A fixed active set does not make the disk
projection affine.

### Error to pass 64 is not objective suboptimality

The paper says pass 64 is a visually accepted finite reference. It should
not be described as the mathematical bottom. In this audit, ordinary pass 64
still has relative texture differences of 8.7%, 22.7%, 18.7%, and 14.9%
from pass 4096 for Barbara, Cameraman, analytic, and crossing respectively.
Those are differences between trajectory checkpoints, not ground-truth
errors. Degeneracy can allow different minimizers to have different images.

The late reference is checked rather than assumed exact: pass 2048 to 4096
texture drift is 0.39–1.73%, while the recovered relative primal-dual gap at
4096 is approximately `2.05e-5` to `1.36e-4`. This is strong objective
evidence but not an image-uniqueness claim.

### The ordinary time frontier was undersampled

Ordinary passes 19, 29, and 64 do not characterize the frontier between
them. In the focused native rerun, **ordinary pass 40 is both faster and
closer to pass 64 than the quality schedule on Barbara, analytic, and
crossing**. Cameraman's ordinary 40 is more accurate and has nearly the
same time. Thus the broad claim of a wall-time Pareto improvement needs a
denser matched-accuracy baseline. CPU contention makes small timing margins
uncertain; the numerical errors do not have that ambiguity.

The objective-gap comparison is more stringent: the quality schedule's gap
is worse than ordinary 32 on every tested image, despite its better pass-64
texture agreement. Its numerical purpose is finite-trajectory prediction,
not direct gap minimization.

## An objective certificate from the retained transport

For the discrete model

`min TV(u) + lambda/2 ||f-u-v||²,  v=D*g, |g(x)|<=mu`,

recover feasible fields from any six-field state:

`p = eta_u Pi_a_u(t_u)`, `q = D*p`,

`g = (eta_w/c_w) Pi_a_w(t_w)`, `v_feasible = D*g`.

The respective pointwise radii are one and mu. An upper and lower bound are

`P = TV(u) + lambda/2 ||f-u-v_feasible||²`,

`L = <f,q> - ||q||²/(2 lambda) - mu TV(q)`.

Thus `L <= F* <= P`. The gap has the independently checked decomposition

`P-L = [TV(u)-<u,q>] + [mu TV(q)-<v_feasible,q>]`

`       + lambda/2 ||f-u-v_feasible-q/lambda||²`.

Every term is nonnegative for the recovered feasible fields. This is a
certificate for the **feasible decomposition recovered from the state**;
it is not silently assigned to the possibly infeasible emitted texture
`f-u-w`. `certificate.py` separately records the RMS difference between
those two textures. No expensive inner solve is needed to evaluate it.

## What was tried

The 64-square exploratory run compared ordinary transport, depth-two and
depth-four finite jumps, full-state Anderson memories 3/5/8, and a factor-1.5
current-state step, using the identical NumPy map and a 2048-pass reference.
The full-state history experiment is distinct from the previously rejected
two-block/subproblem Anderson formulation. It showed some improvement per
evaluation but no compelling time advantage; it was not pursued. These
128-update tests neither reproduce nor refute the user's reported 40,000-step
stall in the earlier formulation. Repository notes explicitly record the
changing-driving-field concern and earlier momentum failures.

The native screen tested common factors 1.25, 1.5, 1.75, and 1.9. The chosen
experimental step is

`z_next = z + alpha (T_f(z)-z)`, with `alpha=1.75`,

after the same four ordinary prefix passes used in the paper. It uses the
current map only. In particular the second screened solve sees the exact
new first primal from the coupled pass before all six components are
advanced together. Mixing the first primal before constructing the second
subproblem would be a different algorithm.

The first implementation built a residual and added it afterward. It was
too expensive. Native mode 3 fuses the pointwise state update and maintains
the primal spectra without additional FFTs. Native mode 4 replaces the last
two relaxed steps by ordinary chart refreshes. All modes remain isolated
from the public library.

## Focused native results

All sources here are 256×256; Barbara is resized from its archived 512 image.
One native M4 thread, two warmup rounds, 15 shuffled timed repetitions per
method, allocation of the returned output included, plan setup and subsequent
diagnostics excluded. Several unrelated CPU-heavy processes were active.
The earlier 8-thread exploratory timings were particularly noisy and are
retained as such. These are not clean reproductions of the paper's 8-thread
millisecond table. The same sources were used during parameter exploration;
this is not a held-out generalization study.

| Source | Quality ms | New 32 ms | Quality E64 | New E64 | Quality gap / new gap |
|---|---:|---:|---:|---:|---:|
| Barbara resized | 71.97 | 66.46 | .04985 | .01741 | 3.31 |
| Cameraman | 64.60 | 64.34 | .09166 | .03738 | 2.78 |
| Analytic | 67.08 | 61.87 | .11111 | .04408 | 3.28 |
| Crossing | 67.38 | 61.69 | .12604 | .04853 | 1.89 |

The error reduction is 59.2–65.1%; new error is 2.45–2.86 times smaller.
Elapsed-time medians are 0.4–8.5% lower, but that small timing gain is not
established beyond contention noise. The robust finding is substantially
better numerical progress at comparable measured time.

At 64 passes, relaxation reduces error to pass 4096 by 11.6–27.1% and the
feasible objective gap by 35.6–53.6% against ordinary 64. It also takes more
time, so these numbers alone are **not** a matched-time deep-speedup claim.
The plotted full frontier shows that objective-gap gains over ordinary
iteration are much smaller and less uniform than gains over the published
finite-flow schedule. This distinction is central to the audit.

![Native time and objective frontier](frontier.png)

![Cartoons and differences to the late reference](comparison.png)

Two ordinary finishing passes lower the objective gap at all tested budgets
24/32/48/64 and all four images. At budget 32 they lower the gap by about
7–15%, while increasing error to pass 64 by 12–15%. They therefore expose
the difference between settling the true state and imitating finite time.
This ablation was checked numerically; it was not separately benchmarked
as a final wall-time product.

## Stress results and limits

The fixed-factor stress screen uses 128-square constant, ramp, edge,
checkerboard, white-noise, low-frequency-carrier, crossing, and noisy-crossing
sources at `(lambda,mu)=(.02,20),(.05,40),(.1,80)`, with 128 total passes.
All 24 runs remain finite for each tested factor. Five ordinary cases are
already at a gap below `1e-8`; exclude those from gain ratios.

- Factor 1.5 improves the gap in 17/19 nontrivial cases.
- Factor 1.75 improves the gap in 16/19. Counterexamples: ramp at (.02,20),
  edge at (.1,80), and low carrier at (.1,80). Gap ratios to ordinary are
  approximately 2.20, 1.02, and 1.86.
- Factor 1.9 improves 17/19, but the (.02,20) ramp gap is approximately
  **94.5 times worse**. Finite output is not sufficient evidence of stability
  or efficiency.

The stress screen uses the residual-form implementation, which the native
oracle checks against the fused version. It measures mathematical behavior,
not fused wall time. No convergence theorem for a universal factor above one
has been established here. There is no reason to promote 1.75 as an
unconditional production default.

## My assessment and the next substantive research question

The strongest idea is the faithful transport state and the elimination of
redundant work around it. The small polynomial is one way of realizing that
idea, not its conceptual source. The new experiment suggests the existing
implementation freezes the transport prediction longer than is worthwhile
on these cases: short, larger, freshly recomputed coupled steps can be more
effective than elaborate long jumps.

The next promising question is how to choose a safe advance from the current
transport, with a cheap certificate or measured nonlinear defect. A larger
constant step cannot solve this universally. A candidate should account for
the radial projection boundary, changing tangent direction, and the true
state's feasibility defect; all rejected trials and gap evaluations must be
charged to its wall time. The duality gap added here supplies an objective
test independent of pass 64. An adaptive rule has **not** been implemented
or claimed successful by this audit.

Before publication, correct the adjoint signs and source initialization,
clarify the conditional remainder estimate, cite the 2011 Gilles–Osher
report and relevant matrix-function/splitting acceleration literature, and
replace sparse baseline timings with equal-error/equal-gap time frontiers.

## Reproduction and files

Run `sh experiments/meyer_transport_audit/run.sh` from the authoritative
checkout. It selects the Mini with `m4host`, synchronizes with `m4build`,
compiles the experimental wrapper against the existing complete Fourier
library, runs tests and final validation, copies `/tmp` outputs back, and
renders the saved results locally. It does not install remote software.

- `model.py`: plotting-free copy of the existing independent six-field oracle.
- `native.cpp`: ordinary, published finite-flow, residual-form relaxation,
  fused relaxation, and final-refresh modes in an experimental wrapper.
- `certificate.py`, `test_audit.py`: feasible primal/dual bounds and six tests.
- `native_audit.py`: native oracle, thread identity, translation, baseline
  equivalence, and exploratory native screen.
- `run_audit.py`, `results64.json`: initial small prototype screen.
- `meyer_native_audit256.json/.npz`: noisy 8-thread exploratory record.
- `meyer_validate256.json/.npz`: unfused relaxation, one thread.
- `meyer_fused_validate256.json/.npz`: principal fused validation.
- `meyer_audit_stress.json`: 24-case parameter/source screen.
- `meyer_settle_probe.json`: final ordinary-refresh ablation.
- `plot_results.py`: renders figures from records without rerunning solvers.

Validation passed: six mathematical unit tests; complete native six-field
agreement with NumPy for multiple relaxation factors; native baseline
agreement with the loaded library; bit identity across one/four threads;
periodic translation equivariance for fused and refreshed modes; and the
two-refresh trajectory oracle. The historical public flow contract was
already represented by the independent implementation; no production code
was edited or promoted during this audit.
