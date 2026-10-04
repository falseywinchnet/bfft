# Finite-pass acceleration of real library routines

This experiment applies the existing Krylov/Bregman transport engine to
actual library recurrences, with public-library controls and total elapsed
cost. It does not substitute a different optimizer for a requested finite
trajectory. All studies include initialization, acquisition, rejected
proposals, feature checks, fallback, and final readout in timing. Data
construction is shared and excluded. Timings are shuffled paired repeats.

The retained environment is NumPy 1.26.4, SciPy 1.13.1, scikit-image 0.24.0,
and an unmodified SPORCO 0.2.2.post1 source checkout on the M4 CPU. BLAS is
single-threaded. SPORCO uses its existing NumPy FFT fallback; PyFFTW was not
installed. Source versions and, where available, source digests are in the
JSON records. Current scikit-image 0.26 source was also inspected; these
measurements are specifically of the installed 0.24 release.

## Candidate selection

| Library routine | Relevant structure | Trial |
| --- | --- | --- |
| scikit-image `denoise_tv_chambolle` | finite differences and local vector normalization | exact adapter, analytic tangent, existing sampled-defect engine |
| scikit-image `richardson_lucy` | two SciPy convolutions per multiplicative pass | exact adapter, analytic tangent, fixed requested pass count |
| SPORCO `TVL2Deconv` | Fourier normal solve and vector shrinkage | exact reduced state, curved feature certificate reused from Meyer |
| SPORCO `BPDN` | factored normal solve and scalar soft threshold | exact reduced state, scalar feature certificate, complete-solve and adaptive-event integration |
| PyLops `splitbregman` | repeated least-squares work and shrinkage | source/documentation candidate; not benchmarked here |
| PyProximal `LinearizedADMM` / TV proximal | gradient/proximal splitting in image restoration | documentation candidate; not benchmarked here |

SciPy's `lsq_linear` and `nnls` were inspected as well. They already use
trust-region or active-set mechanisms and inner linear solvers; treating
them as a stationary Meyer-like map would discard useful existing logic.
They were not used as artificially weakened baselines.

Primary upstream references:

- [scikit-image restoration API](https://scikit-image.org/docs/stable/api/skimage.restoration.html)
- [SPORCO TV classes](https://sporco.readthedocs.io/en/latest/modules/sporco.admm.tvl2.html)
- [SPORCO source release](https://github.com/bwohlberg/sporco/tree/v0.2.2.post1)
- [PyLops split Bregman](https://pylops.readthedocs.io/en/v2.7.1/api/generated/pylops.optimization.sparsity.splitbregman.html)
- [PyProximal image denoising](https://pyproximal.readthedocs.io/en/latest/tutorials/denoising.html)
- [SciPy bounded least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)

## What was measured

`results/initial.json`: 16 scikit-image cases, camera/coins at 128 and 256,
200 requested passes, three timing repeats. TV uses `eps=0` to define a
finite trajectory; its default early stopping is not claimed to be preserved.
Richardson–Lucy uses `clip=False`, no `filter_epsilon`, and Poisson-corrupted
blurred images. Its pass count is a regularization parameter. Increasing it
does not necessarily improve the restored image. The engine is the earlier
sampled-defect `discover`, not a certified path admission. Ordinary adapters
match library outputs exactly in all these cases. TV accepts no jumps and
runs at 0.36–0.39 times library speed. Richardson–Lucy runs at 0.59–0.67
with maximum relative output error 1.07e-4. Both are rejected as accelerators.

`results/sporco_initial.json`: 16 TV denoising/deconvolution cases, fixed rho,
unit relaxation, 200 requested passes, three repeats. This is an exposed
SPORCO configuration, not its default adaptive policy. The quotient matches
library output within 2e-15. The quotient alone is 1.26–1.32 times faster;
adding certified discovery accepts no jumps and loses time. `tv_capacity.json`
checks depths 2 through 24 and later anchors; early curvature, rather than
only too few Krylov vectors, limits useful horizons.

`results/bpdn_initial.json`: 16 image-patch sparse-coding cases, 500 passes,
8x8/16x16 patches, one or sixteen simultaneous signals, camera/coins, two
regularization weights, three repeats. The dictionary is a normalized
separable overcomplete cosine dictionary. Nine cases beat the public library;
only five beat the already-reduced ordinary map. The latter distinction is
necessary to identify actual engine benefit. Finite-pass timing alone is
insufficient: some easy cases have already converged well before 500 passes.

`results/bpdn_solve.json`: 24 held-out cases (camera/coins/moon, seeds 1/2,
one/sixteen signals) stop at primal–dual gap <=1e-5 of the zero-code objective.
Every method checks at the same multiples of 64. The cap is 4096 passes.
Fixed-rho engine and controls all finish 14 cases; ten are censored failures,
not successful solves. On the 14 completed cases, the engine is 1.02–3.32
times faster than the fixed-rho library, and beats the quotient in twelve.
SPORCO's standard adaptive-rho/relaxation configuration finishes all 24 and
is faster than the fixed-rho engine in most cases. This blocks promoting the
fixed-rho engine as a general improvement to the default solver. Raw timing
ratios in capped rows compare work up to the cap, not time to solution.

`results/bpdn_adaptive.json`: the engine is inserted between SPORCO's actual
rho-update events, preserving default relaxation and stopping checkpoints.
The native driver without shortcuts exactly matches the library in all 24
cases. All accelerated runs meet the gap target; four are faster (1.04–1.19
versus the library), twenty are slower, and the median ratio is 0.93. Frequent
control events limit the time available to repay acquisition. This is a
working integration prototype, not a recommended default or upstream-ready
performance claim. Local chart bounds do not certify the entire trajectory
against another run whose adaptive penalty decisions may differ.

## Engine and mathematical contract

`certified_transport.py` reuses the prior `arnoldi_stream` and `reduced_path`.
A library adapter supplies:

- the complete future-driving state and its ordinary map;
- an analytic linear action at the current state;
- nonlinear feature remainders evaluated at every predicted input;
- a proof of nonexpansivity and of the remainder-transfer norm bound;
- an output map and its error amplification bound.

For fixed-parameter ADMM, with shrinkage S, linear response B, and constant b,

    F(v) = v + alpha * (b + B(2*S(v) - v) - S(v)).

For BPDN, B=rho*(D.T*D+rho*I)^(-1). For TV deconvolution,
B=rho*G*(A.T*A+rho*G.T*G)^(-1)*G.T. Both satisfy 0<=B<=I.
For 0<alpha<2, F is an averaged composition of nonexpansive reflected maps.
The nonlinear remainder is alpha*(2*B-I)*R_S, bounded by alpha*||R_S||.
Scalar shrinkage has an explicitly evaluated piecewise-affine remainder;
vector shrinkage reuses the exact disk-projection remainder from Meyer.
A fixed exterior mask is not asserted to make vector shrinkage affine.

Arnoldi compression and nonlinear remainder norms are summed over every
predicted input. Nonexpansivity propagates their sum to a state-error bound.
These are exact-arithmetic results, evaluated numerically in float64 without
outward rounding. An adaptive rho decision is an additional state event:
`adaptive_bpdn.py` never jumps over it and leaves an ordinary native pass to
reconstruct the complete control state. A changed rho invalidates the old
chart. Global error certification across possibly differing adaptive decisions
is not supplied; the complete-solve target is checked independently.

The API currently supports the explicitly tested configurations, including
real scalar data and unweighted shrinkage. Unsupported configurations are
rejected; the experiment is not a replacement for the full public library API.

## Reproduction

Fetch the pinned unmodified SPORCO release and its small missing import
 dependency without installing or changing host software:

```sh
python3 -m experiments.library_acceleration.fetch_sources
```

From the authoritative checkout, use `m4build` with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=tmp/library_sources/deps:tmp/library_sources/sporco-0.2.2.post1:. \
  python3 -m unittest discover -s experiments/library_acceleration -t . -p 'test_*.py' -v
```

Use the same prefix for:

```sh
python3 -m experiments.library_acceleration.study --out /tmp/library_acceleration_initial.json
python3 -m experiments.library_acceleration.study_sporco --out /tmp/library_acceleration_sporco.json
python3 -m experiments.library_acceleration.probe_capacity
python3 -m experiments.library_acceleration.study_bpdn --out /tmp/library_bpdn_initial.json
python3 -m experiments.library_acceleration.solve_bpdn --out /tmp/library_bpdn_solve.json
python3 -m experiments.library_acceleration.study_adaptive --out /tmp/library_bpdn_adaptive.json
python3 -m experiments.library_acceleration.study_reuse --out /tmp/library_sporco_reuse.json
```

Copy `/tmp` results immediately to this directory's `results/` before another
mirror sync. `fetch_sources.py` retains upstream source/licenses in ignored
temporary storage; the adapter modules are independently expressed and the
engine imports the established research implementation.

## Contribution status

No upstream issue, message, or pull request has been sent. The portable
model/feature contract and four adapters are concrete reviewable work. The
engine has real scoped wins in sparse coding, but its adaptive integration
needs a more effective acquisition cost policy before proposing default use.
The measured TV ports do not justify a performance contribution.

`sporco_reuse.py` is a separate small contribution candidate: `xstep` already
computes Xf, so `relax_AX` can pass it to `cnst_A` and avoid one redundant
forward FFT per iteration. It preserves the default adaptive algorithm,
objective/residual statistics, and stopping logic. This is exact Fourier-state
reuse, not evidence for finite-pass Krylov acceleration. Its default-policy
benchmark and real/complex regression checks are retained separately.

## Subsequent direction change

The user explicitly redirected the work from iteration-preserving ports to
an analytical twin of the variational problem and one final correction solve.
That supersedes the contribution plan above as the active engine objective.
See `ANALYTICAL_TWIN_DIRECTION.md`. The library studies remain useful measured
baselines. `chambolle_floor.py` independently distinguishes default stopping,
objective progress, reconstruction distance, and a primal-dual-bracketed floor.
