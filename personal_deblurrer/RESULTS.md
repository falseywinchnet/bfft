# First executable checkpoint

This is the measured result of the global, registered, periodic-boundary
checkpoint in `results/results.json`.  The M4 Mini battery used six image
sources, two noise seeds, 96 by 96 luminance images, and a fixed noise standard
deviation of 0.002 or 0.0025.  Clean images were used only to score results.

## Invariants

All eight tests pass.  They cover positive unit PSF mass, DC conservation,
forward/adjoint closure, identity fixed point, complementary recovery, relative
kernel-pair identification, common-blur gauge detection, and coverage-gated
fallback.

## Reconstruction summary

| Capture family | Joint dead fraction | Best capture PSNR | Matched Fourier inverse | Exposure transport | Gain over inverse |
|---|---:|---:|---:|---:|---:|
| Single Gaussian | 0.863 | 23.509 | 26.772 | 26.772 | fallback |
| Orthogonal motion | 0.102 | 23.438 | 34.257 | 36.869 | +2.612 dB |
| Oblique motion | 0.123 | 22.815 | 33.306 | 34.343 | +1.037 dB |
| Motion plus defocus | 0.240 | 23.823 | 32.158 | 32.568 | +0.410 dB |
| Identical defocus | 0.496 | 23.823 | 29.666 | 29.666 | fallback |
| Three-angle motion | 0.089 | 23.027 | 33.647 | 36.985 | +3.338 dB |

The Fourier inverse and coverage fallback both use regularization `1e-3`.
The exposure method adds twenty warm Split Bregman passes, each consisting of
one exact Fourier latent solve and one bounded isotropic flux update.  It is
not credited with a gain when the coverage gate selects the inverse fallback.

## Blind-pair policy

The phase-preserving pair closure selected the exact ordered kernel pair on all
36 complementary two-capture trials.  It then produced the same aggregate
scores as the known-kernel reconstruction for orthogonal motion, oblique
motion, and motion plus defocus.

For the 12 identical-defocus trials, the estimator detected negligible
relative transport and abstained.  The policy returned the capture mean at
23.826 dB rather than claiming to know the shared defocus.  The 29.666 dB
known-kernel number in the table is an oracle control and is not a blind result.

## Timing and claim boundary

The complete battery took 18.73 seconds on the M4 Mini CPU.  This is an
engineering timing for this implementation, not a comparison with published
systems.

The result supports the mechanism under analytic global blur and matched
synthetic evaluation.  It does not establish single-image blind deblurring,
spatially varying blur, real-camera generalization, non-periodic boundary
handling, or state-of-the-art speed or quality.

## Uncertainty-transport checkpoint

The follow-up battery in `uncertainty_results/uncertainty_results.json` uses
the same six 96 by 96 sources, one seed, a 24-kernel positive-path catalog,
orthogonal length-11 motion, an identical-Gaussian gauge control, and read
noise levels 0, 0.002, and 0.01.

| Read noise | Exact relative pair | Mean true-pair probability | Effective hypotheses | Policy PSNR | Oracle PSNR |
|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 1.000 | 1.00 | 37.647 dB | 37.647 dB |
| 0.002 | 100% | 0.990 | 1.09 | 37.054 dB | 37.074 dB |
| 0.01 | 66.7% | 0.309 | 32.98 | 28.349 dB | 29.171 dB |

This is the intended uncertainty behavior.  The ideal case collapses exactly
to one branch.  At the ordinary synthetic noise level, the law remains sharply
concentrated and loses only 0.020 dB relative to the known-kernel oracle.  At
strong noise it becomes broad instead of reporting a falsely certain kernel,
and its 0.822 dB oracle gap is retained as an explicit failure signal.

The high-pass residual diagnostic estimated read-noise scales of 0,
0.002004, and 0.010021 for the three injected levels.  The identical-blur
gauge was detected in 100% of its trials.  The full M4 Mini CPU battery took
13.22 seconds.

The intervals in this checkpoint are empirical finite-catalog credible
intervals.  They are not calibrated coverage guarantees and are not called
zonotopic enclosures.

## Shift-first / centered-mixing checkpoint

The follow-up in `shift_mix_results/results.json` replaces the desktop
periodic inverse with reflect-boundary positive transport and factors a known
operator into deterministic shift followed by centered mixing. It uses six
96 by 96 sources and read noise sigma 0.002.

| Formation | Observation | Known two-stage | Gain | Blind provisional |
|---|---:|---:|---:|---:|
| Gaussian center mix | 23.990 dB | 26.240 dB | +2.250 dB | 24.252 dB |
| Disk center mix | 24.284 dB | 27.862 dB | +3.578 dB | 24.560 dB |
| Line center mix | 23.591 dB | 29.078 dB | +5.487 dB | 23.609 dB |
| Curve center mix | 23.469 dB | 29.289 dB | +5.820 dB | 23.489 dB |
| Shift then line mix | 20.144 dB | 28.821 dB | +8.677 dB | 19.978 dB |

Every known-operator case improves both PSNR and SSIM. The shifted-line case
recovers `(3,-2)` pixels first and only then applies the centered path inverse.
The single-image blind column is evidence-weighted. It makes small gains on
the four centered cases rather than granting an uncertain phase estimate full
inverse authority. Its shifted score is not alignment-invariant and is an
explicit failure: absolute global translation is unidentifiable from one
unknown image.

A line exposure retains Fourier null lines. At the coverage authority floor,
the 96 by 96 line control marks 15.9% of coefficients unresolved and removes
51.4% of the positive inverse's attempted unsupported energy. The axis-line
solver also differentiates the exposure integral into residue-class path
recurrences. Its raw 25% checkpoint authority is diluted to 10.83% after the
64-pass positive descent, preventing the two sharpening mechanisms from
stacking into faint halos. Complementary exposures are still the only way to
measure the missing bands rather than choose their seed gauge.

The refreshed battery uses a 64-pass maximum for known operators and an
eight-pass ceiling for blind estimates. Known operators all use the common
exact exposure refinement; only the auxiliary line-constraint coefficient
varies continuously. It took 11.22 seconds on the M4 Mini. The invariant suite
now has 91 tests; the Mini skips only the scikit-image GUI-fixture test.

## Oblique and curved path-chart checkpoint

`path_chart_results/results.json` measures twelve line angles at 15-degree
spacing and curves at six orientations with signed bends -16, -8, 2, 4, 8,
12, and 16. Every geometry uses all six 96 by 96 sources with read noise sigma
0.002. The comparison is deliberately against the same 64-pass positive
basin, not only against the blurred observation.

| Exposure set | Observation | Positive basin | Continuous descent | Descent gain | SSIM change |
|---|---:|---:|---:|---:|---:|
| 12 line angles | 23.384 dB | 27.262 dB | 28.153 dB | +0.892 dB | +0.0189 |
| 42 signed curves | 22.994 dB | 29.445 dB | 30.445 dB | +1.000 dB | +0.0086 |

All twelve line angles improve mean PSNR and mean SSIM over the positive-only
basin. Every one of their 72 individual source/angle trials improves PSNR;
the smallest gain is +0.049 dB. Curves use the same Eikonal-ordered exposure
operator whose original PSF atoms form an exact reflected gather/scatter pair.
Every one of the 42 curve geometries and all 252 individual source/geometry
trials improve PSNR; the smallest individual gain is +0.042 dB. Thirty-nine
of 42 curve geometries also improve mean SSIM, and the aggregate SSIM change
is positive. The remaining three are retained as perceptual tradeoffs rather
than hidden by the PSNR result.

The optimized NumPy plan reuses one chart, one reflected index plan, and one
`A*1` normalization across the main and endpoint branches. Flat gather and
weighted occupancy scatter reduce the complete M4 CPU sweep from 324.23 to
222.00 seconds (1.46x) while changing no reported metric by more than
`5.26e-13`. The optimized result is `path_chart_results/results.json`; the
frozen pre-optimization result is retained beside it for the parity audit.

The optional native C++ gather/scatter ABI (now v3) matches the optimized NumPy
oracle to the same `5.26e-13` full-battery metric bound. On the M4 Mini its
96 by 96 RGB forward-plus-adjoint microbench is 4.0x faster for the dense
Gaussian plan and 6.0--7.4x faster for disk, line, and curve plans. A matched
back-to-back workload of 18 complete 64-pass reconstructions took 5.81 seconds
native versus 7.02 seconds NumPy (1.21x end to end). A separate full native
sweep took 257.81 seconds versus the NumPy sweep's 222.00 seconds, demonstrating
substantial whole-run variance and that FFT basin solves, metrics, and source
work remain outside this ABI. The slower aggregate observation is retained in
`path_chart_results/continuous_v1_native.json`; it is not discarded in favor
of the microbenchmark.

The next basin optimization removes a second independent redundancy. The
padded circular positive solve now builds one OTF, carries its terminal forward
prediction, and uses batched real half-spectrum transforms. Frozen Gaussian,
disk, line, and curve basin images differ from the original by at most
`4.3e-15`; their residual traces differ by at most `1.1e-17`. Local basin
runtime improves by 1.97x--3.11x. On the M4 Mini, the full complex-plan battery
fell from the original 324.23 seconds to 131.97 seconds (2.46x) with maximum
metric drift `7.11e-15`. A matched M4 solver workload then measured 8.66
seconds for the real half-spectrum plan versus 9.96 seconds complex (1.15x).
The separate full half-spectrum sweep took 171.43 seconds, again exposing
whole-run variance; that slower aggregate is retained as
`path_chart_results/continuous_v3_rfft.json` and is also the selected current
result rather than being hidden.

The importance is structural as well as numerical: there is no center/line/
curve reconstruction switch. Every nonidentity positive exposure measure uses
one exact reflected operator and one observation-derived discrepancy stop.
The differentiated line recurrence enters only through a continuous
anisotropy/tangent/residual/action/lattice coefficient. The common refinement
normalizes by `A*1`, gates its correction by Fourier support, and transports
endpoint seed sensitivity without using those inferior endpoint branches as
the reconstruction.

A hard vertical step blurred by a horizontal length-11 exposure directly
checks the reported faint-line-halo failure. With read noise sigma 0.002, RMS
error in the constant bands 2--15 pixels beside the edge is 0.13353 in the
observation, 0.01862 after the positive-only basin, and 0.00768 after continuous
exposure descent. The 99th-percentile absolute band error falls from 0.05627
in the positive basin to 0.03060. This is now an executable regression test;
it measures halo suppression where no true texture can disguise ringing.

## Spatial warp/mixing checkpoint

`spatial_results/results.json` evaluates the same six 96 by 96 source
structures at read noise sigma 0.002. Every case is one spatial positive
exposure field; deterministic and mixing labels below describe limiting
controls, not selected reconstruction branches.

| Spatial exposure | Observation | Barycentric pullback | Full exposure transport | Gain |
|---|---:|---:|---:|---:|
| Deterministic shear | 23.173 dB | 35.062 dB | 35.062 dB | +11.889 dB |
| Shear plus centered mix | 24.015 dB | 24.617 dB | 28.570 dB | +4.556 dB |
| Deterministic rotation | 21.618 dB | 32.038 dB | 32.038 dB | +10.420 dB |
| Rotation plus exposure mix | 23.856 dB | 26.960 dB | 30.250 dB | +6.394 dB |
| Centered rotational exposure | 27.743 dB | 27.700 dB | 32.700 dB | +4.957 dB |

All 30 source/case trials improve PSNR. The minimum individual gain is +2.614
dB for shear-plus-mix and +3.464 dB for centered rotational exposure. Every
field has zero fold fraction. Fixed-point inversion of `q=p-m(p)` closes from
`7.1e-15` for shear to `3.4e-8` for the rotational controls. Pure deterministic
fields stop centered descent at zero passes. Three texture trials lose SSIM
despite substantial PSNR gains: tapered hair under shear-plus-mix (-0.0112),
tapered hair under centered rotation (-0.0290), and geometric interfaces under
centered rotation (-0.0192). These perceptual tradeoffs remain promotion work.

The native spatial ABI v3 agrees with the NumPy oracle within `1.42e-14` on
every full-battery metric. It accelerates forward-plus-adjoint by 7.3x--7.9x
and reduces the M4 battery from 2.14 to 1.26 seconds without sensitivity
transport; the selected sensitivity-enabled run takes 1.38 seconds. Its
uncertainty field combines normalized back-transported forward residual and
the disagreement between inverse-coordinate and normalized-adjoint
barycentric pullbacks. It is a sensitivity diagnostic, not a calibrated
interval.

## Estimated rotational-consensus checkpoint

`spatial_estimation_results/results.json` removes the known-field assumption
for one deliberately bounded camera manifold. Three noisy captures have true
relative mean rotations `[-4, 0, 4]` degrees, four-degree exposure extent, and
read-noise sigma 0.002. Every forward and reverse pair registration enters one
weighted cycle-consistency solve; no pair or blur class is selected. The
resulting spatial fields feed one shared-latent positive transport descent.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Unregistered pixel average | 26.404 dB | 0.7643 |
| Best individual capture (evaluation oracle) | 31.898 dB | 0.9288 |
| Estimated rotational consensus | 35.078 dB | 0.9662 |
| Known center-field single-capture oracle | 37.990 dB | 0.9634 |

The estimated consensus beats the best individual capture on all six sources,
by +1.554 to +4.876 dB. Its angle error is 0.00226 degrees mean and 0.00752
degrees maximum; exposure-extent error is 0.00277 degrees mean and 0.00752
degrees maximum. The full M4 Mini battery takes 1.84 seconds. Three identical
captures exercise the common-rotation/common-exposure gauge: the solver
abstains and changes the input by at most `2.22e-16`.

This is not a dense optical-flow claim. The mean gap to the known center-field
single-capture oracle is 2.912 dB, with a 5.842 dB worst case on tapered hair.
That texture also loses 0.0021 SSIM relative to its best capture despite its
+1.554 dB PSNR gain. The GUI therefore runs consensus only on the explicitly
chosen Pair A and Pair B; equal dimensions never imply registration. The
headless API accepts longer verified sequences. Synthetic sequences with
different immutable source truths are rejected.

The spatial inverse now also has an executable fold gate. If the determinant
of `I-grad(m)` is nonpositive anywhere, the deterministic pullback and centered
descent both stop at zero passes, the observation is returned exactly, and the
fold mask enters the sensitivity output. This is abstention, not occlusion
recovery; depth ownership remains outside the current model.

## Continuous dense-flow checkpoint

`dense_estimation_results/results.json` replaces the rotational manifold with
one two-component field. The control combines translation, affine shear,
cross-axis motion, smooth local deformation, and finite exposure mixing in
every trial; none of those components is labelled for the estimator. Forward
and reverse robust registration enter one cycle-consistent connection, whose
confidence is harmonically transported through the image-induced Eikonal
metric. The resulting field becomes the same positive spatial exposure object
used by the known-field and rotational controls.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Best individual capture | 21.493 dB | 0.6084 |
| Unregistered pixel average | 23.497 dB | 0.6428 |
| Estimated dense consensus | 33.462 dB | 0.9702 |
| Known-field consensus oracle | 35.179 dB | 0.9805 |

All six sources improve over the unregistered average by 8.18--12.42 dB and
over the best individual capture by 10.43--13.83 dB. Mean, q90, and maximum
endpoint error against the exact barycentric pair map are 0.259, 0.678, and
1.332 pixels. The optimized M4 Mini run takes 2.76 seconds. Its selected
artifact has SHA-256
`7d2cfc27d4cd263ab4e56d388cfbdb8ce397ca9a51334c8481c477702baf4b1c`.

The mean known-field gap is 1.716 dB. Line drawing is the clear hard case at a
4.109 dB oracle gap, while the other five gaps are 1.51 dB or less. Identical
observations carrying an arbitrary common warp and exposure trigger the common
gauge abstention and change by exactly zero in the battery.

There is still no reconstruction-family switch. Rotation and dense estimation
both call `solve_spatial_field_consensus`; translation, affine motion, and
local deformation are shapes of one field. The desktop exposes only this
continuous Pair A/Pair B route. The older finite kernel-posterior experiment is
retained headlessly as an estimation control rather than offered as a competing
GUI reconstruction mode.

This checkpoint estimates one smooth field. The positive joint-visibility
checkpoint below can route observation support around individual folds, but it
still reconstructs one latent composite and cannot carry distinct appearances
on simultaneous motion sheets. Native promotion remains deferred until that
representation and the current dense operator profile are measured.

The optimization sequence preserves that promotion order. Exact identical
observations take a machine-precision gauge fast path, and a coupled per-pixel
Jacobi preconditioner accelerates the matrix-free normal equation. Both
first-derivative closure equations then join brightness closure at the finest
physical scale; full-step acceptance removes redundant trial warps and the
preconditioned CG ceiling falls from 80 to 60. Battery time falls from 3.27 to
2.76 seconds (1.18x), mean endpoint error falls by 0.0030 pixels, and consensus
PSNR rises by 0.014 dB. At the finest scale only, the quadratic flow metric is
then replaced by a robust Charbonnier action. This leaves the smooth control
essentially invariant while materially improving the folded-visibility
control below. Tapered hair trades -0.023 dB PSNR for +0.00016 SSIM relative
to the brightness-only checkpoint; this is retained rather than hidden.
Frozen unpreconditioned, brightness-only, and derivative-constraint ledgers
remain as `results_v1_unpreconditioned.json`,
`results_v2_preconditioned_brightness.json`, and
`results_v3_derivative_constraints.json`.

## Positive visibility-ownership checkpoint

`visibility_results/results.json` introduces a textured foreground moving over
a stationary background. No mask, foreground label, or layer count enters
estimation. Every observation contributes a positive precision measure, and
its latent ownership is induced by adjoint coverage
`A_i^*w_i / sum_j A_j^*w_j` rather than a winning frame or layer class.

| Moving-layer case | Best capture | Average | Positive ownership | SSIM |
|---|---:|---:|---:|---:|
| Moderate disocclusion | 21.873 dB | 24.059 dB | 28.994 dB | 0.9393 |
| Folded disocclusion | 20.685 dB | 22.843 dB | 25.617 dB | 0.8886 |

For one observation, a nonpositive barycentric Jacobian still aborts inverse
transport. For several observations, it now only removes an invalid coordinate
preconditioner. The same positive normal equation runs through the original
forward/adjoint fields and audits their joint latent coverage. Five of the six
large-disocclusion sources produce individual folds; all five have zero jointly
unsupported pixels and improve over averaging by 2.454--4.151 dB. The M4 Mini
battery takes 4.68 seconds. The selected artifact SHA-256 is
`86cea60e769ec2cc4db1a628d8ad4bf3115957d3b21ccafa64409cafa295bdbe`.
All five direct-fold trials improve; the robust finest-scale flow action is
what raises the folded aggregate without changing the reconstruction law.

A sharper cycle-derived sensor visibility mask was tested and rejected after
it reduced the moderate control by 0.96 dB. Failed correspondence can mark
unique evidence, so cycle closure remains uncertainty rather than authority to
erase a frame. The positive ownership measure comes from transported coverage.

The one honest failure is woven chirps at the larger displacement: it does not
fold, but the result loses 0.465 dB to averaging. This localizes the next gap to
the single smooth flow estimate. Coverage ownership can route a visible sheet;
it cannot represent two simultaneous motions. The next representation must be
a soft, permutation-symmetric positive multi-sheet flow measure—not foreground
classification and not another reconstruction family.

## Positive multi-sheet representation checkpoint

`multisheet_transport.py` now carries several distinct latent appearances
through the exchangeable formation law
`y_i = sum_s pi_i,s A_i,s x_s`. Every ownership is a continuous positive
simplex measure and every `A_i,s` is the existing exact spatial
forward/adjoint operator. Simultaneously permuting the sheet axis changes
nothing; an executable invariant verifies the reconstructed image and sheet
appearances to `1e-12` after undoing the permutation.

The first battery is intentionally a representation oracle. It supplies exact
continuous opacity and exact stationary/moving sheet geometry for two captures
at +/-3 pixels, then asks whether the positive normal equation can reconstruct
their distinct appearances under sigma 0.002 read noise.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Best capture | 22.350 dB | 0.8587 |
| Unregistered average | 24.987 dB | 0.8747 |
| Known-measure multi-sheet oracle | 55.020 dB | 0.9990 |

All six sources improve over averaging. The minimum/mean gains are
27.591/30.033 dB and the M4 Mini run takes 0.18 seconds. The selected artifact
SHA-256 is
`08c5733c725927641b3283b95b373afa9676aaa61f0c5c3c8408945e8dfcc360`.

This oracle closed the appearance-representation gate exposed by the
woven-chirp failure. The blind checkpoint below now estimates straight and
curved sheet support without a hard foreground/background decision.

## Blind Fourier-circle flow-atlas checkpoint

The first global circle fiber recovered translation but failed curved layered
motion: mean/worst changes from dense were -0.398/-1.959 dB, with only 4/12
positive trials. Overlapping reflected Fourier-circle charts now transport a
local vector field alongside the global circle connection. One positive tensor
measure spans displacement scale and global/local atlas coordinate. Its five
deduplicated quadrature plans carry distinct appearances; forward/reverse
cross-prediction distributes soft mass over all of them without an argmax.

| Straight layered case | Dense flow | Unified flow atlas | SSIM |
|---|---:|---:|---:|
| Moderate disocclusion | 28.994 dB | 36.250 dB | 0.9793 |
| Larger disocclusion | 25.617 dB | 33.323 dB | 0.9772 |

All 12 straight layered trials improve. Mean/minimum gains over dense are
7.481/4.254 dB; mean/minimum gains over averaging are 11.336/3.869 dB. Both
exceed the former global-only checkpoint. With radiometric transport active,
the M4 battery takes 5.67 seconds.
Selected artifact SHA-256:
`65d486328f4e363f99819658f7f3dfedaf4304fec1b721e64321e8ff7553bb80`.

| Curved layered case | Dense flow | Unified flow atlas | SSIM |
|---|---:|---:|---:|
| Moderate curvature | 30.645 dB | 32.137 dB | 0.9772 |
| Larger curvature | 28.595 dB | 28.967 dB | 0.9394 |

The curved control rotates one occluding appearance about the image center
over a static background. All 12 trials improve over dense; mean/minimum gains
are 0.932/0.050 dB. Its M4 battery takes 5.83 seconds. Selected artifact
SHA-256:
`771a7112be4e7b35511c439cd785197370df29dffa8fe1de52af43e907b90d46`.
The rejected global-only artifact is retained with SHA-256
`902a3b4b4de22015bd7e69bde7b1b6dbeb77776746871ff65317763488c8eb22`.

The raw atlas is not universally trusted: on smooth single-connection
deformation it loses 3.710 dB on average and 10.985 dB in the worst trial.
Continuous global coherence, dense disagreement with both atlas coordinates,
local/global motion mass, sparse chart observability, and Eikonal fold pressure
form a positive union of authority. The unified output retains 33.462 dB /
0.9702 SSIM with mean/worst changes from dense of only
-0.000022/-0.000064 dB. Selected preservation artifact SHA-256:
`803619d332b463681c49edbd67e8bc8adc6866f8087c24db9a35f56f37175f7a`.

ABI v3 batches exchangeable plans without merging appearances. Its matched
five-plan atlas profile is bit-exact and 1.195x faster than separate native
crossings; the selected profile SHA-256 is
`556f959f9babc90da8fc1377f14c6fa4dbe05d64ca1fe225a020e74cbe80d302`.
The earlier three-plan profile remains beside it as a historical ledger.
The auxiliary appearance descent remains one warm exact sweep, and estimation
plus authority remain auditable Python.

## Radiometric uncertainty checkpoint

`radiometric_transport.py` moves unequal exposures into their symmetric
geometric-mean gauge and supplies a smooth positive precision near sensor
bounds. This precision enters geometry, cross-prediction, ownership
backprojection, and the multi-appearance normal equation; no frame is selected
or discarded.

| Exposure case | Raw average | Censored symmetric average | Unified atlas |
|---|---:|---:|---:|
| Complementary | 21.608 dB | 22.437 dB | 31.645 dB |
| Severe complementary | 19.968 dB | 21.829 dB | 26.719 dB |

All 12 trials improve over the raw average. Mean/minimum gains are
8.394/2.847 dB, and mean/maximum absolute log-gain errors are 0.0191/0.0937.
The M4 battery takes 5.77 seconds. Selected artifact SHA-256:
`0cc049ea46cdd7a0d0a9c523b289a976ee7c457e439938edfd80f736824156e8`.
The equal-exposure preservation batteries change by at most 0.0055 dB from
their pre-radiometric ledgers.

## Accelerated rolling-shutter checkpoint

A shared constant-velocity row warp remains an unidentifiable pair gauge. The
observable control uses row-dependent relative acceleration plus positive
exposure integration. It enters the existing local Fourier-circle atlas rather
than activating a rolling-shutter solver.

| Rolling case | Average | Dense gauge | Unified atlas |
|---|---:|---:|---:|
| Moderate row acceleration | 23.270 dB | 24.862 dB | 30.912 dB |
| Strong row acceleration | 23.326 dB | 24.660 dB | 29.669 dB |

All 12 trials improve over averaging and dense flow. Mean/minimum gains over
averaging are 6.992/0.842 dB; over dense they are 5.530/1.056 dB. The M4
battery takes 5.87 seconds. Selected artifact SHA-256:
`fff5df883a41b265046fcb8bfd34df0c0005926dc97fda7dd72b9a7f88820862`.

Broad real-capture evidence, calibration of common rolling-shutter gauge,
lens aberration, and turbulence remain open gates.

## Immutable real-capture checkpoint

The first field trial uses two recorded 800×800 blurry exposures from Köhler
scene 1. Both source hashes remain unchanged. A center-only run closes the two
observations to 0.00989 RMS versus 0.01904 for their shared radiometric average,
but amplifies the outer three Fourier annuli by 3.269× and scores below the raw
average against the compressed scene-1 sharp sample on the public page.

`relative_mixing_transport.py` fits the low-frequency log-magnitude covariance
difference after center transport and splits its signed eigenspaces into an
exchange-symmetric minimum-trace pair of positive exposure measures. It reduces
outer-annulus amplification to 2.530× and improves the limited reference check
from 24.868/0.6843 to 24.890 dB/0.6876 SSIM. The immutable average remains
better at 25.156 dB/0.7190, so this checkpoint is explicitly rejected rather
than promoted. The remaining common blur is a pair gauge. See
`REAL_CAPTURE_CHECKPOINT.md` and
`real_capture_results/koehler_checkpoint.json`.

## Twelve-capture real transport checkpoint

The failed pair gate motivated a complete-graph estimator rather than a blur
class or reference-frame choice. For every edge among the 12 Köhler scene-1
web captures, the method measures relative exposure, Fourier-circle center,
and low-frequency mixing covariance. Robust graph closure produces zero-mean
coordinates; a directional linear program finds their minimum-trace positive
covariance realization. All 12 positive measures enter one shared latent solve.
The common blur remains an explicit gauge.

| Candidate | Web-reference PSNR | Web-reference SSIM |
|---|---:|---:|
| Unregistered average | 25.088 dB | 0.7337 |
| Center-transport average | 26.820 dB | 0.7896 |
| Best individual capture, evaluation only | 26.679 dB | 0.7716 |
| Global multi-capture positive transport | 27.893 dB | 0.8349 |
| Adaptive spatial covariance atlas | **28.255 dB** | **0.8413** |

The transported result gains 1.074 dB over the center-aligned average and
1.215 dB over the best individual capture. Forward closure is 0.9672 of the
center-average discrepancy, while the outer-three Fourier-circle ratio is
0.3660; the result does not obtain sharpness by amplifying the ringing bands.
All 13 source/reference hashes remain unchanged.

Algorithm-first optimization preserves the restoration while reducing the
M4 wall time from 79.019 seconds for 64 unit multiplicative passes to 34.943
seconds end to end (2.26×). Exact positive-line descent stops at stationarity
after 17 passes. Shared spectral preparation reduces complete-graph FFT work
from 264 transforms to 24. All 12 global exposure operators retain compact
native ABI v3 form, avoiding 1,950,720,000 bytes (1.817 GiB) of replicated
800×800 coefficient arrays. The native compact operator profile separately
measures 6.45× over its NumPy oracle with `1.33e-15` maximum error.

The final deblurred PNG is bit-identical to the earlier optimal checkpoint
(`618315ae81414edafda511faff984f7c357b2d299e94d15a88bb2ff42eb45256`).
The selected JSON SHA-256 is
`a3d911ca98e8ad5563a8a2f2ea0eb0774638b4f695f5719259d188c834f6134b`.
This global result is retained as the compact baseline. The accepted spatial
continuation below supersedes it on quality. This is one compressed scene-1
web reference rather than the official roughly 200-sample trajectory protocol.
See `MULTICAPTURE_TRANSPORT.md`.

The first spatial-mixing continuation closes covariance graphs on 25
overlapping Fourier charts and blends their positive gauges. On a 64×64
four-capture field whose centered mixing changes from horizontal through
oblique to vertical, averaging scores 26.617 dB, one global covariance scores
31.012 dB, and the local atlas scores 32.518 dB. This is a synthetic
representation gate; no real-capture claim is attached yet.

On a read-only 200×200 proxy of the real 12-capture burst, five chart scales
land between 26.474 and 26.482 dB versus 26.437 dB for the global method. The
first 800×800 transfer nevertheless scores only 27.825 dB; hierarchical
positive shrinkage toward the global graph reaches 27.883 dB. Both are rejected
because neither exceeds the 27.893 dB global gate.

The discrepancy is explained by the second-cumulant validity radius. A fixed
0.16-cycle/pixel fit is no longer low-frequency for native-scale exposure.
Every local edge now uses a broad cached covariance scale to continuously set
`f_max * sigma_delta <= 0.25`, then transports only the
`authority^2 * graph_closure` fraction of its deviation from the full-raster
positive graph. This is uncertainty-weighted convex transport, not chart,
capture, or blur-family selection.

The corrected estimator removes deterministic center motion before measuring
finite local mixing charts. With the same 192-pixel/stride-128 atlas it reaches
**28.31305 dB and 0.842440 SSIM**, gaining 0.420 dB/0.00751 over the global
method and 0.0582 dB/0.00117 over the prior atlas. It closes observations to
0.94928 of center-average discrepancy and lowers the outer-three Fourier ratio
to 0.36275. All source hashes remain unchanged. Frequency radii span
0.0551–0.16 cycles/pixel and local-deviation authority spans 0.0419–0.4880.

Native ABI v6 batches the twelve unchanged generated covariance measures and
runs them concurrently. Storage remains 245,760,192 bytes versus a projected
5,713,920,000 materialized bytes. The selected full run takes 46.463 seconds
and 18 optimal-line passes, a 1.46x end-to-end speedup over the unbatched
center-first run. Selected result JSON SHA-256:
`1ae4592de03b861cee540b307042be1f2ef646efd6afae548c31389bcb078ae8`.
The deblurred PNG SHA-256 is
`ba344d5528b45ac6e947c961cbf62820420dcb1e221e678842344cd2862c88ab`.

## Positive fourth-cumulant checkpoint

`quartic_shape_transport.py` moves beyond covariance without choosing a path
family. A positive three-point axis measure uses side mass `w`, center mass
`1-2w`, and extent `sqrt(lambda/(2w))`; it preserves covariance exactly while
transporting normalized fourth cumulant `1/(2w)-3`. Fourier-circle direction
cells are cross-fitted, their common capture cumulant is removed as an exact
gauge, and fold disagreement continuously tempers authority.

The four-capture shifted-line control improves by approximately 0.05 dB over
fixed Gaussian-matched covariance. The first global real fit is rejected at
27.883 dB/0.8341. When shape is estimated in each accepted covariance chart,
all 49 full-resolution chart authorities become exactly zero and every side
mass remains `1/6`. The result changes the selected checkpoint by only
`-1.2e-5` dB and `-4.7e-7` SSIM. This is verified abstention, not a claimed
real-data gain.

ABI v5 executes constant or spatial side-mass fields and preserves native
parity. The updated profile projects 5,713,920,000 bytes to 245,760,192 bytes;
SHA-256 `5c27a448c8ad402f54108d9d5b19ae667631f3cc61fd78fca2048c57f3637a95`.
The remaining fourth-order gap is the full symmetric tensor, including
cross-axis terms needed by curved/asymmetric paths. See
`QUARTIC_SHAPE_TRANSPORT.md`.

## Full symmetric fourth-cumulant checkpoint

`full_quartic_transport.py` estimates all five symmetric tensor coordinates.
A rank audit rejected the first fourfold-symmetric dictionary, which spanned
only three coordinates. The corrected one-axis dictionary has rank five, and
a joint positive program solves one common K4 gauge with an analytic gradient.

The raw relative-K4 member in the 96×96 M4 battery gains 1.851 dB for
moderate directional blur with sensor noise and 3.452 dB for strong directional
blur. It also exposes the common-gauge limit: an opposed-direction case loses
0.670 dB. `quartic_gauge_posterior.py` reconstructs both covariance and K4
positive measures and transports forward closure plus absolute outer
Fourier-circle redistribution into continuous mass. Its posterior gains are
2.129 dB moderate, 2.487 dB strong, and 0.227 dB opposed, while the exact null
changes by only -0.0004 dB. The failure remains visible without becoming a
winner branch.

The real estimation-only gate earns 0.0280 shape authority and reduces held-out
log-magnitude RMS from 0.59193 to 0.57233. The exact even-reflection FFT
operator reduces an 800×800,
153-atom representation from 5.484 GB to 20.535 MB (267.03x); M4 forward and
adjoint times are 0.0380 and 0.0332 seconds. The real posterior retains 0.1402
K4 mass and reaches 27.89381 dB / 0.83501, slightly above its global covariance
member but below the selected spatial atlas. All 49 center-first spatial K4
charts have exactly zero held-out authority, so the higher-capacity spatial
operator is correctly not constructed. Four-way FFT batching cuts posterior
runtime from 134.096 to 70.997 seconds.
Battery SHA-256:
`2d465505b91f352da6d4505bd1a3031dc21568254c39dcfa09693314e7bd849c`.
Real probe SHA-256:
`34cb8e2d32f2bc062b1ea35da25aefaf8e9b88a3e0dee4ce2794621c2a5a1ba5`.
Image posterior SHA-256:
`72ad98da965159682ddcf812b1ababf4508c43c58f41e8a76b157460e267e6a6`.
Spatial K4 probe SHA-256:
`46f7b9aed709a20cb7b642f4c0900b25beedb6c31ad1cd6a2a67b0092bda05a8`.
See `FULL_QUARTIC_TRANSPORT.md`.

## Center / inverse / noise uncertainty transport

One continuous posterior is used for all 23 sources and all five observation
formations. Closure evidence concentrates with capture count, local atlas
authority becomes spatial mass, and FMMT displacement is gated by coherent
innovations and cross-capture fine-scale replication. It chooses no blur
family, source, or capture.

Relative to center transport, mean PSNR changes are +0.6835 dB complementary,
+1.1087 dB spatial covariance, +0.0457 dB rotational warp, +0.0017 dB common
blur, and +0.3041 dB photon-limited; overall gain is +0.4287 dB. The 17-source
V3-era chronological holdout is reported separately with method inheritance
explicitly set to none.

The same fixed posterior scores 28.29267 dB / 0.841890 SSIM on the limited real
checkpoint. It retains 0.98519 mean inverse mass and an explicit 0.01481 center
reserve; FMMT mass is 1.35e-7. This is the uncertainty-aware candidate, while
28.31305 dB remains the maximum-quality atlas checkpoint. Exact bounded RGB
FMMT batching reduces the real run from 98.03 to 77.63 seconds; its posterior
outer-three Fourier ratio is 0.35750 with zero envelope excursion. Generalization JSON
SHA-256: `88d35fd134419526205e1bf03f93b79cc02c11b04b8458d4df09cbdad17a7e68`.
Real JSON SHA-256:
`42a7af005be7f8685680c27ac30d63edcf9303adeb19f7c62f616f1e5d35d54a`.

## Local fixed-point anti-ringing checkpoint

`run_ringing_benchmark.py` adds canonical vertical-step and sparse-point
controls to the six natural-source scaffold. It reports PSNR, SSIM, broad halo
RMS, oscillatory error RMS, and weighted local phase-shift spread for Gaussian,
line, curved, and random positive exposure measures. Noise sigma is 0.002 and
all boundaries use reflection.

The failure was not the auxiliary line recurrence. Positive descent itself
could populate repeated near-null bands, and the exact path refinement could
then treat them as longitudinal evidence. A global Fourier-floor increase was
rejected because it removed valid curve/random detail. The accepted law uses
the exact fixed point of positive transport: locally constant observations
cannot support any inverse correction. Transported local variance continuously
gates corrections in both descent stages at scale 0.004, with no blur or edge
classifier.

| Formation | Previous PSNR / SSIM | Fixed-point PSNR / SSIM | Change |
|---|---:|---:|---:|
| Gaussian | 27.380 / 0.82992 | 27.353 / 0.83017 | -0.028 dB / +0.00025 |
| Line | 29.886 / 0.82023 | 30.160 / 0.86283 | +0.274 dB / +0.04260 |
| Curve | 31.674 / 0.88170 | 32.081 / 0.90261 | +0.407 dB / +0.02091 |
| Random path | 33.174 / 0.89522 | 33.403 / 0.91793 | +0.229 dB / +0.02271 |

On the canonical line edge, the unified result changes from 26.407 dB /
0.76571 SSIM to 26.692 dB / 0.93862 SSIM. Amplified residual images show that
the repeated full-field copies have collapsed to one narrow edge-local error.
The Gaussian control has no broad ring; sparse-point errors remain localized.

First and second transported moments are batched into one operator call. A
second numerical trim omits the rotated straight-line recurrence when its
maximum possible pixel authority is below `1e-6`; random-path authorities were
`1e-9`--`1e-8`, while all line authorities remain active at
`0.0058`--`0.0344`. Three M4 runs average 7.599 seconds versus 8.372 seconds
before the trim, a 9.2% improvement. Maximum measured drift is
`5.96e-6` dB PSNR and `7.17e-9` SSIM. The full suite passes 107 tests with one
intentional fixture skip.

Selected JSON SHA-256:
`bcb81c8562344410c7dec7a4bbab1810e302ddfa21fa1bbdabac73cf7f7e11e5`.
Canonical line PNG SHA-256:
`51794eef02949cff50da446f48b7b3c3daea98c980d54e56b8bc54839127a943`.
