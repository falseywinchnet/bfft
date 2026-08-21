# First immutable real-capture checkpoint

## What was tested

Two 800×800 recorded blurry exposures from Köhler scene 1 were passed to the
same symmetric radiometric, dense-center, Fourier-circle atlas, and positive
transport pipeline. Source hashes were computed before and after every run.
The deblurrer wrote only separate reconstructions, predicted observations,
uncertainty images, and JSON ledgers.

`real_capture_evaluation.py` does not convert no-reference sharpness into a
truth score. It reports:

- matched forward closure against both radiometrically transported captures;
- edge-energy concentration;
- local observation-envelope excursion;
- normalized Fourier-circle amplification; and
- transported residual uncertainty.

The compressed scene-1 sharp JPEG on the benchmark page provides a limited
check with a 32-pixel border crop. The page's other three ground-truth JPEGs
belong to three distinct base images and are not used. This is not the official
Köhler protocol, which searches roughly 200 sharp trajectory samples.

## Center-only result: rejected

The center-only atlas reduced forward residual relative to a shared average,
but amplified the outer three Fourier annuli by 3.269×. Against the scene-1 web
reference it reached 24.868 dB and 0.6843 SSIM, below the untouched average's
25.156 dB and 0.7190 SSIM. This exposed a formation error: setting residual
exposure extent to zero treats a long exposure as a sharp warped sample.

## Relative centered-mixing transport

For centered convolutional exposure at low spatial frequency,

```text
log |Y_2(f) / Y_1(f)|
  = -2 pi^2 f^T (C_2 - C_1) f + O(|f|^4).
```

The scene spectrum and deterministic translation phase cancel. A robust fit on
supported Fourier coefficients estimates only `Delta C = C_2-C_1`. Its signed
eigendecomposition supplies the exchange-symmetric minimum-trace positive
gauge

```text
C_1 = Q max(-Lambda, 0) Q^T,
C_2 = Q max( Lambda, 0) Q^T.
```

Each covariance becomes a centered positive sigma measure with exactly matched
second moment and is convolved with the existing spatial exposure field. This
is neither frame selection nor blur-family classification. Any covariance
common to both observations remains explicitly unidentifiable.

On a held-out 3-pixel versus 9-pixel line-exposure control, automatic relative
mixing improves 24.478 dB averaging to 25.841 dB. Its estimate swaps exactly
under observation exchange.

On the real pair it lowers outer-annulus amplification from 3.269× to 2.530×,
edge concentration from 1.119× to 1.062×, and improves the center-only reference
check to 24.890 dB and 0.6876 SSIM. It still remains below the raw average and
therefore fails the real-capture acceptance gate.

## What the failure establishes

The pair identifies different center translations and part of the differential
mixing covariance. It cannot identify the large blur shared by both long
exposures. Forward closure alone is insufficient: a model can reproduce blurry
observations while remaining blurry or producing high-frequency echoes.

The next justified extension is multi-observation exposure consensus, or an
independent trajectory constraint such as gyro/inertial data. Several blur
covariance differences can close cycles and reduce uncertainty about individual
exposures, but their common positive component remains a gauge unless one
capture or an external measurement constrains it.

## Twelve-capture continuation: accepted limited checkpoint

The multi-observation extension now closes all 66 edges of the 12-capture
scene-1 burst without selecting a reference capture or blur family. It improves
the center-transport average from 26.820 dB/0.7896 SSIM to 27.893 dB/0.8349,
and exceeds the best individual-capture evaluation oracle at 26.679 dB/0.7716.
The outer-three Fourier ratio is 0.3660, so the gain is not a recurrence of the
pair checkpoint's ringing amplification. All source hashes remain unchanged.

This accepts the multi-capture method on one limited field case while leaving
the pair result rejected and the shared-blur gauge unresolved. It does not
replace the official trajectory-sample protocol or establish broad camera
generalization. Full formation equations, optimization measurements, hashes,
and remaining gates are in `MULTICAPTURE_TRANSPORT.md`.

The first accepted continuation made mixing spatial without independently trusting
every chart gauge. Each local covariance deviation is continuously shrunk
toward the full-raster graph by its squared relative authority and graph
closure. Its Fourier ceiling is set by the measured differential extent so the
second-cumulant approximation remains locally valid. This first raised the
immutable checkpoint to 28.255 dB/0.8413 SSIM. Applying deterministic center
transport before measuring finite local charts now raises it to **28.31305
dB/0.842440 SSIM**, with 0.94928 relative forward closure and a suppressive
0.36275 outer-Fourier ratio. The unshrunk 27.825 dB, fixed-band hierarchical
27.883 dB, and pre-center-pullback 28.255 dB runs remain preserved as reasoning
checkpoints.

Native ABI v6 generates the positive nine-point covariance measure directly
from compact eigenaxis fields, reduces projected operator storage by 23.25x,
and runs the twelve capture operators concurrently. The selected 18-pass M4
run takes 46.463 seconds. Its source scope and unresolved common-blur gauge are
unchanged.

Both axis-separable and full-rank fourth-cumulant probes preserve this
selection. The global full-K4 member retains 0.1402 posterior mass but remains
below the atlas; all 49 full-rank spatial chart authorities abstain exactly.
`QUARTIC_SHAPE_TRANSPORT.md` and `FULL_QUARTIC_TRANSPORT.md` record the positive
controlled gains and real-data uncertainty boundary.

## Uncertainty-aware continuation

The center-first atlas now feeds one continuous center/inverse/noise posterior.
No blur type is selected. On this immutable burst it assigns 0.98519 mean mass
to the inverse (0.97327-0.98570 spatially), 0.01481 to center transport, and
1.35e-7 to FMMT noise transport. It reaches 28.29267 dB / 0.841890 SSIM, only
0.02037 dB below the maximum-quality atlas, while explicitly carrying the
between-measure uncertainty. Source hashes remain unchanged. This accepts an
uncertainty-aware candidate without broadening the single-scene claim.
Exact three-way RGB FMMT batching reduces the run from 98.03 to 77.63 seconds.
The posterior also lowers the outer-three Fourier ratio from the atlas's
0.36274 to 0.35750 and has zero local observation-envelope excursion.
