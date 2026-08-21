# Exchange-symmetric multi-capture transport

## Formation model

The multi-capture path treats every exposure as one positive measure acting on
one shared latent image. It does not classify a capture as line, disk, curve,
or sharp, and it never selects a reference frame for reconstruction. For each
complete-graph edge `(i,j)`, it measures three relative quantities:

```text
g_j - g_i       relative log exposure,
c_j - c_i       Fourier-circle phase center,
C_j - C_i       low-frequency centered-mixing covariance.
```

Robust graph least squares closes all edges in exact zero-mean gauges. The
radiometric coordinate transports clipped samples as soft sensor precision.
The center coordinate is removed before mixing. A directional linear program
then finds the minimum-total-trace positive realization of the covariance
graph. This is a computational gauge, not evidence that any capture is sharp:
one positive covariance common to every exposure remains unidentifiable.

Each covariance becomes a centered positive sigma measure. All measures and
all transported sensor precisions enter one shared-latent matched
forward/adjoint solve. Simultaneously permuting the observations only permutes
the graph coordinates and fields; the reconstructed latent is invariant to the
tested tolerance.

## Descent and execution

The solver's `optimal_positive_line` step uses the positive multiplicative
transport direction but solves its weighted quadratic line coefficient
analytically through the same forward operators:

```text
d = x * (correction - 1)
alpha = <y-Ax, A d>_W / <A d, A d>_W.
```

The coefficient is bounded by radiance and positivity, and the run stops at
line stationarity. The real checkpoint uses 17 accepted passes rather than
spending all 64 permitted unit updates.

Observation-only Fourier state is prepared once per capture. For 12 captures,
the complete graph therefore uses 24 FFTs—12 phase spectra and 12 magnitude
spectra—instead of recomputing 264 FFTs across 66 edges.

Global positive measures retain scalar interpolation coefficients through
barycentric pullback. Native ABI v4 then executes 12 compact plans. At 800×800,
this avoids 1,950,720,000 bytes (1.817 GiB) of replicated spatial coefficient
arrays while preserving the NumPy spatial operator as the parity oracle.

## Synthetic acceptance

The four-capture shifted, differently oriented line control improves the raw
average from 22.742 dB to 26.592 dB. A capture permutation changes the latent
by at most `6.38e-15` in the recorded run, and three identical captures take
the exact zero-measure fast path. At eight passes, optimal positive-line
descent improves on the unit multiplicative update.

## Immutable Köhler scene-1 checkpoint

The public 12-capture scene-1 web JPEG burst and its single scene-1 sharp web
JPEG are read without mutation. This is not the official Köhler evaluation,
which searches roughly 200 sharp trajectory samples.

| Candidate | PSNR | SSIM |
|---|---:|---:|
| Unregistered average | 25.088 dB | 0.7337 |
| Center-transport average | 26.820 dB | 0.7896 |
| Best individual capture, evaluation-only oracle | 26.679 dB | 0.7716 |
| Multi-capture positive transport | **27.893 dB** | **0.8349** |

The reconstruction gains 1.074 dB over the center-transport average and 1.215
dB over the best individual-capture oracle. Its forward-closure ratio is
0.9672, and its outer-three Fourier-circle ratio is 0.3660 rather than a
ringing amplification. Every source hash is unchanged.

The final M4 run took 34.943 seconds, versus 79.019 seconds for the original
64-pass multiplicative checkpoint, a 2.26× end-to-end speedup. The saved
8-bit reconstruction is bit-identical to the earlier optimal checkpoint:

```text
deblurred.png SHA-256
618315ae81414edafda511faff984f7c357b2d299e94d15a88bb2ff42eb45256

results.json SHA-256
a3d911ca98e8ad5563a8a2f2ea0eb0774638b4f695f5719259d188c834f6134b
```

The global baseline ledger is
`real_capture_results/personal_deblurrer_koehler_multicapture_compact12/`.
One scene and one compressed reference establish a successful field
checkpoint, not broad real-camera generalization. Spatially varying mixing,
non-covariance exposure shape, saturation ownership, turbulence, and the
common-blur gauge remain open.

## Spatial mixing atlas foundation

The next stage replaces one covariance per capture with overlapping local
covariance graphs. Each chart prepares one magnitude spectrum per capture,
closes every pair edge, solves a chart-local positive gauge, and contributes
through a Gaussian spatial mass. A convex combination of positive covariance
matrices remains positive. Each resulting 2×2 field is realized exactly as a
nine-atom spatial sigma measure and enters the unchanged shared solver.

On the first 64×64 control, centered mixing changes continuously from
horizontal to vertical and oblique across four captures. The raw average is
26.617 dB, the global-covariance method reaches 31.012 dB, and the 25-chart
local atlas reaches 32.518 dB. The estimated covariance eigenvalue floor is
positive and the trace varies spatially by more than 4×. This is a synthetic
formation/representation gate.

A read-only 200×200 Lanczos proxy of the 12-capture scene-1 burst was
directionally consistent: global covariance reached 26.437 dB, while five
local chart scales spanned 26.474–26.482 dB. The first full-resolution atlas
then failed to transfer, reaching only 27.825 dB. Hierarchical shrinkage toward
the full-raster graph improved it to 27.883 dB but still failed the 27.893 dB
global gate. Both artifacts remain in `real_capture_results/` as rejected
reasoning checkpoints.

The failure exposed the fixed Fourier radius rather than a need for a blur
class. The covariance identity is only second-order:

```text
log |Y_j/Y_i| = -2 pi^2 f^T (C_j-C_i) f + O(|f|^4).
```

At native blur scale, fitting every chart through 0.16 cycles/pixel admits too
much fourth-order path shape. Every edge now performs a broad cached scale
probe and continuously contracts its analytical radius so that
`f_max * sigma_delta <= 0.25`, bounded by available Fourier resolution. No
edge or chart changes family or reconstruction branch. Chart covariance is
then shrunk toward the global positive graph by
`authority^2 * graph_closure`; this is a convex positive blend, and every chart
retains positive spatial mass.

## Accepted center-first adaptive spatial checkpoint

The 192-pixel, stride-128 adaptive atlas first cleared the full-resolution
gate at 28.255 dB / 0.8413 SSIM. That version estimated local mixing before
explicitly pulling deterministic center motion out of each finite chart.
Global Fourier magnitude is translation-invariant, but a cropped local chart
is not. The corrected order is now:

```text
radiometric gauge -> deterministic center transport -> local mixing atlas
```

The same atlas and reconstruction gate then improve again:

| Candidate | PSNR | SSIM |
|---|---:|---:|
| Global covariance baseline | 27.893 dB | 0.8349 |
| Prior adaptive atlas | 28.255 dB | 0.8413 |
| Center-first adaptive atlas | **28.313 dB** | **0.84244** |

It gains 0.420 dB and 0.00751 SSIM over the global method, 0.0582 dB and
0.00117 SSIM over the prior atlas, 1.493 dB over the center-transport average,
and 1.635 dB over the best individual-capture oracle. Forward closure improves
from 0.9672 globally and 0.9516 in the prior atlas to 0.94928 of center-average
discrepancy. The outer-three Fourier ratio falls from 0.4104 to 0.36275. All
source hashes remain unchanged. Local-deviation authority spans 0.0419–0.4880
and fitted upper frequency spans 0.0551–0.16 cycles/pixel.

The generated covariance ABI stores the exact established nine-point positive
sigma measure as four eigenaxis components per pixel. It never materializes
atom fields, source-index planes, or coefficient planes. ABI v6 batches all
twelve independent capture operators in one crossing and executes them
concurrently; it does not merge, rank, or select captures. The selected M4 run
takes 46.463 seconds and 18 optimal-line passes, versus 67.841 seconds for the
same center-first algorithm before batching (1.46x end-to-end).

The storage and parity invariants remain those of the generated operator:
245,760,192 bytes instead of a projected 5,713,920,000 materialized bytes,
with forward/adjoint parity below `5e-13`.

Selected artifacts:

```text
real_capture_results/personal_deblurrer_koehler_multicapture_center_first_atlas_v6/
results.json  1ae4592de03b861cee540b307042be1f2ef646efd6afae548c31389bcb078ae8
deblurred.png  ba344d5528b45ac6e947c961cbf62820420dcb1e221e678842344cd2862c88ab
```

The prior accepted artifact is retained under
`personal_deblurrer_koehler_multicapture_adaptive_atlas/`; it is no longer the
selected reasoning checkpoint.

Historically, ABI v4 established `4.40e-13` forward and `2.45e-13` adjoint
parity plus the 23.25x storage reduction. Its 67.387-second result remains a
prior representation checkpoint, not the selected runtime.

Prior artifacts:

```text
real_capture_results/personal_deblurrer_koehler_multicapture_adaptive_atlas/
results.json  7482564f3937ec1429c70f71603f2c58fce500ce6570bc1694e5f9bcdb0c502c
deblurred.png  4f08be0fd11d1e2cea27321dd9de1d6c2545c131bfc5caeb545ceb20ff80ca88
covariance_native_profile.json
                30a2c88f5767aa4a44a7d491b3e843d97448dd54710a641d7b6a3788e62baebe
```

This remains one compressed web reference, not the official roughly
200-sample trajectory evaluation or broad real-camera proof. Non-covariance
path shape, common blur, saturation ownership, turbulence, and additional
recorded bursts remain open.

ABI v5 subsequently adds constant or spatial non-Gaussian axis side masses.
The full-burst cross-fitted quartic field abstains, leaving this adaptive
covariance checkpoint selected. See `QUARTIC_SHAPE_TRANSPORT.md`.

## Continuous center / inverse / noise posterior

`multicapture_posterior.py` keeps deterministic center transport, the positive
spatial mixing inverse, and FMMT noise transport present in one continuous
measure. Forward-closure gain concentrates at resolution `0.08/sqrt(N)`;
overlapping chart authorities reconstruct a spatial inverse-mass field;
coherent closure residual suppresses noise transport; and fine structure
replicated across captures is protected. No blur, noise, source, or capture is
selected.

One fixed policy was measured on 23 sources and five formations. The six
denoiser fixtures are reported separately from 17 V3-era scikit-image files;
the latter are a chronological data holdout with no method inheritance.
Relative to center transport, posterior PSNR gains are 0.42874 dB overall,
0.68350 dB complementary mixing, 1.10874 dB spatial mixing, 0.04565 dB
rotational warp, 0.00170 dB common blur, and 0.30409 dB photon-limited. The
worst complementary loss contracts from -1.4475 to -0.1390 dB and the worst
rotational loss from -3.1210 to -0.2873 dB.

On the twelve-capture Koehler checkpoint, the same posterior carries 0.98519
mean inverse mass (spatial range 0.97327-0.98570), only 1.35e-7 FMMT mass, and
reaches 28.29267 dB / 0.841890 SSIM. This is 0.02037 dB below the unconstrained
atlas while carrying the remaining center uncertainty explicitly. All source
hashes remain unchanged. Bounded three-way RGB FMMT scheduling is bit-exact to
the serial channel oracle and reduces end-to-end runtime from 98.03 to 77.63
seconds. The posterior outer-three Fourier ratio is 0.35750 and its local
observation-envelope excursion is exactly zero.

```text
center_first_generalization_v7.json
  88d35fd134419526205e1bf03f93b79cc02c11b04b8458d4df09cbdad17a7e68
real_capture_results/personal_deblurrer_koehler_multicapture_posterior_v9/
  results.json  42a7af005be7f8685680c27ac30d63edcf9303adeb19f7c62f616f1e5d35d54a
  posterior_deblurred.png  de0f50aa0d3d25c0c3e1b43a7313f506449d18f65e1ac305cdcf14cb5d231927
```
