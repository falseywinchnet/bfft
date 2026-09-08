# Run all three bands before trying to save work

**The direct experiment saves substantial frame time in the classification-heavy
aperture scene. It is not a universal speedup or an equivalent replacement.**

The experimental renderer unconditionally integrates all three wavelength bands
at every primary dielectric hit. It uses N equally weighted midpoint samples per
band, with each sample executing the existing dielectric radiance transport,
including its reflected and transmitted contributions. It performs no initial
three-wavelength classification, no spectral signature comparison, no spectral
subdivision, and no merging of classified wavelength intervals.

The existing optical cutoff and internal-return budgets remain in force. This
experiment does not introduce a new recursive branch-pruning rule. It also does
not implement an exact continuous spectral field: midpoint quadrature is a finite
numerical approximation of each entire band, whose resolution is stated below.
It never substitutes only the three channel-center wavelengths.

## Frame timings

M4 Mini CPU; 800×600; terminal error 1; shared child laws enabled; BVH disabled;
existing spatial coverage, source field, and optical laws. Two complete frame
runs per mode, in forward/reverse order, using the same compiled scene field.
Times include per-frame camera field construction and all camera passes, but
exclude static scene/beam/transport-field construction. These are two-run means,
not a confidence interval or a general hardware guarantee.

| Scene | Current | Direct N=4 | Direct N=8 | Direct N=16 |
|---|---:|---:|---:|---:|
| Aperture canyon | 1,221.5 ms | **483.0 ms** | **693.3 ms** | **1,111.5 ms** |
| Mirror relay | 690.5 ms | **555.6 ms** | 742.8 ms | 1,118.2 ms |
| Occlusion garden | **366.3 ms** | 443.7 ms | 586.8 ms | 878.2 ms |
| Standard | **172.0 ms** | 211.0 ms | 286.4 ms | 438.7 ms |

Aperture N=4 is 2.53× as fast (60.5% less frame time); N=8 is 1.76× as fast
(43.2% less). Mirror N=4 is 1.24× as fast. N=32 and N=64 were also measured;
in aperture they take 1,958.5 and 3,673.3 ms. Unconditional high-resolution
transport is expensive, and cheaper scenes have less classification work to
save. All 48 primary comparison frames and their counters are archived.

In the aperture scene, N=8 changes child classifications from 1,972,999 to
593,357 (69.93% fewer). The remaining calls are **spatial** ownership, edge, and
coverage classifications; this comparison isolates removal of the spectral
prepass. It does not claim to remove every camera classification.

Emitter quadrature samples fall from 19,572,874 to 10,684,071 at N=8, and to
6,338,553 at N=4. Thus the savings include fewer expensive radiance evaluations:
the original spectral trees sometimes produce many more final shading regions
than these unconditional quadrature grids have samples. The result is not a
measurement of classifier overhead alone.

## Image differences and convergence

The methods do not render identical images. The old method conditionally
integrates prism bands and otherwise evaluates channel centers; the new method
integrates bands at every primary dielectric hit. The old output is therefore
not a ground-truth target for this comparison.

A separate aperture frame with **256 samples per band** serves as a denser
numerical reference. It retains the same spatial reconstruction. Differences
below are across the 1,440,000 tone-mapped channel bytes:

| Method | RMS byte difference vs N=256 | Maximum difference | Channel bytes differing by >1 |
|---|---:|---:|---:|
| Current classifier-driven integration | 0.5294 | 232 | 729 |
| Direct N=4 | 0.2643 | 111 | 954 |
| Direct N=8 | 0.1244 | 87 | 528 |
| Direct N=16 | 0.0679 | 37 | 233 |
| Direct N=32 | 0.0426 | 24 | 78 |
| Direct N=64 | 0.0289 | 19 | 31 |

Small whole-frame averages hide sparse local errors. N=8 is both faster and
closer in this RMS measure than the current aperture result, but its maximum
87-byte difference is substantial and it does not meet a universal one-byte
error bound. N=4 has a lower RMS than the current method but more bytes above
one. Neither should be called an equal-quality replacement solely from this
screen.

The comparison figure shows a magnified prism region. Some of the largest
changes from the current result persist in the denser direct evaluations;
direct quadrature is not merely introducing random disagreement with an
otherwise exact baseline. Nevertheless, N=256 is itself approximate, and the
retained spatial reconstruction can introduce additional error.

## Independent ray checks

A separate probe evaluates 56 fixed camera rays: 48 evenly selected from a
coarse raster of primary dielectric hits, plus eight positions selected for
large frame differences. Every direct mode N=4 through N=1024 asserts that it
performs **zero child classifications**, and every returned radiance is finite
and nonnegative. Fixed rays avoid comparing different spatial reconstruction
samples between modes.

Against N=1024, linear-RGB RMS differences are 0.6163 for the old method,
0.1355 at N=4, 0.0701 at N=8, 0.0409 at N=16, and 0.00983 at N=64. They fall
to 0.000838 at N=512. This supports convergence on the measured rays. The
outlier-selected subset is explicitly not a representative image-quality
benchmark or a proof against arbitrarily narrow spectral visibility features.

## Scope and implementation

This is a runnable experiment; production defaults are unchanged. The generator
`prepare_direct_spectral.py` makes a temporary comparison translation unit from
the authoritative native renderer. Every substitution checks for a unique
source anchor. The frame and ray probes compile against that generated source.
No classification algorithm has been replaced by a different admission test.

The result supports the user's hypothesis: trying to reduce transport work can
cost more than simply transporting all bands, and can miss within-band behavior.
The measured limit is equally clear: increasing fixed quadrature resolution
moves the cost back into repeated transport. This experiment is evidence for
direct band transport, not completion of a shared continuous-band representation.

Reproduce on the Mini:

```sh
experiments/photonic_transport_field/run_direct_spectral.sh
```

The script uses `m4build`, copies each run's artifacts immediately, and includes
the denser aperture frame and 56-ray check. Render the local saved artifacts:

```sh
.venv-jpeg/bin/python experiments/photonic_transport_field/plot_direct_spectral.py
```

Raw JSON, PPM frames, compact `summary.json`, and `comparison.png` are in
`direct_spectral_m4/`.

## Follow-up: one unconditional sample per band

The omitted baseline has now been measured: exactly one representative wavelength
at the center of each band, unconditionally, with no spectral classification.
This gives up within-band integration explicitly. The same transport law and
spatial coverage remain in use.

Four scenes, 800×600, three fresh runs per mode in alternating order. The table
uses medians; these fresh current-renderer times supersede cross-run comparisons
with the earlier two-run screen when assessing N=1.

| Scene | Current | One per band | Speedup |
|---|---:|---:|---:|
| Aperture canyon | 1,198.9 ms | **324.8 ms** | **3.69×** |
| Mirror relay | 678.8 ms | **419.9 ms** | **1.62×** |
| Standard | 172.2 ms | **155.2 ms** | **1.11×** |
| Occlusion garden | 357.9 ms | **328.4 ms** | **1.09×** |

One per band is faster in every tested scene. In aperture the three direct
measurements are 324.39–325.33 ms. Child classifications fall from 1,972,999 to
593,045; the remaining classifications belong to spatial coverage. Emitter
quadrature falls from 19,572,874 to 3,173,526 (83.8% fewer evaluations).

At full-frame viewing size the images appear very similar. The aperture N=1
image differs from the current image in 1,389 of 480,000 pixels; 1,090 pixels
have any channel differing by more than one 8-bit level, and 502 by more than
eight. Thus 99.77% of pixels agree within one level per channel. Maximum local
channel difference is 187: small affected area does not mean every difference
is small. Against N=8, 1,370 pixels differ by more than one level (0.285% of the
image), with maximum local difference 232. No perceptual study or motion
stability claim is made.

The output supports N=1 as the practical performance baseline for further work.
Extra spectral samples have not demonstrated enough visible benefit here to
justify assuming them as the starting point. This is an experimental comparison;
no production default was changed by this follow-up.

Reproduce:

```sh
experiments/photonic_transport_field/run_one_per_band.sh
.venv-jpeg/bin/python experiments/photonic_transport_field/plot_one_per_band.py
```

The comparison probe's optional fourth argument selects the number of repeats;
its third argument selects a single direct sample count to compare with the
current renderer. The wrapper compiles via `m4build`, runs all four scenes,
and copies every result immediately. All 24 frame records were checked, and
repeated current-renderer frames matched the first current frame exactly.
Results and images are `one_per_band_*.json*`, `one_per_band_summary.json`,
`one_per_band_comparison.png`, and `one_per_band_scenes.png` in
`direct_spectral_m4/`.
