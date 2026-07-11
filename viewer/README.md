# BFFT IQ Waterfall Viewer

A streaming, zoomable IQ waterfall viewer. The heavy lifting (file I/O, sample
conversion, windowed complex FFTs) lives in a monolithic C++ library,
`libiqwaterfall`, built on the project's BFFT real-FFT kernel. The UI is a thin
DearPyGui front-end.

## Components

| File | Role |
|------|------|
| `iqwaterfall.cpp` | Monolithic backend: RIFF/RF64/BW64 WAV and raw-IQ reader, BFFT waterfall, reassignment, and complex-IQ FCT renderer. Flat C ABI. |
| `dip_algo.cpp` | C++ port of `active_delta_center5_fast1` (DIP/finite-Zak walk + in-house PGHI seed + record OLA), real **and** complex-generalized solvers. Same library. |
| `iqwaterfall.py`  | `ctypes` wrapper: `IQSource`, `Waterfall`, reassignment, DIP, and combined `dip_unified` APIs. |
| `two_lattice.py` | NumPy executable specification: shared one-step DIP seed, independent long/short magnitude projection, tile OLA. |
| `dip_stream.py` | Legacy shared-state/claim experiments and reconstruction utilities; not the live unified viewer path. |
| `superres.py` | Python reference for the phase-aware reassigned spectrogram. |
| `make_sr_fixture.py` | Synthesizes the known-truth SR fixture (tone pair + click train + chirp). |
| `sr_fusion_study.py` | Quantified fusion quality study (click width, halo, tone dip, gate noise). |
| `validate_modes.py` | Headless comparison of the three live modes. |
| `test_fusion_support.py` | Cross-frame short-support routing regression, including NB=1024/NS=512. |
| `archive/validate_dip.py` | ARCHIVED: bisected port check; its Python reference module (`active_delta_center5_cpp_reference`) was never checked in, so it cannot run. `validate_modes.py` + `sr_fusion_study.py` cover the live paths. |
| `iq_waterfall_app.py` | DearPyGui viewer (transport, zoom, dynamic range, colormaps, settings). |
| `archive/fct_view.py` | ARCHIVED intrinsic-FCT viewer mode (stream + endpoint scatter). The FCT library itself stays shipped and tested. |
| `build.sh` | Compiles `libiqwaterfall.{dylib,so}`, linking an isolated Release `libbfft.a`. |

## Build & run

```bash
cd viewer
./build.sh                    # produces libiqwaterfall.dylib
../.venv/bin/pip install dearpygui   # one-time
../.venv/bin/python iq_waterfall_app.py
```

## Notes on the FFT path

BFFT is a real-to-complex kernel. Complex IQ spectra are computed with the
two-real-FFT identity: `X = FFT(I) + j·FFT(Q)`, reconstructing the full N bins
from the two Hermitian half-spectra, then `fftshift`. Verified against
`numpy.fft.fftshift(fft(...))` to ~1e-6 dB.

## Format handling

- **Auto (WAV header)**: parses RIFF, RF64, and BW64 (`ds64`, `fmt `, `data`);
  2-channel ⇒ complex IQ, 1-channel ⇒ real. The header sample rate is used.
- **Raw**: pick the sample format + sample rate + complex/real in the UI. Used
  for headerless SDR captures (`.iq`, `.raw`, `.cs16`, ...).

## Analysis modes

The mode selector exposes three different measurements:

1. **Streaming STFT** — a conventional symmetric-Hann complex spectrogram,
   center-timed and cheap enough for continuous playback.
2. **Super-resolution (two-aperture)** — a waveform-domain combination of the
   two useful SR observables. The Python `active_delta_center5_fast1` port gives
   a smooth coherent shared-state seed (PGHI → one L5 step → consensus). One
   literal Python alternating projection then applies independent long and
   short magnitude families to that seed. Readout is the long reassigned STFT
   on the short COLA lattice. There are no row claims, gates, or image-space
   products. Geometry has one knob: long `N`, short `N/4`, internal short-COLA
   hop `N/8`, and external comparison hop `N/4`. Thus SR 4096/1024 aligns with
   conventional STFT N=2048/H=1024 without changing the solve. Reassignment is
   evaluated directly on those external centers and conservatively splatted to
   the nearest 2×2 time/frequency cells; no internal rows are discarded and no
   post-transform floor is applied.
3. **Reassigned STFT** — ordinary symmetric-Hann phase reassignment, retained
   separately so the effect of the two-aperture solve remains inspectable.

The intrinsic-FCT viewer mode is archived in `archive/fct_view.py`; the exact
FCT transform itself remains a shipped, tested library feature.

The live combined call is native C++ and agrees with the NumPy executable
specification to 1.1e-13 worst case for N=512..4096. Representative 8192-sample
IQ-tile times are about 16/18/18 ms for N=1024/2048/4096. Full complex frame
FFTs use two SIMD-native BFFT real transforms and Hermitian recombination on
every platform (Accelerate remains a benchmark override on macOS). Only the five demanded central
attachment matrices are constructed, using the closed-form geometric sum;
this removed the former 185 ms large-N setup wall.

Mode and transform-size changes leave the position marker sample completely
unchanged. SR tile caches remain alive across A/B mode changes, so returning to
the unified view neither changes the marker nor reruns completed PGHI tiles.

- [x] Monolithic backend, IQ reader, BFFT waterfall — built & verified.
- [x] DearPyGui viewer: open, play/pause/stop, scrub, frequency zoom,
      dynamic range, colormap/window/FFT-size settings.
- [x] Port of `active_delta_center5` (DIP/finite-Zak walk) into the same lib,
      validated to ~1e-13 vs the Python reference (`validate_dip.py`).
- [x] Streaming reconstruction remains available in the backend, but its old
      viewer checkbox was removed: it displayed an STFT of a reconstructed
      record and was not the rolling super-resolution observable its label
      suggested.
- [x] Complex-generalized reconstruction (`dip_run_complex`): full-spectrum PGHI,
      no Hermitian fold. Needed because per-quadrature reconstruction leaks a
      per-frame mirror image; the complex solve is mirror-free (and cheaper: one
      solve, not two).
- [x] Fixed block-boundary striping (overlapping segments + phase-align + Hann
      OLA) and made reconstruction work at zoomed-out spans (raw-fill beyond
      coverage).
- [x] Replaced the live magnitude-claim image with the original Python
      waveform construction (2026-07-10): shared one-step DIP seed followed by
      one independent two-family projection. The old claim result remains a
      study baseline because it is smooth, but its horizontal row modulation
      is not the Python super-resolution operator.
- [x] Reassignment engine ported to C/bfft (`iqw_ra_*`, `iqwaterfall.Reassign`):
      cosine similarity 0.99999998 against the real-valued Python oracle on the
      Dave-and-Simon fixture.  Kept as a separate observable mode.
- [x] Intrinsic FCT endpoint viewer ARCHIVED to `archive/fct_view.py`. The live
      third mode instead uses exact FCT phase jets to guide ordinary
      reassignment. `fct_complex_moment` exposes the selected-support phase
      moment; an endpoint recurrence supplies frequency. The 2N hybrid remains
      slow and fills asynchronously.
- [x] RF64 test: `08-18-24_15100000Hz.wav` opens as stereo int16 IQ; its header
      reports 456,000 Hz (not 390,000 Hz).
- [x] Display pipeline overhaul (2026-07-10): the reference-figure look is a
      transfer function, not a transform. New default "Amplitude (auto)" maps
      linear amplitude against the image's 99.5th percentile (background to
      zero, structure across the palette) — this is exactly what the STFT
      reference figures do; "dB (auto range)" and the manual dB window remain.
      Display resampling is now peak-preserving: max-pool when shrinking (a
      one-bin tone / one-row click can no longer fall between output samples),
      linear-power interpolation when zooming (>= -3 dB worst case; dB-domain
      lerp lost tens of dB). Gamma slider added.
- [x] Removed the live same-parent support restriction entirely. Independent
      frame families need no `M_delta` containment: the selected geometry is
      simply short `N/4`, with its own COLA hop `N/8`.
- [x] File-open UX: concrete extension filter is now the default (".*" was
      unreliable for click-selection), `selections` fallback in the callback,
      and AUTO header-parse failure retries as raw int16 at the UI rate.
- [ ] Solver FFT speed: `fft_pow2` (hand-written) is the per-tile hot path; could
      move to bfft (not Accelerate directly). Only matters for reconstruction throughput.
