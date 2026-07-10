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
| `iqwaterfall.py`  | `ctypes` wrapper: `IQSource`, `Waterfall`, `FctWaterfall`, and DIP APIs. |
| `dip_stream.py` | Asynchronous/cached fusion and FCT-guided reassignment streams; reconstruction remains a library facility. |
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

The mode selector intentionally exposes three different measurements:

1. **Streaming STFT** — a conventional symmetric-Hann complex spectrogram,
   center-timed and cheap enough for continuous playback.
2. **Super-resolution (two-aperture)** — the DIP-Zak F1xT2 fusion.  The user
   sets both analysis windows: the long aperture supplies fine frequency at
   true scale, while state-attached short endpoints distribute it across
   the fine-time rows, and both are read off one refined active-delta state
   (long readout exact to ~1e-14 against the measured Hann spectra; shorts
   attach in-state through `M_delta` to ~1e-13). A short row that crosses its
   display owner's boundary is routed through the neighboring overlapping long
   state containing the same global interval. Thus NB=1024/NS=512 is valid;
   no parallel STFT bank is required.
3. **FCT-guided reassignment** — ordinary Hann reassignment establishes the
   local phase basin. The intrinsic FCT then supplies exact phase-derivative
   coordinates at its selected support: `Re(M/C)` in time and a fixed-support
   endpoint recurrence in frequency. FCT replaces a coordinate only when it
   agrees with the local STFT branch. Energy is accumulated on a 2N half-bin
   raster and the slow transform fills asynchronously.

The intrinsic-FCT viewer mode is archived in `archive/fct_view.py`; the exact
FCT transform itself remains a shipped, tested library feature.

- [x] Monolithic backend, IQ reader, BFFT waterfall — built & verified.
- [x] DearPyGui viewer: open, play/pause/stop, scrub, time & frequency zoom,
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
- [x] Two-aperture DIP-Zak fusion restored as the super-resolution mode with
      manual long/short window controls (2026-07-10).  Fixed the direct seed:
      the former `tap3_adj` pullback made the readout round-trip apply Hann²
      (adjoint ≠ inverse), erasing tone pairs the plain Hann STFT resolves;
      the seed is now the paused state of the record itself, exact to ~1e-14.
      Added all-delta claim partitioning, which removes the ±NB/2
      transient halo.  On the known-truth fixture the claim fusion at
      NB=2048/NS=128 reaches single-row click localization (0.07 ms) AND the
      full long-window tone separation (−38 dB) simultaneously; reassignment
      gives 0.07 ms / −21 dB.  ~55 ms per 160 rows through the tile pool.
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
- [x] Cross-frame aperture routing: the former same-parent restriction made
      NB=1024/NS=512 request invalid local delta 17. That global short interval
      is exactly delta 13 of the next overlapping long state. The router now
      performs this identity generally; only the true DIP attachment condition
      `NS <= NB-128` remains.
- [x] File-open UX: concrete extension filter is now the default (".*" was
      unreliable for click-selection), `selections` fallback in the callback,
      and AUTO header-parse failure retries as raw int16 at the UI rate.
- [ ] Solver FFT speed: `fft_pow2` (hand-written) is the per-tile hot path; could
      move to bfft (not Accelerate directly). Only matters for reconstruction throughput.
