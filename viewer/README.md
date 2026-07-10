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
| `dip_stream.py` | Asynchronous/cached fusion super-resolution and reconstruction streams. |
| `superres.py` | Python reference for the phase-aware reassigned spectrogram. |
| `make_sr_fixture.py` | Synthesizes the known-truth SR fixture (tone pair + click train + chirp). |
| `sr_fusion_study.py` | Quantified fusion quality study (click width, halo, tone dip, gate noise). |
| `validate_modes.py` | Headless, timed comparison of streaming STFT, two-aperture fusion, and reassignment. |
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
   true scale, the short aperture's measured endpoints gate that power across
   the fine-time rows, and both are read off one refined active-delta state
   (long readout exact to ~1e-14 against the measured Hann spectra; shorts
   attach in-state through `M_delta` to ~1e-13).  Default gating is
   claim-normalized: each frame's gates are normalized over its entire delta
   range with long-window weights, so energy outside a frame's owned band goes
   unclaimed instead of being smeared into it — this is what removes the
   ±NB/2 transient halo (measured ~26 dB → ~0 dB on the SR fixture).
3. **Reassigned STFT** — phase derivatives move long-window energy to its
   measured time/frequency centroid.  Kept as a separate observable; sparse
   and speckle-like by construction.

The intrinsic-FCT viewer mode is archived in `archive/fct_view.py`; the exact
FCT transform itself remains a shipped, tested library feature.

- [x] Monolithic backend, IQ reader, BFFT waterfall — built & verified.
- [x] DearPyGui viewer: open, play/pause/stop, scrub, time & frequency zoom,
      dynamic range, colormap/window/FFT-size settings.
- [x] Port of `active_delta_center5` (DIP/finite-Zak walk) into the same lib,
      validated to ~1e-13 vs the Python reference (`validate_dip.py`).
- [x] Streaming reconstruction wired into the viewer ("Reconstruct" toggle):
      background thread pool tiles the signal through the DIP solver, prefetch +
      LRU cache, async fill-in, rendered via `render_mem`.
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
      Added claim-normalized gating (default), which removes the ±NB/2
      transient halo.  On the known-truth fixture the claim-gated fusion at
      NB=2048/NS=128 reaches single-row click localization (0.07 ms) AND the
      full long-window tone separation (−38 dB) simultaneously; reassignment
      gives 0.07 ms / −21 dB.  ~55 ms per 160 rows through the tile pool.
- [x] Reassignment engine ported to C/bfft (`iqw_ra_*`, `iqwaterfall.Reassign`):
      cosine similarity 0.99999998 against the real-valued Python oracle on the
      Dave-and-Simon fixture.  Kept as a separate observable mode.
- [x] Intrinsic FCT viewer mode ARCHIVED (2026-07-10) to `archive/fct_view.py`
      (endpoint-timed display set aside; ~13 s/160 rows was accepted but the
      product direction is the fusion super-resolution).  The exact FCT
      transform, its C ABI, wrapper, and test suite are unchanged.
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
- [x] Aperture validity fixed: owned-band fusion needs `ns <= 7*nb/16 + 32`
      (stricter than the attach guard ES<=EB-4); nb=1024/ns=512 used to crash
      with a delta index out of range. UI snaps, FusionStream raises clearly.
- [x] File-open UX: concrete extension filter is now the default (".*" was
      unreliable for click-selection), `selections` fallback in the callback,
      and AUTO header-parse failure retries as raw int16 at the UI rate.
- [ ] Solver FFT speed: `fft_pow2` (hand-written) is the per-tile hot path; could
      move to bfft (not Accelerate directly). Only matters for reconstruction throughput.
