# BFFT IQ Waterfall Viewer

A streaming, zoomable IQ waterfall viewer. The heavy lifting (file I/O, sample
conversion, windowed complex FFTs) lives in a monolithic C++ library,
`libiqwaterfall`, built on the project's BFFT real-FFT kernel. The UI is a thin
DearPyGui front-end.

## Components

| File | Role |
|------|------|
| `iqwaterfall.cpp` | Monolithic backend: WAV/raw IQ reader (`convert_to_float` port for u8/s8/s16/s24/s32/f32) + BFFT-backed waterfall engine. Flat C ABI. |
| `dip_algo.cpp` | C++ port of `active_delta_center5_fast1` (DIP/finite-Zak walk + in-house PGHI seed + record OLA), real **and** complex-generalized solvers. Same library. |
| `iqwaterfall.py`  | `ctypes` wrapper: `IQSource`, `Waterfall`, `dip_run`, `dip_run_complex`. |
| `dip_stream.py` | Streaming reconstruction: background thread pool, overlapping segments + phase-aligned Hann OLA (kills block striping), prefetch + LRU cache. |
| `superres.py` | Super-resolution readout: complex reassigned spectrogram (the aperture-ladder F1×T2 product). |
| `validate_dip.py` | Bisected numerical check of the port vs the Python reference. |
| `iq_waterfall_app.py` | DearPyGui viewer (transport, zoom, dynamic range, colormaps, settings). |
| `build.sh` | Compiles `libiqwaterfall.{dylib,so}`, linking `../build/libbfft.a`. |

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

- **Auto (WAV header)**: parses RIFF/`fmt `/`data` like `WaveFile.cs`; 2-channel
  ⇒ complex IQ, 1-channel ⇒ real.
- **Raw**: pick the sample format + sample rate + complex/real in the UI. Used
  for headerless SDR captures (`.iq`, `.raw`, `.cs16`, ...).

## Status / roadmap

- [x] Monolithic backend, IQ reader, BFFT waterfall — built & verified.
- [x] DearPyGui viewer: open, play/pause/stop, scrub, time & frequency zoom,
      dynamic range, colormap/window/FFT-size settings.
- [x] Port of `active_delta_center5` (DIP/PGHI reconstruction) into the same lib
      — validated to ~1e-13 vs the Python reference (`validate_dip.py`).
      Use it via `iqwaterfall.dip_run(x)`.
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
- [x] Super-resolution readout ("Super-res readout" toggle): complex reassigned
      spectrogram of the (reconstructed) signal — sharp in both time and
      frequency. Composes with Reconstruct. A plain re-STFT does NOT super-
      resolve and blind geomean fusion degrades (both measured); reassignment of
      the phase-consistent record is the method (per `notes/aperture_ladder_design.md`).
- [x] **Super-resolution = two-window MIN-consensus fusion** (the method): a short
      window (fine time, default 512) and a long window (fine frequency, default
      2048), fused by per-cell MIN in dB. Energy survives only where both windows
      agree, so each window's smearing is suppressed — sharp in BOTH axes. The
      fused image inherits 2048's frequency contrast AND 512's time contrast.
      "Super-res (two-window fusion)" toggle + time/freq window controls; composes
      with Reconstruct. (MIN, not geomean: geomean blends/blurs, MIN intersects.)
- [x] Reassignment engine ported to C/bfft (`iqw_ra_*`, `iqwaterfall.Reassign`):
      matches numpy 0.9999, ~2.4× faster. Available; not the super-res default
      (single-window reassignment is not super-resolution).
- [x] Solver options: up to 3 L5 steps (`n_steps`), short+long seed fusion
      (`fuse_seed`), and warm-start (`dip_run_complex_warm`). All measured inert on
      the 390 kHz IQ path (steps: consensus is a fixed-point; fusion: short seed
      worse; warm-start: complex recon already tracks the signal + PGHI flood ≪
      FFT cost). Retained as options, default off.
- [ ] Solver FFT speed: `fft_pow2` (hand-written) is the per-tile hot path; could
      move to bfft (not Accelerate directly). Only matters for reconstruction throughput.
- [ ] Smooth out the file-open dialog UX (currently needs double-click/retry).
