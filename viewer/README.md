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
| `dip_stream.py` | Asynchronous/cached FCT, active-delta fusion, and reconstruction streams. |
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

- **Auto (WAV header)**: parses RIFF, RF64, and BW64 (`ds64`, `fmt `, `data`);
  2-channel ⇒ complex IQ, 1-channel ⇒ real. The header sample rate is used.
- **Raw**: pick the sample format + sample rate + complex/real in the UI. Used
  for headerless SDR captures (`.iq`, `.raw`, `.cs16`, ...).

## Analysis modes

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
- [x] Active-delta waterfall: measured complex long-aperture endpoints seed the
      paused state directly (no unnecessary PGHI phase invention). Short endpoint
      power partitions each long-bin power over owned fine-time rows, conserving
      power instead of independently normalizing or taking image-space MIN.
- [x] FCT support-search baseline (experimental): the complex correlation selects one shared support
      for I/Q, negative-frequency bins are native, and cells are placed at
      `frame_origin + tau` rather than at an STFT center. Magnitude displays
      `|C| sqrt(N/tau)` (noise/energy normalization); warm color indicates short
      selected support. The wideband false-alarm gate is `log(N)+4` times mean
      power because a prefix search's null maximum grows with `log(N)`. This is
      Claude's dyadic-proxy + explicit endpoint-search algorithm, not the hoped-
      for intrinsic DIP support type.
- [x] Reassignment engine ported to C/bfft (`iqw_ra_*`, `iqwaterfall.Reassign`):
      matches numpy 0.9999, ~2.4× faster. Available; not the super-res default
      (single-window reassignment is not super-resolution).
- [x] FFT sizes through 65,536 are selectable. FCT remains a research mode:
      its checked-in pyramid is O(N log²N) plus signal-adaptive refinement, so
      high-N active scenes may fill asynchronously rather than at playback rate.
- [x] RF64 test: `08-18-24_15100000Hz.wav` opens as stereo int16 IQ; its header
      reports 456,000 Hz (not 390,000 Hz).
- [ ] Solver FFT speed: `fft_pow2` (hand-written) is the per-tile hot path; could
      move to bfft (not Accelerate directly). Only matters for reconstruction throughput.
- [ ] Smooth out the file-open dialog UX (currently needs double-click/retry).
