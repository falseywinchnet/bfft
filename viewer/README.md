# BFFT viewer lab

The repository contains several viewer families because it is both a working
vision tool and a research notebook. These are the supported starting points:

| Goal | Run | Status |
| --- | --- | --- |
| Segment an image into BFFT-guided transport cells | `python viewer/segmenting_veroni_viewer.py` | **Canonical image-segmentation viewer** |
| Play with the two-scale cartoon/texture hierarchy | `python viewer/segmenting_v3_app.py` | Version 3.0 experiment |
| Explore emergent object IDs on the finished cell graph | `python viewer/object_transport_segmentation_viewer.py` | Experimental object-support hierarchy |
| Explore owner-free residual-consuming cells | `python viewer/resource_transport_cells_viewer.py` | Isolated transport-cell experiment |
| Explore cartoon/texture decomposition on still images | `python viewer/meyer_stills.py` | Canonical decomposition explorer |
| Explore 2-D intrinsic time-scale decomposition | `python viewer/voronoi_itd_viewer.py` | Voronoi-supported ITD prototype |
| Process or inspect decomposition on video | `python viewer/meyer_video.py` | Offline video path |
| Apply the real-time webcam effect in OBS | See [`obs-plugin/README.md`](../obs-plugin/README.md) | Native real-time filter |
| Inspect IQ waterfall / super-resolution work | `python viewer/iq_waterfall_app.py` | Separate signal viewer; build first below |

The files named `recursive_*`, `seeded_*`, `error_spent_*`,
`two_population_*`, and `claude_trial_*` are retained research controls. They
are useful for reproducing discarded or intermediate allocation ideas, but
they are not the place to begin another segmentation implementation.

## Voronoi intrinsic time-scale decomposition

```sh
.venv/bin/python viewer/voronoi_itd_viewer.py
```

This image operator replaces separable row/column ITD support with the frozen
Meyer-density geometry already used by the canonical segmentation pipeline.
Two C++ decomposition stages measure a bounded inverse-support tensor. Its
reciprocal ellipse area, `sqrt(det(Q))/pi`, emits the complete germ population
simultaneously; an analytic director-curvature correction shortens only the
supports that cannot remain locally straight. One reduced-basis anisotropic
fast march forms all cells. There is no extrema spacing, candidate search,
birth loop, Lloyd motion, or support diffusion. Actual cell interfaces supply
the intrinsic knot graph, and a convex cell-amplitude readout gives the
baseline.

The panels show the selected baseline and rotation alongside literal hard
Voronoi ownership, extrema polarity, and a live all-level recomposition.
Every extraction is telescoping: the rotations plus the final residual
reconstruct the analyzed lightness plane to floating-point precision.
`voronoi_itd.py` is the standalone operator; `voronoi_itd_app.py` is only its
interactive shell. The production derivation and timing checkpoint are in
[`MEYER_DENSITY_VORONOI.md`](MEYER_DENSITY_VORONOI.md). The rejected
iterative ablation remains recorded in
[`EIKONAL_VORONOI_ITD.md`](EIKONAL_VORONOI_ITD.md).

## Image segmentation quick start

From the repository root:

```sh
python -m pip install -e '.[vision-viewer]'
python viewer/segmenting_veroni_viewer.py
```

Choose a bundled image or open a file, select **full-resolution output** when
desired, and press **Build representation**. The canonical path measures one
finished Meyer/ROF geometry with the optimized one-axis solver. Allocation may
run on a smaller sample of that frozen geometry, then one exact transport
refresh classifies every original-resolution pixel. Large-image output
therefore remains full resolution without making population inference scale
with the source pixel count.

The default allocator reads the complete population directly from the frozen
BFFT support density, `sqrt(det(Q)) / pi`. Its curvature correction shortens a
predicted long tangent support when the measured director turns far enough to
leave that support's normal width; this is what gives a smooth closed contour
more cells without treating every straight edge as texture. A deterministic
local phase turns the resulting continuous density into germs in parallel.
There is no initial
barycenter, candidate list, top-k, requested population, offspring, deletion,
or all-pairs cell operation. The safety ceiling only catches a misspecified
support measure.

Weak detail receives an additional local-null test before it commands
population. Gradients that persist across scale count as positive structural
evidence; finest-scale disagreement is the local null. Attenuation acts only
where total evidence is already weak, allowing smooth sky and similar panels
to command fewer cells without mistaking strong grass or other isotropic
texture for noise.

Those germs create one continuous, same-label first-arrival partition. Each
pixel's support is then carried backward through the achieving causal front,
giving every germ an intrinsic transport force without a runner-up field,
centroid, PCA, or direction bins. The position step is clipped to at most half
the germ's distance from its current hard interface. An exact remarch accepts
the step only when every germ remains alive and the measured transport action
decreases. One accepted pass is the default; the older residual-pressure and
hierarchical split paths remain available as explicit research controls.

At a decisive unchanged-target discontinuity, transport also receives a
finite rank-one jump action across the measured normal. This prevents a
white-side support from cheaply acquiring a few black-side pixels while
leaving travel along the contour available. Once fronts collide, their local
accepted actions determine the subpixel crossing position. Fractional
interface coverage rasterizes that position before the optional soft cover;
it does not propagate or rank a second owner. Both readout refinements remain
guarded by the full RGB/cartoon/texture objective.

The right-panel diagnostics expose the representation itself:

- **Site IDs + boundaries** shows every literal hard transport domain in a
  stable ID colour, matching the diagnostic needed to judge SAD-like panels
  and slivers.
- **Soft Site IDs** evolves those hard indicators through the BFFT-gated
  anisotropic heat cover. It is the literal co-owned partition-of-unity view;
  **Soft Site IDs + hard boundaries** shows which geometric boundaries have
  become functionally invisible without deleting their sites.
- **Hard reconstruction** A/Bs the pre-cover fit against **Reconstruction**.
  Soft support is adopted only when RGB plus the single-stage cartoon and
  texture objective improves.
- **Null evidence confidence**, **Boundary jump confidence**, and
  **Interface coverage** expose broad-panel confidence, crossing discipline,
  and subpixel contour readout separately.
- **Reconstruction + cell boundaries** exposes seams independently of PSNR,
  while **Reconstruction + sites** shows where the transported centers landed.
- **Transport support measure**, **Metric anisotropy**, **Cartoon**,
  **Texture**, and **Transport glass** expose the one frozen geometry that
  controls density, direction, and shape. **Curvature population factor**
  shows exactly where a locally straight anisotropic support expires, while
  **Soft support conductance** shows which boundaries may share territory.
- **Residual energy**, **Reverse residual flow**, and **Refinement demand**
  expose the target-to-reconstruction failure, its predecessor-tree return,
  and the combined local bifurcation signal.
- **Characteristic force**, **Topology clearance**, **Trust-limited step**,
  and **Site motion** expose the causal position relaxation and its literal
  safety bound.

For a headless HD phase trace:

```sh
python viewer/profile_segmenting_veroni.py /path/to/image.png \
  --full-resolution
```

The model and performance rationale are documented in
[`TRANSPORT_CELL_MATH.md`](TRANSPORT_CELL_MATH.md),
[`../notes/FLOW_VOLUME_NUCLEATION.md`](../notes/FLOW_VOLUME_NUCLEATION.md),
[`../notes/CURVATURE_AND_SOFT_SUPPORT.md`](../notes/CURVATURE_AND_SOFT_SUPPORT.md),
[`../notes/NATIVE_SEGMENTING_VIEWER_ROUND.md`](../notes/NATIVE_SEGMENTING_VIEWER_ROUND.md),
[`../notes/NULL_JUMP_INTERFACE_REFINEMENT.md`](../notes/NULL_JUMP_INTERFACE_REFINEMENT.md),
[`../notes/HD_SEGMENTATION_PERFORMANCE.md`](../notes/HD_SEGMENTATION_PERFORMANCE.md).
`segmenting_veroni_app.py` is the current implementation module.
`transport_measure_app.py` retains the canopy/static-overlap research
controls. The `segmenting_veroni_viewer.py` filename is the stable user-facing
entry point. Algorithms awaiting native ports are separated under
[`../port_needed`](../port_needed/README.md).

## Segmenting version 3.0 experiment

```sh
python viewer/segmenting_v3_app.py
```

This viewer runs one lower-resolution cartoon transport, preserves its owner
IDs while upgrading their interfaces directly on the full-resolution target.
It then emits full-resolution texture microcells, assigns each to one cartoon
parent, and transports only among siblings. Its controls retain the former
parent-ridge model for A/B comparison. The panels expose cartoon parents,
texture microcells, texture target and fit, coordinate fields, and the final
residual. See
[`SEGMENTING_V3.md`](SEGMENTING_V3.md) for the measured results.

## Emergent object-support experiment

```sh
python viewer/object_transport_segmentation_viewer.py
```

This separate viewer runs the canonical cell pipeline once, splits each
reconstruction site into connected unchanged-target support fragments,
constructs their literal sparse graph, and tests an evolving
atom-to-part-to-object hierarchy on that graph. The right panel exposes every
cue, hard IDs, boundary-local soft uncertainty, connected-material quotient
failures, core altitude, and best-versus-second saddle confidence. It also
preserves and displays separately connected interface arcs and junctions—the
planar support needed for closure, common-surround, T-junction, and amodal
continuation experiments.

The first eight right-panel views are a pre-hierarchy forensic microscope.
Clicking any rendered pixel anchors its connected canonical support fragment,
then independently displays colour, transport action, metric tensor,
action-plus-metric, complete-state likeness, or a phase-preserving graph
scattering response across every fragment. The local scattering view retains
only the first literal transport neighborhood; the multiscale view retains
all five dyadic bands. These maps consume no object IDs and are not merge
decisions.

The **centered edge relation** views go one step further without forming
objects. They compare the observed joint distribution of scattering patterns
on literal transport interfaces with the independent product distribution.
Positive values mean the two patterns co-occur more than prevalence predicts;
negative values mean they avoid one another. The complete signed spectrum is
retained—there is no candidate list or selected eigenmode.

Use **Connected support fragment IDs** and **Intra-site topology cuts** to
inspect where a reconstruction basis function had crossed or disconnected
scene topology. **Unconstrained path disagreement** A/Bs the old free
two-path readout against the connected first-arrival watershed.

Click the right image to inspect its pixel, cell, hard object, runner-up,
diagnostic material basin, and weakest neighboring interfaces. **Anchored
witnesses only** is a research control that prevents allocation/support
changes from creating a boundary without direct target/cartoon/glass evidence.
The additive and anchored barriers remain visible side by side.

The **Part → parent topology** section retains every visible hard part.
Containment and first-arrival completion through a common surround are the
active parent mechanisms. Embedded T-junctions are retained as directed depth
observations: the continuing region is in front of the two regions whose
contour terminates there. They do not merge the two rear regions. The former
T-junction attraction rule is available only as an explicitly experimental
control because it falsely joined the astronaut's face to the background.

Use **T-junction depth order** and **T-implied contour sides** to inspect the
sparse occlusion hypothesis. The latter paints its proposed front side cyan
and back side red. This is not treated as proof: a material seam terminating
at a silhouette has the same local T geometry. **Enclosed seam attachment
proposals** exposes the competing hypothesis when the terminating pair forms
a bounded union predominantly enclosed by the third region. **T-implied
frontness** aggregates the first hypothesis per part; **Transport-depth
extrapolation** shows the scene-specific support direction fitted from it.
Neither research readout affects object merging. **Surround completion
proposals** shows the separate sparse wavefront collision used for amodal
completion.

Use **Recompute objects only** while tuning the object controls. It reuses the
finished cells and graph rather than decomposing or fitting the image again.
The formal model, current timings, known matched-texture failure, and the
unsigned figure/ground limitation are recorded in
[`../notes/TRANSPORT_OBJECT_SUPPORT.md`](../notes/TRANSPORT_OBJECT_SUPPORT.md).

## Owner-free resource-cell experiment

```sh
.venv/bin/python viewer/resource_transport_cells_viewer.py
```

This separate viewer runs the continuous diffuse/crystalline support model
from `experiments/resource_transport_cells.py`. It has no owners, ranked
candidate allocation, fixed birth batch, deletion, or population-scaled cell
budget. The validated settings are the defaults; rejected controls remain
available for direct A/B testing. Use **Measure C/T objective** when desired,
since the full single-stage decomposition score is intentionally computed on
demand.

## IQ waterfall viewer

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
   The finish selector A/Bs the deployed 75%-relaxed terminal long projection
   against a genuinely palindromic half-long/full-short/half-long cycle.  The
   former empirically preserves weak chirp terminals; the latter removes the
   fitted coefficient and projection-order bias.
   `Normalize rung gain` removes each aperture's exact symmetric-Hann coherent
   power gain before max readout fusion, preventing longer windows from winning
   merely because their FFT amplitude scales with window length. `Seed only`
   displays the shared DIP/PGHI latent directly and skips all unified family
   projections, providing an A/B for what the final fill actually contributes.
3. **Reassigned STFT** — ordinary symmetric-Hann phase reassignment, retained
   separately so the effect of the two-aperture solve remains inspectable.

The intrinsic-FCT viewer mode is archived in `archive/fct_view.py`; the exact
FCT transform itself remains a shipped, tested library feature.

### Minimal SDR integration: raw IQ through the SR readout

This stage has no DIP solver, PGHI, tile cache, warm state, or magnitude-family
projection.  For each display aperture `N`:

1. Create `iqw_ra_create(N)` and optionally enable the conservative bilinear
   splat with `iqw_ra_set_bilinear(engine, 1)`.
2. Supply interleaved complex `float32` IQ and call
   `iqw_ra_render_mem(engine, iq, nsamples, first_center, hop, rows, out_db)`.
3. The engine computes Hann-windowed `Y`, time-weighted `Y_t`, and
   derivative-windowed `Y_d` spectra.  Every non-negligible coefficient moves
   its power `|Y|^2` to

       time = row + Re(Y_t conj(Y)) / (|Y|^2 hop)
       bin  = k - Im(Y_d conj(Y)) N / (2 pi |Y|^2)

   and bilinearly splats it to the surrounding time/frequency cells.
4. For a ladder, repeat with each desired `N`. Before per-cell maximum fusion,
   optionally subtract the exact symmetric-Hann coherent-gain offset
   `20 log10((N-1)/(N_base-1))` dB from each rung.

The output is `rows x N` fft-shifted power in dB.  A single-aperture integration
only needs the `iqw_ra_*` calls; the ladder normalization/resampling/max is a
display policy and can be omitted.  This is reassigned analysis of raw IQ, not
waveform super-resolution—the seed and unified stages are what alter the
latent waveform.

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
The mouse wheel scrubs by one transform hop while hovering the position slider.
Over the waterfall it performs cursor-anchored frequency zoom: the frequency
under the pointer remains fixed, so off-center zooms naturally pan as they
expand or contract.
Streaming and reassigned STFT modes expose `H=N/8`, `N/4`, and `N/2` display
hops.  This changes temporal sampling without moving the position marker and
allows a raw STFT to use the same external hop as an SR comparison.  The SR
The SR stage selector separates raw IQ through the SR reassignment readout,
the shared DIP/PGHI seed through that same readout, and the complete unified
magnitude-family projection.  Raw stage makes no tile requests and therefore
has no asynchronous readiness sweep.

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
