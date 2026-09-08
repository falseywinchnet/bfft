# BFFT Vision for OBS

This module registers BFFT Cartoon for both OBS frame/async sources and
GPU-rendered synchronous sources, plus the native BFFT High Vision temporal
filter. OBS shows only the Cartoon registration compatible with the selected
source, so both menu entries have the same **BFFT Cartoon** name:

- **BFFT Cartoon**, with five display modes:
  - **Cartoon + texture** — the adjustable Meyer layer recomposition.
  - **Fine chrome (optional TVS)** — the earlier chrome-to-TV-projection look,
    driven by the signed defect `cartoon - ROF(cartoon, shading_c)`.
  - **Recursive recovery** — repeats the decomposition on `input - texture`
    and boosts the newly recovered texture. This reproduces the useful
    two-filter recovery behavior without an intervening 8-bit OBS round trip.
  - **Layer interference** — a signed, neutral-gray view of the normalized
    cross term between explicit texture and the cartoon-side TV defect.
  - **Information caustics** — a color-preserving refractive field whose
    displacement comes from non-cartoon geometry and whose carrier comes from
    the local phase between texture and that TV defect.

- **BFFT High Vision**, the continual temporal-imaging framework. Its
  **Synthetic HDR** and **Night integrator** modes transport scene-linear
  radiance, precision, and variance through estimated camera motion while
  releasing support in changing regions.

The Cartoon filter keeps the source color and runs its decomposition on luma.
Its two selectable engines run on the same aspect-preserving spectral work
lattice, so their visual difference is an operator difference rather than a
resolution mismatch:

- **Finite-flow fast (4, 18, 2, 3)** runs four exact fused prefix passes,
  three depth-two semismooth finite-horizon jumps, and two exact refresh passes
  after each jump. Its fixed operator-map cost is 19.
- **Fused passes** runs the ordinary fused Meyer recurrence. The pass slider
  defaults to 16 and may be changed from 1 to 64.

Both expose the exact two-product contract `(cartoon, texture) = (f-v, v)`.
For the fused engine, the texture-side ROF survivor is folded into effective
cartoon at readout. This makes the mixer and A/B comparison identical without
claiming that the underlying fused state lacks that survivor.

The finite-flow chart requires complete power-of-two spectra. **Preview** uses
a 256-class canvas, **Detailed** uses 512 (the default), **High detail** uses
1024, and **Extreme detail** uses 2048. Wide video uses a half-height
power-of-two canvas; the source aspect is fitted inside it and the small unused
margin is filled by symmetric reflection. Thus 16:9 video maps respectively to
`228x128`, `455x256`, `910x512`, or `1820x1024` content inside `256x128`,
`512x256`, `1024x512`, or `2048x1024` spectral canvases. Bilinear maps are
cached when source or lattice dimensions change. The two larger choices are
intended for quality comparisons: transform cost and plan memory grow sharply.
High Vision instead processes the camera's native pixel lattice. Photon
evidence, detector-fixed noise, and motion are not inferred from a resized
surrogate. Night emits monochrome luma; Synthetic HDR retains source chroma.

Both engine readouts are exactly complementary. The main effect mixer is:

```
output = cartoon_gain * cartoon
       + texture_gain * texture
       + shading_gain * (cartoon - ROF(cartoon, shading_c))
```

Cartoon and texture gains at 1 with shading at 0 are an exact bypass. Useful
starting looks from the still viewer are:

- cartoon only: `cartoon=1, texture=0, shading=0`
- texture enhancement: `cartoon=1, texture=3, shading=0`
- flattened/detail-forward: `cartoon=0.7, texture=2, shading=0`
- illumination enhancement: `cartoon=1, texture=1, shading=2`
- smooth cartoon with illumination: `cartoon=1, texture=0, shading=2`

At exactly `cartoon=0`, the remaining texture is signed detail rather than a
color image. The plugin therefore displays that field around
mid-gray and neutralizes chroma. This preserves negative detail and prevents
low luma with retained NV12 chroma from producing false red regions.

Fine chrome is optional and does not alter the selected decomposition engine.
It performs one adjustable TV projection of the effective cartoon and renders
the signed difference as displaced chrome. **TV effect sweeps** controls only
this projection and the other ROF-derived display effects; it does not change
finite-flow or fused pass count. Relief depth and chrome gloss are adjustable.
Its environment carrier uses BFFT's low-error table-polynomial sine rather
than a scalar library `sin` at every pixel. Information caustics also uses
BFFT's slope-tree phase estimator for its TV-defect/texture angle.

Recursive recovery uses **Recovery boost**. Layer interference and Information
caustics use **Information gain**; caustics additionally uses **Information
phase folds** and **Chrome relief depth**. Layer interference is deliberately
monochrome because its positive and negative values are data, not scene color.
Caustics gates its carrier by local non-cartoon magnitude, so smooth areas do
not acquire an arbitrary color or phase.

## Build on macOS

The OBS headers are not shipped inside OBS.app, so use a matching source
checkout. For the currently installed OBS 32.2.1:

```sh
git clone --depth 1 --branch 32.2.1 \
  https://github.com/obsproject/obs-studio.git /tmp/obs-studio-32.2.1

cmake -S obs-plugin -B build-obs \
  -DOBS_SOURCE_DIR=/tmp/obs-studio-32.2.1 \
  -DOBS_APP=/Applications/OBS.app \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-obs --parallel
```

Install for the current user:

```sh
cp -R build-obs/bfft-cartoon.plugin \
  "$HOME/Library/Application Support/obs-studio/plugins/"
```

Restart OBS, open a source's **Filters**, and add **BFFT Cartoon** or **BFFT
High Vision**. For Cartoon, start with **Finite-flow fast**, **Detailed
(512 class)**, cartoon gain 1, texture gain 0, 6 CPU threads, and 8 TV effect
sweeps. Those are the new-filter defaults, so the effect is visible as soon as
it is added. Switch only
**Decomposition engine** to **Fused passes** for the default fused-16 A/B;
the work lattice and display settings remain unchanged. Recursive recovery
performs a second selected decomposition and is correspondingly heavier. For
High Vision, start
with Synthetic HDR; Night integrator retains evidence longer for fixed-exposure
low-light capture.

Night mode does not use HDR's shadow rejection policy: every unclipped frame
contributes one observation unit and low census confidence no longer destroys
transported support. Long-lived Night radiance follows the global camera gauge
rather than being recursively resampled through noisy per-tile flow; this
prevents high evidence support from turning a static scene into an elastic,
"wibbly" image. A signed temporal innovation fusor releases support when
contradictory evidence persists. The comparison is signal-relative, so a dark
object entering a dim region is not hidden behind an absolute linear threshold.
When the camera does not expose gain/shutter metadata, Night changes its
exposure gauge only for a spatially coherent registered brightness step;
low-light frame ratios cannot accumulate into an unbounded AGC random walk.
The **Evidence persistence** control is honored directly; Night no longer
silently raises it to `0.997`. Mode changes reset temporal state. High Vision
uses the exact source dimensions (`1920x1080 → 1920x1080`); OBS scaling belongs
after the filter.

Night registration now uses BFFT's existing high-speed TGFD/Meyer cartoon
carrier for both the incoming frame and the accumulated belief. The oscillatory
side is not inserted into the output; it only keeps sensor-fixed salt from
nucleating a false zero-motion registration. A separate zero-mean nuisance
field remains in detector coordinates and is learned only when accepted camera
motion makes it identifiable. The physical uncertainty law adds read and shot
variances rather than their standard deviations.

Night deliberately does not infer color from photon-starved ISP chroma. YUV
output is neutralized and RGB output receives equal channels, yielding a
monochrome luminance estimate. Synthetic HDR retains the source chroma path
unchanged.

**Night likelihood (experimental)** shares Night's native-resolution
registration, transport, detector-pattern gauge, and monochrome output, but
replaces its signed innovation memory with a sequential generalized
Poisson-Gaussian likelihood-ratio bank. It is a secondary dogfood path, not a
replacement for established Night. In the deterministic 128x96 shadow rig it
raises noise reduction from 17.70 dB to 19.21 dB and frame-2 dark-object
recovery from 73.9% to 86.7%; frame-1 recovery is slower (32.8% versus 40.3%).

**Night moments (420v experimental)** is the fourth High Vision path. It is
for fast camera/adapter formats that expose a noisy frame population instead
of an internally averaged preview. It transports the temporal mean and
population variance in the same global camera gauge, freezes fallback exposure
estimation unless real telemetry is present, and promotes empirical standard
deviation into display radiance. **Moment response power** controls the
mode-dependent mean transfer, **Variance-as-signal gain** controls the second
moment, **Temporal noise floor** removes a known variance floor, and **Moment
bootstrap support** determines how much population must accumulate before
variance participates in change decisions. **Moment integration window** sets
a one-to-sixty-second temporal horizon. It follows actual processed-frame
timestamps rather than nominal camera FPS, while coherent motion/change can
still release stale support locally. The mode remains luma-only.

The entropy-budgeted photon-registration controller is present in the linked
High Vision core but is not applied to ordinary OBS luma. OBS receives
ISP-processed 8-bit video rather than raw photon counts, so enabling the exact
operator here would falsely claim a physical Poisson witness. The installed
High Vision filter is the existing motion-transport integrator; raw/high-bit
capture and an explicitly approximate OBS witness are separate adapters.

The filter currently supports the common 8-bit camera formats (NV12, I420,
YUY2/UYVY, BGRA/RGBA/BGRX/BGR3, and 8-bit planar YUV). Unsupported 10/12/16-bit
formats pass through unchanged. YUV luma is normalized through each OBS
frame's full/limited-range metadata before transfer decoding and is mapped
back to the same code range afterward; limited-range black is not treated as
physical scene light.

The Cartoon path deliberately uses the complete spectral solver rather than
FACR because the current semismooth finite-flow chart is a two-axis spectral
operator. Selecting fused passes uses the same spectral plan. The plan is
rebuilt only when work-lattice dimensions or CPU thread count changes; changing
engine or fused pass count updates the persistent plan in place. Sampling maps
are cached on source/lattice changes. The decoded input plane aliases effective
cartoon, recomposition reuses it, and effect scratch is allocated lazily.

For synchronous GPU sources (including OBS 32's `macos-avcapture-fast`), the
Cartoon registration uses a two-surface Metal staging bridge. It reads back
only the fitted spectral content lattice, processes the previous surface, and
uploads one floating-point luminance correction. The shader applies that
correction to the current native-resolution source, preserving full-resolution
chroma and alpha. The bridge adds one frame of latency; it does not upscale a
256/512 image to impersonate native processing.

The macOS build also produces `bfft-cartoon-smoke`. It loads the actual bundle
through libobs, sends one synthetic async source through finite-flow fast,
fused-16, and Fine chrome, then exercises the synchronous Metal bridge against
a GPU-rendered source and writes PPM comparisons:

```sh
build-obs/bfft-cartoon-smoke \
  build-obs/bfft-cartoon.plugin/Contents/MacOS/bfft-cartoon \
  build-obs/cartoon-smoke
```
