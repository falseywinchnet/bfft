# BFFT Vision for OBS

This module registers two native OBS asynchronous-video filters:

- **BFFT Cartoon**, with two display modes:
  - **Cartoon + texture** — the adjustable Meyer layer recomposition.
  - **Fine chrome** — chrome relief driven by the signed one-step outer-map
    defect `cartoon - ROF(input - texture, shading_c)`.

- **BFFT High Vision**, the continual temporal-imaging framework. Its
  **Synthetic HDR** and **Night integrator** modes transport scene-linear
  radiance, precision, and variance through estimated camera motion while
  releasing support in changing regions.

The Cartoon filter keeps the source color and runs its decomposition on luma.
High Vision instead processes the camera's native pixel lattice. Photon
evidence, detector-fixed noise, and motion are not inferred from a resized
surrogate. Night emits monochrome luma; Synthetic HDR retains source chroma.

The main effect uses the same independent layer mixer as the Python still
viewer:

```
output = residual
       + cartoon_gain * cartoon
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

At exactly `cartoon=0`, the remaining residual and texture are signed detail
rather than a color image. The plugin therefore displays that field around
mid-gray and neutralizes chroma. This preserves negative detail and prevents
low luma with retained NV12 chroma from producing false red regions.

Fine chrome performs one TV projection of `input - texture`, using 8–24
sweeps according to the Quality setting. This yields a useful local transport
direction without running the full slow Gilles loop. Relief depth and chrome
gloss are adjustable.

## Build on macOS

The OBS headers are not shipped inside OBS.app, so use a matching source
checkout. For the currently installed OBS 32.1.1:

```sh
git clone --depth 1 --branch 32.1.1 \
  https://github.com/obsproject/obs-studio.git /tmp/obs-studio-32.1.1

cmake -S obs-plugin -B build-obs \
  -DOBS_SOURCE_DIR=/tmp/obs-studio-32.1.1 \
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
High Vision**. For Cartoon, start with 12 passes and 4 CPU threads. For High
Vision, start with Synthetic HDR; Night integrator retains evidence longer for
fixed-exposure low-light capture.

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

The native engine and all frame buffers are persistent. Quality changes update
the outer pass count in place; they do not recreate plans or image-sized
scratch storage. A plan is rebuilt only when the aspect-derived processing
dimensions or CPU thread count changes, and the incompatible plan is released
before its replacement is allocated.

Source/work coordinate maps are cached on resolution changes. Layer
recomposition and Fine chrome shading run once per work-grid pixel before the
result is expanded to the source frame; the latter avoids repeating identical
trigonometric shading for every full-resolution pixel in a work-grid cell.
