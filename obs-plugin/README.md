# BFFT Cartoon filter for OBS

This is a native OBS asynchronous-video filter around the existing
`bfft_meyer_split` C API. It keeps webcam color and runs the decomposition on
luma at a power-of-two working size no larger than 512x256. A 426x240 camera
feed therefore uses the measured real-time 512x256 path.

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

The display selector also includes:

- **Difference field** - the signed residual between the smooth analytical
  cartoon and the flatter cartoon produced by one terminal TV/ROF solve,
  contrast-normalized around neutral gray.
- **Liquid chrome relief** - treats that field as a height map, derives surface
  normals from its gradient, displaces the camera image, and shades the result
  with a striped chrome environment and specular light.
- **Fine chrome** - uses the same chrome renderer with the fine one-step
  outer-map defect
  `cartoon - ROF(input - texture, shading_c)`. This is the inexpensive local
  counterpart of the converged TGFD-to-Gilles correction field.
- **OKLCH independent** - converts the working image to OKLCH, decomposes L,
  C, and H with separate Meyer solves, and exposes independent cartoon and
  texture gains for each component.
- **OKLCH individual delays** - uses the same independent component solves,
  with separate 0–500 ms controls for L, C, and H.
- **Four hue sectors delayed** - divides hue into four equal 90-degree sectors.
  Each sector receives an independent masked lightness solve, texture gain,
  0–500 ms delay, and hue rotation. A per-sector payload selector can replace
  processed lightness with the signed Meyer residual, chroma data, or the
  texture field.
- **OKLCH echo prism** - accumulates 1–8 decaying taps independently along
  each component's delay.
- **Hue-sector echoes** - gives each hue quadrant its own decaying temporal
  trail while retaining its selected payload and hue rotation.
- **Continuous hue time prism** - interpolates the four sector-delay controls
  continuously around the hue wheel, turning hue into time.
- **Chroma comet trails** - delays saturated pixels more strongly than neutral
  pixels. Chroma delay weight controls the depth law.
- **Hue-time pinwheel** - combines hue and polar image angle into a rotating
  delay field. Pinwheel turns controls its spatial winding.
- **Live cartoon / delayed scene** - keeps a texture-free current cartoon in
  real time over one naturally delayed scene frame. A motion-sensitive mask
  reveals the live cartoon only where current and delayed motion diverge, so
  stationary background remains natural. Scene delay, live strength, motion
  threshold/feather, and live-only hue/chroma are adjustable.
- **Tiled time glass** - replaces the pinwheel's angular delay field with a
  grid of temporal glass panes. Each tile has an ordered or randomized time
  phase, lens displacement clamped inside the pane, and a lightness bevel.
  Tile size, delay depth, refraction, bevel, and randomness are adjustable.

All color modes also expose delayed/current mix, hue rotation, and chroma
scale. Echo modes add tap count and decay.

The temporal modes do not sleep or block for the chosen delay. They retain a
bounded history of at most 40 processed 512x256 OKLCH frames and select by OBS
timestamps. History components are stored as bounded 16-bit values, so the
maximum history is about 30 MiB rather than 60 MiB in 32-bit floats. Existing
luma/chrome modes do not retain this history.

The difference-field and liquid-chrome modes retain the original relation
`difference = cartoon - ROF(cartoon, 0.02)`. This is the one-TV-solve route
from the smooth state to the flat state; it reproduces the liquid flow
geometry without running the multi-second nested Gilles reference on every
frame. Fine chrome instead projects `input - texture` once, using 8–24 TV
sweeps according to the Quality setting. The live TGFD split remains capped
at eight passes to protect frame rate. Relief depth and chrome gloss are
adjustable.

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

Restart OBS, open the webcam source's **Filters**, add **BFFT Cartoon**, and set
the camera to 426x240 at 30 fps. Start with 12 passes and 4 CPU threads.

The filter currently supports the common 8-bit camera formats (NV12, I420,
YUY2/UYVY, BGRA/RGBA/BGRX/BGR3, and 8-bit planar YUV). Unsupported 10/12/16-bit
formats pass through unchanged.

The native engine and all frame buffers are persistent. Quality changes update
the outer pass count in place; they do not recreate FFT plans or image-sized
scratch storage. A plan is rebuilt only when the processing dimensions or CPU
thread count changes, and the incompatible plan is released before its
replacement is allocated.
