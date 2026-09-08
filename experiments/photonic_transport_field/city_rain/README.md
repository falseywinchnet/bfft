# Rain city: a frozen scene behind a dynamic normal-mapped volume

The delivered clip is **18 seconds at 1280×720, 60 fps**:

- Video: `output/final/city_rain_frozen_transport_720p60.mp4`
- Storyboard: `output/final/storyboard.png`
- Normal texture and direct comparison: `output/final/reference_comparison.png`
- Measurements: `output/final/summary.json`
- Replayable frozen field: `build/response.field.gz`

This is a new scene and a separate native bake/playback path using the existing
photonic transport engine. Playback constructs no city `Scene` or live
`TransportField`. It reads the immutable precomputed response and updates the
observer volume's procedural normal texture.

## Scene and observer boundary

The city has 22 building volumes, roads on a plane, textured facades, a distant
landmark, and small street lights: 128 geometric primitives and 111 retained
transport nodes. Its static irradiance, indirect transport, and surface atlases
are built with the existing native backend. The facade texture includes local
window emission; that texture is baked into outgoing appearance, rather than
introducing one area-light solve per window.

The camera is fixed at `(10, 7.5, 17)`, looking toward `(0, 2.5, -9)` with a
23.5-degree vertical half field of view. A flat glass sheet sits 1.20 world units
in front of the eye. It has index 1.52 and thickness 0.006. An adjacent
transparent observer volume has index 1.333 and thickness 0.045. Only its far
surface's normal texture changes.

The fixed entrance stack uses Snell refraction and a parallel-interface Fresnel
transfer. The far surface uses Snell refraction with Fresnel reflection into the
fixed environment. This is an observer-boundary normal-map model: geometry and
city lighting are frozen, so it intentionally does not update city caustics or
solve feedback from the changing observer boundary. Mist changes fine normals;
it is not an added opacity layer or a volumetric scattering simulation.

## What the invisible lens sites retain

The field is indexed by

```
L(surface_x, surface_y, normal_slope_x, normal_slope_y) -> linear RGB
```

Each entry is constructed by evaluating the actual 3-D city from the refracted
exit position and direction. The field therefore contains occlusion and parallax
changes associated with different normals. Playback has no input city image to
warp. The invisible lens sites are response samples, not geometric objects added
to the scene.

The final mapping contains:

| Quantity | Value |
|---|---:|
| Surface sites | 641 × 361 = 231,401 |
| Normal states per site | 49 × 49 = 2,401 |
| Total stored RGB responses | 555,593,801 |
| Packing | Three IEEE half-precision linear values, 6 bytes per response |
| Uncompressed payload | 3,333,562,806 bytes (3.105 GiB) |
| Precompute city evaluations | 494,298,767 |

Some stored states use total-internal-reflection/environment responses and do
not require a city query. The normal domain spans slopes ±0.65 per axis. The
simulator's bounded normal law stays strictly within ±0.55, so all playback
requests remain inside the precomputed domain. A gather interpolates the 16
neighboring entries of this four-dimensional lattice. There is no per-frame
admission test, city fallback, adaptive ray sampling, or field reconstruction.

The array has fixed size from the start; its storage does not grow with the
number of frames. Playback uses a **read-only memory map**, allowing clean
file-backed pages to be managed by the OS instead of copying the entire field
into a second anonymous array. The payload is unchanged across the video.

## Rain animation

The deterministic normal-texture simulator has 70 initially larger droplets,
270 smaller droplets, falling motion, trails, and area-based coalescence. A
periodic micro-droplet derivative texture supplies mist. The microtexture
advects continuously; new random noise is not generated independently each
frame. The final run records 208 coalescences.

The opening is clear. Larger droplets and smaller beads grow in, droplets slide
and merge, and mist gradually increases later in the clip. All visual changes
come from the normal texture and its resulting field gather. No buildings,
lights, camera, sheet geometry, or volume geometry move.

## Measured costs on the M4 Mini

| Stage | Measured cost |
|---|---:|
| Static city transport construction | 1.843 s |
| Dense response-field computation | 19.959 s |
| Read-only file mapping setup | 13.76 ms |
| Initial full-field integrity scan | 3.372 s |
| Warm clear frame | 7.02 ms |
| Median animated frame computation | 7.05 ms |
| Mean animated frame computation | 8.16 ms |
| 95th percentile frame computation | 11.42 ms |
| Maximum frame computation | 19.05 ms |
| All 1,080 frames, including pipe/recording backpressure | 9.51 s |

Frame computation includes simulation-state updates, per-pixel normal
calculation, field interpolation, and tone mapping. Two frames exceeded the
16.667 ms budget; the recording nevertheless contains all 1,080 frames at an
exact 60 fps timeline, with no frame dropping. This is a recorded throughput
measurement, not a hard real-time scheduling guarantee.

The static and field-construction timers exclude compilation, hashing, cache
serialization, archive compression, and transfer. The sequence timer excludes
file mapping, initial/final integrity scans, and final encoder teardown. These
costs are separate from steady playback. A prior anonymous-array playback run
is preserved as `playback_vector.json`; four captured source frames were checked
byte-for-byte against the read-only mapping implementation.

## Freeze verification

The native test mode checks:

- Four-dimensional affine interpolation.
- The parallel-slab direction identity.
- Deterministic rain and bounded normal slopes over 600 simulation steps.
- Frozen-file serialization/reload.
- Rejection of an attempted city evaluation after freeze.

The final recording additionally verifies:

- **0 city evaluations during playback.**
- **0 static transport updates.**
- **0 field rebuilds.**
- **1,080 normal-texture updates.**
- Identical field checksums before and after all frames.
- A valid H.264 MP4 containing exactly 1,080 frames at 60/1 fps, duration 18 s.

The city-evaluation entry point throws if invoked in frozen playback. More
fundamentally, playback is a separate mode that does not construct the objects
needed to perform a city evaluation. The response is mapped with `PROT_READ`.

## Where the stress test is vulnerable

The main observed weakness is finite interpolation across sharp visibility and
facade boundaries. A dense field still blends some neighboring responses where
a small normal change crosses a window edge, silhouette, or disocclusion.

An independent direct city evaluation at **12 seconds**, using the same normal
texture, took 83.91 ms and 921,593 city queries. That diagnostic runs outside
playback. Against it, the recorded gather path has:

| Metric, final 8-bit channel values | Result |
|---|---:|
| Mean absolute difference | 3.44 |
| RMS difference | 11.33 |
| Maximum local difference | 162 |
| Pixels differing by >1 level in any channel | 190,158 / 921,600 |
| Pixels differing by >8 levels in any channel | 136,908 / 921,600 |

The direct check uses the same optical/appearance model and a point evaluation;
it is not an independently validated physical ground truth or a pixel-footprint
integral. The comparison nevertheless exposes real representation error. The
video demonstrates smooth frozen-field playback, not exact equivalence to all
continuous city responses. Both direct and gathered stills, plus the actual
normal texture, are saved. The 512 held-out parameter queries in `precompute.json`
provide a second interpolation diagnostic.

## Reproduce

```sh
experiments/photonic_transport_field/city_rain/run_m4.sh final
.venv-jpeg/bin/python experiments/photonic_transport_field/city_rain/summarize.py
```

Use `run_m4.sh preview` for the smaller 960×540, 30 fps screen. The wrapper uses
`m4build`, the route selected by `m4host`, and the Mini's existing
`/opt/homebrew/bin/ffmpeg`. It installs no software. Results are copied back
immediately. The generated cache goes under `build/`, which is excluded from
both Git and `m4build` synchronization.

The wrapper streams a compressed cache home and verifies it, then removes its
own uncompressed `/tmp` field to recover disk space. Set
`KEEP_CITY_RAIN_FIELD=1` to retain that temporary field for immediate replay.
The captured run's compressed field is saved locally and has passed gzip
integrity verification; its SHA-256 sidecar is beside it. Dense regeneration
needs room for the 3.33 GB uncompressed field plus normal build/recording
headroom. The archive is about 1.1 GB.

After restoring the archive on the compute host, replay needs no city build:

```sh
/tmp/city_rain_native play \
  --field /tmp/city_rain_final/response.field \
  --stats /tmp/city_rain_final/replay.json --duration 18 --fps 60 \
  | /opt/homebrew/bin/ffmpeg -y -f rawvideo -pixel_format rgb24 \
      -video_size 1280x720 -framerate 60 -i pipe:0 \
      -an -c:v h264_videotoolbox -b:v 16000k -pix_fmt yuv420p \
      -movflags +faststart /tmp/city_rain_final/replay.mp4
```

The archive can be restored by streaming `gzip -dc` over SSH into that path,
without creating an extra compressed file on the Mini. Changing the fixed
camera, city, lighting, observer geometry, optical indices, or response domain
requires a new bake. Changing the rain simulation within the saved normal domain
does not invalidate the frozen city field.

## Expanded night city and pinhole-screen experiment

The follow-up uses dozens of buildings, street and vehicle lights, illuminated
interiors and a sun-lit geometric moon. It captures one pinhole illumination
screen and then animates pure screen deformation with a frozen mip pyramid.
See [NIGHT_SCREEN.md](NIGHT_SCREEN.md) for the new optical representation,
reproduction commands, video and limitations. It is a separate experiment from
the 4-D response field documented above.
