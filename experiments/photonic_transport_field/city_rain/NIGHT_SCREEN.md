# Night city through a captured illumination screen

This experiment implements a second representation alongside the original
four-dimensional response field. A fixed pinhole camera captures the illuminated
city into one linear-RGB screen. Rain deforms screen coordinates; all animated
color comes from that screen.

Run from the repository with:

```sh
experiments/photonic_transport_field/city_rain/run_night_screen.sh
```

The script builds and tests on the host selected by `m4host`, captures the scene,
records 1,080 frames at 60 fps, and copies the screen, video and measurements back
to `output/night_screen/`. It uses the existing native renderer and installed
Mini encoder. No extra software is installed.

## Scene

The city has 68 building volumes, including the original core, wider flanking
blocks and additional blocks behind it. It adds rooftop equipment, cornices,
antenna beacons, streetlamp poles, and 36 vehicles across three avenues. Vehicle headlamps,
taillamps, streetlamps and rooftop beacons have visible emissive geometry and
local point sources. During capture, point sources use inverse-square diffuse
illumination and shadow visibility; the road adds a specular highlight lobe.
They remain static in the screen.

Windows have individually assigned warm/cool illumination, dark rooms, curtains
and mullions. Each lit opening evaluates an analytic room behind the facade: the
view ray intersects the room walls or its small luminous lamp globe, and an
interior point lamp illuminates those walls. These local room evaluations do not
add thousands of primitives to the global city BVH. Their outgoing illumination
is captured into the screen; inter-room or room-to-street bounce is not modeled. The main area source and
indirect diffuse field are compiled by the retained native transport backend.
Local point-source shading is evaluated during the capture.

A skybox lies at +/-10 million scene units. Its stable procedural texture
contains a night gradient, faint cloud structure and sparse stars. The moon is
an actual intersected sphere centered at (-90000,115000,-420000), radius 25500,
with procedural lunar albedo and shallow normal relief. It is shaded by light
from a sun at (6000000,2500000,7000000), behind the observer. The moon has a
Lambertian illuminated side, rather than an emissive disk pasted into the sky.
The celestial sizes and sky brightness are composed for this scene, not an
astronomical or photometric calibration. The sun is represented as a distant
point illumination source for the moon. Fog is a fixed depth-based appearance
term captured into the screen.

## Pinhole and droplet relationship

The fixed capture camera is at (10,8,19), looking at (0,4,-10), with a vertical
half field of view of 26 degrees. It represents the city-facing pinhole at the
sheet. The screen stores the illumination seen from that single center of
projection. A virtual screen in front of a pinhole has the same chart orientation
as the scene; a physical screen behind the pinhole would require the usual image
inversion when mounted for viewing.

For output coordinates (x,y), the captured chart is sampled at

```
u = x + f * (1.333 - 1) * normal_slope_x
v = y - f * (1.333 - 1) * normal_slope_y
f = output_height / (2 * tan(26 degrees))
```

This is an explicit paraxial slope-to-screen deformation law. It is not an exact
Snell reconstruction through a finite water/glass stack. The flat state is
identity. Rain support supplies the normal slope; outside droplet/mist support,
that slope is zero. There is no added opacity, reflection overlay, brightness
change, or additional city query in the deformation stage.

The existing rain simulator gives moving large drops, small beads, trails,
coalescence and advected microdroplet normals. The entire normal law is bounded
in magnitude by 0.55. The screen has 192 output pixels of overscan on every side,
which exceeds the maximum possible displacement (about 135.2 pixels at 720p).
The capture uses two samples per output pixel along each axis, yielding a
3328 x 2208 screen. Half-precision linear RGB requires 44,089,344 bytes for the base image.
A deterministic mip pyramid brings the complete resident payload to 58,785,468
bytes, approximately one third above the base image.
Playback computes one shared normal map per frame, then estimates the deformation
Jacobian using neighboring samples of that map,
uses its largest singular value to select an isotropic screen footprint, and
performs trilinear mip sampling in linear RGB before tone mapping. The entire
pyramid is fixed before the warm frame, included in the integrity checksum, and
never updated during animation. The serialized screen contains the base image;
loading deterministically reconstructs its pyramid before playback.

## Invariants and limits

The `play` executable mode constructs no city, beam field, or transport field.
It loads the screen into a const object, verifies its full checksum before and
after recording, and asserts that no city evaluation was attempted. An
out-of-support lookup fails explicitly. Tests check affine screen interpolation,
the clear-screen identity, constant preservation through every mip level, screen
serialization and mip reconstruction, support rejection, the frozen-city guard and the
existing rain/field invariants.

The dimensional reduction deliberately removes the earlier field's alternate
views. Deformation can stretch, compress and fold what the pinhole recorded;
it cannot reveal geometry hidden from that pinhole. Screen supersampling and the mip pyramid improve
spatial sampling, but do not recover disocclusions. Isotropic filtering can blur
more than necessary under strongly anisotropic deformation. Its finite-difference
Jacobian is a local estimate, not an exact integral of a nonlinear pixel footprint.
The initial bilinear stress snapshot is saved as `initial_bilinear_stress.png`;
it exposed severe sparkle under fine mist and motivated the filtered final path.
It precedes the city expansion, final moon and local-room polish, so it is not a matched visual
comparison of filtering alone.

## Delivered recording and measurements

- [18-second 720p60 video](output/night_screen/night_city_screen_rain_720p60.mp4)
- [Clear city](output/night_screen/night_clear.png)
- [Storyboard](output/night_screen/storyboard.png)
- [Frame times](output/night_screen/frame_times.png)
- [Verified measurements](output/night_screen/summary.json)
- Replayable screen: `output/night_screen/illumination.screen`

The final scene contains 68 building volumes (the landmark uses two stacked
volumes), 36 vehicles, 24 rooftop red beacons, 240 local point lights and 2,119
native scene primitives. The local analytic window rooms are additional capture
appearance, not extra global BVH primitives.

| Measurement, M4 Mini CPU | Result |
|---|---:|
| One-time native city/lighting preparation | 800.314 s (13 min 20 s) |
| Pinhole screen capture | 5.875 s |
| Pinhole sample queries | 7,348,224 |
| Base screen payload | 44,089,344 bytes |
| Frozen screen including mip pyramid | 58,785,468 bytes |
| Reduction against the previous 3.33 GB response payload, including mips | 56.7× |
| Median frame computation | 8.095 ms |
| 95th percentile frame computation | 13.006 ms |
| Mean frame computation | 8.743 ms |
| Maximum frame computation | 24.499 ms |
| Frames exceeding 16.667 ms | 1 / 1,080 |
| Recording loop including pipe backpressure and snapshot writes | 10.085 s |
| City evaluations during playback | 0 |
| Screen updates during playback | 0 |

The larger city exposed an expensive serial preparation stage in the existing
native renderer: constructing area-light visibility and surface irradiance
atlases around the detailed geometry. Screen capture and playback are separate
from that work. The screen representation saves runtime memory and evaluation;
it does not make this existing cold-lighting preparation cheap.

The preparation and capture timers exclude compilation, mip construction,
serialization and integrity scans. Frame computation includes rain state update,
normal-map generation, deformation-footprint estimation, mip lookup and tone
mapping. It excludes screen loading/mip reconstruction, pre/post checksum scans,
pipe writes and encoder teardown. The 58.8 MB figure is frozen image storage;
normal scratch storage (about 7.4 MB), output buffers and other process allocations
are additional. The clip is a fixed 60 fps recording, not a hard real-time
scheduling guarantee.

The screen-plus-mip checksum before and after playback was
`2970080582719551191`, matching the bake. FFprobe verified 1,080 H.264 frames,
1280×720, 60/1 fps and 18.000 seconds; a complete decode reported no errors.
The final encoded 6-second frame was inspected in addition to the clear and
rainy source frames. Four saved source checkpoints were byte-identical before
and after sharing the normal map, so that optimization preserved those images.
The earlier repeated-normal timings are retained in
`playback_repeated_normals.json`. Source, cache and video SHA-256 hashes are
saved in `source_manifest.json`.

### Source-extinction acceleration

The subsequent [source-extinction change](SOURCE_EXTINCTION.md) reduces the
800-second native preparation above to approximately one minute. It extinguishes
opaque source intervals before their scene-BVH walks and produces a byte-identical
illumination screen. The video and playback measurements above remain valid.
`run_night_screen.sh` now enables that path; `night_screen_native bake-legacy`
retains the original preparation for comparison.
