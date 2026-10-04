# Compound drop-and-pack accuracy and scaling study

Source: Lucky Iyinbor's 2026-09-28 post,
https://x.com/Luckyballa/status/2104797946281894190 . In the thread he identifies
"Collision accuracy" as the purpose (reply 2105147237823295755), and says the
collision handling method is the main test, with the solver also involved
(reply 2105270286417535380). The post supplies a video, not source geometry,
material parameters, timing, or a reference trajectory. This study reconstructs
the visible household-object packing task; it is not a numerical reproduction
of his unpublished implementation and cannot rank against that implementation.

Freeze the following design before examining comparative results.

* Objects: open cups, shallow bowls, bottles with narrow necks, thin plates,
  slender rods, and small convex pieces. Hollow shapes consist of disjoint
  convex wall sectors and a bottom. All engines receive exactly those sectors;
  no outer convex hull may fill a cavity. Collision geometry is rendered.
* Counts: 32, 64, 128, 256, 512. Seeds: 1, 7, 19. Object sizes, wall thickness,
  density, friction, gravity, and shape resolution remain fixed. Bin dimensions
  scale as the cube root of count, retaining nominal solid-volume fraction and
  aspect ratio. Drop in nonoverlapping batches over a bounded release phase,
  then observe eight more simulated seconds. The same release poses and times
  apply to all engines.
* Configurations: Wrench 60/120 Hz; Rapier 60 Hz/4 iterations and 240 Hz/8
  iterations with its characteristic length set to the object scale; MuJoCo
  120 Hz and 480 Hz with multiccd and elliptic friction (stiff 0.005 s solref
  at 480 Hz). Disable sleeping, restitution, rolling resistance and damping.
  Use fixed settings across sizes, and retain failures/timeouts explicitly.
* Timing: one warmup plus three measured repeats; rotate engine order. Run
  engines sequentially on the same M4. Record step total, median, p95, p99,
  worst, release-phase and all-bodies-active cost separately. Initialization,
  object activation, state export, and geometric scoring are outside step
  timing and reported separately where applicable. Native/WASM/Python call
  boundaries remain part of each measured implementation, as in the first
  study. No concurrent benchmark workloads.
* Geometry: independent exhaustive SAT over convex parts from exported poses,
  with bounding-box rejection only. Score at 10 Hz throughout and at the final
  pose; report maximum sampled penetration, final penetration, penetrating
  body-pair count, floor penetration and escapes. These are sampled geometric
  errors; they do not certify continuous-time absence of tunneling. Compound
  penetration is the maximum convex-part-pair depth, not a global compound
  minimum-translation distance.
* Rest: maximum of linear speed plus bounding radius times angular speed;
  require at least one complete second below 4 mm/s, with all bodies released.
  Record late motion, settling time, mass consistency, nonfinite states and
  native numerical guard activations. Settling alone is not physical accuracy.
* Independent controlled checks: a body falling into a hollow cup (cavity must
  remain empty), analytic free fall, and a thin-floor drop at multiple speeds
  and time steps. These isolate false collision, integration error and missed
  collision from chaotic pile trajectories. They are not a real-world
  validation experiment.
* Report speed and error together at every count. Include observed growth
  between counts and contact/convex-part counts. Do not extrapolate a universal
  asymptotic complexity from five measurements or treat an uncompleted run as
  a successful fast result. Match accuracy only when measured configurations
  actually satisfy the same stated error and settling thresholds.

The original 64-convex-body results remain a separate experiment.

## Measurement details and diagnostic extension

Rapier's solver iterations also perform internal substeps; nominal Hz is the
public `world.step` rate, not a claim of equal internal integration work. The
four/eight iteration settings stay explicit, and their cost is included.
See https://rapier.rs/docs/user_guides/c/integration_parameters/ . Free-fall
measurements independently expose the different integration errors.

The small controlled checks export every solver step. A thin-slab miss is
classified by a downward centre trajectory crossing the slab midplane inside
its footprint, inset by the body's bounding radius. Rolling off an edge is a
separate event. “Final centre below the slab” alone is not a tunneling test.

Fixed release poses can intersect an object that has already been ejected
upwards by an unstable simulation. `audit_release.mjs` measures this insertion
intersection separately. For the 32-body seed-7 Wrench 60 Hz failure, geometric
instability and escapes begin at 2.3 s; the first insertion overlap occurs at
2.8 s. It therefore follows the original instability rather than initiating it.

A separate, narrower 32-object reconstruction uses a 0.24 m-wide, 0.60 m-tall
bin and one release every 0.20 s. It matches the screenshot's object-to-bin
proportions more closely. This extra visualization/diagnostic is not folded
into the frozen 32–512 scaling series, whose bin is 0.42 × 0.42 × 0.68 m at
32 objects and grows geometrically from there.
