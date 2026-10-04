# Finished-engine comparison

The final native engine is compared with MuJoCo 3.2.3 and Rapier 0.21.0 on
three 64-body mixed-convex scenes (seeds 1, 7, 19). Each advances eight seconds
with sleeping, damping, rolling resistance, and restitution disabled. All receive
the same input hull vertices, initial poses, density 2600 kg/m³, friction 0.5,
and gravity 9.81 m/s². One warm-up and three measured runs per seed/configuration;
configuration order rotates each repetition. Only step calls are timed.

The computer is an Apple M4 Mac mini on macOS 26.5. The native engine uses
AppleClang 21, Release, -O3 -DNDEBUG -ffp-contract=off. MuJoCo's native solver
is called through Python; Rapier's WebAssembly distribution through Node.
The timings include these call boundaries and do not include cooking or reporting.

| Engine | Configuration | Median compute seconds | Maximum final overlap mm | Settled scenes |
| --- | --- | ---: | ---: | ---: |
| wrench_transport_cpp | wrench@60Hz | 0.336044 | 0.013589 | 3/3 |
| wrench_transport_cpp | wrench@120Hz | 0.481208 | 0.024438 | 3/3 |
| rapier 0.21.0 | 4 solver iterations, 60 Hz, length unit 1 m | 0.226670 | 6.974646 | 3/3 |
| rapier 0.21.0 | 4 solver iterations, 60 Hz, length unit 0.05 m | 0.180673 | 9.793984 | 0/3 |
| rapier 0.21.0 | 8 solver iterations, 240 Hz, length unit 0.05 m | 0.652784 | 6.329829 | 2/3 |
| mujoco 3.2.3 | newton, elliptic cone, 60 Hz, multiccd | 6.290941 | 30.636573 | 0/3 |
| mujoco 3.2.3 | newton, elliptic cone, 120 Hz, multiccd | 10.075273 | 6.064288 | 0/3 |
| mujoco 3.2.3 | newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | 16.400007 | 0.241008 | 0/3 |

Compute time is the median across nine measured runs. Overlap is the maximum
final depth over all runs. The common scorer uses exhaustive face and edge
separating axes, independently of each engine's manifold generator. Settling
requires the maximum of linear speed plus radius times angular speed to remain
below 4 mm/s through the end of the run. MuJoCo final kinematics are refreshed
after integration before pose export. Kinetic energy is not used to rank engines
because the retained Rapier runner records translational energy only.

Wrench Transport at 60 Hz settles all three scenes in 1.20–1.37 seconds. Rapier's
default 60 Hz configuration settles all three with lower compute cost and larger
final overlap. The highest-rate MuJoCo configuration reduces overlap substantially
but retains motion above the quiet threshold during the observation interval.
The separate native optimization study is in NATIVE_STUDY.md; it is not the
subject of the engine paper.

Reproduction (on the Mini, with the existing external installations):

```sh
python3 experiments/wrench_transport/run_comparison.py \
  --native /tmp/wrench_native/phys_bench \
  --node /opt/homebrew/bin/node \
  --modules /Users/joshuahkuttenkuler/Developer/CodexBuilds/wrench_external \
  --output /tmp/wrench-comparison --repeats 3
```

Raw records, scene exports, common-score reports, and summary are in
`results/wrench-comparison/`. The comparison covers dense small-body packing;
the acceptance tests separately exercise other mechanical scenarios.
