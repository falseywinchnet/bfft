# Compound packing: measured results

This report is generated from saved runs. See `PROTOCOL.md` for source attribution, scene construction and measurement definitions.

| Bodies | Configuration | Completed / planned | Active mean step (ms) | Active compute (s) | Final overlap (mm) | Motion overlap (mm) | Settled seeds | Max escaped |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | Wrench · 60 Hz | 9 / 9 | 0.41401 | 0.19873 | 15938 | 15938 | 2 / 3 | 21 |
| 32 | Wrench · 120 Hz | 9 / 9 | 0.33619 | 0.32274 | 0.020261 | 5.7478 | 3 / 3 | 0 |
| 32 | Rapier · 60 Hz / 4 iterations | 9 / 9 | 0.35834 | 0.172 | 19.859 | 44.869 | 1 / 3 | 1 |
| 32 | Rapier · 240 Hz / 8 iterations | 9 / 9 | 0.34903 | 0.67013 | 9.1777 | 17.93 | 1 / 3 | 1 |
| 32 | MuJoCo · 120 Hz | 9 / 9 | 1.5443 | 1.4825 | 12.821 | 36.973 | 0 / 3 | 3 |
| 32 | MuJoCo · 480 Hz / stiff | 9 / 9 | 1.2645 | 4.8558 | 1.3319 | 8.6163 | 0 / 3 | 1 |
| 64 | Wrench · 60 Hz | 9 / 9 | 0.99147 | 0.47591 | 16883 | 16883 | 2 / 3 | 23 |
| 64 | Wrench · 120 Hz | 9 / 9 | 0.71676 | 0.68809 | 0.020237 | 3.6045 | 3 / 3 | 0 |
| 64 | Rapier · 60 Hz / 4 iterations | 9 / 9 | 0.78661 | 0.37757 | 21.753 | 56.299 | 0 / 3 | 1 |
| 64 | Rapier · 240 Hz / 8 iterations | 9 / 9 | 0.80663 | 1.5487 | 10.664 | 14.406 | 1 / 3 | 1 |
| 64 | MuJoCo · 120 Hz | 9 / 9 | 7.7062 | 7.3979 | 41.159 | 47.034 | 0 / 3 | 9 |
| 64 | MuJoCo · 480 Hz / stiff | 9 / 9 | 4.9318 | 18.938 | 13.452 | 41.37 | 0 / 3 | 3 |
| 128 | Wrench · 60 Hz | 9 / 9 | 2.4748 | 1.1879 | 28408 | 28408 | 0 / 3 | 34 |
| 128 | Wrench · 120 Hz | 9 / 9 | 2.6031 | 2.499 | 0.00058459 | 8.2774 | 1 / 3 | 0 |
| 128 | Rapier · 60 Hz / 4 iterations | 9 / 9 | 1.4283 | 0.68559 | 36.969 | 82.318 | 0 / 3 | 3 |
| 128 | Rapier · 240 Hz / 8 iterations | 9 / 9 | 1.6011 | 3.074 | 12.173 | 21.301 | 0 / 3 | 2 |
| 128 | MuJoCo · 120 Hz | 9 / 9 | 4.6597 | 4.4733 | 9.3101 | 97.837 | 0 / 3 | 10 |
| 128 | MuJoCo · 480 Hz / stiff | 3 / 9 | 0.26369 | 1.0126 | 0 | 22.488 | 0 / 3 | 5 |
| 256 | Wrench · 60 Hz | 9 / 9 | 5.3834 | 2.584 | 50386 | 50386 | 0 / 3 | 111 |
| 256 | Wrench · 120 Hz | 9 / 9 | 6.6033 | 6.3392 | 0.024736 | 14.322 | 3 / 3 | 0 |
| 256 | Rapier · 60 Hz / 4 iterations | 9 / 9 | 3.0644 | 1.4709 | 36.591 | 101.83 | 0 / 3 | 5 |
| 256 | Rapier · 240 Hz / 8 iterations | 9 / 9 | 3.3238 | 6.3817 | 14.71 | 22.462 | 0 / 3 | 2 |
| 256 | MuJoCo · 120 Hz | 9 / 9 | 3.4898 | 3.3502 | 0 | 97.551 | 0 / 3 | 14 |
| 256 | MuJoCo · 480 Hz / stiff | 6 / 9 | 0.20189 | 0.77525 | 10.136 | 24.043 | 0 / 3 | 2 |
| 512 | Wrench · 60 Hz | 9 / 9 | 13.849 | 6.6476 | 46689 | 46689 | 0 / 3 | 174 |
| 512 | Wrench · 120 Hz | 9 / 9 | 23.598 | 22.654 | 30735 | 30735 | 2 / 3 | 122 |
| 512 | Rapier · 60 Hz / 4 iterations | 9 / 9 | 7.3327 | 3.5197 | 26.6 | 96.699 | 0 / 3 | 8 |
| 512 | Rapier · 240 Hz / 8 iterations | 9 / 9 | 7.0345 | 13.506 | 13.194 | 26.325 | 0 / 3 | 3 |
| 512 | MuJoCo · 120 Hz | 9 / 9 | 5.5461 | 5.3242 | 33.146 | 111 | 0 / 3 | 23 |
| 512 | MuJoCo · 480 Hz / stiff | 9 / 9 | 0.31034 | 1.1917 | 0 | 48.393 | 0 / 3 | 2 |

A cross marker in the plots means at least one body escaped or an engine reported numerical instability at that size. Its timing is not successful-packing throughput. Compute shading spans the measured minimum and maximum; lines use medians. Only complete nine-run points are plotted.

## Controlled checks

“Tunneled” means a centre trajectory crosses the slab midplane inside its footprint, with an additional body-radius margin from every edge. Every solver step is exported. Exits over an edge are retained separately.

| Configuration | Cup cavity | 1 m/s slab | 10 m/s slab | 30 m/s slab | Free-fall error (mm) |
|---|---|---|---|---|---:|
| Wrench · 60 Hz | passed | caught / edge exit | caught / edge exit | tunneled | 40.875 |
| Wrench · 120 Hz | passed | caught / edge exit | caught / edge exit | tunneled | 20.437 |
| Rapier · 60 Hz / 4 iterations | passed | caught / edge exit | caught / edge exit | caught / edge exit | 10.223 |
| Rapier · 240 Hz / 8 iterations | passed | caught / edge exit | caught / edge exit | caught / edge exit | 1.2738 |
| MuJoCo · 120 Hz | passed | caught / edge exit | tunneled | tunneled | 20.438 |
| MuJoCo · 480 Hz / stiff | passed | caught / edge exit | caught / edge exit | tunneled | 5.1094 |
| Wrench · 480 Hz | passed | caught / edge exit | caught / edge exit | caught / edge exit | 5.1094 |

Free fall is measured for 0.5 seconds against the analytic centre-of-mass trajectory. Passing these controlled checks does not establish universal collision accuracy.

## Runtime limits

A run has a 180-second process budget. Scoring has a separate 180-second budget. A failed/timed-out scene/configuration is not retried in its measured repetitions; those records remain explicitly not run. Partial results are not counted as completed simulations.

