| Measure | wrench_transport_js<br>wrench@60Hz | wrench_transport_js<br>wrench@120Hz | wrench_transport_js<br>soft_step_best@60Hz | wrench_transport_js<br>soft_step_16@60Hz | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 0.98 | 1.02 | 6.52 | 3.90 | not within 8 s | not within 8 s | not within 8 s | not within 8 s | 3.43 | 5.05 | 1.07 |
| Fastest body in the last second (m/s) | 6.9e-7 | 2.2e-9 | 0.0011 | 0.0010 | 0.1531 | 0.5012 | 0.0163 | 0.4843 | 4.3e-6 | 1.4e-6 | 5.1e-5 |
| Mean kinetic energy in the last second (J) | 1.2e-14 | 4.0e-20 | 2.6e-8 | 3.5e-8 | 6.1e-4 | 3.2e-4 | 1.1e-5 | 0.0014 | 5.8e-13 | 1.8e-13 | 9.8e-11 |
| Deepest overlap at the end (mm) | 2.4e-4 | 3.9e-8 | 0.968 | 0.659 | 7.131 | 1.144 | 0.188 | 6.663 | 12.030 | 3.234 | 4.499 |
| Overlapping pairs (deeper than 0.01 mm) | 0 | 0 | 92 | 94 | 89 | 92 | 78 | 114 | 81 | 99 | 88 |
| Mean overlap of those pairs (mm) | 0.000 | 0.000 | 0.531 | 0.470 | 0.665 | 0.348 | 0.042 | 0.505 | 1.111 | 0.634 | 0.619 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pile top (mm) | 202.5 | 198.3 | 165.5 | 179.6 | 123.0 | 117.5 | 145.1 | 109.7 | 139.2 | 154.5 | 192.2 |
| Mean centre height (mm) | 57.5 | 56.8 | 49.4 | 44.4 | 36.8 | 39.5 | 42.2 | 35.6 | 47.0 | 44.3 | 51.6 |
| Floor force / weight | 1.0180 | 1.0024 | 0.9988 | 0.9981 | 1.0020 | 1.0017 | 0.9863 | 1.0122 | 1.1834 | 1.2265 | 1.1111 |
| Floor and wall vertical force / weight | 1.000000 | 1.000000 | 0.999976 | 0.999998 | 1.003571 | 0.998531 | 0.993155 | 1.016015 | n/a | n/a | n/a |
| Wall-clock for 8 s simulated (s) | 1.58 | 2.82 | 1.49 | 2.01 | 29.50 | 20.45 | 20.93 | 8.50 | 0.23 | 0.16 | 0.66 |
| Simulated time / wall-clock | 5.1 | 2.8 | 5.4 | 4.0 | 0.3 | 0.4 | 0.4 | 0.9 | 34.5 | 49.4 | 12.2 |
| Worst step (ms) | 26.12 | 18.80 | 8.47 | 9.96 | 20.59 | 16.37 | 10.95 | 19.05 | 27.49 | 24.92 | 24.79 |
| Work | stepsPerSecond 60; meanContacts 386; newtonIterations 2819; factorizations 2816; passes 918; lineSearchEvaluations 16067 | stepsPerSecond 120; meanContacts 374; newtonIterations 4767; factorizations 4764; passes 1564; lineSearchEvaluations 34719 | stepsPerSecond 60; meanContacts 430; contactSweeps 7680 | stepsPerSecond 60; meanContacts 449; contactSweeps 15360 | stepsPerSecond 480; solverIterations 37185; meanContacts 182 | stepsPerSecond 480; solverIterations 24800; meanContacts 226 | stepsPerSecond 480; solverIterations 30944; meanContacts 215 | stepsPerSecond 120; solverIterations 9467; meanContacts 229 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 |
