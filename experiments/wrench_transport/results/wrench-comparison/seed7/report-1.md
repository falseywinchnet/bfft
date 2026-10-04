| Measure | wrench_transport_cpp<br>wrench@120Hz | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m | mujoco 3.2.3<br>newton, elliptic cone, 60 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | wrench_transport_cpp<br>wrench@60Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 1.33 | 2.22 | not within 8 s | not within 8 s | not within 8 s | not within 8 s | not within 8 s | 1.33 |
| Fastest body in the last second (m/s) | 8.1e-6 | 7.1e-6 | 0.1767 | 0.2850 | 0.4780 | 0.1451 | 0.0446 | 3.7e-6 |
| Mean kinetic energy in the last second (J) | 1.1e-12 | 1.6e-12 | 2.2e-4 | 4.3e-5 | 0.0041 | 5.9e-4 | 1.1e-5 | 6.8e-13 |
| Deepest overlap at the end (mm) | 0.024 | 6.088 | 7.269 | 6.330 | 10.148 | 1.358 | 0.137 | 6.5e-4 |
| Overlapping pairs (deeper than 0.01 mm) | 1 | 81 | 103 | 86 | 110 | 115 | 63 | 0 |
| Mean overlap of those pairs (mm) | 0.024 | 1.110 | 0.858 | 0.747 | 0.937 | 0.350 | 0.041 | 0.000 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pile top (mm) | 183.2 | 146.1 | 132.1 | 141.2 | 112.1 | 112.5 | 137.5 | 184.8 |
| Mean centre height (mm) | 51.7 | 46.0 | 41.3 | 45.2 | 33.6 | 35.2 | 41.9 | 56.8 |
| Floor force / weight | 1.0625 | 1.1587 | 1.1917 | 1.2202 | 1.0494 | 0.9627 | 1.0023 | 1.0995 |
| Floor and wall vertical force / weight | n/a | n/a | n/a | n/a | 1.002047 | 0.980371 | 1.008312 | n/a |
| Wall-clock for 8 s simulated (s) | 0.48 | 0.24 | 0.18 | 0.65 | 6.30 | 8.37 | 16.39 | 0.33 |
| Simulated time / wall-clock | 16.8 | 33.7 | 44.3 | 12.2 | 1.3 | 1.0 | 0.5 | 24.5 |
| Worst step (ms) | 4.54 | 27.49 | 24.63 | 24.62 | 20.15 | 18.76 | 8.71 | 7.09 |
| Work | stepsPerSecond 120; newtonIterations 5591; factorizations 5588; passes 1718 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 | stepsPerSecond 60; solverIterations 5107; meanContacts 228 | stepsPerSecond 120; solverIterations 8768; meanContacts 230 | stepsPerSecond 480; solverIterations 30545; meanContacts 207 | stepsPerSecond 60; newtonIterations 3194; factorizations 3191; passes 923 |
