| Measure | wrench_transport_cpp<br>wrench@120Hz | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m | mujoco 3.2.3<br>newton, elliptic cone, 60 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | wrench_transport_cpp<br>wrench@60Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 1.10 | 1.72 | not within 8 s | 6.87 | not within 8 s | not within 8 s | not within 8 s | 1.20 |
| Fastest body in the last second (m/s) | 8.1e-6 | 1.3e-5 | 0.1658 | 0.0021 | 0.6909 | 0.8432 | 0.0370 | 3.4e-6 |
| Mean kinetic energy in the last second (J) | 1.5e-12 | 1.5e-12 | 3.9e-4 | 9.3e-9 | 0.0087 | 0.0033 | 1.5e-5 | 1.1e-12 |
| Deepest overlap at the end (mm) | 0.002 | 6.975 | 6.769 | 5.444 | 13.948 | 3.740 | 0.241 | 0.011 |
| Overlapping pairs (deeper than 0.01 mm) | 0 | 76 | 86 | 90 | 102 | 101 | 76 | 1 |
| Mean overlap of those pairs (mm) | 0.000 | 1.108 | 1.145 | 0.611 | 1.001 | 0.463 | 0.046 | 0.011 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| Pile top (mm) | 153.6 | 180.6 | 164.0 | 161.7 | 114.6 | 110.7 | 141.7 | 170.8 |
| Mean centre height (mm) | 50.5 | 50.1 | 42.6 | 47.8 | 29.2 | 35.5 | 44.0 | 52.7 |
| Floor force / weight | 1.0521 | 1.1499 | 1.1941 | 1.3521 | 1.0196 | 0.9876 | 0.9952 | 1.0326 |
| Floor and wall vertical force / weight | n/a | n/a | n/a | n/a | 0.987844 | 0.985582 | 1.003793 | n/a |
| Wall-clock for 8 s simulated (s) | 0.46 | 0.22 | 0.17 | 0.64 | 6.54 | 11.68 | 18.11 | 0.34 |
| Simulated time / wall-clock | 17.6 | 36.9 | 47.2 | 12.6 | 1.2 | 0.7 | 0.4 | 23.9 |
| Worst step (ms) | 6.12 | 26.59 | 24.10 | 24.37 | 23.39 | 20.09 | 8.60 | 5.82 |
| Work | stepsPerSecond 120; newtonIterations 5660; factorizations 5655; passes 1634 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 | stepsPerSecond 60; solverIterations 5157; meanContacts 233 | stepsPerSecond 120; solverIterations 9619; meanContacts 224 | stepsPerSecond 480; solverIterations 32826; meanContacts 210 | stepsPerSecond 60; newtonIterations 3239; factorizations 3234; passes 972 |
