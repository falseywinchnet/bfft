| Measure | wrench_transport_cpp<br>wrench@60Hz | wrench_transport_cpp<br>wrench@120Hz | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m | mujoco 3.2.3<br>newton, elliptic cone, 60 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 1.20 | 1.10 | 1.72 | not within 8 s | 6.87 | not within 8 s | not within 8 s | not within 8 s |
| Fastest body in the last second (m/s) | 3.4e-6 | 8.1e-6 | 1.3e-5 | 0.1658 | 0.0021 | 0.6909 | 0.8432 | 0.0370 |
| Mean kinetic energy in the last second (J) | 1.1e-12 | 1.5e-12 | 1.5e-12 | 3.9e-4 | 9.3e-9 | 0.0087 | 0.0033 | 1.5e-5 |
| Deepest overlap at the end (mm) | 0.011 | 0.002 | 6.975 | 6.769 | 5.444 | 13.948 | 3.740 | 0.241 |
| Overlapping pairs (deeper than 0.01 mm) | 1 | 0 | 76 | 86 | 90 | 102 | 101 | 76 |
| Mean overlap of those pairs (mm) | 0.011 | 0.000 | 1.108 | 1.145 | 0.611 | 1.001 | 0.463 | 0.046 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| Pile top (mm) | 170.8 | 153.6 | 180.6 | 164.0 | 161.7 | 114.6 | 110.7 | 141.7 |
| Mean centre height (mm) | 52.7 | 50.5 | 50.1 | 42.6 | 47.8 | 29.2 | 35.5 | 44.0 |
| Floor force / weight | 1.0326 | 1.0521 | 1.1499 | 1.1941 | 1.3521 | 1.0196 | 0.9876 | 0.9952 |
| Floor and wall vertical force / weight | n/a | n/a | n/a | n/a | n/a | 0.987844 | 0.985582 | 1.003793 |
| Wall-clock for 8 s simulated (s) | 0.32 | 0.44 | 0.22 | 0.17 | 0.63 | 6.37 | 11.46 | 17.88 |
| Simulated time / wall-clock | 24.7 | 18.1 | 37.1 | 46.8 | 12.8 | 1.3 | 0.7 | 0.4 |
| Worst step (ms) | 5.53 | 6.04 | 27.21 | 24.60 | 24.65 | 22.91 | 19.78 | 8.59 |
| Work | stepsPerSecond 60; newtonIterations 3239; factorizations 3234; passes 972 | stepsPerSecond 120; newtonIterations 5660; factorizations 5655; passes 1634 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 | stepsPerSecond 60; solverIterations 5157; meanContacts 233 | stepsPerSecond 120; solverIterations 9619; meanContacts 224 | stepsPerSecond 480; solverIterations 32826; meanContacts 210 |
