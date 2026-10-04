| Measure | wrench_transport_js<br>wrench@60Hz | wrench_transport_js<br>wrench@120Hz | wrench_transport_js<br>soft_step_best@60Hz | wrench_transport_js<br>soft_step_16@60Hz | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 2.07 | 1.25 | 5.83 | 6.33 | not within 8 s | not within 8 s | not within 8 s | not within 8 s | 1.72 | not within 8 s | 6.87 |
| Fastest body in the last second (m/s) | 0.0011 | 7.6e-8 | 4.1e-4 | 2.8e-4 | 0.3582 | 0.1841 | 0.0370 | 0.8432 | 1.3e-5 | 0.1658 | 0.0021 |
| Mean kinetic energy in the last second (J) | 2.3e-8 | 9.9e-16 | 2.3e-9 | 1.1e-8 | 0.0012 | 7.7e-5 | 1.5e-5 | 0.0033 | 1.5e-12 | 3.9e-4 | 9.3e-9 |
| Deepest overlap at the end (mm) | 6.5e-4 | 8.5e-5 | 1.393 | 0.742 | 10.217 | 10.038 | 0.237 | 3.611 | 6.975 | 6.769 | 5.444 |
| Overlapping pairs (deeper than 0.01 mm) | 0 | 0 | 81 | 87 | 105 | 114 | 75 | 102 | 76 | 86 | 90 |
| Mean overlap of those pairs (mm) | 0.000 | 0.000 | 0.557 | 0.501 | 0.611 | 0.432 | 0.046 | 0.450 | 1.109 | 1.145 | 0.611 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Pile top (mm) | 174.9 | 152.0 | 150.1 | 142.7 | 101.1 | 132.1 | 141.7 | 110.7 | 180.6 | 164.0 | 161.7 |
| Mean centre height (mm) | 51.1 | 51.2 | 47.2 | 46.5 | 35.5 | 37.8 | 44.0 | 35.5 | 50.1 | 42.6 | 47.8 |
| Floor force / weight | 1.1036 | 1.0347 | 0.9483 | 0.9664 | 1.0081 | 1.0043 | 0.9952 | 0.9876 | 1.1499 | 1.1941 | 1.3521 |
| Floor and wall vertical force / weight | 0.999998 | 1.000000 | 1.000000 | 1.000088 | 1.003654 | 0.998767 | 1.003793 | 0.985582 | n/a | n/a | n/a |
| Wall-clock for 8 s simulated (s) | 2.00 | 2.96 | 1.41 | 1.96 | 30.37 | 28.16 | 18.40 | 11.90 | 0.22 | 0.17 | 0.66 |
| Simulated time / wall-clock | 4.0 | 2.7 | 5.7 | 4.1 | 0.3 | 0.3 | 0.4 | 0.7 | 35.9 | 45.9 | 12.1 |
| Worst step (ms) | 22.63 | 18.50 | 6.68 | 9.34 | 19.14 | 19.77 | 8.65 | 20.13 | 26.69 | 24.13 | 24.32 |
| Work | stepsPerSecond 60; meanContacts 399; newtonIterations 4237; factorizations 4232; passes 1293; lineSearchEvaluations 23950 | stepsPerSecond 120; meanContacts 407; newtonIterations 6260; factorizations 6255; passes 1819; lineSearchEvaluations 38442 | stepsPerSecond 60; meanContacts 432; contactSweeps 7680 | stepsPerSecond 60; meanContacts 445; contactSweeps 15360 | stepsPerSecond 480; solverIterations 39286; meanContacts 190 | stepsPerSecond 480; solverIterations 28947; meanContacts 241 | stepsPerSecond 480; solverIterations 32826; meanContacts 210 | stepsPerSecond 120; solverIterations 9619; meanContacts 224 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 |
