| Measure | wrench_transport_js<br>wrench@60Hz | wrench_transport_js<br>wrench@120Hz | wrench_transport_js<br>soft_step_best@60Hz | wrench_transport_js<br>soft_step_16@60Hz | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 1.27 | 1.70 | 2.53 | 2.10 | not within 8 s | not within 8 s | not within 8 s | not within 8 s | 1.08 | not within 8 s | not within 8 s |
| Fastest body in the last second (m/s) | 3.9e-7 | 9.0e-9 | 1.1e-4 | 5.2e-4 | 0.1653 | 0.1225 | 0.0641 | 0.7963 | 1.8e-6 | 0.2249 | 0.2937 |
| Mean kinetic energy in the last second (J) | 7.6e-15 | 5.8e-18 | 1.7e-9 | 1.9e-8 | 2.7e-4 | 1.4e-4 | 3.0e-5 | 0.0030 | 6.8e-13 | 5.3e-4 | 1.1e-4 |
| Deepest overlap at the end (mm) | 2.9e-4 | 1.0e-6 | 0.845 | 0.726 | 6.860 | 1.388 | 0.200 | 7.838 | 7.221 | 7.351 | 5.621 |
| Overlapping pairs (deeper than 0.01 mm) | 0 | 0 | 85 | 91 | 114 | 102 | 85 | 110 | 87 | 95 | 92 |
| Mean overlap of those pairs (mm) | 0.000 | 0.000 | 0.526 | 0.491 | 0.568 | 0.318 | 0.038 | 0.526 | 1.240 | 0.949 | 0.910 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pile top (mm) | 170.4 | 139.9 | 165.3 | 168.5 | 112.1 | 113.4 | 127.7 | 120.4 | 154.7 | 167.5 | 147.4 |
| Mean centre height (mm) | 55.6 | 51.3 | 45.6 | 46.4 | 34.7 | 35.8 | 41.2 | 36.2 | 48.1 | 43.9 | 50.4 |
| Floor force / weight | 1.0524 | 1.0432 | 0.9829 | 0.9707 | 1.0139 | 0.9702 | 0.9807 | 1.0036 | 1.1329 | 1.1772 | 1.1206 |
| Floor and wall vertical force / weight | 1.000000 | 1.000000 | 1.000001 | 0.999902 | 0.977347 | 0.992884 | 0.998218 | 1.022065 | n/a | n/a | n/a |
| Wall-clock for 8 s simulated (s) | 1.66 | 3.11 | 1.48 | 2.01 | 24.70 | 11.81 | 30.57 | 10.45 | 0.23 | 0.17 | 0.66 |
| Simulated time / wall-clock | 4.8 | 2.6 | 5.4 | 4.0 | 0.3 | 0.7 | 0.3 | 0.8 | 35.2 | 45.8 | 12.2 |
| Worst step (ms) | 32.18 | 19.43 | 7.68 | 8.74 | 11.10 | 7.16 | 17.91 | 22.86 | 27.07 | 24.33 | 24.36 |
| Work | stepsPerSecond 60; meanContacts 410; newtonIterations 2793; factorizations 2793; passes 909; lineSearchEvaluations 16443 | stepsPerSecond 120; meanContacts 401; newtonIterations 5522; factorizations 5519; passes 1712; lineSearchEvaluations 35190 | stepsPerSecond 60; meanContacts 471; contactSweeps 7680 | stepsPerSecond 60; meanContacts 438; contactSweeps 15360 | stepsPerSecond 480; solverIterations 34270; meanContacts 199 | stepsPerSecond 480; solverIterations 24441; meanContacts 236 | stepsPerSecond 480; solverIterations 33105; meanContacts 209 | stepsPerSecond 120; solverIterations 9479; meanContacts 222 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 |
