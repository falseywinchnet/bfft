| Measure | wrench_transport_cpp<br>wrench@120Hz | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 1 m | rapier 0.21.0<br>4 solver iterations, 60 Hz, length unit 0.05 m | rapier 0.21.0<br>8 solver iterations, 240 Hz, length unit 0.05 m | mujoco 3.2.3<br>newton, elliptic cone, 60 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 120 Hz, multiccd | mujoco 3.2.3<br>newton, elliptic cone, 480 Hz, multiccd, solref 0.005 s | wrench_transport_cpp<br>wrench@60Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Settled (max speed stays below 4 mm/s) at (s) | 1.17 | 2.12 | not within 8 s | 2.55 | not within 8 s | not within 8 s | not within 8 s | 1.37 |
| Fastest body in the last second (m/s) | 1.0e-5 | 2.9e-6 | 0.2365 | 1.6e-4 | 2.6181 | 0.4199 | 0.0225 | 4.2e-6 |
| Mean kinetic energy in the last second (J) | 1.6e-12 | 2.0e-12 | 1.3e-4 | 3.0e-10 | 0.0064 | 0.0024 | 1.6e-5 | 9.8e-13 |
| Deepest overlap at the end (mm) | 0.014 | 6.890 | 9.794 | 4.618 | 30.637 | 6.064 | 0.157 | 0.014 |
| Overlapping pairs (deeper than 0.01 mm) | 2 | 74 | 102 | 89 | 101 | 105 | 72 | 1 |
| Mean overlap of those pairs (mm) | 0.012 | 1.115 | 0.943 | 0.726 | 0.659 | 0.493 | 0.034 | 0.014 |
| Bodies outside the container | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| Pile top (mm) | 139.9 | 167.1 | 126.8 | 131.9 | 109.4 | 116.8 | 125.5 | 204.9 |
| Mean centre height (mm) | 46.6 | 49.4 | 41.5 | 44.1 | 32.0 | 34.8 | 38.7 | 50.4 |
| Floor force / weight | 1.0634 | 1.2246 | 1.2096 | 1.0742 | 1.5301 | 1.1872 | 1.0043 | 0.9965 |
| Floor and wall vertical force / weight | n/a | n/a | n/a | n/a | 0.989629 | 1.006848 | 1.004026 | n/a |
| Wall-clock for 8 s simulated (s) | 0.50 | 0.22 | 0.18 | 0.66 | 5.53 | 10.06 | 12.58 | 0.37 |
| Simulated time / wall-clock | 16.1 | 35.7 | 44.1 | 12.1 | 1.4 | 0.8 | 0.6 | 21.6 |
| Worst step (ms) | 4.92 | 27.22 | 25.32 | 24.40 | 18.47 | 20.91 | 6.20 | 8.89 |
| Work | stepsPerSecond 120; newtonIterations 5519; factorizations 5516; passes 1718 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 60; solverIterations 1920 | stepsPerSecond 240; solverIterations 15360 | stepsPerSecond 60; solverIterations 4741; meanContacts 240 | stepsPerSecond 120; solverIterations 9681; meanContacts 226 | stepsPerSecond 480; solverIterations 28164; meanContacts 215 | stepsPerSecond 60; newtonIterations 3253; factorizations 3250; passes 909 |
