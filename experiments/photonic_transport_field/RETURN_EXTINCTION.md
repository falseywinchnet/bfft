# Return extinction: rejected as an expensive-pixel accelerator

The user proposed stopping reversed/returning transport as an extinction
condition for further information flow. This experiment directly tests that
mechanism, separately from preserving a recurring path's accumulated radiance.
All modes are experimental; the default is `--return-mode off`.

## Mechanisms actually tested

* `path-extinction`: stop at the existing A-B-A face return with the existing
  direction-return test, dot product greater than 1-1e-9. This does not require
  the same surface position and is not a certificate of repeated optical state.
* `state-extinction`: stop when an ancestor has exactly matching primitive,
  hit position, incident direction, normal, geometric normal, front/back flag,
  and area-emitter exclusion state. Channel, wavelength, scene, and field are
  fixed for the enclosing trace. Equality is numerical, with no tolerance.
* `state-closure`: at that same exact return, replace further scalar feedback
  by its geometric-series contribution. The cycle may have terminal diffuse,
  emissive, or empty side exits. Nonterminal side exits, incomplete/pruned
  cycles, gains above one, and driven unit-gain loops are rejected and marched
  normally. A zero-source unit-gain loop contributes zero under this renderer's
  steady source-driven interpretation; it does not model initial stored light.

Closure adds B*g/(1-g), where B is the weighted source and terminal-exit
contribution of the already traversed cycle, and g is the returning-to-ancestor
weight ratio. Existing first-cycle exits remain represented. This is an
infinite-tail extension, not bitwise equivalence to the ordinary finite-depth,
finite-weight-cutoff tracer. It does not handle arbitrary coupled recurrences.

## Established scenes, 800x600 on the M4 CPU

Four modes, four static default-camera scenes, three rotated repeats give 48
measured frames. Fields were compiled once per scene, outside raster timing.
All used linear primitive intersection and the accepted retained transport.
The table reports median complete raster milliseconds; this is a focused
three-repeat screen, not a robust claim of sub-percent speedup.

| Scene | Off | Path extinction | Exact-state extinction | Exact-state closure |
|---|---:|---:|---:|---:|
| Standard | 185.585 | 185.008 | 189.172 | 188.544 |
| Aperture canyon | 1070.055 | 1065.331 | 1068.274 | 1069.647 |
| Mirror relay | 752.123 | 744.535 | 754.781 | 754.039 |
| Occlusion garden | 411.717 | 413.548 | 416.588 | 416.155 |

All 48 frames match their scene's off-mode 8-bit RGB reference byte for byte.
That does not establish equality of unquantized radiance.

Path extinction fired 246, 469, 246, and 249 times per frame, respectively.
In aperture canyon it reduced secondary events from 1,556,936 to 1,555,604
(0.086%) and source quadrature from 16,186,047 to 16,109,622 (0.472%).
The full-frame counters include work beyond the expensive boundary fallback.
It barely reaches the dominant work.

Neither exact-state mode found an exact return in those scene/camera cases.
This does not rule out geometric near-returns: the matcher intentionally does
not merge floating-point states or drifting positions. It means this exact
criterion has no measured acceleration coverage in this screen.

## Controlled recurrent-light fixtures

The reference uses cutoff 1e-12 and up to 512 interactions. Both fixtures
contain exact returns, and both have nonzero contributions from repeated cycles.

| Fixture | Deep reference | State extinction | State closure |
|---|---:|---:|---:|
| Rough reflectors receiving side illumination | .01558076586658 | .01014442376016 | .01558076586660 |
| Dielectric cavity with terminal source exits | 4.07692307691875 | 3.27160493827160 | 4.07692307692308 |

Extinction loses approximately 34.89% and 19.75% of radiance, respectively.
Closure agrees with the deep reference within 1e-9 relative-plus-absolute test
tolerance and uses four or five traced events. The dielectric result also
matches the independent analytic geometric series to 1e-12.

This distinguishes eliminating duplicate path expansion from eliminating
light. Exact recurrence can justify a compact response; setting that response
to zero is not generally correct.

Additional checks reject displaced hit positions, changed source exclusion,
opposite incident directions, drifting paths, nonterminal side exits, pruned
cycles, and driven unit-gain feedback. Optimized and ASan/UBSan fixture runs
pass. Existing scene, update, camera, boundary, and source-support regressions
pass. The new executable with return mode off produces a byte-identical
800x600 aperture-canyon frame to the saved pre-experiment executable.

## Decision and reproduction

Rejected as a way to cheaply terminate the overactive pixels. The weak face
criterion removes too little work; exact-state recurrence has no measured
coverage in the established frames; unconditional extinction loses substantial
real contributions in the controlled fixtures. The mode stays off by default.

Implementation: `native/return_extinction.hpp`, with optional hooks in
`native/regime_scene_native.cpp`. Tests and benchmark:
`native/return_extinction_probe.cpp`. Raw JSON, summaries, logs, and aperture
PPMs are in `return_extinction_m4/`.

Run the full screen and copy results back using the selected M4 route:

```sh
sh experiments/photonic_transport_field/run_return_extinction.sh
```

The focused native fixture checks also have a `return-test` Make target;
run that through m4build. This experiment does not alter the scattering model
or establish dynamic-scene or 30-fps performance.
