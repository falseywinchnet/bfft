# Active 2-D FMMT acceleration

## Outcome

The now-rejected GUI control—continuous-support FMMT—was accelerated without
changing its denoising equations, posterior, physical constants, or default
quality behavior. Work remains in NumPy/SciPy because the 2-D theory is still
evolving; this is representation optimization, not premature C++ promotion.

On the M4 Mini, the matched 256-square Cameraman mixed-corruption run now has a
three-repeat median kernel time of `1.205 s`. The pre-change matched single run
was `1.390 s`; this is approximately 13% lower latency, with the repeat record
being the authoritative current measurement.

The GUI previously computed the complete continuous support field once for its
visual panel and then again inside FMMT. Sharing that exact field lowers median
GUI latency:

| size | duplicated support | shared support | latency reduction | speedup |
|---:|---:|---:|---:|---:|
| 128 square | 0.2781 s | 0.2346 s | 15.7% | 1.186x |
| 256 square | 1.3303 s | 1.2152 s | 8.7% | 1.095x |

The shared and internally computed paths are pixel-for-pixel identical in the
record (`maximum difference = 0`).

## Representation changes

### One front workspace

SciPy Dijkstra returns a batch-by-pixel distance matrix. The old path allocated
a second matrix for attenuation. Distance is now transformed into attenuation
in place after the finite-distance mask is recorded. The explicit front
workspace therefore falls from two float64 fields per anchor/pixel to one.

The saved workspace admits a 512-anchor batch at 256 square under the same
256-MiB policy. Batch size remains a reported function of pixel count and
workspace, not an image-content or quality setting.

### One joint packet multiplication

Signal and residual packets already travel on identical ordered fronts. They
are now concatenated into one vector packet and accumulated by one matrix
multiplication per front batch, then split back into their exact marginals.
This removes duplicated dispatch and improves the dense packet kernel without
changing its graph or attenuation.

### Reused recurrence geometry

The separable histogram bootstrap evaluates the same horizontal and vertical
edge transmission laws in both sweep orderings. Those laws are now formed once
per transport call and reused by both recurrences. The scalar Numba recurrence
remains the exact representation oracle; the accelerated vector recurrence is
bit-identical in its invariant test.

### One compiled histogram filter

All empirical histogram bins share the same spatial box operator. A single
three-dimensional `uniform_filter` with unit channel extent replaces one
SciPy call per bin. It is bit-identical to the per-bin reference for the signal
and residual packet configurations.

### Complementary support lanes

The two checkerboard witness lanes are exact complements. Gaussian filtering
is linear and reflection preserves constants, so the second denominator and
masked numerator are derived from the first lane and full-field filter. The
optimized lane estimate agrees with the two-lane reference within `1e-15`.
Residue masks are cached by image shape.

### GUI support sharing

`transport_support_birth` accepts an already evaluated support field plus its
diagnostics. `denoise_2d_fmmt` exposes that only as an optional representation
input. The GUI passes the same field to the denoiser that it displays, removing
the duplicate continuum-scale calculation. Direct API calls still compute the
field internally exactly once.

## Current profile

For the 256-square three-repeat record, median stage times are:

| stage | seconds | share |
|---|---:|---:|
| ordered-front transport | 0.6250 | 51.9% |
| bootstrap chart | 0.3001 | 24.9% |
| support birth | 0.1452 | 12.1% |
| residual scale, graph, packets, posterior | 0.1341 | 11.1% |
| total | 1.2045 | 100% |

The next serious acceleration target is therefore unambiguous: the repeated
sparse multi-source Dijkstra front and its dense packet contraction. A future
native syscall should fuse limited-radius graph arrival, attenuation, mass,
and joint packet accumulation without materializing the complete batch-by-
pixel matrix. The bootstrap recurrence is the second native kernel. Neither
requires changing the estimator or inventing a denoising control.

## Verification

- 139 relevant denoiser invariants pass on the M4 Mini.
- The vector recurrence and batched histogram are bit-identical to references.
- Complement-lane support agrees within `1e-15`.
- In-place joint packets agree with split accumulation within `2e-15`.
- Front batch changes remain bounded within `5e-16`.
- The Dear PyGui continuous-support 2-D path completes end to end.

The repeatable measurement is `benchmark_2d_acceleration.py`; the copied M4
record is `2d_acceleration_m4.json`.
