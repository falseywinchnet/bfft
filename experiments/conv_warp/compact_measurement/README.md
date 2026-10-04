# Streaming joint CONV measurement

This implementation evaluates rectangular destination-pixel averages of the
existing joint CONV construction without allocating its full-image control
lattice. It retains both ordered factors, their source-dependent blend, the
shared range projection, entity capacities and the common completion parameter.
The executable is a freestanding WebAssembly SIMD module using binary64
arithmetic and the reference's binary32 source and proposal storage points.

## Representation

Each scalar channel uses two first-factor sample grids. The second-factor
admission at each source interval is recorded in ten bits: five active-face
bits and five current-sign bits, stored in a uint16. The selected mass-fibre
projection is recovered from its raw currents and those bits using the same
finite formula and arithmetic order as the reference. No new face search,
classifier, adaptive subdivision or iterative solve is introduced during replay.

Six-by-six patches are reconstructed from those grids and faces. The inverse
collocation operations preserve the reference's intermediate binary32 rounding;
independent lane pairs use SIMD with fused multiply-add disabled. Shared controls
receive the same incident source-range intersections. Original vertex values are
restored exactly as in the reference.

The geometry plan integrates each quintic Bernstein basis over each destination
basin's intersection with a source cell. The endpoint convention and single-pixel
whole-axis basin are unchanged. Reverse sparse banks associate each shared
control with its destination contributions. A shared control is emitted once,
using the first incident cell in row-major order.

## Shared admission in rolling rows

Let C be a bounded proposal, B its bilinear baseline, and alpha_g the simultaneous
capacity of shared entity g. Define

    Q_e = B_e + alpha_g (C_e - B_e),
    D_e = C_e - Q_e.

The completed control is P_e = Q_e + tau D_e, where tau is the channel's common
completion parameter. Vertices have D_e = 0. For every destination-pixel
functional L,

    L(P) = L(Q) + tau L(D).

The two terms are accumulated before tau is final. Once all completion
constraints have contributed their minimum, one destination-sized combination
produces the image. If the original candidate passes the channel-wide current
check, tau is set to one, retaining the reference's unchanged-candidate branch.

Each constraint row belongs to a source cell and only touches that cell's
vertices, edges and interior entity. An interior capacity is therefore final
after its own cell; a vertical-edge capacity after its incident cells in the
same row; and a horizontal-edge capacity after its two incident rows. After row
r has contributed all its capacities, every entity needed by row r-1 is final.
The compiler then evaluates row r-1's completion constraints and accumulates
L(Q), L(D). The final row is flushed explicitly.

Two patch rows suffice. Horizontal capacities use three rolling boundary rows;
vertical and interior capacities use two rows. Cone metadata uses two rows.
The wraparound schedules overwrite an entity only after its final consumer.
The channel-wide completion minimum remains global, but it requires one scalar
and no retained completed surface.

Constraint topology is fixed once for the 36 local positions. The compiler
reuses its indices and integer derivative coefficients. The existing angular
cone order uses stable merge passes instead of repeated insertion, with the same
comparison, tolerances and early full-plane test. Source analysis is unchanged.

## Storage and precision

For N source pixels and M output pixels, the main arena is approximately
132 N + 40 M bytes plus line, sparse-bank and rolling-row storage. The first-factor
grids account for 80 N bytes. Channels reuse the same work buffers. No allocation
has the size of the two-dimensional fifth-step control lattice.

The default measurement fuses completion with integration. It omits the final
binary32 rounding of each completed control. The underlying real-arithmetic
functional is unchanged. Positive normalized area weights do not amplify that
omitted control-rounding error; binary64 reassociation adds its own rounding.
Near-zero alpha can amplify an error during unpremultiplication, so encoded
channels are checked independently rather than inferred from the float error.

`exactStorage: true` is an independent validation mode. It reconstructs transient
completed patches with the final binary32 rounding and measures them directly.
It retains full capacity/cone arrays for replay but still allocates no full-image
control atlas. `compact_probe_patch` exposes its transient controls for direct
comparison. The fused mode does not perform or report a final stored-control
margin scan because it does not store those controls.

## Validation

`validation.mjs` checks 120 seeded combinations of image dimensions, destination
sizes, flat colours, random colours, steps, checker patterns, ramps, waves,
small quantized variations, isolated features and transparency. Source dimensions
range from 5 through 28; destination axes include one pixel and unreduced sizes.

- 3,000,816 transient controls match the frozen reference exactly.
- Maximum storage-mode pixel-average difference: 1.91e-14.
- Maximum fused pixel-average difference: 6.07e-9.
- No encoded byte differs in either mode across the validation set.

The larger 513 × 321 browser comparison also passes with no encoded byte
differences and a maximum float difference of 8.63e-9. These observations are
finite validation results, not a universal bit-identity assertion for fused
floating-point arithmetic.

The frozen reference is the website's existing native joint source compiler and
WASM binary. `reference-provenance.json` records their source hashes.
`instrumented_source.h` retains that construction with admitted-face recording,
optional omission of unused second-factor samples, a fused initial margin count,
and the stable angular merge implementation. Its unused atlas entry points are
never called by `CompactMeasurement`.

## Timing

M4 Mini, Node v25.2.1, warmed modules and arenas, alternating implementation order.
Seven measurements per case, except three for 1025 × 641. Module fetch,
compilation and image decoding are excluded. The old timing ends when the atlas
is prepared; the compact timing includes its completed pixel-area measurement.

| Source → output | Image | Reference preparation | Compact preparation + measurement | Reference / compact WASM memory |
| --- | --- | ---: | ---: | ---: |
| 257 × 171 → 193 × 129 | waves | 542.0 ms | 409.3 ms | 25.89 / 7.21 MB |
| 513 × 321 → 385 × 241 | waves | 2052.0 ms | 1558.1 ms | 96.53 / 26.21 MB |
| 513 × 321 → 129 × 81 | noise | 672.0 ms | 827.4 ms | 96.53 / 22.87 MB |
| 1025 × 641 → 257 × 161 | waves | 8341.4 ms | 6209.5 ms | 384.37 / 89.72 MB |

The smooth-pattern cases complete about 24–26% sooner while using about one
quarter of the reference memory. The noise case is 23% slower: its existing
reference needs little joint-current work, so replay overhead is more visible.
No source-dependent backend selector is introduced.

The initial local Chromium browser's 513 × 321 run took 1829.9 ms through canvas drawing,
versus 2385.5 ms for reference preparation. Its compact arena was 24.94 MiB;
allocated WASM memory was 25 MiB, versus 92.0625 MiB for the reference.
`browser-initial-results.json` retains that observation; `browser-results.json`
contains the final browser rerun. Browser
reference timing also includes loading its module. CPU timing does not establish
a GPU preparation speed or real-time video throughput.

## Integration scope

The browser demonstrator is `research/conv-webgpu/compact.html` in the website
checkout. It accepts an image and output dimensions and offers an explicit
reference comparison. The authoritative runtime is copied with
`python3 tools/sync_conv_compact.py` from that checkout.

This implementation is directed at one requested rectangular output. Changing
output dimensions currently rebuilds its compact state. It does not replace the
public perspective engine's cached arbitrary-footprint query path. Retaining a
compact reusable source and moving the remaining source/admission passes to GPU
execution are subsequent work; the current CPU preparation is not near-instant.

## Reproduction

The small freestanding compile takes less than a second with the installed SDK:

```sh
sh experiments/conv_warp/compact_measurement/build.sh
```

Run the full numerical work on the Mini from this authoritative checkout:

```sh
/Users/ultimussecundai/.local/bin/m4build -- /opt/homebrew/bin/node experiments/conv_warp/compact_measurement/validation.mjs
/Users/ultimussecundai/.local/bin/m4build -- /opt/homebrew/bin/node experiments/conv_warp/compact_measurement/timing.mjs
```

Copy `validation-results.json` and `timing-results.json` from the printed mirror
back before the next synchronization. The earlier `results-*.json` and
`timing-results-before-simd.json` retain intermediate performance observations;
they do not describe the final binary. Final results and runtime hashes are
recorded in `receipt.json`.
