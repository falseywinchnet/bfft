# Geometry-fixed paired one-sided cells

## Where the BFFT phase coordinate applies

The current residual ridge searches 16 arbitrary directions and 161 offsets.
Its histogram storage is:

```text
cells * directions * offsets * 3 * sizeof(double)
```

At 16,806 cells that is about 1.04 GB. Replacing `sin` and `cos` alone would
not help: the direction table is already precomputed and the expensive object
is the arbitrary basis family.

Each contour cell already has a symmetric boundary tensor. The implementation
integrates its traceless components over every pixel owned by the literal
cell; sampling only at the germ is wrong because germs normally lie in flat
interiors where the boundary tensor is zero. Its normalized doubled-angle
vector is

```text
u = (Bxx - Byy) / hypot(Bxx - Byy, 2 Bxy) = cos(2 theta)
v = 2 Bxy          / hypot(Bxx - Byy, 2 Bxy) = sin(2 theta)
```

The normal follows by the stable covering-map half angle:

```text
nx = sqrt((1 + u) / 2)
ny = sign(v) sqrt((1 - u) / 2)
```

No `atan2`, `sin`, or `cos` is required. When a general vector rather than a
tensor supplies the direction, cache BFFT's `(octant, slope leaf, h)` and use
the table center plus projective half-angle residual.

## Proposed cell coordinate

For cell `i`:

```text
s_i(x) = n_i dot (x - c_i)
```

Accumulate the residual once into a one-dimensional histogram over `s_i`.
Prefix sums give the best finite split offset. The two side weights should use
a narrow fractional pixel footprint, not a hard sign:

```text
r = clip(kappa * (s_i - offset_i), -1, 1)
w_minus = (1 - r) / 2
w_plus  = (1 + r) / 2
```

Changing the sign of `n_i` only swaps the two coefficients, so the line-field
orientation ambiguity disappears.

Cell-ordered execution reuses one offset histogram, reducing workspace from
`O(cells * directions * offsets)` to `O(offsets)`. It also removes the
16-direction pixel loop.

## Implemented path

`bfft_vision_scan_paired_offsets` counting-sorts pixels by owner once, then
streams each cell through one `offsets x 3` histogram. The canonical
`fit_regions` path uses it whenever the frozen BFFT boundary tensor is
available. Callers without geometry retain the old free-angle path as a
compatibility/reference control.

On the 475 x 475, 605-cell Pikachu partition, the isolated native measurement
takes a median 1.77 ms, versus 8.49 ms for 16 directions at the same 161
offsets (4.8x). Histogram workspace falls from 35.7 MiB to 3.8 KiB. At 16,806
cells the former tensor would occupy about 0.97 GiB (1.04 decimal GB), while
the paired histogram remains 3.8 KiB plus the pixel ordering.

## Prototype evidence

On default full-size Pikachu:

| Readout | Objective | Critical-row RGB MSE |
|---|---:|---:|
| former free ridge, 41 offsets | 0.002363 | 0.09952 |
| geometry-fixed fractional pair | 0.001706 | 0.03268 |
| free ridge, 161 offsets | 0.001551 | 0.03283 |

The paired coordinate recovers nearly all of the critical-edge improvement.
Its remaining objective gap comes from cells whose best residual axis differs
from the frozen boundary normal. A possible bounded fallback is a small
slope-leaf neighborhood: test the cached leaf and its immediate neighbors,
rather than sixteen unrelated global directions. The current canonical path
deliberately uses only the intrinsic direction. The free-angle compatibility
control caps its accumulator at 256 MiB and retains an odd offset count.
