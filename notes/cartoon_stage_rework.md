# Reworking the BFFT cartoon stage for large images

Files: `experiments/cartoon_stage_study.py` (measurement),
`experiments/cartoon_stage_tridiag.py` (the rework, proven and benchmarked).
Nothing in the library was modified.

## What the stage is

`src/detail/meyer_kernel.hpp::run_passes` is a Gilles-Osher / split-Bregman
TGFD alternation. Per pass it does exactly **four real 2-D transforms** — a
forward on each of the two divergence fields, an inverse on each of the two
spectra — plus two pointwise shrinkages. The linear subproblem

    (c - eta * Laplacian) u = rhs

is solved exactly in the spectrum by `symbol()`, which is the **periodic** DFT
symbol `1 / (c - eta*(lx + ly))`. The column stage is neat: because the symbol
is real, the row-rfft's Re and Im planes can be column-transformed separately
with real FFTs, so the whole pipeline stays real.

`_meyer_padded` rounds **each dimension up to the next power of two** by
symmetric reflection, because the Bruun kernel is radix-2.

## Measured, before proposing anything

**Padding is the largest single waste.** Transformed area against area
supplied: 1080p 2.02x, 1440p 2.28x, 4K 2.02x, and anything just above a power
of two 3.99x. Note that standard video resolutions are all 5-smooth
(1920 = 2^7·3·5, 1080 = 2^3·3^3·5, 2160 = 2^4·3^3·5), so a mixed-radix
transform would need no padding at all — the waste is entirely an artifact of
radix-2.

**The cost per pixel per pass gets worse with size, not better.**

| size | per pass | ns/pixel/pass |
|---|---:|---:|
| 256² | 1.65 ms | 25.25 |
| 512² | 3.57 ms | 13.63 |
| 1024² | 20.22 ms | 19.29 |
| 2048² | 165.24 ms | **39.40** |

From 512² to 2048², `log N` grows 22% but the per-pixel cost grows 189%. That
is a memory cliff, not an arithmetic one: the engine holds ~17 full-size plane
buffers plus five spectra, which at 2048² is over a gigabyte of working set.
**At large sizes this stage is bandwidth-bound, not flop-bound**, and the
right optimizations are the ones that move fewer bytes.

**There is no early-exit saving; the opposite.** `run_passes` has no
convergence test, unlike `rof_from_spec`. I expected slack. There is none —
the alternation is still moving at pass 48:

| pass | relative step | distance to the 48-pass answer |
|---|---:|---:|
| 6 | 1.07e-2 | 3.71e-2 |
| 24 | 9.69e-4 | 8.85e-3 |
| 48 | 2.43e-4 | 0 |

At the shipped `passes=24` the cartoon is still 8.9e-3 away from where it is
heading. Convergence is sublinear, as ADMM on a non-strongly-convex composite
should be. So a stopping test would save nothing, and the expensive tail makes
**per-pass cost reduction worth more, not less**.

## The rework: transform one axis, sweep the other

The operator is separable, but only the *transform* has to be. Transform along
rows only. For each row frequency `k` the remaining equation in `y` is

    -eta*u[y-1] + (c - eta*lx(k) + 2*eta)*u[y] - eta*u[y+1] = rhs[k, y]

a symmetric tridiagonal Toeplitz system. It is strictly diagonally dominant
for every `k`, because `lx(k)` lies in `[-4, 0]` so the diagonal is at least
`c + 2*eta` against an off-diagonal sum of `2*eta`. Thomas needs no pivoting.
This is the Hockney/FACR construction and it costs `O(H)` where the column FFT
costs `O(H log H)`.

Three consequences, in order of size:

1. **The swept axis stops being padded.** A tridiagonal solve does not care
   whether `H` is a power of two. Which axis to sweep is a free choice and it
   matters — sweep the *worse-padded* one. 1080p and 4K fall from 2.02x the
   image to 1.07x; just above a power of two falls from 3.99x to 2.00x.
2. **Half the transform work disappears** even with no padding at all, because
   one of the two directions stops being transformed.
3. **Neumann boundaries in that axis become free** — fold the first and last
   rows of the tridiagonal instead of wrapping. The current periodic symbol is
   exactly the wrapped finite difference that `TRANSPORT_CELL_MATH.md` lists
   as defect 7. In the swept direction it stops existing.

### Proven, not asserted

The periodic sweep reproduces the full 2-D spectral solve to **1e-15
relative** at every size and every `(c, eta)` tested, via a Sherman-Morrison
correction for the wrapped corners. The Neumann variant satisfies its own
operator to 5.4e-16.

### Measured

| image | spectral (padded) | row + sweep | measured | compiled ceiling |
|---|---:|---:|---:|---:|
| 1024² | 8.8 ms | 11.4 ms | 0.77x | 2.29x |
| 1080×1920 | 55.8 ms | 21.9 ms | **2.55x** | 5.77x |
| 1440×2560 | 99.0 ms | 55.0 ms | **1.80x** | 3.64x |
| 1025² | 51.8 ms | 21.0 ms | **2.47x** | 5.44x |
| 2048² | 50.6 ms | 41.5 ms | 1.22x | 2.81x |

The prototype's sweep is a Python-level loop over `H` rows, so it leaves most
of the win on the table. "Compiled ceiling" is the spectral time against the
transforms alone — what remains once the sweep costs what an `O(H)` streaming
pass in C should cost. The honest read is that the measured column is a floor
and the ceiling column is the target.

At an exact power of two the prototype **loses** (0.77x) because there is no
padding to recover and the Python sweep cannot outrun a tuned FFT. So the
route should be selected, not switched to: sweep when it removes padding, keep
the present path when both dimensions are already powers of two.

## Also measured, smaller

**Cascadic warm start: ~19%, and it needs an API that does not exist.** A
half-resolution solve costs 10% of the full one and lands where fine pass 7 of
24 sits — buying 7 passes for the price of 2.4. The correct coarse fidelity is
`lam * 2` per halving (the discrete ROF functional at pixel width `h` is
`sum|grad u| + (c*h/2) sum|u-f|^2`), but the measurement barely distinguishes
it: distance 0.0366 at `lam*2` against 0.0369 unchanged and 0.0395 at `lam/2`.
Worth doing eventually, but `bfft_meyer_split` always starts from zeros and
has no way to accept an initial `u`, `w`, and Bregman state.

## Recommended order

1. **Sweep the worse-padded axis** (this note's rework). Largest win, exact,
   already proven equivalent, and it fixes the wrap artifact in that axis.
   Selected per-image against the present path.
2. **Mixed-radix (3 and 5) transform lengths.** Independent of 1 and composes
   with it — together they would eliminate padding entirely on every standard
   resolution.
3. **Warm-start entry points** for `run_passes`, then cascadic multigrid.
4. Not worth doing: a convergence test in `run_passes`. Measured, and there is
   nothing to reclaim.
