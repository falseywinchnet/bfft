# FFT and C++ opportunities for the HD sigma pipeline

Date: 2026-07-26  
Machine: Apple M3, 8 logical CPUs, macOS 14.8.7, arm64  
Runtime: Python 3.13, NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0

This takes over the interrupted FFT survey.  No viewer, core, or production
file was changed.  The reproducible benchmark is
`experiments/sigma_opt/bench_fft_cpp_opportunities.py`.

## Executive result

There is a real FFT opportunity, but it is narrow:

* use it for the full-frame expected-affine-gain Gaussian triplet when the
  cell budget stays fixed and image resolution makes sigma broad;
* do **not** use it for the per-cell ridge scan, graph walk, normal assembly,
  or small fixed-sigma geometry filters;
* do **not** implement the Gaussian path by composing the current BFFT 1-D
  radix-2 API.  Its power-of-two-only geometry makes HD zero padding much
  larger than a general-length convolution, and the row/column plumbing is
  not presently a reusable 2-D convolution provider.

The first HD C++ work should instead be a reusable vision-graph plan that
builds ownership topology once, then drives the ridge scan, block normal
assembly, and rendering.  At 1080p / 2,400 cells, the current already-fused
JIT kernels spend:

| component | time |
|---|---:|
| fused ridge scan | 274.7 ms |
| co-ownership pattern build | 234.3 ms |
| normal/RHS accumulation, one field | 74.8 ms |
| two-sided render, one field | 41.7 ms |

The 1080p pattern metadata itself is 32.4 MiB.  A plan is therefore important
for both time and allocation pressure; a collection of stateless C calls
would preserve much of the waste.

## Expected-gain Gaussian: measured crossover

The allocation field requests, per input plane:

1. Gaussian mean;
2. Gaussian x derivative;
3. Gaussian y derivative.

The FFT benchmark does one input transform and broadcasts it against the
three sampled kernels.  It reproduces SciPy's exact finite kernel
(`truncate=4`) and `mode="reflect"` halo.  Maximum absolute disagreement was
between `3e-17` and `2e-15`, depending on sigma.

Representative warm best times for one float64 plane:

| image | sigma | radius | separable direct | fused FFT | FFT speedup |
|---|---:|---:|---:|---:|---:|
| 512 x 512 | 0.8 | 3 | 15.4 ms | 16.6 ms | 0.93x |
| 512 x 512 | 4 | 16 | 16.0 ms | 22.6 ms | 0.71x |
| 512 x 512 | 8 | 32 | 26.7 ms | 24.4 ms | 1.09x |
| 512 x 512 | 16 | 64 | 64.3 ms | 27.8 ms | 2.31x |
| 1280 x 720 | 4 | 16 | 62.7 ms | 78.0 ms | 0.80x |
| 1280 x 720 | 8 | 32 | 81.2 ms | 59.5 ms | 1.36x |
| 1280 x 720 | 16 | 64 | 200.6 ms | 103.6 ms | 1.94x |
| 1920 x 1080 | 4 | 16 | 129.7 ms | 296.8 ms | 0.44x |
| 1920 x 1080 | 8 | 32 | 241.9 ms | 203.3 ms | 1.19x |
| 1920 x 1080 | 16 | 64 | 424.1 ms | 165.5 ms | 2.56x |
| 1920 x 1080 | 32 | 128 | 880.7 ms | 302.8 ms | 2.91x |

The non-monotone FFT times are real: padding to FFT-friendly dimensions makes
the crossover depend on both image geometry and radius.  The selection rule
must be based on planned transform dimensions or a short cached calibration,
not sigma alone.

The viewer chooses

```
sigma = max(0.55 * sqrt(npixels / ncells), 0.8)
```

At a fixed 2,400-cell budget, the exact operating points measured separately
were:

| image | sigma | direct | FFT | speedup |
|---|---:|---:|---:|---:|
| 512 x 512 | 5.75 | 18.0 ms | 14.1 ms | 1.28x |
| 1280 x 720 | 10.78 | 108.5 ms | 62.5 ms | 1.74x |
| 1920 x 1080 | 16.17 | 399.8 ms | 239.1 ms | 1.67x |

Those are per plane.  Ordinary RGB expected gain uses three planes, so the
1080p saving is roughly 0.48 seconds per pressure update in this prototype.
“RGB + decomposition gain” uses two additional planes and raises the
potential saving to roughly 0.8 seconds.  A native implementation should
accumulate the scalar gain directly instead of materializing 3 or 5 sets of
three HD output images.

### Why current BFFT is not the FFT backend for this yet

`bfft_plan_create` accepts power-of-two 1-D lengths only.  A 1080p,
sigma=16.17 convolution has radius 65.  After the reflect halo and linear
convolution extension, it needs approximately a 1340 x 2180 transform.
Power-of-two padding expands that to 2048 x 4096, 8.39 million samples,
versus about 2.9 million samples for FFT-friendly general lengths.  This
nearly 3x grid inflation is before row/column scratch and transposes.

A worthwhile FFT implementation therefore needs one of:

* a general-length 2-D provider with cached plans;
* a platform provider such as Accelerate, with a portable equivalent;
* or a deliberately separate convolution backend.

Until then, the exact SciPy fused-FFT experiment is a better HD path than
forcing the operation through the current BFFT transform API.

### Direct convolution is still worth a native implementation

The three requested fields share work.  A dedicated separable kernel can
compute the smooth and derivative horizontal partials together, then the
three vertical outputs together.  It can also accumulate

```
mean^2 + sigma^2 * (gx^2 + gy^2)
```

without storing the fields.  This is an excellent threaded/SIMD C++ kernel
for small and medium radii.  A hybrid can choose direct or FFT from the
planned padded dimensions.  Small fixed filters elsewhere in geometry
(sigma 0.7, 1.25, 1.5, 2.0) should always use this direct path.

## Native-sideport measurements

The synthetic HD fixture used deterministic square ownership regions and one
adjacent runner per pixel.  It measures the exact optimized experiment
kernels rather than the old NumPy baselines:

| image / cells | ridge | pattern build | normal accumulate | render | pattern memory |
|---|---:|---:|---:|---:|---:|
| 512 x 512 / 2,400 | 52.6 ms | 28.3 ms | 8.7 ms | 8.3 ms | 4.7 MiB |
| 1280 x 720 / 2,400 | 66.2 ms | 197.1 ms | 23.2 ms | 18.3 ms | 14.8 MiB |
| 1920 x 1080 / 2,400 | 274.7 ms | 234.3 ms | 74.8 ms | 41.7 ms | 32.4 MiB |

The timings fluctuate with memory pressure, but the conclusion is stable:
at HD the topology construction and per-pixel passes dominate the cell-sized
factorization bookkeeping.

The exact monotone-bucket graph walk was also measured on a uniform graph:

| image / cells | adjacency pack | one walk | pushes | packed adjacency |
|---|---:|---:|---:|---:|
| 512 x 512 / 2,400 | 43.0 ms | 82.1 ms | 632,692 | 24.0 MiB |
| 1280 x 720 / 2,400 | 191.7 ms | 348.2 ms | 2,150,362 | 84.4 MiB |

One Newton step performs seven walks.  The walk is thus an HD wall-clock
priority, but not automatically a C++ speedup: Numba has already compiled the
same serial bucket algorithm to native code.  A literal translation mainly
removes dependency/startup cost.  The C++ version needs a memory/layout or
parallelism improvement to justify itself.

## Prioritized integration targets

### 1. `bfft_vision_graph_plan`: topology and stable pixel incidence

Highest-confidence target.

The plan should own:

* image shape and cell count;
* owner/runner arrays or validated views of them;
* the undirected co-ownership edge set;
* fixed 3x3-block CSR pattern and per-pixel block slots;
* stable cell-to-pixel and block-to-pixel incidence lists;
* reusable ridge, RHS, block, and render workspaces.

Build the pair set with a linear hash/radix pass rather than
`np.unique(low*n + high)` followed by sorting.  Stable incidence lists make
each cell/block independently writable, enabling deterministic parallel
ridge and normal assembly without atomics.  This one object amortizes the
234 ms pattern build and enables the next two targets.

### 2. Coupled block assembly plus two-sided render

High-confidence target.

Port Claude's finite-element assembly exactly—never recreate the design
matrix.  Keep SciPy/SuperLU as the first factorization backend: native code
should return or expose the fixed CSR values and RHS, accept solved
coefficients, then render both predictions and their blend.

At 1080p, one field's accumulate+render is 116.5 ms.  A seven-probe Newton
step over cartoon and texture executes fourteen fields, about 1.63 seconds
of these two kernels before pattern builds or factorization.  Stable
block/pixel incidence permits multithreading while retaining a fixed
summation order inside every output block.

### 3. Cell-local ridge scan

High-confidence target; explicitly **not FFT/Radon**.

The optimized scan is already the right algorithm: one pixel sweep with all
16 angles innermost, then cumulative bins.  A global Radon transform loses
the cell restriction, and performing one transform per cell is absurd for
supports of only roughly tens to hundreds of pixels.

Use the graph plan's stable cell-to-pixel lists and parallelize by cell.  Each
thread then writes one cell's 16 x 41 x 3 accumulator, with no races or
thread-private full-model sinograms.  Keep the existing first-angle,
first-bin tie behavior.

### 4. Expected-affine-gain statistic

High confidence as a core vision method; medium confidence that FFT belongs
in the first implementation.

Expose one planned operation that accepts multiple planes, plane weights,
sigma, and returns the accumulated scalar gain.  Implement a threaded fused
separable path first.  Add FFT only behind a measured planner when a
general-length provider exists.  Do not expose mean/dx/dy arrays unless a
diagnostic explicitly requests them.

### 5. Exact two-label graph walk

High wall-clock value, lower implementation confidence.

Do not claim a speedup from a line-for-line C++ port.  First fix the layout:
the current packed adjacency costs 96 bytes per pixel (eight float64 costs
plus eight int32 neighbor indices).  At 1080p that is about 190 MiB.  A
padded grid/sentinel border can derive neighbors by fixed offsets and retain
only node-major costs, cutting roughly one third of that storage.

After that, compare:

* the existing circular monotone bucket queue;
* a monotone radix heap over ordered float64 bit patterns;
* parallel processing within a bucket.  With bucket width no larger than the
  minimum edge cost, no same-bucket relaxation exists, which is the useful
  opening for exact parallelism.  The two-label updates still require a
  careful deterministic claim/merge scheme.

Keep the Numba bucket kernel until the C++ version beats it warm on actual
perturbed metric fields.

### 6. Tree accumulation and Schur candidate pricing

Defer.

Tree accumulation is already 12.6x faster at 256 px and is small relative to
HD walks/topology.  Schur pricing is cell/candidate dominated after Claude's
local-support rewrite.  Both are reasonable later additions to a graph plan,
but neither should delay the four targets above.

## Core/API and build shape

The repository already has the right public pattern: a flat C ABI, opaque
plans, C++ implementation, and ctypes wrapper.  Vision should follow it,
for example with a new `bfft_vision_plan` rather than being tied to the
DearPyGui app.  CMake must add the public header and source to both static and
shared targets, and the Python build hook must add the source too.

There is an existing packaging discrepancy to fix during integration:
`CMakeLists.txt` includes `src/meyer.cpp`, but `setup.py`'s `SOURCES` list
does not, even though `_core.py` declares the Meyer symbols.  A vision
backport should not copy that omission.

The sparse factorization should remain outside the first core slice.  The
native plan provides deterministic block CSR assembly; SciPy factors the
cell-sized matrix; the native plan renders.  That captures the HD,
pixel-proportional work without introducing a large sparse-solver dependency
into BFFT.

## HD viewer consequences

The current viewer cannot request full resolution:

* `Config.max_side` defaults to 256;
* the UI slider is capped at 512;
* `_fit_rgb` always downsizes to that cap.

Removing the cap alone will make the UI appear hung.  A full-resolution mode
needs the invariant score/decomposition cache, reusable graph/native
workspaces, and a display-sized preview separate from the full-resolution
model buffers.

Float64 HD storage is material.  One 1920 x 1080 RGB array is about
47.5 MiB; one scalar plane is 15.8 MiB.  The current model holds many such
arrays, while the packed graph can add roughly 190 MiB.  The native plan
should reuse workspaces and return only requested diagnostics.  Ownership and
indices should remain int32 where their range permits it.  Changing solver
fields to float32 is a separate numerical decision and should not be bundled
into the performance port.

## Negatives worth preserving

* No global Radon/FFT ridge scan.
* No FFT for the many sigma <= 2 geometry filters.
* No forced 2-D convolution through power-of-two-only BFFT plans.
* No claim that C++ automatically beats Numba for the bucket walk.
* No materialized design matrix.
* No stateless native API that rebuilds ownership topology for cartoon and
  texture separately.
* No HD diagnostic arrays by default; calculate scalar summaries in the
  backend.

