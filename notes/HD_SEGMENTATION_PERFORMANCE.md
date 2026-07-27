# HD segmentation performance audit

This note describes the canonical
`viewer/segmenting_veroni_viewer.py` path. Measurements below are from an
Apple-silicon Mac at 1280×720, 180 initial cells, 24 BFFT passes. They are
phase measurements, not promises about another machine.

## What grew at HD

Let `P` be working pixels, `S` cells, `B` the split batch, `K` BFFT passes,
and `w` the local basis width (normally 3).

| Phase | Current cost | What matters |
| --- | --- | --- |
| BFFT cartoon/texture geometry | `O(KP)` | Linear in pixels and passes. At 24 passes this is about 2.2–2.6 s at 720p. |
| Structure tensors and graph weights | `O(P)` | Several image-wide filters and eight edge planes. |
| Initial blue-noise coverage | `O(SC)`, `C=max(2048,16S)` | The former exact farthest scan was `O(SP)`. The deterministic R2 candidate pool makes initialization independent of HD pixel count. |
| Two-best geodesic ownership | `O(E + R/δ)` | Exact Dial walk, where `E≈8P`, `δ` is the minimum edge cost, and `R` is the active distance range. The binary-heap reference is `O(P log P)`. |
| Local affine fit and render | `O(Pw²)` / `O(Pw)` | Linear in pixels for fixed `w=3`. |
| Expected affine allocation map | `O(P)` | It is evaluated on the field's physical scale. The former full-grid FIR form was `O(P × spacing)` at coarse cell counts. |
| Subdivision pricing | `O(P+S log S)` | A single reduction now finds every cell's best pixel. The former loop could rescan all `P` pixels once per requested child (`O(BP)`). |
| Coupled normal assembly | `O(Pw²)` | Direct finite-element-style scatter into the measured co-ownership graph. |
| Sparse factorization | topology dependent; dense worst case `O((wS)³)` | The only potentially cubic piece. At current cell budgets it remains well below the HD pixel walk; it becomes important as cell count, not resolution, grows. |

There is no all-pairs cell calculation in the normal HD round. The apparent
“bloom” came from repeated full-image work and generic heap bookkeeping.

## Changes made

- Replaced Numba's typed-list heap first with a packed monomorphic heap, then
  with an exact monotone bucket queue. Since every relaxation advances by at
  least the minimum graph-edge cost, entries in one bucket need no internal
  comparison. Queue storage is recycled after each pop. Owner, runner, and
  distances match the binary heap exactly; repeated 720p assignment fell from
  about 2.59 s to 0.31–0.32 s.
- Replaced full-image farthest-point initialization with deterministic
  low-discrepancy candidates. On the 128px camera control, mean and 95th
  percentile coverage improved slightly; the worst uncovered distance changed
  by less than one pixel.
- Changed subdivision from repeated `owner == cell` image scans to one linear
  per-cell argmax reduction.
- Store the eight-plane HD edge graph as float32 while retaining float64 path
  sums. This halves that persistent allocation; tested ownership and runner-up
  labels were identical, with path changes around `1e-6`.
- Removed duplicate edge tables, unused coordinate grids, eager legacy
  full-colour splits, and eager residual-memory construction.
- Cartoon/texture MSE is now explicit or objective-driven. Previously every
  render decomposed the reconstruction again even when the weight was zero.
  At 24 passes this silently added roughly 2.5 s to each render.
- Expected affine gain now evaluates its wide blur on a correspondingly coarse
  support grid, with derivatives converted back to native-pixel units. At
  512px its field correlates 0.9997 with the former full-grid result, overlaps
  97% of the top percentile, and chooses the same peak to within one pixel.
- The native vision wrapper no longer converts cell labels int32 → int64 →
  int32 before each C++ pass.
- Replaced the OKLab 3×3 colour transform's batched tiny BLAS call with a
  direct fixed contraction. Accelerate had routed the former `(H,W,3) @
  (3,3)` expression through a fragile tiny-matrix batch path.

The same 720p initialization moved from roughly 10–13 s and around 1 GiB
resident memory to about 3.3–3.7 s and 0.6 GiB peak in the measured control.
A warmed default assign/fit/render pass is about 1.1 s. Results vary with
image topology because the exact graph walk processes a data-dependent heap.

The **Composite** strong-round acceptance remains intentionally expensive:
each nonlinear proposal receives a fresh cartoon/texture decomposition. The
viewer exposes **RGB only** acceptance for interactive HD tuning; that keeps
the exact full-resolution reconstruction and graph solve but skips the
additional decomposition score. Use the explicit MSE button to audit a state,
or return to Composite for final research measurements.

## SIMD/compiler check

No manual intrinsics were added. Release builds use `-O3`; the Apple clang
vectorization report confirms that both inner normal-assembly block loops are
vectorized (NEON width 2 for double precision). The three-element affine
render dot product is deliberately left scalar: clang's cost model reports
that vectorizing a runtime-width loop of length three is not beneficial.
Validation and Dijkstra loops have dependencies and irregular gathers, so a
SIMD directive would be dishonest and usually slower.

Use this command to repeat the compiler report:

```sh
/usr/bin/c++ -std=c++17 -O3 -DNDEBUG -Iinclude \
  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
  -c src/vision.cpp -o /tmp/bfft_vision_vec.o
```

## Repeat the phase trace

```sh
python viewer/profile_segmenting_veroni.py --gallery astronaut
python viewer/profile_segmenting_veroni.py /path/to/photo.png --full-resolution
python viewer/profile_segmenting_veroni.py /path/to/photo.png \
  --full-resolution --measure-decomposition
```

The last flag makes the expensive single-stage-to-single-stage cartoon and
texture comparison visible instead of charging every ordinary render for it.
