# Existing engine inventory

## Optimized posterizer

The current implementation is `../posterizer/src/posterizer` on branch
`codex/posterizer`:

- `core.py:_perceptual_importance` computes spatial OKLab gradient/contrast
  saliency and sublinear occupied-color population mass.
- `oklch.py:bifurcate_palette` constructs the deterministic weighted binary
  palette tree from exact one-dimensional split proposals plus two-means
  refinement.
- `core.py:_assign_oklch` performs matrix-form cylindrical OKLCH assignment.
- `core.py:_cleanup_components` absorbs small spatial islands using the
  equal-neighbor component graph.
- `core.py:posterize_array` owns the complete raster pipeline and diagnostics.

The defining commits are `a25e385` (initial perceptual method), `701e00f`
(foreground/detail allocation), and `98ae5d5` (optimized implementation).

## Optimal/error-bounded SVG maker

The first complete vectorizer is `../svg_converter/src/tlvector/core.py`:

- `_seed_palette`, `_regularized_assign`, and `_lift_labels` establish coarse
  topology and lift it to the source lattice.
- `_nested_residual_children` and `_adaptive_quality_children` add
  parent-locked residual cells, optionally under an exact RGBA MSE target.
- `_boundary_loops` compiles exact oriented boundaries of raster-square unions.
- `_simplify_closed`, `_relax_subpixel`, and `_loop_path` simplify and emit
  accepted linear/quadratic contour events.
- `_svg_document` and `vectorize_array` compile the final layered SVG.

This engine entered at `157cf3f` and gained adaptive error bounds at
`b816221`/`e417296`.

## Compact SVG V2

`../svg_converter_v2/src/tlvector_v2` is a second static SVG engine:

- `merge.py:merge_error_per_byte` removes components by measured RGBA-error
  cost per estimated serialized byte.
- `lattice.py:compact_lattice_loop` emits compact exact H/V/L lattice paths;
  `deterministic_svgz` adds stable SVGZ output.
- `affine.py:fit_rank1_affine_cells` and `affine_gradient_svg` paint native
  segmenter cells with rank-one SVG gradients.

It is valuable for asynchronous export, but its measured 11.21-second city
conversion confirms that it cannot be placed directly on a 33.3 ms hot path.

## SVG oscilloscope animation engine

The missing temporal implementation is the self-contained browser app now
preserved verbatim at `legacy/svg_oscilloscope_renderer_v5.html` (SHA-256
`dcd0ed37de32ba02defc52b6a41547d20ffaa0619bf533ec6c328ee3f88206f2`). It was
recovered from the user's Downloads folder after task history identified the
task titled **SVG Oscilloscope Renderer**
(`6a8159e1-cea8-83ea-b3ed-4ff5cd03d2db`). Its defining code is:

- `buildLinearPathRuns`: exact `M/H/V/L/Z` parsing for lattice SVGs, with
  browser geometry sampling as the fallback for curved paths;
- independent scheduling of every disconnected `M…Z` subpath, specifically to
  avoid false pen-up bridges inside compound color paths;
- `reroll`/`startNextLine`: a shuffled complete visit cycle, random direction,
  random starting segment, and a 45–180 screen-pixel phrase (with occasional
  16–54 pixel flicks) rather than a complete path redraw;
- `segmentCenterDash`/`advanceTracer`: a short dash traveling through each
  visited segment with a hard 26,000-mark per-frame rendering budget;
- `fadeHistory`: refresh-rate-independent exponential decay controlled by a
  half-life, defaulting to 3.6 seconds;
- `nearestWordColor`/`drawMark`: Lab-nearest Office-like colors, watercolor
  density passes, and a faint green phosphor accent;
- `syncTurbo`: logarithmic 1×–100× turbo, with reduced glow/pigment work above
  18× and 55×.

The city fixture discussed in that task had 217 compound color paths but
34,479 independent subpaths. The v5 correction removed a stale 70,000-segment
loader cutoff and scheduled those independent strokes rather than DOM path
elements.

`realtime_vector_fx` is the native merge target for that engine: stable lattice
edge IDs replace browser DOM geometry, maximal trace runs preserve pen-up
separation, and the persistent visit scheduler must retain the complete-cycle,
random-local-slice, and history-decay semantics above.
