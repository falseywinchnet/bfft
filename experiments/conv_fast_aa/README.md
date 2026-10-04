# CONV / CONVSTAR fast antialiasing investigation

This is a first-principles research experiment, not a promoted CONV replacement.
The objective remains FXAA-class complete cost with 4× MSAA-class quality.
Two distinct input contracts are tested; renderer geometry is never supplied to
an image-only method. See `FINDINGS.md` for retained M4 measurements and failures.

## What comes from CONV

The current CONV* operator consists of the compact source jet, ordered sign
ledger, signed-fibre admission, and integrated Bernstein synthesis. The current
jet is already a finite FIR bank. Its cost is not just evaluation of a quintic:
source construction and admission must be counted. The joint warp construction
also retains factor order, shared controls, and global completion. We do not
replace any of those constraints or alter the existing implementation.

The useful structural principle is **construct a faithful representation, then
apply the pixel functional directly**. A pixel is a measure, not a point. A
cheaper interpolant does not recover geometric coverage that was never observed.
The existing `conv_warp/line_coverage/FINDINGS.md` documents the opposite source
contract too: a raster already containing coverage must not be treated as nodal
point data and averaged a second time.

The image-only prototypes below borrow the existing first jet/tensor, but are
new experimental filters. They are not equivalent compilations of CONV*.
`core.conv_box` is a separate reference: exact half-cell quadrature of the
original admitted one-dimensional CONV factors, averaged across the two orders.
It is not the more expensive joint 2-D warp atlas.

## Image-only tangent contraction

Use the CONV fourth-order first jet on the existing radius-two cross:

    gx = [8(E-W) - (E2-W2)] / 12
    gy = [8(S-N) - (S2-N2)] / 12.

Freeze a tangent at the source pixel, scale it so `max(|tx|,|ty|)=1`, and
integrate the Q1 reconstruction along `p+s*t`, `s in [-1,1]`. No additional
classification, iterative fitting, search, or adaptive subdivision is used.
The segment visits only the two opposite bilinear cells incident to the pixel.
Writing `a=|tx|`, `b=|ty|`, with D1/D2 the two tangent-side diagonals, gives

    output = C + wx(E+W-2C) + wy(N+S-2C) + wd(D1+D2-2C)
    wx = a*m1/2 - a*b*m2/2
    wy = b*m1/2 - a*b*m2/2
    wd = a*b*m2/2.

For a uniform line measure, `(m1,m2)=(E|s|,Es²)=(1/2,1/3)`.
For the normalized quintic-step derivative `rho(s)=15/16*(1-s²)²`, they are
`(5/16,1/7)`. These are exact polynomial contractions, not quadrature guesses.
The resulting stencil is nonnegative and sums to one, so it preserves constants
and range, and fixes affine images away from the clamped boundary.

The RGB variant uses the existing tensor
`Gxx=sum(gx²), Gxy=sum(gx*gy), Gyy=sum(gy²)`. Its minor eigenvector supplies
the tangent. The eigenvalue gap divided by the trace weights the contraction
against the identity. Isotropic tensors have no preferred tangent. This fixes
equal-luminance color blindness at a measurable GPU cost.

Direction is ill-conditioned near a zero first jet. The final implementation
uses paired differences and a float32-relative zero rule: derivative magnitude
must exceed eight float32 epsilons times the already-read local variation.
The RGB rule uses squared magnitude. This is a numerical conditioning convention,
not a fitted perceptual threshold. The reference must receive the same float32
source and luma values as Metal. Failed v1/v2 comparison receipts are retained.

The filter does not conserve arbitrary global image mass: its weights vary with
position. It does not preserve all already-antialiased input, and its invariants
do not imply good subpixel reconstruction. Those failures are measured.

## A concrete information bound

Let a white half-plane begin at x=3.1 or x=3.4. Both produce the same entire
integer-lattice point raster: pixel 3 is black and pixel 4 is white. Nevertheless,
pixel 3 has true coverage 0.4 or 0.1. Any deterministic image-only operator gives
one answer to both inputs, so its worst absolute coverage error is at least 0.15.
The rotated four-sample pattern used here returns 0.5 or 0, with error 0.1 in
each case. Thus universal 4×-coverage accuracy cannot follow from this point
raster, regardless of solver, polynomial order, or exact integration downstream.
This does not rule out an excellent average-case postprocess method.

## Retained boundary current

The second experiment starts from the renderer's triangle vertices, which are
already available before rasterization. It does not infer boundaries from colors.
The prototype input contract is nondegenerate triangles with positive signed
area in screen coordinates; a general integration must normalize winding.
For a triangle T and pixel square Q, the exact coverage is

    A(T intersect Q) = (1/2) integral_boundary(T intersect Q) (x dy - y dx).

Its boundary consists of pieces of at most three triangle edges and four square
edges. A segment's endpoints are clipped with finite min/max arithmetic. Its
contribution is half their cross product. There is no polygon sorting, iterative
optimization, subdivision, or image-analysis pass. Coincident edges are owned by
the square to prevent double counting. The independent CPU oracle instead
constructs a clipped polygon using Sutherland-Hodgman and measures its area.

Most covered pixels are either wholly inside or cut by only one edge. The latter
use the exact box-projected half-plane CDF. If `a=max(|nx|,|ny|)`,
`b=min(|nx|,|ny|)`, `z=|d|`, the lesser-side coverage is

    1/2-z/a                         when z <= (a-b)/2
    max((a+b)/2-z,0)^2 / (2*a*b)   otherwise.

The sign of d selects that tail or its complement; b=0 uses the linear limit.
Pixels cut by multiple edges use the complete seven-segment formula, avoiding
the incorrect multiplication of independent edge coverages at corners.

Unchanged primitive edge quantities are hoisted to the vertex stage. A
conservative draw support contains every pixel that can intersect the original
triangle. For twice-area D and edge pixel support radii r_i, dilation about the
centroid by `1+3*max(r_i)/D` moves every supporting edge out by at least r_i.
The vertex shader picks this support when its area is smaller than the bounding
rectangle; otherwise it keeps the rectangle. Both supports evaluate the same
original triangle. This is fixed algebra on supplied geometry, not feature
analysis or a change in the reconstructed source.

## Coverage is additive; opacity composition is not

If two adjacent triangles partition one surface within a pixel, their coverages
satisfy `a+b=1`. Alpha-over gives `a+b-a*b`, leaving a seam; for a=b=1/2 it
produces 3/4. Adding their color-weighted areas gives the correct answer.
`boundary_sum` tests this on a triangulated surface using additive accumulation.

That rule does **not** solve visibility. Two coincident opaque triangles represent
one visible footprint, not twice its coverage. Both naive alpha-over and additive
area accumulation fail that explicit counterexample. General depth overlap,
varying shading, textured interiors, transparency, and complete moving-scene
validation remain required before this could replace an engine's antialiasing.
No hidden visibility oracle or per-pixel fragment solver is added.

## Baselines, provenance, and scope

- NVIDIA FXAA 3.11 PC quality preset 12, default subpixel/edge thresholds, actual
  shader source adapted through Metal type/texture macros. It is not a small
  homemade filter. The preserved BSD notice is in `vendor/Fxaa3_11.h`.
- Source: https://github.com/GameTechDev/CMAA2/blob/master/Projects/CMAA2/FXAA/Fxaa3_11.h
  (the optional separate-luma/gather modifications are disabled).
- FXAA context: https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf
- SMAA context: https://www.iryoku.com/smaa/downloads/SMAA-Enhanced-Subpixel-Morphological-Antialiasing.pdf
- Prior analytical/distance coverage work:
  https://www.iryoku.com/aacourse/downloads/Filtering-Approaches-for-Real-Time-Anti-Aliasing.pdf
  and https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-22-fast-prefiltered-lines

Directional postfilters and analytic coverage are established ideas. Neither the
half-plane CDF nor Green's theorem is claimed as a new invention. The concrete
result here is a tested CONV-motivated contraction/representation investigation
and a fixed-work M4 implementation, not a claim of literature novelty.

The 456-case image battery measures numerical coverage-space error. FXAA is
normally used in perceptual/display space; this scalar diagnostic is not a
complete color-managed game comparison. Four geometric samples in that battery
are an ideal coverage control, not hardware timing; on the sine/checker shader
tests they act as supersampling, which ordinary MSAA need not provide. The
separate primitive experiment uses actual Metal 4× MSAA, including resolve.
Its M4 sample positions match the rotated pattern in the image battery.

GPU timing uses warmed pipelines, 21 rotating-order observations and eight
dependent draws/dispatches per command buffer. It includes image reads/writes,
primitive draws, clears, and the corresponding resolve/postprocess. It excludes
shader compilation, CPU geometry generation/upload, and display presentation.
The final image throughput uses RGBA8Unorm. v1 RGBA32Float timing was strongly
bandwidth dominated and is not substituted for the final result.

## Reproduction on the selected M4 host

From the repository root:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh experiments/conv_fast_aa/build.sh
/Users/ultimussecundai/.local/bin/m4build -- python3 -m unittest \
  experiments.conv_fast_aa.test_core experiments.conv_fast_aa.test_geometry -v
/Users/ultimussecundai/.local/bin/m4build -- python3 -m experiments.conv_fast_aa.study \
  generate --out /tmp/conv_fast_aa
```

Before another sync, execute the built runner in the printed mirror:

```sh
ssh "$(/Users/ultimussecundai/.local/bin/m4host)" \
  'cd /Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539 && \
   /tmp/conv_fast_aa_build/gpu experiments/conv_fast_aa/aa.metal \
     /tmp/conv_fast_aa/input.bin /tmp/conv_fast_aa/gpu benchmark && \
   python3 -m experiments.conv_fast_aa.study analyze --out /tmp/conv_fast_aa'
```

For actual hardware rendering and coverage checks:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  'mkdir -p /tmp/conv_fast_aa_geometry && \
   /tmp/conv_fast_aa_build/geometry_gpu experiments/conv_fast_aa /tmp/conv_fast_aa_geometry && \
   python3 -m experiments.conv_fast_aa.geometry_reference analyze --out /tmp/conv_fast_aa_geometry'
```

Copy each `/tmp` output immediately into `output/support_geometry/conv_fast_aa/`
using the host selected by `m4host`. The M4 standalone Metal SDK component is
absent; the executables compile shader source through the installed Metal runtime.
No remote software installation is necessary. Local report rendering uses
`.venv-jpeg/bin/python -m experiments.conv_fast_aa.study plot --out <saved-output>`
and the corresponding `geometry_reference plot` entry point.

The second-stage Metal optimization, bounded exact visibility/composition, and
raw-current thin-line augmentation investigation is in [FOLLOWUP.md](FOLLOWUP.md).
