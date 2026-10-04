# Entropy-guided decorrelation stretch for OBS

The filter **Entropy-Guided Decorrelation Stretch** (`entropy_decorrelation_stretch`)
is part of the existing `realtime-vector-fx` OBS module. It reuses the GPU filter
lifecycle, target rendering, staging-surface readback, build, and packaging
framework in `obs/gpu-filter.cpp`. The mathematical implementation is independent
of OBS in `src/entropy_stretch.cpp`; the integration is `obs/entropy-filter.cpp`.

This is a live false-color enhancement. It expands subtle RGB differences and
allocates more of the available output color range to colors associated with
locally high entropy. It does not reconstruct missing colors or identify objects.

## The original decorrelation-stretch algorithm

For color vectors x, compute mean μ and covariance C. Diagonalize

    C = V diag(λ₁, λ₂, λ₃) Vᵀ.

Center the colors, rotate to principal-component coordinates, scale each
component by its inverse standard deviation, and rotate back:

    y = μ_target + σ_target V diag(1 / sqrt(λᵢ)) Vᵀ (x − μ).

Without regularization or clipping, Cov(y) = σ_target² I. Highly correlated RGB
colors form a thin cloud: the low-variance directions are the subtle color
distinctions. Expanding those directions makes these distinctions visible.
Returning to the RGB axes avoids displaying an arbitrarily signed or reordered
set of principal components. A separate contrast stretch can use more of the
available output range. Regularization is essential for flat or rank-deficient
video, where unrestricted inverse variance can amplify tiny noise enormously.

Sources:

- [MathWorks explanation](https://www.mathworks.com/help/images/enhance-color-separation-using-decorrelation-stretching.html)
- [MathWorks decorrstretch reference](https://www.mathworks.com/help/images/ref/decorrstretch.html)
- [NASA algorithm theoretical basis document record](https://ntrs.nasa.gov/citations/20060034467)
- [Jon Harman's DStretch algorithm description](https://www.dstretch.com/AlgorithmDescription.html)
- [NASA account of the rock-art application](https://www.nasa.gov/technology/tech-transfer-spinoffs/nasa-technique-for-manipulating-satellite-photos-now-reveals-ancient-images/)

The linked Gizmodo article could not be fetched by the research browser; the
NASA and DStretch primary sources supply the underlying technique. The entropy
allocation and temporal-profile rules below are this implementation's additions,
not claims about the original DStretch algorithm.

## The implemented live algorithm

### 1. Sparse point sampling

Capture the unmodified filter target once per OBS video frame, on the GPU.
At the analysis cadence, render a **128 × 72** point-sample lattice for a 16:9
source. Each lattice point samples a native source pixel, without averaging a
thumbnail first. Lattice height follows aspect ratio and is capped at 256;
width is adjustable from 32 to 256 and never exceeds source width.

The four phase offsets are (¼,¼), (¾,¼), (¼,¾), and (¾,¾) within each lattice
cell. This reduces persistent alignment bias; it does not guarantee immunity
to aliasing. Read back the small RGBA8 lattice on a subsequent frame. Default
analysis frequency is 10 Hz. At 1080p, 9,216 samples are about 0.44% of the image;
a sample update transfers 36 KiB back to the CPU. There is no full-frame CPU
readback in the filter.

### 2. Estimate local entropy and associate it with color

Partition the lattice into 8 × 8 blocks. Quantize each RGB component to 32 levels
and count the joint RGB symbols in a block. For n valid samples, compute

    H = −Σ p(k) log₂ p(k) / log₂(n).
    h = H × variance_RGB / (variance_RGB + noise_floor²).

The normalized Shannon entropy measures local color diversity at this sampled
scale. The variance factor suppresses tiny fluctuations near the configured
noise floor. Transparent samples below alpha 0.5 are omitted; other samples are
unpremultiplied before analysis. Boundary blocks use their actual sample count.

Accumulate population n_b and entropy sum in an 8 × 8 × 8 RGB-family histogram.
A family's entropy estimate is

    e_b = Σ h / (n_b + 4).

This is conditional mean entropy with a four-sample zero-entropy prior. It is
an explicit association between a color family and local entropy, rather than
using global color rarity as a substitute for entropy.

### 3. Regularized decorrelation

Compute the RGB mean and covariance with sample weights 0.05 + h². Form

    σ = sqrt(trace(C) / 3)
    gᵢ = clamp(σ / sqrt(max(λᵢ, noise_floor²)), 0.25, max_gain)
    M = (1 − decorrelation) I + decorrelation V diag(gᵢ) Vᵀ
    z = μ + M (x − μ).

A fixed maximum of twelve 3 × 3 Jacobi sweeps computes the eigendecomposition.
There is no per-pixel eigendecomposition or full-frame optimization. The symmetric
matrix reconstruction is invariant to eigenvector sign and ordering; rotations
inside exactly repeated eigenspaces do not change the transform.

### 4. Allocate output range according to entropy

Accumulate 64-bin histograms of each transformed RGB channel using sample mass

    w_sample = e_b² / n_b^0.75.

Thus a color family's total mass is e_b² n_b^0.25. Large flat backgrounds cannot
win simply by occupying many pixels, and a single rare observation gets little
weight. Smooth each channel histogram twice with the kernel [¼, ½, ¼].

Convert the histogram to a positive allocation density:

    d_j = 0.2 + 0.8 min(8, 64 hist_j / Σ hist).
    F_j = cumulative_sum(d) / Σ d.

If there is no entropy evidence, use the identity curve. Otherwise F is strictly
increasing, starts at zero, and ends at one. Its slope is larger in ranges
supported by high-entropy colors. Those ranges get more output interval; ranges
with little entropy support get less. The positive floor reserves some range
even for unsupported colors. The cap limits concentration before normalization;
it is not a claim that the final normalized slope is bounded by eight.

The proposal is a regularized decorrelation followed by channel allocation:

    proposal(x) = (1 − allocation) clamp(z) + allocation F(clamp(z)).

Bake this entire composition into a **33³ RGB LUT** in fixed input-RGB coordinates.
This is a trilinear approximation to the proposal, not exact evaluation between
LUT vertices. Channel-wise allocation is a marginal approximation to the joint
color association; it is not an optimal 3-D perceptual allocation.

### 5. Interpolate the actual assignment profile

Start with the identity LUT. Every rendered frame, with elapsed time dt:

    a = 1 − exp(−dt / adaptation_seconds)
    a = min(a, max_change_per_second × dt / max_abs(target_LUT − current_LUT))
    current_LUT += a × (target_LUT − current_LUT).

Cap dt at 0.1 seconds so a paused/hidden source cannot produce a large catch-up
jump. Interpolate the finished RGB mapping, not eigenvectors, palette IDs, or
video frames. This avoids eigenvector sign/order flicker and does not create
frame-blending ghost trails. Convex interpolation keeps LUT values in gamut.
The per-node change bound also bounds the change at any fixed input color under
trilinear interpolation, before output quantization. Source motion or changing
source colors are not bounded by this profile-only guarantee.

Upload the current LUT as a 1089 × 33 RGBA16 UNORM texture (about 281 KiB).
The CPU profile remains float; upload rounding error is at most 0.5/65535 per
component. RGBA16 avoids an OBS 32.2.1 Metal RGBA32F upload failure in which
`MTLPixelFormat.bytesPerPixel` has no 128-bit case. Rendering uses one source
sample and two bilinear LUT samples, with interpolation between blue slices.
The displayed frame retains full source resolution and alpha.

## Controls and operational scope

| Control | Default | Meaning |
|---|---:|---|
| Effect strength | 1.0 | Blend the enhanced result with the original; zero is exact bypass |
| Decorrelation stretch | 0.65 | Blend identity with the regularized covariance transform |
| Entropy color-range allocation | 0.80 | Blend with the entropy-weighted cumulative maps |
| Profile adaptation | 1.5 s | Exponential time constant; a stable target reaches about 63% in one time constant when the slew cap is inactive |
| Maximum color change | 0.25/s | Maximum profile change per second in normalized channel units |
| Lattice columns | 128 | Sparse-analysis density; height follows aspect ratio |
| Analysis updates | 10/s | Target-profile refresh rate |
| Noise floor | 0.015 | Suppress very small variations and regularize covariance |
| Maximum decorrelation gain | 6 | Limit weak-component expansion |
| Freeze current color assignment | Off | Hold the current LUT exactly |

A single scene-wide color mapping is deliberate: equal input colors receive
equal output colors, even in different regions. High-entropy regions benefit
when their colors can be distinguished statistically from low-entropy regions.
If both regions use the same colors, this global mapping cannot assign them
independent contrast. There are no spatially varying mapping seams or local
halo kernels, but amplified noise and clipping can still create visible artifacts.

Entropy is not semantic importance: sensor noise, foliage, compression artifacts,
and unwanted text can all have high entropy. The noise floor is a heuristic,
not a noise estimator. The sampled entropy scale depends on lattice density,
and spatial arrangement inside each block is discarded by the histogram.

This filter operates on SDR encoded display RGB, after OBS source color
conversion. HDR and floating-point SDR source spaces are bypassed rather than
silently quantized. No HDR processing is implemented. The output gamut clamp
can discard extreme transformed distinctions; this is an enhancement, not a
lossless or perceptually uniform transform. Scene cuts deliberately retain the
slow profile transition, so the old mapping can influence a new scene briefly.

The profile stays current once per OBS video frame even if preview and output
both render the source. Resolution changes recreate only analysis resources and
discard stale readbacks; they retain the current color assignment. Settings are
handed off under a mutex, while graphics and profile work stay on the render
thread. Small staging maps can still synchronize with the graphics backend;
double buffering is not a universal nonblocking guarantee.

## Build and verification

Portable core, existing regression suite, and PPM reference renderer:

```sh
cmake -S realtime_vector_fx -B /tmp/rvfx-entropy -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rvfx-entropy -j4
ctest --test-dir /tmp/rvfx-entropy --output-on-failure
/tmp/rvfx-entropy/rvfx_entropy_demo input.ppm output.ppm
```

macOS OBS bundle, using headers that match the installed OBS version:

```sh
OBS_SOURCE_DIR=/path/to/obs-studio-32.2.1 \
OBS_CONFIG_INCLUDE_DIR=/path/to/generated-obsconfig \
SIMDE_INCLUDE_DIR=/path/to/simde-0.8.2 \
RVFX_RUN_ENTROPY_SMOKE=1 \
sh realtime_vector_fx/tools/build_obs_macos.sh
```

The bundle is `realtime_vector_fx/dist/realtime-vector-fx.plugin`. Copy it to
`~/Library/Application Support/obs-studio/plugins/`, replacing the previous
bundle of the same module if applicable, restart OBS, then add **Entropy-Guided
Decorrelation Stretch** under the source's **Effect Filters**. The new filter
shares the module with the existing RVFX filters and keeps their IDs intact.
Installing an update does not modify saved scene/filter settings.

`RVFX_RUN_OBS_SMOKE=1` separately runs the pre-existing GPU/posterizer regression.
The new smoke test verifies identity/gamma/alpha, a changing assignment profile,
CPU/GPU transform agreement, freeze, and zero-strength bypass using actual
libobs and Metal. It needs a GUI-capable macOS session and a matching OBS runtime.

Measured results are recorded in `ENTROPY_VERIFICATION.md`.

## Same-output optimization

The implementation now reuses analysis storage and histogram terms, computes
weights once per RGB family, and uses ARM64 SIMD for profile interpolation and
packing. Exact profile-distance reuse and unchanged-table detection avoid
redundant scans/uploads. On Metal, the staging copy is deferred to the start of
the next frame while retaining the same sample-to-analysis latency. See
[the optimization audit](ENTROPY_OPTIMIZATION.md) for equivalence and timings.
The mathematical definitions and all controls above are unchanged.

## Optional Chains

The checkable **Chains** group adds independently sampled RGB, brightness, and
contrast stages, individual amounts, a brightness reference, and all six
application orders. It defaults off; enabling with the default amounts preserves
the existing RGB look. See [ENTROPY_CHAINS.md](ENTROPY_CHAINS.md) for equations,
controls, composition, preservation checks, and measured approximation limits.
