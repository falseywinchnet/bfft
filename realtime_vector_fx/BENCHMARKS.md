# Performance ledger

Measured 2026-08-17 on the repository's M4 Mac mini using Apple Clang, `-O3
-ffast-math`, eight palette priors, a 480-pixel-wide trace lattice, 4,096
palette samples, and up to 2,048 selected segments per frame. Each run contains
12 warmup frames followed by 180 measured synthetic frames. Changing and
exactly static variants separate worst-case analysis from temporal reuse.

The total includes persistent-prior update, OKLab lattice posterization, all
boundary updates, randomized trace/effect generation, full-resolution CPU
posterized compositing, and trace/glow drawing. Synthetic input generation is
outside the timed interval, matching an OBS filter that receives an existing
frame.

The following table is the pre-bifurcation-port reference baseline for the
combined FX engine.

| Format | Scene | Resolution | Core p50 | Core p95 | Composited mean | Composited p95 | 30 fps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| RGBA | changing | 1280×720 | 2.779 ms | 2.808 ms | 4.035 ms | 4.084 ms | pass |
| RGBA | static | 1280×720 | 1.202 ms | 1.241 ms | 2.392 ms | 2.431 ms | pass |
| RGBA | changing | 1920×1080 | 2.804 ms | 2.848 ms | 5.466 ms | 5.534 ms | pass |
| RGBA | static | 1920×1080 | 1.209 ms | 1.334 ms | 3.844 ms | 3.980 ms | pass |
| NV12 | changing | 1280×720 | 3.453 ms | 3.503 ms | 4.363 ms | 4.421 ms | pass |
| NV12 | static | 1280×720 | 1.252 ms | 1.282 ms | 2.164 ms | 2.196 ms | pass |
| NV12 | changing | 1920×1080 | 3.558 ms | 3.613 ms | 5.616 ms | 5.681 ms | pass |
| NV12 | static | 1920×1080 | 1.237 ms | 1.314 ms | 3.088 ms | 3.142 ms | pass |

The worst 1080p p95 currently consumes 17.0% of the 33.333 ms budget, leaving
about 27.65 ms of headroom. Exact source-token reuse lowers static core p95 by
52.8% for RGBA and 63.8% for NV12. This is an algorithmic reference benchmark, not yet an
OBS capture-to-present latency measurement.

After porting the occupied-space OKLCH bifurcation controls, current M4 Mini
changing-RGBA measurements at 1920×1080 are:

| Mode | Colors | Core p95 | Composited p95 | 30 fps |
| --- | ---: | ---: | ---: | ---: |
| Combined FX | 8 | 5.211 ms | 7.902 ms | pass |
| Posterizer only | 24 | 9.211 ms | 11.612 ms | pass |
| Posterizer only | 64 | 26.224 ms | 28.751 ms | pass |

Posterizer-only mode skips topology, segment, glyph, trail-history, and overlay
work. The 64-color row records the slower of two final 60-frame runs after 12
warmups; the 8- and 24-color rows use 180 measured frames.

The OBS 32.2.1 CPU and graphics adapters were also compiled against matching
source headers, linked to the installed OBS framework, bundled, ad-hoc signed,
and verified. The graphics path keeps full-resolution posterization and line
overlay on the GPU and reads back only the bounded analysis lattice. Live
capture-to-present profiling remains outstanding. A Metal/libobs smoke harness
successfully loaded the non-installed bundle, instantiated both filters,
compiled both embedded effects, allocated the dynamic textures/vertex buffer,
and rendered eight frames through a synthetic source plus the GPU filter with
nonempty staged output and no OBS error-level logs.
The non-installed verification bundle is
`dist/realtime-vector-fx.plugin` on the development MacBook.

The same harness was then run four times at 1920×1080, each with 12 warmups and
120 measured frames, on the development Apple A18 Pro host. A static synthetic
RGBA source had a 0.767 ms median bare-source frame; the complete GPU filter
chain had a 2.903 ms median total frame (2.516–3.161 ms range), for 2.079 ms
median incremental overhead (1.779–2.523 ms range). Each measurement ends with
a staging-map synchronization, so queued Metal work is included, but this is
still a headless libobs render benchmark rather than OBS UI capture-to-present
latency. All phosphor/liquid-metal/emboss modes and falling/arcing/mixed glyph
motions were exercised without error, and the filter result was staged to a
1920×1080 image for visual inspection.
