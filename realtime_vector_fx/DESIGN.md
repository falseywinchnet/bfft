# Frame-budget design

The 30 fps target is a deadline, not an average-throughput target. The initial
CPU implementation assigns the bounded trace lattice every frame, but caps
palette samples and emitted segments. SVG strings and DOM-like paths never
exist on the hot path; `TraceSegment` is the vector representation and
`svg_snapshot()` is an explicitly off-hot-path consumer.

| Stage | Bounded by | Persistent state |
| --- | --- | --- |
| Palette prior | `palette_samples * palette_colors` | OKLab centroids |
| Poster ownership | `trace_width * trace_height * palette_colors` | source token, cached OKLab, label lattice |
| Trace update | at most two edges per lattice cell | dense edge age/phase |
| Run compiler | at most two edges per lattice cell | stable constituent edge slots |
| Reveal scheduler | `segments_per_frame` after run compilation | frame/run phase |
| Base effects | 1–2 commands per selected run | none beyond phase |
| Glyph effects | `glyph_particles` | position, velocity, curvature, life, glyph |
| CPU composite | output pixels + selected geometry | packed RGB or native 4:2:0 frame |

The primary architectural rule is that temporal coherence is state, not an
extra optimization pass. Palette indices remain stable through EMA priors and
trace IDs remain stable through lattice addresses. This avoids the two costs
that make one-shot SVG methods unsuitable for live video: rebuilding a global
color hierarchy and matching arbitrary contour objects across frames.

Each lattice address also retains the exact packed source sample that produced
its OKLab state. An identical RGBA or YUV token bypasses transfer decoding and
cube roots, but the cached perceptual value is still compared with every
current palette prior. Static regions accelerate without freezing ownership
while palette centroids evolve.

The plugin exposes two implementations. The async CPU filter is the deployable
fallback for packed RGB, NV12, and I420 frames without a full-resolution RGB
staging allocation. The graphics filter downsamples the source into an OBS
texrender, pipelines two low-resolution staging surfaces so analysis is one
frame delayed rather than blocking on the current render, uploads only the
label lattice and 64-entry palette, posterizes the full-resolution source with
a point-sampled shader, and overlays trace/glyph commands with one reusable
dynamic vertex buffer. A later compute backend can eliminate the bounded CPU
analysis readback while preserving the same temporal engine contract.
