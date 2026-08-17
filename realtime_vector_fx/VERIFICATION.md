# Completion evidence

This ledger maps the original realtime unification requirements to current,
reproducible evidence. It deliberately distinguishes the requested end state
from optional future optimizations.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Identify the optimized posterizer | `posterizer/src/posterizer/core.py`, `oklch.py`; commits `a25e385`, `701e00f`, `98ae5d5` | Function/commit map in `SOURCE_INVENTORY.md` |
| Identify the optimal SVG maker | `svg_converter/src/tlvector/core.py` and compact V2 in `svg_converter_v2/src/tlvector_v2` | Function/commit map in `SOURCE_INVENTORY.md` |
| Identify and preserve the SVG animation engine | `legacy/svg_oscilloscope_renderer_v5.html` | Exact 30,212-byte import; SHA-256 `dcd0ed37de32ba02defc52b6a41547d20ffaa0619bf533ec6c328ee3f88206f2` matches the recovered Downloads artifact |
| One high-speed C++ program/folder | `include/rvfx/engine.hpp`, `src/engine.cpp`, `obs/*.cpp` | C++17 core and both OBS adapters build from `tools/build_obs_macos.sh` |
| Hold and update posterization priors | Persistent OKLab centroids with weighted EMA and stable palette identity | Core test bounds centroid movement across a shifted frame |
| Avoid recomputing unchanged perceptual state | Exact packed RGBA/YUV tokens and cached OKLab lattice | Tests require zero changed cells on an identical frame; M4 static benchmarks record the speedup |
| Posterize, trace, and update existing traces | Dense stable edge slots retain age/phase; maximal compatible runs are rebuilt from them | Tests require a stable ID to survive with increasing age and require joined runs longer than one cell |
| Random subset without starving lines | Persistent Fisher-Yates visit order over stable run IDs; random local phrase per visit | Tests require an entire cycle of unique `source_id` values before any repeat and a slice shorter than a long source line |
| Off-hot-path SVG traces | `Engine::svg_snapshot()` | Tests require SVG paths and glyph text; demo snapshot is XML-validated |
| Continual smooth/fading trace history | Two full-resolution GPU texrenders ping-pong previous energy through exponential decay before depositing new geometry | Metal filter-chain smoke test renders and stages persistent output; 1080p captures cover all three modes |
| Separate `#82b361` glowing glyph engine | Persistent particles, 5×7 glyph atlas, independent falling/arcing/mixed motion and trails | Tests prove downward falling motion and nonzero arcing curvature; all motion modes pass the Metal smoke test |
| Phosphor, liquid metal, emboss/source-color sheen | Effect-specific C++ commands plus width-aware GPU triangle quads and glow | Core mode assertions plus isolated 1080p Metal captures for all three modes |
| Less than 1/30 second | Bounded 480-wide analysis lattice, capped samples/visits, no hot-path SVG serialization | Worst changing M4 CPU fallback p95 is 5.681 ms at 1080p; median synchronized Metal/libobs filtered frame is 2.903 ms |
| OBS plugin | Async packed/YUV CPU filter and texture-native Metal filter in one module | OBS 32.2.1 loads the signed arm64 bundle, compiles all three effects, creates both filters, attaches the GPU filter to a source, renders/stages frames, cycles every effect/motion mode, and tears down with no error-level log |

The deliverable is `dist/realtime-vector-fx.plugin`. It is ad-hoc signed for
local use and intentionally not copied into the user's OBS plugin directory by
the build/test workflow.

Optional next-generation work—compute-shader palette/edge extraction,
P010/HDR, notarization, and platform-specific installers—can improve the
product but is not required to establish the requested realtime C++/OBS engine.
