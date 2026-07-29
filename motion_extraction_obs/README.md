# Posy Motion Extraction for OBS

An independent OBS async-video filter implementing Posy's live motion
extraction vocabulary, plus an acceleration-energy visualization.

Modes include signed and absolute delayed difference, frozen-baseline
extraction, motion overlay, large-feature blur, glow, RGB time separation,
recoloring, color-preserving extraction, and acceleration brightness/warp.

`Capture frozen baseline`, `Clear frozen baseline`, and `Reset delay history`
make edit-time operations usable on a live camera.

The delay history always remains on the source's native pixel lattice. Its
memory budget controls temporal sampling for long delays: one-frame effects
remain exact, while multi-second delays use the closest retained full-resolution
snapshot.
