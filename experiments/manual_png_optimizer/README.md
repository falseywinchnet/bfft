# Ownership-aware PNG optimizer

This is the PNG-native sibling of `manual_jpeg_optimizer`. It always writes a
standard PNG and requires no custom decoder.

The shared idea is constituent ownership, not JPEG machinery. PNG has no DCT
or quantization table, so the exact transport domain is its indexed raster:

1. allocate a perceptual palette, using composited/premultiplied color plus
   alpha for nonopaque images so hidden RGB consumes no palette capacity;
2. treat every pixel as ownership of one palette constituent;
3. optionally bloom-fill ownership across edge-gated neighbors when the saved
   boundary cost repays the perceptual regret;
4. allocate extra palette capacity to boundary colors with edge-weighted
   k-means++ and deterministic full-image Lloyd refinement;
5. optionally phase-mix the remaining smooth-region residual on a compressible
   ordered lattice, with exponential barriers at ownership edges;
6. spectrally reorder palette identities and inverse-remap all pixel labels;
7. choose PNG scanline filters and DEFLATE strategy by actual output bytes;
8. decode and select the measured rate/distortion winner.

Step 6 is lossless. The palette and label permutation cancel exactly, but the
new index field can be substantially easier for DEFLATE to represent. The
lossy spatial pass is anchored to the quantizer's original owner, so zero
pressure is the exact identity and a color-basis change cannot masquerade as
compression.

## CLI

```sh
# Pixel-exact metadata/filter/DEFLATE optimization
python -m experiments.manual_png_optimizer lossless input.png output.png \
  --report output.json

# A guided rate/distortion search
python -m experiments.manual_png_optimizer optimize input.png output.png \
  --target-bytes 300000 --minimum-ssim 0.85 --report output.json

# One directly controlled palette/flow point
python -m experiments.manual_png_optimizer optimize input.png output.png \
  --colors 64 --ownership-strength 0.0015

# Edge-aware 256-color palette with banding-controlled residual transport
python -m experiments.manual_png_optimizer optimize input.png output.png \
  --colors 256 --quantizer auto --ownership-strength 0 \
  --dither selective --diffusion-strength 0.9 \
  --diffusion-edge-barrier 3

python -m experiments.manual_png_optimizer compare input.png output.png
python -m experiments.manual_png_optimizer gui input.png
```

The automatic search uses a descending color bracket followed by measured
integer rate correction, refines a bounded atlas of edge priors with
deterministic full-image Lloyd steps, and runs a short ownership-pressure
ladder. It does not enumerate hundreds of filter conjectures: palette orders
are first probed with level-1 DEFLATE, the two best reach terminal encoding,
and the terminal pass searches a small measured set of DEFLATE memory levels.
Under the byte ceiling, SSIM losses greater than 0.00025 are rejected; within
that narrow band PSNR and edge PSNR resolve structural near-ties.

When the repository's native library is available, weighted k-means++ and
nearest-palette ownership run in C++ while retaining NumPy's seeded center
sequence and SciPy's first-code tie rule. Independent terminal DEFLATE trials
run on a bounded four-worker pool; results are consumed in their original
order, so exact-size ties remain deterministic. On the eight-scene synthetic
control this reduced the M4 Mini sweep from about 107 seconds to 48 seconds,
and all eight optimized PNG files remained byte-for-byte identical. On the
representative mixed scene, the local profile fell from 29.6 to 9.9 seconds.

`selective` diffusion is deliberately different from Floyd--Steinberg. For
each smooth-region pixel it finds the palette constituent on the opposite side
of the current quantization residual, computes the mixture that reconstructs
the source color in expectation, and realizes that mixture on an 8×8 phase
lattice. Edge conductance decays exponentially, so residual ownership travels
within a region rather than bleeding across its boundary. The report's
`smooth_transition_coverage` exposes the banding/entropy trade instead of
hiding it behind SSIM alone.

On the 1714×823 city control, the balanced preset writes 574,070 bytes in about
8.7 seconds versus TinyPNG's 582,613 bytes in about 10 seconds. It reaches SSIM
0.90736, PSNR 34.70 dB, edge PSNR 28.09 dB, and smooth-transition coverage
0.383; TinyPNG measures 0.90213, 34.72 dB, 28.34 dB, and 0.424 respectively.

## GUI

The Dear PyGui interface runs optimization on a worker thread, exposes the
rate, quality, palette, ownership, edge-protection, filter, and lossless
controls, previews original and optimized pixels, and has an explicit **Save
processed result** button. Saving also writes the JSON measurement report.

Install the optional desktop dependencies with:

```sh
python -m pip install -e '.[png-lab]'
```
