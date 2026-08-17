# Cat-image rate–distortion result

Source: `istockphoto-508030340-612x612.jpg`, 612×409, 32,530 bytes,
progressive 4:4:4. The embedded luma table is exactly the standard quality-70
table. The stated external target was TinyPNG at 29,200 bytes.

All fidelity values below compare decoded RGB against the decoded source JPEG,
not against an unavailable pre-JPEG original. SSIM is the mean local RGB SSIM;
PSNR is RGB; edge PSNR is measured on luma Sobel components.

| Candidate | Bytes | RGB SSIM | RGB PSNR | Edge PSNR |
|---|---:|---:|---:|---:|
| Source | 32,530 | 1.000000 | 99.00 | 99.00 |
| Coefficient-phase control, q70 4:4:4 optimized | 30,807 | 0.999962 | 66.75 | 55.91 |
| Plain q70 4:2:2 | 27,159 | 0.987077 | 41.38 | 48.64 |
| Conserved ownership optimum, λ=0 | **27,122** | 0.987073 | 41.35 | 48.36 |
| Aligned projection 0.35, q70 4:2:2 | **27,152** | **0.987118** | 41.37 | 48.63 |
| Smallest aligned phase candidate | 27,119 | 0.987102 | 41.34 | 48.61 |
| Spatial/frequency ownership, balanced knee | **27,100** | **0.987172** | 41.33 | **48.69** |
| Ordinary q76 4:2:2 near 29.2 KB | 29,146 | 0.984635 | 41.22 | 35.84 |

The original selected 27,152-byte artifact is 5,378 bytes (16.5%) smaller than the
source and 2,048 bytes (7.0%) below the 29,200-byte target. The byte counts are
decimal; filesystem displays may round these differently.

The later ownership relaxation improves the byte result to 27,122 while
making zero routed coefficient changes at its exact zero-pressure optimum.
Its pre-quantization constituent composition residual is
`1.42e-14`.  This result is important as a control: it proves that transport
cannot improve a nearest-lattice distortion objective at zero rate pressure,
and prevents the relaxation from manufacturing gains by silently deleting
constituents.  A positive-pressure sweep exhibits a sharp first routed phase
near λ=0.5; that 25,525-byte point falls to SSIM 0.97637 and is not the
quality-matched winner.

The direct DCT ownership follow-up supersedes that auxiliary-route result.
With transport λ=0.009, frequency permeability 0.3, Y mobility 0.15, and
Cb/Cr mobility 1.75/1.75, it moves signed constituent ownership over 496,377
spatial and 248,248 within-block frequency edges simultaneously. The result
is 27,100 bytes: 52 bytes smaller than the earlier aligned winner while also
improving SSIM from 0.987118 to 0.987172 and edge PSNR from 48.63 to 48.69 dB.
Positive and negative mass errors are below `1e-10`, and recorded flow
reconstructs coefficient displacement with residual below `1e-10`.

This is the first measured improvement attributable to redistribution itself,
not merely q70 lattice matching or 4:2:2 sampling. The globally optimal claim
applies to continuous signed-mass transport for fixed graph weights and
mobilities; choosing those hyperparameters and the later JPEG projection
remains a measured Pareto search.

## What worked

The dominant result is quantizer phase matching. Re-encoding with the source's
quality-70 bins avoids a second set of incompatible bin centers. Changing only
the chroma lattice from 4:4:4 to 4:2:2 then removes enough chroma entropy to
cross the target with good decoded fidelity. A superficially higher quality of
76 is larger and measurably worse because it requantizes already-quantized
coefficients onto a different lattice.

The v3-inspired connected chroma projection is a very small positive result at
this operating point: relative to the plain q70 4:2:2 control it improves RGB
SSIM by 0.000041 and saves 7 bytes, while losing 0.008 dB RGB PSNR and 0.008 dB
edge PSNR. That is not evidence for a broadly superior transform. It is enough
to retain the method as an inspectable A/B control.

## What did not work

The nonzero global phase rotations at ±22.5° did not win the collective
objective. Moving texture variability into one chroma direction often lowers a
rate proxy, but baseline JPEG cannot undo a spatially varying color rotation.
After rotating back into standard YCbCr, only the deliberately discarded weak
component changes the rate. Most of the hoped-for channel packing cancels.

This establishes a constraint for a follow-up: a genuinely reversible local
three-channel packing needs either a decoder-side transform (and paid side
information) or a codec whose color transform itself is signaled. For ordinary
JPEG, the useful search space is distortion-aware preprocessing plus native
quantizer/sampling/Huffman choices.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.manual_jpeg_optimizer optimize \
  experiments/manual_jpeg_optimizer/assets/istockphoto-508030340-612x612.jpg \
  /tmp/manual_jpeg_cat.jpg --target-bytes 29200 --exhaustive \
  --report /tmp/manual_jpeg_cat.json
```

The checked-in winning JPEG and report are in `results/`. The `stages/`
directory contains all five forward-cascade renders and structural overlays.
TinyPNG's output was not supplied, so equal-quality superiority to TinyPNG is
not established. Use the CLI `compare` command on that file for the exact same
metrics, then inspect the cat's whiskers, eye rings, ear fur, and blanket weave
side by side.
