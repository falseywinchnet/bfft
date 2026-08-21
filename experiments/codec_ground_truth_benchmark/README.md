# Synthetic codec ground-truth benchmark

This benchmark replaces “that edge looks smoother” with a controlled,
reference-grounded comparison. It renders eight deterministic scenes that
separately stress oblique edges, chroma boundaries, low-slope gradients, thin
lines and text, flat/texture transitions, phase/frequency structure, alpha,
and mixed photo/graphic content.

Each scene produces:

- a lossless PNG reference rendered from a 4x supersampled source;
- a pixel-identical but deliberately uncompressed PNG upload;
- a high-quality 4:4:4, non-optimized JPEG upload.

The upload files are inefficient containers, not intentionally damaged
images. This gives every optimizer the same high-information starting point.
JPEG is necessarily a first-generation capture; all final metrics still use
the untouched PNG reference.

## Protocol

Generate a suite:

```sh
python3 -m experiments.codec_ground_truth_benchmark generate \
  ~/Desktop/tinypng_synthetic_benchmark
```

Upload all 16 files in `upload/` to TinyPNG without renaming them, then place
the returned files together in `candidates/tinypng/`. The `png__` and
`jpeg__` prefixes let the evaluator distinguish both codecs in one folder.

Run our optimizers at each corresponding TinyPNG result's exact byte count:

```sh
python3 -m experiments.codec_ground_truth_benchmark run-ours \
  ~/Desktop/tinypng_synthetic_benchmark
```

When running through a mirrored compute host, place results outside the mirror
so a later sync cannot delete earlier cases:

```sh
python3 -m experiments.codec_ground_truth_benchmark run-ours SUITE \
  --output-dir /tmp/codec-ground-truth-ours
```

Evaluate both implementations against the ground truth:

```sh
python3 -m experiments.codec_ground_truth_benchmark evaluate \
  ~/Desktop/tinypng_synthetic_benchmark
```

Before TinyPNG results exist, validate the inputs and metric pipeline with:

```sh
python3 -m experiments.codec_ground_truth_benchmark evaluate \
  ~/Desktop/tinypng_synthetic_benchmark --include-upload
```

The command writes `evaluation.json` and `evaluation.csv`. Besides bytes,
color SSIM, PSNR, and luma edge PSNR, it reports chroma-edge PSNR, edge bias,
edge-band error, ringing, alpha error, false plateaus and curvature on known
gradient probes, texture energy/correlation, and spurious texture in known
flat regions. This makes “smooth” falsifiable: blur, ringing suppression,
banding, chroma bleed, and good antialiasing no longer collapse into one score.

Three known-bad PNG controls are generated in `controls/`: Gaussian blur,
coarse channel banding, and an exaggerated unsharp halo. Include them when
calibrating the metrics:

```sh
python3 -m experiments.codec_ground_truth_benchmark evaluate \
  ~/Desktop/tinypng_synthetic_benchmark --include-upload --include-controls
```

Any archived implementation can participate without changing the harness.
Put its files in a directory under the same `png__CASE.png` / `jpeg__CASE.jpg`
names and pass repeatable labels:

```sh
python3 -m experiments.codec_ground_truth_benchmark evaluate SUITE \
  --candidate old=OLD_RESULTS --candidate new=NEW_RESULTS \
  --candidate tinypng=SUITE/candidates/tinypng
```
