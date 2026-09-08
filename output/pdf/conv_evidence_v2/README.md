# CONV evidence packet

This directory contains the deterministic synthetic evidence used by the
paper. `results.json` holds every per-case analytic-truth record and the
aggregates plotted in the PDF. The three `eikonal_convstar_conformance_*.json`
files compare CONV* with the formal all-source Eikonal operator at a common
65-by-65 target. `conv_evidence_native_timing.json` contains the M4 native
timing records, including warmup and repeat counts, platform, backend, median,
MAD, and minimum time. `green_variable_metric_reference.json` and
`GREEN_KERNEL_ADMISSION_ANALYSIS.md` give the anisotropic cardinal-Green
audit and its derivation.

The figures are generated directly from those records. No natural-image case
is part of the scored population. `evidence_manifest.json` records the exact
artifact sizes and SHA-256 digests.

Rebuild commands:

```sh
python -m experiments.conv_synthetic_evidence --out output/pdf/conv_evidence_v2
python -m experiments.build_conv_evidence_packet
python experiments/compose_conv_paper.py
```

The conformance and native-timing runs use the M4 Mini commands documented in
the repository's `AGENTS.md` and the generator command strings stored in the
manifest.
