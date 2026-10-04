# Mini legacy experiment recovery

These files were untracked in `/Users/joshuahkuttenkuler/code/bfft` at
`7e410729c2a182b13e0447742797fc252ebc7da8`. They were copied read-only through
the `m4host` route selector and verified against pre/post SHA-256. Exact origins
and duplicate mappings are in `recovery/2026-10-03/remote-manifest.json`.

The RFHT headers, scalar/NEON comparison programs, fast-sincos demonstration,
STFT stress script, and retained CSV are historical experiments. They are not
newly validated improvements to the current BFFT API. The Mini's old committed
HEAD is already an ancestor of this repository's main history, so its older
library implementation was not substituted for the current one.

Identical copies of the audio roundtrip benchmark and two interpolation scripts
already exist at `tests/audio_magphase_roundtrip_benchmark.cpp`,
`experiments/self_geometric_harmonic_interpolation.py`, and
`experiments/conv_conservative_multiresolution.py`. The two identical NEON
comparison copies were deduplicated to `experiments/compare_rfht_neon_bfft_audio.cpp`
within this directory.

`sbench.py` expects a sibling `stft.py` or one directory above it. Its original
location satisfied that assumption. To reproduce the historical program without
editing it, stage it and the repository's `stft.py` together in a temporary
directory, and put the repository on `PYTHONPATH`. The saved CSV is historical
evidence; it has not been regenerated against today's library.

`patch_atan2f_ulp_check.py` modifies a named scratch file in `/tmp`. It is kept
as provenance, not run automatically. Its external scratch input was not among
the recovered files. Build the standalone trigonometric demonstration with:

```sh
c++ -O2 -std=c++17 experiments/mini_legacy_recovery/fast_sincos_demo.cpp \
  -o /tmp/bfft-recovered-fast-sincos
```

The RFHT comparison programs require the current BFFT library plus
`-Iinclude -Iexperiments/mini_legacy_recovery`; the NEON version additionally
requires an AArch64 compiler/host. Preserve all reported numerical error, not
only transform timings.
