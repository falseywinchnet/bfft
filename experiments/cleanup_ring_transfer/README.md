# Recovered Cleanup ring-transform experiment

This source snapshot preserves the two BFFT extensions that remained only in
the Cleanup consumer checkout: the lane-valued Bruun DIT entry point and a
16-point complex codelet. Its independent ring-field probe, minimal header
dependencies, original provenance, mathematical derivation, and two compact
receipts are included. The primary BFFT kernels are unchanged by this recovery.

Original root: `/Users/ultimussecundai/Downloads/cleanup2`. Exact copied paths
and hashes are in `recovery/2026-10-03/consumer-manifest.json`. The original
`vendor/bfft/PROVENANCE.json` describes its baseline, while
`vendor/bfft/LOCAL-EXTENSIONS.md` identifies the later extensions; neither was
rewritten to imply that the extensions were already upstream.

With a built BFFT library, compile the probe from the BFFT repository root:

```sh
c++ -O2 -std=c++17 -Iinclude \
  -Iexperiments/cleanup_ring_transfer/native/src \
  -Iexperiments/cleanup_ring_transfer/vendor/bfft/src \
  experiments/cleanup_ring_transfer/research/true_superresolution/ring_transform_probe.cpp \
  -Lbuild -lbfft -Wl,-rpath,"$PWD/build" -o /tmp/bfft-recovered-ring-probe
/tmp/bfft-recovered-ring-probe
```

Use `m4build` for this build/run in this workspace, and keep its output separate
from other tasks. The probe checks complete fields, literal-DFT samples, winning
peaks and competitor values. Its timing loop is an isolated ring-bank comparison,
not a complete audio callback or a live SDR application test. The retained
historical receipts and their limitations are described in
`docs/ring-forward-transform.md`.
