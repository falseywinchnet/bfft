# Relaxed Chip Design

This is the BFFT-side research capsule for treating standard-cell placement as
a two-way relaxation problem:

1. relax cells and legal capacity into a low-rank transported measure;
2. preserve each cell's identity as a phase in the reference measure;
3. carry that phase through the transported measure;
4. unrelax directly to a physical support target, then legalize capacity.

The information about *what goes where* is in the transport itself. The
decoder therefore does not enumerate candidate placements or evaluate local
alternatives. Its core operation is CDF conjugation on a bounded active
support.

## What is here

- `unrelaxation.py` contains the reusable rank-prefix phase evaluator and
  identity-preserving support-CDF inverse.
- `basin_walk.py` contains fixed spherical initial-chart walks and a weighted
  low-rank far-field projection used for initializer diagnostics.
- `check_unrelaxation.py` checks identity, directed transport, and rank-prefix
  phase semantics.
- `check_basin_walk.py` checks spherical endpoints, axis controls, and the
  chart projection.
- `example.py` is a minimal synthetic transport.
- `DESIGN.md` records the representation and scaling argument.
- `RESULTS.md` separates best raw WB quality from the best transferable law.
- `SETUP.md` covers this standalone capsule and the full circuit harness.

The production experiment remains in `HypersphericalCircuitLab`, where DEF
emission, legal capacity, native sparse kernels, and frozen datasets live. This
folder intentionally contains no benchmark inputs or generated DEF files.

## Quick check

From the BFFT repository root:

```sh
python3 -m relaxed_chip_design.check_unrelaxation
python3 -m relaxed_chip_design.check_basin_walk
python3 -m relaxed_chip_design.example
```

See [RESULTS.md](RESULTS.md) before comparing individual HPWL figures: the
lowest WB-only number and the best cross-density transport law are different
results answering different questions.
