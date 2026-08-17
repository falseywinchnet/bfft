# Fourier-shell Eikonal transport

This experiment tests whether cross-observation uncertainty can define a useful
transport metric on Fourier shells. It reconstructs a conjugate-symmetric polar
spectrum from several noisy observations with both private missing wedges and a
shared blind wedge.

See `RESULTS.md` for the completed 40-trial experiment, the nine-cell regime
sweep, and the boundary between helpful and unreliable transport.

All atlas methods use the same estimated complex edge connection and the same
seed selection. `isotropic_atlas` uses flat path costs, `shuffled_metric` moves
the inferred costs to incorrect locations, and `eikonal_atlas` uses the costs
where cross-observation coherence placed them. This isolates the effect of the
transport metric from the effect of angular interpolation itself.

Run tests and the full CPU experiment on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/fourier_eikonal_transport/test_fourier_eikonal_transport.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/fourier_eikonal_transport/run_experiment.py
```

Map the transport regime over observation count and noise with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/fourier_eikonal_transport/run_regime_sweep.py
```
