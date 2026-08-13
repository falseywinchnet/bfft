# Mechanical cone-FFT experiments

Companion code for [`notes/mechanical_cone_fft.md`](../../notes/mechanical_cone_fft.md).

## Algebraic certificate

`certificate.py` depends only on NumPy. It checks the cone quotient algebra,
the exact normalized `N=8` Bruun factorization, both diagonal gauges,
reciprocal virtual work, the 44-bar convex vocabulary, and a dense reciprocal
spring existence result.

```sh
python experiments/mechanical_cone_fft/certificate.py
```

## Beam model

`beam_model.py` requires NumPy, SciPy, and Matplotlib. It builds the force and
displacement whiffletree forms, sweeps hinge stiffness and output loading, and
runs tap-position Monte Carlo.

```sh
python experiments/mechanical_cone_fft/beam_model.py \
  --trials 1000 --out build/mechanical-cone-fft/beam
```

## Spring control experiment

`spring_surrogate.py` proves that isolated and dense balanced spring maps can be
exact while a naive cascade is corrupted by reciprocal back-loading.

```sh
python experiments/mechanical_cone_fft/spring_surrogate.py \
  --trials 500 --out build/mechanical-cone-fft/springs
```

The beam and spring programs are linear research models, not continuum FEA or
fabrication CAD. Monte Carlo uses the fixed seed `20260812`. Generated plots
and CSV files are intentionally not tracked.
