# Repository compute notes

The authoritative tree is the MacBook checkout. Use the AWDL build mirror for
the SciPy terminal-measurement runs; do not edit the mirrored checkout.

## Manual JPEG optimizer

Run its NumPy/Pillow/SciPy tests and target-image CLI sweep on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/manual_jpeg_optimizer/test_manual_jpeg_optimizer.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.manual_jpeg_optimizer optimize \
  experiments/manual_jpeg_optimizer/assets/istockphoto-508030340-612x612.jpg \
  /tmp/manual_jpeg_cat.jpg --target-bytes 29200
```

The GUI additionally needs Dear PyGui on the MacBook. Copy the remote JPEG and
JSON result from `/tmp` immediately after a sweep.

## Seventeen-square experiment

Run its complete test file on the M4 Mini from anywhere in this repository:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/Developer/CodexBuilds/.venv-bfft/bin/python \
  experiments/square17_transport/test_square17_transport.py
```

The reproducible occupation, exact packet/GDL, frontier-preimage, and lifted
runs and their parameters are documented in
`experiments/square17_transport/README.md`. Invoke them through `m4build` with
the Mini venv above so NumPy evolution and SciPy terminal measurement use the
same runtime.

`m4build` does not copy generated results back. Copy selected JSON/SVG
artifacts from
`/Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539/`
immediately after a run, before the next mirror sync.

## Geometric pi-collision experiment

Run the mpmath oracle tests on the M4 Mini system Python:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/pi_collision/test_pi_collision.py
```

Build the GMP/MPFR C++ benchmark in the mirror:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  make -C experiments/pi_collision clean all
```

The generated executable remains in the mirror. Run it before the next sync:

```sh
ssh m4mini-awdl \
  'cd /Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539/experiments/pi_collision && ./pi_collision_cpp --bits 100000 --m 2'
```

## Elliptic pi experiment

Run the radical-free isogeny tests and build the GMP/MPFR benchmark with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/elliptic_pi/test_elliptic_pi.py

/Users/ultimussecundai/.local/bin/m4build -- \
  make -C experiments/elliptic_pi clean all
```

Run the generated benchmark before another mirror sync:

```sh
ssh m4mini-awdl \
  'cd /Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539/experiments/elliptic_pi && ./elliptic_pi_cpp --bits 100000'
```

## Relational witness spiral experiment

The experiment is NumPy-only and must run on the M4 Mini CPU. From anywhere in
this repository, run:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/relational_witness_spiral/run_experiment.py
```

Run its tests with the same mirrored system Python:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/relational_witness_spiral/test_relational_witness_spiral.py
```

The learned-subspace follow-up uses the Mini's existing CPU Torch installation:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_learned_subspace.py
```

Run the learned jet-transport follow-up and its tests with the same Torch path:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_jet_transport.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_jet_transport.py
```

Run the probabilistic connection-hypothesis follow-up with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_probabilistic_jet.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_probabilistic_jet.py \
  --out /tmp/probabilistic_jet_full
```

Copy the generated `/tmp/probabilistic_jet_full` artifacts back immediately;
they are outside the mirrored checkout so later syncs do not remove them.

Run the associative-memory shell experiment with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_associative_shells.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_associative_shells.py \
  --out /tmp/associative_shells_full
```

Run the projective-labeling quotient and its tests with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_projective_quotient.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_projective_quotient.py \
  --out /tmp/projective_quotient_full
```

Run the projective role-transition experiment and tests with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_projective_transition.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_projective_transition.py \
  --out /tmp/projective_transition_full
```

Add `--crossings-only` to remove within-role self adjacency from the quotient
operator; this matched diagnostic is documented in `RESULTS.md`.

Run the continuous Banach-eikonal sieve and its tests with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_banach_eikonal_sieve.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_banach_eikonal_sieve.py \
  --out /tmp/banach_eikonal_sieve_full
```

Run the structure-agnostic hypersphere-atlas transport and its plot-producing
double-spiral sweep with the same CPU Torch installation:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/test_hypersphere_atlas.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_hypersphere_atlas.py
```

## Fourier-shell Eikonal transport experiment

Run the NumPy tests and multiview transport sweep on the M4 Mini CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/fourier_eikonal_transport/test_fourier_eikonal_transport.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/fourier_eikonal_transport/run_experiment.py

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/fourier_eikonal_transport/run_regime_sweep.py
```

## Omni-inducement benchmark

Run the LELU-only operator comparison and its tests on the M4 Mini CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/omni_inducement_benchmark/test_benchmark.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/omni_inducement_benchmark/run_benchmark.py \
  --out /tmp/omni_inducement_full --widths 16,36 --seeds 3 --steps 600
```

Copy the remote `/tmp/omni_inducement_full` directory back immediately after
the sweep.

## Parameter-matched soft Eikonal study

Run the exact-budget affine-versus-soft-Eikonal comparison on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_matched/test_matched.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_matched/run_benchmark.py \
  --out /tmp/soft_eikonal_vs_mlp --widths 16,36 --seeds 3 --steps 800

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_matched/run_visual_probes.py \
  --out /tmp/soft_eikonal_vs_mlp_visual_probes.json --width 36 --seed 0 --steps 800
```

Copy both remote `/tmp` artifacts back immediately after their runs.

## Soft Eikonal instructive-compartment screen

Run the exact-budget compartment and relational variants on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_instructive/test_instructive.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_instructive/run_screen.py \
  --out /tmp/soft_eikonal_instructive_screen --widths 16 --seeds 2 --steps 400

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/soft_eikonal_instructive/run_visual_probes.py \
  --out /tmp/soft_eikonal_instructive_probes.json
```

## Self-context Eikonal superset

Run the union-catalog exact-budget comparison and tests on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/self_context_superset/test_superset.py

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/self_context_superset/run_benchmark.py \
  --out /tmp/self_context_superset --widths 16 --seeds 2 --steps 400
```
