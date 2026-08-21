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

## Personal deblurrer

Run the complete invariant suite and the spatial estimation batteries on the
M4 Mini CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest discover -s personal_deblurrer -t . -p 'test_*.py'

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_spatial_estimation_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_spatial_estimation

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_dense_estimation_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_dense_estimation

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_visibility_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_visibility
```

Copy selected `/tmp` JSON artifacts back immediately. The Dear PyGui interface
uses the MacBook's `.venv-jpeg` environment.

## Continuous-support FMMT denoiser experiment

Run the support, relation-simmer, and representation invariants plus the 1-D/2-D
probes on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_sample_series \
  denoiser.test_transport_support denoiser.test_affine_relation_transport \
  denoiser.test_cross_predictive_transport \
  denoiser.test_cross_predictive_transport_2d \
  denoiser.test_causal_ancestry \
  denoiser.test_causal_parity_transport \
  denoiser.test_causal_predictive_geometry \
  denoiser.test_continuous_source_transport \
  denoiser.test_crossfit_characteristic_transport_2d \
  denoiser.test_witnessed_characteristic_transport_2d \
  denoiser.test_fmmt_representation \
  denoiser.test_fused_transport_geometry \
  denoiser.test_reflection_consistent_posterior_2d \
  denoiser.test_probe_nuisance_geometry_2d \
  denoiser.test_continual_eikonal_noise_transport_2d \
  denoiser.test_continual_fabada_eikonal_2d \
  denoiser.test_canonical_variance_transport_2d \
  denoiser.test_backward_moment_smoother_2d \
  denoiser.test_causal_information_lineage_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser probes --out /tmp/denoiser_transport_probes

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_1d_foundational_simmer \
  --out /tmp/denoiser_1d_foundational_simmer.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_predictive_seed_geometry \
  --out /tmp/predictive_seed_under_corruption.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_1d_cross_predictive_battery \
  --size 256 --seeds 3 \
  --out /tmp/denoiser_cross_predictive_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_2d_denoiser_battery \
  --size 96 --seeds 2 --out /tmp/denoiser_2d_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_causal_predictive_geometry \
  --size 40 --seeds 2 --out /tmp/causal_predictive_geometry.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_causal_fixed_point \
  --size 20 --quantiles 8 --continuations 16 \
  --out /tmp/causal_fixed_point.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_crossfit_characteristic_transport \
  --size 32 --seeds 1 \
  --out /tmp/crossfit_characteristic_transport.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_continuous_tangent_convergence \
  --size 20 --out /tmp/parallel_jet_tangent_convergence.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_continuous_tangent_information_geometry \
  --size 20 --quantile-counts 8,16,32 \
  --out /tmp/continuous_tangent_lineage_jet_geometry.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_nuisance_geometry_2d \
  --size 40 --seeds 2 --out /tmp/nuisance_geometry_40.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_complete_moment_lineage_2d \
  --size 16 --phase-count 2 \
  --sources 'cameraman,tapered hair,geometric interfaces,woven chirps,line drawing,multiscale blobs' \
  --out /tmp/complete_moment_lineage_36_phase2.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_zero_residual_component_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_zero_residual_component_2d \
  --size 16 --phase-count 1 \
  --sources 'cameraman,tapered hair,woven chirps' \
  --out /tmp/zero_component_terminal_18_phase1.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_terminal_component_visual_2d \
  --size 64 --out /tmp/terminal_component_visual64
```

The V3 support-corruption probe imports the full BFFT Meyer and vision ABI.
Build or point to the complete shared library, run the probe on the Mini, and
copy its `/tmp` JSON immediately:

```sh
python3 -m denoiser.probe_v3_support_under_corruption \
  --out /tmp/v3_support_under_corruption.json
```

The vector recurrence is the measured FMMT bootstrap. The scalar Numba form is
retained as an exact representation oracle. The Dear PyGui interface uses the
MacBook's `.venv-jpeg` environment.

Run the active 2-D representation timing gate on the M4 Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.benchmark_2d_acceleration \
  --sizes 128,256 --repeats 3 \
  --out /tmp/denoiser_2d_acceleration_m4.json
```

Copy `/tmp/denoiser_2d_acceleration_m4.json` back immediately after the run.
