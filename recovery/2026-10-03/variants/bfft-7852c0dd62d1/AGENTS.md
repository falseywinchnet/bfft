# Repository compute notes

The authoritative tree is the local checkout. For CPU-heavy builds and tests,
use `/Users/ultimussecundai/.local/bin/m4build -- <command>`; the helper selects
the route using `m4host` and mirrors this checkout. For read-only inspection or
copying results, use `ssh "$(/Users/ultimussecundai/.local/bin/m4host)" ...`.
Do not edit the mirrored tree as the primary copy or install remote software.

## CONV fixed-order family after Version 1.0

The fixed source/current configurations and measured transition/cost tradeoffs
are documented in `experiments/conv_fixed_orders/FINDINGS.md` and `README.md`.
Version 1.0 is the six-source/five-current construction with the rolling ledger.
The family has no runtime selector and changes source coefficients, derivative
order, Bernstein controls, current count, ledger dimensions, and exact basin
quadrature as appropriate to each fixed configuration.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.conv_fixed_orders.test_orders -v
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_fixed_orders.run_study \
  --out /tmp/conv_fixed_orders_final4 --repeats 31
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_fixed_orders.timing \
  --out /tmp/conv_fixed_orders_final1.json --repeats 31
```

Copy each result immediately to `output/support_geometry/conv_fixed_orders/`.
The first shared-mirror sync failed; the retained runs used an isolated copy
of the authoritative required files in `/tmp/conv_fixed_orders_workspace` on
the Mini. No remote source was edited. The initial uncached timing screen is
not a performance result; use `conv_fixed_orders_final4` and `final1` records.
Render retained results locally with:

```sh
.venv-jpeg/bin/python -m experiments.conv_fixed_orders.report \
  output/support_geometry/conv_fixed_orders/conv_fixed_orders_final4
```

Do not equate raw transition width with a universal nonlinear transfer function.
Report admitted 2-D carrier behavior and ringing/strip excursions alongside it.
No family member is promoted as a universal replacement for Version 1.0.

## CONV continuous-current audit

The exact ledger-cost recurrence is promoted in the native backend. Its
proof, bit-equivalence checks, and 10–20% measured complete-pipeline time
reduction are documented in `experiments/conv_exact_fusion/FINDINGS.md`.
Run the focused and existing native regressions with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=standalone_conv_resize_demo:. \
  python3 -m unittest experiments.conv_exact_fusion.test_fusion \
  standalone_conv_resize_demo.test_backend experiments.test_conv_distilled_core
```

Its benchmark reconstructs the pre-fusion ledger for a stable comparison:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_exact_fusion.study \
  --methods base,ledger,rolling --repeats 31 --out /tmp/conv_exact_fusion.json
```

Copy that JSON immediately to `output/support_geometry/conv_exact_fusion/`.

The larger algebraic fusion proofs and conservative phase-path certificate
are in `experiments/conv_exact_fusion/LARGER_FUSION_PROOFS.md`. Run their
exact-rational verification and small independent 2-D coverage probe with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.conv_exact_fusion.test_polynomial_proof -v
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_exact_fusion.study_polynomial_proof \
  --out /tmp/conv_polynomial_proof.json
```

Copy that JSON immediately to `output/support_geometry/conv_exact_fusion/`.
This is a proof oracle, not a production accelerator or a native bitwise
certificate. Successful eighth-cell certificates do not establish useful
amortization at 8x; report interval size as well as acceptance coverage.


The admissibility/transition-band study is documented in
`experiments/conv_admission_band/FINDINGS.md`. Run its tests and native census
on the Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.conv_admission_band.test_refinement
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_admission_band.run_study \
  --out /tmp/conv_admission_band_full --repeats 21
```

Copy `/tmp/conv_admission_band_full/` immediately into
`output/support_geometry/conv_admission_band/`. Additional jet and stability
probes are listed in the findings. The certificate and wider-jet variants
are diagnostics, not accepted replacements: the former does not tighten the
measured carrier transition, while the latter has mixed natural-image
results and larger thin-structure excursions.

The **rejected** budgeted mode-catalog experiment is `experiments.conv_budget_modes.resize` (also
`conv_budget_resize`). It uses direct scalar characteristic modes and the
unchanged `convstar_resize` fallback, including multichannel input. The new
mode applies to scalar images; it has no solver or iterative admission.
Run its mathematical checks and complete end-to-end budget gate on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m unittest experiments.test_conv_budget_modes \
  experiments.test_conv_joint_characteristic_modes experiments.test_conv_joint_finite

/Users/ultimussecundai/.local/bin/m4build -- env \
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m experiments.validate_conv_budget_modes \
  --sizes 9,13,17,33,65,129 --scales 2,3,6 --repeats 15 \
  --out /tmp/conv_budget_gate.json
```

Copy the JSON immediately to
`output/support_geometry/conv_joint_characteristic_modes/budget_gate.json`.
See `experiments/CONV_BUDGET_REPLACEMENT.md` for the measured cost envelope,
coverage, and remaining gradient-error tradeoff. The 1.5x budget is an
empirical gate against the same Python/NumPy CONV implementation, not a
universal wall-clock guarantee or a native-kernel claim.

CONV design constraint from the user: **we do not do solvers**. Do not propose
iterative optimization or a linear-system solve as the CONV construction.
The archived joint-potential QP results are diagnostic evidence only, not an
accepted operator. Pursue direct algebraic representations and finite admission
rules; do not relax the original constraints to make a finite baseline pass.

The characteristic-mode follow-up builds direct C2 potentials `F = phi(G)`
and certifies their actual current, including zero-amplitude phase ranges:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.test_conv_joint_characteristic_modes experiments.test_conv_joint_finite
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_joint_characteristic_modes
```

Copy `/tmp/conv_joint_characteristic_modes/` immediately into
`output/support_geometry/conv_joint_characteristic_modes/`. Render the saved
results locally with `.venv-jpeg/bin/python -m experiments.plot_conv_joint_characteristic_modes`.
This finite mode catalog recognizes some monotone straight/radial geometry;
it rejects unsupported cases and is not a universal resizer. Report acceptance
coverage alongside accuracy. See `experiments/CONV_JOINT_CHARACTERISTIC_MODES.md`.

The direct shared-jet capacity experiment runs with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.test_conv_joint_finite
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_joint_finite --out /tmp/conv_joint_finite.json
```

Copy the JSON immediately to `output/support_geometry/conv_joint_finite/`.
`--grouping node`, `order`, and `component` test finite shared-jet allocations.
This is an incomplete construction: the zero-jet baseline can violate the
original directional cones, and those cases must be rejected, not repaired by
a solver or presented as successful admission.

The joint-potential follow-up uses one source-defined tensor-quintic field,
structurally shared derivative traces, and a sparse gradient-metric solve.
Run its invariants and the fixed natural-boundary matched/validation battery:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.test_conv_joint_potential experiments.test_conv_compatible_current experiments.test_conv_functional_current
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.validate_conv_joint_potential
```

Copy `/tmp/conv_joint_final/` immediately into
`output/support_geometry/conv_joint_potential/final/`. The separate certificate
and continuity ablations run with `python3 -m experiments.study_conv_joint_potential`
through `m4build`; copy `/tmp/conv_joint_study/` into that output folder's
`study/` subdirectory. The first archived study used explicit C1 equalities;
the current implementation eliminates them structurally, with the same
mathematical objective and feasible set. Render saved results locally with
`.venv-jpeg/bin/python -m experiments.plot_conv_joint_potential`.
See `experiments/CONV_JOINT_POTENTIAL_FINDINGS.md` for numerical tolerances,
scope, and the distinction between range control and complete topology control.

The compatible-current follow-up adds a global quintic proposal and a
continuous excursion envelope. Run the full invariant suite and probes with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.test_conv_functional_current experiments.test_conv_compatible_current
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_compatible_current --out /tmp/conv_compatible_discovery_v3.json
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_compatible_current --validation --out /tmp/conv_compatible_validation_v3.json
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_compatible_current --joint-probe --out /tmp/conv_compatible_joint.json
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.probe_conv_compatible_2d --out /tmp/conv_compatible_2d.json
```

Copy each JSON immediately into `output/support_geometry/conv_compatible_current/`.
Render the saved discovery/validation results locally with:

```sh
.venv-jpeg/bin/python -m experiments.plot_conv_compatible_current \
  output/support_geometry/conv_compatible_current
```

Run the functional-current projection tests and analytic slope battery on
the M4 Mini system Python (NumPy/SciPy):

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.test_conv_functional_current

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.conv_functional_current --out /tmp/conv_functional_current
```

Copy `/tmp/conv_functional_current/results.json` back immediately using the
host selected by `m4host`. Render the saved results locally with:

```sh
.venv-jpeg/bin/python -m experiments.conv_functional_current --plot-only \
  --out output/support_geometry/conv_functional_current
```

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

## Periodic N-D commuting-chart study

Run the focused periodic N-D diagnosis and its structural tests on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/periodic_nd_study.py \
  --out /tmp/periodic_nd_study --steps 2000 --seeds 3

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m unittest ML_experiment.test_periodic_nd_study
```

Copy the generated `/tmp/periodic_nd_study` artifacts back immediately before
another mirror sync. The measured interpretation and the nonperiodic
commuting-chart construction are documented in
`ML_experiment/PERIODIC_ND_DIAGNOSIS.md`.

## Sparse-observation sine geometry study

Run the progressively thinned 1-D sine acquisition study and its focused tests
on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 ML_experiment/sparse_sine_study.py \
  --out /tmp/sparse_sine_study --steps 1000 --seeds 3

/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m unittest ML_experiment.test_sparse_sine_study
```

Copy the `/tmp/sparse_sine_study` artifacts back before another mirror sync.
The task, acquisition diagnosis, and measured separation between observed-tail
recovery and unsupported extrapolation are documented in
`ML_experiment/SPARSE_SINE_GEOMETRY.md`.

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

## Native retained photonic transport

Build the compiled retained-block plan, run its C++ and Python equivalence
tests, and dogfood the packed field on the M4 Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  'make -B -C experiments/photonic_transport_field/native retained-test retained && \
   python3 -m unittest discover -s experiments/photonic_transport_field -t . -p "test_*.py" -q && \
   python3 -m experiments.photonic_transport_field.run_native_retained_experiment \
     --repeats 200 --out /tmp/native_retained_scenes_m4.json \
     --image /tmp/native_retained_scenes_m4.png'
```

Copy both `/tmp/native_retained_scenes_m4` artifacts back immediately after
the run. The native library is built inside the mirrored tree and must not be
treated as an authoritative source artifact.

Render the established 3-D room through its integrated retained field with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  sh -c 'make -B -C experiments/photonic_transport_field/native regime_scene_native && \
  experiments/photonic_transport_field/native/regime_scene_native \
  --width 1920 --height 1280 --terminal-error 1 \
  --transport-backend retained --out /tmp/regime_integrated_retained_1920.ppm \
  > /tmp/regime_integrated_retained_1920.json'
```

Copy both `/tmp/regime_integrated_retained_1920` artifacts back immediately.

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
  denoiser.test_causal_information_lineage_2d \
  denoiser.test_causal_scale_transport_2d

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

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_compressed_eikonal_observer_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_compressed_eikonal_observer_2d \
  --size 32 --out /tmp/compressed_eikonal_phase_union_residual_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_causal_scale_transport_2d \
  --size 32 --out /tmp/causal_scale_transport_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_scale_retention_audit_2d \
  --size 32 --out /tmp/scale_retention_audit_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_scale_retention_visual_2d \
  --size 96 --out /tmp/scale_retention_visual_96.png

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_residual_erosion_transport_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_residual_erosion_transport_2d \
  --size 32 --seeds 1 --out /tmp/residual_erosion_transport_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_conservative_exchange_transport_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_conservative_exchange_transport_2d \
  --size 32 --numerical-cycle-ceiling 3 --all-corruptions \
  --out /tmp/conservative_exchange_transport_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_exchange_transfer_laws_2d \
  --size 32 --cycles 3 --out /tmp/exchange_transfer_laws_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_zonotopic_edge_flux_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_zonotopic_edge_flux_2d \
  --size 32 --out /tmp/zonotopic_edge_flux_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_continuous_scale_zonotope_transport_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_continuous_scale_zonotope_transport_2d \
  --size 32 --out /tmp/continuous_scale_zonotope_transport_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_continuous_scale_edge_family_transport_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_continuous_scale_edge_family_transport_2d \
  --size 20 --out /tmp/continuous_scale_edge_family_transport_20.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_joint_value_jet_zonotope_contractor_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_joint_value_jet_zonotope_contractor_2d \
  --size 20 --out /tmp/joint_value_jet_zonotope_contractor_20.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_lifted_scale_moment_transport_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_lifted_scale_moment_transport_2d \
  --size 20 --out /tmp/lifted_scale_moment_transport_20.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_lifted_endpoint_action_transport_2d

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_gui_evaluation

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_dcnt

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_transport_chambolle

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_selling_chambolle \
  denoiser.test_hodge_zonotope_chambolle

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_transport_chambolle \
  --size 32 --out /tmp/transport_chambolle_32.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_dcnt \
  --size 32 --cycles 3 --out /tmp/dcnt_first_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_lifted_endpoint_action_transport_2d \
  --size 20 --out /tmp/lifted_endpoint_action_transport_20.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_temporal_action_barycentres_2d \
  --size 20 --out /tmp/temporal_action_barycentres_20.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_temporal_action_barycentres_2d \
  --size 32 --out /tmp/temporal_action_barycentres_32.json
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

## Anchor optimizer battery

Run the paired 24-task, two-architecture, three-seed Anchor battery on the M4
Mini with its system Python, which already provides CPU Torch and NumPy:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m ML_experiment.run_benchmark \
  --out /tmp/anchor_battery_24x2x3 --widths 24 --seeds 3 \
  --steps 500 --batch 256 --eval-every 5 \
  --variants ordinary_mlp,self_context \
  --optimizers adamw,sgd,anchor \
  --optimizer-lrs adamw=0.003,sgd=0.03,anchor=1.0
```

Copy `results.json` back immediately, then run
`python3 -m ML_experiment.analyze_anchor_battery` locally to regenerate the
paired acquisition and iteration-efficiency summary.

## DIP packet-gauge audit

Generate isolated variants from the authoritative DIP header, compile on the
M4 Mini, validate with ASan/UBSan, and run the complete-transform and cell
experiments with:

```sh
experiments/dip_packet_gauge/run.sh
```

The script uses `m4build` and `m4host`, copies its `/tmp` artifacts back, and
regenerates `experiments/dip_packet_gauge/RESULTS.md`. The source generators
leave the production kernels unchanged. Avoid treating timings under heavy
concurrent Mini load as a kernel-promotion gate.

## Meyer transport novelty and acceleration audit

Build the isolated native current-state relaxation runner, check its complete
state against the NumPy oracle, run the time/objective-gap comparison and
parameter stress screen, and copy the records back with:

```sh
sh experiments/meyer_transport_audit/run.sh
```

This uses the existing complete `build/libbfft.so` in the Mini mirror for
the Fourier ABI, compiles the authoritative Meyer header into an experimental
wrapper, and leaves the production API unchanged. The first September 4 timings were collected under competing CPU load.
The later freed-CPU rerun is recorded in meyer_clean256.json; consult
DYNAMIC_FACTORS.md and the revised paper for the current interpretation.

## Mechanical FFT HTML simulator

The standalone eight-sample simulator has a dependency-free JavaScript model.
Run its short exact-geometry and Fourier checks locally, then rebuild its
single-file HTML bundle:

```sh
node experiments/mechanical_cone_fft/simulator/test_mechanics.cjs
python3 experiments/mechanical_cone_fft/simulator/build.py
```

Run the original NumPy mechanical certificate on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/mechanical_cone_fft/certificate.py
```

## Meyer transport dynamic-factor investigation

Run the directional-certificate tests and the initial two-resolution screen
on the M4 Mini CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.meyer_transport_audit.test_dynamic_factors -v

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/dynamic_factors.py \
  --size 64 --steps 128 --out /tmp/meyer_dynamic64.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/dynamic_factors.py \
  --size 128 --steps 128 --out /tmp/meyer_dynamic128.json
```

Copy each /tmp JSON back immediately. These are exploratory identical-backend
NumPy comparisons, not production speed claims. The local acceptance contract
and measured trajectory/cost failures are in
experiments/meyer_transport_audit/DYNAMIC_FACTORS.md.

## Meyer terminal transport geometry

The factor-free terminal-condition diagnostic and Poisson capacity certificate
run on the Mini CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.meyer_transport_audit.test_terminal_geometry -v

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/terminal_geometry.py \
  --size 32 --out /tmp/meyer_terminal_geometry32.json
```

Copy the /tmp JSON back immediately. The certified special regime, failed
terminal reconstruction, and unresolved general problem are documented in
experiments/meyer_transport_audit/TERMINAL_GEOMETRY.md.

## Meyer intermediate-state investigation

Run the within-pass identity, Hodge, and affine-pattern tests plus the
phase-resolved screen on the M4 CPU:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.meyer_transport_audit.test_intermediate_state -v

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/intermediate_state.py \
  --size 64 --out /tmp/meyer_intermediate64.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/intermediate_chart.py \
  --size 64 --out /tmp/meyer_intermediate_chart64.json
```

Copy each /tmp JSON back immediately. The dense one-dimensional chart solve
is a structural oracle, not a production timing result. Derivations and
measured limitations are in experiments/meyer_transport_audit/INTERMEDIATE_STATE.md.
Render the saved records locally with .venv-jpeg/bin/python
experiments/meyer_transport_audit/plot_intermediate.py.

## Coupled transport-of-transport investigation

Run the exact finite-increment lift and carried-response tests on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m unittest \
  experiments.meyer_transport_audit.test_finite_transport_lift \
  experiments.meyer_transport_audit.test_transport_of_transport -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/probe_finite_transport_lift.py \
  --size 64 --out /tmp/meyer_finite_lift64.json

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/transport_of_transport.py \
  --size 64 --repeats 3 --out /tmp/meyer_carried64_full_memory.json
```

Copy each /tmp JSON back immediately. The exact lift has no demonstrated
speedup; the reduced-response variants are negative performance evidence.
Read experiments/meyer_transport_audit/TRANSPORT_OF_TRANSPORT.md for the
coupled finite-transfer identities and scope of each test.


## Diagonal packet walk audit

Run the exact packet/carry tests and scalar schedule experiment on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/diagonal_packet_walk/test_packet.py

/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  'clang++ -O3 -std=c++17 -fno-vectorize -fno-slp-vectorize \
   experiments/diagonal_packet_walk/native.cpp -o /tmp/diagonal_packet_native && \
   /tmp/diagonal_packet_native 8 > /tmp/diagonal_packet_fused.json'
```

Copy the JSON back immediately using the host selected by `m4host`, then run
`python3 experiments/diagonal_packet_walk/audit_counts.py PATH_TO_JSON` locally.
The exact family, carry growth, cancellation identities, and movement model
are in `experiments/diagonal_packet_walk/README.md`. Scalar schedule timings
are not production real-input SIMD BFFT benchmarks.

The explicit NEON follow-up, independent exported-output oracle, and assembly
checks are reproduced by the command in
`experiments/diagonal_packet_walk/NEON_LEDGER.md`. Compile with `-DPACKET_NEON`;
copy `/tmp/diagonal_packet_neon_final.json` and the oracle/assembly JSON files
back immediately. The assembly audit is specific to Apple-clang AArch64 output.

## Short Meyer transport observation

Record the ordinary six-field recurrence on a periodic 64-sample edge:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/watch_edge.py \
  --out /tmp/meyer_watch_edge
```

Copy both `/tmp/meyer_watch_edge.json` and `/tmp/meyer_watch_edge.npz` back
immediately using `m4host`. The measured shoulder release and the distinction
between steady transport and convergence are documented in
`experiments/meyer_transport_audit/WATCH_EDGE.md`.

## Formal synthetic Meyer transport dynamics

Run the discrete-jet, invariant-drift, and first-release diagnostic with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/formal_edge.py \
  --out /tmp/meyer_formal_edge.json
```

Copy the JSON back immediately using `m4host`. Exact identities, admission
conditions, numerical checks and limitations are documented in
`experiments/meyer_transport_audit/FORMAL_EDGE_DYNAMICS.md`.


## Arbitrary real Fourier chart traversal

Run the exact partition-graph and moving-chart certificates on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_atlas/run.py \
  --out /tmp/real_fourier_atlas.json
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_atlas/test_moving.py \
  --out /tmp/real_fourier_moving.json
```

Copy both JSON files back immediately using the route selected by `m4host`.
The existing Mini SymPy installation provides exact real algebraic arithmetic.
These are state-space/composition certificates, not timing benchmarks.
The transition law and collision/jet conditions are in
`experiments/real_fourier_atlas/README.md`.

## Meyer compatibility-defect relaxation

Run the invariant suite and retained-memory validation on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.meyer_transport_audit.test_defect_relaxation -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/validate_defect_relaxation.py \
  --size 128 --steps 1024 --seed 3029 \
  --methods ordinary,refresh,aligned_refresh,remembered_refresh \
  --out /tmp/meyer_defect_validation128_cached.json

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/probe_defect_edge.py \
  --out /tmp/meyer_defect_edge_tail.json
```

Copy each /tmp JSON back immediately with m4host. The exact memory updates,
fixed-point argument, failed controls, time-to-certificate results and
regressions are in `experiments/meyer_transport_audit/DEFECT_RELAXATION.md`.


## Local real Fourier walk and bounded quartic FFT

Run the exact local-exchange checks, real FFT controls, and quartic candidate
on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_walk/run.py \
  --out /tmp/real_fourier_walk.json
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_walk/test_quartic.py \
  --out /tmp/real_quartic_fft.json
```

Copy both JSON files back immediately using the host selected by `m4host`.
The unrestricted Newton walk has intentionally retained conditioning failures;
its quadratic operation count must not be presented as a fast FFT. The bounded
quartic candidate has O(N log N) real arithmetic and selectable regrouping,
but native speed and novelty are not established. Read
`experiments/real_fourier_walk/QUARTIC_CANDIDATE.md` for the actual construction.

## Ramp refresh regression diagnosis

Run branch/stage interventions, local feedback spectra, and invariants:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/diagnose_ramp_refresh.py \
  --out /tmp/meyer_ramp_diagnosis.json

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/probe_ramp_ringing.py \
  --out /tmp/meyer_ramp_ringing.json

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.meyer_transport_audit.test_ramp_diagnosis -v
```

Copy each JSON back immediately using m4host. The dense derivative is only
an explanatory diagnostic, not an added runtime solver. The certificate
phase correction and causal interventions are documented in
`experiments/meyer_transport_audit/RAMP_REFRESH_DIAGNOSIS.md`. Plot copied
records locally with `.venv-jpeg/bin/python experiments/meyer_transport_audit/plot_ramp_diagnosis.py`.

## Normalized quartic real FFT NEON experiment

Run the exact factorization certificate, ASan/UBSan and optimized native checks,
and the existing BFFT benchmark with the experimental comparison enabled:

```sh
experiments/real_fourier_walk/run_native.sh
```

The script uses m4build/m4host, compiles the current production DIF and the
experimental kernel together, then immediately copies all /tmp artifacts into
`experiments/real_fourier_walk/native_m4/`. It regenerates `NATIVE_RESULTS.md`.
See `NATIVE_FACTORIZATION.md` for the native real representation, limitations
of the Bruun-equivalent factorization, and complete-transform timing contract.

## Unrestricted real Fourier coordinate walks

Run the synthesis, turnover, frame-gauge, and exact-program tests on the M4:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest discover -s experiments/fourier_free_walk -p "test_*.py" -v

experiments/fourier_free_walk/run.sh
```

The script regenerates the finite searches and copies /tmp results back into
`experiments/fourier_free_walk/reproduced/`. See that experiment's README for
which freedoms are exercised, explicit numerical admission limits, and the
distinction between actual placement reduction and remaining exchange cells.
No newly discovered FFT family or production speed result is established.

## Retained raytracer scene updates

Run the integrated geometry, signed residual, source-response, finite emitter
window, image-equivalence, and large geometry update checks on the M4 Mini:

```sh
sh experiments/photonic_transport_field/run_retained_updates.sh
```

This uses m4build/m4host and places executables in /tmp so another task's mirror
sync cannot replace them with old local binaries between compilation and tests.
It copies results to experiments/photonic_transport_field/retained_updates_m4/.
See RETAINED_SCENE_UPDATES.md in that experiment for the current math and limits.
The CPU 30 fps waterfall/grass/flying-observer goal at a minimum of 800×600
remains an end-to-end gate;
geometry-only timings and encoded animation frame rates do not satisfy it.

## Retained camera demand gathering

Run the exact eager/demand camera comparison at 800×600, four observer poses,
three additional scenes, geometry-update integration, and concurrent sample
publication checks on the M4 CPU:

```sh
sh experiments/photonic_transport_field/run_retained_camera.sh
```

The script builds to /tmp through m4build/m4host, runs ASan/UBSan and TSan, and
copies JSON records into experiments/photonic_transport_field/camera_gather_m4/.
See CAMERA_DEMAND_GATHER.md in that experiment. Retain the eager atlas and dense
crossing modes as numerical oracles. Compare alternating runs at the same
resolution and report end-to-end timing separately from camera-only timing.

## Fourier arithmetic by twos

Run the exact guarded Fermat-ring lift, compact cyclotomic packet, inverse,
and phase-shift checks on the selected M4 Mini:

```sh
sh experiments/fourier_twos/run.sh
```

The script copies /tmp test and experiment records back immediately.
See experiments/fourier_twos/README.md for the exact algebraic output contract,
range guards, and explicitly separate numeric projection. These representation
timings are not an end-to-end BFFT speed comparison. The earlier Fourier walk
search was set aside at the user's request; this does not extend that search.

## Boundary information reuse

Run the shared-filter and exact optical-query invariants, alternating four-mode
800×600 benchmark, integrated updates, and ASan/UBSan/TSan checks on the M4 CPU:

```sh
sh experiments/photonic_transport_field/run_retained_boundary.sh
```

Results are copied immediately to experiments/photonic_transport_field/boundary_discovery_m4/.
See BOUNDARY_INFORMATION_REUSE.md. The benchmark records traversal policy:
small command-line scenes use linear traversal under auto, while earlier camera
and retained-update helpers used Scene's default BVH. Compare matching policies.
Boundary stage timings include shared-field construction; worker-time audit
sums are not frame wall times. The exact query cache belongs to one immutable
scene/render pass and must not survive geometry changes without invalidation.

## Retained camera source expansion audit

Run the per-pixel optical/source expansion diagnosis, exact source-support
checks, and paired 800×600 camera/compilation benchmark on the selected M4 CPU:

```sh
experiments/photonic_transport_field/run_source_expansion.sh
```

The script uses `m4build`, puts executable and results in `/tmp`, and copies JSON
and path-word diagnostics back immediately. `--source-cones angles
--source-zero-elision off` selects the unchanged source evaluator as an oracle.
The default uses algebraic source-cone overlap and verified opaque-suffix zero
elision. Optical cutoffs, spectral partitions, quadrature nodes, coverage,
and geometry stay unchanged. See `experiments/photonic_transport_field/SOURCE_EXPANSION.md`
for the nested-work diagnosis and the rejected cutoff-topology experiment.

## Native return-extinction screen

The return-extinction mode is rejected as an expensive-pixel accelerator and
remains off by default. Findings: `experiments/photonic_transport_field/RETURN_EXTINCTION.md`.
Run the selected-M4 build, fixture tests, 48-frame screen, and immediate result
copy with `sh experiments/photonic_transport_field/run_return_extinction.sh`.
Run focused checks with `/Users/ultimussecundai/.local/bin/m4build -- make -C experiments/photonic_transport_field/native return-test`.

## Native expensive-pixel redundancy census

Run `sh experiments/photonic_transport_field/run_pixel_redundancy.sh` to compile
and measure the two worst aperture pixels on the selected M4 host and copy the
JSON back immediately. Findings: `experiments/photonic_transport_field/PIXEL_REDUNDANCY.md`.
The compile-time census is not a promoted renderer optimization.

## Randomized defect-gate decisions

Keep the existing gated candidates fixed and randomize only the alignment
threshold. Run the checks and seed screen on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.meyer_transport_audit.test_noisy_defect_gate \
  experiments.meyer_transport_audit.test_defect_relaxation -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/noisy_defect_gate.py \
  --out /tmp/meyer_noisy_defect_gate128.json

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/probe_noisy_gate_decisions.py \
  --out /tmp/meyer_noisy_gate_census.json
```

Copy each JSON back immediately using m4host. The decision law, negative
speed result, and distinction between shared and local noise are documented
in `experiments/meyer_transport_audit/NOISY_DEFECT_GATE.md`.

## Native child-boundary diagnostic

The owned prism/sheet/jelly boundary route is opt-in (`--child-boundaries on`).
It is not promoted: exact-state unit-response retention exceeds 1 GB on the
800x600 aperture-canyon frame and does not fix its source-integration cost.
See `experiments/photonic_transport_field/NATIVE_CHILD_BOUNDARY.md`. Preserve the
native default `off` until the compactness and full-pipeline gate is met.

Run the focused tests, existing regressions, and 24-frame M4 comparison with:

```sh
experiments/photonic_transport_field/run_child_boundary.sh
```

The script uses `m4build` and immediately copies the final `/tmp` artifacts to
`experiments/photonic_transport_field/child_boundary_m4/`. Render the saved
comparison locally with:

```sh
.venv-jpeg/bin/python -m experiments.photonic_transport_field.plot_child_boundary
```

### Shared child-law observer registration

The child route now defaults to `--child-response shared`. It registers three
child laws once and never inserts observation query states. The old large cache
is available only with `--child-response exact` for comparison. The earlier
compactness failure above describes that archived backend. The broader child
optical route remains opt-in; its optics differ from the legacy renderer.
See `experiments/photonic_transport_field/OBSERVER_REGISTRATION.md`.

```sh
experiments/photonic_transport_field/run_observer_registration.sh
```

This runs on the M4 through `m4build` and copies the 24-frame matched screen back
to `experiments/photonic_transport_field/observer_registration_m4/`. Report law
registrations, evaluations, and internal intersection steps separately. Fewer
registrations must never be presented as one radiance calculation per object.

## Direct spectral transport comparison

Run the unconditional three-band experiment, four-scene frame comparisons,
denser aperture reference, and fixed-ray classification checks on the Mini:

```sh
experiments/photonic_transport_field/run_direct_spectral.sh
```

The wrapper uses `m4build` and copies every run immediately into
`experiments/photonic_transport_field/direct_spectral_m4/`. Plot saved results
locally with `.venv-jpeg/bin/python experiments/photonic_transport_field/plot_direct_spectral.py`.
See `experiments/photonic_transport_field/DIRECT_SPECTRAL_TRANSPORT.md` for the
measured speed/error tradeoff. This is finite unconditional band quadrature,
not an exact continuous spectral representation or a promoted default.

The N=1 follow-up uses `experiments/photonic_transport_field/run_one_per_band.sh`
for three alternating repeats across all four scenes. Plot locally with
`.venv-jpeg/bin/python experiments/photonic_transport_field/plot_one_per_band.py`.
This deliberately uses one representative wavelength per band and does not
claim within-band integration.

## Retained planar reflected-view source field

Run `experiments/photonic_transport_field/run_reflected_gather.sh` on the Mini
via its `m4build` wrapper. It compares one-per-band rendering with and without
retained virtual-eye source-response fields for rectangular mirrors, checks the
reflection identity, deterministic reuse, and stale illumination rejection,
and immediately copies each scene's artifacts. Plot saved output locally with
`.venv-jpeg/bin/python experiments/photonic_transport_field/plot_reflected_gather.py`.
See `experiments/photonic_transport_field/REFLECTED_CAMERA_GATHER.md`: report
first-frame construction and warm gathers separately. This is a scoped prototype,
not an assertion that the entire directional/refraction field is retained.

## Frozen city rain / dynamic observer-volume normals

Run the static bake, structural checks, direct-reference diagnostic, and
60 fps recording with:

```sh
experiments/photonic_transport_field/city_rain/run_m4.sh final
```

The wrapper uses `m4build` and the Mini's existing `/opt/homebrew/bin/ffmpeg`.
It records 1280×720 for 18 seconds from a read-only position/normal response
field, with no city object constructed during playback. It copies results and
streams the compressed response cache home before removing its own large
uncompressed `/tmp` field. `KEEP_CITY_RAIN_FIELD=1` retains that file. Cache
archives live in `city_rain/build/`, excluded from mirror sync and Git.

Render saved diagnostics locally with
`.venv-jpeg/bin/python experiments/photonic_transport_field/city_rain/summarize.py`.
See `experiments/photonic_transport_field/city_rain/README.md` for the optical
model, field-domain invalidation, 3.33 GB dense field cost, and measured errors
at visibility boundaries. This is the deliberately frozen-city/normal-only
animation case, not general dynamic-scene invalidation.

## Night city / pinhole illumination screen

Build, test, capture and record the expanded night city on the selected M4 with:

```sh
experiments/photonic_transport_field/city_rain/run_night_screen.sh
```

The independent `night_screen_native play` mode constructs no scene and animates
only rain-driven screen deformation. It uses one captured pinhole image and a
frozen mip pyramid, with integrity checks before and after playback. All output,
including the replayable screen, is copied from `/tmp/night_screen/` immediately.
This replaces the earlier 4-D response only within this new experiment; it
cannot reconstruct disoccluded geometry absent from the pinhole image. See
`experiments/photonic_transport_field/city_rain/NIGHT_SCREEN.md` for the exact
mapping, optical scope, filtering and measured costs. Render saved diagnostics
locally with `.venv-jpeg/bin/python experiments/photonic_transport_field/city_rain/summarize_night_screen.py`.

The night city's opaque source-extinction comparison and full accelerated bake
run with `experiments/photonic_transport_field/city_rain/run_source_extinction.sh`.
It compares 2,177 city queries and the complete transport/irradiance fields of
four native scenes with extinction on and off. The night `bake` mode enables the
new source-chart zero certificates; `bake-legacy` retains the original path.
Results are copied from `/tmp/night_screen_extinction/` immediately. See
`experiments/photonic_transport_field/city_rain/SOURCE_EXTINCTION.md`.

## Earth-contact home architectural scene

The editable first model and native architecture lighting adapter are in
`experiments/photonic_transport_field/earth_home/`. Build and validate geometry
locally with `python3 experiments/photonic_transport_field/earth_home/build_model.py`
and `python3 experiments/photonic_transport_field/earth_home/validate_model.py`.
Build the CPU renderer on the Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c 'make -C experiments/photonic_transport_field/native conv_native.o && c++ -std=c++20 -O2 -mcpu=native experiments/photonic_transport_field/earth_home/render_home.cpp experiments/photonic_transport_field/native/retained_transport.cpp experiments/photonic_transport_field/native/conv_native.o -pthread -o /tmp/earth_home_render'
```

Run `/tmp/earth_home_render SCENE.bin OUTPUT.ppm VIEW WIDTH` through SSH to the
host returned by `m4host`. Views: site, exterior, plan, commons, greenhouse,
dome, services. Copy `/tmp` output back immediately. This reuses native BFFT
scene/BVH/intersections with a direct sky/sun adapter; do not characterize its
images or timings as retained-field global illumination or calibrated daylight.
See the scene README for assumptions and the offline inspection viewer.

## Conifer / registered Fourier smoke

Build and test the isolated one-tree, one-light smoke scene and run its
independent no-scene replay on the selected M4 host with:

```sh
experiments/photonic_transport_field/conifer_projectors/run_m4.sh
```

The wrapper uses `m4build` and `m4host` and copies `/tmp/conifer_projectors/`
back immediately. Render the saved diagnostics locally with:

```sh
.venv-jpeg/bin/python experiments/photonic_transport_field/conifer_projectors/summarize.py
```

The projector/direct test isolates source transfer reuse; it shares the diffuse
registration and does not certify full GI. Report the separate diffuse error,
quadrature residual, capture cost and memory alongside replay performance.
`play` must construct no scene and perform no geometry or density queries.
See the experiment README for the diffuse-identity termination contract,
Fourier/positive-angular model, cache invalidation and viewpoint support limits.

## Generic Blender scene import

The geometry authority is the user's `.blend`, outside renderer code. Do not
rebuild the historical home generator when rendering Blender edits. Export
actual evaluated Blender state with
`experiments/photonic_transport_field/blender_import/export_blender.py`.
Build and test the generic native scene consumer on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  'make -C experiments/photonic_transport_field/blender_import && python3 experiments/photonic_transport_field/blender_import/test_pipeline.py -v'
```

Use `blender_import/render_m4.sh SNAPSHOT.bffts OUTPUT.ppm CAMERA WIDTH`
(relative to `experiments/photonic_transport_field/`) for an existing export.
It copies the generated render and inspection receipts back immediately.
The Blender round-trip fixture test additionally requires
`BFFT_BLENDER_FIXTURE` pointing to actual exports made by
`create_blender_fixture.py`; a skipped test is not Blender validation.
See that directory's README for supported materials, camera/normal/units
contract and the direct-light preview renderer's limitations.

## Meyer pre-refresh transport contracts

Run the exact phase, nonlinear-remainder, transverse-memory, and finite-secant
oracle tests on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- python3 -m unittest \
  experiments.meyer_transport_audit.test_intermediate_state.IntermediateTests.test_within_pass_and_remainder_identities \
  experiments.meyer_transport_audit.test_intermediate_state.IntermediateTests.test_transverse_field_can_change_projected_divergence \
  experiments.meyer_transport_audit.test_finite_transport_lift
```

See `experiments/meyer_transport_audit/PRE_REFRESH_INTEGRATION.md` for scope.
These are mathematical oracle regressions, not native equivalence tests or
evidence for a new runtime speedup. Refresh, gating, and interval variants
remain experimental.

## Projectile motion / joint noise transport

Run the causal tossed-ball comparison and the independent Kalman-limit,
component-covariance, Jacobian, and missing-observation tests on the selected M4:

```sh
sh experiments/projectile_transport/run_m4.sh
```

The wrapper uses m4build/m4host and copies `/tmp/projectile_transport_first/`
back immediately into `experiments/projectile_transport/results/`. Render saved
records locally with `.venv-jpeg/bin/python -m experiments.projectile_transport.plot`.
See that experiment's README and FINDINGS for the fixed priors, mode catalog,
component-identifiability limitations, forecast protocol and measured outcomes.
This is an IMM Gaussian-mixture EKF experiment; it has no zonotopic enclosure
guarantee and is not a new universal nonlinear prediction theorem. Report true
future forecast error and calibration alongside current-state tracking error.

## Fixed downstream CONV comparison

Keep the six-source/five-current Version 1 construction fixed. The ledger,
projection metric, order blend, and basin formulas in `experiments/conv_downstream`
are isolated experiments; none is promoted. See its `FINDINGS.md` for exact
reformulations versus changed operators, quality, cost, and proof scope.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.conv_downstream.test_downstream -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_downstream.run_study \
  --out /tmp/conv_downstream4 --repeats 31

/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_downstream.timing \
  --out /tmp/conv_downstream1.json --repeats 31

/Users/ultimussecundai/.local/bin/m4build -- env \
  CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_downstream.probe_mixtures \
  --out /tmp/conv_downstream_mixtures.json
```

Copy those artifacts into `output/support_geometry/conv_downstream/` immediately
using the host selected by `m4host`. Render locally with
`.venv-jpeg/bin/python -m experiments.conv_downstream.report`.

## Civilian 3-D position-only geometric transport

The user rejected the earlier `projectile_transport` derivative-chain candidate
as a formulation of the requested transport geometry. Do not use its results
as fulfillment of that objective. The separate construction and first measured
comparison are in `experiments/civilian_transport/THEORY.md` and `FINDINGS.md`.

```sh
sh experiments/civilian_transport/run_m4.sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.civilian_transport.test_transport \
  experiments.civilian_transport.test_io -v
```

The wrapper uses m4build/m4host, snapshots authoritative source in an isolated
`/tmp` workspace before the benchmark, and copies `/tmp/civilian_transport_final/`
back immediately. Render locally with `.venv-jpeg/bin/python -m
experiments.civilian_transport.report`. Development selection and test seeds are
separate. Position, future forecast, component attribution, boundary Brier score,
calibration and cost must be reported together. This restricted geometric-action
particle filter has no universal forecasting or set-enclosure guarantee; its
first candidate is not promoted over the strongest comparators. The optional
particle-budget/noise-law probe is development-only, with its command in README.

## Civilian transport observation/noise degradation

Run the frozen matched noise and nested-observation sweep with proved synthetic
motion bounds on the Mini, copying the outputs home through the wrapper:

```sh
sh experiments/civilian_transport/run_degradation_m4.sh
```

It runs the 22 focused/existing checks, snapshots the authoritative source into
an isolated remote `/tmp` workspace, and copies results immediately into
`experiments/civilian_transport/degradation_results/`. Render locally with:

```sh
.venv-jpeg/bin/python -m experiments.civilian_transport.degradation_report
```

See `BOUND_EXAMINATION.md` for conditional coverage and the distinction between
synthetic parameter guarantees and unestablished real hardware/target bounds.

## Relational-current Kalman experiment

The first explicit current-prior implementation and its limits are documented in
`experiments/civilian_transport/RELATIONAL_FINDINGS.md`. The state retains a full
current covariance; the Gaussian proximal map is a structural special case,
not an established nonlinear BTB motion denoiser. Run the invariant tests and
fresh paired screen on the Mini, with immediate result copying, using:

```sh
sh experiments/civilian_transport/run_relational_m4.sh
```

Render retained results locally with:

```sh
.venv-jpeg/bin/python -m experiments.civilian_transport.relational_report
```

Do not treat Zak-coordinate equivalence as an accuracy gain, repeated refinement
as new evidence, or the slow proximal-iteration residual as a truth certificate.

## Local witness geometry for current corrections

BTB supplies an observation about local restoration; do not treat its iteration
as an implementation requirement. The finite independent-witness construction
and its continuation/sensor-identifiability limits are documented in
`experiments/civilian_transport/WITNESS_GEOMETRY.md` and `WITNESS_FINDINGS.md`.
Run its focused tests and retained exploration on the Mini, copying results
home immediately, with:

```sh
sh experiments/civilian_transport/run_witness_m4.sh
```

Render the retained research figures locally with:

```sh
.venv-jpeg/bin/python -m experiments/civilian_transport.witness_report
```

The witness certificate applies to clean error at witness times under a stated
noise contract. It is not a future-position guarantee or a detector that can
separate an unrestricted coherent sensor drift from identical physical motion.

## Boundary-triggered acquisition and integration tradeoff

The irregular acquisition study retains one current-Kalman state and varies
only sampling cadence after an estimated boundary crossing. Its first version
uses the ordinary covariance-weighted innovation as the shortening rule.
See `experiments/civilian_transport/ACQUISITION_PROTOCOL.md` and
`ACQUISITION_FINDINGS.md` for the measured cost and settling tradeoff.

```sh
sh experiments/civilian_transport/run_acquisition_m4.sh
.venv-jpeg/bin/python -m experiments.civilian_transport.acquisition_report
```

The runner uses the Mini and immediately copies retained results home. Report
both sample counts and CPU cost, with the post-trigger settling interval; do
not present the later accuracy as if it were attained at the instant of crossing.

## Exact civilian relational tracker optimization

The streaming tracker uses the exact four-state realization documented in
`experiments/civilian_transport/EXACT_OPTIMIZATION.md`. Preserve the original
finite-cell prior, five length branches, evidence, and acquisition decisions.
Historical smoothing/full coefficient exports lazily use the dense reference;
do not present their cost as constant-state filtering cost.

Run the Mini build, 23 focused regressions, 480 matched acquisition cases, and
31-repeat cost study with:

```sh
./experiments/civilian_transport/run_optimization_m4.sh
```

The script copies `/tmp` results back immediately. Render locally with:

```sh
.venv-jpeg/bin/python -m experiments.civilian_transport.optimization_report
```

Build the small optional local native runtime with
`make -C experiments/civilian_transport/native libmarkov.so`. Without it the public
tracker uses the same four-state algebra in NumPy. Generated libraries are not
source artifacts. The simple CV control is for cost comparison only.

## Wrench transport standalone engine

`experiments/wrench_transport/phys/` is the current C++20 engine. The adjacent
JavaScript implementation is the historical prototype, not the native baseline.
Preserve general convex and compound geometry: do not reshape objects to make
stacking demonstrations pass. Preserve carried-mass and retained-load scaling.
All native checks run on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  '/opt/homebrew/bin/cmake -S experiments/wrench_transport/phys -B /tmp/wrench_native -DCMAKE_BUILD_TYPE=Release && \
   /opt/homebrew/bin/cmake --build /tmp/wrench_native -j4 && \
   /opt/homebrew/bin/ctest --test-dir /tmp/wrench_native --output-on-failure && \
   /tmp/wrench_native/wrench_minimal'
```

Keep both the independent SAT/dense-system numerical checks and acceptance tests
passing before trusting benchmark changes. See `NATIVE_STUDY.md` for the frozen
baseline, matched timing protocol, discarded optimization, and reproduction.
Run `phys_bench scene.txt result.json 60 8 replay.json` to export real native
poses for a browser replay. A replay is not a browser port of the engine.
