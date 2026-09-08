# ML experiment: self-context Eikonal acquisition

This folder is a self-contained checkpoint of the structure-learning study. It
contains the model, all 24 tasks, five additions to the self-context baseline,
the exact-budget ordinary LELU MLP control, raw results, fitted probes, an
interactive visualization, and the research interpretation.

The comparison is deliberately strict. Within a task every model has exactly
the same number of trainable parameters. The control is an ordinary
encode-expand-LELU-contract-decode MLP. No model receives a task-specific
Fourier basis, periodic feature map, unseen-support label, or GELU activation.

## Start here

- [`IDEAS.md`](IDEAS.md): hypothesis, epistemic constraints, and the five
  experiments.
- [`REPORT.md`](REPORT.md): results and interpretation from 462 M4 CPU fits.
- [`CURVATURE_STATE_REPORT.md`](CURVATURE_STATE_REPORT.md): parameter-free
  curvature-state and nested-self-context follow-up (194 benchmark fits plus
  30 matched visual-probe fits).
- [`CURVATURE_SUPERSET_REPORT.md`](CURVATURE_SUPERSET_REPORT.md): direct
  self-context versus curvature-self-context comparison on all 22 problems.
- [`NESTED_CHART_CHECK.md`](NESTED_CHART_CHECK.md): rapid scratch-versus-staged
  nested-selection check on radial stripes and multiscale 1-D.
- [`TRANSPORT_STUDY.md`](TRANSPORT_STUDY.md): differential diagnosis and
  parameter-matched eikonal-ray transport experiments.
- [`RADIAL_DOGFOOD.md`](RADIAL_DOGFOOD.md): topology-first 500-step radial
  dogfood and the continuous full-space frame-flow result.
- [`CONTINUOUS_FRAME_FULL.md`](CONTINUOUS_FRAME_FULL.md): full 23-task matched
  comparison and the transport lessons from segmentation and Meyer splitting.
- [`FRAME_REFINEMENT_REPORT.md`](FRAME_REFINEMENT_REPORT.md): 84-fit optimizer,
  capacity, and speed study, including polynomial drifted chirp and the spatial
  3-D spiral.
- [`RESPONSE_ENHANCED_REPORT.md`](RESPONSE_ENHANCED_REPORT.md): 184-fit,
  two-seed comparison of original self-context, relational SCL, baseline CFF,
  and relational/deeper-response CFF, including CPU Pareto measurements.
- [`ANCHOR_BATTERY_REPORT.md`](ANCHOR_BATTERY_REPORT.md): 432-fit paired
  AdamW/SGD/Anchor iteration-efficiency battery, including the complete winner
  rule, aggregate result, and failure boundary.
- [`ANCHOR_FORMAL_DECOMPOSITION.md`](ANCHOR_FORMAL_DECOMPOSITION.md): exact
  recurrence, Euclidean/Bregman comparison, Meyer-transport boundary, scalar
  stability analysis, and falsifiable next experiments for Anchor.
- [`MUON_GEOMETRY_NOTE.md`](MUON_GEOMETRY_NOTE.md): faithful Muon recurrence,
  spectral/Shampoo equivalence, precise non-Bregman boundary, and the proposed
  restraint-shaped matrix momentum geometry.
- [`MUON_TRANSPORT_LEPTON_RESULT.md`](MUON_TRANSPORT_LEPTON_RESULT.md):
  Transport-Muon, smooth Bregman Lepton, transported dual-state Lepton, the
  60-run paired result, and the resulting mirror-shaped Anchor hypothesis.
- [`MUON_PRIMACY_PROBLEM.md`](MUON_PRIMACY_PROBLEM.md): exact one-step and
  rank-deficient block-sweep theorems for anisotropic operator recovery,
  numerical certificate, random-streaming boundary, and the resulting moving
  block research program.
- [`optimizer.py`](optimizer.py): standalone `ResidualPolarTransport`
  implementation with explicit training-residual observation, optional
  layer-local activation/cotangent hooks, reversible Fusion/polar control,
  cached matrix roots, and a copyable `Optimizer` alias.
- [`MATRIX_TRANSPORT_BREGMAN_TRANSLATION.md`](MATRIX_TRANSPORT_BREGMAN_TRANSLATION.md):
  the Fusion/Muon crossover-state decomposition, residual-operator anisotropy
  governor, and the first governed self-context result.
- [`RESIDUAL_POLAR_RIPPLE.md`](RESIDUAL_POLAR_RIPPLE.md): five-seed AdamW,
  Muon, fixed Matrix Transport, and `optimizer.py` Ripple stability test,
  including the live entropy-rank/polar-weight trace and wall-time boundary.
- [`OPTIMIZER_GEOMETRY_BATTERY_REPORT.md`](OPTIMIZER_GEOMETRY_BATTERY_REPORT.md):
  864-fit six-arm neural battery plus 72 operator-witness runs, the restrained
  slow-memory Anchor ablation, exact/practical Muon boundary, complete paired
  result, and all-problem validation/loss atlas.
- [`response_enhanced.py`](response_enhanced.py): the new relational SCL and
  response-enhanced CFF layers.
- [`response_enhanced.html`](response_enhanced.html): complete timing table,
  acquisition curves, task deltas, and four-way fitted-function atlas.
- [`frame_refinement.html`](frame_refinement.html): paired acquisition deltas
  and fitted geometry with an explicit observed/extrapolated boundary.
- [`complex_spiral_3d.html`](complex_spiral_3d.html): the supplied Matplotlib
  Axes3D/Turbo/depth/camera plot applied to truth and retained model fits.
- [`continuous_frame_flow.py`](continuous_frame_flow.py): standalone PyTorch
  reference/fast layer and hybrid CPU Muon helper.
- [`continuous_frame_full.html`](continuous_frame_full.html): complete truth,
  vanilla MLP, self-context, and continuous-frame-flow fitted-function atlas.
- [`radial_dogfood.html`](radial_dogfood.html): truth, fitted fields, radial
  profiles, and learning curves for the decisive mechanism sequence.
- [`transport_study.html`](transport_study.html): 11-task deltas, acquisition,
  chart-observation efficiency, mechanism ablations, and fitted probes.
- [`nested_chart_check.html`](nested_chart_check.html): graphical learning,
  endpoint, radial-field, and multiscale-continuation results for that check.
- [`visualization.html`](visualization.html): the complete 22-problem atlas,
  with truth, MLP, self-context, hard-gate, and chart-curvature fits.
- [`summary_visualization.html`](summary_visualization.html): aggregate score,
  tail-tradeoff, spiral, and multiscale summary plots.
- [`curvature_state.html`](curvature_state.html): matched confirmation metrics
  and by-eye fits for the curvature-state follow-up.
- [`curvature_superset.html`](curvature_superset.html): acquisition, endpoint,
  tail, and fitted-function differences across the complete suite.
- `results_confirm/`: 308 confirmation fits, paired-seed summary, and 88 fitted
  probes covering all 22 tasks and four representative models.
- `results_screen/`: 154 preliminary fits.

## Models

Seven parameter-identical variants are compared:

1. ordinary LELU MLP;
2. self-context Eikonal baseline;
3. harder continuous allocation;
4. a second anchored context refinement;
5. uncertainty-gated context injection;
6. output-secant supervision;
7. allocation-chart curvature regularization.

`models.py` contains LELU, the soft Eikonal layer, and the exact-budget MLP.
`variants.py` constructs the seven matched models. `tasks.py` contains the full
24-problem suite. `build_problem_atlas.py` compacts the fitted probes into the
responsive all-problem visualization.

## Reproduce on the M4 Mini CPU

Run from the repository root. The checkout on this Mac remains authoritative.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.test_experiment
```

Confirmation benchmark:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_benchmark \
  --out /tmp/ml_experiment_confirm --widths 36 --seeds 2 \
  --steps 600 --batch 256 --eval-every 25
```

Selected visual probes:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_probes \
  --out /tmp/ml_experiment_probes.json --width 36 --seed 0 --steps 600 \
  --variants ordinary_mlp,self_context,self_context_hard,self_context_chart
```

Curvature-state confirmation:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_benchmark \
  --out /tmp/curvature_state_confirm --widths 24 --seeds 2 \
  --steps 500 --batch 256 --eval-every 25 \
  --tasks spiral,checkerboard,nd_spiral_low_rank,nd_spiral_high_rank,radial_stripes,swiss_cheese,ripple,multiscale_1d,chirp_1d,localized_steps_1d,fourier_mix_1d \
  --variants ordinary_mlp,self_context,self_context_jet_factor,self_context_jet_curvature_context,self_context_nested
```

Radial-only dogfood winner:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_radial_dogfood \
  --out /tmp/radial_dogfood_stiefel_flow.json \
  --variants self_context_stiefel_flow_curvature \
  --width 24 --seed 0 --steps 500 --grid 81
```

Rebuild its visualization fragment after copying the JSON result back:

```sh
python3 -m ML_experiment.build_radial_dogfood
```

Full continuous-frame-flow benchmark:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_benchmark \
  --out /tmp/continuous_frame_full --widths 24 --seeds 2 \
  --steps 500 --batch 256 --eval-every 25 \
  --variants ordinary_mlp,self_context,self_context_stiefel_flow_curvature
```

Optimizer/capacity/speed refinement:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_frame_refinement \
  --out /tmp/frame_refinement_full --seeds 2 --steps 500
```

After copying those results back, regenerate its paired summary and report:

```sh
python3 -m ML_experiment.analyze_frame_refinement \
  ML_experiment/results_frame_refinement/results.json \
  --out ML_experiment/results_frame_refinement/summary.json
python3 -m ML_experiment.build_frame_refinement
```

After copying its results and probes back, rebuild the complete report with:

```sh
python3 -m ML_experiment.analyze \
  ML_experiment/results_continuous_frame_full/results.json \
  --out ML_experiment/results_continuous_frame_full/summary.json
python3 -m ML_experiment.build_continuous_frame_full
```

The corresponding compact visual is regenerated locally with:

```sh
python3 -m ML_experiment.build_curvature_state
```

`m4build` does not copy `/tmp` results back. Copy the result files immediately
after a remote run and before another mirror synchronization.

## Regenerate the analysis

```sh
python3 -m ML_experiment.analyze \
  ML_experiment/results_confirm/results.json \
  --out ML_experiment/results_confirm/summary.json
python3 -m ML_experiment.write_report
python3 -m ML_experiment.build_problem_atlas
```

The benchmark uses CPU Torch, AdamW, paired seeds, explicit held-out support,
mean class recall for classification, and normalized MSE-derived score for
regression. See `REPORT.md` for the boundaries on what the results establish.

## Anchor optimizer experiment

`anchor.py` contains the standalone transported lead--lag optimizer.  The
`anchor` factory uses a fixed maximum LR with a local dynamic trust controller;
`anchor_fixed` and `anchor_isotropic` retain the mechanism ablations.  The
derivation and pre-battery contrast screen are in
`../experiments/sgd_transport_restraint/ANCHOR.md`.

The completed three-optimizer battery is documented in
`ANCHOR_BATTERY_REPORT.md`.  Recompute its frozen summary with:

```sh
python3 -m ML_experiment.analyze_anchor_battery \
  ML_experiment/results_anchor_battery/results.json \
  --out ML_experiment/results_anchor_battery/summary.json
```

## Source-aware self-context optimizer battery

`context_backprop.py` decomposes the parameter cotangent into the derivative
with the self-context chart held fixed and the feedback caused by moving that
chart. `split_trust` applies the locally nonexpansive normalization derivative,
then bounds chart feedback once per parameter tensor without elementwise
clipping. The forward function is bit-identical across backward modes.

The four-arm battery compares Anchor, Transport-Lepton, AdamW, and Transported
AdamW on the same self-context model. `TransportedAdamW` replaces AdamW's
elementwise second moment with a transported full covariance inside each row;
it does not couple rows or apply Muon's matrix polar map.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 -m ML_experiment.run_benchmark \
  --out /tmp/cured_four_arm_full --variants self_context --widths 24 \
  --seeds 3 --steps 500 --batch 256 --eval-every 5 \
  --context-backward-mode nonexpansive \
  --context-gradient-mode split_trust \
  --optimizers anchor_cured,lepton_transport_cured,adamw,adamw_transport \
  --optimizer-lrs anchor_cured=1.0,lepton_transport_cured=.003,adamw=.003,adamw_transport=.003
```

After copying the M4 result back, rebuild the atlas with:

```sh
python3 ML_experiment/build_cured_optimizer_atlas.py \
  ML_experiment/results_cured_four_arm_full/results.json \
  --operator ML_experiment/results_transported_adam_operator_rms.json \
  --out ML_experiment/cured_optimizer_atlas.html
```

The mathematical and experimental notes are
`SELF_CONTEXT_BACKPROP_TRANSPORT.md` and `TRANSPORTED_ADAMW.md`.
