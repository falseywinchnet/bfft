# ML experiment: self-context Eikonal acquisition

This folder is a self-contained checkpoint of the structure-learning study. It
contains the model, all 23 tasks, five additions to the self-context baseline,
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
23-problem suite. `build_problem_atlas.py` compacts the fitted probes into the
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
