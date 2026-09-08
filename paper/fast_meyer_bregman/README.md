# Transport-state fusion for Meyer–Bregman decomposition

`main.tex` is the September 4 engineering-style revision. It leads with the
retained transport construction and exact spectral fusion, credits the author's
transport/superresolution origin separately from established numerical
machinery, and distinguishes finite-reference agreement from objective progress.
The original manuscript and PDF are preserved as `august_revision.tex` and
`august_revision.pdf`.

The new evidence table uses the clean CPU rerun in
`../../experiments/meyer_transport_audit/meyer_clean256.json`. This supersedes
loaded timing comparisons for current performance interpretation. At 32 passes,
fixed 1.75 improves finite-reference error but takes slightly longer than the
published quality preset. A fixed factor is only an experimental control.
The first dynamic-factor derivation, tests, cost measurements, and failure cases
are documented in `../../experiments/meyer_transport_audit/DYNAMIC_FACTORS.md`.
The paper states the requirements and leaves that investigation separate.

Primary reproduction records:

- `../../experiments/meyer_hard_jump_defect_proof.py` proves and renders the
  exact retained-carrier and signed-halo identity.
- `../../experiments/meyer_semismooth_state_jump.py` is the independent
  six-field finite-flow oracle and schedule/constraint ablation.
- `../../experiments/benchmark_meyer_flow_jump.py` produces the native M4
  scaling record, including ordinary fused 19, 29, and 64 controls.
- `flow_natural_benchmark.py` applies the current native methods to Barbara,
  Cameraman, and an analytic crossing.  It reads the archived cold nested
  Gilles output and folds every survivor into effective cartoon `f-v` so all
  displayed outputs are exact two-product decompositions.
- `results/flow_natural_m4/` contains the complete M4 timing samples and all
  arrays used in the natural comparison.
- `render_revision.py` regenerates the current figures and clean timing table from serialized records. `render_figures.py` retains the earlier rendering workflow.

Build and inspect:

```sh
.venv-jpeg/bin/python paper/fast_meyer_bregman/render_revision.py
/opt/homebrew/bin/tectonic -o tmp/pdfs/fast_meyer \
  paper/fast_meyer_bregman/main.tex
pdftoppm -png tmp/pdfs/fast_meyer/main.pdf tmp/pdfs/fast_meyer/page
```

The final artifact is copied to
`output/pdf/fast_meyer_bregman_cartoon_texture.pdf` after visual QA.
