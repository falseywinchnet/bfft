# Exact Markov Elimination for Relational Current Tracking

Research manuscript in the CONV house style: IEEEtran journal,
two columns, the same theorem families and opening convention, and compact
numbered mathematical exposition. The author is `Anonymous`, matching the
reference CONV draft. Edit the author field and PDF metadata together when
preparing a named submission.

## Manuscript build

The source is `main.tex`; it includes generated rows from `tables/` and vector
PDFs from `figures/`. References are embedded with `thebibliography`, so no
BibTeX or Biber run is needed. No absolute paths, operating-system fonts,
network calls, or shell escape are required by the document.

With a standard TeX Live installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The delivered PDF was compiled and visually checked with Tectonic 0.17.0:

```sh
tectonic -X compile main.tex --outdir build --keep-logs
```

`IEEEtran`, Latin Modern, amsmath, amssymb, amsthm, bm, microtype, graphicx,
booktabs, and hyperref are standard TeX packages. The source is compatible
with the PDF figure workflow; local validation used Tectonic/XeTeX, not an
actual arXiv submission service. The source archive contains only the TeX
file, table inputs, and required figures. No generated main PDF, build log,
private path, or experiment dependency is needed to compile it.

The arXiv preparation guidance used was
https://info.arxiv.org/help/submit_tex.html (accessed 7 September 2026).
The manuscript has not been submitted or published.

## Evidence and regeneration

`evidence/manifest.json` records SHA-256 digests and repository provenance
for eight frozen input files. Run the table/figure generator with Python,
NumPy, and Matplotlib:

```sh
python generate_assets.py
```

It validates the input digests, aggregates the retained raw case records,
and regenerates ten tables plus three vector figures. It does not train
or rerun a tracker. IMM/robust-IMM bootstrap intervals retain the original
reported resamples. Additional CV/CA/UKF intervals use 4000 paired resamples
within each of the five fixed families, seeded with 880042. Those additional
intervals are saved separately in `derived_paired_intervals.json`.

The four empirical protocols are kept separate:

1. Matched relational forecast comparison: 720 method/case runs, 30 trajectories
   in each of three conditions, including five conventional controls and two
   current-prior ablations. Forecast baseline paths number 2048 per query run.
2. Adaptive acquisition: 480 policy/case records; count/accuracy/settling
   comparisons with the same retained update law.
3. Exact optimization: 480 cases run through both implementations (960 runs),
   plus separate 31-repeat microbenchmarks (five for 512-cell dense cases).
4. Historical geometric confirmation: a separate 80-trajectory cohort with
   corrupted observations; the current model was absent. This appendix is
   contextual, not a direct current-versus-geometric result.

The tiny CV cost control differs from the augmented CV implementation in
protocol 1. The paper does not convert their differing workload timings into
a common accuracy/latency ranking. It makes no universal contemporary SOTA,
real-flight, or certified Gaussian-mixture coverage claim.

The evidence archive supplied alongside the arXiv source contains the input
records, this generator, the generated assets, and a snapshot of the relevant
experiment Python/C++ source for inspection. The paper's numerals are not
silently updated by rerunning the experiments.

The uncertainty section proves a sharp conditional bounded-acceleration error
certificate, its horizon and sensor-error growth, and a distinct posterior
growth envelope for the finite current mixture. It does not claim a superior
worst-case physical rate. The acquisition fixtures do not establish the
acceleration premise. `derived_degradation.json` contains within-method
noise/sparsity inflation and 4000 paired within-family bootstrap intervals
(seed 880043), regenerated from the existing forecast records.
