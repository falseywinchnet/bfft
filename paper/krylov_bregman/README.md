# Manuscript source

**Discoverable Transport: Closure, Observability, and the Cost of Acceleration**

The September 30 theory revision uses the CONV paper's compact IEEE journal
house style. `main.tex` is now the authoritative, self-contained manuscript:
no external section files, images, or bibliography are required to compile it.
Open that file in the Codex LaTeX editor for editable source and PDF preview.
The native compiler successfully builds it.

The argument is organized by closure, acquisition/observability, and complete
cost rather than by the chronology of experiments. The main results include
finite observable and positive-weighted-degree closure, mirror compatibility,
trajectory defects, conditional-energy bounds, fixed-point horizon-uniform
second-jet transport, exact linear observability, a tolerance-dependent
directional law, and finite joint-pattern request algebras. The empirical
section has one selected-outcome table. Specialized proofs and protocols are
appendices. Supplied, structurally discovered, and numerically learned
representations are distinguished explicitly.

The finite-energy theorem is fixed-point-centered; its arbitrary-anchor
counterpart is pathwise. Pattern-algebra and observability dimensions are
representation-specific statements, not universal lower bounds on acceleration.
Float64 verification is distinguished from interval certification throughout.

## Export

From the repository root, using the existing Tectonic installation:

```sh
tectonic -X compile paper/krylov_bregman/main.tex --outdir output/pdf --keep-logs
```

The delivered PDF is `output/pdf/krylov_bregman_research_draft.pdf`.
`output/pdf/krylov_bregman_source.zip` contains the self-contained source and
this README. The editor's preview uses its own compiler; command-line export
is for a shareable PDF and inspection of layout diagnostics.

The other `.tex` files and figure scripts in this folder preserve the earlier
modular exposition and full experiment tables. They are not inputs to the
current manuscript. Running an earlier report generator does not update the
new theory text; reconcile any new measurements explicitly.

## Evidence

Implementation, proof checks, and unaltered run records remain under:

- `experiments/krylov_bregman/`: mirror/Krylov identities, Huber discovery,
  coupled Meyer certificates, and rational chart synthesis.
- `experiments/entropic_transport_closure/`: conditional evolution, finite
  energy, directional/observability diagnostics, Gaussian precision closure,
  primitive-response reuse, frame conditioning, and mixed request algebra.

The manuscript's final appendix describes comparison conditions directly,
without requiring access to these internal records. Different stopping norms, finite-trajectory targets, and timing
protocols are not pooled. Initial invalid axis-control ablations are excluded.

The revision followed an explicit consultation with Claude in the existing
research conversation. Its proposed closure/observability/cost organization
was adopted; the fixed-point, finite-lift, and representation-specific scope
corrections were agreed before editing. The mathematical claims remain subject
to their stated hypotheses and the retained proof checks.
