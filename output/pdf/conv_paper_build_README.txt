CONV paper — main-text revision, 2026-09-07

Files
  conv_paper_fused.tex  All paper sections and bibliography fused into one TeX file.
  conv_paper_assets/    The seven original figure assets referenced by that file.

Build from this directory:
  tectonic conv_paper_fused.tex

Alternatively, use XeLaTeX with IEEEtran, fontspec, and the packages listed
in the preamble; run enough passes to resolve the cross-references.
The existing Hebrew dedication uses macOS /System/Library/Fonts/SFHebrew.ttf.
On another platform, change that one font declaration to an installed Hebrew
font. No external section files or bibliography database are required.

Main-text additions
  Section VI.A: exact incremental ledger-cost recurrence and its derivation.
  Section VI.B: continuous derivative-energy bound for the Euclidean metric.
  Section X.A: reduced arithmetic count and measured native pipeline savings.
  Section X.D: finite-precision near-tie safeguard and its scope.

The construction, source stencil, signed fibre, blend, and basin rule are
preserved. The energy bound is not a universal optimality claim.
