# Results So Far

These are frozen one-thread measurements from the standard-cell harness. HPWL
is in micrometers. The labels matter: a low score on one circuit is not evidence
of a valid inverse if the same law fails the other density regime.

| Method | `gcd` HPWL | `wb_dma_top` HPWL | Interpretation |
| --- | ---: | ---: | --- |
| Native exact all-pairs | 7,222.3325 | 32,686.9825 | Best quality reference; not support-sparse |
| Common legal baseline | 7,322.7625 | 45,353.1100 | Shared comparison baseline |
| Legacy endpoint/FIFO inverse | 7,267.3375 | 45,162.1200 | Sparse-friendly, weak dense transfer |
| Raw global occupied-marginal quotient | 10,367.0250 | **40,800.8575** | Real WB result; loses cell identity |
| Global quotient with x gauge/support projection | 9,147.7100 | **40,432.3000** | Best WB-only aggregate control; still fails GCD |
| Identity-preserving CDF conjugation | **7,280.1875** | **41,716.6300** | Selected transferable support-sparse law |

So `40,800.8575` was not disregarded as a bad measurement. A later variant
even reached `40,432.3`. Both are retained as controls showing that the global
transported occupied marginal contains strong capacity information. They are
not the selected algorithm because collapsing the coupling to that marginal
throws away which cell owns which transported mass, producing catastrophic
GCD regressions.

`41,716.63` is therefore not the best WB number. It is the best result so far
from one cell-identity-preserving, support-sparse inverse that improves both
circuits without a utilization branch: 0.58% over baseline on `gcd` and 8.02%
on `wb_dma_top`.

## Completion time and memory

| Design | Active width | Moves | Transport + readout | Guarded wall | Peak RSS | Explicit state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gcd` | <= 6 | 27 | 0.274 s | 0.54 s | 152.4 MiB | 13.60 MiB |
| `wb_dma_top` | <= 6 | 323 | 0.216 s | 0.73 s | 129.4 MiB | 8.52 MiB |

The rank-prefix reference-phase evaluation alone took 2.3 ms on `gcd` and
5.3 ms on `wb_dma_top`; it is not the present readout bottleneck.

## Remaining gaps

Relative to native exact all-pairs, CDF conjugation remains 57.8550 um behind
on `gcd` and 9,029.6475 um behind on `wb_dma_top`. The next useful work is to
preserve the coupled measure across another relaxation and improve capacity
emission, not to enumerate alternative destinations after decoding.
