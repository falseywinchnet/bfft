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
| Far-field conditioner at support fixed point | **7,280.1875** | **40,907.2600** | Current transferable law; no candidate DEF selection |
| Exact-row oracle initializer + current law | -- | **32,724.5075** | Diagnostic only; proves initial row chart dominates |

So `40,800.8575` was not disregarded as a bad measurement. A later variant
even reached `40,432.3`. Both are retained as controls showing that the global
transported occupied marginal contains strong capacity information. They are
not the selected algorithm because collapsing the coupling to that marginal
throws away which cell owns which transported mass, producing catastrophic
GCD regressions.

`41,716.63` is therefore not the best WB number. It was the first result from
one cell-identity-preserving, support-sparse inverse that improved both density
regimes.  The far-field conditioner now preserves the same 7,280.1875 `gcd`
result and reaches 40,907.2600 on `wb_dma_top`, a 9.80% improvement over the
common baseline, without a utilization branch.

The conditioner uses the unrestricted capacity displacement and exact HPWL
subgradient as two circular witnesses, propagates scalar confidence over the
net graph, and evolves the soft quotient to a transported-support residual of
0.01 row height.  `wb_dma_top` contracts through 465.8382, 53.0463, and 6.9136
DBU mean support evolution.  A second-pass DEF happened to score 40,895.8325,
but the selected stopping rule never compares hard outputs.

## Completion time and memory

| Design | Active width | Moves | Transport + readout | Guarded wall | Peak RSS | Explicit state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gcd` | <= 6 | 27 | 0.274 s | 0.54 s | 152.4 MiB | 13.60 MiB |
| `wb_dma_top` | <= 6 | 323 | 0.216 s | 0.73 s | 129.4 MiB | 8.52 MiB |

The current far-field fixed-point run is:

| Design | Soft passes | Direct HPWL | Transport + readout | Guarded wall | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gcd` | 1 | 7,280.1875 | 0.528 s | 0.79 s | 154.2 MiB |
| `wb_dma_top` | 3 | 40,907.2600 | 0.524 s | 1.04 s | 130.2 MiB |

The rank-prefix reference-phase evaluation alone took 2.3 ms on `gcd` and
5.3 ms on `wb_dma_top`; it is not the present readout bottleneck.

## Initial-DEF basin finding

A fixed spherical walk from the RePlAce initial chart to the exact all-pairs
chart isolates the remaining limitation.  On `wb_dma_top`, the unchanged law
returns 40,907.2600 at the original chart, 41,558.0175 at the midpoint,
39,218.7400 at three quarters, and 34,083.6475 at the exact chart.  The path is
not a smooth scalar descent.

The axis control is decisive.  Supplying only the exact x coordinate worsens
the result to 43,253.2525.  Supplying only the exact row coordinate while
retaining the original x gauge yields **32,724.5075**, just 37.5250 um above
the exact all-pairs reference.  That run finishes in 1.04 seconds with
133,532 KiB peak RSS.  It is an oracle experiment, not a deployable method.

At the original initial chart, 99.32% of `gcd` identities are already within
the active radius of their exact target row, versus 68.19% on `wb_dma_top`.
At three quarters of the WB walk that coverage reaches 99.68% and the hard
result crosses to 39,218.7400.  The sparse transport is therefore a fast basin
refiner; the missing global operation is cell-specific competitive row
assignment.

Two reduced substitutes failed to recover it.  A streamed global row-CDF
conjugation converged to 41,068.6125, and using only the old monotone
quotient's row assignment reached 40,648.0200 on WB while collapsing `gcd` to
8,940.3400.  Marginal row mass and monotone occupancy both lose identity.

## Diffusive initializer falsification

A paired WB experiment tested whether short physical net support was deceptive
input to the RePlAce initializer.  The control command regenerated the frozen
baseline DEF byte-for-byte.  The experimental command changed only per-net
wirelength weight, using the heat high-pass survival

```text
w(span, sigma) = 1 - exp(-span^2 / (4 sigma^2)).
```

The scale was derived from the transport rather than tuned: the median
nonzero vertical initializer-to-support displacement was 2.119803 um, or
1.239651 rows.  RePlAce accepted 2,075 custom weights; 59.17% of included nets
received less than half weight.

| WB measurement | Exact control | Short-support eroded | Change |
| --- | ---: | ---: | ---: |
| RePlAce reported HPWL (um) | 21,859.1191 | 28,584.1523 | +30.77% |
| Common-legal initial HPWL (um) | 45,353.1100 | 52,429.3400 | +15.60% |
| Final support-sparse HPWL (um) | **40,907.2600** | **48,986.4300** | **+19.75%** |
| Mean row distance from exact chart | 2.1582 | 4.9311 | +128.48% |
| Within two rows of exact chart | 68.19% | 37.03% | -31.16 points |
| RePlAce + support guarded wall | 1.15 s | 1.08 s | -0.07 s |
| Sequential peak RSS | 134,212 KiB | 134,380 KiB | +168 KiB |

This falsifies the proposed erosion.  It does more than worsen local
wirelength: it moves the initializer away from the exact row chart, halving
the exact-row fraction from 17.06% to 8.83%.  The far field is therefore not
the subset of long nets left after local detail is removed.  It is the
coherent phase learned through the connected hierarchy of short constraints.
Local support is a boundary condition for global row ownership.

## Backward transport pre-image

The exact-row/original-x oracle chart was then fixed as a representational
target.  No HPWL value was consulted during the backward walk.  Each step
measured one transport secant, projected the fixed coordinate residual onto
that response, and used the least-squares secant coefficient.  Final HPWL was
read only after the predetermined residual step was emitted.

The first row probe separated forward transport from unrelaxation.  Along the
natural row pullback, 67.97% of the target residual energy was visible in the
soft innovation map, but only 36.10% was visible after anchor-CDF quantile
emission.  A transport-derived step of 0.358580 improved the innovation's
target-row fraction from 91.77% to 97.36% and its common-legal HPWL from
33,551.4500 to 32,305.3950.  The quantile result did not inherit that gain: it
changed from 32,724.5075 to 32,753.0900.

Reading the already-computed innovation row directly fixed the interface.  It
emits one nearest active segment per cell and propagates capacity only along
that fixed edge; it does not score destinations.  The result was 32,017.0850
with 1,857 of 1,858 cells on the fixed target row.

The horizontal coordinate behaved differently.  It is a largely decoupled
source gauge: the first x secant exposed 78.72% of its residual energy and
caused zero row response.  Three response-calibrated gauge pullbacks reduced
the fixed-target horizontal residual to 0.668 sites mean and two sites at p90.

| WB chart/readout | Initial HPWL (um) | Emitted HPWL (um) | Interpretation |
| --- | ---: | ---: | --- |
| Exact all-pairs reference | -- | 32,686.9825 | Previous exact-path reference |
| Exact-row chart + quantile inverse | 31,344.9300 | 32,724.5075 | Oracle axis diagnostic |
| Row pre-image + innovation endpoint | 31,344.9300 | 32,017.0850 | Soft row phase crosses readout |
| Row + calibrated x pre-image | 32,801.9025 | **31,427.9000** | Best backward-transport diagnostic |

The final run takes 0.904 seconds internally, 1.03 seconds guarded, and
134,564 KiB peak RSS.  Independent DEF re-parsing confirms 31,427.9000.  It is
1,259.0825 um, or 3.85%, below exact all-pairs and descends 1,374.0025 um from
its own common-legal initializer.  Only one cell remains one row from the
fixed target; the readout moves one cell and leaves one fixed edge blocked.

This is not deployable quality evidence because the fixed row target came from
the exact-path oracle.  It is stronger representational evidence: a better
hard placement than the previous exact reference already exists in the
support-sparse transport's reachable image.  The missing operation is learning
the global row chart without an oracle.  Once that chart is supplied, backward
transport can precondition both row phase and horizontal gauge without a
candidate-placement search.

Relative to native exact all-pairs, the current transferable result remains
57.8550 um behind on `gcd` and 8,220.2775 um behind on `wb_dma_top`. The next
useful work is to encode multiscale competitive row ownership in the
transported state while retaining local support at every scale, not to erase
short nets or enumerate alternative destinations after decoding.
