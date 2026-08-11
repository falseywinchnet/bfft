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
| Far-field conditioner at support fixed point | **7,280.1875** | **40,907.2600** | Transferable first-pass law; no candidate DEF selection |
| Residual-gated row self-distillation | **7,280.1875** | **40,411.0225** | Former selected law; one score-free pullback after multi-pass soft continuation |
| Vector connection diffusion + residual pullback | **7,280.1875** | **39,932.2000** | New selected law; transports net-graph orientation while retaining each cell's capacity-orbit radius |
| Exact-row oracle initializer + current law | -- | **32,724.5075** | Diagnostic only; proves initial row chart dominates |

So `40,800.8575` was not disregarded as a bad measurement. A later variant
even reached `40,432.3`. Both are retained as controls showing that the global
transported occupied marginal contains strong capacity information. They are
not the selected algorithm because collapsing the coupling to that marginal
throws away which cell owns which transported mass, producing catastrophic
GCD regressions.

`41,716.63` is therefore not the best WB number. It was the first result from
one cell-identity-preserving, support-sparse inverse that improved both density
regimes.  The far-field conditioner preserves the same 7,280.1875 `gcd`
result and its first pass reaches 40,907.2600 on `wb_dma_top`. Residual-gated
row self-distillation retains GCD byte-for-byte and reaches 40,411.0225 on WB.
Replacing its scalar confidence diffusion by a connection-valued orientation
diffusion retains the same GCD DEF and reaches **39,932.2000** on WB, an 11.95%
improvement over the common baseline, without a utilization branch.

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

The first far-field fixed-point pass is:

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

## Residual-gated row self-distillation

The deployable row target now comes from the first transport itself.  After
the ordinary soft continuation reaches its residual stop, the algorithm reads
only the number of continuation passes.  A one-pass chart is emitted
unchanged.  A genuinely multi-pass chart is pulled backward exactly once: its
hard rows are attached to the original RePlAce continuous x gauge, then the
unchanged support-CDF transport runs once more.

The gate never reads HPWL, utilization, a circuit label, or alternative hard
placements:

```text
self_distill = first_soft_continuation_steps > 1
```

| Design | First soft passes | Pullback | Previous HPWL (um) | Final HPWL (um) | Guarded wall | Peak RSS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `gcd` | 1 | skipped | 7,280.1875 | **7,280.1875** | 0.77 s | 155,644 KiB |
| `wb_dma_top` | 3 | applied once | 40,907.2600 | **40,411.0225** | 1.93 s | 131,652 KiB |

The GCD output DEF is byte-identical to the prior selected result.  WB improves
496.2375 um, or 1.21%, over the former transferable law and 4,942.0875 um over
the common legal baseline.  Its two transport passes take 1.797 seconds
internally, including 22 ms to graft the transported rows into the original x
gauge.

Repeating the row self-map is not continuation.  Its mean row change contracts
from 0.0856 through 0.0560, 0.0474, and 0.0431 rows, but it progressively
replaces the original far-field evidence with decoded state and worsens hard
quality.  The production operation is therefore one backward preconditioning
step, conditionally enabled by the first solve's continuation history, not a
fixed-point loop.

A separate dyadic row hierarchy located the remaining error.  Absolute child
affinity lost identity and returned 65,437.7725.  Carrying hard reference phase
through the transported tree recovered 43,867.6600.  Reinforcing only
transport/net phases that agree circularly reached 41,573.9400 with 16,722
bytes of achieving-path state.  The hierarchy improved row-occupancy error but
still assigned the wrong identities; full net-cost ALS was noncontractive.
These are retained falsifications, not selected variants.

## Vector connection diffusion

The literature round tested whether the initial conditioner was diffusing the
wrong object.  The previous law transported a scalar confidence through the
net graph, then projected it back onto each cell's original capacity direction.
The new law transports the `U(1)` displacement orientation itself:

1. the unrestricted capacity displacement and exact HPWL subgradient produce
   the same circular cross-supported gain;
2. each net averages admitted **unit directions**, not destinations;
3. one vector diffusion transports that orientation to ambiguous cells; and
4. every cell applies the result with its own bounded capacity-orbit radius.

No row, site, or DEF alternative is evaluated.  The existing soft residual
still determines continuation, and the existing one-time backward row graft
still determines whether a second transport runs.

| Design | First-pass HPWL (um) | Final HPWL (um) | Guarded wall | Peak RSS | Added conditioner workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gcd` | 7,280.1875 | **7,280.1875** | 0.79 s | 154,244 KiB | 47,592 bytes |
| `wb_dma_top` | 40,994.1250 | **39,932.2000** | 2.01 s | 130,640 KiB | 298,992 bytes |

On WB, 82.78% of cells receive a redirected orientation, 62.16% gain graph
confidence, and the mean cosine to the former local direction is 0.6574.  The
new result improves the former transferable record by 478.8225 um, or 1.18%.
GCD's output is byte-identical to the former selected DEF.

A repeated WB run produced an identical DEF with SHA-256
`352957bb9853d93bc55d122c8b796a579a3799dab85a755dbe18c70b47edcf04`.
Independent DEF parsing confirms 39,932.2000 um.

The fixed paper-derived ablations explain the gain:

| Ablation | GCD HPWL (um) | WB HPWL (um) | Finding |
| --- | ---: | ---: | --- |
| Exact circular cut on both passes | 7,280.1875 | 40,436.6275 | A real secondary cut ambiguity, not the ownership recovery |
| Deterministic degree-2 far field | 7,279.4150 | 41,174.0400 | Better kernel approximation does not imply a better row chart |
| Cell-net rotation sheaf | audit only | audit only | Collapses HPWL's interval boundary to quadratic wirelength; WB exact-row fraction falls to 9.04% |
| Vector diffusion + circular cut | 7,280.1875 | 40,598.6650 | Absolute row phase suppresses useful vertical connection transport |
| Vector diffusion + conductance gate | 7,280.1875 | 40,438.4950 | Weak cancelling resultants still contain useful orientation |

The selected operation is therefore the un-gated vector connection field.
It confirms the original intuition precisely: the information about direction
lives in the transport relation, and scalar confidence diffusion had erased
it before unrelaxation.

## Phase-boundary microscope

An internal transport trace recorded 18 states across both WB passes: the
lifted chart, unrestricted capacity prediction, every conditioner/retransport
step, local support coupling, and hard readout.  The soft support fixed point
contracts rapidly (0.2329 to 0.0265 to 0.0035 um on pass one, then 0.1698 to
0.0170 um on pass two), while exact row identity stays near 17--20%.  The solve
is converging; it is converging in a phase-blurred identity chart.

When signed support gain/loss is aggregated after sorting cells by the exact
all-pairs target row, it forms a striking two-sided band with a phase reversal
near the middle row.  A per-cell audit shows why that image must not be treated
as a decoder.  The exact result is used only as the post-run diagnostic axis:

| Local-coupling information audit | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| Exact row inside active support | 69.27% | 69.05% |
| Signed moment / required row displacement correlation | 0.0026 | -0.0851 |
| Correct moment direction when target is inside support and nonzero | 62.72% | 63.08% |
| Exact row has positive gain when it is inside support | 51.90% | 51.68% |
| Exact row is the strongest transported/reference ratio | 20.67% | 20.19% |
| CDF inverse reaches exact row | 16.74% | 17.55% |

Thus the band is a collective conditional statistic revealed by oracle
ordering, not an absolute phase stored in each local transport row.  It carries
a weak direction bit but neither competitive identity nor displacement
magnitude.

Three direct, score-free phase inverses falsified simpler recoveries.  Carrying
the first circular support moment reached 45,889.3400.  Pulling back the
continuous innovation row reached 44,349.2575.  Carrying the exact within-bin
CDF residual through the capacity-safe row pullback reached 41,680.5250.
Finally, a fine global site-CDF inverse used rank-prefix sums and binary
inversion, required no `cells x sites` state and chose exactly one target, but
destroyed locality: it moved 92.47% of first-pass cells by 4.99 rows on average
and reached 65,340.2125.  That run took 3.03 s guarded and peaked at 148,619 KiB.

These results narrow the recovery requirement.  Local support is the boundary
condition that keeps an identity meaningful, but local transport does not yet
contain the missing global ownership coordinate.  The next representation
must transport a multiscale competitive identity path into the local chart;
post-hoc sharpening, a single global phase, or a new physical ordering cannot
manufacture it after the coupling has been compressed.

Relative to native exact all-pairs, the current transferable result remains
57.8550 um behind on `gcd` and 7,245.2175 um behind on `wb_dma_top`. Vector
diffusion recovers part of the missing global relation by transporting
orientation instead of scalar confidence. The next useful work is to extend
that connection state with an oriented HPWL interval/achieving-boundary record,
while retaining local support at every scale--not to erase short nets,
collapse a net to one center, or enumerate destinations after decoding.
