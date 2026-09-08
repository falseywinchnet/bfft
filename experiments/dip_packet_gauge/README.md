# DIP packet-gauge experiment

This is an isolated exploration of `src/detail/bruun_dip_kernel.hpp`.
It changes where a packet carries its phase, while preserving the complete
real-input → natural-order complex-spectrum contract. Production kernels and
public dispatch are unchanged.

Read the [audit and derivation](../../src/detail/bruun_dip_gauge_audit.md) and
[measured results](RESULTS.md). The generated header is intentionally ignored:
`generate.py` reads the authoritative DIP header, records its SHA256, and
constructs a separate class template with public experimental hooks.

## What is being compared

| Name | Change |
|---|---|
| `dip` | Unmodified production DIP header |
| `dip_clone` | Same algorithm through generated template; code-layout/noise control |
| `fused8` | Fuse the first two terminal levels using the existing four-rotation cell |
| `gauge8` | Shifted polynomial / fixed positive DFT8 terminal |
| `gauge16` | DFT16 where the descent reaches width 16; otherwise gauge8 |
| `gauge32` | DFT32/16/8 terminals, selected by actual descent width |
| `gauge_core` | Three-rotation two-level cells, including the width-8 terminal's first two levels |
| `gauge_hybrid` | Three-rotation interior plus gauge8 leaves |
| `unfolded_packet` | Entire boundary packet is prephased; ordinary positive complex DIF within two real rows; bit reversal and conjugate folding fused into comb output |
| `gauge_selective` | Three-rotation cell only with at least eight columns per stream; original mirrored terminal retained |
| `dif`, `dit` | Actual repository kernels, same natural input/output, caller-provided scratch |

The gauge is complex **notation for the existing two real coordinates**.
No extra degrees of freedom, complex input buffer, or full complex N-point FFT
is introduced. `unfolded_packet` uses a generic radix-2 inner loop and explicit
bit-index arithmetic; its negative result is about this implementation, not a
lower bound on every realization of the gauge identity.

## Reproduce

From the repository root:

```sh
experiments/dip_packet_gauge/run.sh
```

This generates the isolated header, uses `m4build` to compile on the Mini,
runs the ASan/UBSan cell tests, runs the full transform and local cell
benchmarks sequentially, copies `/tmp` results back using `m4host`, and
regenerates the summary. It does not install software or change library defaults.

Manual build after generation, on the Mini mirror:

```sh
clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast \
  experiments/dip_packet_gauge/benchmark.cpp -o /tmp/dip_packet_gauge
/tmp/dip_packet_gauge 20 15
```

Arguments are maximum log2(N), timing rounds, and optional minimum log2(N).
The report excludes sizes below 256 from geometric means because call and
code-placement overhead dominate the smallest transforms. The benchmark also
checks those sizes by default. Apple runs request interactive thread QoS;
this is not CPU pinning or an assurance of an idle machine.

## Validation and measurement limits

- 275 ASan/UBSan cell/packet cases cover general packet coefficients, q=1 and
  vector tails, deliberately offset pointers, guard sentinels, two-level
  energy gain of four, inversion by the original cell, and invalid plan sizes.
- Each complete-transform size checks eight signals: impulse, DC, Nyquist,
  mixed sinusoids, and independent random signals including scales 1e-100 and
  1e100. A separate long-double DFT checks every bin through N=256 and sampled
  bins thereafter. All variants are compared with DIT and inverted with the
  original DIP inverse. The inverse itself is not a new candidate.
- Full transforms always start from the same finite input during timing.
  There is no recurrent unnormalized transform, NaN accumulation, setup,
  allocation, or inverse in the timed region. Each method gets the same input,
  output, and work arrays. DIF has its additional caller-provided native scratch.
- Timing order rotates each round. Medians are reported, not best samples.
  Raw per-round samples were added for the final selective follow-up; earlier
  files contain median/min/max only. The isolated cell timing includes a
  common reset memcpy and precomputes its phases; complete transforms include
  their dispatch and coefficient costs.
- The M4 was heavily occupied by unrelated jobs (load averages roughly 30–40).
  No other processes were interrupted. The A18 Pro corroboration ran the M4-
  compiled executable on the MacBook and was also not an isolated-machine run.
  A clone varying by a few percent makes small improvements inconclusive.
- These are double-precision ARM/NEON measurements. The new streamed core has
  V2 and scalar loops; it does **not** have a dedicated AVX2-wide implementation.
  No cross-platform speed claim is made.

`screen_m4.jsonl` records the initial, especially noisy screen. `core_*`
records the first three-rotation follow-up. `final_*` adds the fully unfolded
packet and extends M4 correctness through N=1,048,576. `selective_m4.jsonl`
records the final selective policy with raw timing samples. Historical files
are preserved so unsuccessful branches and run-to-run variance remain visible.
