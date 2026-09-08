# Native real quartic factorization and NEON experiment

2026-09-05. The executable is `normalized_quartic.hpp`. Measurements are in
`NATIVE_RESULTS.md` and `native_m4/`; the pre-inlining ablation is retained in
`native_m4_before_inline/`. Production BFFT dispatch is unchanged. The only
production-file edit is an opt-in block in `examples/benchmark.cpp`.

## What was factored

The coefficient-frame candidate stores a residue modulo

    Q(y) = q_a(y) q_b(y), y=t^h,
    q_theta(y) = y² - 2 cos(theta)y + 1.

For each of its two quadratic residues u(y-independent) + t^h v, use

    A = u + cos(theta) v,
    B = sin(theta) v.

Each of A and B has h real coefficients. The pair of factors therefore still
uses exactly 4h real coordinates. This is a change of coordinates on the
quartic packet, not an enlargement to independent complex samples. The ridge
q_R(y)=y²-1 retains its two real coefficient streams instead.

Let h=2w and split each of A and B into width-w halves. For one unit-circle
quadratic define C=cos(theta/2), S=sin(theta/2). Then

    r = C A1 - S B1
    i = S A1 + C B1
    low  = (A0+r, B0+i)
    high = (A0-r, i-B0).

These are the normalized residues at theta/2 and pi-theta/2. Apply that cell
to each old factor, then assign the four child pairs using any of

    (0,1 | 2,3), (0,2 | 1,3), (0,3 | 1,2).

If G denotes the packet coordinate change, R the original coefficient-frame
remainder transition, B the real butterfly and P the chosen pair permutation,
the exact identity is

    G_children R_choice = P_choice (B_a direct_sum B_b) G_parent.

The SymPy certificate checks the actual formulas against independent polynomial
remainders, for every monomial t^j, j=0..7, using symbolic C,S with C²+S²=1.
It also checks the ridge separately: 64 scalar-coordinate identities total.
Since all three pairings only assign these certified child residues to packets,
the identity preserves all three choices. The older coefficient-frame
certificate separately tests all three remainder maps and all 27 N=16 plans.

At the input, t^N-1 splits into ridge and quarter-turn factors by sums and
differences of the input halves. At the output, each normalized pair is already
(Re X, -Im X). The code writes standard bins directly and emits real DC/Nyquist.

## Cost and what this says about the candidate

For a generic packet with eight width-w input streams, the literal coefficient
reducer used 32w multiplies and 32w subtracts to produce both quartic children.
The normalized transition uses 8w multiplies and 12w adds/subtracts: a fourfold
reduction in multiplication count at this cell. The ridge is cheaper still.
Counts are real arithmetic counts, not CPU instruction or timing predictions.
Geometry, trigonometry, choices and terminal-bin indices are planned once.
No allocation, trigonometric call, division or coefficient-frame reconstruction
occurs in a timed transform.

This cheap factorization is precisely two normalized Bruun cells followed by
routing. It exposes a limitation as well as making the experiment practical:
in these coordinates, cross-parent grouping does not create a new dependency
between the two quadratic residues. This implementation is a Bruun arithmetic
graph with selectable packet layout and fused destination writes. It establishes
neither a new arithmetic FFT primitive nor a proof that another packet frame
cannot expose useful cancellations. It is also not a DIT/DIF hybrid.

A final-leaf pairing can be bypassed when writing individual Fourier bins.
`crossings` counts all selected cross-pairings in the abstract plan;
`executed_crossings` counts only those above the final leaf that actually route
intermediate data. This distinction prevents terminal labels from being counted
as performed transport.

## NEON implementation

The implementation uses BFFT's actual `bruun_simd_backend.hpp` primitives.
Streaming vector lanes hold two adjacent real coefficients. Each generic
packet iteration uses four vector multiplies, four vector fused multiply-add
or subtract operations, and eight vector additions/subtractions, plus eight
loads and eight stores. Pair routing is compiled into destination stores;
there is no separate permutation buffer or global sort pass. Indexed output
stores are still real transport costs and are included in measurements.

The final sixteen-sample codelet fuses two tree levels and performs both
child routes before writing standard Fourier bins. Leaf SIMD vectors process
the two real quadratic factors together, using transposes to align lanes.
The register helper is explicitly inlined after assembly inspection found
Clang otherwise emitted calls across the intended fusion boundary. The archived
assembly contains actual two-double FMUL, FMLA/FMLS and transpose instructions.
The unfused ending remains a selectable ablation.

These intrinsic semantics follow Arm's primary reference:
https://arm-software.github.io/acle/neon_intrinsics/
In particular BFFT's V2_MADD(a,b,c) means a+b*c, and V2_MSUB(a,b,c) means a-b*c.
No cost claim is inferred from the instruction names; whole-transform timings
are measured on the M4.

The plan stores 64-byte operation records, N/4-1 records, including the child
records consumed by the fused codelet. Its metadata footprint is therefore
16N-64 bytes, in addition to N real work values and N/2+1 output complex bins.
Metadata streaming, upper-level work-buffer traffic, leaf instruction count
and output locality remain opportunities to inspect against BFFT's mature
fused kernels. The benchmark is not an attribution experiment separating them.

## Benchmark contract and validation

Run from the authoritative checkout:

    experiments/real_fourier_walk/run_native.sh

This uses m4build/m4host, compiles current `src/bfft.cpp` with its default DIF
dispatch, and compiles the real `examples/benchmark.cpp` with
`BFFT_BENCH_QUARTIC_WALK`. Both use -O3, C++17 and -ffp-contract=fast; the
experiment does not use -ffast-math. No prebuilt stale BFFT library is linked.
The standard benchmark columns continue to run, including its optional FFTW
comparison. The opt-in QUARTIC JSON records add the paired experiment.

For each size 64 through 1,048,576, all four variants run on the same generated
real signal: sibling, separated, random, and separated with the unfused ending.
Each comparison uses seven chunks, alternating which engine is timed first;
the existing benchmark's timing function supplies three warmups per chunk.
Each process reports median chunk timings. Three process runs are made, with
size order reversed in the middle run. The summary takes medians across runs
and retains run ranges in `summary.json`. Timed execution includes input-to-work
movement, arithmetic, packet routing, standard output and matching input/sink
perturbations. Planning and allocation are excluded for both engines. This is
warm repeated-transform timing, not cold-plan, single-shot or multi-core timing.

The validation program covers powers of two from 4 through 65,536, both endings,
all three policies, 32 random plan seeds at small sizes, impulses/basis vectors,
random input, constants, alternating input, a sinusoid, zero input, and work
bounds. It performs 10,432 transform checks in each build. Small cases are also
compared directly to a long-double O(N²) DFT. Both ASan/UBSan and the optimized
build pass. The benchmark additionally verifies each measured size against
DIF, including sizes larger than the standalone validation sweep.

Source hashes, compiler flags, Apple M4 identity, load averages, complete output,
raw timings, mathematical certificate and both validation reports are saved.
The machine was not exclusively reserved; three-run variability is retained,
so a ratio indistinguishable from one is treated as a tie, not a speed claim.

## Measured interpretation

The final main-benchmark median walk/DIF ratio is 0.956 at N=64 (the three-run
range is 0.951–0.958). Every measured larger size is slower: about 1.096 at
N=128, 1.302 at N=4096 and 1.640 at N=1,048,576. This is a narrow small-transform
win, not a general crossover. Sibling and random routes also improve on DIF at
N=64, so that win cannot be assigned specifically to cross-parent routing.

A separate diagnostic reuses `examples/dif_vs_dip_benchmark.cpp`'s paired timing
utilities and calls `bruun::DIF_RFFT_kernel::forward_standard` directly, removing
the public C/C++ plan wrapper from the baseline. It times both A/B orders and
repeats five times. Median walk/DIF ratios are 0.907 at N=64, 1.092 at N=128,
and 1.305 at N=4096. This corroborates the narrow small-transform result; the
diagnostic uses that harness's signal and loop, so its nanoseconds are not
interchangeable with the main benchmark's nanoseconds. Raw rows are retained
in `native_m4/direct_kernel_comparison.jsonl`.

Within this experimental kernel, separated routing reduces median execution
time versus sibling routing by about 26% at N=65,536 and 25% at N=262,144.
At N=4096 the two are essentially tied. Thus routing freedom demonstrably
matters in this implementation even though the real arithmetic is unchanged.
The runs do not isolate whether output locality, cache conflicts, instruction
scheduling, or another layout-dependent effect causes the larger-size gain.
They do not show that routing is free or that sorting has been eliminated.

The next useful native direction is to keep the measured cross-parent layout
advantage while reducing upper-level buffer traffic and plan metadata toward
BFFT's fused schedule. That is a concrete performance direction. A distinct
arithmetic FFT would need something more than the block permutation exposed
by this factorization, such as a maintained frame that creates reusable mixed
terms without paying to reconstruct the old residues at every level.

Reproduce the direct-kernel diagnostic with:

    /Users/ultimussecundai/.local/bin/m4build -- sh -c 'clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast experiments/real_fourier_walk/compare_direct.cpp -o /tmp/quartic_compare_direct && /tmp/quartic_compare_direct > /tmp/quartic_compare_direct.jsonl'

Copy `/tmp/quartic_compare_direct.jsonl` back immediately using m4host.
