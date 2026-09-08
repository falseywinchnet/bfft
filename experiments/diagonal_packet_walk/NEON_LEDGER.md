# Explicit NEON movement ledger

The `PACKET_NEON` build of `native.cpp` implements the same six schedules as
the scalar experiment. One complex double is represented by a full 128-bit
register with lanes `[real, imaginary]`. All arrays retain that layout,
including natural input/output, packet buffers, and coefficient tables.
There is no boundary AoS/SoA conversion, padding, or hidden batch of transforms.
This is one specified SIMD implementation, not a claim of optimal packing.

## Arithmetic and register movement

For x=(xr,xi), w=(wr,wi), complex multiplication is

    swapped = exchange_lanes(x)
    signed_swap = (-xi, xr)
    result = x * wr + signed_swap * wi.

The actual NEON intrinsics use one lane exchange (`vextq_f64`), one sign-bit
XOR, one two-lane multiply by coefficient lane 0, and one two-lane fused
multiply-add by coefficient lane 1. Coefficient lanes are directly addressed
by the instructions; no coefficient broadcast/shuffle instruction is needed.

A quarter turn exchanges lanes and XORs one sign bit. Conjugating a coefficient
XORs its imaginary sign bit without exchanging lanes. Add/subtract butterflies
operate on whole vectors without lane permutations. General multiplication
uses a fused operation, so roundoff need not be bit-identical to scalar code.

With M general complex multiplies, J quarter turns, B butterflies, and R/W
logical sample reads/writes, the implementation's ledger is:

| Item | Count |
|---|---:|
| 128-bit sample loads | R |
| 128-bit sample stores | W |
| 128-bit coefficient reads | M |
| Complex arithmetic lane exchanges | M+J |
| Coefficient by-element operands | 2M |
| Explicit coefficient broadcast instructions | 0 |
| Boundary layout conversion operations | 0 |
| Vector butterfly additions/subtractions | 2B |
| Vector product multiplies | M |
| Vector product fused multiply-adds | M |

The small sign masks are constants, not data-dependent arrays or additional
N-element transport passes. Compiler setup loads/materialization of these
masks, general-purpose register moves, integer address calculations, ABI
register saves, and loop-control instructions appear in the retained static
assembly inventory. They are not silently equated with sample accesses.

R, W, M, J, B and index-table references for each complete schedule are given
in README and checked against the instrumented native execution by
`audit_counts.py`. In particular entry gathering, both internal routing
passes, and exit scattering remain charged to the fused packet route.
Its removed carry-diagonal pass is fused into the first DIF input loads;
the coefficient loads and multiplications remain charged.

These are required logical operations of the specified implementation.
Compiler scheduling, instruction pairing, reuse, and caches can change
physical instruction/load counts. This is not a hardware performance-counter
trace, a DRAM bandwidth measurement, or a claim about every possible NEON FFT.

## Generated-code check

`audit_assembly.py` reads actual Apple-clang assembly, verifies the complex
multiply lowers to one EXT, one FMUL and one FMLA, and verifies one EXT for
quarter turns. It inventories the uninstrumented dispatch and packet kernels
and rejects vector stack accesses in those bodies. The exact assembly hash
and opcode inventories are retained in `neon_assembly_m4.json`.

This check initially exposed vector stores of unused instrumentation counters
in the timed dispatcher. Those counters are now an empty type when counting
is disabled, and the uninstrumented dispatcher returns void. This removes
measurement bookkeeping from the timed transform rather than overlooking it
in the ledger. Integer control-state spills, if emitted, are still represented
by the assembly inventory; they are not complex sample spills.

## Closed-form comparison at N=65536

N=4^r with r=8, a=b=16. For conventional radix-2 DIT/DIF/Stockham,

    M+J = N log2(N)/2 - N + 1 = 458753.

For the reduced/fused packet route,

    M+J = Nr - N + 2N/a + 2N/b = 475136.

Thus this packing needs about 3.57% more arithmetic lane exchanges for the
fused packet route, alongside 25% more logical sample transfers than Stockham.
Lane exchange cost does not rescue the routing cost identified by the scalar
experiment. It also does not introduce a new asymptotic penalty in this packing.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
 'clang++ -O3 -std=c++17 -DPACKET_NEON -fno-vectorize -fno-slp-vectorize \
  experiments/diagonal_packet_walk/native.cpp -o /tmp/diagonal_packet_neon && \
  /tmp/diagonal_packet_neon 8 /tmp/diagonal_packet_neon.bin \
  > /tmp/diagonal_packet_neon_final.json && \
  python3 experiments/diagonal_packet_walk/verify_native.py \
  /tmp/diagonal_packet_neon.bin --out /tmp/diagonal_packet_neon_oracle.json && \
  clang++ -O3 -std=c++17 -DPACKET_NEON -fno-vectorize -fno-slp-vectorize -S \
  experiments/diagonal_packet_walk/native.cpp -o /tmp/diagonal_packet_neon.s && \
  python3 experiments/diagonal_packet_walk/audit_assembly.py \
  /tmp/diagonal_packet_neon.s --out /tmp/diagonal_packet_neon_assembly.json'
```

Copy the JSON artifacts immediately using the route selected by `m4host`.
Run the local count audit on the copied timing JSON. The temporary binary
input/output dump is only needed for the independent NumPy comparison.
Numerical validation covers all six native schedules at eight sizes through
N=65536; it compares explicitly exported inputs/outputs to NumPy's FFT.
