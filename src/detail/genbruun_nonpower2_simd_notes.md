# Non-power-of-two SIMD kernel notes

This note tracks the non-power-of-two SIMD work beside `src/detail/genbruun_nonpower2_simd.hpp`.

## Guardrails

- Do not touch `bruun_fastplan.cpp`, `bruun_fastplan_safe_api.cpp`, or `bruun_fastplan_benchmark_standard.cpp`.
- Do not alter the power-of-two Bruun kernels. The new kernels live in a separate namespace and file.
- Keep the non-power-of-two work as reusable SIMD building blocks until the exact mixed-radix/Bruun factor scheduler is settled.

## Layout decision

The power-of-two normalized Bruun kernel is fast because every quadratic split streams contiguous `[A0 | B0 | A1 | B1]` quarters and lowers the rotation to two FMA expressions:

- `R = c * B0 - s * B1`
- `I = s * B0 + c * B1`

The non-power-of-two structures should keep that same arithmetic shape, but the coefficients must be streams instead of one scalar angle per node. Odd factors and mixed CRT branches can then use the same register schedule even when branch widths are not powers of two.

## Implemented first step

`src/detail/genbruun_nonpower2_simd.hpp` adds three isolated kernels:

1. `quadratic_fwd_variable`, a one-level variable-angle quadratic split.
2. `quadratic_inv_variable`, the inverse of the one-level variable-angle quadratic split.
3. `quadratic_two_level_fused_variable`, a two-level fused split that mirrors the FMA-optimal register lifetime of the main Bruun `norm2` kernel while allowing streamed parent and child coefficient tables.

The forward, inverse, and fused kernels have scalar tails so odd block lengths remain valid. The backend staging now mirrors the power-of-two kernel: scalar, 128-bit SSE2/NEON, then AVX2/FMA.

## Next optimization targets

- Build a descriptor scheduler that groups non-power-of-two CRT nodes by equal block shape so the variable-angle kernels receive long contiguous runs.
- Extend inverse coverage to the two-level fused kernel once the full mixed-radix tree ordering is fixed.
- Keep the benchmark harness under `benchmarks/` and compare scalar, NEON/SSE2, and AVX2 lowering without linking to or modifying fastplan.

## Descriptor scheduler pass

The scheduler now accepts `(offset, coefficient_offset, length)` descriptors over shared `A0/B0/A1/B1/C/S` arenas. It drops zero-length work, validates negative inputs early, and can stable-sort descriptors by descending length before execution. The immediate goal is not a final global planner; it is a low-risk grouping layer that lets future non-power-of-two CRT nodes feed the FMA kernels in longer contiguous runs.

## Validation snapshot

Commands run on this machine after adding the scheduler and benchmark harness:

- NEON/FMA harness: `blocks=384 lanes=7300 ns_per_lane=2.043`.
- Scalar harness: `blocks=384 lanes=7300 ns_per_lane=2.054`.
- AVX2/FMA harness: `blocks=384 lanes=7300 ns_per_lane=1.295`.
- Assembly check found `vfmsub231pd` and `vfmadd231pd` instructions in the AVX2 object, confirming the intended fused multiply-add lowering.

The benchmark mutates data repeatedly, so the absolute values are only a local smoke test. The useful signal is that the AVX2 path is active, measurably faster on this descriptor workload, and lowers to packed FMA instructions.

## Inverse counterpart pass

The one-level inverse reconstructs the original lanes with:

- `A0 = 0.5 * (U0 + V0)`
- `R = 0.5 * (U0 - V0)`
- `I = 0.5 * (W0 + X0)`
- `A1 = 0.5 * (W0 - X0)`
- `B0 = c * R + s * I`
- `B1 = c * I - s * R`

The SIMD paths lower `B0` and `B1` with packed multiply-add instructions. The scheduler has a matching `execute_inverse` method, and the harness now round-trips both a single odd-length block and the mixed descriptor workload.

## GenBruun path integration

`src/detail/genbruun_kernel.hpp` now sweeps the odd `MINUS_SPLIT` and `BRUUN_ODD` projection loops along contiguous `m` lanes. The helpers use the same `BRUUN_LEVEL` 128-bit abstraction as the power-of-two kernel, so ARM NEON gets a first real acceleration path while the generic scalar tail remains the reference behavior.

Fixed-radix NEON-side codelets are in place for radices 3, 5, 7, 11, 13, 17,
and 19.
`MINUS_SPLIT` uses fused/hoisted small-radix projections; `BRUUN_ODD` has matching
forward and adjoint child helpers. The plan stores a codelet kind per odd op so
execution dispatch does not repeatedly classify `p`. Larger odd primes still use
the generic vectorized projection helpers.

Latest local smoke results on this machine:

- `N=125`, 32 iterations: forward `0.58 us`, inverse `0.93 us`, roundtrip
  `1.39 us`, roundtrip error `5.6e-16`.
- `N=625`, 32 iterations: forward `3.07-3.09 us`, inverse `4.45-4.69 us`,
  roundtrip `7.50-7.75 us`, roundtrip error `6.7e-16`. Earlier short-run
  `N=625` roundtrip timing was an outlier.
- `N=2401`, 4 iterations: forward `14.6 us`, inverse `79.4 us`, roundtrip
  `41.9 us`, roundtrip error `1.0e-15`. This points at inverse-side work on
  deeper radix-7 chains even though correctness is clean.
- `N=289`, 8 iterations: forward `7.0 us`, inverse `5.9 us`, roundtrip
  `12.9 us`, roundtrip error `4.4e-16`.
- `N=361`, 8 iterations: forward `3.7 us`, inverse `3.0 us`, roundtrip
  `6.7 us`, roundtrip error `5.6e-16`.
- `scripts/odd_prime_accuracy_probe.py 121 169 289 361 625 2401` stayed
  FFT-grade, with max Bruun errors in the `1.8e-16..2.8e-16` range.

Remaining inverse-side work:

- The fixed-radix inverse/adjoint codelets are implemented for 3, 5, 7, 11, 13,
  17, and 19. Correctness and high-precision probes pass.
- The rooted `BRUUN_POW2` cascade still uses one-level `norm_q_inv`; port the
  same `norm2_inv_fused` staging used by the power-of-two kernel.
- Larger odd primes still use the generic vectorized inverse fallback.
- The deeper radix-7 chain (`N=2401`) needs an assembly/profiling pass; inverse
  timing is currently the most suspicious result.

## Follow-up measurements

- A two-vector latency-hiding unroll in the fixed-radix template codelets was
  tested and reverted on this NEON target. It increased register pressure and
  slowed smooth odd cases, especially radix-7-heavy paths.
- Rooted `BRUUN_POW2` cascade use of `norm2_fused` / `norm2_inv_fused` was also
  tested at `q >= 16` and `q >= 64`, then reverted. Correctness stayed clean,
  but local timings were unstable and slower for mixed sizes such as `N=3000`.
  The next attempt should start from assembly inspection rather than a blind
  breadth-loop substitution.
- Arbitrary-N now routes through the public native-order, magnitude, and f32 APIs.
  Internally, arbitrary-N native order is the standard order for now; f32 has a
  separate float GenBruun arena and op walker, using float pow2 tails where the
  arbitrary-N factorization reaches the native Bruun subtree.

## Benchmark shape

`benchmarks/odd_prime_power_fftw_benchmark.cpp` now mirrors the power-of-two
benchmark columns. With no size argument it sweeps primes adjacent to powers of
two; with a power-of-two argument it sweeps odd sizes in the interval `(N/2, N)`;
with a non-power-of-two argument it benchmarks that single arbitrary-N size.
Native-order, f32, PFFFT, and MKL columns print `n/a` where the arbitrary-N path
does not support the corresponding mode yet.
