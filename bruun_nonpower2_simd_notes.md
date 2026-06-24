# Non-power-of-two SIMD kernel notes

This note tracks the non-power-of-two SIMD work beside `bruun_nonpower2_simd.cpp`.

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

`bruun_nonpower2_simd.cpp` adds two isolated kernels:

1. `quadratic_fwd_variable`, a one-level variable-angle quadratic split.
2. `quadratic_inv_variable`, the inverse of the one-level variable-angle quadratic split.
3. `quadratic_two_level_fused_variable`, a two-level fused split that mirrors the FMA-optimal register lifetime of the main Bruun `norm2` kernel while allowing streamed parent and child coefficient tables.

The forward, inverse, and fused kernels have scalar tails so odd block lengths remain valid. The AVX2 path is opt-in through `BRUUN_NP2_SIMD_AVX2` and deliberately separate from `BRUUN_SIMD_*` to avoid coupling with the power-of-two implementation.

## Next optimization targets

- Build a descriptor scheduler that groups non-power-of-two CRT nodes by equal block shape so the variable-angle kernels receive long contiguous runs.
- Extend inverse coverage to the two-level fused kernel once the full mixed-radix tree ordering is fixed.
- Add a benchmark harness that compares scalar and AVX2 lowering without linking to or modifying fastplan.

## Descriptor scheduler pass

The scheduler now accepts `(offset, coefficient_offset, length)` descriptors over shared `A0/B0/A1/B1/C/S` arenas. It drops zero-length work, validates negative inputs early, and can stable-sort descriptors by descending length before execution. The immediate goal is not a final global planner; it is a low-risk grouping layer that lets future non-power-of-two CRT nodes feed the FMA kernels in longer contiguous runs.

## Validation snapshot

Commands run on this machine after adding the scheduler and benchmark harness:

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

The AVX2 path lowers `B0` and `B1` with packed FMA instructions. The scheduler has a matching `execute_inverse` method, and the harness now round-trips both a single odd-length block and the mixed descriptor workload.
