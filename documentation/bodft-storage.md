# Explicit BODFT provisioning

The original `bodft_plan_create` API remains available. The new API in
`include/bfft/bodft.h` separates immutable coefficients from mutable execution
scratch and performs no allocation or deallocation itself.

1. Call `bodft_query_storage(N, precision, &requirements)`.
2. Allocate `plan_bytes` with `plan_alignment` and `workspace_bytes` with
   `workspace_alignment` through the application's allocator or arena.
3. Call `bodft_prepare`, then `bodft_prepare_workspace`.
4. Reuse the plan and workspace with the prepared forward/inverse entry points.
5. After all calls and borrows end, release the two original allocations. There
   is no separate destroy call: the stored objects have trivial destructors.

The plan owns no scratch. Independent threads may share it with distinct
workspaces. A workspace can serve any plan of the same size and precision;
overlapping calls on the same workspace are prohibited. The owner must keep the
storage at a stable address and must not modify plan bytes. Handles are valid
only after successful preparation and until storage is released or reused.

The queried sizes include metadata, live coefficient/index arrays, scratch and
alignment padding. They exclude the caller's input/output and allocator
bookkeeping. Sizes depend on build and precision; never persist hard-coded
requirements across library revisions. Query checks size arithmetic without
allocating. Preparation checks extent and alignment before writing. Execution
checks counts, precision, workspace shape and non-overlap before writing either
output or workspace. See the header for output-slot aliasing and error contracts.

Only the requested precision is constructed. Coefficient tables are built for
the radix-4 stages actually executed. The legacy allocating API now checks each
internal allocation and translates constructor failure to `BFFT_ERROR_ALLOCATION`
at the C boundary, releasing partially constructed storage.

## Validation and reviewed scope

**MEASURED, 2026-10-07, native Windows x64, Clang 22.1.8:** all ten CMake library
tests pass, including the new provisioning test. The new test also passes with
AddressSanitizer and UndefinedBehaviorSanitizer. It exercises both precisions,
all powers of two from 2 through 8192, exact/short/misaligned/overflowing storage,
execution overlap/count/precision/shape rejection, same-shape workspace reuse,
round trips, agreement with the legacy API, and concurrent execution with a
byte-immutable shared plan. This is not macOS/Linux validation or a speed claim.

House-style source review covers the new C API declarations,
`src/bodft_prepared.cpp`, `src/detail/bodft_storage.hpp`,
`tests/bodft_storage.cpp`, and the changed allocation/storage/type sections in
the legacy kernel and wrappers. It checks explicit types, named execution,
ownership/lifetimes, initialized storage, overflow/conversions, failure state,
and setup versus repeated-loop work. C++17 remains the provider language mode.
The pre-existing kernel/wrappers contain older spelling and compact expressions;
this change does not certify or reformat their untouched implementation.

Native reproduction:

```sh
cmake -S . -B build-bounded -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBFFT_BUILD_EXAMPLES=OFF -DBFFT_BUILD_PROBES=OFF \
  -DBFFT_BUILD_HIGH_VISION=OFF -DBFFT_ENABLE_AUTO_SIMD=OFF
cmake --build build-bounded --parallel 4
ctest --test-dir build-bounded --output-on-failure
```

For codec consumers, charge both allocations and conversion buffers to the same
bounded resource as setup and packet storage. Provision before processing, retain
immutable plans for the stream lifetime, and keep scratch on each decoder. A
separate accounting-only reservation is not a substitute for allocator ownership.
