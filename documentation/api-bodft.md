# BODFT API reference

BODFT is the native half-bin-shifted real transform. It maps `N` real samples to
`N / 2` packed complex bins at half-bin frequencies.

Include the C API with:

```c
#include <bfft/bodft.h>
```

Include the C++ wrapper with:

```cpp
#include <bfft/bodft.hpp>
```

## C API

- `bodft_backend_name()` returns the selected SIMD backend.
- `bodft_plan_create(n, &plan)` creates a plan for power-of-two `n >= 2`.
- `bodft_plan_destroy(plan)` destroys a plan. Passing `NULL` is allowed.
- `bodft_plan_size(plan)` returns `N`.
- `bodft_plan_bins(plan)` returns `N / 2`.
- `bodft_forward(plan, input, output)` computes double-precision output.
- `bodft_inverse(plan, input, output)` computes double-precision inverse output.
- `bodft_forward_f32(plan, input, output)` computes single-precision output.
- `bodft_inverse_f32(plan, input, output)` computes single-precision inverse output.

## C++ API

Use `bfft::bodft` for RAII ownership and exception-based errors. It
provides size queries and double-precision and single-precision forward and
inverse methods.
