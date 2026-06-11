# BFFT public API

BFFT exposes a small C ABI and a lightweight C++ wrapper.

## Headers

```c
#include <bfft/bfft.h>
```

```cpp
#include <bfft/bfft.hpp>
```

## Plans

Create one plan per transform size and reuse it for repeated transforms.
Transform sizes must be powers of two and at least four samples.

```c
bfft_plan* plan = NULL;
bfft_status status = bfft_plan_create(4096, &plan);
```

```cpp
bfft::plan plan(4096);
```

Destroy C plans with `bfft_plan_destroy`. C++ plans are RAII objects.

## Memory sizes

- `bfft_plan_size(plan)` / `plan.size()` returns `N`.
- `bfft_plan_bins(plan)` / `plan.bins()` returns `N / 2 + 1` complex bins.
- `bfft_plan_work_size(plan)` / `plan.work_size()` returns the double scratch
  count needed by forward calls.
- `bfft_plan_native_scratch_size(plan)` / `plan.native_scratch_size()` returns
  the complex scratch count needed by standard-output forward calls when the
  fused scatter policy is selected.

## Standard forward transform

The standard forward transform writes ordinary FFT r2c bins from `0` to `N/2`.
This is the easiest API to use and is comparable to `fftw_plan_dft_r2c_1d` or
`pocketfft` r2c output.

```c
bfft_forward(plan, input, output, work, native_scratch);
```

The library automatically chooses its packing policy. You can inspect it with:

```c
const char* policy = bfft_plan_standard_policy(plan);
```

## Native and residue APIs

Advanced users can avoid standard-order conversion:

- `bfft_forward_native` writes the heap-optimized native spectrum order.
- `bfft_inverse_native` reads that native order.
- `bfft_forward_residues` writes Bruun residue coordinates as `N` doubles.
- `bfft_inverse_residues` converts residue coordinates back to time samples.

Use layout conversion helpers when needed:

- `bfft_native_to_standard`
- `bfft_standard_to_native`

## Filtering

BFFT can convert a standard frequency response into residue-domain filter
coefficients once, then apply the filter with a streaming multiply.

```c
size_t filter_size = bfft_filter_size(plan);
bfft_residue_filter_from_standard(plan, response, residue_filter);
bfft_filter_signal(plan, input, residue_filter, output);
```

For real zero-phase responses, use `bfft_residue_filter_from_real`.

## Error handling

C functions return `bfft_status`. Use `bfft_status_string` for diagnostics.
C++ methods throw `bfft::error` for invalid arguments or internal failures.
