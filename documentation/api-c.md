# C API reference

Include the main C API with:

```c
#include <bfft/bfft.h>
```

## Types

| Type | Description |
| --- | --- |
| `bfft_plan` | Opaque reusable real FFT plan. |
| `bfft_workspace` | Opaque aligned scratch workspace. |
| `bfft_complex` | Double-precision complex value with `re` and `im`. |
| `bfft_complex_f32` | Single-precision complex value with `re` and `im`. |
| `bfft_status` | Return status enum. |
| `bfft_layout` | Descriptive layout enum. |

## Version and diagnostics

- `bfft_version_string()` returns the library version string.
- `bfft_backend_name()` returns the selected SIMD backend name.
- `bfft_status_string(status)` returns a readable status string.

## Plan lifecycle

- `bfft_plan_create(n, &plan)` creates a plan for power-of-two `n >= 4`.
- `bfft_plan_destroy(plan)` destroys a plan. Passing `NULL` is allowed.

## Plan queries

- `bfft_plan_size(plan)` returns `N`.
- `bfft_plan_bins(plan)` returns `N / 2 + 1`.
- `bfft_plan_work_size(plan)` returns double work buffer length.
- `bfft_plan_work_size_f32(plan)` returns float work buffer length.
- `bfft_plan_native_scratch_size(plan)` returns standard transform scratch size.
- `bfft_plan_standard_policy(plan)` returns the standard-output packing policy.

## Forward transforms

- `bfft_forward` computes standard FFT-order double-precision output.
- `bfft_forward_native` computes native-order double-precision output.
- `bfft_forward_native_workspace` computes native-order output with workspace storage.
- `bfft_forward_f32` computes standard FFT-order single-precision output.
- `bfft_forward_native_f32` computes native-order single-precision output.

## Magnitude transforms

- `bfft_forward_magnitude` writes standard-order double magnitudes.
- `bfft_forward_magnitude_f32` writes standard-order float magnitudes.

## Inverse transforms

- `bfft_inverse` inverts standard FFT-order double-precision input.
- `bfft_inverse_f32` inverts standard FFT-order single-precision input.
- `bfft_inverse_native` inverts native-order double-precision input.
- `bfft_inverse_native_f32` inverts native-order single-precision input.

## Layout conversion

- `bfft_native_to_standard` converts double native output to standard output.
- `bfft_standard_to_native` converts double standard input to native input.
- `bfft_native_to_standard_f32` converts float native output to standard output.
- `bfft_standard_to_native_f32` converts float standard input to native input.

## Residue transforms and filters

- `bfft_forward_residues` transforms real input to residue coordinates.
- `bfft_inverse_residues` transforms residue coordinates back in place.
- `bfft_filter_size` returns the residue filter length.
- `bfft_residue_filter_from_standard` converts a complex response to a residue filter.
- `bfft_residue_filter_from_real` converts a real zero-phase response to a residue filter.
- `bfft_apply_residue_filter` applies a residue filter in place.
- `bfft_filter_signal` transforms, filters, and inverts a signal.
