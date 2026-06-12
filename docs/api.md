# BFFT public API

BFFT exposes a small C ABI and a lightweight C++ wrapper.

## Headers

```c
#include <bfft/bfft.h>
```

```cpp
#include <bfft/bfft.hpp>
```

## Version and backend

The C header defines `BFFT_VERSION_MAJOR`, `BFFT_VERSION_MINOR`, and
`BFFT_VERSION_PATCH`. Runtime helpers return stable diagnostic strings:

```c
const char* version = bfft_version_string();
const char* backend = bfft_backend_name();
```

The C++ wrappers are:

```cpp
std::string version = bfft::version_string();
std::string backend = bfft::backend_name();
```

`bfft_backend_name` reports the SIMD backend compiled into the library, such as
`avx2-fma-256`, `sse2-128`, `neon-128`, or `scalar`.

## Types

- `bfft_plan` is an opaque reusable transform plan.
- `bfft_complex` stores one complex value as `double re` and `double im`.
- `bfft_complex_f32` stores one single-precision complex value as `float re`
  and `float im`.
- `bfft_status` is the C return-code enum.
- `bfft_layout` names the public representations:
  `BFFT_LAYOUT_STANDARD`, `BFFT_LAYOUT_NATIVE`, and `BFFT_LAYOUT_RESIDUES`.

The C++ header aliases these as `bfft::complex`, `bfft::status`, and
`bfft::complex_f32`, `bfft::status`, and `bfft::layout`.

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

`bfft_plan_create` returns `BFFT_ERROR_INVALID_ARGUMENT` for invalid sizes or a
NULL output pointer, and `BFFT_ERROR_ALLOCATION` if plan allocation fails. On
failure, the output plan pointer is set to NULL.

Destroy C plans with `bfft_plan_destroy`. Passing NULL is allowed. C++ plans are
RAII objects and throw `bfft::error` if construction fails.

## Memory sizes

- `bfft_plan_size(plan)` / `plan.size()` returns `N`.
- `bfft_plan_bins(plan)` / `plan.bins()` returns `N / 2 + 1` complex bins.
- `bfft_plan_work_size(plan)` / `plan.work_size()` returns the double scratch
  count needed by forward calls.
- `bfft_plan_work_size_f32(plan)` / `plan.work_size_f32()` returns the float
  scratch count needed by single-precision forward calls.
- `bfft_plan_native_scratch_size(plan)` / `plan.native_scratch_size()` returns
  the complex scratch count to allocate for standard-output forward calls.
- `bfft_filter_size(plan)` / `plan.filter_size()` returns the double count for
  residue-domain filters.

The size query functions return zero for a NULL C plan. For simple callers,
allocate every buffer from these helpers and pass the buffers to the matching
function.

## Standard forward transform

The standard forward transform writes ordinary FFT r2c bins from `0` to `N/2`.
This is the easiest API to use and is comparable to `fftw_plan_dft_r2c_1d` or
`pocketfft` r2c output.

```c
bfft_forward(plan, input, output, work, native_scratch);
```

Buffer sizes:

- `input`: `N` doubles.
- `output`: `bfft_plan_bins(plan)` complex values.
- `work`: `bfft_plan_work_size(plan)` doubles.
- `native_scratch`: `bfft_plan_native_scratch_size(plan)` complex values.

The library automatically chooses its packing policy. You can inspect it with:

```c
const char* policy = bfft_plan_standard_policy(plan);
```

The policy string is either `fused-scatter-plus-layout-convert` or
`two-phase-standard-pack`. `native_scratch` may be NULL only for
`two-phase-standard-pack`; portable code should allocate and pass the scratch
buffer unconditionally.

The C++ wrappers are:

```cpp
plan.forward(input, output, work, native_scratch);
std::vector<bfft::complex> spectrum = plan.forward(input_vector);
```

The vector overload validates that `input_vector.size() == plan.size()` and
allocates the work buffers internally.

## Magnitude-only forward transform

If an application only needs amplitudes and not phases, use the magnitude-only
forward transform. It writes standard FFT-order magnitudes from bin `0` to
`N/2` directly into a real output buffer. This path still uses the same Bruun
forward transform, but it avoids complex output storage and the native complex
scratch buffer required by the ordinary standard-output double transform.

```c
bfft_forward_magnitude(plan, input, magnitudes, work);
```

Buffer sizes:

- `input`: `N` doubles.
- `magnitudes`: `bfft_plan_bins(plan)` doubles.
- `work`: `bfft_plan_work_size(plan)` doubles.

The output values are `abs(X[k])` for the ordinary real-to-complex FFT bins. DC
and Nyquist bins are real, so their magnitudes are absolute values. Interior
bins use `sqrt(re * re + im * im)`. The C++ wrappers are:

```cpp
plan.forward_magnitude(input, magnitudes, work);
std::vector<double> magnitudes = plan.forward_magnitude(input_vector);
```

The single-precision equivalent is:

```c
bfft_forward_magnitude_f32(plan, input, magnitudes, work);
```

with `N` floats for `input`, `bfft_plan_bins(plan)` floats for `magnitudes`, and
`bfft_plan_work_size_f32(plan)` floats for `work`. The C++ wrappers are
`plan.forward_magnitude_f32(input, magnitudes, work)` and the allocating vector
overload `plan.forward_magnitude_f32(input_vector)`.

## Single-precision transforms

The float32 API uses the same plan size and layout policy as the double API, but
all signal, work, and complex spectrum buffers are single precision.

```c
bfft_forward_f32(plan, input, output, work, native_scratch);
bfft_inverse_f32(plan, output, roundtrip);
```

Buffer sizes:

- `input`: `N` floats.
- `output`: `bfft_plan_bins(plan)` `bfft_complex_f32` values.
- `work`: `bfft_plan_work_size_f32(plan)` floats.
- `native_scratch`: accepted for API symmetry and may be NULL.

The C++ wrappers are:

```cpp
plan.forward_f32(input, output, work);
std::vector<bfft::complex_f32> spectrum = plan.forward_f32(input_vector);
plan.inverse_f32(spectrum, output);
std::vector<float> signal = plan.inverse_f32(spectrum_vector);
```

Native-order float32 calls and converters mirror the double API:

- `bfft_forward_native_f32(plan, input, output, work)`
- `bfft_inverse_native_f32(plan, input, output)`
- `bfft_native_to_standard_f32(plan, native_input, standard_output)`
- `bfft_standard_to_native_f32(plan, standard_input, native_output)`

The implementation keeps float32 storage and float32 butterfly arithmetic. Plans
own rounded float32 twiddle tables so repeated transforms avoid float recurrence
drift. The current native float32 BH7 regression target is 144 dB SFDR after
native-to-standard conversion; the tracked probe measured 144.95579274 dB at
`N = 4194304` on the Apple M4 NEON build used for the first release notes.

## Standard inverse transform

The standard inverse reads ordinary FFT r2c bins and writes `N` time-domain
samples. A forward followed by an inverse returns the original real signal up to
floating-point roundoff.

```c
bfft_inverse(plan, spectrum, output);
```

Buffer sizes:

- `spectrum`: `bfft_plan_bins(plan)` complex values in standard order.
- `output`: `N` doubles.

The C++ wrappers are:

```cpp
plan.inverse(spectrum, output);
std::vector<double> signal = plan.inverse(spectrum_vector);
```

## Native and residue APIs

Advanced users can avoid standard-order conversion:

- `bfft_forward_native(plan, input, output, work)` writes
  `bfft_plan_bins(plan)` complex values in native spectrum order.
- `bfft_inverse_native(plan, input, output)` reads native order and writes `N`
  doubles.
- `bfft_forward_residues(plan, input, residues)` writes Bruun residue
  coordinates as `N` doubles.
- `bfft_inverse_residues(plan, residues_signal)` converts residue coordinates
  back to time samples in place.

Use layout conversion helpers when needed:

- `bfft_native_to_standard(plan, native_input, standard_output)`
- `bfft_standard_to_native(plan, standard_input, native_output)`

Both conversion helpers read and write `bfft_plan_bins(plan)` complex values.

The C++ wrappers keep the same names without the `bfft_` prefix:
`forward_native`, `inverse_native`, `forward_residues`, `inverse_residues`,
`native_to_standard`, and `standard_to_native`.

## Filtering

BFFT can convert a standard frequency response into residue-domain filter
coefficients once, then apply the filter with a streaming multiply.

```c
size_t filter_size = bfft_filter_size(plan);
bfft_residue_filter_from_standard(plan, response, residue_filter);
bfft_filter_signal(plan, input, residue_filter, output);
```

Buffer sizes:

- `response`: `bfft_plan_bins(plan)` complex values in standard order.
- `residue_filter`: `bfft_filter_size(plan)` doubles.
- `input` and `output`: `N` doubles.

For real zero-phase responses, use `bfft_residue_filter_from_real` with
`bfft_plan_bins(plan)` doubles. `bfft_apply_residue_filter` applies a residue
filter in place to an existing `N`-double residue vector.

The C++ wrappers are `filter_size`, `residue_filter_from_standard`,
`residue_filter_from_real`, `apply_residue_filter`, and `filter_signal`.

## Error handling

C functions return `bfft_status`:

- `BFFT_OK`
- `BFFT_ERROR_INVALID_ARGUMENT`
- `BFFT_ERROR_ALLOCATION`
- `BFFT_ERROR_INTERNAL`

Use `bfft_status_string` for diagnostics. C functions return
`BFFT_ERROR_INVALID_ARGUMENT` when a required plan or buffer pointer is NULL.
C++ methods throw `bfft::error`; call `error.code()` to retrieve the original
`bfft_status`.
