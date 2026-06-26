# C++ API reference

Include the C++ wrapper with:

```cpp
#include <bfft/bfft.hpp>
```

The C++ API wraps the C ABI with RAII and exceptions.

## Aliases

- `bfft::complex` aliases `bfft_complex`.
- `bfft::complex_f32` aliases `bfft_complex_f32`.
- `bfft::layout` aliases `bfft_layout`.
- `bfft::status` aliases `bfft_status`.

## Diagnostics

- `bfft::version_string()` returns the library version.
- `bfft::backend_name()` returns the selected SIMD backend.
- `bfft::error` is thrown when a wrapped C API call returns an error.

## `bfft::plan`

Create a real FFT plan with any size `N >= 2`:

```cpp
bfft::plan plan(1024);
```

Power-of-two plans use the native Bruun kernel. Other sizes use the generalized
Bruun plan behind the same methods.

Important methods:

- `size()` returns `N`.
- `bins()` returns `N / 2 + 1`.
- `work_size()` returns double work buffer length.
- `work_size_f32()` returns float work buffer length.
- `native_scratch_size()` returns standard-output scratch length.
- `standard_policy()` returns the standard-output packing policy.
- `create_workspace()` creates aligned scratch storage.

## Transform methods

Double precision:

- `forward(...)`
- `forward_native(...)`
- `forward_magnitude(...)`
- `inverse(...)`
- `inverse_native(...)`
- `forward_residues(...)`
- `inverse_residues(...)`

Single precision:

- `forward_f32(...)`
- `forward_native_f32(...)`
- `forward_magnitude_f32(...)`
- `inverse_f32(...)`
- `inverse_native_f32(...)`

Convenience overloads that accept `std::vector` allocate temporary work buffers
and return output vectors. Pointer overloads use caller-owned buffers.

## Residue filtering methods

- `filter_size()` returns the residue filter length.
- `residue_filter_from_standard(...)` converts a complex response.
- `residue_filter_from_real(...)` converts a real zero-phase response.
- `apply_residue_filter(...)` applies a filter in place.
- `filter_signal(...)` filters an input signal into an output buffer.

Residue filtering is power-of-two-only. Standard, native-order, magnitude, and
single-precision transform methods route for arbitrary-N plans.


## STFT plan

Include `<bfft/stft.hpp>` to use `bfft::stft_plan`, a fixed-configuration
short-time transform with an internal streaming inverse buffer.

```cpp
#include <bfft/stft.hpp>
#include <vector>

bfft::stft_plan tf(24576, 512, 128);
std::vector<double> x(tf.n());
std::vector<bfft::complex> Zx = tf.forward(x); // row-major bins x segments
std::vector<double> y = tf.inverse(Zx);
tf.reset_buffer();
```

Construct with `bfft::stft_odft` to use the half-bin ODFT frame transform, or
pass a window pointer/vector of length `n_fft`. `bfft::hann_window(n_fft)`
returns the default Hann analysis window.
