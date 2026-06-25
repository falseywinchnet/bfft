# Core concepts

## Plans

A plan owns reusable transform metadata. Create one plan for a given transform
size and reuse it for repeated transforms.

BFFT real FFT plans accept any size `N >= 2`; power-of-two sizes use the native
Bruun fast path, while other sizes use the generalized Bruun path. BODFT plans
require a power-of-two size `N >= 2`.

## Buffer ownership

The C API uses caller-owned buffers. Query the plan for every buffer size:

| Function | Meaning |
| --- | --- |
| `bfft_plan_size` | Transform length `N`. |
| `bfft_plan_bins` | Standard real-to-complex bin count, `N / 2 + 1`. |
| `bfft_plan_work_size` | Double-precision work buffer length. |
| `bfft_plan_work_size_f32` | Single-precision work buffer length. |
| `bfft_plan_native_scratch_size` | Scratch complex values for standard output. |
| `bfft_filter_size` | Residue-domain filter length. |

The C++ wrapper provides the same values as methods on `bfft::plan`.

## Layouts

BFFT exposes three spectrum layouts.

### Standard layout

Standard layout is ordinary FFT-order real-to-complex output. It has `N / 2 + 1`
complex bins and is the right default for most applications.

### Native layout

Native layout is the internal BFFT order. Use it when performance matters and
when downstream code can consume native order directly.

### Residue layout

Residue layout is used by residue-domain transforms and filtering. It is useful
for pipelines that can avoid conversion back to standard complex spectrum form.

## Threading and reentrancy

Plans are reusable metadata. Transform calls use caller-provided buffers. For
concurrent transforms, give each thread its own work buffers and workspace.
