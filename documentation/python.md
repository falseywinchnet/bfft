# Python bindings

BFFT ships a small ctypes-based Python package that exposes numpy-friendly
drop-in transforms. No prebuilt binaries are distributed: installation compiles
the native library from source on your machine.

## Install

From a clone of the repository:

```sh
pip install .
```

This compiles `src/bfft.cpp` and `src/bodft.cpp` with your C++ compiler and
bundles the resulting shared library inside the installed package. The only
requirements are a C++17 compiler and NumPy.

### Optimization flags

Because the library is compiled on your own machine (no prebuilt binaries are
distributed), the build tunes for the local CPU by default. It selects, when the
compiler accepts them:

- `-O3`
- `-march=native` (or `-mcpu=native` on Apple-silicon clang) — emit AVX2/AVX-512
  and other host-specific instructions.
- `-ffast-math` — relaxed floating-point for faster math.

Each flag is probed against your compiler first, so the build degrades
gracefully on toolchains that lack them. Control the defaults with environment
variables:

| Variable | Effect |
| --- | --- |
| `BFFT_NO_NATIVE=1` | Skip `-march=native` / `-mcpu=native` (portable codegen). |
| `BFFT_NO_FAST_MATH=1` | Keep strict IEEE math (drop `-ffast-math`). |
| `CXX=...` | Choose the compiler. |
| `BFFT_CXXFLAGS="..."` | Append extra flags to the compile. |

```sh
CXX=clang++ BFFT_NO_FAST_MATH=1 pip install .
```

`-ffast-math` assumes no NaNs/infinities and reorders operations, so results may
differ in the last bits from a strict-IEEE build (still accurate to
floating-point precision for these transforms). It also enables
flush-to-zero/denormals-are-zero for the process when the library loads, which
can affect denormal handling elsewhere. Set `BFFT_NO_FAST_MATH=1` if you need
bit-reproducible or strict denormal behavior.

### Using a system-installed library

If you prefer to build the native library separately, install it first and the
Python loader will discover it automatically:

```sh
make && sudo make install PREFIX=/usr/local
pip install .
```

You can also point the loader at a specific shared object:

```sh
export BFFT_LIBRARY=/path/to/libbfft.so
```

The loader searches, in order: `$BFFT_LIBRARY`, the library bundled in the
package, a sibling `build/` directory in a source checkout, and finally the
system library search path.

## Usage

```python
import numpy as np
import bfft

x = np.random.randn(1024)          # power-of-two length

X = bfft.rfft(x)                   # == numpy.fft.rfft(x)        -> N/2 + 1 bins
x_back = bfft.irfft(X)             # == numpy.fft.irfft(X)       -> N samples

H = bfft.odft(x)                   # half-bin-shifted transform  -> N/2 bins
x_back2 = bfft.iodft(H)            # inverse of odft             -> N samples
```

The module-level functions cache the plan, the transform sizes, and reusable
scratch buffers per length internally, so repeated calls at the same size avoid
re-creating that state. They stay safe to call from multiple threads: a
concurrent call at the same size that cannot reuse the shared scratch falls back
to a private buffer.

### Planned objects (hot loops)

For the lowest per-call overhead -- transforming the same size repeatedly in a
tight loop -- use a planned object. It caches everything except the unavoidable
output allocation and input-pointer fetch:

```python
plan = bfft.Plan(N)                # standard real FFT at fixed size N
X = plan.rfft(x)                   # == numpy.fft.rfft(x)
x_back = plan.irfft(X)             # == numpy.fft.irfft(X)

oplan = bfft.OdftPlan(N)           # half-bin transform at fixed size N
H = oplan.odft(x)
x_back2 = oplan.iodft(H)
```

A planned object owns shared scratch and is **not thread-safe**: create one plan
per thread (or use the module-level functions, which guard against concurrent
use).

Pass a caller-owned output buffer with ``out=`` to avoid the per-call output
allocation entirely (the pyfftw-style zero-allocation loop):

```python
plan = bfft.Plan(N)
out = np.empty(plan.bins, np.complex128)   # allocate once
for chunk in stream:                        # chunk has length N
    plan.rfft(chunk, out=out)               # writes into out, no allocation
```

`out=` is accepted by `Plan.rfft`/`irfft` and `OdftPlan.odft`/`iodft`; it must
be a C-contiguous array of the right dtype and length, and is returned as-is.

### Calling BFFT from Numba (`@njit`)

`numpy.fft` cannot be called from `@njit(nopython=True)` code -- it is a Python
C-extension that only exists in object mode. BFFT *can*, because it is a plain C
ABI taking raw pointers, which Numba lowers through its cffi support. `cffi` is a
dependency of BFFT, so only `numba` itself needs to be installed alongside it:

```python
import numpy as np
from numba import njit
import bfft.numba_support as bn
from bfft.numba_support import bfft_forward, ffi

N = 4096
plan, bins, work_n, scratch_n = bn.make_plan(N)   # plan is an int address

@njit(cache=True)
def rfft_into(plan, x, out_f64, work, scratch_f64):
    bfft_forward(plan,
                 ffi.from_buffer(x), ffi.from_buffer(out_f64),
                 ffi.from_buffer(work), ffi.from_buffer(scratch_f64))

x = np.random.randn(N)
out = np.empty(bins, np.complex128)
work = np.empty(work_n, np.float64)
scratch = np.empty(scratch_n, np.complex128)
rfft_into(plan, x, out.view(np.float64), work, scratch.view(np.float64))
# out == numpy.fft.rfft(x)
```

Two rules make it work with Numba: pass the **plan as the integer address** from
`make_plan` (Numba can type an int but not a raw cffi pointer), and pass complex
buffers as their **real view** (`buf.view(np.float64)`, or `buf.view(np.float32)`
for single precision) so `ffi.from_buffer` yields the pointer type the C function
expects. A JIT-compiled loop then performs each transform with no Python-object
interaction -- in practice at the bare C transform speed.

All four transforms are available from `@njit` in both precisions:

| double (`float64` / `complex128`) | single (`float32` / `complex64`) | transform |
| --- | --- | --- |
| `bfft_forward` | `bfft_forward_f32` | real FFT (rfft) |
| `bfft_inverse` | `bfft_inverse_f32` | inverse real FFT (irfft) |
| `bodft_forward` | `bodft_forward_f32` | half-bin ODFT (odft) |
| `bodft_inverse` | `bodft_inverse_f32` | inverse ODFT (iodft) |

Create plans with `make_plan(N)` for the standard real FFT (pass
`dtype=np.float32` to size the single-precision work buffer) and
`make_odft_plan(N)` for the ODFT. Both plan helpers return the same
`(plan, bins, work_n, scratch_n)` tuple shape. The ODFT helpers return zero
for `work_n` and `scratch_n`, and the exported `bodft_forward` /
`bodft_forward_f32` callables accept the same `work` and `native_scratch`
arguments as `bfft_forward` / `bfft_forward_f32` while ignoring them. This lets
the same jitted call site switch between rfft and odft by changing only the plan
factory and transform function.

## API

| Function | Equivalent | Notes |
| --- | --- | --- |
| `bfft.rfft(x)` | `numpy.fft.rfft(x)` | Power-of-two `N >= 4`. Returns `N/2 + 1` complex bins. |
| `bfft.irfft(X, n=None)` | `numpy.fft.irfft(X, n)` | `n` defaults to `2 * (len(X) - 1)`. Returns `N` real samples. |
| `bfft.odft(x)` | half-bin phase shift + `rfft` | `H[k] = sum_n x[n] exp(-2j*pi*(k+1/2)*n/N)`, `N >= 2`. Returns `N/2` complex bins. |
| `bfft.iodft(H, n=None)` | inverse of `bfft.odft` | `n` defaults to `2 * len(H)`. Returns `N` real samples. |

Python real FFT transforms operate on power-of-two `N >= 4` in double precision. The
forward and inverse pairs round-trip to floating-point precision, and `rfft` and
`irfft` match `numpy.fft` to within floating-point error.


## Short-time transforms

`bfft.STFTPlan` is a reusable native plan for streaming short-time Fourier
transforms. It uses the same BFFT real FFT or BODFT half-bin transform for every
frame, returns a two-dimensional NumPy `complex128` spectrogram with shape
`(n_bins, n_segs)`, and stores the inverse overlap-add buffer inside the plan.
Call `reset_buffer()` before starting a fresh inverse stream.

```python
import numpy as np
import bfft

tf = bfft.STFTPlan(n=24576, n_fft=512, hop_length=128)
x = np.random.randn(tf.n)
Zx = tf.stft(x)       # complex128, shape (tf.n_bins, tf.n_segs)
y = tf.istft(Zx)     # float64, length tf.n_segs * tf.hop_length
tf.reset_buffer()    # clear streaming overlap state
```

Pass `transform="odft"` to use the half-bin ODFT path. Pass a 1-D float64
window of length `n_fft` to override the default Hann window; the native plan
derives the matching MSE-optimal synthesis window. `bfft.hann_window(n_fft)`
returns the exact default window used by native STFT plans.
