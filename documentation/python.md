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

Override the compiler with the `CXX` environment variable, and pass extra flags
with `BFFT_CXXFLAGS` if needed:

```sh
CXX=clang++ BFFT_CXXFLAGS="-march=native" pip install .
```

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

## API

| Function | Equivalent | Notes |
| --- | --- | --- |
| `bfft.rfft(x)` | `numpy.fft.rfft(x)` | Power-of-two `N >= 4`. Returns `N/2 + 1` complex bins. |
| `bfft.irfft(X, n=None)` | `numpy.fft.irfft(X, n)` | `n` defaults to `2 * (len(X) - 1)`. Returns `N` real samples. |
| `bfft.odft(x)` | half-bin phase shift + `rfft` | `H[k] = sum_n x[n] exp(-2j*pi*(k+1/2)*n/N)`, `N >= 2`. Returns `N/2` complex bins. |
| `bfft.iodft(H, n=None)` | inverse of `bfft.odft` | `n` defaults to `2 * len(H)`. Returns `N` real samples. |

All Python transforms operate on power-of-two lengths in double precision. The
forward and inverse pairs round-trip to floating-point precision, and `rfft` and
`irfft` match `numpy.fft` to within floating-point error.
