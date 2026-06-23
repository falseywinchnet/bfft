"""Call BFFT from inside Numba ``@njit(nopython=True)`` code.

``numpy.fft`` cannot be used from nopython Numba: it is a Python C-extension that
only exists in object mode, so Numba has no way to lower it. BFFT is the
opposite -- a plain C ABI that takes raw pointers -- so it can be invoked from
nopython code through Numba's first-class cffi support.

This module is optional and imported on demand (``import bfft.numba_support``).
It requires ``cffi``; Numba is only needed to actually JIT-compile callers.

Usage
-----

Plans and buffers are created once in normal Python, then the transform is
called from inside an ``@njit`` function. Two rules make it work with Numba:

* Pass the **plan as the integer address** returned by ``make_plan`` (Numba
  cannot type a raw cffi pointer, but it types an int fine; the C ABI takes a
  pointer-sized integer).
* Pass complex buffers as their **float64 view** (``buf.view(np.float64)``),
  since ``ffi.from_buffer`` yields a ``double*`` the C function expects.

::

    import numpy as np
    from numba import njit
    import bfft.numba_support as bn
    from bfft.numba_support import bfft_forward, ffi

    N = 4096
    plan, bins, work_n, scratch_n = bn.make_plan(N)

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

The buffers are caller-owned, so a JIT-compiled loop that transforms many frames
performs each FFT with no Python-object interaction at all -- something
``numpy.fft`` cannot do from nopython mode.
"""

from __future__ import annotations

import cffi
import numpy as np

from ._core import _candidate_paths

ffi = cffi.FFI()

# Pointer arguments are declared as the element type the C ABI uses. cffi/Numba
# accept arrays passed through ffi.from_buffer for these.
# Complex buffers are declared as double* (complex128 is bit-identical to a pair
# of doubles): Numba marshals double* from a float64 array via ffi.from_buffer,
# whereas a named struct pointer cannot be typed. Pass the float64 *view* of any
# complex buffer (e.g. out.view(np.float64)) at the call site.
ffi.cdef(
    """
    int    bfft_plan_create(size_t n, void** plan);
    void   bfft_plan_destroy(void* plan);
    size_t bfft_plan_bins(void* plan);
    size_t bfft_plan_work_size(void* plan);
    size_t bfft_plan_native_scratch_size(void* plan);
    size_t bfft_plan_work_size_f32(void* plan);
    int    bfft_forward(size_t plan, double* input, double* output,
                        double* work, double* native_scratch);
    int    bfft_inverse(size_t plan, double* input, double* output);
    int    bfft_forward_f32(size_t plan, float* input, float* output,
                            float* work, float* native_scratch);
    int    bfft_inverse_f32(size_t plan, float* input, float* output);

    int    bodft_plan_create(size_t n, void** plan);
    void   bodft_plan_destroy(void* plan);
    size_t bodft_plan_bins(void* plan);
    int    bodft_forward(size_t plan, double* input, double* output);
    int    bodft_inverse(size_t plan, double* input, double* output);
    int    bodft_forward_f32(size_t plan, float* input, float* output);
    int    bodft_inverse_f32(size_t plan, float* input, float* output);
    """
)


def _open():
    last_err = None
    for path in _candidate_paths():
        try:
            return ffi.dlopen(path)
        except OSError as exc:
            last_err = exc
    raise OSError(
        "bfft.numba_support could not load the BFFT native library. "
        f"Last error: {last_err}"
    )


lib = _open()

# Numba types cffi functions only when they are referenced as direct globals
# (not via attribute access on the library object), so bind each one to a
# module-level name. Inside @njit code, import and call these directly:
#     from bfft.numba_support import bfft_forward, ffi
bfft_forward = lib.bfft_forward
bfft_inverse = lib.bfft_inverse
bodft_forward = lib.bodft_forward
bodft_inverse = lib.bodft_inverse
# Single-precision counterparts. Pass complex64 buffers as their float32 view.
bfft_forward_f32 = lib.bfft_forward_f32
bfft_inverse_f32 = lib.bfft_inverse_f32
bodft_forward_f32 = lib.bodft_forward_f32
bodft_inverse_f32 = lib.bodft_inverse_f32

# Register the functions with Numba so they are callable from nopython code.
try:  # pragma: no cover - exercised only where numba is present
    from numba.core.typing import cffi_utils as _cffi_utils

    for _fn in (bfft_forward, bfft_inverse, bodft_forward, bodft_inverse,
                bfft_forward_f32, bfft_inverse_f32,
                bodft_forward_f32, bodft_inverse_f32):
        _cffi_utils.register_function(_fn)
except Exception:  # numba missing or API drift: cffi still works in object mode
    pass


# Keep created plan cdata alive for the process lifetime; callers hold only the
# integer address, which would not otherwise pin the underlying object.
_live_plans = []


def _plan(create, n):
    pp = ffi.new("void**")
    if create(n, pp) != 0:
        raise RuntimeError(f"plan creation failed for N={n}")
    plan = pp[0]
    _live_plans.append(plan)
    return plan


def make_plan(n, dtype=np.float64):
    """Create a standard real-FFT plan. Returns ``(plan_addr, bins, work_n,
    scratch_n)`` where ``plan_addr`` is the plan's address as an int (pass it as
    the first argument to ``bfft_forward`` / ``bfft_inverse`` inside ``@njit``
    code) and the integers are the required buffer lengths.

    ``dtype`` selects the work-buffer sizing for the precision you will use:
    ``np.float64`` for ``bfft_forward`` or ``np.float32`` for
    ``bfft_forward_f32``. ``bins`` and ``scratch_n`` count complex elements (of
    the matching precision); allocate ``scratch_n`` complex values and pass its
    real view. ``work_n`` counts real elements of the chosen precision."""
    plan = _plan(lib.bfft_plan_create, n)
    if np.dtype(dtype) == np.float32:
        work_n = int(lib.bfft_plan_work_size_f32(plan))
    else:
        work_n = int(lib.bfft_plan_work_size(plan))
    return (
        int(ffi.cast("uintptr_t", plan)),
        int(lib.bfft_plan_bins(plan)),
        work_n,
        int(lib.bfft_plan_native_scratch_size(plan)),
    )


def make_odft_plan(n):
    """Create a half-bin ODFT plan. Returns ``(plan_addr, bins)``."""
    plan = _plan(lib.bodft_plan_create, n)
    return int(ffi.cast("uintptr_t", plan)), int(lib.bodft_plan_bins(plan))
