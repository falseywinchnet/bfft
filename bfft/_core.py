"""ctypes binding to the BFFT shared library.

Loads the native library (bundled in this package, or a system install from
`make install`) and exposes typed access to the forward transforms used by the
public shim in :mod:`bfft`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent


class _Complex(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


def _candidate_paths():
    # 1. Explicit override.
    env = os.environ.get("BFFT_LIBRARY")
    if env:
        yield env
    # 2. Library bundled inside this package (the pip-install path).
    for pat in ("_libbfft.so", "_libbfft.dylib", "_libbfft.dll"):
        for hit in glob.glob(str(_PKG_DIR / pat)):
            yield hit
    # 3. A build tree next to a source checkout (editable / dev use).
    for rel in ("../build/libbfft.so", "../build/libbfft.dylib"):
        p = _PKG_DIR / rel
        if p.exists():
            yield str(p)
    # 4. A system install from `make install`.
    found = ctypes.util.find_library("bfft")
    if found:
        yield found
    yield "libbfft.so"
    if sys.platform == "darwin":
        yield "libbfft.dylib"


def _load_library() -> ctypes.CDLL:
    last_err = None
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(path)
        except OSError as exc:  # pragma: no cover - depends on platform
            last_err = exc
    raise OSError(
        "Could not locate the BFFT native library. Build it with `pip install .` "
        "or `make && make install`, or set BFFT_LIBRARY to the shared object. "
        f"Last error: {last_err}"
    )


_lib = _load_library()


def _decl(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_dbl_p = ctypes.POINTER(ctypes.c_double)
_cplx_p = ctypes.POINTER(_Complex)
_plan_p = ctypes.c_void_p

# --- standard real FFT (bfft.h) ---
_bfft_plan_create = _decl("bfft_plan_create", ctypes.c_int,
                          [ctypes.c_size_t, ctypes.POINTER(_plan_p)])
_bfft_plan_destroy = _decl("bfft_plan_destroy", None, [_plan_p])
_bfft_plan_bins = _decl("bfft_plan_bins", ctypes.c_size_t, [_plan_p])
_bfft_plan_work_size = _decl("bfft_plan_work_size", ctypes.c_size_t, [_plan_p])
_bfft_plan_native_scratch_size = _decl(
    "bfft_plan_native_scratch_size", ctypes.c_size_t, [_plan_p])
_bfft_forward = _decl("bfft_forward", ctypes.c_int,
                      [_plan_p, _dbl_p, _cplx_p, _dbl_p, _cplx_p])
_bfft_inverse = _decl("bfft_inverse", ctypes.c_int, [_plan_p, _cplx_p, _dbl_p])

# --- half-bin ODFT (bodft.h) ---
_bodft_plan_create = _decl("bodft_plan_create", ctypes.c_int,
                           [ctypes.c_size_t, ctypes.POINTER(_plan_p)])
_bodft_plan_destroy = _decl("bodft_plan_destroy", None, [_plan_p])
_bodft_plan_bins = _decl("bodft_plan_bins", ctypes.c_size_t, [_plan_p])
_bodft_forward = _decl("bodft_forward", ctypes.c_int, [_plan_p, _dbl_p, _cplx_p])
_bodft_inverse = _decl("bodft_inverse", ctypes.c_int, [_plan_p, _cplx_p, _dbl_p])

_OK = 0


def _check(status, what):
    if status != _OK:
        raise RuntimeError(f"{what} failed with BFFT status {status}")


def _as_f64_1d(x):
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("bfft transforms expect a 1-D real input array")
    return a


def _as_c128_1d(x):
    a = np.ascontiguousarray(x, dtype=np.complex128)
    if a.ndim != 1:
        raise ValueError("bfft inverse transforms expect a 1-D complex spectrum")
    return a


def _is_pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


# Plans are reusable per size; cache them so repeated calls avoid setup cost.
_bfft_plans: dict[int, _plan_p] = {}
_bodft_plans: dict[int, _plan_p] = {}


def _bfft_plan(n):
    p = _bfft_plans.get(n)
    if p is None:
        p = _plan_p()
        _check(_bfft_plan_create(n, ctypes.byref(p)), "bfft_plan_create")
        _bfft_plans[n] = p
    return p


def _bodft_plan(n):
    p = _bodft_plans.get(n)
    if p is None:
        p = _plan_p()
        _check(_bodft_plan_create(n, ctypes.byref(p)), "bodft_plan_create")
        _bodft_plans[n] = p
    return p


def rfft(x):
    """Real-to-complex FFT. Drop-in for :func:`numpy.fft.rfft` (power-of-two)."""
    a = _as_f64_1d(x)
    n = a.shape[0]
    if not _is_pow2(n) or n < 4:
        raise ValueError("bfft.rfft requires a power-of-two length N >= 4")
    plan = _bfft_plan(n)
    bins = _bfft_plan_bins(plan)
    out = np.empty(bins, dtype=np.complex128)
    work = np.empty(_bfft_plan_work_size(plan), dtype=np.float64)
    scratch = np.empty(_bfft_plan_native_scratch_size(plan), dtype=np.complex128)
    _check(_bfft_forward(
        plan,
        a.ctypes.data_as(_dbl_p),
        out.ctypes.data_as(_cplx_p),
        work.ctypes.data_as(_dbl_p),
        scratch.ctypes.data_as(_cplx_p),
    ), "bfft_forward")
    return out


def irfft(x, n=None):
    """Inverse real FFT. Drop-in for :func:`numpy.fft.irfft` (power-of-two).

    ``x`` holds ``N/2 + 1`` complex bins. ``n`` is the length of the real output;
    when omitted it defaults to ``2 * (len(x) - 1)``, matching numpy."""
    a = _as_c128_1d(x)
    if n is None:
        n = 2 * (a.shape[0] - 1)
    if not _is_pow2(n) or n < 4:
        raise ValueError("bfft.irfft requires a power-of-two output length N >= 4")
    plan = _bfft_plan(n)
    bins = _bfft_plan_bins(plan)
    if a.shape[0] != bins:
        raise ValueError(
            f"bfft.irfft expected {bins} bins for N={n}, got {a.shape[0]}")
    out = np.empty(n, dtype=np.float64)
    _check(_bfft_inverse(
        plan,
        a.ctypes.data_as(_cplx_p),
        out.ctypes.data_as(_dbl_p),
    ), "bfft_inverse")
    return out


def odft(x):
    """Half-bin-shifted real transform: H[k] = sum_n x[n] exp(-2j*pi*(k+1/2)*n/N),
    k = 0 .. N/2-1. Equivalent to a half-bin phase shift followed by an rfft."""
    a = _as_f64_1d(x)
    n = a.shape[0]
    if not _is_pow2(n) or n < 2:
        raise ValueError("bfft.odft requires a power-of-two length N >= 2")
    plan = _bodft_plan(n)
    bins = _bodft_plan_bins(plan)
    out = np.empty(bins, dtype=np.complex128)
    _check(_bodft_forward(
        plan,
        a.ctypes.data_as(_dbl_p),
        out.ctypes.data_as(_cplx_p),
    ), "bodft_forward")
    return out


def iodft(x, n=None):
    """Inverse of :func:`odft`. Maps ``N/2`` packed half-bin bins back to the
    ``N`` real samples. ``n`` defaults to ``2 * len(x)``."""
    a = _as_c128_1d(x)
    if n is None:
        n = 2 * a.shape[0]
    if not _is_pow2(n) or n < 2:
        raise ValueError("bfft.iodft requires a power-of-two output length N >= 2")
    plan = _bodft_plan(n)
    bins = _bodft_plan_bins(plan)
    if a.shape[0] != bins:
        raise ValueError(
            f"bfft.iodft expected {bins} bins for N={n}, got {a.shape[0]}")
    out = np.empty(n, dtype=np.float64)
    _check(_bodft_inverse(
        plan,
        a.ctypes.data_as(_cplx_p),
        out.ctypes.data_as(_dbl_p),
    ), "bodft_inverse")
    return out
