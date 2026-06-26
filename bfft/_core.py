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
import threading
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
_void_p = ctypes.c_void_p
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
                      [_plan_p, _void_p, _void_p, _void_p, _void_p])
_bfft_inverse = _decl("bfft_inverse", ctypes.c_int, [_plan_p, _void_p, _void_p])

# --- half-bin ODFT (bodft.h) ---
_bodft_plan_create = _decl("bodft_plan_create", ctypes.c_int,
                           [ctypes.c_size_t, ctypes.POINTER(_plan_p)])
_bodft_plan_destroy = _decl("bodft_plan_destroy", None, [_plan_p])
_bodft_plan_bins = _decl("bodft_plan_bins", ctypes.c_size_t, [_plan_p])
_bodft_forward = _decl("bodft_forward", ctypes.c_int, [_plan_p, _void_p, _void_p])
_bodft_inverse = _decl("bodft_inverse", ctypes.c_int, [_plan_p, _void_p, _void_p])

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


def _out_buffer(out, length, dtype, what):
    """Return a length-``length`` ``dtype`` output buffer: validate a caller-
    supplied ``out`` (reused, zero-allocation path) or allocate a fresh one."""
    if out is None:
        return np.empty(length, dtype=dtype)
    if out.dtype != dtype or out.shape != (length,):
        raise ValueError(
            f"{what} out must be a C-contiguous {dtype} array of shape ({length},)")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{what} out must be C-contiguous")
    return out


def _check_rfft_n(n):
    # Power-of-two N >= 4 uses the native Bruun kernel; any other N >= 2 uses the
    # arbitrary-N generalized-Bruun plan (z^N-1 factorization). Both go through the
    # same C ABI plan; the arbitrary-N plan owns its scratch (work_size == 0).
    if n < 2:
        raise ValueError("bfft real FFT requires a length N >= 2")


def _check_odft_n(n):
    if not _is_pow2(n) or n < 2:
        raise ValueError("bfft ODFT requires a power-of-two length N >= 2")


# --------------------------------------------------------------------------
# Planned objects: cache the plan, sizes, and reusable scratch buffers (with
# their raw pointers) so the only per-call costs are one output allocation and
# one input-pointer fetch -- matching numpy's own minimum and chasing pyfftw.
#
# A Plan owns shared scratch and is therefore NOT thread-safe: use one plan per
# thread (or use the module-level rfft/odft, which guard against concurrent use).
# --------------------------------------------------------------------------

class Plan:
    """Reusable plan for the standard real FFT at a fixed size N (any N >= 2;
    power-of-two N uses the native kernel, other N the generalized Bruun plan).

    ``Plan(N).rfft(x)`` and ``Plan(N).irfft(X)`` are the low-overhead, hot-loop
    counterparts of :func:`rfft` / :func:`irfft`. Not thread-safe; create one
    plan per thread."""

    __slots__ = ("n", "bins", "_plan", "_work", "_scratch", "_wp", "_sp")

    def __init__(self, n):
        n = int(n)
        _check_rfft_n(n)
        self.n = n
        self._plan = _bfft_plan(n)
        self.bins = _bfft_plan_bins(self._plan)
        self._work = np.empty(_bfft_plan_work_size(self._plan), dtype=np.float64)
        self._scratch = np.empty(
            _bfft_plan_native_scratch_size(self._plan), dtype=np.complex128)
        self._wp = self._work.ctypes.data
        self._sp = self._scratch.ctypes.data

    def rfft(self, x, out=None):
        a = _as_f64_1d(x)
        if a.shape[0] != self.n:
            raise ValueError(f"Plan(N={self.n}).rfft expects length {self.n}")
        out = _out_buffer(out, self.bins, np.complex128, "Plan.rfft")
        _check(_bfft_forward(self._plan, a.ctypes.data, out.ctypes.data,
                             self._wp, self._sp), "bfft_forward")
        return out

    def irfft(self, x, out=None):
        a = _as_c128_1d(x)
        if a.shape[0] != self.bins:
            raise ValueError(
                f"Plan(N={self.n}).irfft expects {self.bins} bins")
        out = _out_buffer(out, self.n, np.float64, "Plan.irfft")
        _check(_bfft_inverse(self._plan, a.ctypes.data, out.ctypes.data),
               "bfft_inverse")
        return out


class OdftPlan:
    """Reusable plan for the half-bin ODFT at a fixed power-of-two size N.

    ``OdftPlan(N).odft(x)`` / ``.iodft(H)`` mirror :func:`odft` / :func:`iodft`
    with cached state. Not thread-safe; create one plan per thread."""

    __slots__ = ("n", "bins", "_plan")

    def __init__(self, n):
        n = int(n)
        _check_odft_n(n)
        self.n = n
        self._plan = _bodft_plan(n)
        self.bins = _bodft_plan_bins(self._plan)

    def odft(self, x, out=None):
        a = _as_f64_1d(x)
        if a.shape[0] != self.n:
            raise ValueError(f"OdftPlan(N={self.n}).odft expects length {self.n}")
        out = _out_buffer(out, self.bins, np.complex128, "OdftPlan.odft")
        _check(_bodft_forward(self._plan, a.ctypes.data, out.ctypes.data),
               "bodft_forward")
        return out

    def iodft(self, x, out=None):
        a = _as_c128_1d(x)
        if a.shape[0] != self.bins:
            raise ValueError(
                f"OdftPlan(N={self.n}).iodft expects {self.bins} bins")
        out = _out_buffer(out, self.n, np.float64, "OdftPlan.iodft")
        _check(_bodft_inverse(self._plan, a.ctypes.data, out.ctypes.data),
               "bodft_inverse")
        return out



# --- short-time Fourier transform (stft.h) ---
_bfft_stft_plan_p = ctypes.c_void_p
_bfft_stft_plan_create = _decl("bfft_stft_plan_create", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _dbl_p, ctypes.c_int,
     ctypes.POINTER(_bfft_stft_plan_p)])
_bfft_stft_plan_destroy = _decl("bfft_stft_plan_destroy", None, [_bfft_stft_plan_p])
_bfft_stft_plan_bins = _decl("bfft_stft_plan_bins", ctypes.c_size_t, [_bfft_stft_plan_p])
_bfft_stft_plan_segments = _decl("bfft_stft_plan_segments", ctypes.c_size_t, [_bfft_stft_plan_p])
_bfft_stft_plan_buffer_length = _decl("bfft_stft_plan_buffer_length", ctypes.c_size_t, [_bfft_stft_plan_p])
_bfft_stft_hann_window = _decl("bfft_stft_hann_window", ctypes.c_int, [ctypes.c_size_t, _dbl_p])
_bfft_stft_reset_buffer = _decl("bfft_stft_reset_buffer", ctypes.c_int, [_bfft_stft_plan_p])
_bfft_stft_forward = _decl("bfft_stft_forward", ctypes.c_int,
    [_bfft_stft_plan_p, _void_p, _void_p])
_bfft_stft_inverse = _decl("bfft_stft_inverse", ctypes.c_int,
    [_bfft_stft_plan_p, _void_p, _void_p])


def hann_window(n_fft):
    """Return BFFT's default STFT Hann window as a float64 NumPy array."""
    n_fft = int(n_fft)
    out = np.empty(n_fft, dtype=np.float64)
    _check(_bfft_stft_hann_window(n_fft, out.ctypes.data_as(_dbl_p)),
           "bfft_stft_hann_window")
    return out


class STFTPlan:
    """Reusable BFFT-backed STFT/ISTFT plan.

    Parameters
    ----------
    n : int
        Real input block length. It must be positive and divisible by
        ``hop_length``.
    n_fft : int
        Frame and FFT length. It must be an even power of two at least 4.
    hop_length : int
        Samples between adjacent frames. It must divide ``n_fft``.
    window : array_like or None
        Optional analysis window of length ``n_fft``. If omitted, BFFT generates
        a Hann window. The matching MSE-optimal synthesis window is derived in
        the native plan.
    transform : {"rfft", "odft"}
        Per-frame transform. ``"rfft"`` returns ``n_fft // 2 + 1`` bins;
        ``"odft"`` returns ``n_fft // 2`` half-bin shifted bins.

    Notes
    -----
    The inverse owns a persistent overlap-add buffer inside the plan. Repeated
    ``istft`` calls continue the stream; call :meth:`reset_buffer` before
    starting a fresh stream. One plan is not thread-safe because it owns scratch
    and streaming state.
    """

    __slots__ = ("n", "n_fft", "hop_length", "n_bins", "n_segs", "buffer_length",
                 "transform", "_plan", "_window")

    def __init__(self, n=24576, n_fft=512, hop_length=128, window=None,
                 transform="rfft"):
        n = int(n)
        n_fft = int(n_fft)
        hop_length = int(hop_length)
        t = str(transform).lower()
        if t == "rfft":
            kind = 0
        elif t == "odft":
            kind = 1
        else:
            raise ValueError("transform must be either 'rfft' or 'odft'")
        if window is None:
            w_ptr = None
            self._window = None
        else:
            w = np.ascontiguousarray(window, dtype=np.float64)
            if w.shape != (n_fft,):
                raise ValueError(f"window must be a 1-D array of length {n_fft}")
            self._window = w
            w_ptr = w.ctypes.data_as(_dbl_p)
        plan = _bfft_stft_plan_p()
        _check(_bfft_stft_plan_create(n, n_fft, hop_length, w_ptr, kind,
                                      ctypes.byref(plan)),
               "bfft_stft_plan_create")
        self._plan = plan
        self.n = n
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = int(_bfft_stft_plan_bins(plan))
        self.n_segs = int(_bfft_stft_plan_segments(plan))
        self.buffer_length = int(_bfft_stft_plan_buffer_length(plan))
        self.transform = t

    def __del__(self):
        plan = getattr(self, "_plan", None)
        if plan:
            _bfft_stft_plan_destroy(plan)
            self._plan = None

    def reset_buffer(self):
        """Zero the native persistent ISTFT overlap buffer."""
        _check(_bfft_stft_reset_buffer(self._plan), "bfft_stft_reset_buffer")
        return self

    def stft(self, x):
        """Return a ``(n_bins, n_segs)`` complex128 spectrogram."""
        a = np.ascontiguousarray(x, dtype=np.float64)
        if a.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},)")
        out = np.empty((self.n_bins, self.n_segs), dtype=np.complex128)
        _check(_bfft_stft_forward(self._plan, a.ctypes.data, out.ctypes.data),
               "bfft_stft_forward")
        return out

    def istft(self, Zx):
        """Invert a spectrogram block and return a float64 signal block.

        The plan's internal overlap buffer is updated as part of this call.
        """
        a = np.ascontiguousarray(Zx, dtype=np.complex128)
        if a.shape != (self.n_bins, self.n_segs):
            raise ValueError(f"Zx must have shape ({self.n_bins}, {self.n_segs})")
        out = np.empty(self.n_segs * self.hop_length, dtype=np.float64)
        _check(_bfft_stft_inverse(self._plan, a.ctypes.data, out.ctypes.data),
               "bfft_stft_inverse")
        return out


# Native plans are reusable per size; cache them so repeated calls avoid setup.
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


# Cache the planned objects (and thus their scratch buffers) per size for the
# stateless module-level functions. Each cached plan carries a non-blocking lock:
# a concurrent call on the same size that cannot grab the lock transparently
# falls back to a fresh private plan, so results stay correct under threads
# while the common single-threaded path reuses the cached buffers.
_rfft_cache: dict[int, "Plan"] = {}
_odft_cache: dict[int, "OdftPlan"] = {}
_rfft_locks: dict[int, threading.Lock] = {}
_odft_locks: dict[int, threading.Lock] = {}
_cache_guard = threading.Lock()


def _cached_rfft_plan(n):
    plan = _rfft_cache.get(n)
    if plan is None:
        with _cache_guard:
            plan = _rfft_cache.get(n)
            if plan is None:
                plan = Plan(n)
                _rfft_locks[n] = threading.Lock()
                _rfft_cache[n] = plan
    return plan, _rfft_locks[n]


def _cached_odft_plan(n):
    plan = _odft_cache.get(n)
    if plan is None:
        with _cache_guard:
            plan = _odft_cache.get(n)
            if plan is None:
                plan = OdftPlan(n)
                _odft_locks[n] = threading.Lock()
                _odft_cache[n] = plan
    return plan, _odft_locks[n]


def rfft(x):
    """Real-to-complex FFT. Drop-in for :func:`numpy.fft.rfft` (any N >= 2)."""
    a = _as_f64_1d(x)
    _check_rfft_n(a.shape[0])
    plan, lock = _cached_rfft_plan(a.shape[0])
    if lock.acquire(blocking=False):
        try:
            return plan.rfft(a)
        finally:
            lock.release()
    return Plan(a.shape[0]).rfft(a)  # concurrent use: private scratch


def irfft(x, n=None):
    """Inverse real FFT. Drop-in for :func:`numpy.fft.irfft` (any N >= 2).

    ``x`` holds ``N/2 + 1`` complex bins. ``n`` is the length of the real output;
    when omitted it defaults to ``2 * (len(x) - 1)``, matching numpy."""
    a = _as_c128_1d(x)
    if n is None:
        n = 2 * (a.shape[0] - 1)
    _check_rfft_n(n)
    plan, _ = _cached_rfft_plan(n)  # irfft needs no shared scratch
    return plan.irfft(a)


def odft(x):
    """Half-bin-shifted real transform: H[k] = sum_n x[n] exp(-2j*pi*(k+1/2)*n/N),
    k = 0 .. N/2-1. Equivalent to a half-bin phase shift followed by an rfft."""
    a = _as_f64_1d(x)
    _check_odft_n(a.shape[0])
    plan, _ = _cached_odft_plan(a.shape[0])  # odft needs no shared scratch
    return plan.odft(a)


def iodft(x, n=None):
    """Inverse of :func:`odft`. Maps ``N/2`` packed half-bin bins back to the
    ``N`` real samples. ``n`` defaults to ``2 * len(x)``."""
    a = _as_c128_1d(x)
    if n is None:
        n = 2 * a.shape[0]
    _check_odft_n(n)
    plan, _ = _cached_odft_plan(n)
    return plan.iodft(a)
