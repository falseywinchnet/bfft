"""ctypes binding to the BFFT shared library.

Loads the native library (bundled in this package, or a system install from
`make install`) and exposes typed access to the forward transforms used by the
public shim in :mod:`bfft`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import concurrent.futures
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


def _has_required_symbols(lib: ctypes.CDLL) -> bool:
    for name in ("bfft_plan_create", "bodft_plan_create", "bfft_stft_plan_create"):
        try:
            getattr(lib, name)
        except AttributeError:
            return False
    return True


def _load_library() -> ctypes.CDLL:
    last_err = None
    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:  # pragma: no cover - depends on platform
            last_err = exc
            continue
        if _has_required_symbols(lib):
            return lib
        last_err = OSError(
            f"{path} is an older BFFT library without the STFT symbols; "
            "rebuild/install the native library or set BFFT_LIBRARY to the new one"
        )
    raise OSError(
        "Could not locate a compatible BFFT native library. Build it with "
        "`pip install .` or `make && make install`, or set BFFT_LIBRARY to the "
        "shared object. "
        f"Last error: {last_err}"
    )


_lib = _load_library()


def _decl(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _decl_optional(name, restype, argtypes):
    """Declare a newer optional kernel while accepting an older shared lib."""
    try:
        return _decl(name, restype, argtypes)
    except AttributeError:
        return None


_dbl_p = ctypes.POINTER(ctypes.c_double)
_flt_p = ctypes.POINTER(ctypes.c_float)
_cplx_p = ctypes.POINTER(_Complex)
_void_p = ctypes.c_void_p
_plan_p = ctypes.c_void_p
_i32_p = ctypes.POINTER(ctypes.c_int32)
_i64_p = ctypes.POINTER(ctypes.c_int64)
_u8_p = ctypes.POINTER(ctypes.c_uint8)
_size_p = ctypes.POINTER(ctypes.c_size_t)

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

# --- fast correlated transform (fct.h) ---
_fct_plan_create = _decl("fct_plan_create", ctypes.c_int,
                         [ctypes.c_size_t, ctypes.POINTER(_plan_p)])
_fct_plan_create_ex = _decl("fct_plan_create_ex", ctypes.c_int,
                            [ctypes.c_size_t, ctypes.c_int, ctypes.c_double,
                             ctypes.c_double, ctypes.POINTER(_plan_p)])
_fct_plan_destroy = _decl("fct_plan_destroy", None, [_plan_p])
_fct_plan_bins = _decl("fct_plan_bins", ctypes.c_size_t, [_plan_p])
_fct_forward = _decl("fct_forward", ctypes.c_int,
                     [_plan_p, _void_p, _void_p, _void_p])
_fct_forward_complex = _decl("fct_forward_complex", ctypes.c_int,
                             [_plan_p, _void_p, _void_p, _void_p])
_fct_forward_complex_moment = _decl(
    "fct_forward_complex_moment", ctypes.c_int,
    [_plan_p, _void_p, _void_p, _void_p, _void_p])

# --- Meyer G-norm decomposer (meyer.h) ---
_meyer_plan_create = _decl("bfft_meyer_plan_create", ctypes.c_int,
                           [ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_double, ctypes.c_double, ctypes.c_int,
                            ctypes.c_int, ctypes.c_double, ctypes.c_int,
                            ctypes.POINTER(_plan_p)])
_meyer_plan_destroy = _decl("bfft_meyer_plan_destroy", None, [_plan_p])
_meyer_plan_set_solver = _decl_optional(
    "bfft_meyer_plan_set_solver", ctypes.c_int, [_plan_p, ctypes.c_int])
_meyer_plan_solver = _decl_optional(
    "bfft_meyer_plan_solver", ctypes.c_int, [_plan_p])
_meyer_split = _decl("bfft_meyer_split", ctypes.c_int,
                     [_plan_p, _void_p, _void_p, _void_p])
_meyer_split_trace = _decl_optional(
    "bfft_meyer_split_trace", ctypes.c_int,
    [_plan_p, _void_p, _void_p, _void_p])
_meyer_trace_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, _dbl_p, _dbl_p, ctypes.c_size_t, _void_p)
_meyer_split_visit = _decl_optional(
    "bfft_meyer_split_visit", ctypes.c_int,
    [_plan_p, _void_p, _meyer_trace_callback, _void_p])
_meyer_decompose = _decl("bfft_meyer_decompose", ctypes.c_int,
                         [_plan_p, _void_p, _void_p, _void_p, _void_p,
                          _void_p, _void_p])
_meyer_rof = _decl("bfft_meyer_rof", ctypes.c_int,
                   [_plan_p, _void_p, _void_p, ctypes.c_double,
                    ctypes.c_double, ctypes.c_int, ctypes.c_double])
_meyer_rof_accelerated = _decl_optional(
    "bfft_meyer_rof_accelerated", ctypes.c_int,
    [_plan_p, _void_p, _void_p, ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_double, ctypes.c_int])
_meyer_last_rof_sweeps = _decl_optional(
    "bfft_meyer_plan_last_rof_sweeps", ctypes.c_int, [_plan_p])
_meyer_last_rof_hodge_applied = _decl_optional(
    "bfft_meyer_plan_last_rof_hodge_applied", ctypes.c_int, [_plan_p])

# --- measured-graph vision kernels (vision.h; optional for old installs) ---
_vision_assemble_normal = _decl_optional(
    "bfft_vision_assemble_normal", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _i32_p, _u8_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p,
     _i64_p, _i64_p, _i64_p, _dbl_p, _dbl_p])
_vision_render_affine = _decl_optional(
    "bfft_vision_render_affine", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p,
     _dbl_p, _dbl_p, _dbl_p])
_vision_scan_residual_ridges = _decl_optional(
    "bfft_vision_scan_residual_ridges", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     ctypes.c_double, ctypes.c_double, _i32_p, _dbl_p, _dbl_p, _dbl_p,
     _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _i32_p, _i32_p])
_vision_scan_paired_offsets = _decl_optional(
    "bfft_vision_scan_paired_offsets", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     ctypes.c_double, _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p,
     _dbl_p, _i32_p])
_vision_support_forward = _decl_optional(
    "bfft_vision_support_forward", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p])
_vision_support_transpose = _decl_optional(
    "bfft_vision_support_transpose", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p])
_vision_support_normal_apply = _decl_optional(
    "bfft_vision_support_normal_apply", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p])
_vision_curvature_population_f32 = _decl_optional(
    "bfft_vision_curvature_population_f32", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t,
     _flt_p, _flt_p, _flt_p, _flt_p, ctypes.c_double,
     _flt_p, _flt_p, _flt_p, _flt_p, _dbl_p])
_vision_soft_support_diffuse = _decl_optional(
    "bfft_vision_soft_support_diffuse", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     ctypes.c_size_t,
     ctypes.c_double, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p,
     _dbl_p, _dbl_p])
_vision_fast_march_first_label = _decl_optional(
    "bfft_vision_fast_march_first_label", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _dbl_p, _i32_p, _dbl_p, _dbl_p,
     _i32_p, _dbl_p, _u8_p, _dbl_p,
     _i64_p, ctypes.c_size_t, _i32_p,
     _dbl_p, _dbl_p, _dbl_p,
     _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p,
     _i32_p, _i32_p, _dbl_p, _i32_p,
     _size_p, _size_p, _size_p])
_vision_fast_march_labels = _decl_optional(
    "bfft_vision_fast_march_labels", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _dbl_p, _i32_p, _dbl_p, _dbl_p,
     _i32_p, _dbl_p, _u8_p, _dbl_p,
     _i64_p, ctypes.c_size_t, _i32_p,
     _dbl_p, _dbl_p, _dbl_p,
     _i32_p, _dbl_p, _size_p, _size_p])
_vision_metric_edge_costs_f32 = _decl_optional(
    "bfft_vision_metric_edge_costs_f32", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t,
     _flt_p, _flt_p, _flt_p, _flt_p, _flt_p, _flt_p,
     ctypes.c_double, ctypes.c_double, _flt_p])
_vision_bucket_first_label = _decl_optional(
    "bfft_vision_bucket_first_label", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i64_p, _dbl_p, _flt_p,
     ctypes.c_double, ctypes.c_size_t, ctypes.c_double,
     _i32_p, _dbl_p, _i32_p, _size_p])
_vision_hard_affine_fit = _decl_optional(
    "bfft_vision_hard_affine_fit", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p])
_vision_hard_basis_refit = _decl_optional(
    "bfft_vision_hard_basis_refit", ctypes.c_int,
    [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
     _i32_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p, _dbl_p])

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
    if not _is_pow2(n) or n < 4:
        raise ValueError("bfft real FFT requires a power-of-two length N >= 4")


def _check_odft_n(n):
    if not _is_pow2(n) or n < 2:
        raise ValueError("bfft ODFT requires a power-of-two length N >= 2")


def _check_fct_n(n):
    if not _is_pow2(n) or n < 16:
        raise ValueError("bfft FCT requires a power-of-two length N >= 16")


# --------------------------------------------------------------------------
# Planned objects: cache the plan, sizes, and reusable scratch buffers (with
# their raw pointers) so the only per-call costs are one output allocation and
# one input-pointer fetch -- matching numpy's own minimum and chasing pyfftw.
#
# A Plan owns shared scratch and is therefore NOT thread-safe: use one plan per
# thread (or use the module-level rfft/odft, which guard against concurrent use).
# --------------------------------------------------------------------------

class Plan:
    """Reusable plan for the standard real FFT at a fixed power-of-two size N >= 4.

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


class FctPlan:
    """Reusable plan for the Fast Correlated Transform at a fixed
    power-of-two size N >= 16.

    FORWARD-ONLY.  ``FctPlan(N).fct(x)`` returns ``(C, tau)``: for each of
    the ``N/2 + 1`` standard bins, ``C[k]`` is the leading-edge correlation
    ``sum_{t < tau[k]} x[t] * exp(-2j*pi*k*t/N)`` at the slice
    ``tau[k] in [1, N]`` selected for high correlation under the score
    ``|C|^2 / tau``.  Bins with no coherent leading edge default to
    ``tau = N`` (the plain FFT bin). The intrinsic phase-disk selector certifies
    the global ``|C|^2/tau`` argmax for every bin by default. With an explicit
    positive ``act`` floor, bins proven below that floor emit full Fourier
    support. An explicit ``t_min`` restricts the feasible domain to
    ``tau in [t_min, N]`` without changing the exact objective; ``rel`` is kept
    only for compatibility with the former selector. The selection is
    nonlinear and no inverse exists.

    A plan owns native scratch and is not thread-safe; create one plan per
    thread."""

    __slots__ = ("n", "bins", "_plan")

    def __init__(self, n, t_min=None, rel=None, act=None):
        n = int(n)
        _check_fct_n(n)
        self.n = n
        if t_min is None and rel is None and act is None:
            self._plan = _fct_plan(n)
        else:
            p = _plan_p()
            _check(_fct_plan_create_ex(
                n,
                1 if t_min is None else int(t_min),
                0.5 if rel is None else float(rel),
                0.0 if act is None else float(act),
                ctypes.byref(p)), "fct_plan_create_ex")
            self._plan = p
        self.bins = int(_fct_plan_bins(self._plan))

    def fct(self, x, out=None, tau_out=None):
        a = _as_f64_1d(x)
        if a.shape[0] != self.n:
            raise ValueError(f"FctPlan(N={self.n}).fct expects length {self.n}")
        out = _out_buffer(out, self.bins, np.complex128, "FctPlan.fct")
        tau_out = _out_buffer(tau_out, self.bins, np.int64, "FctPlan.fct tau")
        _check(_fct_forward(self._plan, a.ctypes.data, out.ctypes.data,
                            tau_out.ctypes.data),
               "fct_forward")
        return out, tau_out

    def fct_complex(self, x, out=None, tau_out=None):
        """Adaptive leading-edge analysis of a length-N complex-IQ frame.

        Returns ``(C, tau)`` with N full-spectrum bins.  Tau is selected once
        from the complex correlation; analyzing I and Q independently would be
        incorrect because FCT selection is nonlinear.
        """
        a = _as_c128_1d(x)
        if a.shape[0] != self.n:
            raise ValueError(
                f"FctPlan(N={self.n}).fct_complex expects length {self.n}")
        out = _out_buffer(out, self.n, np.complex128,
                          "FctPlan.fct_complex")
        tau_out = _out_buffer(tau_out, self.n, np.int64,
                              "FctPlan.fct_complex tau")
        _check(_fct_forward_complex(self._plan, a.ctypes.data, out.ctypes.data,
                                    tau_out.ctypes.data),
               "fct_forward_complex")
        return out, tau_out

    def fct_complex_moment(self, x, out=None, tau_out=None, moment_out=None):
        """Return ``(C, tau, M)`` with the exact phase moment at selected tau.

        ``M[k] = sum(t*x[t]*exp(-2j*pi*k*t/N), t < tau[k])`` and therefore
        ``Re(M/C)`` is the FCT phase group delay. Support selection is performed
        once on ``x``; ``M`` is evaluated afterward at that same support.
        """
        a = _as_c128_1d(x)
        if a.shape[0] != self.n:
            raise ValueError(
                f"FctPlan(N={self.n}).fct_complex_moment expects length "
                f"{self.n}")
        out = _out_buffer(out, self.n, np.complex128,
                          "FctPlan.fct_complex_moment")
        tau_out = _out_buffer(tau_out, self.n, np.int64,
                              "FctPlan.fct_complex_moment tau")
        moment_out = _out_buffer(moment_out, self.n, np.complex128,
                                 "FctPlan.fct_complex_moment moment")
        _check(_fct_forward_complex_moment(
            self._plan, a.ctypes.data, out.ctypes.data,
            tau_out.ctypes.data, moment_out.ctypes.data),
            "fct_forward_complex_moment")
        return out, tau_out, moment_out



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
    transform : {"rfft", "odft", "fct"}
        Per-frame transform. ``"rfft"`` returns ``n_fft // 2 + 1`` bins;
        ``"odft"`` returns ``n_fft // 2`` half-bin shifted bins; ``"fct"``
        returns ``n_fft // 2 + 1`` bins where each bin is emitted at its
        maximally correlated leading-edge slice (forward-only: ``istft``
        raises; frames are gathered in natural time order, no fftshift).

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
        elif t == "fct":
            kind = 2
        else:
            raise ValueError("transform must be 'rfft', 'odft', or 'fct'")
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
        Plans created with ``transform="fct"`` are forward-only and reject
        this call.
        """
        if self.transform == "fct":
            raise ValueError(
                "STFTPlan(transform='fct') is forward-only: the FCT's "
                "per-bin slice selection is nonlinear and has no inverse")
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
_fct_plans: dict[int, _plan_p] = {}


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


def _fct_plan(n):
    p = _fct_plans.get(n)
    if p is None:
        p = _plan_p()
        _check(_fct_plan_create(n, ctypes.byref(p)), "fct_plan_create")
        _fct_plans[n] = p
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
    """Real-to-complex FFT for power-of-two lengths N >= 4."""
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
    """Inverse real FFT for power-of-two lengths N >= 4.

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


def fct(x):
    """Fast Correlated Transform (forward-only) for power-of-two lengths
    N >= 16.  Returns ``(C, tau)`` where ``C[k]`` is the leading-edge
    correlation of standard bin ``k`` at its maximally correlated slice
    ``tau[k]`` (see :class:`FctPlan`).  There is no inverse."""
    a = _as_f64_1d(x)
    _check_fct_n(a.shape[0])
    return FctPlan(a.shape[0]).fct(a)


class MeyerPlan:
    """Planned Meyer G-norm cartoon + texture decomposer (TGFD).

    ``MeyerPlan((H, W)).decompose(image)`` runs the Gilles-Osher two-projector
    alternation as warm interleaved Split Bregman sweeps (one per subproblem
    per pass, persistent Bregman states, spectra of f/u/w maintained so each
    sweep costs one forward + one inverse 2-D transform), then splits the
    texture layer along the ratio-4 rung ladder {mu, mu/4, mu/16} with
    independent ROF solves.  Solver 0 requires power-of-two dimensions;
    solvers 1 and 2 transform one power-of-two axis and sweep the other.
    Use :func:`meyer` for automatic arbitrary-size padding.  Images are
    [0, 255]-scaled floats.

    Returns ``(cartoon, texture, band_coarse, band_mid, band_fine)`` with
    ``cartoon + band_coarse + band_mid + band_fine == cartoon-layer u +
    texture-layer v`` exactly (cartoon includes the coarsest rung survivor;
    ``image - cartoon - bands`` is the model residual).  Not thread-safe;
    create one plan per thread."""

    __slots__ = ("shape", "lam", "mu", "passes", "rung_sweeps", "rung_tol",
                 "threads", "solver", "_plan")

    def __init__(self, shape, lam=0.05, mu=40.0, passes=64, rung_sweeps=600,
                 rung_tol=1e-5, threads=0, solver=0):
        h, w = int(shape[0]), int(shape[1])
        solver = int(solver)
        if solver not in (0, 1, 2):
            raise ValueError("solver must be 0 (spectral), 1 (periodic FACR), "
                             "or 2 (Neumann FACR)")
        transformed = [n >= 8 and _is_pow2(n) for n in (h, w)]
        if h < 2 or w < 2 or (
                solver == 0 and not all(transformed)) or (
                solver != 0 and not any(transformed)):
            raise ValueError(
                f"MeyerPlan shape {shape} is incompatible with solver "
                f"{solver}; solver 0 needs two power-of-two axes >= 8, "
                "FACR needs at least one")
        self.shape = (h, w)
        self.lam = float(lam)
        self.mu = float(mu)
        self.passes = int(passes)
        self.rung_sweeps = int(rung_sweeps)
        self.rung_tol = float(rung_tol)
        self.threads = int(threads)
        self.solver = solver
        plan = _plan_p()
        _check(_meyer_plan_create(h, w, self.lam, self.mu, self.passes,
                                  self.rung_sweeps, self.rung_tol,
                                  self.threads, ctypes.byref(plan)),
               "bfft_meyer_plan_create")
        self._plan = plan
        if solver:
            if _meyer_plan_set_solver is None:
                _meyer_plan_destroy(self._plan)
                self._plan = None
                raise RuntimeError(
                    "installed BFFT native library predates FACR solvers; "
                    "rebuild/reinstall the package")
            _check(_meyer_plan_set_solver(self._plan, solver),
                   "bfft_meyer_plan_set_solver")

    def __del__(self):
        plan = getattr(self, "_plan", None)
        if plan:
            _meyer_plan_destroy(plan)

    def split(self, image):
        """The model decomposition alone: ``(cartoon, texture) = (u, v)``,
        exactly the pair the Gilles-Osher alternation produces, with no
        ladder.  Note :meth:`decompose` reports a different cartoon --
        ``u`` plus the ladder's coarsest rung survivor -- so that its
        cartoon and bands sum to ``u + v``."""
        a = np.ascontiguousarray(image, dtype=np.float64)
        if a.shape != self.shape:
            raise ValueError(
                f"MeyerPlan{self.shape}.split expects shape {self.shape}")
        outs = (np.empty(self.shape, dtype=np.float64),
                np.empty(self.shape, dtype=np.float64))
        _check(_meyer_split(self._plan, a.ctypes.data,
                            *(o.ctypes.data for o in outs)),
               "bfft_meyer_split")
        return outs

    def trace(self, image):
        """Return every intermediate ``(cartoon, texture)`` state.

        Both arrays have shape ``(passes, height, width)``.  The native
        engine executes the pass sequence once; this is exactly equivalent
        to separate 1..passes splits without their quadratic repeated work.
        """
        if _meyer_split_trace is None:
            raise RuntimeError(
                "installed BFFT native library predates split tracing; "
                "rebuild/reinstall the package")
        a = np.ascontiguousarray(image, dtype=np.float64)
        if a.shape != self.shape:
            raise ValueError(
                f"MeyerPlan{self.shape}.trace expects shape {self.shape}")
        shape = (self.passes, *self.shape)
        cartoon = np.empty(shape, dtype=np.float64)
        texture = np.empty(shape, dtype=np.float64)
        _check(_meyer_split_trace(
            self._plan,
            a.ctypes.data,
            cartoon.ctypes.data,
            texture.ctypes.data,
        ), "bfft_meyer_split_trace")
        return cartoon, texture

    def visit(self, image, callback):
        """Visit every intermediate split state without a pass-deep array.

        ``callback(pass_number, cartoon, texture)`` receives read-only NumPy
        views that are valid only until the callback returns.
        """
        if _meyer_split_visit is None:
            raise RuntimeError(
                "installed BFFT native library predates split visitors; "
                "rebuild/reinstall the package")
        a = np.ascontiguousarray(image, dtype=np.float64)
        if a.shape != self.shape:
            raise ValueError(
                f"MeyerPlan{self.shape}.visit expects shape {self.shape}")
        failure = []

        @_meyer_trace_callback
        def visitor(pass_number, cartoon_ptr, texture_ptr, count, user):
            try:
                cartoon = np.ctypeslib.as_array(
                    cartoon_ptr, shape=(count,)).reshape(self.shape)
                texture = np.ctypeslib.as_array(
                    texture_ptr, shape=(count,)).reshape(self.shape)
                cartoon.flags.writeable = False
                texture.flags.writeable = False
                callback(int(pass_number), cartoon, texture)
            except BaseException as exc:  # ctypes cannot propagate callbacks
                failure.append(exc)

        _check(_meyer_split_visit(
            self._plan, a.ctypes.data, visitor, None),
            "bfft_meyer_split_visit")
        if failure:
            raise failure[0]

    def rof(self, image, c, eta=0.0, sweeps=200, tol=1e-5,
            hodge_after=0):
        """A plain ROF solve, ``argmin_x TV(x) + (c/2)|x - image|^2``, run
        by the same Split Bregman solver the ladder rungs use.  ``eta`` is
        the Bregman penalty (0 selects the ladder convention, ``10 * c``);
        sweeps stop early once the relative iterate change falls below
        ``tol``.  ``hodge_after > 0`` enables one objective-checked
        Fourier/Hodge closure after that many sweeps, then continues the same
        ROF problem; it currently requires solver 0.  ``image - rof(image,
        c)`` is the ROF residual."""
        a = np.ascontiguousarray(image, dtype=np.float64)
        if a.shape != self.shape:
            raise ValueError(
                f"MeyerPlan{self.shape}.rof expects shape {self.shape}")
        out = np.empty(self.shape, dtype=np.float64)
        hodge_after = int(hodge_after)
        if hodge_after:
            if _meyer_rof_accelerated is None:
                raise RuntimeError(
                    "this bfft build does not provide ROF Hodge acceleration")
            _check(_meyer_rof_accelerated(
                self._plan, a.ctypes.data, out.ctypes.data, float(c),
                float(eta), int(sweeps), float(tol), hodge_after),
                "bfft_meyer_rof_accelerated")
        else:
            _check(_meyer_rof(self._plan, a.ctypes.data, out.ctypes.data,
                              float(c), float(eta), int(sweeps), float(tol)),
                   "bfft_meyer_rof")
        return out

    @property
    def last_rof_sweeps(self):
        """Ordinary Split-Bregman sweeps used by the most recent ROF call."""
        return (None if _meyer_last_rof_sweeps is None
                else int(_meyer_last_rof_sweeps(self._plan)))

    @property
    def last_rof_hodge_applied(self):
        """Whether the most recent ROF call accepted its Hodge proposal."""
        return (None if _meyer_last_rof_hodge_applied is None
                else bool(_meyer_last_rof_hodge_applied(self._plan)))

    def decompose(self, image):
        a = np.ascontiguousarray(image, dtype=np.float64)
        if a.shape != self.shape:
            raise ValueError(
                f"MeyerPlan{self.shape}.decompose expects shape {self.shape}")
        outs = tuple(np.empty(self.shape, dtype=np.float64)
                     for _ in range(5))
        _check(_meyer_decompose(self._plan, a.ctypes.data,
                                *(o.ctypes.data for o in outs)),
               "bfft_meyer_decompose")
        return outs


_MEYER_PLANS = {}
_MEYER_BATCH_PLANS = {}
_MEYER_LOCK = threading.Lock()
_MEYER_BATCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _meyer_padded(image, lam, mu, passes, rung_sweeps, rung_tol, threads,
                  solver=0):
    """Shared arbitrary-size front end: returns (plan, padded, top, left,
    h, w).  The spectral solver pads both dimensions; FACR pads only its
    transformed dimension.  Padding is symmetric with the image centered."""
    a = np.ascontiguousarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("meyer expects a 2-D grayscale image")
    h, w = a.shape
    if h < 2 or w < 2:
        raise ValueError("meyer expects an image at least 2x2")

    def _next_pow2(n):
        p = 8
        while p < n:
            p *= 2
        return p

    solver = int(solver)
    if solver not in (0, 1, 2):
        raise ValueError("solver must be 0, 1, or 2")
    nh, nw = _next_pow2(h), _next_pow2(w)
    if solver == 0:
        ph, pw = nh, nw
    elif h * nw <= nh * w:
        ph, pw = h, nw       # sweep height, transform width
    else:
        ph, pw = nh, w       # sweep width, transform height
    top, left = (ph - h) // 2, (pw - w) // 2
    if (ph, pw) != (h, w):
        padded = np.pad(a, ((top, ph - h - top), (left, pw - w - left)),
                        mode="symmetric")
    else:
        padded = a
    key = (ph, pw, float(lam), float(mu), int(passes), int(rung_sweeps),
           float(rung_tol), int(threads), solver)
    with _MEYER_LOCK:
        plan = _MEYER_PLANS.get(key)
        if plan is None:
            plan = MeyerPlan((ph, pw), lam=lam, mu=mu, passes=passes,
                             rung_sweeps=rung_sweeps, rung_tol=rung_tol,
                             threads=threads, solver=solver)
            _MEYER_PLANS[key] = plan
    return plan, padded, top, left, h, w


def meyer_split(image, lam=0.05, mu=40.0, passes=64, threads=0, solver=0):
    """Meyer cartoon + texture decomposition of an arbitrary-size grayscale
    image, without the scale ladder: returns ``(cartoon, texture)``, the
    pair the Gilles-Osher alternation produces.  This is the fast path --
    the ladder in :func:`meyer` costs an order of magnitude more than the
    decomposition itself. ``solver=1`` enables periodic FACR and avoids
    padding the worse-padded axis (falling back to spectral when neither
    axis needs padding); ``solver=2`` additionally uses Neumann boundaries
    on that swept axis."""
    plan, padded, top, left, h, w = _meyer_padded(
        image, lam, mu, passes, 1, 0.0, threads, solver)
    outs = plan.split(padded)
    return tuple(o[top:top + h, left:left + w].copy() for o in outs)


def meyer_split_batch(images, lam=0.05, mu=40.0, passes=64, threads=0,
                      solver=0):
    """Split equal-shaped planes concurrently with independent native plans.

    A Meyer plan owns mutable scratch, so ordinary cached scalar plans cannot
    safely be called from two Python workers. This routine caches one plan per
    plane and divides the requested native worker lanes among them. The native
    engine is bit-identical across thread counts.
    """
    planes = np.ascontiguousarray(images, dtype=np.float64)
    if planes.ndim != 3 or planes.shape[0] < 1:
        raise ValueError("meyer_split_batch expects KxHxW planes")
    channels, h, w = planes.shape
    lanes = int(threads)
    if channels == 1 or lanes < channels:
        split = [
            meyer_split(
                planes[channel], lam=lam, mu=mu, passes=passes,
                threads=threads, solver=solver)
            for channel in range(channels)
        ]
        return (
            np.stack([item[0] for item in split]),
            np.stack([item[1] for item in split]),
        )
    if h < 2 or w < 2:
        raise ValueError("meyer expects images at least 2x2")

    def next_pow2(value):
        result = 8
        while result < value:
            result *= 2
        return result

    solver = int(solver)
    if solver not in (0, 1, 2):
        raise ValueError("solver must be 0, 1, or 2")
    nh, nw = next_pow2(h), next_pow2(w)
    if solver == 0:
        ph, pw = nh, nw
    elif h * nw <= nh * w:
        ph, pw = h, nw
    else:
        ph, pw = nh, w
    top, left = (ph - h) // 2, (pw - w) // 2
    if (ph, pw) == (h, w):
        padded = planes
    else:
        padded = np.stack([
            np.pad(
                planes[channel],
                ((top, ph - h - top), (left, pw - w - left)),
                mode="symmetric",
            )
            for channel in range(channels)
        ])
    plan_threads = max(lanes // channels, 1)
    key = (
        channels, ph, pw, float(lam), float(mu), int(passes),
        plan_threads, solver,
    )
    with _MEYER_LOCK:
        entry = _MEYER_BATCH_PLANS.get(key)
        if entry is None:
            plans = [
                MeyerPlan(
                    (ph, pw), lam=lam, mu=mu, passes=passes,
                    rung_sweeps=1, rung_tol=0.0,
                    threads=plan_threads, solver=solver,
                )
                for _ in range(channels)
            ]
            entry = (plans, threading.Lock())
            _MEYER_BATCH_PLANS[key] = entry
    plans, use_lock = entry
    with use_lock:
        futures = [
            _MEYER_BATCH_EXECUTOR.submit(plans[channel].split, padded[channel])
            for channel in range(channels)
        ]
        split = [future.result() for future in futures]
    return (
        np.stack([
            item[0][top:top + h, left:left + w]
            for item in split
        ]),
        np.stack([
            item[1][top:top + h, left:left + w]
            for item in split
        ]),
    )


def meyer_trace(image, lam=0.05, mu=40.0, passes=64, threads=0, solver=0):
    """All intermediate Meyer split states from one native pass sequence.

    Returns two arrays shaped ``(passes, height, width)``.  Arbitrary input
    sizes use the padding policy selected by ``solver``.
    """
    plan, padded, top, left, h, w = _meyer_padded(
        image, lam, mu, passes, 1, 0.0, threads, solver)
    cartoon, texture = plan.trace(padded)
    return (
        cartoon[:, top:top + h, left:left + w].copy(),
        texture[:, top:top + h, left:left + w].copy(),
    )


def rof(image, c=0.025, eta=0.0, sweeps=200, tol=1e-5, threads=0,
        solver=0, hodge_after=0):
    """Total-variation (Rudin-Osher-Fatemi) solve on an arbitrary-size
    grayscale image: ``argmin_x TV(x) + (c/2)|x - image|^2``.

    Smaller ``c`` smooths harder.  ``image - rof(image, c)`` is the ROF
    residual; the Meyer decomposition is built from exactly this solver, and
    subtracting a ROF solve of a cartoon layer isolates the smooth
    illumination a flat cartoon discards.  ``hodge_after > 0`` applies the
    optional one-shot static-ROF accelerator after that many ordinary
    sweeps; it requires ``solver=0``.  Arbitrary sizes are handled by the
    padding policy selected by ``solver``."""
    plan, padded, top, left, h, w = _meyer_padded(
        image, 0.05, 40.0, 1, 1, 0.0, threads, solver)
    out = plan.rof(
        padded, c, eta=eta, sweeps=sweeps, tol=tol,
        hodge_after=hodge_after)
    return out[top:top + h, left:left + w].copy()


def meyer(image, lam=0.05, mu=40.0, passes=64, rung_sweeps=600,
          rung_tol=1e-5, threads=0, solver=0):
    """Meyer G-norm cartoon + texture decomposition of an arbitrary-size
    grayscale image (see :class:`MeyerPlan` for the algorithm and outputs).

    Solver 0 symmetrically pads each dimension to a power of two.  Solver 1
    is periodic FACR and pads only the transformed axis; solver 2 uses the
    same one-axis padding with Neumann boundaries on the swept axis.  The
    five outputs are cropped back to the input size.  Plans are cached per
    (padded shape, parameters, solver)."""
    plan, padded, top, left, h, w = _meyer_padded(
        image, lam, mu, passes, rung_sweeps, rung_tol, threads, solver)
    outs = plan.decompose(padded)
    return tuple(o[top:top + h, left:left + w].copy() for o in outs)
