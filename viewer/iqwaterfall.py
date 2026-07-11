"""ctypes wrapper around the monolithic libiqwaterfall backend.

Exposes an IQ file source (WAV auto-detect or raw formats) and a BFFT-backed
waterfall engine that turns a file window into a rows x n_fft dB image.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

# Sample formats (must match iqw_fmt in iqwaterfall.cpp).
FMT_AUTO = 0
FMT_U8   = 1
FMT_S8   = 2
FMT_S16  = 3
FMT_S24  = 4
FMT_S32  = 5
FMT_F32  = 6

FORMAT_NAMES = {
    "Auto (WAV header)": FMT_AUTO,
    "uint8":  FMT_U8,
    "int8":   FMT_S8,
    "int16":  FMT_S16,
    "int24":  FMT_S24,
    "int32":  FMT_S32,
    "float32": FMT_F32,
}

# Window types (must match iqw_window in iqwaterfall.cpp).
WINDOWS = {
    "Hann": 0,
    "Hamming": 1,
    "Blackman": 2,
    "Blackman-Harris": 3,
    "Rectangular": 4,
}

_PKG = Path(__file__).resolve().parent


def _load():
    env = os.environ.get("IQW_LIBRARY")
    names = [env] if env else []
    names += ["libiqwaterfall.dylib", "libiqwaterfall.so"]
    last = None
    for nm in names:
        if not nm:
            continue
        p = nm if os.path.isabs(nm) else str(_PKG / nm)
        try:
            return ctypes.CDLL(p)
        except OSError as exc:
            last = exc
    raise OSError(f"could not load libiqwaterfall: {last}. Run viewer/build.sh")


_lib = _load()

_vp = ctypes.c_void_p
_ll = ctypes.c_longlong


def _decl(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_open = _decl("iqw_open", _vp, [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int])
_close = _decl("iqw_close", None, [_vp])
_num = _decl("iqw_num_samples", _ll, [_vp])
_rate = _decl("iqw_sample_rate", ctypes.c_double, [_vp])
_iscplx = _decl("iqw_is_complex", ctypes.c_int, [_vp])
_bits = _decl("iqw_bits", ctypes.c_int, [_vp])
_read = _decl("iqw_read", _ll, [_vp, _ll, _ll, _vp])

_wf_create = _decl("iqw_wf_create", _vp, [ctypes.c_int, ctypes.c_int])
_wf_destroy = _decl("iqw_wf_destroy", None, [_vp])
_wf_render = _decl("iqw_wf_render", None,
                   [_vp, _vp, _ll, _ll, ctypes.c_int, _vp, ctypes.c_int])
_wf_render_mem = _decl("iqw_wf_render_mem", None,
                       [_vp, _vp, _ll, _ll, _ll, ctypes.c_int, _vp, ctypes.c_int])

_fct_create = _decl("iqw_fct_create", _vp, [ctypes.c_int])
_fct_create_ex = _decl("iqw_fct_create_ex", _vp,
                       [ctypes.c_int, ctypes.c_int, ctypes.c_double])
_fct_destroy = _decl("iqw_fct_destroy", None, [_vp])
_fct_render = _decl("iqw_fct_render", None,
                    [_vp, _vp, _ll, _ll, ctypes.c_int, _vp, _vp, ctypes.c_int])
_fct_render_mem = _decl(
    "iqw_fct_render_mem", None,
    [_vp, _vp, _ll, _ll, _ll, ctypes.c_int, _vp, _vp, ctypes.c_int])

_ra_create = _decl("iqw_ra_create", _vp, [ctypes.c_int])
_ra_destroy = _decl("iqw_ra_destroy", None, [_vp])
_ra_set_bilinear = _decl("iqw_ra_set_bilinear", None, [_vp, ctypes.c_int])
_ra_render_mem = _decl("iqw_ra_render_mem", None,
                       [_vp, _vp, _ll, _ll, _ll, ctypes.c_int, _vp])

_fct_ra_create = _decl("iqw_fct_ra_create", _vp,
                       [ctypes.c_int, ctypes.c_int])
_fct_ra_destroy = _decl("iqw_fct_ra_destroy", None, [_vp])
_fct_ra_render_mem = _decl(
    "iqw_fct_ra_render_mem", None,
    [_vp, _vp, _ll, _ll, _ll, ctypes.c_int, _vp, ctypes.c_int])

_dp = ctypes.POINTER(ctypes.c_double)
_dip_run = _decl("iqw_dip_run", None,
                 [_dp, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                  ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                  _dp, _dp])
_ci = ctypes.c_int
_cdd = ctypes.c_double
_cip = ctypes.POINTER(ctypes.c_int)
_dip_run_complex = _decl("iqw_dip_run_complex", None,
                         [_dp, _ci, _cip, _ci, _ci, _cdd, _ci, _ci, _ci, _ci,
                          _dp, _dp])
_dip_unified = _decl(
    "iqw_dip_unified", None,
    [_dp, _ci, _cip, _ci, _ci, _cdd, _ci, _ci, _ci, _ci, _ci, _cdd, _cdd,
     _dp, _dp])
_dip_frames_long = _decl("iqw_dip_frames_long", _ci, [_ci, _ci])
_dip_run_complex_warm = _decl(
    "iqw_dip_run_complex_warm", None,
    [_dp, _ci, _cip, _ci, _ci, _cdd, _ci, _ci, _ci, _ci, _ci,
     _dp, _dp, _dp, _dp])

# Default (reference) long-aperture window. Apertures are runtime-selectable:
# NB = EB*32, NS = ES*32 (q* = 32 shared), HB = NB/8, HS = 32.
NB_LONG = 1024
HB_LONG = 128


_dip_waterfall_complex = _decl(
    "iqw_dip_waterfall_complex", None,
    [_dp, _ci, _cip, _ci, _ci, _cdd, _ci, _ci, _ci, _ci, _dp])
_dip_waterfall_proj = _decl(
    "iqw_dip_waterfall_proj", None,
    [_dp, _ci, _cip, _ci, _ci, _ci, _cdd, _ci, _ci, _ci, _ci, _dp])
_dip_fusion = _decl(
    "iqw_dip_fusion", None,
    [_dp, _ci, _cip, _ci, _ci, _ci, _cdd, _ci, _ci, _ci, _ci, _dp, _dp])


def ndelta_of(nb, ns):
    return nb // 32 - ns // 32 + 1


def frames_long(L, nb=NB_LONG):
    """Number of long-aperture frames for record length L and window nb."""
    return int(_dip_frames_long(int(L), int(nb)))


def dip_waterfall_complex(z, dsel=None, renorm=True, steepest_scale=2.5e-4,
                          n_steps=1, fuse_seed=False, nb=NB_LONG, ns=128):
    """Super-resolution waterfall readout: the long-aperture endpoint magnitude
    read directly off the refined active-delta state (no record synthesis).

    z is a complex record. nb/ns are the long/short aperture windows (multiples
    of 32, ns < nb, eb-es >= 4). Returns an (Ib, nb) float64 magnitude array,
    Ib = frames_long(L, nb); fftshift NOT applied. n_steps=0 = pure seed.
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    L = z.size
    Ib = frames_long(L, nb)
    out = np.empty((Ib, int(nb)), dtype=np.float64)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    _dip_waterfall_complex(z.ctypes.data_as(_dp), L, ds_ptr, nds,
                           1 if renorm else 0, float(steepest_scale),
                           int(n_steps), 1 if fuse_seed else 0, int(nb), int(ns),
                           out.ctypes.data_as(_dp))
    return out


def dip_waterfall_proj(z, mode=3, alpha=1.0, n_steps=1, direct_seed=True,
                       dsel=None, renorm=True, nb=NB_LONG, ns=128):
    """Projection-style active-delta waterfall / diagnostic readout.

    mode: 0 seed only, 1 short-only correction, 2 long-only, 3 long+short.
    nb/ns are the long/short aperture windows. Returns (Ib, nb) magnitudes.
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    L = z.size
    Ib = frames_long(L, nb)
    out = np.empty((Ib, int(nb)), dtype=np.float64)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    _dip_waterfall_proj(z.ctypes.data_as(_dp), L, ds_ptr, nds,
                        1 if renorm else 0, int(mode), float(alpha),
                        int(n_steps), 1 if direct_seed else 0, int(nb), int(ns),
                        out.ctypes.data_as(_dp))
    return out


class IQSource:
    """An open IQ file. WAV headers are auto-parsed when fmt is FMT_AUTO."""

    def __init__(self, path, fmt=FMT_AUTO, sample_rate=0.0, is_complex=True):
        h = _open(path.encode(), int(fmt), float(sample_rate), 1 if is_complex else 0)
        if not h:
            raise IOError(f"could not open IQ file: {path} (fmt={fmt})")
        self._h = h
        self.path = path
        self.num_samples = int(_num(h))
        self.sample_rate = float(_rate(h))
        self.is_complex = bool(_iscplx(h))
        self.bits = int(_bits(h))

    def read(self, start, count, out=None):
        """Read `count` complex samples from index `start` -> complex128 array."""
        if out is None or out.shape != (count * 2,):
            out = np.zeros(count * 2, dtype=np.float32)
        _read(self._h, int(start), int(count), out.ctypes.data)
        return out[0::2] + 1j * out[1::2]

    def close(self):
        if getattr(self, "_h", None):
            _close(self._h)
            self._h = None

    def __del__(self):
        self.close()


class Waterfall:
    """BFFT-backed waterfall engine for a fixed FFT size and window."""

    def __init__(self, n_fft=1024, window=0):
        h = _wf_create(int(n_fft), int(window))
        if not h:
            raise ValueError(f"invalid n_fft={n_fft} (must be power of two >= 4)")
        self._h = h
        self.n_fft = int(n_fft)
        self.window = int(window)

    def render(self, src: IQSource, start, hop, n_rows, remove_dc=False, out=None):
        """Return (n_rows, n_fft) float32 dB image, fftshifted (col0 = -Fs/2)."""
        if out is None or out.shape != (n_rows, self.n_fft):
            out = np.empty((n_rows, self.n_fft), dtype=np.float32)
        _wf_render(self._h, src._h, int(start), int(hop), int(n_rows),
                   out.ctypes.data, 1 if remove_dc else 0)
        return out

    def render_mem(self, iq_interleaved, start, hop, n_rows, remove_dc=False,
                   out=None):
        """Render a waterfall from an in-memory interleaved-IQ float32 buffer.

        iq_interleaved is a 1-D float32 array of I,Q pairs (length 2*nsamples).
        """
        iq = np.ascontiguousarray(iq_interleaved, dtype=np.float32)
        nsamples = iq.size // 2
        if out is None or out.shape != (n_rows, self.n_fft):
            out = np.empty((n_rows, self.n_fft), dtype=np.float32)
        _wf_render_mem(self._h, iq.ctypes.data, int(nsamples), int(start),
                       int(hop), int(n_rows), out.ctypes.data,
                       1 if remove_dc else 0)
        return out

    def close(self):
        if getattr(self, "_h", None):
            _wf_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


class FctWaterfall:
    """Complex-IQ adaptive leading-edge waterfall engine.

    ``render`` returns ``(db, support)`` in fftshifted full-spectrum order.
    ``support`` is tau/N; callers should place a cell at frame_origin + tau,
    rather than pretending every adaptive aperture has STFT-center timing.
    """

    def __init__(self, n_fft=1024, min_support=1, activity=0.0):
        h = _fct_create_ex(int(n_fft), int(min_support), float(activity))
        if not h:
            raise ValueError(
                f"invalid FCT n_fft={n_fft}, min_support={min_support}, "
                f"activity={activity}")
        self._h = h
        self.n_fft = int(n_fft)
        self.min_support = int(min_support)
        self.activity = float(activity)

    def render(self, src: IQSource, start, hop, n_rows, remove_dc=False,
               db_out=None, support_out=None):
        shape = (int(n_rows), self.n_fft)
        if db_out is None or db_out.shape != shape:
            db_out = np.empty(shape, dtype=np.float32)
        if support_out is None or support_out.shape != shape:
            support_out = np.empty(shape, dtype=np.float32)
        _fct_render(self._h, src._h, int(start), int(hop), int(n_rows),
                    db_out.ctypes.data, support_out.ctypes.data,
                    1 if remove_dc else 0)
        return db_out, support_out

    def render_mem(self, iq_interleaved, start, hop, n_rows, remove_dc=False,
                   db_out=None, support_out=None):
        iq = np.ascontiguousarray(iq_interleaved, dtype=np.float32)
        shape = (int(n_rows), self.n_fft)
        if db_out is None or db_out.shape != shape:
            db_out = np.empty(shape, dtype=np.float32)
        if support_out is None or support_out.shape != shape:
            support_out = np.empty(shape, dtype=np.float32)
        _fct_render_mem(self._h, iq.ctypes.data, iq.size // 2, int(start),
                        int(hop), int(n_rows), db_out.ctypes.data,
                        support_out.ctypes.data, 1 if remove_dc else 0)
        return db_out, support_out

    def close(self):
        if getattr(self, "_h", None):
            _fct_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


def _dsel_ptr(dsel):
    if dsel is None:
        return None, 0, None
    ds = np.ascontiguousarray(dsel, dtype=np.int32)
    return ds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ds.size, ds


class Reassign:
    """BFFT-backed complex reassigned-spectrogram (super-resolution) readout."""

    def __init__(self, n_fft=1024):
        h = _ra_create(int(n_fft))
        if not h:
            raise ValueError(f"invalid n_fft={n_fft} (power of two >= 4)")
        self._h = h
        self.n_fft = int(n_fft)

    def render_mem(self, iq_interleaved, start, hop, n_rows, out=None):
        """Return (n_rows, n_fft) float32 dB reassigned image, fftshifted."""
        iq = np.ascontiguousarray(iq_interleaved, dtype=np.float32)
        nsamples = iq.size // 2
        if out is None or out.shape != (n_rows, self.n_fft):
            out = np.empty((n_rows, self.n_fft), dtype=np.float32)
        _ra_render_mem(self._h, iq.ctypes.data, int(nsamples), int(start),
                       int(hop), int(n_rows), out.ctypes.data)
        return out

    def set_bilinear(self, enabled=True):
        _ra_set_bilinear(self._h, 1 if enabled else 0)

    def close(self):
        if getattr(self, "_h", None):
            _ra_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


class FctReassign:
    """Hann-STFT energy relocated by intrinsic-FCT phase derivatives.

    FCT selects a leading-edge support per bin. Its correlation moment gives
    the phase group delay, while an exact fixed-support endpoint recurrence
    gives instantaneous frequency. This is deliberately a hybrid observable:
    STFT supplies energy; FCT supplies the reassignment coordinates. Output
    has ``2*N`` columns on a half-bin frequency raster.
    """

    def __init__(self, n_fft=1024, min_support=None):
        n_fft = int(n_fft)
        if min_support is None:
            min_support = max(4, n_fft // 32)
        h = _fct_ra_create(n_fft, int(min_support))
        if not h:
            raise ValueError(
                f"invalid FCT reassignment n_fft={n_fft}, "
                f"min_support={min_support}")
        self._h = h
        self.n_fft = n_fft
        self.bins = 2 * n_fft
        self.min_support = int(min_support)

    def render_mem(self, iq_interleaved, start, hop, n_rows,
                   remove_dc=False, out=None):
        iq = np.ascontiguousarray(iq_interleaved, dtype=np.float32)
        nsamples = iq.size // 2
        if out is None or out.shape != (n_rows, self.bins):
            out = np.empty((n_rows, self.bins), dtype=np.float32)
        _fct_ra_render_mem(self._h, iq.ctypes.data, int(nsamples), int(start),
                           int(hop), int(n_rows), out.ctypes.data,
                           1 if remove_dc else 0)
        return out

    def close(self):
        if getattr(self, "_h", None):
            _fct_ra_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


def dip_run(x, dsel=None, renorm=True, steepest_scale=2.5e-4, n_steps=1,
            fuse_seed=False):
    """Run active_delta_center5 (DIP walk + PGHI seed) on a real record.

    x is a real float array (length L, typically 8192). Returns (u, loss0).
    n_steps in [1,3] iterates the L5 grad/step/consensus; fuse_seed blends the
    short- and long-aperture PGHI seeds. n_steps=1, fuse_seed=False reproduces
    the reference to ~1e-13.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    L = x.size
    u = np.empty(L, dtype=np.float64)
    loss0 = ctypes.c_double(0.0)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    _dip_run(x.ctypes.data_as(_dp), L, ds_ptr, nds, 1 if renorm else 0,
             float(steepest_scale), int(n_steps), 1 if fuse_seed else 0,
             u.ctypes.data_as(_dp), ctypes.byref(loss0))
    return u, loss0.value


def dip_fusion(z, dsel, mode=3, alpha=1.0, n_steps=1, direct_seed=True,
               renorm=True, nb=NB_LONG, ns=128):
    """F1xT2 fusion readout off the refined active-delta state.

    dsel is the active delta set (also what the readout emits). Returns
    (long_mag [Ib, nb], short_mag [Ib, Dsel, ns]): long endpoint = fine
    frequency amplitude, short endpoints at the CONSTRAINED deltas = fine-time
    gates (measured, not model-implied). Both are views of one refined Zb.
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    L = z.size
    Ib = frames_long(L, nb)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    long_mag = np.empty((Ib, int(nb)), dtype=np.float64)
    short_mag = np.empty((Ib, int(nds), int(ns)), dtype=np.float64)
    _dip_fusion(z.ctypes.data_as(_dp), L, ds_ptr, nds, 1 if renorm else 0,
                int(mode), float(alpha), int(n_steps), 1 if direct_seed else 0,
                int(nb), int(ns), long_mag.ctypes.data_as(_dp),
                short_mag.ctypes.data_as(_dp))
    return long_mag, short_mag


def dip_run_complex(z, dsel=None, renorm=True, steepest_scale=2.5e-4,
                    n_steps=1, fuse_seed=False, nb=NB_LONG, ns=128):
    """Complex-generalized DIP/PGHI reconstruction of a complex baseband record.

    Returns (u, loss0). nb/ns select the long/short aperture windows.
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    L = z.size
    u = np.empty(L, dtype=np.complex128)
    loss0 = ctypes.c_double(0.0)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    _dip_run_complex(z.ctypes.data_as(_dp), L, ds_ptr, nds, 1 if renorm else 0,
                     float(steepest_scale), int(n_steps), 1 if fuse_seed else 0,
                     int(nb), int(ns), u.ctypes.data_as(_dp),
                     ctypes.byref(loss0))
    return u, loss0.value


def dip_unified(z, dsel=None, renorm=True, steepest_scale=2.5e-4,
                shared_steps=1, nb=NB_LONG, ns=None, h_short=None,
                unified_steps=1, beta=0.88, final_long_relax=0.75):
    """Shared fast1 seed plus independent two-lattice waveform projections."""
    z = np.ascontiguousarray(z, dtype=np.complex128)
    if ns is None:
        ns = int(nb) // 4
    if h_short is None:
        h_short = int(ns) // 2
    u = np.empty(z.size, dtype=np.complex128)
    loss0 = ctypes.c_double(0.0)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    _dip_unified(
        z.ctypes.data_as(_dp), z.size, ds_ptr, nds, 1 if renorm else 0,
        float(steepest_scale), int(shared_steps), int(nb), int(ns),
        int(h_short), int(unified_steps), float(beta), float(final_long_relax),
        u.ctypes.data_as(_dp), ctypes.byref(loss0))
    return u, loss0.value


def dip_run_complex_warm(z, warm_internal=None, dsel=None, renorm=True,
                         steepest_scale=2.5e-4, n_steps=1, fuse_seed=False,
                         nb=NB_LONG, ns=128):
    """Warm-started complex reconstruction with cross-tile phase continuity.

    warm_internal: previous tile's long-aperture internal phase (nb, n_warm) or
    None. Returns (u, internal, loss0); internal is (nb, frames_long(L, nb)).
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    L = z.size
    Ib = frames_long(L, nb)
    u = np.empty(L, dtype=np.complex128)
    internal = np.empty((int(nb), Ib), dtype=np.float64)
    loss0 = ctypes.c_double(0.0)
    ds_ptr, nds, _keep = _dsel_ptr(dsel)
    if warm_internal is None:
        warm_ptr, n_warm = None, 0
    else:
        w = np.ascontiguousarray(warm_internal, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != nb:
            raise ValueError(f"warm_internal must be ({nb}, n_warm)")
        warm_ptr, n_warm = w.ctypes.data_as(_dp), w.shape[1]
    _dip_run_complex_warm(
        z.ctypes.data_as(_dp), L, ds_ptr, nds, 1 if renorm else 0,
        float(steepest_scale), int(n_steps), 1 if fuse_seed else 0,
        int(nb), int(ns), int(n_warm), warm_ptr, internal.ctypes.data_as(_dp),
        u.ctypes.data_as(_dp), ctypes.byref(loss0))
    return u, internal, loss0.value
