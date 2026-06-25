"""A small, fully-nopython STFT/ISTFT built on BFFT -- a librosa-style analysis
that runs inside Numba ``@njit`` code.

Unlike ``numpy.fft`` (object-mode only), BFFT is a C-ABI real FFT that Numba can
call in nopython mode, so the entire short-time transform compiles -- no
``objmode`` round-trips. The transform follows the ssqueezepy convention: each
frame is fftshifted before the FFT (a zero-phase / "modulated" STFT). See
https://github.com/OverLordGoldDragon/ssqueezepy for the background.

Usage::

    import numpy as np
    from stft import STFT

    tf = STFT(n=24576, n_fft=512, hop_length=128)   # default Hann window
    x  = np.random.randn(tf.n)
    Zx = tf.stft(x)                  # (n_bins, n_segs) complex128
    xr, buf = tf.istft(Zx)          # xr length n_segs*hop; xr ~= x after opening

Pass your own analysis window (length ``n_fft``) to the constructor; the
matching MSE-optimal synthesis window is derived and stored automatically.

Requires ``bfft`` (which provides the njit FFT) and ``numba``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# BFFT's nopython-callable FFT entry points and the cffi handle.
from bfft.numba_support import (
    bfft_forward,
    bfft_inverse,
    bodft_forward,
    bodft_inverse,
    ffi,
    make_odft_plan,
    make_plan,
)


# --------------------------------------------------------------------------
# fftshift helpers (numba-compatible). For even length these equal a roll by
# n // 2 and fftshift == ifftshift; the general forms are provided for reuse.
# --------------------------------------------------------------------------
@njit(cache=True)
def fftshift(a):
    n = a.shape[0]
    s = n // 2
    out = np.empty_like(a)
    for i in range(n):
        out[i] = a[(i + (n - s)) % n]
    return out


@njit(cache=True)
def ifftshift(a):
    n = a.shape[0]
    s = n // 2
    out = np.empty_like(a)
    for i in range(n):
        out[i] = a[(i + s) % n]
    return out


# --------------------------------------------------------------------------
# nopython kernels. These take only arrays / scalars / the plan address, so they
# are callable from other @njit code as well as from the STFT class.
# --------------------------------------------------------------------------
# cache=False: these call BFFT through cffi function pointers, which Numba
# treats as dynamic globals and refuses to cache.
@njit(cache=False)
def stft_kernel(
    x, ana, plan, n_fft, hop, n_bins, n_segs, work, scratch, transform_kind,
):
    """Reflect-pad, fftshift-frame, window, and transform each segment via BFFT.
    Returns a (n_bins, n_segs) complex128 spectrogram."""
    n = x.shape[0]
    half = n_fft // 2
    pl = half
    pr = half - 1

    # Centered reflect padding (no edge-sample duplication), matching np.pad
    # mode='reflect' with pad widths (n_fft//2, n_fft//2 - 1).
    xp = np.empty(n + n_fft - 1, np.float64)
    for i in range(n):
        xp[pl + i] = x[i]
    for i in range(pl):
        xp[i] = x[pl - i]
    for i in range(pr):
        xp[pl + n + i] = x[n - 2 - i]

    seg = np.empty(n_fft, np.float64)
    out_f = np.empty(2 * n_bins, np.float64)   # complex spectrum as re/im pairs
    Zx = np.empty((n_bins, n_segs), np.complex128)

    for jx in range(n_segs):
        base = jx * hop
        for a in range(n_fft):
            idx = a + half          # fftshift the frame as we gather it
            if idx >= n_fft:
                idx -= n_fft
            seg[a] = xp[base + idx] * ana[a]
        if transform_kind == 0:
            bfft_forward(plan,
                         ffi.from_buffer(seg), ffi.from_buffer(out_f),
                         ffi.from_buffer(work), ffi.from_buffer(scratch))
        else:
            bodft_forward(plan,
                          ffi.from_buffer(seg), ffi.from_buffer(out_f),
                          ffi.from_buffer(work), ffi.from_buffer(scratch))
        for b in range(n_bins):
            Zx[b, jx] = out_f[2 * b] + 1j * out_f[2 * b + 1]
    return Zx


@njit(cache=False)
def istft_kernel(
    Zx, syn, plan, n_fft, hop, n_bins, n_segs, buffer, transform_kind,
):
    """Inverse: invert each column via BFFT, undo the frame fftshift, apply the
    synthesis window, then stream out ``hop`` samples per frame using a
    persistent overlap buffer (length ``n_fft - hop``). Returns the
    ``(n_segs * hop,)`` signal and the updated buffer, so successive blocks can
    be inverted continuously. Only the opening of a fresh stream carries a
    transient; steady-state reconstruction is exact."""
    half = n_fft // 2
    blen = n_fft - hop          # persistent buffer length

    inb_f = np.empty(2 * n_bins, np.float64)
    frame = np.empty(n_fft, np.float64)
    proc = np.empty(n_fft, np.float64)
    x = np.empty(n_segs * hop, np.float64)

    pos = 0
    for jx in range(n_segs):
        for b in range(n_bins):
            z = Zx[b, jx]
            inb_f[2 * b] = z.real
            inb_f[2 * b + 1] = z.imag
        if transform_kind == 0:
            bfft_inverse(plan, ffi.from_buffer(inb_f), ffi.from_buffer(frame))
        else:
            bodft_inverse(plan, ffi.from_buffer(inb_f), ffi.from_buffer(frame))
        for a in range(n_fft):
            idx = a + half          # undo the frame fftshift, then window
            if idx >= n_fft:
                idx -= n_fft
            proc[a] = frame[idx] * syn[a]

        # Emit the first hop samples (overlapped with the carried buffer),
        # shift the buffer down by hop, and fold in this frame's tail.
        for a in range(hop):
            x[pos + a] = proc[a] + buffer[a]
        for a in range(blen - hop):
            buffer[a] = buffer[a + hop]
        for a in range(blen - hop, blen):
            buffer[a] = 0.0
        for a in range(blen):
            buffer[a] += proc[hop + a]
        pos += hop

    return x, buffer


# --------------------------------------------------------------------------
# Configurable class
# --------------------------------------------------------------------------
class STFT:
    """Short-time Fourier transform with a fixed configuration.

    Parameters
    ----------
    n : int
        Length of the real input signal.
    n_fft : int
        Segment / FFT length. Must be an even power of two >= 4.
    hop_length : int
        Hop between segments. Must divide ``n_fft`` (constant-overlap-add).
    window : array_like or None
        Analysis window of length ``n_fft``. Defaults to a Hann window. The
        MSE-optimal synthesis window is derived from it.
    transform : {"rfft", "odft"}
        Transform used for each frame. ``"odft"`` uses the half-bin ODFT path
        with ``n_fft // 2`` bins; ``"rfft"`` uses the standard real FFT path
        with ``n_fft // 2 + 1`` bins.

    Notes
    -----
    Reconstruction is exact (~1e-15) in steady state. The streaming inverse has
    an inherent latency of ``n_fft // 2`` samples (the half-frame fftshift
    centering): ``istft(stft(x))[n_fft // 2:]`` matches ``x``. The only other
    deviation is an opening transient over the first frames, where fewer frames
    overlap than the constant-overlap-add denominator assumes -- expected when a
    stream starts, not an error. The intended overlap is 3/4
    (``hop_length == n_fft // 4``); other overlaps redistribute the transient.
    """

    def __init__(
        self, n=24576, n_fft=512, hop_length=128, window=None, transform="rfft",
    ):
        n = int(n)
        n_fft = int(n_fft)
        hop_length = int(hop_length)

        assert n_fft >= 4 and (n_fft & (n_fft - 1)) == 0, \
            "n_fft must be a power of two >= 4"
        assert n_fft % 2 == 0, "n_fft must be even"
        assert 1 <= hop_length <= n_fft, "hop_length must be in [1, n_fft]"
        assert n_fft % hop_length == 0, \
            "n_fft must be an integer multiple of hop_length (COLA)"
        assert n > 0, "n must be positive"

        if transform == "rfft":
            transform_kind = 0
            plan_factory = make_plan
        else:
            if transform == "odft":
                transform_kind = 1
                plan_factory = make_odft_plan
            else:
                raise ValueError("transform must be either 'rfft' or 'odft'")

        if window is None:
            window = np.hanning(n_fft)
        window = np.ascontiguousarray(window, dtype=np.float64)
        assert window.ndim == 1 and window.shape[0] == n_fft, \
            f"window must be a 1-D array of length n_fft={n_fft}"

        self.n = n
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = 0
        self.n_segs = (n - 1) // hop_length + 1
        self.transform = transform
        self._transform_kind = transform_kind

        # COLA denominator: sum of squared analysis window over all hop shifts
        # that cover each sample, then the MSE-optimal synthesis window.
        den = np.zeros(n_fft, dtype=np.float64)
        idx = np.arange(n_fft)
        kmax = n_fft // hop_length
        for k in range(-kmax, kmax + 1):
            j = idx - k * hop_length
            m = (j >= 0) & (j < n_fft)
            den[m] += window[j[m]] ** 2
        assert np.all(den > 0), \
            "window/hop_length violate COLA (zero in the synthesis denominator)"
        synthesis = window / den

        self.window = window
        self.synthesis_window = synthesis
        # Analysis window is pre-shifted (ifftshift) and applied as the frame is
        # gathered with its fftshift; the synthesis window is applied after the
        # inverse frame is un-fftshifted, so it is stored un-shifted.
        self._ana = np.ascontiguousarray(np.fft.ifftshift(window))
        self._syn = np.ascontiguousarray(synthesis)
        # Persistent overlap buffer length for streaming inversion.
        self.buffer_length = n_fft - hop_length

        # One reusable BFFT plan and scratch (float64-staged, complex as pairs).
        plan, bins, work_n, scratch_n = plan_factory(n_fft)
        self.n_bins = bins
        self._plan = plan
        self._work = np.empty(work_n, dtype=np.float64)
        self._scratch = np.empty(2 * scratch_n, dtype=np.float64)

    def stft(self, x):
        """Forward transform. ``x`` has length ``n``; returns
        ``(n_bins, n_segs)`` complex128."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        assert x.shape == (self.n,), f"x must have shape ({self.n},)"
        return stft_kernel(x, self._ana, self._plan, self.n_fft,
                           self.hop_length, self.n_bins, self.n_segs,
                           self._work, self._scratch, self._transform_kind)

    def new_buffer(self):
        """Allocate a zeroed persistent overlap buffer for streaming inversion."""
        return np.zeros(self.buffer_length, dtype=np.float64)

    def istft(self, Zx, buffer=None):
        """Inverse transform. ``Zx`` has shape ``(n_bins, n_segs)``. Returns
        ``(x, buffer)`` where ``x`` has length ``n_segs * hop_length`` and
        ``buffer`` is the updated overlap state. Pass the returned ``buffer``
        back in to invert the next block continuously; omit it (or call
        :meth:`new_buffer`) to start a fresh stream."""
        Zx = np.ascontiguousarray(Zx, dtype=np.complex128)
        assert Zx.shape == (self.n_bins, self.n_segs), \
            f"Zx must have shape ({self.n_bins}, {self.n_segs})"
        if buffer is None:
            buffer = self.new_buffer()
        else:
            buffer = np.ascontiguousarray(buffer, dtype=np.float64)
            assert buffer.shape == (self.buffer_length,), \
                f"buffer must have shape ({self.buffer_length},)"
        return istft_kernel(Zx, self._syn, self._plan, self.n_fft,
                            self.hop_length, self.n_bins, self.n_segs, buffer,
                            self._transform_kind)
