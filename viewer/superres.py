"""Super-resolution readout: the complex reassigned spectrogram.

Per the aperture-ladder method (notes/aperture_ladder_design.md S6), the
super-resolution product is a *reassigned* spectrogram of the (phase-consistent)
reconstructed record: a long analysis window supplies fine frequency, and
reassignment relocates each energy point to its instantaneous frequency and
group-delay-corrected time -- sharpening both axes at once. Blind geometric-mean
fusion of raw window sizes DEGRADES (measured); this does not, because
reassignment uses the local phase structure.

This is the complex (IQ) variant: full n-bin FFT, no Hermitian fold.
"""
from __future__ import annotations

import numpy as np


def reassigned_db(z, n, hop, n_rows, eps=1e-12):
    """Complex reassigned spectrogram as a (n_rows, n) fftshifted dB image.

    z is a complex baseband buffer whose sample 0 aligns with display row 0.
    Column c maps to bin (c - n/2) after fftshift (col 0 = -Fs/2), matching the
    plain waterfall convention.
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    g = np.hanning(n).astype(np.float64)
    dg = np.gradient(g)
    tg = (np.arange(n) - (n - 1) / 2.0) * g

    pad = n
    zp = np.concatenate([np.zeros(pad, np.complex128), z, np.zeros(pad, np.complex128)])
    centers = np.arange(n_rows) * hop
    base = centers[:, None] + pad - n // 2 + np.arange(n)[None, :]   # [n_rows, n]
    segs = zp[base]

    Y = np.fft.fft(g * segs, axis=1)
    Yt = np.fft.fft(tg * segs, axis=1)
    Yd = np.fft.fft(dg * segs, axis=1)
    E = Y.real ** 2 + Y.imag ** 2
    inv = 1.0 / (E + 1e-30)

    # Reassignment: local group delay (time) and instantaneous frequency (freq).
    that = centers[:, None] + np.real(Yt * np.conj(Y)) * inv
    khat = np.arange(n)[None, :] - np.imag(Yd * np.conj(Y)) * inv * n / (2 * np.pi)

    good = E > 1e-9 * (E.max() + 1e-30)
    cols = np.clip(np.round(that / hop), 0, n_rows - 1).astype(np.int64)
    rows = np.mod(np.round(khat).astype(np.int64), n)
    flat = (cols * n + rows)[good]                # scatter into [n_rows, n]
    P = np.bincount(flat, weights=E[good], minlength=n_rows * n).reshape(n_rows, n)

    P = np.fft.fftshift(P, axes=1)                # col 0 -> -Fs/2
    return (10.0 * np.log10(P + eps)).astype(np.float32)
