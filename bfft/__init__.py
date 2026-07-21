"""BFFT: real Fourier transforms with a numpy-friendly API.

Power-of-two lengths N >= 4 use the native Bruun kernel. Other real FFT
lengths are intentionally rejected.

Public functions (stateless drop-ins, with cached plans/buffers under the hood):
    bfft.rfft(x)      -- drop-in equivalent of numpy.fft.rfft for power-of-two N >= 4.
    bfft.irfft(x, n)  -- drop-in equivalent of numpy.fft.irfft.
    bfft.odft(x)      -- half-bin-shifted real transform (phase shift + rfft).
    bfft.iodft(x, n)  -- inverse of odft.
    bfft.fct(x)       -- Fast Correlated Transform (forward-only): (C, tau)
                         with each standard bin at its maximally correlated
                         leading-edge slice. No inverse exists.
    bfft.meyer_split(img) -- Meyer G-norm cartoon + texture decomposition
                         (TGFD) of an arbitrary-size grayscale image;
                         returns (cartoon, texture).  The fast path.
    bfft.meyer(img)   -- the same decomposition plus the 3-rung scale
                         ladder; returns (cartoon, texture, band_coarse,
                         band_mid, band_fine).  The ladder costs an order
                         of magnitude more than the decomposition.

Planned objects (lowest per-call overhead for hot loops; one per thread):
    bfft.Plan(N)      -- .rfft(x) / .irfft(X) at a fixed power-of-two size N.
    bfft.OdftPlan(N)  -- .odft(x) / .iodft(H) at a fixed power-of-two size N.
    bfft.FctPlan(N)   -- .fct(x) at a fixed power-of-two size N >= 16.
    bfft.MeyerPlan((H, W)) -- .decompose(img) at fixed power-of-two dims.
"""

from ._core import (FctPlan, MeyerPlan, OdftPlan, Plan, STFTPlan, fct,
                    hann_window, iodft, irfft, meyer, meyer_split, odft,
                    rfft)

__all__ = ["rfft", "irfft", "odft", "iodft", "fct", "meyer",
           "meyer_split", "Plan", "OdftPlan", "FctPlan", "MeyerPlan",
           "STFTPlan", "hann_window"]
__version__ = "1.0"
