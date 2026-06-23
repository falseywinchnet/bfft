"""BFFT: power-of-two real Fourier transforms with a numpy-friendly API.

Public functions:
    bfft.rfft(x)      -- drop-in equivalent of numpy.fft.rfft for power-of-two N.
    bfft.irfft(x, n)  -- drop-in equivalent of numpy.fft.irfft.
    bfft.odft(x)      -- half-bin-shifted real transform (phase shift + rfft).
    bfft.iodft(x, n)  -- inverse of odft.
"""

from ._core import iodft, irfft, odft, rfft

__all__ = ["rfft", "irfft", "odft", "iodft"]
__version__ = "0.1.0"
