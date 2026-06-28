"""BFFT: real Fourier transforms with a numpy-friendly API.

Power-of-two lengths N >= 4 use the native Bruun kernel. Other real FFT
lengths are intentionally rejected.

Public functions (stateless drop-ins, with cached plans/buffers under the hood):
    bfft.rfft(x)      -- drop-in equivalent of numpy.fft.rfft for power-of-two N >= 4.
    bfft.irfft(x, n)  -- drop-in equivalent of numpy.fft.irfft.
    bfft.odft(x)      -- half-bin-shifted real transform (phase shift + rfft).
    bfft.iodft(x, n)  -- inverse of odft.

Planned objects (lowest per-call overhead for hot loops; one per thread):
    bfft.Plan(N)      -- .rfft(x) / .irfft(X) at a fixed power-of-two size N.
    bfft.OdftPlan(N)  -- .odft(x) / .iodft(H) at a fixed power-of-two size N.
"""

from ._core import OdftPlan, Plan, STFTPlan, hann_window, iodft, irfft, odft, rfft

__all__ = ["rfft", "irfft", "odft", "iodft", "Plan", "OdftPlan", "STFTPlan", "hann_window"]
__version__ = "1.0"
