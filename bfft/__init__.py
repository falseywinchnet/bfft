"""BFFT: power-of-two real Fourier transforms with a numpy-friendly API.

Public functions (stateless drop-ins, with cached plans/buffers under the hood):
    bfft.rfft(x)      -- drop-in equivalent of numpy.fft.rfft for power-of-two N.
    bfft.irfft(x, n)  -- drop-in equivalent of numpy.fft.irfft.
    bfft.odft(x)      -- half-bin-shifted real transform (phase shift + rfft).
    bfft.iodft(x, n)  -- inverse of odft.

Planned objects (lowest per-call overhead for hot loops; one per thread):
    bfft.Plan(N)      -- .rfft(x) / .irfft(X) at a fixed size N.
    bfft.OdftPlan(N)  -- .odft(x) / .iodft(H) at a fixed size N.
"""

from ._core import OdftPlan, Plan, iodft, irfft, odft, rfft

__all__ = ["rfft", "irfft", "odft", "iodft", "Plan", "OdftPlan"]
__version__ = "0.1.0"
