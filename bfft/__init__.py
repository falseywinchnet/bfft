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
    bfft.meyer_split(img) -- fixed-cost jump-measure Meyer cartoon + texture
                         split of an arbitrary-size grayscale image; returns
                         (cartoon, texture).  This is the default fast path.
    bfft.meyer(img)   -- legacy Gilles-Osher decomposition plus the 3-rung
                         scale ladder; returns (cartoon, texture,
                         band_coarse, band_mid, band_fine).
    bfft.rof(img, c)  -- the plain total-variation solve the decomposition
                         is built from, on its own.

Recomposition effects (bfft.effects, pure numpy over the above):
    bfft.meyer_channels(img)      -- the split applied per channel of a
                         colour image in RGB, OKLab, or OKLab luma+chroma.
    bfft.recompose(...) / bfft.recompose_channels(...)
                      -- reassemble a split with independent gains on the
                         cartoon, texture, and shading layers.
    bfft.shade(cartoon, c)        -- the smooth illumination a flat cartoon
                         discards: cartoon - ROF(cartoon, c).

Planned objects (lowest per-call overhead for hot loops; one per thread):
    bfft.Plan(N)      -- .rfft(x) / .irfft(X) at a fixed power-of-two size N.
    bfft.OdftPlan(N)  -- .odft(x) / .iodft(H) at a fixed power-of-two size N.
    bfft.FctPlan(N)   -- .fct(x) at a fixed power-of-two size N >= 16.
    bfft.MeyerPlan((H, W)) -- .decompose(img) with spectral or one-axis
                         FACR screened-Poisson solves.
"""

from ._core import (FctPlan, MeyerPlan, OdftPlan, Plan, STFTPlan, fct,
                    hann_window, iodft, irfft, meyer, meyer_split,
                    meyer_split_conditioned_first,
                    meyer_split_jump_measure,
                    meyer_split_legacy,
                    meyer_split_preconditioned,
                    meyer_trace, odft,
                    rfft, rof)
from .effects import (lab_to_srgb, meyer_channels, recompose,
                      recompose_channels, shade, srgb_to_lab)
from .vision import (CoownershipGraph, SingleStageDecompositionObjective,
                     assemble_normal, compact_support_operators,
                     coownership_graph, deletion_prices,
                     measure_residual_ridges, render_partition,
                     selected_inverse_blocks, vision_backend)

__all__ = ["rfft", "irfft", "odft", "iodft", "fct", "meyer",
           "meyer_split", "meyer_split_conditioned_first", "meyer_trace",
           "meyer_split_jump_measure",
           "meyer_split_legacy",
           "meyer_split_preconditioned",
           "rof", "Plan", "OdftPlan",
           "FctPlan", "MeyerPlan",
           "STFTPlan", "hann_window", "meyer_channels", "recompose",
           "recompose_channels", "shade", "srgb_to_lab", "lab_to_srgb",
           "CoownershipGraph", "coownership_graph", "assemble_normal",
           "compact_support_operators",
           "render_partition", "measure_residual_ridges",
           "selected_inverse_blocks", "deletion_prices",
           "SingleStageDecompositionObjective", "vision_backend"]
__version__ = "1.0"
