"""Analytic signal families for the one-dimensional resize laboratory."""

from __future__ import annotations

import numpy as np


SIGNALS = (
    "Transported multiband chirp",
    "Subpixel edge and carrier",
    "Crossing phase packets",
    "Near-Nyquist sweep",
)


def evaluate(name: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a named deterministic signal exactly at normalized coordinates."""

    t = np.asarray(x, dtype=np.float64)
    if name == SIGNALS[0]:
        phase = 2.0*np.pi*(7.0*t + 43.0*t*t + 81.0*t*t*t/3.0)
        return (
            0.43*np.sin(phase)
            + 0.23*np.cos(2.0*np.pi*(31.0*t + 0.37))
            + 0.17*np.sin(2.0*np.pi*(113.0*t - 0.21))
            + 0.12*np.exp(-((t-0.68)/0.11)**2)*np.cos(2.0*np.pi*241.0*t)
        )
    if name == SIGNALS[1]:
        edge = 0.66*np.tanh((t-0.4173)/0.0027)
        packet = 0.22*np.exp(-((t-0.59)/0.085)**2)*np.cos(
            2.0*np.pi*(79.0*t+0.31)
        )
        return edge + packet + 0.08*np.sin(2.0*np.pi*13.0*t)
    if name == SIGNALS[2]:
        left = np.exp(-((t-0.36)/0.12)**2)*np.cos(
            2.0*np.pi*(48.0*t+27.0*t*t)
        )
        right = np.exp(-((t-0.64)/0.12)**2)*np.cos(
            2.0*np.pi*(71.0*t-23.0*t*t+0.19)
        )
        return 0.58*left + 0.48*right + 0.09*np.sin(2.0*np.pi*9.0*t)
    if name == SIGNALS[3]:
        # Instantaneous frequency rises from 0.08 to 0.47 cycles per source
        # interval when the default 4097-anchor lattice is used.
        phase = 2.0*np.pi*(330.0*t + 790.0*t*t)
        return 0.72*np.sin(phase) + 0.13*np.cos(2.0*np.pi*71.0*t)
    raise ValueError(f"unknown signal {name!r}")


def samples(name: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    if length < 5:
        raise ValueError("signal length must be at least five")
    coordinate = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
    return coordinate, evaluate(name, coordinate)
