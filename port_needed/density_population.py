"""Emit the complete site population directly from frozen BFFT support.

The local density ``sqrt(det(Q)) / pi`` is already measured by the one-stage
geometry.  A fixed low-discrepancy phase converts that continuous density to
germs in parallel.  There is no candidate list, ranking, top-k selection,
offspring, deletion, or requested population target.
"""

from __future__ import annotations

import numpy as np


def _hash01(x: np.ndarray, y: np.ndarray, salt: np.ndarray) -> np.ndarray:
    value = np.sin(
        (x + 1.0) * 12.9898
        + (y + 1.0) * 78.233
        + 17.123
        + salt * 31.7
    ) * 43758.5453123
    return value - np.floor(value)


def emit_density_population(
    geometry: dict,
    *,
    safety_cells: int,
    phase_shift: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Locally quantize the tensor-implied population into normalized germs."""
    measure = np.maximum(
        np.asarray(geometry["measure"], dtype=np.float64), 0.0)
    height, width = measure.shape
    implied = max(float(geometry["implied_cells"]), 1.0)
    ceiling = max(int(safety_cells), 1)
    commanded = min(implied, float(ceiling))
    density = measure * commanded / max(float(np.sum(measure)), 1e-30)

    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    phase = np.mod(
        52.9829189 * np.mod(
            0.06711056 * xx + 0.00583715 * yy + phase_shift, 1.0),
        1.0,
    )
    whole = np.floor(density).astype(np.int32)
    fractional = density - whole
    count_at = whole + (phase < fractional)
    flat_count = count_at.ravel()
    realized = int(np.sum(flat_count, dtype=np.int64))
    pixel = np.repeat(np.arange(height * width), flat_count)
    if pixel.size == 0:
        pixel = np.array([int(np.argmax(density))], dtype=np.int64)
        occurrence = np.zeros(1, dtype=np.float64)
    else:
        start = np.cumsum(flat_count, dtype=np.int64) - flat_count
        occurrence = (
            np.arange(pixel.size, dtype=np.float64)
            - np.repeat(start, flat_count)
        )
    py = pixel // width
    px = pixel - py * width
    jitter_x = _hash01(
        px.astype(np.float64),
        py.astype(np.float64),
        occurrence + float(phase_shift),
    ) - 0.5
    jitter_y = _hash01(
        px.astype(np.float64) + 19.0,
        py.astype(np.float64) - 7.0,
        occurrence + 2.0 * float(phase_shift),
    ) - 0.5
    center_x = np.clip(
        px.astype(np.float64) + 0.5 + 0.8 * jitter_x,
        0.5,
        width - 0.5,
    )
    center_y = np.clip(
        py.astype(np.float64) + 0.5 + 0.8 * jitter_y,
        0.5,
        height - 0.5,
    )
    centers = np.column_stack((center_x / width, center_y / height))
    return centers, {
        "implied_cells": implied,
        "commanded_cells": commanded,
        "realized_cells": int(len(centers)),
        "safety_limit_hit": implied > ceiling or realized > ceiling,
        "quantization_error": float(len(centers) - commanded),
        "maximum_pixel_density": float(np.max(density)),
        "density": density,
        "phase": phase,
    }
