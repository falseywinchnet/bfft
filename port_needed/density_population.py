"""Emit the complete site population directly from frozen BFFT support.

The local density ``sqrt(det(Q)) / pi`` is already measured by the one-stage
geometry.  A fixed low-discrepancy phase converts that continuous density to
germs in parallel.  There is no candidate list, ranking, top-k selection,
offspring, deletion, or requested population target.
"""

from __future__ import annotations

import numpy as np

from bfft.vision import curvature_population_native


def _centered_difference(field: np.ndarray, axis: int) -> np.ndarray:
    """Unit-grid derivative with a one-sided boundary limit."""
    derivative = np.empty_like(field, dtype=np.float64)
    if axis == 0:
        derivative[1:-1] = 0.5 * (field[2:] - field[:-2])
        derivative[0] = field[1] - field[0]
        derivative[-1] = field[-1] - field[-2]
    else:
        derivative[:, 1:-1] = 0.5 * (
            field[:, 2:] - field[:, :-2])
        derivative[:, 0] = field[:, 1] - field[:, 0]
        derivative[:, -1] = field[:, -1] - field[:, -2]
    return derivative


def curvature_limited_geometry(geometry: dict) -> dict:
    """Limit a straight anisotropic support by measured director curvature.

    If the tensor predicts tangent and normal semi-spans ``a`` and ``b``,
    respectively, a contour with curvature ``kappa`` departs from its tangent
    by approximately ``kappa * a**2 / 2``.  A single straight support is only
    valid while that sagitta is no larger than ``b``.  Shortening the tangent
    span to satisfy the bound raises the required population by

        sqrt(max(1, kappa * a**2 / (2*b))).

    The director derivative is evaluated through its doubled-angle line field,
    so the result is invariant to the arbitrary sign of an eigenvector.
    """
    base_implied = max(float(geometry["implied_cells"]), 1e-30)
    native = curvature_population_native(
        geometry["precision_xx"],
        geometry["precision_xy"],
        geometry["precision_yy"],
        geometry["measure"],
        base_implied,
    )
    if native is not None:
        result = dict(geometry)
        result.update(native)
        result["straight_implied_cells"] = base_implied
        result["curvature_backend"] = "native C++"
        return result

    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    trace = qxx + qyy
    discriminant = np.hypot(qxx - qyy, 2.0 * qxy)
    safe_discriminant = np.maximum(discriminant, 1e-30)
    coherence = discriminant / np.maximum(trace, 1e-30)

    # (u, v) = (cos(2 theta), sin(2 theta)).  Its covariant derivative gives
    # d theta without angle unwrapping or a branch cut.
    u = (qxx - qyy) / safe_discriminant
    v = 2.0 * qxy / safe_discriminant
    du_dx = _centered_difference(u, 1)
    du_dy = _centered_difference(u, 0)
    dv_dx = _centered_difference(v, 1)
    dv_dy = _centered_difference(v, 0)
    theta_x = 0.5 * (u * dv_dx - v * du_dx)
    theta_y = 0.5 * (u * dv_dy - v * du_dy)

    # Recover either representative of the principal normal.  Reversing it
    # also reverses the tangent and disappears under the absolute value.
    normal_x = np.sqrt(np.maximum(0.5 * (1.0 + u), 0.0))
    normal_y = np.copysign(
        np.sqrt(np.maximum(0.5 * (1.0 - u), 0.0)),
        np.where(np.abs(v) > 1e-30, v, 1.0),
    )
    tangent_x = -normal_y
    tangent_y = normal_x
    curvature = coherence * np.abs(
        tangent_x * theta_x + tangent_y * theta_y)

    high = np.maximum(0.5 * (trace + discriminant), 1e-30)
    low = np.maximum(0.5 * (trace - discriminant), 1e-30)
    tangent_span = 1.0 / np.sqrt(low)
    normal_span = 1.0 / np.sqrt(high)
    sagitta_ratio = (
        curvature * tangent_span * tangent_span
        / np.maximum(2.0 * normal_span, 1e-30)
    )
    population_factor = np.sqrt(np.maximum(1.0, sagitta_ratio))

    raw_measure = (
        np.maximum(np.asarray(geometry["measure"], dtype=np.float64), 0.0)
        * base_implied
    )
    corrected_measure = raw_measure * population_factor
    corrected_implied = max(float(np.sum(corrected_measure)), 1e-30)

    result = dict(geometry)
    result["measure"] = np.ascontiguousarray(
        corrected_measure / corrected_implied, dtype=np.float32)
    result["implied_cells"] = corrected_implied
    result["straight_implied_cells"] = base_implied
    result["director_curvature"] = np.ascontiguousarray(
        curvature, dtype=np.float32)
    result["curvature_sagitta_ratio"] = np.ascontiguousarray(
        sagitta_ratio, dtype=np.float32)
    result["curvature_population_factor"] = np.ascontiguousarray(
        population_factor, dtype=np.float32)
    result["curvature_backend"] = "NumPy reference"
    return result


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
