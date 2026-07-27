"""PORT 02: assemble the fixed eight-neighbour transport metric.

This is a streaming stencil, O(pixels), with no cell-count dependence.
Output layout is direction-major ``float32[8, height, width]`` to match the
current exact walk; a native port should benchmark node-major packing too.
"""

from __future__ import annotations

import numpy as np

from experiments.wasserstein_allocation_tree import (
    _edge_cost_stack as _reference_edge_cost_stack,
)


def build_edge_costs(geometry: dict, metric_strength: float) -> np.ndarray:
    return np.ascontiguousarray(
        _reference_edge_cost_stack(geometry, metric_strength),
        dtype=np.float32,
    )


def build_residual_pressure_costs(
    geometry: dict,
    metric_strength: float,
    residual_energy: np.ndarray,
    pressure_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strengthen the pointwise BFFT metric where reconstruction fails.

    Residual changes the local normal/tangent contrast; it never supplies a
    per-cell axis.  A coherent narrow feature therefore becomes a cheap
    tangent channel and an expensive normal crossing at every point along its
    own curved or branching geometry.
    """
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    residual = np.sqrt(np.maximum(
        np.asarray(residual_energy, dtype=np.float64), 0.0))
    robust_scale = max(float(np.percentile(residual, 90.0)), 1e-20)
    trace = qxx + qyy
    coherence = np.hypot(qxx - qyy, 2.0 * qxy) / np.maximum(trace, 1e-30)
    # Squaring makes ambiguous/isotropic texture rapidly lose geometric
    # authority while leaving coherent contours close to unit strength.
    coherence_gate = coherence * coherence
    pressure = residual / (residual + robust_scale) * coherence_gate
    tensor_scale = max(float(np.percentile(qxx + qyy, 90.0)), 1e-12)
    horizon = float(geometry["max_support_px"])
    strength = (
        max(float(metric_strength), 0.0)
        * horizon * horizon
        * (1.0 + max(float(pressure_gain), 0.0) * pressure)
    )
    mxx = 1.0 + strength * qxx / tensor_scale
    mxy = strength * qxy / tensor_scale
    myy = 1.0 + strength * qyy / tensor_scale
    height, width = qxx.shape
    costs = np.full((8, height, width), np.inf, dtype=np.float32)
    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for index, (dy, dx) in enumerate(directions):
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        yd = slice(max(0, dy), min(height, height + dy))
        xd = slice(max(0, dx), min(width, width + dx))
        ax = 0.5 * (mxx[ys, xs] + mxx[yd, xd])
        axy = 0.5 * (mxy[ys, xs] + mxy[yd, xd])
        ay = 0.5 * (myy[ys, xs] + myy[yd, xd])
        costs[index, ys, xs] = np.sqrt(np.maximum(
            dx * dx * ax + 2.0 * dx * dy * axy + dy * dy * ay,
            1e-8,
        ))
    return costs, pressure, coherence_gate
