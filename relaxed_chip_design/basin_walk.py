"""Low-dimensional probes for initial-placement transport basins.

These routines construct fixed diagnostic paths.  They do not score points on
the path or select a placement by an objective.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FarFieldFit:
    """A displacement projected onto a supplied physical chart."""

    operator: np.ndarray
    predicted_displacement: np.ndarray
    explained_energy_fraction: float
    basis_condition: float


def inverse_stereographic(xy: np.ndarray) -> np.ndarray:
    """Lift planar coordinates to the unit two-sphere."""

    raw = np.asarray(xy, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("xy must be a points x 2 matrix")
    radius2 = np.sum(raw * raw, axis=1)
    denominator = 1.0 + radius2
    return np.column_stack(
        (
            2.0 * raw[:, 0] / denominator,
            2.0 * raw[:, 1] / denominator,
            (radius2 - 1.0) / denominator,
        )
    )


def spherical_chart_walk(
    initial_xy: np.ndarray,
    target_xy: np.ndarray,
    alpha: float,
    axes: str = "xy",
) -> np.ndarray:
    """Spherically interpolate two normalized physical charts.

    ``axes`` may retain only the walked x or y coordinate.  This is useful for
    fixed axis ablations that distinguish horizontal gauge from row phase.
    """

    initial = np.asarray(initial_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if initial.shape != target.shape or initial.ndim != 2:
        raise ValueError("initial_xy and target_xy must be equal matrices")
    if initial.shape[1] != 2:
        raise ValueError("physical charts must have two coordinates")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if axes not in {"xy", "x", "y"}:
        raise ValueError("axes must be 'xy', 'x', or 'y'")

    left = inverse_stereographic(1.5 * (2.0 * initial - 1.0))
    right = inverse_stereographic(1.5 * (2.0 * target - 1.0))
    cosine = np.clip(np.sum(left * right, axis=1), -1.0, 1.0)
    angle = np.arccos(cosine)
    sine = np.sin(angle)
    near = np.abs(sine) <= 1e-12
    walked = np.empty_like(left)
    walked[near] = (1.0 - alpha) * left[near] + alpha * right[near]
    walked[~near] = (
        np.sin((1.0 - alpha) * angle[~near])[:, None]
        / sine[~near, None]
        * left[~near]
        + np.sin(alpha * angle[~near])[:, None]
        / sine[~near, None]
        * right[~near]
    )
    walked /= np.maximum(
        np.linalg.norm(walked, axis=1, keepdims=True), 1e-300
    )
    raw = walked[:, :2] / np.maximum(1.0 - walked[:, 2:3], 1e-12)
    result = np.clip(0.5 * (raw / 1.5 + 1.0), 0.0, 1.0)
    if axes == "x":
        result[:, 1] = initial[:, 1]
    elif axes == "y":
        result[:, 0] = initial[:, 0]
    return result


def fit_chart_displacement(
    initial_xy: np.ndarray,
    target_xy: np.ndarray,
    directions: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> FarFieldFit:
    """Fit target displacement in a fixed low-dimensional physical chart."""

    initial = np.asarray(initial_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    chart = np.asarray(directions, dtype=np.float64)
    if initial.shape != target.shape or initial.ndim != 2:
        raise ValueError("initial_xy and target_xy must be equal matrices")
    if initial.shape[1] != 2 or chart.ndim != 2:
        raise ValueError("physical points and directions must be matrices")
    if len(chart) != len(initial):
        raise ValueError("directions must have one row per point")
    cell_weights = (
        np.ones(len(initial), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if cell_weights.shape != (len(initial),) or np.any(cell_weights <= 0.0):
        raise ValueError("weights must be one positive value per point")

    basis = np.column_stack((np.ones(len(chart)), chart))
    root = np.sqrt(cell_weights / np.mean(cell_weights))[:, None]
    displacement = target - initial
    operator, _, _, singular = np.linalg.lstsq(
        root * basis, root * displacement, rcond=None
    )
    predicted = basis @ operator
    total = float(np.sum(cell_weights[:, None] * displacement**2))
    residual = float(
        np.sum(cell_weights[:, None] * (displacement - predicted) ** 2)
    )
    condition = (
        float(singular[0] / singular[-1])
        if len(singular) and singular[-1] > 0.0
        else float("inf")
    )
    return FarFieldFit(
        operator=operator,
        predicted_displacement=predicted,
        explained_energy_fraction=(
            1.0 - residual / total if total > 0.0 else 1.0
        ),
        basis_condition=condition,
    )
