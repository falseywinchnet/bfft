"""Backward residual steps through a fixed relaxed transport map.

These primitives never score decoded placements.  A target representation is
fixed first; one transport secant then separates the target residual into the
component visible along that response and its orthogonal remainder.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TransportSecant:
    """Projection of a fixed target residual onto one transport response."""

    response: np.ndarray
    visible_residual: np.ndarray
    orthogonal_residual: np.ndarray
    optimal_coefficient: float
    visible_energy_fraction: float
    orthogonal_energy_fraction: float


def requires_row_self_distillation(soft_continuation_steps: int) -> bool:
    """Return whether a learned row chart contains multi-pass history."""

    return int(soft_continuation_steps) > 1


def graft_transported_rows(
    source_chart: np.ndarray,
    transported_chart: np.ndarray,
) -> np.ndarray:
    """Attach transported row phase to the source horizontal gauge once."""

    source = np.asarray(source_chart, dtype=np.float64)
    transported = np.asarray(transported_chart, dtype=np.float64)
    if source.shape != transported.shape:
        raise ValueError("source and transported charts must have equal shapes")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("physical charts must be points x 2 matrices")
    result = source.copy()
    result[:, 1] = transported[:, 1]
    return result


def project_transport_residual(
    target: np.ndarray,
    base_output: np.ndarray,
    trial_output: np.ndarray,
) -> TransportSecant:
    """Project ``target - base_output`` onto one measured map secant."""

    fixed_target = np.asarray(target, dtype=np.float64)
    base = np.asarray(base_output, dtype=np.float64)
    trial = np.asarray(trial_output, dtype=np.float64)
    if fixed_target.shape != base.shape or base.shape != trial.shape:
        raise ValueError("target and transport outputs must have equal shapes")
    residual = fixed_target - base
    response = trial - base
    residual_flat = residual.ravel()
    response_flat = response.ravel()
    response_energy = float(np.dot(response_flat, response_flat))
    residual_energy = float(np.dot(residual_flat, residual_flat))
    coefficient = (
        float(np.dot(residual_flat, response_flat)) / response_energy
        if response_energy > 1e-300
        else 0.0
    )
    visible = coefficient * response
    orthogonal = residual - visible
    visible_energy = float(np.dot(visible.ravel(), visible.ravel()))
    orthogonal_energy = float(
        np.dot(orthogonal.ravel(), orthogonal.ravel())
    )
    return TransportSecant(
        response=response,
        visible_residual=visible,
        orthogonal_residual=orthogonal,
        optimal_coefficient=coefficient,
        visible_energy_fraction=(
            visible_energy / residual_energy
            if residual_energy > 1e-300 else 1.0
        ),
        orthogonal_energy_fraction=(
            orthogonal_energy / residual_energy
            if residual_energy > 1e-300 else 0.0
        ),
    )


def pullback_residual(
    initial: np.ndarray,
    target: np.ndarray,
    observed_output: np.ndarray,
    step: float,
    *,
    axes: str = "xy",
    maximum_step: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pull a fixed output residual back to the initial chart once.

    ``axes`` freezes a gauge that the transport does not represent.  Optional
    per-axis bounds keep the pullback inside the chart used to measure the
    secant.
    """

    source = np.asarray(initial, dtype=np.float64)
    fixed_target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed_output, dtype=np.float64)
    if source.shape != fixed_target.shape or source.shape != observed.shape:
        raise ValueError("initial, target, and output must have equal shapes")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("physical charts must be points x 2 matrices")
    if axes not in {"x", "y", "xy"}:
        raise ValueError("axes must be 'x', 'y', or 'xy'")
    correction = float(step) * (fixed_target - observed)
    if axes == "x":
        correction[:, 1] = 0.0
    elif axes == "y":
        correction[:, 0] = 0.0
    if maximum_step is not None:
        bound = np.asarray(maximum_step, dtype=np.float64)
        if bound.shape != (2,) or np.any(bound < 0.0):
            raise ValueError("maximum_step must be a nonnegative x/y pair")
        correction = np.clip(correction, -bound, bound)
    return source + correction
