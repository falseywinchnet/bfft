"""Continuous scalar caustic readout of transported 2-D branch action.

The causal HJ law lives on branch Haar measure, while the requested image is a
scalar pushforward through each branch's signal value.  A branch barycenter
silently ignores that change of measure; a hard branch mode is discontinuous.
This probe instead transports the collision law through the scalar quantile
map and integrates its continuous caustic density.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_lineage_law_2d,
    nested_population_phases,
)
from .fused_transport_geometry import weighted_empirical_quantiles
from .witnessed_characteristic_transport_2d import _validate


def scalar_pushforward_collision_barycenter(
    signal: np.ndarray,
    reference_mass: np.ndarray,
    path_score: np.ndarray,
    collision_order: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project branch action through scalar amplitude before collision.

    If ``Q(u)`` is the quantile map of the order-one terminal path measure,
    its scalar density is ``p(Q(u)) = 1 / Q'(u)``.  An order-alpha collision
    has scalar barycenter

        integral Q(u) p(Q(u))**(alpha - 1) du
        ------------------------------------------------.
        integral      p(Q(u))**(alpha - 1) du

    The branch count resolves the quantile integral.  Machine precision is the
    only derivative floor; there is no amplitude bandwidth or concentration
    setting.
    """
    value = np.asarray(signal, dtype=np.float64)
    haar = np.asarray(reference_mass, dtype=np.float64)
    action = np.asarray(path_score, dtype=np.float64)
    order = np.asarray(collision_order, dtype=np.float64)
    if value.ndim != 3 or haar.shape != value.shape or action.shape != value.shape:
        raise ValueError("signal, Haar mass, and path action must align HxWxK")
    if order.shape != value.shape[:2]:
        raise ValueError("collision order must align with the image domain")
    if (
        not np.all(np.isfinite(value))
        or not np.all(np.isfinite(haar))
        or not np.all(np.isfinite(action))
        or not np.all(np.isfinite(order))
        or np.any(haar < 0.0)
        or np.any(order < 1.0)
    ):
        raise ValueError("terminal caustic state must be finite and physical")
    if np.any(np.sum(haar, axis=-1) <= 0.0):
        raise ValueError("every terminal Haar law needs positive mass")

    # The HJ action is defined up to an additive targetwise gauge.  Recenter
    # before exponentiation and reconstruct the order-one terminal measure.
    centered_action = action - np.max(action, axis=-1, keepdims=True)
    base_mass = haar * np.exp(centered_action)
    base_mass /= np.sum(base_mass, axis=-1, keepdims=True)
    quantile = weighted_empirical_quantiles(
        value, base_mass, value.shape[-1])

    quantile_speed = np.empty_like(quantile)
    quantile_speed[..., 0] = 2.0 * (
        quantile[..., 1] - quantile[..., 0])
    quantile_speed[..., -1] = 2.0 * (
        quantile[..., -1] - quantile[..., -2])
    quantile_speed[..., 1:-1] = (
        quantile[..., 2:] - quantile[..., :-2])
    quantile_speed = np.maximum(quantile_speed, 0.0)
    span = np.ptp(quantile, axis=-1, keepdims=True)
    derivative_floor = np.maximum(
        np.sqrt(np.finfo(float).eps) * span,
        np.finfo(float).tiny,
    )
    log_authority = (
        (1.0 - order[..., None])
        * np.log(np.maximum(quantile_speed, derivative_floor))
    )
    log_authority -= np.max(log_authority, axis=-1, keepdims=True)
    authority = np.exp(log_authority)
    authority /= np.sum(authority, axis=-1, keepdims=True)
    readout = np.sum(authority * quantile, axis=-1)
    return readout, {
        "mean_collision_order": float(np.mean(order)),
        "mean_quantile_speed": float(np.mean(quantile_speed)),
        "zero_speed_fraction": float(np.mean(quantile_speed == 0.0)),
        "maximum_authority": float(np.max(authority)),
    }


def phase_integrated_terminal_caustic_readout_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    phase_count: int = 4,
    memory_ceiling_bytes: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Integrate scalar-caustic sections over population representation."""
    image = _validate(observation)
    phases = nested_population_phases(phase_count)
    sections = []
    diagnostics = []
    for phase in phases:
        law, law_diagnostic = causal_information_lineage_law_2d(
            image,
            angular_count=angular_count,
            quantile_count=quantile_count,
            population_phase=phase,
            memory_ceiling_bytes=memory_ceiling_bytes,
        )
        section, section_diagnostic = scalar_pushforward_collision_barycenter(
            law["signal"],
            law["reference_mass"],
            law["hj_path_score"],
            law["hj_simplex_collision_order"],
        )
        sections.append(section)
        diagnostics.append({**section_diagnostic, "law": law_diagnostic})
    stack = np.stack(sections)
    result = np.mean(stack, axis=0)
    lower = float(np.min(image))
    upper = float(np.max(image))
    result = np.clip(result, lower, upper)
    phase_rms = np.sqrt(np.mean((stack - result[None, ...]) ** 2, axis=(1, 2)))
    return result, {
        "status": "phase-integrated scalar caustic of causal HJ branch action",
        "physical_parameters": "none",
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "phase_count": int(phase_count),
        "phases": phases,
        "phase_mean_rms": float(np.mean(phase_rms)),
        "phase_maximum_rms": float(np.max(phase_rms)),
        "mean_collision_order": float(np.mean([
            row["mean_collision_order"] for row in diagnostics])),
        "mean_zero_speed_fraction": float(np.mean([
            row["zero_speed_fraction"] for row in diagnostics])),
        "theory_status": (
            "2-D terminal-measure probe; not promoted to FMMT or the GUI"
        ),
    }
