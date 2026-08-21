"""Predictive laws transported through shared-label causal ancestry.

The parity forest's direct value barycenter is rejected. This module retains
the complete ancestry-weighted root-value law instead, converts that atomic law
to a common quantile coordinate, and only then evaluates Wasserstein volume.
Quantile count is numerical measure resolution and must converge; it is not a
noise or image setting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_ancestry import shared_label_causal_forest
from .causal_parity_transport import parity_root_pixels
from .cross_predictive_transport_2d import relation_transport_metric_2d
from .fused_transport_geometry import (
    predictive_horizontal_wasserstein_geometry,
    predictive_wasserstein_geometry,
)


def weighted_quantile_particles(
    weights: np.ndarray,
    values: np.ndarray,
    quantile_count: int,
) -> np.ndarray:
    """Resolve an ancestry-weighted scalar law on common midpoint quantiles."""
    mass = np.asarray(weights, dtype=np.float64)
    support = np.asarray(values, dtype=np.float64).reshape(-1)
    count = int(quantile_count)
    if mass.ndim != 3 or mass.shape[-1] != support.size:
        raise ValueError("weights must be HxWxK aligned with K support values")
    if count < 2:
        raise ValueError("at least two quantile particles are required")
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("predictive weights must be finite and nonnegative")
    total = np.sum(mass, axis=-1)
    if not np.allclose(total, 1.0, rtol=2e-13, atol=2e-13):
        raise ValueError("predictive weights must conserve unit mass")
    order = np.argsort(support, kind="stable")
    sorted_values = support[order]
    cumulative = np.cumsum(mass[..., order], axis=-1)
    cumulative[..., -1] = 1.0
    particles = np.empty(mass.shape[:2] + (count,), dtype=np.float64)
    for index in range(count):
        quantile = (index + 0.5) / count
        atom = np.argmax(cumulative >= quantile, axis=-1)
        particles[..., index] = sorted_values[atom]
    return particles


def causal_parity_predictive_geometry(
    observation: np.ndarray,
    *,
    quantile_count: int = 32,
    memory_ceiling_bytes: int | None = None,
    transport_metric: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transport held-out root laws and extract their Wasserstein geometry."""
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("causal predictive geometry expects an HxW field")
    if not np.all(np.isfinite(image)):
        raise ValueError("causal predictive geometry requires finite samples")
    count = int(quantile_count)
    if count < 2:
        raise ValueError("quantile_count is numerical resolution and must exceed one")

    from port_needed.continuous_eikonal_transport import prepare_continuous_metric

    relation_geometry = relation_transport_metric_2d(image)
    metric = relation_geometry if transport_metric is None else transport_metric
    for key in ("metric_xx", "metric_xy", "metric_yy"):
        if key not in metric or np.asarray(metric[key]).shape != image.shape:
            raise ValueError("transport metric fields must align with observation")
    prepared = prepare_continuous_metric(
        metric["metric_xx"],
        metric["metric_xy"],
        metric["metric_yy"],
        consistency_limit=np.finfo(float).max,
    )
    particles = np.empty(image.shape + (count,), dtype=np.float64)
    collision_population = np.empty_like(image)
    flat_image = image.ravel()
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    target_parity = (yy + xx) & 1
    lane_records = []
    for root_parity in (0, 1):
        roots = parity_root_pixels(image.shape, root_parity)
        forest, ancestry = shared_label_causal_forest(
            roots,
            prepared,
            memory_ceiling_bytes=memory_ceiling_bytes,
        )
        lane_particles = weighted_quantile_particles(
            ancestry.weights,
            flat_image[roots],
            count,
        )
        target = target_parity != root_parity
        particles[target] = lane_particles[target]
        collision_population[target] = ancestry.collision_population[target]
        lane_records.append({
            "root_parity": root_parity,
            "root_count": int(roots.size),
            "front_pushes": forest["front_pushes"],
            "dense_ancestry_bytes": forest["dense_ancestry_bytes"],
        })
    wasserstein = predictive_wasserstein_geometry(particles)
    horizontal_wasserstein = predictive_horizontal_wasserstein_geometry(
        particles)
    return particles, {
        "status": "causal held-out predictive law geometry",
        "theory_status": "fixed-point precursor; not a denoiser",
        "quantile_count": count,
        "predictive_geometry": wasserstein,
        "horizontal_predictive_geometry": horizontal_wasserstein,
        "relation_metric_determinant_max_error": float(np.max(np.abs(
            relation_geometry["metric_determinant"] - 1.0))),
        "transport_metric_source": (
            "full-scale relation conductance"
            if transport_metric is None
            else "supplied fixed-point state"
        ),
        "collision_population": collision_population,
        "lanes": lane_records,
        "unresolved": [
            "relation metric still reads held-out target",
            "direction/curvature Sasaki connection is not yet transported",
            "dense topological parity roots saturate collision population",
            "quantile refinement must converge",
        ],
    }


def causal_predictive_fixed_point(
    observation: np.ndarray,
    *,
    quantile_count: int = 32,
    maximum_continuations: int = 8,
    memory_ceiling_bytes: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Iterate causal law and horizontal metric to numerical equilibrium.

    The physical residual is self-consistency between the marching metric
    ``M`` and the horizontal geometry ``H[mu(M)]`` generated by its transported
    law. A determinant-normalized line search accepts a remarch only when

        ||M - H[mu(M)]||_F^2 / ||M||_F^2

    decreases. Step halving is numerical descent machinery, not physical time;
    its lower limit is derived from floating-point resolution. The continuation
    ceiling remains an explicit failure guard.
    """
    image = np.asarray(observation, dtype=np.float64)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("fixed-point continuation ceiling must be positive")

    def horizontal_metric(diagnostic: dict[str, Any]) -> dict[str, np.ndarray]:
        horizontal = diagnostic["horizontal_predictive_geometry"]
        return {
            "metric_xx": horizontal["metric_xx"],
            "metric_xy": horizontal["metric_xy"],
            "metric_yy": horizontal["metric_yy"],
        }

    def residual(
        metric: dict[str, np.ndarray],
        requested: dict[str, np.ndarray],
    ) -> float:
        numerator = np.mean(
            (metric["metric_xx"] - requested["metric_xx"]) ** 2
            + 2.0 * (metric["metric_xy"] - requested["metric_xy"]) ** 2
            + (metric["metric_yy"] - requested["metric_yy"]) ** 2
        )
        denominator = np.mean(
            metric["metric_xx"] ** 2
            + 2.0 * metric["metric_xy"] ** 2
            + metric["metric_yy"] ** 2
        )
        return float(numerator / max(float(denominator), np.finfo(float).tiny))

    def blend(
        current: dict[str, np.ndarray],
        requested: dict[str, np.ndarray],
        step: float,
    ) -> dict[str, np.ndarray]:
        xx = (1.0 - step) * current["metric_xx"] + step * requested["metric_xx"]
        xy = (1.0 - step) * current["metric_xy"] + step * requested["metric_xy"]
        yy = (1.0 - step) * current["metric_yy"] + step * requested["metric_yy"]
        volume = np.sqrt(np.maximum(xx * yy - xy * xy, np.finfo(float).tiny))
        return {
            "metric_xx": xx / volume,
            "metric_xy": xy / volume,
            "metric_yy": yy / volume,
        }

    def evaluate(metric: dict[str, np.ndarray]):
        particles, diagnostic = causal_parity_predictive_geometry(
            image,
            quantile_count=quantile_count,
            memory_ceiling_bytes=memory_ceiling_bytes,
            transport_metric=metric,
        )
        requested = horizontal_metric(diagnostic)
        return particles, diagnostic, requested, residual(metric, requested)

    metric = relation_transport_metric_2d(image)
    state, state_diagnostic, requested, action = evaluate(metric)
    records = []
    equilibrium = False
    noncontractive = False
    numerical_floor = np.finfo(float).eps
    minimum_step = np.sqrt(np.finfo(float).eps)
    for continuation in range(ceiling):
        record = {
            "continuation": continuation,
            "self_consistency_action": action,
            "horizontal_implied_support": float(
                state_diagnostic["horizontal_predictive_geometry"][
                    "implied_support"]),
            "accepted_step": None,
        }
        records.append(record)
        if action <= numerical_floor:
            equilibrium = True
            break
        if continuation + 1 == ceiling:
            break
        step = 1.0
        accepted = None
        while step >= minimum_step:
            candidate_metric = blend(metric, requested, step)
            candidate = evaluate(candidate_metric)
            candidate_action = candidate[3]
            if candidate_action < action - numerical_floor * max(action, 1.0):
                accepted = (candidate_metric, *candidate)
                break
            step *= 0.5
        if accepted is None:
            noncontractive = True
            break
        metric, state, state_diagnostic, requested, action = accepted
        record["accepted_step"] = step
    ceiling_hit = len(records) == ceiling and not equilibrium
    return state, {
        "status": (
            "causal horizontal predictive equilibrium"
            if equilibrium
            else "fixed-point seed rejected or unresolved"
        ),
        "equilibrium": equilibrium,
        "noncontractive": noncontractive,
        "continuation_ceiling_hit": ceiling_hit,
        "continuations": records,
        "final_state": state_diagnostic,
        "numerical_resolution": {
            "quantile_count": int(quantile_count),
            "maximum_continuations": ceiling,
            "memory_ceiling_bytes": memory_ceiling_bytes,
            "minimum_line_search_step": minimum_step,
        },
    }
