"""Causal affine atlas commanded by continuous predictive information volume.

This is a theorem probe, not a promoted denoiser. The held-out continuous
tangent law supplies scalar horizontal Wasserstein geometry. Its volume is
quantized only for a population-phase refinement experiment; all emitted germs
share one eikonal label while retaining distinct causal identities. Root signal
and jet state is then parallel transported as an affine chart through the
recorded Hopf--Lax ancestry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_ancestry import shared_label_continuous_causal_forest
from .continuous_tangent_transport_2d import (
    continuous_tangent_jet_field_2d,
    continuous_tangent_signal_population_2d,
)
from .fused_transport_geometry import (
    predictive_horizontal_wasserstein_geometry,
    predictive_lineage_jet_geometry,
    weighted_empirical_quantiles,
)
from .witnessed_characteristic_transport_2d import _validate


def _sample_centers(field: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Bilinearly sample a raster field at normalized continuous centers."""
    image = np.asarray(field, dtype=np.float64)
    points = np.asarray(centers, dtype=np.float64)
    height, width = image.shape
    x = np.clip(points[:, 0] * width - 0.5, 0.0, width - 1.0)
    y = np.clip(points[:, 1] * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    a = x - x0
    b = y - y0
    return (
        (1.0 - a) * (1.0 - b) * image[y0, x0]
        + a * (1.0 - b) * image[y0, x1]
        + (1.0 - a) * b * image[y1, x0]
        + a * b * image[y1, x1]
    )


def continuous_tangent_causal_atlas_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
    quantile_count: int = 32,
    population_phase: float = 0.0,
    memory_ceiling_bytes: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transport held-out affine charts through a shared-label causal forest."""
    image = _validate(observation)
    signal, signal_diagnostic = continuous_tangent_signal_population_2d(
        image, angular_count=angular_count)
    quantiles = weighted_empirical_quantiles(
        signal["prediction"], signal["mass"], quantile_count)
    horizontal = predictive_horizontal_wasserstein_geometry(quantiles)

    # The import is delayed because this experimental bridge uses the native
    # V3 runtime, while the tangent and information kernels do not require it.
    from port_needed.continuous_eikonal_transport import prepare_continuous_metric
    from port_needed.density_population import emit_density_population

    centers, population = emit_density_population(
        {
            "measure": horizontal["measure"],
            "implied_cells": horizontal["implied_support"],
        },
        safety_cells=image.size,
        phase_shift=float(population_phase),
    )
    if population["safety_limit_hit"]:
        raise RuntimeError("domain-sized numerical population ceiling was hit")
    prepared = prepare_continuous_metric(
        horizontal["metric_xx"],
        horizontal["metric_xy"],
        horizontal["metric_yy"],
        consistency_limit=np.finfo(float).max,
    )
    forest, ancestry = shared_label_continuous_causal_forest(
        centers,
        prepared,
        memory_ceiling_bytes=memory_ceiling_bytes,
    )

    prior_population = {
        "mass": signal["mass"],
        "directional_derivative": signal["directional_derivative"],
        "tangent": signal["tangent"],
    }
    gradient_x, gradient_y, jet_diagnostic = continuous_tangent_jet_field_2d(
        prior_population)
    barycenter = np.sum(signal["mass"] * signal["prediction"], axis=-1)
    root_value = _sample_centers(barycenter, centers)
    root_gradient_x = _sample_centers(gradient_x, centers)
    root_gradient_y = _sample_centers(gradient_y, centers)

    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    center_x = np.clip(centers[:, 0] * width - 0.5, 0.0, width - 1.0)
    center_y = np.clip(centers[:, 1] * height - 0.5, 0.0, height - 1.0)
    root_chart = (
        root_value[None, None, :]
        + root_gradient_x[None, None, :] * (xx[..., None] - center_x)
        + root_gradient_y[None, None, :] * (yy[..., None] - center_y)
    )
    field = np.sum(ancestry.weights * root_chart, axis=-1)
    causal_jet_geometry = predictive_lineage_jet_geometry(
        ancestry.weights,
        root_gradient_x,
        root_gradient_y,
        quantile_count=quantile_count,
    )
    residual = image - field
    return np.clip(field, 0.0, 1.0), {
        "status": "shared-label Hopf--Lax affine atlas phase probe",
        "theory_status": (
            "continuous predictive metric with numerical germ realization; "
            "joint residual transport and phase convergence pending"
        ),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "population_phase": float(population_phase),
        "signal": signal_diagnostic,
        "horizontal_implied_support": float(horizontal["implied_support"]),
        "horizontal_geometry": horizontal,
        "population": population,
        "centers": centers,
        "forest": forest,
        "mean_collision_population": float(np.mean(
            ancestry.collision_population)),
        "maximum_collision_population": float(np.max(
            ancestry.collision_population)),
        "causal_jet_implied_support": float(
            causal_jet_geometry["implied_support"]),
        "causal_jet_geometry": causal_jet_geometry,
        "jet": jet_diagnostic,
        "root_value": root_value,
        "root_gradient_x": root_gradient_x,
        "root_gradient_y": root_gradient_y,
        "observation_graph_maximum_error": float(np.max(np.abs(
            field + residual - image))),
        "unresolved": [
            "population germ phase must converge",
            "root chart sampling is not target-excluded for every destination",
            "residual atoms are not yet transported through the causal DAG",
            "vertical jet population is diagnostic only",
        ],
    }
