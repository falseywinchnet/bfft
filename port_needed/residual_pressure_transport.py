"""PORT 09: fixed-population residual-pressure transport.

Residual is a density in the same transport problem as the original support;
it is not an event that creates descendants.  The population is conserved.
Every pass performs three simultaneous operations:

1. exact two-label anisotropic power transport;
2. soft mass/barycenter reduction under support + residual density;
3. position and additive power-weight updates for every site.

The BFFT metric remains pointwise throughout.  No cell covariance, principal
axis, split, merge, candidate ranking, or deletion appears in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .two_label_transport import hard_partition_with_forest


def pressure_density(
    support_measure: np.ndarray,
    residual_energy: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.maximum(
        np.asarray(support_measure, dtype=np.float64), 0.0)
    support /= max(float(np.sum(support)), 1e-30)
    residual = np.maximum(
        np.asarray(residual_energy, dtype=np.float64), 0.0)
    # Square-root compression prevents a single edge pixel from becoming a
    # population command while retaining the entire spatial residual field.
    residual = np.sqrt(residual)
    residual /= max(float(np.sum(residual)), 1e-30)
    gain = max(float(strength), 0.0)
    density = support + gain * residual
    density /= max(float(np.sum(density)), 1e-30)
    return density, residual


def _soft_mass_and_barycenter(
    forest: dict,
    density_2d: np.ndarray,
    cells: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    owner = np.asarray(forest["labels"], dtype=np.int32).ravel()
    runner = np.asarray(forest["runner"], dtype=np.int32).ravel()
    first = np.asarray(forest["distance"], dtype=np.float64).ravel()
    second = np.asarray(
        forest["second_distance"], dtype=np.float64).ravel()
    density = np.asarray(density_2d, dtype=np.float64).ravel()
    height, width = density_2d.shape
    valid = runner >= 0
    runner_safe = np.where(valid, runner, owner)
    gap = np.zeros_like(first)
    gap[valid] = second[valid] - first[valid]
    owner_weight = np.ones_like(first)
    owner_weight[valid] = expit(np.clip(
        gap[valid] / max(float(temperature), 1e-6), 0.0, 40.0))
    runner_weight = np.where(valid, 1.0 - owner_weight, 0.0)
    yy, xx = np.mgrid[:height, :width]
    x = (xx.ravel().astype(np.float64) + 0.5) / width
    y = (yy.ravel().astype(np.float64) + 0.5) / height

    def both(value):
        return (
            np.bincount(
                owner,
                weights=density * owner_weight * value,
                minlength=cells,
            )
            + np.bincount(
                runner_safe,
                weights=density * runner_weight * value,
                minlength=cells,
            )
        )

    mass = both(np.ones_like(density))
    safe = np.maximum(mass, 1e-30)
    barycenter = np.column_stack((both(x) / safe, both(y) / safe))
    return mass, barycenter


def relax_residual_pressure(
    centers: np.ndarray,
    costs: np.ndarray,
    support_measure: np.ndarray,
    residual_energy: np.ndarray,
    *,
    passes: int = 4,
    pressure_strength: float = 1.0,
    temperature: float = 2.0,
    position_relaxation: float = 0.35,
    capacity_relaxation: float = 0.5,
    initial_reach: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, list[dict], dict]:
    """Relax one conserved site population under unexplained-energy pressure."""
    centers = np.asarray(centers, dtype=np.float64).copy()
    starting_centers = centers.copy()
    cells = len(centers)
    initial_cells = cells
    reach = (
        np.zeros(cells, dtype=np.float64)
        if initial_reach is None
        else np.asarray(initial_reach, dtype=np.float64).copy()
    )
    if reach.shape != (cells,):
        raise ValueError("initial_reach must have one value per site")
    density, residual_density = pressure_density(
        support_measure, residual_energy, pressure_strength)
    height, width = density.shape
    target_mass = 1.0 / max(cells, 1)
    position_step = float(np.clip(position_relaxation, 0.0, 1.0))
    capacity_step = max(float(capacity_relaxation), 0.0)
    trace = []
    forest = None

    for pass_index in range(max(int(passes), 0)):
        forest = hard_partition_with_forest(centers, costs, reach)
        mass, barycenter = _soft_mass_and_barycenter(
            forest, density, cells, temperature)
        valid = mass > 1e-12
        displacement = np.zeros_like(centers)
        displacement[valid] = barycenter[valid] - centers[valid]
        centers[valid] += position_step * displacement[valid]
        centers[:, 0] = np.clip(
            centers[:, 0], 0.5 / width, 1.0 - 0.5 / width)
        centers[:, 1] = np.clip(
            centers[:, 1], 0.5 / height, 1.0 - 0.5 / height)

        log_error = np.log(
            target_mass / np.maximum(mass, 1e-30))
        log_error = np.clip(log_error, -2.0, 2.0)
        reach += capacity_step * float(temperature) * log_error
        reach -= float(np.mean(reach))
        trace.append({
            "pass": pass_index + 1,
            "cells": cells,
            "mass_rms_error": float(np.sqrt(np.mean(
                (mass / target_mass - 1.0) ** 2))),
            "mass_p10": float(np.percentile(mass / target_mass, 10.0)),
            "mass_p90": float(np.percentile(mass / target_mass, 90.0)),
            "motion_rms_px": float(np.sqrt(np.mean(
                (displacement[:, 0] * width) ** 2
                + (displacement[:, 1] * height) ** 2))),
            "reach_rms": float(np.sqrt(np.mean(reach * reach))),
        })

    forest = hard_partition_with_forest(centers, costs, reach)
    if len(centers) != initial_cells:
        raise AssertionError("fixed-population pressure changed site count")
    return centers, reach, forest, trace, {
        "density": density,
        "residual_density": residual_density,
        "initial_cells": initial_cells,
        "final_cells": len(centers),
        "initial_centers": starting_centers,
        "motion": centers - starting_centers,
    }
