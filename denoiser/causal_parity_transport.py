"""Observation-excluded parity transport through a shared-label eikonal DAG.

This is the first executable pre-ownership image experiment. The two parity
lanes arise from grid topology. Each target is read only from the opposite
lane; distinct root values and identities are transported through the same
continuous-simplex law under the determinant-one relation metric.

The metric still reads the complete observation and the ancestry matrix is a
dense research representation. Consequently this module is a falsification
candidate, not a promoted denoiser.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_ancestry import shared_label_causal_forest
from .cross_predictive_transport_2d import relation_transport_metric_2d


def parity_root_pixels(shape: tuple[int, int], parity: int) -> np.ndarray:
    """Return one of the two disjoint topological observation lanes."""
    height, width = int(shape[0]), int(shape[1])
    lane = int(parity)
    if height < 2 or width < 2 or lane not in (0, 1):
        raise ValueError("parity roots need a 2-D domain and parity 0 or 1")
    yy, xx = np.mgrid[:height, :width]
    return np.flatnonzero(((yy + xx) & 1).ravel() == lane)


def denoise_causal_parity_transport(
    observation: np.ndarray,
    *,
    memory_ceiling_bytes: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-predict both parity lanes by exact shared-label ancestry."""
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("causal parity transport expects an HxW field")
    if not np.all(np.isfinite(image)):
        raise ValueError("causal parity transport requires finite samples")

    from port_needed.continuous_eikonal_transport import prepare_continuous_metric

    geometry = relation_transport_metric_2d(image)
    prepared = prepare_continuous_metric(
        geometry["metric_xx"],
        geometry["metric_xy"],
        geometry["metric_yy"],
        consistency_limit=np.finfo(float).max,
    )
    estimate = np.empty_like(image)
    collision_population = np.empty_like(image)
    entropy_population = np.empty_like(image)
    lane_records = []
    flat_image = image.ravel()
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    target_parity = (yy + xx) & 1
    for root_parity in (0, 1):
        roots = parity_root_pixels(image.shape, root_parity)
        forest, ancestry = shared_label_causal_forest(
            roots,
            prepared,
            memory_ceiling_bytes=memory_ceiling_bytes,
        )
        transported = np.tensordot(
            ancestry.weights,
            flat_image[roots],
            axes=([-1], [0]),
        )
        target = target_parity != root_parity
        estimate[target] = transported[target]
        collision_population[target] = ancestry.collision_population[target]
        entropy_population[target] = ancestry.entropy_population[target]
        lane_records.append({
            "root_parity": root_parity,
            "root_count": int(roots.size),
            "target_count": int(np.count_nonzero(target)),
            "front_pushes": forest["front_pushes"],
            "front_maximum_heap": forest["front_maximum_heap"],
            "dense_ancestry_bytes": forest["dense_ancestry_bytes"],
            "target_collision_population_mean": float(np.mean(
                ancestry.collision_population[target])),
            "target_collision_population_max": float(np.max(
                ancestry.collision_population[target])),
        })
    return np.clip(estimate, 0.0, 1.0), {
        "status": "shared-label pre-ownership parity transport seed",
        "theory_status": "falsification candidate; not promoted",
        "observation_exclusion": (
            "each value target is reconstructed only from opposite parity"
        ),
        "metric_observation_exclusion": False,
        "metric_determinant_max_error": float(np.max(np.abs(
            geometry["metric_determinant"] - 1.0))),
        "collision_population": collision_population,
        "entropy_population": entropy_population,
        "lanes": lane_records,
        "unresolved": [
            "metric still reads the held-out target observation",
            "four-direction tangent quadrature is crystalline",
            "dense ancestry is a research representation",
            "one parity-scale source population has no refinement study",
        ],
    }
