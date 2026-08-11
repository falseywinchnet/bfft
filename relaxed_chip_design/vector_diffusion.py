"""Connection-valued orientation diffusion without destination search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def diffuse_connection_orientations(
    local_directions: np.ndarray,
    local_radii: np.ndarray,
    confidence: np.ndarray,
    nets: Sequence[np.ndarray],
    endpoint_counts: np.ndarray,
    *,
    net_survival: np.ndarray | None = None,
    diffusion: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Diffuse unit directions while retaining every node's local radius.

    ``nets`` contains only movable node indices; ``endpoint_counts`` also
    counts fixed endpoints so high-fanout authority matches the production
    conditioner.  The operation is linear in total movable pins and returns
    one flux vector per node.  It never constructs or scores destinations.
    """
    directions = np.asarray(local_directions, dtype=np.float64)
    radii = np.asarray(local_radii, dtype=np.float64)
    gain = np.asarray(confidence, dtype=np.float64)
    endpoint_counts = np.asarray(endpoint_counts, dtype=np.int64)
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("local directions must be nodes x 2")
    if radii.shape != (len(directions),):
        raise ValueError("local radii must match nodes")
    if gain.shape != (len(directions),):
        raise ValueError("confidence must match nodes")
    if endpoint_counts.shape != (len(nets),):
        raise ValueError("endpoint counts must match nets")
    if np.any(radii < 0.0):
        raise ValueError("local radii must be nonnegative")
    if np.any((gain < 0.0) | (gain > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")
    if not 0.0 <= diffusion <= 1.0:
        raise ValueError("diffusion must lie in [0, 1]")
    if np.any(endpoint_counts <= 0):
        raise ValueError("endpoint counts must be positive")
    norms = np.linalg.norm(directions, axis=1)
    unit = np.zeros_like(directions)
    covered = norms > 1e-15
    unit[covered] = directions[covered] / norms[covered, None]
    if net_survival is None:
        survival = np.ones(len(nets), dtype=np.float64)
    else:
        survival = np.asarray(net_survival, dtype=np.float64)
        if survival.shape != (len(nets),):
            raise ValueError("net survival must match nets")
        if np.any((survival < 0.0) | (survival > 1.0)):
            raise ValueError("net survival must lie in [0, 1]")

    direction_sum = np.zeros_like(unit)
    confidence_sum = np.zeros(len(unit), dtype=np.float64)
    neighbor_weight = np.zeros(len(unit), dtype=np.float64)
    for net_index, raw_cells in enumerate(nets):
        cells = np.asarray(raw_cells, dtype=np.int64)
        if not len(cells):
            continue
        if np.any((cells < 0) | (cells >= len(unit))):
            raise ValueError("net contains an invalid node")
        admitted = gain[cells]
        admitted_total = float(np.sum(admitted))
        if admitted_total <= 1e-300:
            continue
        mean_direction = np.sum(
            admitted[:, None] * unit[cells], axis=0
        ) / admitted_total
        mean_norm = float(np.linalg.norm(mean_direction))
        if mean_norm <= 1e-15:
            continue
        mean_direction /= mean_norm
        weight = float(survival[net_index]) / int(endpoint_counts[net_index])
        if weight <= 1e-300:
            continue
        direction_sum[cells] += weight * mean_direction
        confidence_sum[cells] += weight * float(np.mean(admitted))
        neighbor_weight[cells] += weight

    has_neighbor = neighbor_weight > 0.0
    neighbor_direction = np.zeros_like(unit)
    neighbor_direction[has_neighbor] = (
        direction_sum[has_neighbor] / neighbor_weight[has_neighbor, None]
    )
    neighbor_norm = np.linalg.norm(neighbor_direction, axis=1)
    valid_neighbor = neighbor_norm > 1e-15
    neighbor_direction[valid_neighbor] /= neighbor_norm[
        valid_neighbor, None
    ]
    neighbor_confidence = np.zeros(len(unit), dtype=np.float64)
    neighbor_confidence[has_neighbor] = (
        confidence_sum[has_neighbor] / neighbor_weight[has_neighbor]
    )
    relaxation = float(diffusion) * (1.0 - gain) ** 2
    relaxation[~valid_neighbor] = 0.0
    propagated = gain + relaxation * np.maximum(
        neighbor_confidence - gain, 0.0
    )
    mixed = (
        (1.0 - relaxation)[:, None] * unit
        + relaxation[:, None] * neighbor_direction
    )
    mixed_norm = np.linalg.norm(mixed, axis=1)
    valid_mixed = mixed_norm > 1e-15
    mixed[valid_mixed] /= mixed_norm[valid_mixed, None]
    flux = radii[:, None] * propagated[:, None] * mixed
    cosine = np.einsum("nd,nd->n", mixed, unit, optimize=True)
    working_bytes = int(
        unit.nbytes
        + survival.nbytes
        + direction_sum.nbytes
        + confidence_sum.nbytes
        + neighbor_weight.nbytes
        + neighbor_direction.nbytes
        + neighbor_confidence.nbytes
        + relaxation.nbytes
        + propagated.nbytes
        + mixed.nbytes
        + flux.nbytes
    )
    return flux, {
        "method": "u1_net_vector_diffusion_local_radius",
        "graph_propagated_fraction": float(np.mean(
            propagated > gain + 1e-15
        )),
        "redirected_fraction": float(np.mean(
            valid_mixed & (cosine < 1.0 - 1e-9)
        )),
        "mean_direction_cosine": (
            float(np.mean(cosine[valid_mixed]))
            if np.any(valid_mixed) else 1.0
        ),
        "working_bytes": working_bytes,
        "candidate_destinations_materialized": False,
    }
