"""Subpixel coverage from collisions of already-accepted transport fronts.

No second owner is propagated and no site competes for a ranked pixel list.
When two first-arrival regions meet on a lattice edge, their accepted action
and the cost of that one edge determine where the continuous equality surface
crossed it.  Only the pixel footprint cut by that surface receives fractional
coverage from the colour across the interface.
"""

from __future__ import annotations

import numpy as np


def _accumulate_axis(
    reconstruction: np.ndarray,
    labels: np.ndarray,
    distance: np.ndarray,
    edge_cost: np.ndarray,
    confidence: np.ndarray,
    *,
    axis: int,
    strength: float,
    neighbour_sum: np.ndarray,
    coverage_sum: np.ndarray,
) -> int:
    if axis == 1:
        first_slice = (slice(None), slice(0, -1))
        second_slice = (slice(None), slice(1, None))
    else:
        first_slice = (slice(0, -1), slice(None))
        second_slice = (slice(1, None), slice(None))

    first_label = labels[first_slice]
    second_label = labels[second_slice]
    collision = first_label != second_label
    if not np.any(collision):
        return 0

    cost = np.maximum(np.asarray(edge_cost, dtype=np.float64), 1e-12)
    delta = distance[second_slice] - distance[first_slice]
    crossing = np.clip(0.5 + 0.5 * delta / cost, 0.0, 1.0)
    edge_confidence = 0.5 * (
        confidence[first_slice] + confidence[second_slice])
    amount_first = np.where(
        collision & (crossing < 0.5),
        (0.5 - crossing) * edge_confidence * strength,
        0.0,
    )
    amount_second = np.where(
        collision & (crossing > 0.5),
        (crossing - 0.5) * edge_confidence * strength,
        0.0,
    )

    neighbour_sum[first_slice] += (
        amount_first[..., None] * reconstruction[second_slice])
    coverage_sum[first_slice] += amount_first
    neighbour_sum[second_slice] += (
        amount_second[..., None] * reconstruction[first_slice])
    coverage_sum[second_slice] += amount_second
    return int(np.count_nonzero(collision))


def fractional_interface_coverage(
    reconstruction: np.ndarray,
    labels: np.ndarray,
    distance: np.ndarray,
    cardinal_costs: np.ndarray,
    boundary_confidence: np.ndarray,
    *,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Rasterize a continuous first-arrival interface into pixel coverage."""
    field = np.asarray(reconstruction, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    distance = np.asarray(distance, dtype=np.float64)
    confidence = np.clip(
        np.asarray(boundary_confidence, dtype=np.float64), 0.0, 1.0)
    cardinal = np.asarray(cardinal_costs, dtype=np.float64)
    if (
        field.shape[:2] != labels.shape
        or distance.shape != labels.shape
        or confidence.shape != labels.shape
        or cardinal.shape[:2] != labels.shape
    ):
        raise ValueError("interface fields must share one image geometry")

    applied_strength = max(float(strength), 0.0)
    neighbour_sum = np.zeros_like(field)
    coverage_sum = np.zeros(labels.shape, dtype=np.float64)
    horizontal_edges = _accumulate_axis(
        field,
        labels,
        distance,
        cardinal[:, :-1, 0],
        confidence,
        axis=1,
        strength=applied_strength,
        neighbour_sum=neighbour_sum,
        coverage_sum=coverage_sum,
    )
    vertical_edges = _accumulate_axis(
        field,
        labels,
        distance,
        cardinal[:-1, :, 2],
        confidence,
        axis=0,
        strength=applied_strength,
        neighbour_sum=neighbour_sum,
        coverage_sum=coverage_sum,
    )

    # Corners can be cut by more than one interface.  Preserve a nonnegative
    # contribution from the pixel's own support and scale all incident
    # coverage together rather than selecting one neighbour.
    scale = np.minimum(
        1.0,
        0.95 / np.maximum(coverage_sum, 1e-30),
    )
    effective = coverage_sum * scale
    proposal = (
        (1.0 - effective)[..., None] * field
        + scale[..., None] * neighbour_sum
    )
    return np.clip(proposal, 0.0, 1.0), {
        "coverage": effective,
        "horizontal_collision_edges": horizontal_edges,
        "vertical_collision_edges": vertical_edges,
        "covered_pixels": int(np.count_nonzero(effective > 1e-12)),
        "coverage_mean": float(np.mean(effective)),
        "coverage_p99": float(np.percentile(effective, 99.0)),
        "strength": applied_strength,
    }
