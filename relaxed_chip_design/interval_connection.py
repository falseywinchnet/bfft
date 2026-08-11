"""Oriented HPWL interval stalks for connection-valued transport."""

from __future__ import annotations

import numpy as np


def decompose_interval_stalk(
    local_directions: np.ndarray,
    confidence: np.ndarray,
    movable_points: np.ndarray,
    fixed_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Separate common direction from identifiable boundary-normal detail.

    The returned state is ``(even, odd, face_sign)``.  ``even`` is the
    confidence-weighted common two-vector.  An ``odd`` coordinate is admitted
    only when supported movable witnesses attain both faces of that HPWL
    interval.  ``face_sign`` is +1 on a lower face, -1 on an upper face, and
    zero away from an achieving face.

    The components remain separate by design.  A transport may diffuse them
    in a four-dimensional stalk and call :func:`synthesize_interval_stalk`
    only during unrelaxation.
    """
    directions = np.asarray(local_directions, dtype=np.float64)
    weights = np.asarray(confidence, dtype=np.float64)
    movable = np.asarray(movable_points, dtype=np.float64)
    fixed = (
        np.empty((0, 2), dtype=np.float64)
        if fixed_points is None
        else np.asarray(fixed_points, dtype=np.float64)
    )
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("local directions must be endpoints x 2")
    if weights.shape != (len(directions),):
        raise ValueError("confidence must match local directions")
    if movable.shape != directions.shape:
        raise ValueError("movable points must match local directions")
    if fixed.ndim != 2 or fixed.shape[1] != 2:
        raise ValueError("fixed points must be endpoints x 2")
    if np.any(weights < 0.0):
        raise ValueError("confidence must be nonnegative")
    if not len(directions):
        raise ValueError("at least one movable endpoint is required")

    norms = np.linalg.norm(directions, axis=1)
    unit = np.zeros_like(directions)
    covered = norms > 1e-15
    unit[covered] = directions[covered] / norms[covered, None]
    total_weight = float(np.sum(weights))
    even = np.zeros(2, dtype=np.float64)
    if total_weight > 1e-300:
        even = np.sum(weights[:, None] * unit, axis=0) / total_weight

    endpoints = np.vstack((movable, fixed)) if len(fixed) else movable
    face_sign = np.zeros_like(movable)
    odd = np.zeros(2, dtype=np.float64)
    paired_axes = 0
    supported_face_incidences = 0
    for axis in range(2):
        coordinate = endpoints[:, axis]
        minimum = float(np.min(coordinate))
        maximum = float(np.max(coordinate))
        if maximum - minimum <= 1e-12:
            continue
        low = np.abs(movable[:, axis] - minimum) <= 1e-9
        high = np.abs(movable[:, axis] - maximum) <= 1e-9
        face_sign[low, axis] = 1.0
        face_sign[high, axis] = -1.0
        low_supported = low & covered & (weights > 0.0)
        high_supported = high & covered & (weights > 0.0)
        low_weight = float(np.sum(weights[low_supported]))
        high_weight = float(np.sum(weights[high_supported]))
        supported_face_incidences += int(np.sum(low_supported | high_supported))
        if low_weight <= 1e-300 or high_weight <= 1e-300:
            continue
        low_mode = float(np.sum(
            weights[low_supported] * unit[low_supported, axis]
        ) / low_weight)
        high_mode = float(np.sum(
            weights[high_supported] * unit[high_supported, axis]
        ) / high_weight)
        odd[axis] = 0.5 * (low_mode - high_mode)
        paired_axes += 1

    working_bytes = int(
        unit.nbytes + endpoints.nbytes + face_sign.nbytes + odd.nbytes
    )
    return even, odd, face_sign, {
        "method": "paired_even_odd_hpwl_interval_stalk",
        "paired_axis_count": paired_axes,
        "supported_face_incidence_count": supported_face_incidences,
        "working_bytes": working_bytes,
        "candidate_destinations_materialized": False,
    }


def synthesize_interval_stalk(
    even: np.ndarray,
    odd: np.ndarray,
    face_sign: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Restrict a transported even/odd stalk into endpoint directions."""
    common = np.asarray(even, dtype=np.float64)
    detail = np.asarray(odd, dtype=np.float64)
    signs = np.asarray(face_sign, dtype=np.float64)
    if common.shape != (2,) or detail.shape != (2,):
        raise ValueError("even and odd must be two-vectors")
    if signs.ndim != 2 or signs.shape[1] != 2:
        raise ValueError("face signs must be endpoints x 2")
    directions = common[None, :] + signs * detail[None, :]
    if normalize:
        norms = np.linalg.norm(directions, axis=1)
        covered = norms > 1e-15
        directions[covered] /= norms[covered, None]
    return directions
