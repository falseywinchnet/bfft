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


def parallel_transport_interval_detail(
    transported_support: np.ndarray,
    reference_support: np.ndarray,
    active_segments: np.ndarray,
    anchor_segments: np.ndarray,
    segment_centers: np.ndarray,
    odd_detail: np.ndarray,
    row_mass: np.ndarray,
    *,
    marginal_tolerance: float = 0.01,
    maximum_iterations: int = 400,
) -> tuple[np.ndarray, dict]:
    """Carry odd detail as a marginal-preserving sparse transport tangent.

    Let ``T`` and ``R`` be the transported and reference conditionals and let
    ``h`` be the incidence restriction of the odd two-vector.  The lifted
    sparse kernel is

    ``row_mass * T * exp(h * log(T / R))``.

    One sparse KL projection restores the original cell and segment marginals.
    Thus zero odd detail reproduces ``T``, ``T == R`` remains the identity, and
    phase cannot create uncompensated capacity.  The tolerance is measured in
    the same physical capacity units as ``row_mass``; 0.01 is one hundredth of
    an indivisible standard-cell site in the production harness.
    """
    transported = np.asarray(transported_support, dtype=np.float64)
    reference = np.asarray(reference_support, dtype=np.float64)
    active = np.asarray(active_segments, dtype=np.int64)
    anchors = np.asarray(anchor_segments, dtype=np.int64)
    centers = np.asarray(segment_centers, dtype=np.float64)
    odd = np.asarray(odd_detail, dtype=np.float64)
    masses = np.asarray(row_mass, dtype=np.float64)
    if transported.shape != reference.shape or transported.shape != active.shape:
        raise ValueError("support and active segment shapes must match")
    if transported.ndim != 2 or transported.shape[1] == 0:
        raise ValueError("support must be cells x nonempty fixed width")
    if anchors.shape != (len(active),) or odd.shape != (len(active), 2):
        raise ValueError("anchor and odd state must match cells")
    if masses.shape != (len(active),) or np.any(masses <= 0.0):
        raise ValueError("positive row mass must match cells")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("segment centers must be segments x 2")
    if not np.isfinite(marginal_tolerance) or marginal_tolerance <= 0.0:
        raise ValueError("marginal tolerance must be positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum iterations must be positive")

    valid = active >= 0
    if np.any(~np.any(valid, axis=1)):
        raise ValueError("every cell must have active support")
    safe = np.maximum(active, 0)
    if np.any(safe >= len(centers)) or np.any(
        (anchors < 0) | (anchors >= len(centers))
    ):
        raise ValueError("segment index is out of range")
    transported = np.where(valid, np.maximum(transported, 0.0), 0.0)
    reference = np.where(valid, np.maximum(reference, 0.0), 0.0)
    transported /= np.maximum(np.sum(transported, axis=1)[:, None], 1e-300)
    reference /= np.maximum(np.sum(reference, axis=1)[:, None], 1e-300)

    incidence = centers[safe] - centers[anchors, None]
    incidence_norm = np.linalg.norm(incidence, axis=2)
    unit_incidence = np.zeros_like(incidence)
    nonzero = incidence_norm > 1e-15
    unit_incidence[nonzero] = incidence[nonzero] / incidence_norm[nonzero, None]
    restriction = np.einsum(
        "cda,ca->cd", unit_incidence, odd, optimize=True
    )
    restriction = np.clip(restriction, -1.0, 1.0)
    restriction[~valid] = 0.0

    log_transport = np.log(np.maximum(transported, 1e-300))
    log_reference = np.log(np.maximum(reference, 1e-300))
    log_kernel = (
        np.log(masses)[:, None]
        + log_transport
        + restriction * (log_transport - log_reference)
    )
    log_kernel[~valid] = -np.inf
    row_max = np.max(log_kernel, axis=1)
    kernel = np.exp(log_kernel - row_max[:, None])
    kernel[~valid] = 0.0

    segment_count = len(centers)
    active_cell, active_slot = np.nonzero(valid)
    active_group = active[active_cell, active_slot]
    base_coupling = masses[:, None] * transported
    column_mass = np.bincount(
        active_group,
        weights=base_coupling[active_cell, active_slot],
        minlength=segment_count,
    )
    right = np.ones(segment_count, dtype=np.float64)
    left = np.ones(len(active), dtype=np.float64)
    column_error = float("inf")
    relative_error = float("inf")
    iterations = 0
    for iteration in range(maximum_iterations):
        row_denominator = np.sum(kernel * right[safe], axis=1)
        left = masses / np.maximum(row_denominator, 1e-300)
        coupling = left[:, None] * kernel * right[safe]
        column_actual = np.bincount(
            active_group,
            weights=coupling[active_cell, active_slot],
            minlength=segment_count,
        )
        column_error = float(np.max(np.abs(column_actual - column_mass)))
        relative_error = float(np.max(
            np.abs(column_actual - column_mass)
            / np.maximum(column_mass, 1.0)
        ))
        iterations = iteration + 1
        if column_error <= marginal_tolerance:
            break
        covered = column_mass > 0.0
        right[covered] *= column_mass[covered] / np.maximum(
            column_actual[covered], 1e-300
        )
        right[~covered] = 0.0

    lifted = coupling / masses[:, None]
    lifted[~valid] = 0.0
    row_error = float(np.max(np.abs(np.sum(coupling, axis=1) - masses)))
    delta = lifted - transported
    working_bytes = int(
        incidence.nbytes
        + incidence_norm.nbytes
        + unit_incidence.nbytes
        + restriction.nbytes
        + log_transport.nbytes
        + log_reference.nbytes
        + log_kernel.nbytes
        + kernel.nbytes
        + base_coupling.nbytes
        + column_mass.nbytes
        + right.nbytes
        + left.nbytes
        + coupling.nbytes
        + lifted.nbytes
    )
    return lifted, {
        "method": "sparse_kl_parallel_transport_of_odd_ratio_tangent",
        "iterations": iterations,
        "converged": column_error <= marginal_tolerance,
        "maximum_iterations": int(maximum_iterations),
        "segment_marginal_tolerance": float(marginal_tolerance),
        "row_mass_max_error": row_error,
        "segment_mass_max_error": column_error,
        "segment_mass_max_relative_error": relative_error,
        "mean_support_l1_shift": float(np.mean(np.sum(np.abs(delta), axis=1))),
        "changed_support_cell_fraction": float(np.mean(
            np.sum(np.abs(delta), axis=1) > 1e-12
        )),
        "identity_preserving_when_measures_equal": True,
        "base_transport_preserved_when_odd_zero": True,
        "reference_phase_chart_preserved": True,
        "candidate_destinations_materialized": False,
        "working_bytes": working_bytes,
    }
