"""Cross-fitted joint posterior/residual contraction of scale-edge zonotopes.

For four primitive orientations, every target edge is validated by the two
parallel edges shifted one lattice unit along its transverse covector.  The
witness edges share no endpoint with the target and lie in the opposite
transverse parity fold.  Their values and signed first jets determine a
tangent in the joint chart.  Only its normal covector is contracted; lawful
amplitude transport along the witnessed tangent remains free.  The current
target coordinate is included, so zero transfer remains feasible.

The same local scale-edge coefficients must satisfy both coordinates:

    lower_value <= r_i - (A alpha)_i <= upper_value,
    lower_jet <= (r_j-r_i) - ((A alpha)_j-(A alpha)_i) <= upper_jet.

Posterior and residual restrict the same exchange coefficients with opposite
signs; neither action is converted into a scalar penalty.  The complete sparse
slab intersection is retained because its axis-aligned coefficient shadow can
remain almost unchanged.  Cross-fold target-exclusion remains weaker than
statistical independence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .continuous_scale_edge_family_transport_2d import (
    _contract_sparse_generator_box,
    _factorized_push_enclosure,
    continuous_scale_edge_family_transport_state_2d,
)
from .witnessed_characteristic_transport_2d import _validate


_VALIDATION_FRAMES = (
    ((0, 1), (1, 0), "horizontal"),
    ((1, 0), (0, 1), "vertical"),
    ((1, 1), (1, -1), "positive_diagonal"),
    ((1, -1), (1, 1), "negative_diagonal"),
)


def _crossfit_value_jet_constraints(
    residual: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return joint coordinate operator and safe opposite-fold bounds."""
    field = np.asarray(residual, dtype=np.float64)
    if field.ndim != 2 or min(field.shape) < 5:
        raise ValueError("cross-fitted value/jet constraints need a 5x5 field")
    height, width = field.shape
    vertex = np.arange(field.size, dtype=np.int64).reshape(field.shape)
    operator_rows: list[int] = []
    operator_columns: list[int] = []
    operator_values: list[float] = []
    coordinate_lower: list[float] = []
    coordinate_upper: list[float] = []
    current_coordinate: list[float] = []
    target_sources: list[int] = []
    target_targets: list[int] = []
    witness_vertices: list[tuple[int, int, int, int]] = []
    orientations: list[int] = []
    folds: list[int] = []
    row = 0
    for orientation, (direction, transverse, _name) in enumerate(
        _VALIDATION_FRAMES
    ):
        dy, dx = direction
        ny, nx = transverse
        for y in range(height):
            for x in range(width):
                points = (
                    (y, x),
                    (y + dy, x + dx),
                    (y - ny, x - nx),
                    (y - ny + dy, x - nx + dx),
                    (y + ny, x + nx),
                    (y + ny + dy, x + nx + dx),
                )
                if not all(
                    0 <= py < height and 0 <= px < width
                    for py, px in points
                ):
                    continue
                (p0, p1, minus0, minus1, plus0, plus1) = points
                target_set = {p0, p1}
                witness_set = {minus0, minus1, plus0, plus1}
                if target_set & witness_set:
                    raise RuntimeError("cross-fit witness shares a target endpoint")
                i = int(vertex[p0])
                j = int(vertex[p1])
                mi = int(vertex[minus0])
                mj = int(vertex[minus1])
                pi = int(vertex[plus0])
                pj = int(vertex[plus1])
                value_current = float(field[p0])
                value_witness = (float(field[minus0]), float(field[plus0]))
                jet_current = float(field[p1] - field[p0])
                jet_witness = (
                    float(field[minus1] - field[minus0]),
                    float(field[plus1] - field[plus0]),
                )
                value_lower = min(*value_witness, value_current)
                value_upper = max(*value_witness, value_current)
                jet_lower = min(*jet_witness, jet_current)
                jet_upper = max(*jet_witness, jet_current)

                # Value row q_value = r_i.
                operator_rows.append(row)
                operator_columns.append(i)
                operator_values.append(1.0)
                coordinate_lower.append(value_lower)
                coordinate_upper.append(value_upper)
                current_coordinate.append(value_current)
                row += 1
                # First-jet row q_jet = r_j-r_i.
                operator_rows.extend((row, row))
                operator_columns.extend((i, j))
                operator_values.extend((-1.0, 1.0))
                coordinate_lower.append(jet_lower)
                coordinate_upper.append(jet_upper)
                current_coordinate.append(jet_current)
                row += 1

                target_sources.append(i)
                target_targets.append(j)
                witness_vertices.append((mi, mj, pi, pj))
                orientations.append(orientation)
                # Transverse shifts change this coordinate's parity.
                fold_coordinate = y if ny != 0 else x
                folds.append(int(fold_coordinate & 1))

    operator = sparse.coo_matrix(
        (operator_values, (operator_rows, operator_columns)),
        shape=(row, field.size),
    ).tocsr()
    lower = np.asarray(coordinate_lower, dtype=np.float64)
    upper = np.asarray(coordinate_upper, dtype=np.float64)
    current = np.asarray(current_coordinate, dtype=np.float64)
    target_sources_array = np.asarray(target_sources, dtype=np.int64)
    target_targets_array = np.asarray(target_targets, dtype=np.int64)
    witness_array = np.asarray(witness_vertices, dtype=np.int64)
    target_exclusion_error = (
        int(np.count_nonzero(
            (witness_array == target_sources_array[:, None])
            | (witness_array == target_targets_array[:, None])
        ))
        if witness_array.size else 0
    )
    fold_array = np.asarray(folds, dtype=np.int64)
    orientation_array = np.asarray(orientations, dtype=np.int64)
    return operator, current - upper, current - lower, {
        "target_edge_count": int(target_sources_array.size),
        "joint_constraint_count": int(operator.shape[0]),
        "target_exclusion_error": target_exclusion_error,
        "fold_zero_edge_count": int(np.sum(fold_array == 0)),
        "fold_one_edge_count": int(np.sum(fold_array == 1)),
        "orientation_edge_counts": {
            name: int(np.sum(orientation_array == ordinal))
            for ordinal, (_direction, _transverse, name) in enumerate(
                _VALIDATION_FRAMES)
        },
        "mean_value_interval_width": float(np.mean(
            upper[0::2] - lower[0::2])) if lower.size else 0.0,
        "mean_jet_interval_width": float(np.mean(
            upper[1::2] - lower[1::2])) if lower.size else 0.0,
        "operator": operator,
        "coordinate_lower": lower,
        "coordinate_upper": upper,
        "current_coordinate": current,
        "target_sources": target_sources_array,
        "target_targets": target_targets_array,
        "witness_vertices": witness_array,
        "fold": fold_array,
        "orientation": orientation_array,
    }


def _covariance_normal_constraints(
    field: np.ndarray,
    rectangle: dict[str, Any],
    *,
    additive_transfer: bool,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
    """Contract only transverse to the witnessed value/jet tangent.

    The two target-excluded parallel witnesses determine a direction in
    ``(value, first jet)`` space.  Its Euclidean normal is the sole contracted
    coordinate; amplitude along the witnessed tangent remains free.  If both
    witnesses coincide, the target-to-witness direction supplies the tangent.
    A completely degenerate triple contributes a zero row and therefore no
    invented direction.
    """
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    source = rectangle["target_sources"]
    target = rectangle["target_targets"]
    witness = rectangle["witness_vertices"]
    minus_source, minus_target, plus_source, plus_target = witness.T
    witness_minus = np.column_stack((
        values[minus_source],
        values[minus_target] - values[minus_source],
    ))
    witness_plus = np.column_stack((
        values[plus_source],
        values[plus_target] - values[plus_source],
    ))
    current = np.column_stack((
        values[source],
        values[target] - values[source],
    ))
    tangent = witness_plus - witness_minus
    tangent_norm = np.linalg.norm(tangent, axis=1)
    degenerate = tangent_norm <= np.finfo(float).tiny
    tangent[degenerate] = (
        current[degenerate] - witness_minus[degenerate])
    tangent_norm = np.linalg.norm(tangent, axis=1)
    nonzero = tangent_norm > np.finfo(float).tiny
    normal = np.zeros_like(tangent)
    normal[nonzero, 0] = tangent[nonzero, 1] / tangent_norm[nonzero]
    normal[nonzero, 1] = -tangent[nonzero, 0] / tangent_norm[nonzero]
    minus_coordinate = np.sum(normal * witness_minus, axis=1)
    plus_coordinate = np.sum(normal * witness_plus, axis=1)
    current_coordinate = np.sum(normal * current, axis=1)
    coordinate_lower = np.minimum.reduce((
        minus_coordinate, plus_coordinate, current_coordinate))
    coordinate_upper = np.maximum.reduce((
        minus_coordinate, plus_coordinate, current_coordinate))

    row = np.repeat(np.arange(source.size, dtype=np.int64), 2)
    column = np.column_stack((source, target)).reshape(-1)
    coefficient = np.column_stack((
        normal[:, 0] - normal[:, 1],
        normal[:, 1],
    )).reshape(-1)
    operator = sparse.coo_matrix(
        (coefficient, (row, column)),
        shape=(source.size, values.size),
    ).tocsr()
    if additive_transfer:
        transfer_lower = coordinate_lower - current_coordinate
        transfer_upper = coordinate_upper - current_coordinate
    else:
        transfer_lower = current_coordinate - coordinate_upper
        transfer_upper = current_coordinate - coordinate_lower
    return operator, transfer_lower, transfer_upper, {
        "operator": operator,
        "coordinate_lower": coordinate_lower,
        "coordinate_upper": coordinate_upper,
        "current_coordinate": current_coordinate,
        "normal": normal,
        "degenerate_edge_fraction": float(np.mean(~nonzero)),
        "mean_normal_interval_width": float(np.mean(
            coordinate_upper - coordinate_lower)),
    }


def contract_joint_value_jet_scale_edge_state_2d(
    observation: np.ndarray,
    *,
    initial_posterior: np.ndarray | None = None,
    trace_refinement: int = 0,
) -> dict[str, Any]:
    """Add cross-fitted joint value/jet contraction to the local edge state."""
    image = _validate(observation)
    state = continuous_scale_edge_family_transport_state_2d(
        image,
        initial_posterior=initial_posterior,
        trace_refinement=trace_refinement,
    )
    coordinate_operator, residual_transfer_lower, residual_transfer_upper, (
        residual_witness
    ) = (
        _crossfit_value_jet_constraints(state["residual_after_erosion"]))
    posterior_operator, _posterior_reverse_lower, _posterior_reverse_upper, (
        posterior_witness
    ) = _crossfit_value_jet_constraints(state["posterior_after_erosion"])
    if (
        posterior_operator.shape != coordinate_operator.shape
        or (posterior_operator != coordinate_operator).nnz
    ):
        raise RuntimeError("posterior and residual coordinate charts diverged")
    # _crossfit_value_jet_constraints expresses subtraction from its field.
    # Posterior exchange is additive, so its admissible transfer interval has
    # the opposite sign.
    posterior_transfer_lower = (
        posterior_witness["coordinate_lower"]
        - posterior_witness["current_coordinate"]
    )
    posterior_transfer_upper = (
        posterior_witness["coordinate_upper"]
        - posterior_witness["current_coordinate"]
    )
    rectangle_transfer_lower = np.maximum(
        residual_transfer_lower, posterior_transfer_lower)
    rectangle_transfer_upper = np.minimum(
        residual_transfer_upper, posterior_transfer_upper)
    if np.any(rectangle_transfer_lower > rectangle_transfer_upper):
        raise RuntimeError("safe joint posterior/residual interval is empty")
    residual_active = residual_transfer_lower >= posterior_transfer_lower
    posterior_active = posterior_transfer_upper <= residual_transfer_upper
    witness = {
        **residual_witness,
        "residual": residual_witness,
        "posterior": posterior_witness,
        "residual_transfer_lower": residual_transfer_lower,
        "residual_transfer_upper": residual_transfer_upper,
        "posterior_transfer_lower": posterior_transfer_lower,
        "posterior_transfer_upper": posterior_transfer_upper,
        "residual_lower_active_fraction": float(np.mean(residual_active)),
        "posterior_upper_active_fraction": float(np.mean(posterior_active)),
    }
    residual_normal_operator, residual_normal_lower, residual_normal_upper, (
        residual_normal
    ) = _covariance_normal_constraints(
        state["residual_after_erosion"],
        residual_witness,
        additive_transfer=False,
    )
    posterior_normal_operator, posterior_normal_lower, posterior_normal_upper, (
        posterior_normal
    ) = _covariance_normal_constraints(
        state["posterior_after_erosion"],
        posterior_witness,
        additive_transfer=True,
    )
    covariance_operator = sparse.vstack((
        residual_normal_operator,
        posterior_normal_operator,
    ), format="csr")
    transfer_lower = np.concatenate((
        residual_normal_lower, posterior_normal_lower))
    transfer_upper = np.concatenate((
        residual_normal_upper, posterior_normal_upper))
    joint_generator = (covariance_operator @ state["generator"]).tocsc()
    coefficient_lower, coefficient_upper, contraction = (
        _contract_sparse_generator_box(
            joint_generator,
            transfer_lower,
            transfer_upper,
            initial_lower=state["coefficient_lower"],
            initial_upper=state["coefficient_upper"],
        ))
    positive = joint_generator.maximum(0.0)
    negative = joint_generator.minimum(0.0)
    support_lower_before = np.asarray(
        positive @ state["coefficient_lower"]
        + negative @ state["coefficient_upper"]
    ).ravel()
    support_upper_before = np.asarray(
        positive @ state["coefficient_upper"]
        + negative @ state["coefficient_lower"]
    ).ravel()
    intersected_lower = np.maximum(support_lower_before, transfer_lower)
    intersected_upper = np.minimum(support_upper_before, transfer_upper)
    support_width_before = np.maximum(
        support_upper_before - support_lower_before, 0.0)
    support_width_after = np.maximum(
        intersected_upper - intersected_lower, 0.0)
    positive_width = support_width_before > np.finfo(float).tiny
    support_width_ratio = np.divide(
        support_width_after,
        support_width_before,
        out=np.ones_like(support_width_after),
        where=positive_width,
    )
    full_coordinate = np.asarray(
        joint_generator @ np.ones(joint_generator.shape[1])).ravel()
    full_violation = (
        (full_coordinate < transfer_lower)
        | (full_coordinate > transfer_upper)
    )
    zero_feasible = bool(np.all(
        (transfer_lower <= 0.0) & (0.0 <= transfer_upper)))
    coefficient_center = 0.5 * (
        coefficient_lower + coefficient_upper)
    coefficient_radius = 0.5 * (
        coefficient_upper - coefficient_lower)
    generator = state["generator"]
    center_transfer = np.asarray(generator @ coefficient_center).ravel()
    radius_transfer = np.asarray(abs(generator) @ coefficient_radius).ravel()
    pushed_center, pushed_radius = _factorized_push_enclosure(
        state["edge_response"],
        state["pushed_zero_response"],
        state["representation"],
        coefficient_lower,
        coefficient_upper,
    )
    shape = image.shape
    value_width_before = state["coefficient_upper"] - state[
        "coefficient_lower"]
    value_width_after = coefficient_upper - coefficient_lower
    transfer_enclosure_lower = (
        center_transfer - radius_transfer).reshape(shape)
    transfer_enclosure_upper = (
        center_transfer + radius_transfer).reshape(shape)
    pushed_transfer_enclosure_lower = (
        pushed_center - pushed_radius).reshape(shape)
    pushed_transfer_enclosure_upper = (
        pushed_center + pushed_radius).reshape(shape)
    return {
        **state,
        "status": (
            "local scale-edge zonotope with a cross-fitted joint "
            "posterior/residual value/first-jet contractor branch"
        ),
        "theory_status": (
            "target edges are excluded from opposite-fold parallel witnesses; "
            "normal covariance is retained but statistical independence and "
            "the component-mixture action law remain unresolved"
        ),
        "coefficient_lower_before_joint": state["coefficient_lower"],
        "coefficient_upper_before_joint": state["coefficient_upper"],
        "coefficient_lower": coefficient_lower,
        "coefficient_upper": coefficient_upper,
        "transfer_enclosure_lower": transfer_enclosure_lower,
        "transfer_enclosure_upper": transfer_enclosure_upper,
        "pushed_transfer_enclosure_lower": pushed_transfer_enclosure_lower,
        "pushed_transfer_enclosure_upper": pushed_transfer_enclosure_upper,
        "branches": {
            "uncontracted_parent_identity_lineage": state["branches"][
                "identity_lineage"],
            "uncontracted_parent_positive_push_lineage": state["branches"][
                "positive_push_lineage"],
            "joint_noise_identity_lineage": {
                "posterior_base": state["posterior_after_erosion"],
                "transfer_lower": transfer_enclosure_lower,
                "transfer_upper": transfer_enclosure_upper,
                "coefficient_constraints": "constrained_zonotope",
            },
            "joint_noise_positive_push_lineage": {
                "posterior_base": state["pushed_posterior_base"],
                "transfer_lower": pushed_transfer_enclosure_lower,
                "transfer_upper": pushed_transfer_enclosure_upper,
                "coefficient_constraints": "constrained_zonotope",
            },
        },
        "branch_roles": {
            "uncontracted_parent_identity_lineage": "support explanation",
            "uncontracted_parent_positive_push_lineage": (
                "transported support explanation"),
            "joint_noise_identity_lineage": "noise-consistent stability",
            "joint_noise_positive_push_lineage": (
                "transported noise-consistent stability"),
        },
        "joint_contraction": contraction,
        "joint_witness": witness,
        "constrained_zonotope": {
            "coordinate_operator": covariance_operator,
            "constraint_matrix": joint_generator,
            "constraint_lower": transfer_lower,
            "constraint_upper": transfer_upper,
            "constraint_semantics": (
                "remaining residual and updated posterior must both remain "
                "inside target-excluded transverse value/jet covariance "
                "slabs; witnessed tangent amplitude remains unconstrained"
            ),
            "constraint_block_edge_count": int(
                residual_normal_operator.shape[0]),
            "residual_normal": residual_normal,
            "posterior_normal": posterior_normal,
            "coefficient_lower": coefficient_lower,
            "coefficient_upper": coefficient_upper,
            "zero_transfer_feasible": zero_feasible,
            "full_transfer_violation_fraction": float(np.mean(
                full_violation)),
            "mean_constraint_support_width_ratio": float(np.mean(
                support_width_ratio[positive_width]
            )) if np.any(positive_width) else 1.0,
            "median_constraint_support_width_ratio": float(np.median(
                support_width_ratio[positive_width]
            )) if np.any(positive_width) else 1.0,
            "constraint_support_width_before": support_width_before,
            "constraint_support_width_after": support_width_after,
        },
        "mean_coefficient_width_before_joint": (
            float(np.mean(value_width_before))
            if value_width_before.size else 0.0
        ),
        "mean_coefficient_width_after_joint": (
            float(np.mean(value_width_after))
            if value_width_after.size else 0.0
        ),
        "additional_contracted_coefficient_fraction": (
            float(np.mean(
                value_width_after
                < value_width_before - 64.0 * np.finfo(float).eps
            ))
            if value_width_after.size else 0.0
        ),
    }


__all__ = [
    "contract_joint_value_jet_scale_edge_state_2d",
]
