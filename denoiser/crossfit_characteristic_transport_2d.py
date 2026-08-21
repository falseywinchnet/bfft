"""Strict direction-lane cross-fitted characteristic transport in 2-D.

For each primitive tangent ``d``, choose a binary lattice coordinate that flips
under ``d``. Odd-lag source points then lie in the lane opposite the target.
Validation points lie at nonzero even multiples of ``d`` and their predictors
also use the opposite lane. Consequently the target observation is absent from
both its prediction values and their validation action.

No reflected boundary values, square diagonal surrogate, selected scale, noise
class, or validation band is used. Missing boundary characteristics simply
carry no mass.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .cross_predictive_transport import debiased_transport_coefficient


PRIMITIVE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def primitive_directions_2d(order: int = 1) -> tuple[tuple[int, int], ...]:
    """Return the complete unoriented primitive lattice tangent quadrature."""
    angular_order = int(order)
    if angular_order < 1:
        raise ValueError("angular quadrature order must be positive")
    if angular_order == 1:
        return PRIMITIVE_DIRECTIONS
    directions = []
    for dy in range(0, angular_order + 1):
        for dx in range(-angular_order, angular_order + 1):
            if dy == 0 and dx <= 0:
                continue
            if dx == 0 and dy == 0:
                continue
            if math.gcd(abs(dy), abs(dx)) != 1:
                continue
            directions.append((dy, dx))
    return tuple(directions)


def primitive_direction_weights_2d(
    directions: tuple[tuple[int, int], ...],
) -> tuple[float, ...]:
    """Return exact projective-circle Voronoi widths for tangent quadrature."""
    if not directions:
        raise ValueError("direction quadrature cannot be empty")
    angle = np.mod(np.array([
        math.atan2(dy, dx) for dy, dx in directions
    ], dtype=np.float64), math.pi)
    if np.unique(angle).size != angle.size:
        raise ValueError("directions must be distinct on the projective circle")
    order = np.argsort(angle)
    sorted_angle = angle[order]
    previous = np.roll(sorted_angle, 1)
    previous[0] -= math.pi
    following = np.roll(sorted_angle, -1)
    following[-1] += math.pi
    sorted_weight = 0.5 * (following - previous)
    weight = np.empty_like(sorted_weight)
    weight[order] = sorted_weight
    return tuple(float(value) for value in weight)


def _validate(observation: np.ndarray) -> np.ndarray:
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("cross-fitted characteristic transport expects HxW")
    if not np.all(np.isfinite(image)):
        raise ValueError("cross-fitted characteristic transport needs finite data")
    return image


def _sample(
    field: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    offset_y: int,
    offset_x: int,
) -> tuple[np.ndarray, np.ndarray]:
    sy = yy + int(offset_y)
    sx = xx + int(offset_x)
    valid = (
        (0 <= sy) & (sy < field.shape[0])
        & (0 <= sx) & (sx < field.shape[1])
    )
    value = field[
        np.clip(sy, 0, field.shape[0] - 1),
        np.clip(sx, 0, field.shape[1] - 1),
    ]
    return value, valid


def crossfit_characteristic_population_2d(
    observation: np.ndarray,
    *,
    angular_order: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return every strictly cross-fitted prediction and conductance mass."""
    image = _validate(observation)
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    prediction_fields = []
    conductance_fields = []
    directional_derivative_fields = []
    tangent_vectors = []
    action_means = []
    validation_population = []
    scale_count = 0

    directions = primitive_directions_2d(angular_order)
    direction_weights = primitive_direction_weights_2d(directions)
    for (dy, dx), direction_weight in zip(directions, direction_weights):
        direction_length = math.hypot(dy, dx)
        unit_tangent = (dy / direction_length, dx / direction_length)
        directional_extent = min(
            height - 1 if dy else np.iinfo(np.int32).max,
            width - 1 if dx else np.iinfo(np.int32).max,
        )
        maximum_lag = max(int(directional_extent) // 2, 1)
        for lag in range(1, maximum_lag + 1, 2):
            minus_one, valid_minus_one = _sample(
                image, yy, xx, -dy * lag, -dx * lag)
            plus_one, valid_plus_one = _sample(
                image, yy, xx, dy * lag, dx * lag)
            minus_three, valid_minus_three = _sample(
                image, yy, xx, -3 * dy * lag, -3 * dx * lag)
            plus_three, valid_plus_three = _sample(
                image, yy, xx, 3 * dy * lag, 3 * dx * lag)
            characteristics = (
                (
                    0.5 * (minus_one + plus_one),
                    0.5 * np.abs(plus_one - minus_one),
                    valid_minus_one & valid_plus_one,
                    (plus_one - minus_one) / (2.0 * lag * direction_length),
                ),
                (
                    1.5 * minus_one - 0.5 * minus_three,
                    0.5 * np.abs(minus_one - minus_three),
                    valid_minus_one & valid_minus_three,
                    (minus_one - minus_three) / (2.0 * lag * direction_length),
                ),
                (
                    1.5 * plus_one - 0.5 * plus_three,
                    0.5 * np.abs(plus_one - plus_three),
                    valid_plus_one & valid_plus_three,
                    (plus_three - plus_one) / (2.0 * lag * direction_length),
                ),
            )
            for (
                prediction, variation, prediction_valid,
                directional_derivative,
            ) in characteristics:
                error = np.abs(prediction - image)
                action_sum = np.zeros_like(image)
                action_count = np.zeros_like(image)
                for path_index in range(1, lag + 1):
                    path_offset = 2 * path_index
                    for sign in (-1, 1):
                        sampled_error, position_valid = _sample(
                            error,
                            yy,
                            xx,
                            sign * dy * path_offset,
                            sign * dx * path_offset,
                        )
                        sampled_predictor_valid, _unused = _sample(
                            prediction_valid,
                            yy,
                            xx,
                            sign * dy * path_offset,
                            sign * dx * path_offset,
                        )
                        valid = position_valid & sampled_predictor_valid
                        action_sum += np.where(valid, sampled_error, 0.0)
                        action_count += valid
                valid_action = prediction_valid & (action_count > 0.0)
                action = np.divide(
                    action_sum,
                    action_count,
                    out=np.zeros_like(action_sum),
                    where=action_count > 0.0,
                )
                conductance = np.where(
                    valid_action,
                    direction_weight
                    / (lag * np.maximum(action + variation, floor)),
                    0.0,
                )
                prediction_fields.append(prediction)
                conductance_fields.append(conductance)
                directional_derivative_fields.append(directional_derivative)
                tangent_vectors.append(unit_tangent)
                if np.any(valid_action):
                    action_means.append(float(np.mean(action[valid_action])))
                    validation_population.append(float(np.mean(
                        action_count[valid_action])))
            scale_count += 1

    predictions = np.stack(prediction_fields, axis=-1)
    conductance = np.stack(conductance_fields, axis=-1)
    total = np.sum(conductance, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise RuntimeError("cross-fitted population left a point unsupported")
    mass = conductance / total
    return {
        "prediction": predictions,
        "mass": mass,
        "directional_derivative": np.stack(
            directional_derivative_fields, axis=-1),
        "tangent": np.asarray(tangent_vectors, dtype=np.float64),
    }, {
        "status": "strict direction-lane cross-fitted population",
        "directions": [list(direction) for direction in directions],
        "angular_quadrature_order": int(angular_order),
        "direction_count": len(directions),
        "direction_quadrature_weight": list(direction_weights),
        "direction_quadrature_mass": float(sum(direction_weights)),
        "scale_measure": "all admissible odd lags under ds/s conductance",
        "scale_count": scale_count,
        "characteristic_count": int(predictions.shape[-1]),
        "target_identity_excluded": True,
        "boundary_condition": "no reflection; invalid characteristics have zero mass",
        "predictive_action_mean": float(np.mean(action_means)),
        "validation_population_mean": float(np.mean(validation_population)),
    }


def _weighted_median(prediction: np.ndarray, mass: np.ndarray) -> np.ndarray:
    order = np.argsort(prediction, axis=-1, kind="stable")
    values = np.take_along_axis(prediction, order, axis=-1)
    weights = np.take_along_axis(mass, order, axis=-1)
    index = np.argmax(np.cumsum(weights, axis=-1) >= 0.5, axis=-1)
    return np.take_along_axis(values, index[..., None], axis=-1)[..., 0]


def crossfit_characteristic_measure_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "mean",
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the cross-fitted law by its W2 mean or W1 median barycenter."""
    population, diagnostic = crossfit_characteristic_population_2d(
        observation, angular_order=angular_order)
    if barycenter == "mean":
        readout = np.sum(
            population["mass"] * population["prediction"], axis=-1)
    elif barycenter == "median":
        readout = _weighted_median(
            population["prediction"], population["mass"])
    else:
        raise ValueError("barycenter must be 'mean' or 'median'")
    return readout, {
        **diagnostic,
        "barycenter": barycenter,
    }


def denoise_crossfit_characteristic_transport_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "median",
    maximum_continuations: int = 32,
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore only globally cross-predictable residual mass."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial = crossfit_characteristic_measure_2d(
        image, barycenter=barycenter, angular_order=angular_order)
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        prediction, measure = crossfit_characteristic_measure_2d(
            residual, barycenter="mean", angular_order=angular_order)
        transport = debiased_transport_coefficient(
            residual.ravel(), prediction.ravel())
        coefficient = transport["coefficient"]
        if coefficient == 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                **transport,
            })
            break
        candidate = np.clip(state + coefficient * prediction, 0.0, 1.0)
        candidate_action = float(np.mean((image - candidate) ** 2))
        if candidate_action >= action - np.finfo(float).eps * max(action, 1.0):
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                **transport,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            **transport,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return state, {
        "status": (
            "strict cross-fitted covariance equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "theory_status": "strict 2-D cross-fitting experiment; not promoted",
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "unresolved": [
            "global covariance still treats spatial validators as independent",
            "four primitive tangents are a crystalline quadrature",
            "odd-lag boundary population needs refinement convergence",
        ],
    }
