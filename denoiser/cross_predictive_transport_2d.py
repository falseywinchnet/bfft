"""Minimal 2-D lift of full-scale characteristic transport.

This module is a falsification seed, not the fused image denoiser. It lifts the
successful 1-D scale state onto four primitive lattice tangent families and
retains all admissible lags and three first-jet characteristics. Its ordinary
conductance barycenter and image-global covariance continuation are measurable
precursors to the required V3 causal-particle law.

The expected failure is explicit: sparse interfaces occupy too little global
mass, while the four-direction quadrature is crystalline. A valid successor
must obtain effective population from causal ancestry and use horizontal
Wasserstein transport under a continuous eikonal metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import numpy as np
from scipy import ndimage

from .cross_predictive_transport import debiased_transport_coefficient


PRIMITIVE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class CrossPredictive2DResolution:
    """Numerical failure guard for the deliberately minimal image lift."""

    maximum_continuations: int = 32


def _validate_image(observation: np.ndarray) -> np.ndarray:
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("2-D characteristic transport expects an HxW field")
    if not np.all(np.isfinite(image)):
        raise ValueError("2-D characteristic transport requires finite samples")
    return image


def _floor(image: np.ndarray) -> float:
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    return (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )


def relation_characteristic_measure_2d(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize four directional full-scale first-jet populations."""
    prediction, diagnostic, _direction_mass, _population = _relation_characteristics_2d(
        observation)
    return prediction, diagnostic


def _relation_characteristics_2d(
    observation: np.ndarray,
    *,
    retain_population: bool = False,
    exclude_target_action: bool = False,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, np.ndarray] | None]:
    """Return the readout and its unnormalized directional transport mass."""
    image = _validate_image(observation)
    height, width = image.shape
    if float(np.ptp(image)) == 0.0:
        characteristic_count = 3 * (
            width // 2 + height // 2 + 2 * (min(height, width) // 2))
        return image.copy(), {
            "directions": [list(value) for value in PRIMITIVE_DIRECTIONS],
            "direction_count": 4,
            "characteristics_per_scale": 3,
            "characteristic_count": characteristic_count,
            "predictive_action_mean": 0.0,
            "path_variation_mean": 0.0,
        }, np.ones((4, height, width), dtype=np.float64), (
            {
                "prediction": image[..., None].copy(),
                "mass": np.ones(image.shape + (1,), dtype=np.float64),
            }
            if retain_population else None
        )

    maximum_lag = max(height, width) // 2
    padded = np.pad(image, 2 * maximum_lag, mode="reflect")
    yy = np.arange(height)[:, None] + 2 * maximum_lag
    xx = np.arange(width)[None, :] + 2 * maximum_lag
    floor = _floor(image)
    numerator = np.zeros_like(image)
    denominator = np.zeros_like(image)
    direction_mass = np.zeros((4, height, width), dtype=np.float64)
    action_means = []
    variation_means = []
    characteristic_count = 0
    population_prediction = []
    population_conductance = []

    for direction_index, (dy, dx) in enumerate(PRIMITIVE_DIRECTIONS):
        direction_lag = min(
            height // 2 if dy else np.iinfo(np.int32).max,
            width // 2 if dx else np.iinfo(np.int32).max,
        )
        for lag in range(1, int(direction_lag) + 1):
            left_one = padded[yy - dy * lag, xx - dx * lag]
            right_one = padded[yy + dy * lag, xx + dx * lag]
            left_two = padded[yy - 2 * dy * lag, xx - 2 * dx * lag]
            right_two = padded[yy + 2 * dy * lag, xx + 2 * dx * lag]
            characteristics = (
                (0.5 * (left_one + right_one),
                 0.5 * np.abs(right_one - left_one)),
                (2.0 * left_one - left_two,
                 np.abs(left_one - left_two)),
                (2.0 * right_one - right_two,
                 np.abs(right_one - right_two)),
            )
            for characteristic_index, (prediction, variation) in enumerate(
                characteristics
            ):
                error = np.abs(prediction - image)
                if dy == 0:
                    action = ndimage.uniform_filter1d(
                        error, 2 * lag + 1, axis=1, mode="reflect")
                    action_population = 2 * lag + 1
                elif dx == 0:
                    action = ndimage.uniform_filter1d(
                        error, 2 * lag + 1, axis=0, mode="reflect")
                    action_population = 2 * lag + 1
                else:
                    # This square surrogate is deliberately exposed as the
                    # crystalline seed's central defect.
                    action = ndimage.uniform_filter(
                        error, 2 * lag + 1, mode="reflect")
                    action_population = (2 * lag + 1) ** 2
                if exclude_target_action:
                    # Remove every validation error whose predictor reads the
                    # target source identity. Interpolation reads it at the two
                    # opposite lag endpoints; the one-sided jets read it at
                    # the corresponding opposite endpoint. The central error
                    # itself is always excluded. At axial lag one, interpolation
                    # has no independent validator and therefore carries zero
                    # conductance rather than borrowing target evidence.
                    error_pad = np.pad(error, lag, mode="symmetric")
                    ey = np.arange(height)[:, None] + lag
                    ex = np.arange(width)[None, :] + lag
                    if characteristic_index == 0:
                        offsets = ((0, 0), (-dy * lag, -dx * lag),
                                   (dy * lag, dx * lag))
                    elif characteristic_index == 1:
                        offsets = ((0, 0), (dy * lag, dx * lag))
                    else:
                        offsets = ((0, 0), (-dy * lag, -dx * lag))
                    independent = action_population - len(offsets)
                    if independent <= 0:
                        action = np.full_like(action, np.inf)
                    else:
                        excluded = np.zeros_like(error)
                        for offset_y, offset_x in offsets:
                            excluded += error_pad[
                                ey + offset_y, ex + offset_x]
                        action = np.maximum(
                            (action_population * action - excluded)
                            / independent,
                            0.0,
                        )
                conductance = 1.0 / (
                    lag * np.maximum(action + variation, floor))
                numerator += conductance * prediction
                denominator += conductance
                direction_mass[direction_index] += conductance
                if retain_population:
                    population_prediction.append(prediction)
                    population_conductance.append(conductance)
                action_means.append(float(np.mean(action)))
                variation_means.append(float(np.mean(variation)))
                characteristic_count += 1

    population = None
    if retain_population:
        prediction_field = np.stack(population_prediction, axis=-1)
        conductance_field = np.stack(population_conductance, axis=-1)
        population = {
            "prediction": prediction_field,
            "mass": conductance_field / denominator[..., None],
        }
    return numerator / denominator, {
        "directions": [list(value) for value in PRIMITIVE_DIRECTIONS],
        "direction_count": 4,
        "characteristics_per_scale": 3,
        "characteristic_count": characteristic_count,
        "predictive_action_mean": float(np.mean(action_means)),
        "path_variation_mean": float(np.mean(variation_means)),
        "scale_measure": "complete directionwise ds/s",
        "known_defect": "diagonal action uses square-footprint surrogate",
        "target_validation_excluded": bool(exclude_target_action),
        "target_identity_exclusion": (
            "central error and every in-path validator whose predictor reads "
            "the target"
            if exclude_target_action else "not enforced"
        ),
    }, direction_mass, population


def relation_characteristic_population_2d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the full weighted prediction law before barycentric collapse."""
    _readout, diagnostic, _direction_mass, population = (
        _relation_characteristics_2d(observation, retain_population=True))
    assert population is not None
    return population, {
        **diagnostic,
        "population_representation": (
            "all direction/scale/first-jet predictions with normalized "
            "conductance mass"
        ),
    }


def heldout_relation_characteristic_measure_2d(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read out the full-scale law with target prediction error excluded."""
    prediction, diagnostic, _direction_mass, _population = (
        _relation_characteristics_2d(
            observation, exclude_target_action=True))
    return prediction, diagnostic


def heldout_relation_characteristic_population_2d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the strictly held-out weighted characteristic population."""
    _readout, diagnostic, _direction_mass, population = (
        _relation_characteristics_2d(
            observation,
            retain_population=True,
            exclude_target_action=True,
        ))
    assert population is not None
    return population, {
        **diagnostic,
        "population_representation": (
            "all direction/scale/first-jet predictions; target excluded from "
            "both value and validation action"
        ),
    }


def relation_transport_metric_2d(observation: np.ndarray) -> dict[str, Any]:
    """Derive a determinant-one path metric from relation conductance.

    Directional conductance forms the tangent covariance

        C = sum_d g_d dhat_d dhat_d^T / sum_d g_d.

    The eikonal metric is ``M = sqrt(det(C)) inv(C)``. Predictable tangents are
    therefore cheap directions, while ``det(M)=1`` prevents an empirical
    strength parameter from changing path volume. This is the crystalline
    four-direction precursor to a continuous tangent-sphere integral.
    """
    return _relation_transport_metric_2d(observation, exclude_target_action=False)


def heldout_relation_transport_metric_2d(
    observation: np.ndarray,
) -> dict[str, Any]:
    """Derive the metric from strictly held-out characteristic conductance."""
    return _relation_transport_metric_2d(observation, exclude_target_action=True)


def _relation_transport_metric_2d(
    observation: np.ndarray,
    *,
    exclude_target_action: bool,
) -> dict[str, Any]:
    image = _validate_image(observation)
    _prediction, measure, direction_mass, _population = (
        _relation_characteristics_2d(
            image, exclude_target_action=exclude_target_action))
    covariance_xx = np.zeros_like(image)
    covariance_xy = np.zeros_like(image)
    covariance_yy = np.zeros_like(image)
    total = np.sum(direction_mass, axis=0)
    for mass, (dy, dx) in zip(direction_mass, PRIMITIVE_DIRECTIONS):
        norm = math.hypot(dx, dy)
        ux = dx / norm
        uy = dy / norm
        covariance_xx += mass * ux * ux
        covariance_xy += mass * ux * uy
        covariance_yy += mass * uy * uy
    covariance_xx /= total
    covariance_xy /= total
    covariance_yy /= total
    determinant = np.maximum(
        covariance_xx * covariance_yy - covariance_xy * covariance_xy,
        np.finfo(float).tiny,
    )
    root_determinant = np.sqrt(determinant)
    metric_xx = covariance_yy / root_determinant
    metric_xy = -covariance_xy / root_determinant
    metric_yy = covariance_xx / root_determinant
    metric_determinant = metric_xx * metric_yy - metric_xy * metric_xy
    return {
        "metric_xx": np.ascontiguousarray(metric_xx),
        "metric_xy": np.ascontiguousarray(metric_xy),
        "metric_yy": np.ascontiguousarray(metric_yy),
        "metric_determinant": np.ascontiguousarray(metric_determinant),
        "tangent_covariance_xx": np.ascontiguousarray(covariance_xx),
        "tangent_covariance_xy": np.ascontiguousarray(covariance_xy),
        "tangent_covariance_yy": np.ascontiguousarray(covariance_yy),
        "direction_mass": np.ascontiguousarray(direction_mass),
        "characteristic_measure": measure,
        "geometry": (
            "M=sqrt(det(C))*inv(C), C is full-scale relation tangent mass"
        ),
        "theory_status": (
            "determinant-normalized crystalline precursor; tangent-sphere "
            "convergence pending"
        ),
        "target_validation_excluded": bool(exclude_target_action),
    }


def denoise_cross_predictive_transport_2d(
    observation: np.ndarray,
    resolution: CrossPredictive2DResolution = CrossPredictive2DResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Advance the minimal image lift to image-global covariance equilibrium."""
    image = _validate_image(observation)
    ceiling = int(resolution.maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial_measure = relation_characteristic_measure_2d(image)
    action = float(np.mean((image - state) ** 2))
    records = []
    ceiling_hit = True
    for continuation in range(ceiling):
        residual = image - state
        prediction, measure = relation_characteristic_measure_2d(residual)
        transport = debiased_transport_coefficient(residual, prediction)
        coefficient = transport["coefficient"]
        if coefficient == 0.0:
            ceiling_hit = False
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                **transport,
                "measure": measure,
            })
            break
        candidate = state + coefficient * prediction
        candidate_action = float(np.mean((image - candidate) ** 2))
        if candidate_action > action + np.finfo(float).eps * max(action, 1.0):
            raise RuntimeError("2-D characteristic continuation violated descent")
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            **transport,
            "measure": measure,
        })
        state = candidate
        action = candidate_action

    return np.clip(state, 0.0, 1.0), {
        "status": (
            "numerical continuation ceiling reached; seed rejected"
            if ceiling_hit
            else "minimal four-direction covariance equilibrium"
        ),
        "theory_status": "2-D falsification seed; not promoted",
        "initial_measure": initial_measure,
        "accepted_continuations": int(sum(
            record["accepted"] for record in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "numerical_resolution": asdict(resolution),
        "unresolved": [
            "four-direction crystalline tangent quadrature",
            "diagonal square-footprint action surrogate",
            "image-global covariance dilutes sparse support",
            "effective population is not derived from causal ancestry",
        ],
    }
