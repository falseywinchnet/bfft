"""Transport characteristic branch probability without smoothing amplitude."""

from __future__ import annotations

from typing import Any

import numpy as np

from .continuous_source_transport import source_measure_operator
from .continuous_tangent_transport_2d import (
    continuous_tangent_joint_population_2d,
)
from .crossfit_characteristic_transport_2d import (
    crossfit_characteristic_population_2d,
)
from .fused_transport_geometry import (
    predictive_horizontal_wasserstein_geometry,
    weighted_empirical_quantiles,
)
from .witnessed_characteristic_transport_2d import (
    _crps_against_witness,
    _validate,
)


def _backtransport_branch_action(
    action: np.ndarray,
    scale_conductance: np.ndarray,
    source_identity: np.ndarray,
    source_coefficient: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pull each branch action through the affine chart that generated it.

    Absolute affine coefficients transport nonnegative action; their signs
    remain part of amplitude reconstruction, not action mass.  A source can
    contribute only where the same branch exists.  This is a chart-incidence
    operation, not a spatial window or a direction catalogue added afterward.
    """
    local = np.asarray(action, dtype=np.float64)
    conductance = np.asarray(scale_conductance, dtype=np.float64)
    identity = np.asarray(source_identity, dtype=np.int64)
    coefficient = np.asarray(source_coefficient, dtype=np.float64)
    if (
        local.ndim != 3
        or conductance.shape != local.shape
        or identity.shape[:3] != local.shape
        or coefficient.shape != identity.shape
    ):
        raise ValueError("branch action and affine source charts must align")
    pixels = local.shape[0] * local.shape[1]
    branches = local.shape[2]
    flat_action = local.reshape(pixels, branches)
    flat_valid = conductance.reshape(pixels, branches) > 0.0
    flat_identity = identity.reshape(pixels, branches, -1)
    flat_coefficient = coefficient.reshape(pixels, branches, -1)
    branch = np.broadcast_to(
        np.arange(branches, dtype=np.int64)[None, :, None],
        flat_identity.shape,
    )
    source_action = flat_action[flat_identity, branch]
    source_valid = flat_valid[flat_identity, branch]
    transported_weight = np.abs(flat_coefficient) * source_valid
    total = np.sum(transported_weight, axis=-1)
    transported = np.divide(
        np.sum(transported_weight * source_action, axis=-1),
        total,
        out=flat_action.copy(),
        where=total > 0.0,
    )
    return transported.reshape(local.shape), {
        "represented_chart_fraction": float(np.mean(total > 0.0)),
        "mean_chart_action": float(np.mean(transported)),
        "maximum_chart_action": float(np.max(transported)),
        "transport_law": (
            "absolute affine-incidence pullback on the identical branch"
        ),
    }


def denoise_causal_branch_action_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select amplitude only after branch action traverses its source chart."""
    image = _validate(observation)
    population, population_diagnostic = continuous_tangent_joint_population_2d(
        image, angular_count=angular_count)
    local_action = np.asarray(population["joint_action"], dtype=np.float64)
    source_action, source_diagnostic = _backtransport_branch_action(
        local_action,
        population["scale_conductance"],
        population["source_identity"],
        population["source_coefficient"],
    )
    path_action = local_action + source_action
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else np.sqrt(np.finfo(float).eps) * magnitude
    )
    conductance = np.asarray(
        population["scale_conductance"], dtype=np.float64)
    path_mass = conductance / np.maximum(path_action, floor)
    path_mass /= np.sum(path_mass, axis=-1, keepdims=True)
    branch = np.argmax(path_mass, axis=-1)
    estimate = np.take_along_axis(
        population["signal"], branch[..., None], axis=-1)[..., 0]
    lower = float(np.min(image))
    upper = float(np.max(image))
    local_branch = np.argmax(population["mass"], axis=-1)
    return np.clip(estimate, lower, upper), {
        "status": "one-step causal characteristic-branch action",
        "theory_status": (
            "local joint action plus exact affine-chart ancestral action"
        ),
        "angular_count": int(angular_count),
        "population": population_diagnostic,
        "source_action": source_diagnostic,
        "selected_branch_count": int(np.unique(branch).size),
        "branch_change_fraction": float(np.mean(branch != local_branch)),
        "mean_path_action": float(np.mean(path_action)),
        "maximum_target_self_coefficient": float(np.max(np.abs(np.where(
            population["source_identity"]
            == np.arange(image.size).reshape(image.shape + (1, 1)),
            population["source_coefficient"],
            0.0,
        )))),
        "unresolved": [
            "single ancestral pullback is a first characteristic-step probe",
            "absolute affine action transport needs a continuum chart derivation",
            "hard branch readout remains discontinuous",
        ],
    }


def _row_half_pair_action(
    value: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    """One half expected absolute pair distance for aligned row laws."""
    order = np.argsort(value, axis=-1, kind="stable")
    ordered_value = np.take_along_axis(value, order, axis=-1)
    ordered_mass = np.take_along_axis(mass, order, axis=-1)
    preceding_mass = np.cumsum(ordered_mass, axis=-1) - ordered_mass
    preceding_moment = (
        np.cumsum(ordered_mass * ordered_value, axis=-1)
        - ordered_mass * ordered_value
    )
    return np.sum(
        ordered_mass
        * (ordered_value * preceding_mass - preceding_moment),
        axis=-1,
    )


def _branch_energy_score(
    probability: np.ndarray,
    prediction: np.ndarray,
    point_crps: np.ndarray,
    witness_half_pair: np.ndarray,
) -> float:
    """Proper scalar energy score of a branch mixture against its witness."""
    mass = np.asarray(probability, dtype=np.float64)
    cross = np.sum(mass * point_crps, axis=-1) + witness_half_pair
    self_action = _row_half_pair_action(prediction, mass)
    return float(np.mean(np.maximum(cross - self_action, 0.0)))


def denoise_branch_posterior_transport_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
    quantile_count: int = 32,
    maximum_transports: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move branch probability to proper-score equilibrium, then read one branch."""
    image = _validate(observation)
    ceiling = int(maximum_transports)
    if ceiling < 1:
        raise ValueError("maximum branch transports must be positive")
    population, population_diagnostic = continuous_tangent_joint_population_2d(
        image, angular_count=angular_count)
    prediction = np.asarray(population["signal"], dtype=np.float64)
    state = np.asarray(population["mass"], dtype=np.float64).reshape(
        image.size, -1)
    signal_prior = np.asarray(population["prior_mass"], dtype=np.float64)
    quantiles = weighted_empirical_quantiles(
        prediction, signal_prior, quantile_count)
    horizontal = predictive_horizontal_wasserstein_geometry(quantiles)
    operator, operator_diagnostic = source_measure_operator(
        horizontal["metric_xx"],
        horizontal["metric_xy"],
        horizontal["metric_yy"],
    )
    witness, witness_diagnostic = crossfit_characteristic_population_2d(image)
    point_crps, witness_half_pair = _crps_against_witness(
        prediction, witness["prediction"], witness["mass"])
    flat_prediction = prediction.reshape(image.size, -1)
    flat_point_crps = point_crps.reshape(image.size, -1)
    flat_witness_half_pair = witness_half_pair.ravel()

    def risk(branch_mass: np.ndarray) -> float:
        return _branch_energy_score(
            branch_mass,
            flat_prediction,
            flat_point_crps,
            flat_witness_half_pair,
        )

    action = risk(state)
    records = []
    equilibrium = False
    numerical = np.finfo(float).eps * max(action, 1.0)
    for transport in range(ceiling):
        destination = np.asarray(operator @ state, dtype=np.float64)
        destination /= np.sum(destination, axis=1, keepdims=True)
        midpoint = 0.5 * (state + destination)
        endpoint_action = risk(destination)
        midpoint_action = risk(midpoint)
        curvature = max(
            2.0 * (endpoint_action - 2.0 * midpoint_action + action),
            0.0,
        )
        linear = endpoint_action - action - curvature
        step = (
            float(np.clip(-linear / (2.0 * curvature), 0.0, 1.0))
            if curvature > np.finfo(float).tiny
            else (1.0 if linear < 0.0 else 0.0)
        )
        candidate = (1.0 - step) * state + step * destination
        candidate_action = risk(candidate)
        accepted = candidate_action < action - numerical
        records.append({
            "transport": transport,
            "accepted": accepted,
            "proper_score_before": action,
            "proper_score_after": candidate_action,
            "endpoint_proper_score": endpoint_action,
            "analytic_transport_step": step if accepted else 0.0,
            "quadratic_curvature": curvature,
            "mean_branch_collision_population": float(np.mean(
                1.0 / np.sum(candidate * candidate, axis=1))),
        })
        if not accepted:
            equilibrium = True
            break
        state = candidate
        action = candidate_action

    branch = np.argmax(state, axis=1)
    estimate = flat_prediction[np.arange(image.size), branch].reshape(image.shape)
    lower = float(np.min(image))
    upper = float(np.max(image))
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(estimate, lower, upper), {
        "status": (
            "branch probability at proper-score transport equilibrium"
            if equilibrium else "branch transport ceiling reached; unresolved"
        ),
        "theory_status": (
            "amplitude-frozen characteristic branch transport; foundational gate"
        ),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "accepted_branch_transports": int(sum(
            record["accepted"] for record in records)),
        "branch_transport_ceiling_hit": ceiling_hit,
        "terminal_proper_score": action,
        "branch_transports": records,
        "population": population_diagnostic,
        "witness": witness_diagnostic,
        "horizontal_geometry": horizontal,
        "source_operator": operator_diagnostic,
        "selected_branch_count": int(np.unique(branch).size),
        "unresolved": [
            "branch index uses the common angular/radial quadrature chart",
            "hard maximum-probability readout is not yet a continuous branch section",
            "dense branch population is a research representation",
        ],
    }
