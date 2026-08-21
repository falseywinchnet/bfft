"""Dense jet proposals judged by a strictly disjoint transport witness.

The proposal law contains every admissible first-jet characteristic over the
domain's complete ``ds/s`` scale measure.  Its values and intrinsic variation
never read the target observation.  A direction-lane cross-fitted law then
assigns each proposal its continuous ranked probability score (CRPS).  Thus
validation is a proper transport score against an independent empirical law,
not a noise class, acceptance band, or fitted likelihood width.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .crossfit_characteristic_transport_2d import (
    crossfit_characteristic_population_2d,
    primitive_direction_weights_2d,
    primitive_directions_2d,
)
from .cross_predictive_transport import debiased_transport_coefficient
from .cross_predictive_transport_2d import relation_transport_metric_2d
from .continuous_source_transport import (
    _exclude_target_identity,
    source_measure_operator,
)


def _validate(observation: np.ndarray) -> np.ndarray:
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("witnessed characteristic transport expects HxW")
    if not np.all(np.isfinite(image)):
        raise ValueError("witnessed characteristic transport needs finite data")
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


def dense_characteristic_proposals_2d(
    observation: np.ndarray,
    *,
    angular_order: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return target-independent dense first-jet proposals and base mass."""
    image = _validate(observation)
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    predictions = []
    variation_fields = []
    scale_conductance = []
    source_identity = []
    source_coefficient = []
    lags = []
    directions = primitive_directions_2d(angular_order)
    direction_weights = primitive_direction_weights_2d(directions)
    for (dy, dx), direction_weight in zip(directions, direction_weights):
        extent = min(
            height - 1 if dy else np.iinfo(np.int32).max,
            width - 1 if dx else np.iinfo(np.int32).max,
        )
        for lag in range(1, max(int(extent) // 2, 1) + 1):
            minus_one, valid_minus_one = _sample(
                image, yy, xx, -dy * lag, -dx * lag)
            plus_one, valid_plus_one = _sample(
                image, yy, xx, dy * lag, dx * lag)
            minus_two, valid_minus_two = _sample(
                image, yy, xx, -2 * dy * lag, -2 * dx * lag)
            plus_two, valid_plus_two = _sample(
                image, yy, xx, 2 * dy * lag, 2 * dx * lag)
            characteristics = (
                (
                    0.5 * (minus_one + plus_one),
                    0.5 * np.abs(plus_one - minus_one),
                    valid_minus_one & valid_plus_one,
                    ((-dy * lag, -dx * lag), (dy * lag, dx * lag)),
                    (0.5, 0.5),
                ),
                (
                    2.0 * minus_one - minus_two,
                    np.abs(minus_one - minus_two),
                    valid_minus_one & valid_minus_two,
                    ((-dy * lag, -dx * lag), (-2 * dy * lag, -2 * dx * lag)),
                    (2.0, -1.0),
                ),
                (
                    2.0 * plus_one - plus_two,
                    np.abs(plus_one - plus_two),
                    valid_plus_one & valid_plus_two,
                    ((dy * lag, dx * lag), (2 * dy * lag, 2 * dx * lag)),
                    (2.0, -1.0),
                ),
            )
            for prediction, variation, valid, offsets, coefficients in characteristics:
                predictions.append(prediction)
                variation_fields.append(variation)
                scale_conductance.append(np.where(
                    valid, direction_weight / lag, 0.0))
                indices = []
                for offset_y, offset_x in offsets:
                    sy = np.clip(yy + offset_y, 0, height - 1)
                    sx = np.clip(xx + offset_x, 0, width - 1)
                    indices.append((sy * width + sx).astype(np.int64))
                source_identity.append(np.stack(indices, axis=-1))
                source_coefficient.append(np.broadcast_to(
                    np.asarray(coefficients, dtype=np.float64),
                    image.shape + (2,),
                ))
                lags.append(lag)
    prediction = np.stack(predictions, axis=-1)
    variation = np.stack(variation_fields, axis=-1)
    conductance = np.stack(scale_conductance, axis=-1)
    if np.any(np.sum(conductance, axis=-1) <= 0.0):
        raise RuntimeError("dense proposal law left a point unsupported")
    return {
        "prediction": prediction,
        "variation": variation,
        "scale_conductance": conductance,
        "source_identity": np.stack(source_identity, axis=-2),
        "source_coefficient": np.stack(source_coefficient, axis=-2),
    }, {
        "proposal_count": int(prediction.shape[-1]),
        "directions": [list(direction) for direction in directions],
        "angular_quadrature_order": int(angular_order),
        "direction_count": len(directions),
        "direction_quadrature_weight": list(direction_weights),
        "direction_quadrature_mass": float(sum(direction_weights)),
        "scale_measure": "every admissible lag with ds/s base mass",
        "target_identity_excluded": True,
        "boundary_condition": "no reflection; invalid proposals have zero mass",
        "lag_minimum": int(min(lags)),
        "lag_maximum": int(max(lags)),
    }


def _crps_against_witness(
    proposal: np.ndarray,
    witness_value: np.ndarray,
    witness_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CRPS(Q, z) and Q's half pairwise absolute transport action."""
    order = np.argsort(witness_value, axis=-1, kind="stable")
    value = np.take_along_axis(witness_value, order, axis=-1)
    mass = np.take_along_axis(witness_mass, order, axis=-1)
    preceding_mass = np.cumsum(mass, axis=-1) - mass
    preceding_moment = np.cumsum(mass * value, axis=-1) - mass * value
    self_action = np.sum(
        mass * (value * preceding_mass - preceding_moment), axis=-1)

    expected_absolute = np.empty_like(proposal)
    # Stream proposal particles so the implementation scales in memory as
    # O(H W K_witness), rather than materializing their Cartesian product.
    for index in range(proposal.shape[-1]):
        expected_absolute[..., index] = np.sum(
            witness_mass * np.abs(
                witness_value - proposal[..., index, None]),
            axis=-1,
        )
    return np.maximum(
        expected_absolute - self_action[..., None], 0.0), self_action


def _weighted_median(prediction: np.ndarray, mass: np.ndarray) -> np.ndarray:
    order = np.argsort(prediction, axis=-1, kind="stable")
    values = np.take_along_axis(prediction, order, axis=-1)
    weights = np.take_along_axis(mass, order, axis=-1)
    index = np.argmax(np.cumsum(weights, axis=-1) >= 0.5, axis=-1)
    return np.take_along_axis(values, index[..., None], axis=-1)[..., 0]


def witnessed_characteristic_population_2d(
    observation: np.ndarray,
    *,
    angular_order: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score dense proposals only against the disjoint predictive witness."""
    image = _validate(observation)
    proposal, proposal_diagnostic = dense_characteristic_proposals_2d(
        image, angular_order=angular_order)
    witness, witness_diagnostic = crossfit_characteristic_population_2d(
        image, angular_order=angular_order)
    score, self_action = _crps_against_witness(
        proposal["prediction"], witness["prediction"], witness["mass"])
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    conductance = proposal["scale_conductance"] / np.maximum(
        proposal["variation"] + score, floor)
    total = np.sum(conductance, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise RuntimeError("witness score removed the complete proposal law")
    return {
        "prediction": proposal["prediction"],
        "mass": conductance / total,
        "crps": score,
        "variation": proposal["variation"],
        "scale_conductance": proposal["scale_conductance"],
        "source_identity": proposal["source_identity"],
        "source_coefficient": proposal["source_coefficient"],
    }, {
        "status": "dense jet law under strict cross-fitted CRPS conductance",
        "proposal": proposal_diagnostic,
        "witness": witness_diagnostic,
        "transport_score": (
            "CRPS(Q,z)=E_Q|Z-z|-0.5 E_Q|Z-Z'|"
        ),
        "mean_witness_self_action": float(np.mean(self_action)),
        "mean_proposal_crps": float(np.mean(score)),
        "target_identity_excluded": True,
        "theory_status": "proper-score witness experiment; not promoted",
    }


def witnessed_characteristic_measure_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "mean",
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the witnessed proposal law by its W2 or W1 barycenter."""
    population, diagnostic = witnessed_characteristic_population_2d(
        observation, angular_order=angular_order)
    if barycenter == "mean":
        readout = np.sum(
            population["mass"] * population["prediction"], axis=-1)
    elif barycenter == "median":
        readout = _weighted_median(
            population["prediction"], population["mass"])
    else:
        raise ValueError("barycenter must be 'mean' or 'median'")
    return readout, {**diagnostic, "barycenter": barycenter}


def _weighted_half_pair_action(value: np.ndarray, mass: np.ndarray) -> float:
    order = np.argsort(value, kind="stable")
    value = value[order]
    mass = mass[order]
    preceding_mass = np.cumsum(mass) - mass
    preceding_moment = np.cumsum(mass * value) - mass * value
    return float(np.sum(
        mass * (value * preceding_mass - preceding_moment)))


def _sorted_expected_absolute(
    value: np.ndarray,
    mass: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Evaluate E|V-query| for one sorted finite empirical measure."""
    order = np.argsort(value, kind="stable")
    value = value[order]
    mass = mass[order]
    cumulative_mass = np.cumsum(mass)
    cumulative_moment = np.cumsum(mass * value)
    index = np.searchsorted(value, query, side="right") - 1
    clipped = np.clip(index, 0, value.size - 1)
    left_mass = np.where(index >= 0, cumulative_mass[clipped], 0.0)
    left_moment = np.where(index >= 0, cumulative_moment[clipped], 0.0)
    total_moment = cumulative_moment[-1]
    return (
        query * left_mass - left_moment
        + (total_moment - left_moment) - query * (1.0 - left_mass)
    )


def _four_colour_leave_one_out_residual_crps(
    image: np.ndarray,
    witness: dict[str, np.ndarray],
    residual_query: np.ndarray,
) -> np.ndarray:
    """Score residual queries against an exactly target-independent prior.

    Each source contributes its unweighted valid odd-characteristic residual
    law.  Only sources in the same four-colour lattice lane are pooled.  A
    target can enter neither another source's odd-lag prediction nor its static
    validity mask in that lane, and its complete local law is removed before
    scoring.  The returned CRPS is therefore a prior; dependence on ``y_x``
    enters only through the caller's residual query ``y_x-z``.
    """
    height, width = image.shape
    if residual_query.shape[:2] != image.shape:
        raise ValueError("residual queries must align with the image")
    prediction = witness["prediction"]
    valid = witness["mass"] > 0.0
    residual_particle = image[..., None] - prediction
    score = np.empty_like(residual_query)
    yy, xx = np.mgrid[:height, :width]
    colour = 2 * (yy & 1) + (xx & 1)

    for lane in range(4):
        sites = np.argwhere(colour == lane)
        source_count = int(sites.shape[0])
        if source_count < 2:
            raise RuntimeError("four-colour residual lane needs two sources")
        global_values = []
        global_weights = []
        local_laws = []
        for sy, sx in sites:
            local_value = residual_particle[sy, sx, valid[sy, sx]]
            local_mass = np.full(
                local_value.size, 1.0 / local_value.size, dtype=np.float64)
            local_laws.append((local_value, local_mass))
            global_values.append(local_value)
            global_weights.append(local_mass / source_count)
        global_value = np.concatenate(global_values)
        global_mass = np.concatenate(global_weights)
        global_self = _weighted_half_pair_action(global_value, global_mass)
        removed_mass = 1.0 / source_count
        remaining_mass = 1.0 - removed_mass

        for (sy, sx), (local_value, local_mass) in zip(sites, local_laws):
            local_to_global = float(np.sum(
                local_mass * _sorted_expected_absolute(
                    global_value, global_mass, local_value)))
            local_self = _weighted_half_pair_action(local_value, local_mass)
            remaining_self = (
                global_self
                - removed_mass * local_to_global
                + removed_mass * removed_mass * local_self
            ) / (remaining_mass * remaining_mass)
            query = residual_query[sy, sx]
            global_absolute = _sorted_expected_absolute(
                global_value, global_mass, query)
            local_absolute = np.sum(
                local_mass[None, :]
                * np.abs(query[:, None] - local_value[None, :]),
                axis=-1,
            )
            remaining_absolute = (
                global_absolute - removed_mass * local_absolute
            ) / remaining_mass
            score[sy, sx] = np.maximum(
                remaining_absolute - remaining_self, 0.0)
    return score


def _lineage_residual_crps(
    image: np.ndarray,
    witness: dict[str, np.ndarray],
    residual_query: np.ndarray,
    lineage: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Score residuals against source laws transported by target lineage."""
    height, width = image.shape
    pixels = image.size
    if residual_query.shape[:2] != image.shape:
        raise ValueError("residual queries must align with the image")
    if lineage.shape != (pixels, pixels):
        raise ValueError("source lineage must be an NxN transport law")
    prediction = witness["prediction"].reshape(pixels, -1)
    valid = (witness["mass"] > 0.0).reshape(pixels, -1)
    residual_particle = image.reshape(pixels, 1) - prediction
    query = residual_query.reshape(pixels, -1)
    score = np.empty_like(query)
    source_populations = []
    collision_populations = []
    for target in range(pixels):
        source = np.flatnonzero(lineage[target] > 0.0)
        values = []
        weights = []
        for identity in source:
            source_value = residual_particle[identity, valid[identity]]
            source_mass = lineage[target, identity] / source_value.size
            values.append(source_value)
            weights.append(np.full(
                source_value.size, source_mass, dtype=np.float64))
        value = np.concatenate(values)
        mass = np.concatenate(weights)
        mass /= np.sum(mass)
        self_action = _weighted_half_pair_action(value, mass)
        score[target] = np.maximum(
            _sorted_expected_absolute(value, mass, query[target]) - self_action,
            0.0,
        )
        source_populations.append(float(source.size))
        collision_populations.append(float(
            1.0 / np.sum(lineage[target] * lineage[target])))
    return score.reshape(residual_query.shape), {
        "mean_residual_source_population": float(np.mean(source_populations)),
        "mean_residual_source_collision_population": float(np.mean(
            collision_populations)),
        "minimum_residual_source_collision_population": float(np.min(
            collision_populations)),
        "maximum_target_self_lineage": float(np.max(np.abs(
            np.diag(lineage)))),
    }


def joint_characteristic_population_2d(
    observation: np.ndarray,
    *,
    angular_order: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Disintegrate witnessed signal and residual laws on ``y=z+r``."""
    image = _validate(observation)
    signal, signal_diagnostic = witnessed_characteristic_population_2d(
        image, angular_order=angular_order)
    witness, witness_diagnostic = crossfit_characteristic_population_2d(
        image, angular_order=angular_order)
    residual = image[..., None] - signal["prediction"]
    residual_score = _four_colour_leave_one_out_residual_crps(
        image, witness, residual)
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    action = signal["variation"] + signal["crps"] + residual_score
    conductance = signal["scale_conductance"] / np.maximum(action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "signal": signal["prediction"],
        "residual": residual,
        "mass": mass,
        "signal_crps": signal["crps"],
        "residual_crps": residual_score,
        "source_identity": signal["source_identity"],
        "source_coefficient": signal["source_coefficient"],
    }, {
        "status": "joint signal/residual characteristic disintegration",
        "signal_law": signal_diagnostic,
        "residual_witness": witness_diagnostic,
        "observation_graph_maximum_error": float(np.max(np.abs(
            signal["prediction"] + residual - image[..., None]))),
        "residual_prior": (
            "four-colour leave-one-source-out empirical CRPS"
        ),
        "target_identity_excluded_from_prior": True,
        "theory_status": "joint-measure experiment; not promoted",
    }


def joint_characteristic_measure_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "mean",
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the signal marginal of the exact joint observation measure."""
    population, diagnostic = joint_characteristic_population_2d(
        observation, angular_order=angular_order)
    if barycenter == "mean":
        readout = np.sum(population["mass"] * population["signal"], axis=-1)
    elif barycenter == "median":
        readout = _weighted_median(population["signal"], population["mass"])
    else:
        raise ValueError("barycenter must be 'mean' or 'median'")
    return readout, {**diagnostic, "barycenter": barycenter}


def lineage_joint_characteristic_population_2d(
    observation: np.ndarray,
    *,
    angular_order: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Disintegrate residual mass through the signal law's source lineage."""
    image = _validate(observation)
    signal, signal_diagnostic = witnessed_characteristic_population_2d(
        image, angular_order=angular_order)
    witness, witness_diagnostic = crossfit_characteristic_population_2d(
        image, angular_order=angular_order)
    _influence, lineage = _source_influence_and_lineage(
        signal["mass"],
        signal["source_identity"],
        signal["source_coefficient"],
    )
    residual = image[..., None] - signal["prediction"]
    residual_score, lineage_diagnostic = _lineage_residual_crps(
        image, witness, residual, lineage)
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    action = signal["variation"] + signal["crps"] + residual_score
    conductance = signal["scale_conductance"] / np.maximum(action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "signal": signal["prediction"],
        "residual": residual,
        "mass": mass,
        "signal_crps": signal["crps"],
        "residual_crps": residual_score,
        "source_identity": signal["source_identity"],
        "source_coefficient": signal["source_coefficient"],
        "prior_lineage": lineage,
    }, {
        "status": "joint disintegration under transported source-lineage residual law",
        "signal_law": signal_diagnostic,
        "residual_witness": witness_diagnostic,
        "residual_prior": "target source-lineage transported empirical CRPS",
        "observation_graph_maximum_error": float(np.max(np.abs(
            signal["prediction"] + residual - image[..., None]))),
        "target_identity_excluded_from_prior": (
            lineage_diagnostic["maximum_target_self_lineage"] == 0.0
        ),
        **lineage_diagnostic,
        "theory_status": "lineage-local residual experiment; not promoted",
    }


def lineage_joint_characteristic_measure_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "mean",
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the signal marginal of the lineage-local joint measure."""
    population, diagnostic = lineage_joint_characteristic_population_2d(
        observation, angular_order=angular_order)
    if barycenter == "mean":
        readout = np.sum(population["mass"] * population["signal"], axis=-1)
    elif barycenter == "median":
        readout = _weighted_median(population["signal"], population["mass"])
    else:
        raise ValueError("barycenter must be 'mean' or 'median'")
    return readout, {**diagnostic, "barycenter": barycenter}


def _transport_residual_lineage(
    image: np.ndarray,
    signal: dict[str, np.ndarray],
    witness: dict[str, np.ndarray],
    lineage: np.ndarray,
    *,
    maximum_transports: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Broaden source lineage only while held-out residual CRPS descends."""
    ceiling = int(maximum_transports)
    if ceiling < 1:
        raise ValueError("maximum lineage transports must be positive")
    held_out_signal = np.sum(
        signal["mass"] * signal["prediction"], axis=-1)
    held_out_residual = image - held_out_signal
    geometry = relation_transport_metric_2d(image)
    operator, operator_diagnostic = source_measure_operator(
        geometry["metric_xx"], geometry["metric_xy"], geometry["metric_yy"])

    def risk(source_law: np.ndarray) -> float:
        score, _ = _lineage_residual_crps(
            image,
            witness,
            held_out_residual[..., None],
            source_law,
        )
        return float(np.mean(score))

    state = lineage
    action = risk(state)
    records = []
    equilibrium = False
    for transport in range(ceiling):
        candidate = _exclude_target_identity(operator @ state)
        candidate_action = risk(candidate)
        numerical = np.finfo(float).eps * max(action, 1.0)
        accepted = candidate_action < action - numerical
        records.append({
            "transport": transport,
            "accepted": accepted,
            "held_out_residual_crps_before": action,
            "held_out_residual_crps_after": candidate_action,
            "mean_collision_population_after": float(np.mean(
                1.0 / np.sum(candidate * candidate, axis=1))),
        })
        if not accepted:
            equilibrium = True
            break
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return state, {
        "accepted_lineage_transports": int(sum(
            record["accepted"] for record in records)),
        "lineage_transport_ceiling_hit": ceiling_hit,
        "terminal_held_out_residual_crps": action,
        "lineage_transports": records,
        "source_operator": operator_diagnostic,
        "metric": geometry["geometry"],
        "metric_theory_status": geometry["theory_status"],
    }


def transported_lineage_joint_characteristic_population_2d(
    observation: np.ndarray,
    *,
    angular_order: int = 1,
    maximum_lineage_transports: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Joint law under CRPS-stopped determinant-one lineage transport."""
    image = _validate(observation)
    signal, signal_diagnostic = witnessed_characteristic_population_2d(
        image, angular_order=angular_order)
    witness, witness_diagnostic = crossfit_characteristic_population_2d(
        image, angular_order=angular_order)
    _influence, direct_lineage = _source_influence_and_lineage(
        signal["mass"],
        signal["source_identity"],
        signal["source_coefficient"],
    )
    lineage, transport_diagnostic = _transport_residual_lineage(
        image,
        signal,
        witness,
        direct_lineage,
        maximum_transports=maximum_lineage_transports,
    )
    residual = image[..., None] - signal["prediction"]
    residual_score, lineage_diagnostic = _lineage_residual_crps(
        image, witness, residual, lineage)
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    action = signal["variation"] + signal["crps"] + residual_score
    conductance = signal["scale_conductance"] / np.maximum(action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "signal": signal["prediction"],
        "residual": residual,
        "mass": mass,
        "signal_crps": signal["crps"],
        "residual_crps": residual_score,
        "source_identity": signal["source_identity"],
        "source_coefficient": signal["source_coefficient"],
        "transported_prior_lineage": lineage,
    }, {
        "status": "joint disintegration under CRPS-stopped source transport",
        "signal_law": signal_diagnostic,
        "residual_witness": witness_diagnostic,
        "residual_prior": (
            "determinant-one Selling transport stopped by held-out CRPS"
        ),
        "observation_graph_maximum_error": float(np.max(np.abs(
            signal["prediction"] + residual - image[..., None]))),
        "target_identity_excluded_from_prior": (
            lineage_diagnostic["maximum_target_self_lineage"] == 0.0
        ),
        **lineage_diagnostic,
        **transport_diagnostic,
        "theory_status": (
            "transported-lineage residual experiment; relation metric still "
            "uses the crystalline precursor"
        ),
    }


def transported_lineage_joint_characteristic_measure_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "mean",
    angular_order: int = 1,
    maximum_lineage_transports: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the signal marginal after intrinsic residual-lineage transport."""
    population, diagnostic = transported_lineage_joint_characteristic_population_2d(
        observation,
        angular_order=angular_order,
        maximum_lineage_transports=maximum_lineage_transports,
    )
    if barycenter == "mean":
        readout = np.sum(population["mass"] * population["signal"], axis=-1)
    elif barycenter == "median":
        readout = _weighted_median(population["signal"], population["mass"])
    else:
        raise ValueError("barycenter must be 'mean' or 'median'")
    return readout, {**diagnostic, "barycenter": barycenter}


def denoise_joint_characteristic_transport_2d(
    observation: np.ndarray,
    *,
    maximum_continuations: int = 32,
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue the same joint law until residual covariance vanishes."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial = joint_characteristic_measure_2d(
        image, barycenter="median", angular_order=angular_order)
    lower = float(np.min(image))
    upper = float(np.max(image))
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        prediction, measure = joint_characteristic_measure_2d(
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
        candidate = np.clip(state + coefficient * prediction, lower, upper)
        candidate_action = float(np.mean((image - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
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
            "measure": measure,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "joint residual covariance equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "theory_status": "self-similar joint continuation experiment; not promoted",
    }


def denoise_joint_authority_transport_2d(
    observation: np.ndarray,
    *,
    maximum_continuations: int = 32,
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue only positive debiased energy of the local signal marginal."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial = joint_characteristic_measure_2d(
        image, barycenter="median", angular_order=angular_order)
    lower = float(np.min(image))
    upper = float(np.max(image))
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        population, measure = joint_characteristic_population_2d(
            residual, angular_order=angular_order)
        prediction = np.sum(
            population["mass"] * population["signal"], axis=-1)
        square_mass = population["mass"] * population["mass"]
        uncertainty = np.sum(
            square_mass
            * (population["signal"] - prediction[..., None]) ** 2,
            axis=-1,
        )
        collision_population = 1.0 / np.sum(square_mass, axis=-1)
        energy = prediction * prediction
        excess = np.maximum(energy - uncertainty, 0.0)
        authority = np.divide(
            excess,
            energy,
            out=np.zeros_like(excess),
            where=energy > 0.0,
        )
        update = authority * prediction
        update_energy = float(np.mean(update * update))
        projection = float(np.mean(residual * update))
        local = {
            "positive_authority_fraction": float(np.mean(authority > 0.0)),
            "mean_authority": float(np.mean(authority)),
            "mean_signal_variance": float(np.mean(uncertainty)),
            "mean_positive_predictive_energy": float(np.mean(excess)),
            "mean_collision_population": float(np.mean(collision_population)),
            "minimum_collision_population": float(np.min(collision_population)),
        }
        if update_energy == 0.0 or projection <= 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": 0.0,
                **local,
            })
            break
        descent = min(1.0, projection / update_energy)
        candidate = np.clip(state + descent * update, lower, upper)
        candidate_action = float(np.mean((image - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": descent,
                **local,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            "global_descent_coefficient": descent,
            **local,
            "measure": measure,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "joint local-authority equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "authority_law": (
            "(q^2-sum_i w_i^2 (z_i-q)^2)_+/q^2"
        ),
        "theory_status": "local joint-authority experiment; not promoted",
    }


def _signed_source_ancestry(
    mass: np.ndarray,
    source_identity: np.ndarray,
    source_coefficient: np.ndarray,
) -> np.ndarray:
    """Aggregate repeated characteristic particles by conserved source ID."""
    height, width, particles = mass.shape
    pixels = height * width
    ancestry = np.zeros((pixels, pixels), dtype=np.float64)
    target = np.broadcast_to(
        np.arange(pixels, dtype=np.int64).reshape(height, width, 1),
        (height, width, particles),
    )
    for endpoint in range(source_identity.shape[-1]):
        np.add.at(
            ancestry,
            (target.ravel(), source_identity[..., endpoint].ravel()),
            (mass * source_coefficient[..., endpoint]).ravel(),
        )
    return ancestry


def _source_influence_and_lineage(
    mass: np.ndarray,
    source_identity: np.ndarray,
    source_coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact signed influence and positive conserved source lineage."""
    height, width, particles = mass.shape
    pixels = height * width
    influence = np.zeros((pixels, pixels), dtype=np.float64)
    lineage = np.zeros((pixels, pixels), dtype=np.float64)
    target = np.broadcast_to(
        np.arange(pixels, dtype=np.int64).reshape(height, width, 1),
        (height, width, particles),
    )
    coefficient_norm = np.sum(np.abs(source_coefficient), axis=-1)
    for endpoint in range(source_identity.shape[-1]):
        source = source_identity[..., endpoint]
        coefficient = source_coefficient[..., endpoint]
        np.add.at(
            influence,
            (target.ravel(), source.ravel()),
            (mass * coefficient).ravel(),
        )
        np.add.at(
            lineage,
            (target.ravel(), source.ravel()),
            (mass * np.abs(coefficient) / coefficient_norm).ravel(),
        )
    return influence, lineage


def _lineage_overlap_uncertainty(
    influence: np.ndarray,
    lineage: np.ndarray,
    held_out_error: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pull source errors through the normalized lineage Gram covariance.

    If ``B`` is nonnegative source lineage and ``D=diag(B.T B)``, then
    ``C=D^-1/2 B.T B D^-1/2`` is a positive-semidefinite source correlation
    with unit diagonal on every represented source.  For signed estimator
    influence ``A`` and held-out source error ``e``, uncertainty is

    ``diag((A diag(e)) C (A diag(e)).T)``.

    The factorized evaluation avoids constructing ``C`` explicitly.
    """
    signed_error = influence * np.asarray(held_out_error, dtype=np.float64)[None, :]
    column_energy = np.sum(lineage * lineage, axis=0)
    if np.any(column_energy <= 0.0):
        raise RuntimeError("every source must participate in positive lineage")
    whitened = signed_error / np.sqrt(column_energy)[None, :]
    correlated = whitened @ lineage.T
    uncertainty = np.sum(correlated * correlated, axis=1)
    diagonal = np.sum(signed_error * signed_error, axis=1)
    return uncertainty, {
        "mean_independent_source_uncertainty": float(np.mean(diagonal)),
        "mean_overlap_source_uncertainty": float(np.mean(uncertainty)),
        "mean_overlap_inflation": float(np.mean(np.divide(
            uncertainty,
            diagonal,
            out=np.zeros_like(uncertainty),
            where=diagonal > 0.0,
        ))),
        "minimum_lineage_column_energy": float(np.min(column_energy)),
        "maximum_lineage_column_energy": float(np.max(column_energy)),
    }


def denoise_joint_source_authority_transport_2d(
    observation: np.ndarray,
    *,
    maximum_continuations: int = 32,
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue residual energy debiased by exact characteristic ancestry."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial = joint_characteristic_measure_2d(
        image, barycenter="median", angular_order=angular_order)
    lower = float(np.min(image))
    upper = float(np.max(image))
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        population, measure = joint_characteristic_population_2d(
            residual, angular_order=angular_order)
        prediction = np.sum(
            population["mass"] * population["signal"], axis=-1)
        ancestry = _signed_source_ancestry(
            population["mass"],
            population["source_identity"],
            population["source_coefficient"],
        )
        strict, _ = crossfit_characteristic_population_2d(
            residual, angular_order=angular_order)
        strict_mean = np.sum(strict["mass"] * strict["prediction"], axis=-1)
        held_out_error = (residual - strict_mean).ravel()
        uncertainty = (ancestry * ancestry) @ (held_out_error * held_out_error)
        uncertainty = uncertainty.reshape(image.shape)
        energy = prediction * prediction
        excess = np.maximum(energy - uncertainty, 0.0)
        authority = np.divide(
            excess,
            energy,
            out=np.zeros_like(excess),
            where=energy > 0.0,
        )
        update = authority * prediction
        update_energy = float(np.mean(update * update))
        projection = float(np.mean(residual * update))
        row_mass_error = float(np.max(np.abs(np.sum(ancestry, axis=1) - 1.0)))
        diagonal_influence = float(np.max(np.abs(np.diag(ancestry))))
        collision_population = 1.0 / np.sum(ancestry * ancestry, axis=1)
        local = {
            "positive_authority_fraction": float(np.mean(authority > 0.0)),
            "mean_authority": float(np.mean(authority)),
            "mean_source_uncertainty": float(np.mean(uncertainty)),
            "mean_positive_predictive_energy": float(np.mean(excess)),
            "mean_source_collision_population": float(np.mean(
                collision_population)),
            "source_ancestry_row_mass_maximum_error": row_mass_error,
            "maximum_target_self_influence": diagonal_influence,
        }
        if update_energy == 0.0 or projection <= 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": 0.0,
                **local,
            })
            break
        descent = min(1.0, projection / update_energy)
        candidate = np.clip(state + descent * update, lower, upper)
        candidate_action = float(np.mean((image - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": descent,
                **local,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            "global_descent_coefficient": descent,
            **local,
            "measure": measure,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "joint conserved-source authority equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "authority_law": (
            "signed source ancestry squared against held-out source error"
        ),
        "theory_status": "conserved-source authority experiment; not promoted",
    }


def denoise_joint_overlap_covariance_transport_2d(
    observation: np.ndarray,
    *,
    maximum_continuations: int = 32,
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue under covariance induced by common conserved source lineage."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial = joint_characteristic_measure_2d(
        image, barycenter="median", angular_order=angular_order)
    lower = float(np.min(image))
    upper = float(np.max(image))
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        population, measure = joint_characteristic_population_2d(
            residual, angular_order=angular_order)
        prediction = np.sum(
            population["mass"] * population["signal"], axis=-1)
        influence, lineage = _source_influence_and_lineage(
            population["mass"],
            population["source_identity"],
            population["source_coefficient"],
        )
        strict, _ = crossfit_characteristic_population_2d(
            residual, angular_order=angular_order)
        strict_mean = np.sum(strict["mass"] * strict["prediction"], axis=-1)
        held_out_error = (residual - strict_mean).ravel()
        uncertainty, overlap = _lineage_overlap_uncertainty(
            influence, lineage, held_out_error)
        uncertainty = uncertainty.reshape(image.shape)
        energy = prediction * prediction
        excess = np.maximum(energy - uncertainty, 0.0)
        authority = np.divide(
            excess,
            energy,
            out=np.zeros_like(excess),
            where=energy > 0.0,
        )
        update = authority * prediction
        update_energy = float(np.mean(update * update))
        projection = float(np.mean(residual * update))
        local = {
            "positive_authority_fraction": float(np.mean(authority > 0.0)),
            "mean_authority": float(np.mean(authority)),
            "mean_positive_predictive_energy": float(np.mean(excess)),
            "influence_row_mass_maximum_error": float(np.max(np.abs(
                np.sum(influence, axis=1) - 1.0))),
            "lineage_row_mass_maximum_error": float(np.max(np.abs(
                np.sum(lineage, axis=1) - 1.0))),
            "maximum_target_self_influence": float(np.max(np.abs(
                np.diag(influence)))),
            "maximum_target_self_lineage": float(np.max(np.abs(
                np.diag(lineage)))),
            **overlap,
        }
        if update_energy == 0.0 or projection <= 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": 0.0,
                **local,
            })
            break
        descent = min(1.0, projection / update_energy)
        candidate = np.clip(state + descent * update, lower, upper)
        candidate_action = float(np.mean((image - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": descent,
                **local,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            "global_descent_coefficient": descent,
            **local,
            "measure": measure,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "joint lineage-overlap covariance equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "authority_law": (
            "PSD normalized Gram covariance of conserved source lineage"
        ),
        "theory_status": "source-overlap covariance experiment; not promoted",
    }


def _lineage_covariance_authority(
    lineage: np.ndarray,
    residual: np.ndarray,
    held_out_prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Debias local residual--prediction covariance under source lineage."""
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    q = np.asarray(held_out_prediction, dtype=np.float64).reshape(-1)
    if lineage.shape != (r.size, r.size) or q.size != r.size:
        raise ValueError("lineage and covariance fields must share one domain")
    product = r * q
    covariance = lineage @ product
    covariance_variance = np.sum(
        lineage * lineage * (product[None, :] - covariance[:, None]) ** 2,
        axis=1,
    )
    excess = np.maximum(covariance * covariance - covariance_variance, 0.0)
    authority = np.divide(
        excess,
        covariance * covariance,
        out=np.zeros_like(excess),
        where=covariance > 0.0,
    )
    return authority, {
        "mean_lineage_covariance": float(np.mean(covariance)),
        "mean_lineage_covariance_variance": float(np.mean(covariance_variance)),
        "mean_positive_covariance_energy": float(np.mean(excess)),
        "positive_authority_fraction": float(np.mean(authority > 0.0)),
        "mean_authority": float(np.mean(authority)),
    }


def denoise_joint_lineage_covariance_transport_2d(
    observation: np.ndarray,
    *,
    maximum_continuations: int = 32,
    angular_order: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue joint residuals under cross-fitted causal covariance evidence."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    state, initial = joint_characteristic_measure_2d(
        image, barycenter="median", angular_order=angular_order)
    lower = float(np.min(image))
    upper = float(np.max(image))
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        posterior, measure = joint_characteristic_population_2d(
            residual, angular_order=angular_order)
        prediction = np.sum(
            posterior["mass"] * posterior["signal"], axis=-1)

        # Authority is a prior object.  Its weights and predictions come from
        # the witnessed signal law before observation-graph disintegration, so
        # source y_s cannot validate its own covariance contribution.
        prior, _ = witnessed_characteristic_population_2d(
            residual, angular_order=angular_order)
        held_out_prediction = np.sum(
            prior["mass"] * prior["prediction"], axis=-1)
        _influence, lineage = _source_influence_and_lineage(
            prior["mass"],
            prior["source_identity"],
            prior["source_coefficient"],
        )
        authority, covariance = _lineage_covariance_authority(
            lineage, residual, held_out_prediction)
        authority = authority.reshape(image.shape)
        update = authority * prediction
        update_energy = float(np.mean(update * update))
        projection = float(np.mean(residual * update))
        local = {
            **covariance,
            "lineage_row_mass_maximum_error": float(np.max(np.abs(
                np.sum(lineage, axis=1) - 1.0))),
            "maximum_target_self_lineage": float(np.max(np.abs(
                np.diag(lineage)))),
        }
        if update_energy == 0.0 or projection <= 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": 0.0,
                **local,
            })
            break
        descent = min(1.0, projection / update_energy)
        candidate = np.clip(state + descent * update, lower, upper)
        candidate_action = float(np.mean((image - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": descent,
                **local,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            "global_descent_coefficient": descent,
            **local,
            "measure": measure,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "joint lineage covariance equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "authority_law": (
            "positive debiased residual-prediction covariance under prior lineage"
        ),
        "theory_status": "lineage-local covariance experiment; not promoted",
    }
