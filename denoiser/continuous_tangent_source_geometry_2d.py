"""CRPS-stopped continuous source transport for tangent information geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .continuous_source_transport import (
    _exclude_target_identity,
    source_measure_operator,
)
from .continuous_tangent_transport_2d import (
    continuous_tangent_jet_field_2d,
    continuous_tangent_jet_particles_2d,
    continuous_tangent_signal_population_2d,
)
from .crossfit_characteristic_transport_2d import (
    crossfit_characteristic_population_2d,
)
from .fused_transport_geometry import (
    combine_information_geometries,
    predictive_horizontal_wasserstein_geometry,
    predictive_lineage_jet_geometry,
    predictive_lineage_prolongation_geometry,
    weighted_empirical_quantiles,
)
from .witnessed_characteristic_transport_2d import (
    _lineage_residual_crps,
    _source_influence_and_lineage,
    _validate,
)


def _weighted_support_crps(
    weights: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Exact scalar CRPS for row-wise laws on one shared support."""
    mass = np.asarray(weights, dtype=np.float64)
    value = np.asarray(support, dtype=np.float64).reshape(-1)
    target = np.asarray(query, dtype=np.float64).reshape(-1)
    if mass.shape != (target.size, value.size):
        raise ValueError("CRPS weights, support, and target must align")
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("CRPS weights must be finite and nonnegative")
    row_mass = np.sum(mass, axis=1)
    if not np.allclose(row_mass, 1.0, atol=4e-13, rtol=4e-13):
        raise ValueError("CRPS source laws must conserve unit mass")
    absolute = np.sum(
        mass * np.abs(value[None, :] - target[:, None]), axis=1)
    order = np.argsort(value, kind="stable")
    ordered_value = value[order]
    ordered_mass = mass[:, order]
    cumulative_mass = np.cumsum(ordered_mass, axis=1) - ordered_mass
    weighted_value = ordered_mass * ordered_value[None, :]
    cumulative_value = np.cumsum(weighted_value, axis=1) - weighted_value
    half_pair = np.sum(ordered_mass * (
        ordered_value[None, :] * cumulative_mass - cumulative_value
    ), axis=1)
    return np.maximum(absolute - half_pair, 0.0)


def _weighted_support_half_pair(
    weights: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Return one half of the row-wise expected pair distance."""
    mass = np.asarray(weights, dtype=np.float64)
    value = np.asarray(support, dtype=np.float64).reshape(-1)
    order = np.argsort(value, kind="stable")
    ordered_value = value[order]
    ordered_mass = mass[:, order]
    cumulative_mass = np.cumsum(ordered_mass, axis=1) - ordered_mass
    weighted_value = ordered_mass * ordered_value[None, :]
    cumulative_value = np.cumsum(weighted_value, axis=1) - weighted_value
    return np.sum(ordered_mass * (
        ordered_value[None, :] * cumulative_mass - cumulative_value
    ), axis=1)


def _projective_jet_crps(
    lineage: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    tangent: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Integrate held-out directional-jet CRPS over the projective circle."""
    directions = np.unique(np.asarray(tangent, dtype=np.float64), axis=0)
    gx = np.asarray(gradient_x, dtype=np.float64)
    gy = np.asarray(gradient_y, dtype=np.float64)
    longest = float(max(gx.shape))
    directional_scores = []
    for dy, dx in directions:
        jet = longest * (dx * gx + dy * gy).ravel()
        directional_scores.append(_weighted_support_crps(
            lineage, jet, jet))
    score = np.stack(directional_scores, axis=1)
    angular_weight = np.pi / len(directions)
    point_action = 0.5 * angular_weight * np.sum(score, axis=1)
    return float(np.mean(point_action)), {
        "mean_projective_jet_crps": float(np.mean(point_action)),
        "maximum_projective_jet_crps": float(np.max(point_action)),
        "projective_jet_directions": int(len(directions)),
        "projective_energy_identity": (
            "one half integral_0^pi of directional CRPS"
        ),
    }


def _projective_jet_law_crps(
    lineage: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    signal_mass: np.ndarray,
    directional_derivative: np.ndarray,
    tangent: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Score transported source jets against held-out local jet laws."""
    directions, inverse = np.unique(
        np.asarray(tangent, dtype=np.float64), axis=0, return_inverse=True)
    gx = np.asarray(gradient_x, dtype=np.float64)
    gy = np.asarray(gradient_y, dtype=np.float64)
    mass = np.asarray(signal_mass, dtype=np.float64)
    derivative = np.asarray(directional_derivative, dtype=np.float64)
    pixels = gx.size
    longest = float(max(gx.shape))
    point_score = np.zeros(pixels, dtype=np.float64)
    represented = np.zeros(pixels, dtype=np.float64)
    for direction_index, (dy, dx) in enumerate(directions):
        member = inverse == direction_index
        query_value = (
            longest * derivative[..., member]).reshape(pixels, -1)
        query_mass = mass[..., member].reshape(pixels, -1)
        query_total = np.sum(query_mass, axis=1)
        valid = query_total > 0.0
        query_mass[valid] /= query_total[valid, None]
        source_jet = longest * (dx * gx + dy * gy).ravel()
        half_pair = _weighted_support_half_pair(lineage, source_jet)
        cross = np.zeros(pixels, dtype=np.float64)
        for target in np.flatnonzero(valid):
            source_to_query = (
                np.abs(source_jet[:, None] - query_value[target][None, :])
                @ query_mass[target]
            )
            cross[target] = lineage[target] @ source_to_query
        point_score[valid] += np.maximum(
            cross[valid] - half_pair[valid], 0.0)
        represented[valid] += 1.0
    if np.any(represented <= 0.0):
        raise RuntimeError("held-out jet law left a target unsupported")
    point_action = 0.5 * np.pi * point_score / represented
    return float(np.mean(point_action)), {
        "mean_projective_jet_law_crps": float(np.mean(point_action)),
        "maximum_projective_jet_law_crps": float(np.max(point_action)),
        "minimum_represented_jet_directions": int(np.min(represented)),
        "projective_jet_law_directions": int(len(directions)),
        "projective_energy_identity": (
            "one half integral_0^pi of local-law directional CRPS"
        ),
    }


def _joint_bundle_energy_score(
    lineage: np.ndarray,
    source_residual: np.ndarray,
    source_gradient_x: np.ndarray,
    source_gradient_y: np.ndarray,
    query_residual: np.ndarray,
    query_gradient_x: np.ndarray,
    query_gradient_y: np.ndarray,
    query_mass: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Proper energy score on joint residual/full-jet source particles."""
    source_distance, cross_cost = _joint_bundle_costs(
        source_residual,
        source_gradient_x,
        source_gradient_y,
        query_residual,
        query_gradient_x,
        query_gradient_y,
        query_mass,
    )
    score = _joint_bundle_score_field(lineage, source_distance, cross_cost)
    return float(np.mean(score)), {
        "mean_joint_bundle_energy_score": float(np.mean(score)),
        "maximum_joint_bundle_energy_score": float(np.max(score)),
        "joint_bundle_coordinates": "(residual, physical jet x, physical jet y)",
        "joint_bundle_query_particles": int(np.asarray(query_mass).shape[-1]),
    }


def _joint_bundle_costs(
    source_residual: np.ndarray,
    source_gradient_x: np.ndarray,
    source_gradient_y: np.ndarray,
    query_residual: np.ndarray,
    query_gradient_x: np.ndarray,
    query_gradient_y: np.ndarray,
    query_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source self-distance and target-to-source joint cross costs."""
    residual = np.asarray(source_residual, dtype=np.float64)
    gx = np.asarray(source_gradient_x, dtype=np.float64)
    gy = np.asarray(source_gradient_y, dtype=np.float64)
    mass = np.asarray(query_mass, dtype=np.float64)
    height, width = residual.shape
    pixels = residual.size
    longest = float(max(height, width))
    source = np.column_stack((
        residual.ravel(),
        longest * gx.ravel(),
        longest * gy.ravel(),
    ))
    query = np.stack((
        np.asarray(query_residual, dtype=np.float64),
        longest * np.asarray(query_gradient_x, dtype=np.float64),
        longest * np.asarray(query_gradient_y, dtype=np.float64),
    ), axis=-1).reshape(pixels, mass.shape[-1], 3)
    probability = mass.reshape(pixels, -1).copy()
    probability /= np.sum(probability, axis=1, keepdims=True)
    source_distance = np.linalg.norm(
        source[:, None, :] - source[None, :, :], axis=-1)
    cross_cost = np.empty((pixels, pixels), dtype=np.float64)
    for target in range(pixels):
        distance = np.linalg.norm(
            source[:, None, :] - query[target][None, :, :], axis=-1)
        cross_cost[target] = distance @ probability[target]
    return source_distance, cross_cost


def _joint_bundle_score_field(
    lineage: np.ndarray,
    source_distance: np.ndarray,
    cross_cost: np.ndarray,
) -> np.ndarray:
    """Evaluate pointwise joint energy from precomputed bundle costs."""
    ancestry = np.asarray(lineage, dtype=np.float64)
    half_pair = 0.5 * np.sum(
        ancestry * (ancestry @ source_distance), axis=1)
    cross = np.sum(ancestry * cross_cost, axis=1)
    return np.maximum(cross - half_pair, 0.0)


def _analytic_local_joint_transport(
    lineage: np.ndarray,
    transported: np.ndarray,
    source_distance: np.ndarray,
    cross_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Minimize each target's joint energy along its transport chord.

    Euclidean distance is conditionally negative definite, so the energy
    score restricted to every mass-conserving chord is a convex quadratic.
    The clipped stationary point therefore supplies a continuous local
    transport authority without a step size or an acceptance threshold.
    """
    state = np.asarray(lineage, dtype=np.float64)
    destination = np.asarray(transported, dtype=np.float64)
    delta = destination - state
    distance_delta = delta @ source_distance
    linear = (
        np.sum(delta * cross_cost, axis=1)
        - np.sum(state * distance_delta, axis=1)
    )
    curvature = np.maximum(
        -0.5 * np.sum(delta * distance_delta, axis=1), 0.0)
    step = np.zeros(state.shape[0], dtype=np.float64)
    curved = curvature > np.finfo(float).tiny
    step[curved] = np.clip(
        -linear[curved] / (2.0 * curvature[curved]), 0.0, 1.0)
    step[~curved & (linear < 0.0)] = 1.0
    candidate = state + step[:, None] * delta
    before = _joint_bundle_score_field(state, source_distance, cross_cost)
    after = _joint_bundle_score_field(candidate, source_distance, cross_cost)
    return candidate, step, {
        "maximum_local_energy_increase": float(np.max(after - before)),
        "mean_local_energy_decrease": float(np.mean(before - after)),
    }


def _causal_support_joint_transport(
    lineage: np.ndarray,
    transported: np.ndarray,
    support_labels: np.ndarray,
    source_distance: np.ndarray,
    cross_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Minimize joint energy after reducing its variation by causal support.

    The labels are not score bands.  They are the irreversible first-arrival
    cells emitted by the predictive information volume and its own eikonal
    metric.  A cell shares one transport coordinate, so witness fluctuations
    cannot independently move individual pixels before support exists.
    """
    state = np.asarray(lineage, dtype=np.float64)
    destination = np.asarray(transported, dtype=np.float64)
    label = np.asarray(support_labels, dtype=np.int64).reshape(-1)
    if label.size != state.shape[0] or np.any(label < 0):
        raise ValueError("causal support labels must cover every target")
    cells = int(np.max(label)) + 1
    delta = destination - state
    distance_delta = delta @ source_distance
    linear = (
        np.sum(delta * cross_cost, axis=1)
        - np.sum(state * distance_delta, axis=1)
    )
    curvature = np.maximum(
        -0.5 * np.sum(delta * distance_delta, axis=1), 0.0)
    support_linear = np.bincount(
        label, weights=linear, minlength=cells)
    support_curvature = np.bincount(
        label, weights=curvature, minlength=cells)
    support_step = np.zeros(cells, dtype=np.float64)
    curved = support_curvature > np.finfo(float).tiny
    support_step[curved] = np.clip(
        -support_linear[curved] / (2.0 * support_curvature[curved]),
        0.0,
        1.0,
    )
    support_step[~curved & (support_linear < 0.0)] = 1.0
    step = support_step[label]
    candidate = state + step[:, None] * delta
    before = _joint_bundle_score_field(state, source_distance, cross_cost)
    after = _joint_bundle_score_field(candidate, source_distance, cross_cost)
    support_change = np.bincount(
        label, weights=after - before, minlength=cells)
    return candidate, step, {
        "maximum_support_energy_increase": float(np.max(support_change)),
        "maximum_point_energy_increase": float(np.max(after - before)),
        "mean_support_transport_step": float(np.mean(support_step)),
        "positive_support_step_fraction": float(np.mean(support_step > 0.0)),
    }


def _joint_bundle_graph_gradient_flow(
    lineage: np.ndarray,
    operator: sparse.spmatrix,
    stationary_mass: np.ndarray,
    source_distance: np.ndarray,
    cross_cost: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Take the exact admissible energy-gradient flux on the source graph.

    For every target law, joint energy supplies a cotangent potential on
    source identity.  Reversible Selling conductance carries probability from
    higher to lower potential with donor mobility.  The continuity equation
    conserves mass; target-incident edges are removed exactly.  Positivity and
    the convex quadratic energy along that flux determine the step without a
    learning rate.
    """
    state = np.asarray(lineage, dtype=np.float64)
    transition = sparse.csr_matrix(operator, dtype=np.float64)
    pixels = state.shape[0]
    if state.shape != (pixels, pixels) or transition.shape != state.shape:
        raise ValueError("lineage and source operator must be square and aligned")
    stationary = np.asarray(stationary_mass, dtype=np.float64).reshape(-1)
    if stationary.size != pixels or np.any(stationary <= 0.0):
        raise ValueError("stationary source mass must be positive and aligned")
    stationary /= np.sum(stationary)
    conductance = (
        0.5
        * (
            sparse.diags(stationary) @ transition
            + transition.T @ sparse.diags(stationary)
        )
    ).tocoo()
    edge = conductance.row < conductance.col
    first = conductance.row[edge]
    second = conductance.col[edge]
    weight = conductance.data[edge]
    candidate = state.copy()
    accepted_targets = 0
    maximum_increase = 0.0
    total_variation = np.zeros(pixels, dtype=np.float64)
    step_values = np.zeros(pixels, dtype=np.float64)
    before = _joint_bundle_score_field(state, source_distance, cross_cost)
    for target in range(pixels):
        valid = (first != target) & (second != target)
        a = first[valid]
        b = second[valid]
        conductance_value = weight[valid]
        law = state[target]
        potential = cross_cost[target] - law @ source_distance
        difference = potential[a] - potential[b]
        donor = np.where(difference > 0.0, a, b)
        receiver = np.where(difference > 0.0, b, a)
        rate = (
            conductance_value
            * np.abs(difference)
            * law[donor]
        )
        positive = rate > 0.0
        if not np.any(positive):
            continue
        donor = donor[positive]
        receiver = receiver[positive]
        rate = rate[positive]
        delta = np.zeros(pixels, dtype=np.float64)
        np.add.at(delta, donor, -rate)
        np.add.at(delta, receiver, rate)
        linear = float(potential @ delta)
        if linear >= 0.0:
            continue
        distance_delta = source_distance @ delta
        curvature = max(float(-0.5 * delta @ distance_delta), 0.0)
        outgoing = np.bincount(
            donor, weights=rate, minlength=pixels)
        losing = outgoing > 0.0
        positivity_time = float(np.min(
            law[losing] / outgoing[losing]))
        energy_time = (
            -linear / (2.0 * curvature)
            if curvature > np.finfo(float).tiny
            else np.inf
        )
        time = min(positivity_time, energy_time)
        if not np.isfinite(time) or time <= 0.0:
            continue
        candidate[target] = np.maximum(law + time * delta, 0.0)
        candidate[target, target] = 0.0
        candidate[target] /= np.sum(candidate[target])
        accepted_targets += 1
        step_values[target] = time
        total_variation[target] = 0.5 * np.sum(
            np.abs(candidate[target] - law))
    after = _joint_bundle_score_field(
        candidate, source_distance, cross_cost)
    maximum_increase = float(np.max(after - before))
    return candidate, {
        "positive_gradient_flow_fraction": float(
            accepted_targets / pixels),
        "mean_gradient_flow_time": float(np.mean(
            step_values[step_values > 0.0])) if accepted_targets else 0.0,
        "maximum_gradient_flow_time": float(np.max(step_values)),
        "mean_gradient_flow_total_variation": float(
            np.mean(total_variation)),
        "maximum_gradient_flow_total_variation": float(
            np.max(total_variation)),
        "maximum_gradient_flow_energy_increase": maximum_increase,
        "selling_gradient_edge_count": int(len(first)),
    }


def continuous_tangent_source_geometry_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
    quantile_count: int = 32,
    maximum_source_transports: int = 32,
    remetricize: bool = False,
    joint_jet_action: bool = False,
    joint_jet_law_action: bool = False,
    joint_bundle_action: bool = False,
    strict_joint_bundle_action: bool = False,
    line_search: bool = False,
    local_joint_transport: bool = False,
    causal_support_joint_transport: bool = False,
    strict_joint_graph_gradient_flow: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Transport source identity before admitting vertical jet information."""
    image = _validate(observation)
    ceiling = int(maximum_source_transports)
    if ceiling < 1:
        raise ValueError("maximum source transports must be positive")
    if sum(map(bool, (
        joint_jet_action,
        joint_jet_law_action,
        joint_bundle_action,
        strict_joint_bundle_action,
    ))) > 1:
        raise ValueError("select at most one joint source-transport action")
    if local_joint_transport and not strict_joint_bundle_action:
        raise ValueError(
            "local joint transport requires the strict joint bundle action")
    if causal_support_joint_transport and not strict_joint_bundle_action:
        raise ValueError(
            "causal support transport requires the strict joint bundle action")
    if strict_joint_graph_gradient_flow and not strict_joint_bundle_action:
        raise ValueError(
            "joint graph gradient flow requires the strict joint bundle action")
    if sum(map(bool, (
        local_joint_transport,
        causal_support_joint_transport,
        strict_joint_graph_gradient_flow,
    ))) > 1:
        raise ValueError("select one strict joint transport realization")
    signal, signal_diagnostic = continuous_tangent_signal_population_2d(
        image, angular_count=angular_count)
    quantiles = weighted_empirical_quantiles(
        signal["prediction"], signal["mass"], quantile_count)
    horizontal = predictive_horizontal_wasserstein_geometry(quantiles)
    horizontal_operator, horizontal_operator_diagnostic = source_measure_operator(
        horizontal["metric_xx"],
        horizontal["metric_xy"],
        horizontal["metric_yy"],
    )
    _influence, lineage = _source_influence_and_lineage(
        signal["mass"],
        signal["source_identity"],
        signal["source_coefficient"],
    )
    witness, witness_diagnostic = crossfit_characteristic_population_2d(image)
    strict_gradient_x, strict_gradient_y, strict_jet_diagnostic = (
        continuous_tangent_jet_field_2d(witness))
    held_out_signal = np.sum(
        signal["mass"] * signal["prediction"], axis=-1)
    held_out_residual = image - held_out_signal
    prior_population = {
        "mass": signal["mass"],
        "directional_derivative": signal["directional_derivative"],
        "tangent": signal["tangent"],
    }
    gradient_x, gradient_y, jet_diagnostic = continuous_tangent_jet_field_2d(
        prior_population)
    initial_vertical = predictive_lineage_jet_geometry(
        lineage.reshape(image.shape + (image.size,)),
        gradient_x,
        gradient_y,
        quantile_count=quantile_count,
    )
    initial_fused = combine_information_geometries(
        horizontal, initial_vertical)
    initial_prolongation = predictive_lineage_prolongation_geometry(
        lineage.reshape(image.shape + (image.size,)), held_out_signal)
    initial_prolongation_fused = combine_information_geometries(
        horizontal, initial_prolongation)
    joint_particles, joint_particle_diagnostic = (
        continuous_tangent_jet_particles_2d(signal))
    joint_query_residual = image[..., None] - joint_particles["signal"]
    strict_query_residual = image[..., None] - witness["prediction"]
    strict_query_gradient_x = np.broadcast_to(
        strict_gradient_x[..., None], witness["prediction"].shape)
    strict_query_gradient_y = np.broadcast_to(
        strict_gradient_y[..., None], witness["prediction"].shape)
    if (
        local_joint_transport
        or causal_support_joint_transport
        or strict_joint_graph_gradient_flow
    ):
        local_source_distance, local_cross_cost = _joint_bundle_costs(
            held_out_residual,
            gradient_x,
            gradient_y,
            strict_query_residual,
            strict_query_gradient_x,
            strict_query_gradient_y,
            witness["mass"],
        )
    support_diagnostic: dict[str, Any] | None = None
    support_labels: np.ndarray | None = None
    if causal_support_joint_transport:
        from port_needed.continuous_eikonal_transport import (
            continuous_first_partition_prepared,
            prepare_continuous_metric,
        )
        from port_needed.density_population import emit_density_population

        support_centers, support_population = emit_density_population(
            {
                "measure": initial_fused["measure"],
                "implied_cells": initial_fused["implied_support"],
            },
            safety_cells=image.size,
        )
        support_metric = prepare_continuous_metric(
            initial_fused["metric_xx"],
            initial_fused["metric_xy"],
            initial_fused["metric_yy"],
            consistency_limit=np.finfo(float).max,
        )
        support_forest = continuous_first_partition_prepared(
            support_centers, support_metric, compact=True)
        support_labels = np.asarray(
            support_forest["labels"], dtype=np.int64)
        support_diagnostic = {
            "population": support_population,
            "center_count": int(len(support_centers)),
            "front_pushes": int(support_forest["front_pushes"]),
            "front_maximum_heap": int(
                support_forest["front_maximum_heap"]),
            "geometry": "initial fused predictive information geometry",
        }

    def risk(source_law: np.ndarray) -> tuple[float, dict[str, float]]:
        score, diagnostic = _lineage_residual_crps(
            image,
            witness,
            held_out_residual[..., None],
            source_law,
        )
        residual_action = float(np.mean(score))
        if strict_joint_bundle_action:
            bundle_action, bundle_risk = _joint_bundle_energy_score(
                source_law,
                held_out_residual,
                gradient_x,
                gradient_y,
                strict_query_residual,
                strict_query_gradient_x,
                strict_query_gradient_y,
                witness["mass"],
            )
            action = bundle_action
            return action, {
                **diagnostic,
                "mean_held_out_residual_crps": residual_action,
                **bundle_risk,
                "selected_projective_jet_action": 0.0,
                "joint_transport_action": action,
                "joint_bundle_witness": "strict odd/even direction-lane law",
            }
        if joint_bundle_action:
            bundle_action, bundle_risk = _joint_bundle_energy_score(
                source_law,
                held_out_residual,
                gradient_x,
                gradient_y,
                joint_query_residual,
                joint_particles["gradient_x"],
                joint_particles["gradient_y"],
                joint_particles["mass"],
            )
            action = bundle_action
            return action, {
                **diagnostic,
                "mean_held_out_residual_crps": residual_action,
                **bundle_risk,
                "selected_projective_jet_action": 0.0,
                "joint_transport_action": action,
            }
        if joint_jet_law_action:
            jet_action, jet_risk = _projective_jet_law_crps(
                source_law,
                gradient_x,
                gradient_y,
                signal["mass"],
                signal["directional_derivative"],
                signal["tangent"],
            )
            jet_action_key = "mean_projective_jet_law_crps"
        else:
            jet_action, jet_risk = _projective_jet_crps(
                source_law,
                gradient_x,
                gradient_y,
                signal["tangent"],
            )
            jet_action_key = "mean_projective_jet_crps"
        use_jet = joint_jet_action or joint_jet_law_action
        action = residual_action + (jet_action if use_jet else 0.0)
        return action, {
            **diagnostic,
            "mean_held_out_residual_crps": residual_action,
            **jet_risk,
            "selected_projective_jet_action": jet_risk[jet_action_key],
            "joint_transport_action": action,
        }

    state = lineage
    action, risk_diagnostic = risk(state)
    records = []
    equilibrium = False
    operator_diagnostic = horizontal_operator_diagnostic
    minimum_step = np.sqrt(np.finfo(float).eps)
    for transport in range(ceiling):
        if remetricize:
            current_vertical = predictive_lineage_jet_geometry(
                state.reshape(image.shape + (image.size,)),
                gradient_x,
                gradient_y,
                quantile_count=quantile_count,
            )
            current_geometry = combine_information_geometries(
                horizontal, current_vertical)
            operator, operator_diagnostic = source_measure_operator(
                current_geometry["metric_xx"],
                current_geometry["metric_xy"],
                current_geometry["metric_yy"],
            )
        else:
            current_geometry = horizontal
            operator = horizontal_operator
        transported = _exclude_target_identity(operator @ state)
        numerical = np.finfo(float).eps * max(action, 1.0)
        if strict_joint_graph_gradient_flow:
            candidate, gradient_flow_diagnostic = (
                _joint_bundle_graph_gradient_flow(
                    state,
                    operator,
                    operator_diagnostic["stationary_mass"],
                    local_source_distance,
                    local_cross_cost,
                )
            )
            candidate_action, candidate_diagnostic = risk(candidate)
            transported = candidate
            full_step_action = candidate_action
            step = gradient_flow_diagnostic[
                "mean_gradient_flow_total_variation"]
        elif causal_support_joint_transport:
            candidate, local_step, local_step_diagnostic = (
                _causal_support_joint_transport(
                    state,
                    transported,
                    support_labels,
                    local_source_distance,
                    local_cross_cost,
                )
            )
            candidate_action, candidate_diagnostic = risk(candidate)
            full_step_action = float(np.mean(_joint_bundle_score_field(
                transported, local_source_distance, local_cross_cost)))
            step = float(np.mean(local_step))
        elif local_joint_transport:
            candidate, local_step, local_step_diagnostic = (
                _analytic_local_joint_transport(
                    state,
                    transported,
                    local_source_distance,
                    local_cross_cost,
                )
            )
            candidate_action, candidate_diagnostic = risk(candidate)
            full_step_action = float(np.mean(_joint_bundle_score_field(
                transported, local_source_distance, local_cross_cost)))
            step = float(np.mean(local_step))
        else:
            step = 1.0
            candidate = transported
            candidate_action, candidate_diagnostic = risk(candidate)
            full_step_action = candidate_action
            while line_search and candidate_action >= action - numerical:
                step *= 0.5
                if step < minimum_step:
                    break
                candidate = (1.0 - step) * state + step * transported
                candidate_action, candidate_diagnostic = risk(candidate)
        accepted = candidate_action < action - numerical
        records.append({
            "transport": transport,
            "accepted": accepted,
            "transport_action_before": action,
            "transport_action_after": candidate_action,
            "full_step_transport_action": full_step_action,
            "accepted_step": step if accepted else 0.0,
            "positive_local_step_fraction": (
                float(np.mean(local_step > 0.0))
                if (local_joint_transport or causal_support_joint_transport)
                else None
            ),
            "maximum_local_step": (
                float(np.max(local_step))
                if (local_joint_transport or causal_support_joint_transport)
                else None
            ),
            "maximum_local_energy_increase": (
                local_step_diagnostic["maximum_local_energy_increase"]
                if local_joint_transport else None
            ),
            "mean_local_energy_decrease": (
                local_step_diagnostic["mean_local_energy_decrease"]
                if local_joint_transport else None
            ),
            "maximum_support_energy_increase": (
                local_step_diagnostic["maximum_support_energy_increase"]
                if causal_support_joint_transport else None
            ),
            "maximum_point_energy_increase": (
                local_step_diagnostic["maximum_point_energy_increase"]
                if causal_support_joint_transport else None
            ),
            "positive_support_step_fraction": (
                local_step_diagnostic["positive_support_step_fraction"]
                if causal_support_joint_transport else None
            ),
            "positive_gradient_flow_fraction": (
                gradient_flow_diagnostic["positive_gradient_flow_fraction"]
                if strict_joint_graph_gradient_flow else None
            ),
            "mean_gradient_flow_total_variation": (
                gradient_flow_diagnostic[
                    "mean_gradient_flow_total_variation"]
                if strict_joint_graph_gradient_flow else None
            ),
            "maximum_gradient_flow_energy_increase": (
                gradient_flow_diagnostic[
                    "maximum_gradient_flow_energy_increase"]
                if strict_joint_graph_gradient_flow else None
            ),
            "held_out_residual_crps_before": risk_diagnostic[
                "mean_held_out_residual_crps"],
            "held_out_residual_crps_after": candidate_diagnostic[
                "mean_held_out_residual_crps"],
            "projective_jet_crps_before": risk_diagnostic[
                "selected_projective_jet_action"],
            "projective_jet_crps_after": candidate_diagnostic[
                "selected_projective_jet_action"],
            "joint_bundle_energy_before": risk_diagnostic.get(
                "mean_joint_bundle_energy_score", 0.0),
            "joint_bundle_energy_after": candidate_diagnostic.get(
                "mean_joint_bundle_energy_score", 0.0),
            "mean_collision_population_after": float(np.mean(
                1.0 / np.sum(candidate * candidate, axis=1))),
            "transport_geometry_implied_support": float(
                current_geometry["implied_support"]),
            **candidate_diagnostic,
        })
        if not accepted:
            equilibrium = True
            break
        state = candidate
        action = candidate_action
        risk_diagnostic = candidate_diagnostic

    vertical = predictive_lineage_jet_geometry(
        state.reshape(image.shape + (image.size,)),
        gradient_x,
        gradient_y,
        quantile_count=quantile_count,
    )
    fused = combine_information_geometries(horizontal, vertical)
    prolongation = predictive_lineage_prolongation_geometry(
        state.reshape(image.shape + (image.size,)), held_out_signal)
    prolongation_fused = combine_information_geometries(
        horizontal, prolongation)
    ceiling_hit = len(records) == ceiling and not equilibrium
    return {
        "horizontal": horizontal,
        "initial_vertical": initial_vertical,
        "vertical": vertical,
        "initial_fused": initial_fused,
        "fused": fused,
        "initial_prolongation": initial_prolongation,
        "prolongation": prolongation,
        "initial_prolongation_fused": initial_prolongation_fused,
        "prolongation_fused": prolongation_fused,
        "lineage": state,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
    }, {
        "status": (
            "continuous source geometry at held-out CRPS equilibrium"
            if equilibrium else "source transport ceiling reached; unresolved"
        ),
        "theory_status": (
            "held-out horizontal metric -> Selling source transport -> "
            "vertical jet volume"
            + (
                " with per-step fused remetricization"
                if remetricize else "; fixed-point remetricization pending"
            )
        ),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "remetricize": bool(remetricize),
        "joint_jet_action": bool(joint_jet_action),
        "joint_jet_law_action": bool(joint_jet_law_action),
        "joint_bundle_action": bool(joint_bundle_action),
        "strict_joint_bundle_action": bool(strict_joint_bundle_action),
        "line_search": bool(line_search),
        "local_joint_transport": bool(local_joint_transport),
        "causal_support_joint_transport": bool(
            causal_support_joint_transport),
        "strict_joint_graph_gradient_flow": bool(
            strict_joint_graph_gradient_flow),
        "causal_support": support_diagnostic,
        "minimum_line_search_step": float(minimum_step),
        "accepted_source_transports": int(sum(
            record["accepted"] for record in records)),
        "source_transport_ceiling_hit": ceiling_hit,
        "terminal_transport_action": action,
        "terminal_held_out_residual_crps": risk_diagnostic[
            "mean_held_out_residual_crps"],
        "terminal_projective_jet_crps": risk_diagnostic[
            "selected_projective_jet_action"],
        "source_transports": records,
        "maximum_target_self_lineage": float(np.max(np.abs(
            np.diag(state)))),
        "lineage_row_mass_maximum_error": float(np.max(np.abs(
            np.sum(state, axis=1) - 1.0))),
        "signal": signal_diagnostic,
        "witness": witness_diagnostic,
        "source_operator": operator_diagnostic,
        "horizontal_source_operator": horizontal_operator_diagnostic,
        "terminal_risk": risk_diagnostic,
        "jet": jet_diagnostic,
        "joint_jet_particles": joint_particle_diagnostic,
        "strict_joint_jet": strict_jet_diagnostic,
        "unresolved": [
            "Selling reduced stencil changes discretely with the metric",
            "source ancestry is dense quadratic research state",
            *([] if remetricize else [
                "fused geometry has not yet been remarched to fixed point"
            ]),
        ],
    }
