"""Scale-state relation transport for the one-dimensional foundation.

This is an experimental denoiser, not a promoted final law.  Unlike the first
affine-relation simmer, it never chooses a maximum physical relation horizon:
every lag from one grid edge to half of the interval participates.  Lag is a
state coordinate with the dilation-invariant measure ``ds / s``.

At lag ``s`` three first-jet characteristics compete: opposite observations
transport to their common midpoint, while the two one-sided chords extrapolate
from the left and right.  Their path variation is transport dispersion, and
their cross-predictive error is accumulated over the interval that each
relation claims to describe.  Reciprocal total action is conductance; the
normalized conductance measure is marginalized only at readout.  These are
topological paths, not branches selected by a named corruption.

One pass deliberately removes unpredictable material.  Predictable structure
left in the residual is then transported again.  Continuation is not a chosen
round count: residual--prediction covariance is debiased by its finite-sample
variance, and only positive excess covariance may move mass.  The process is
at equilibrium when that excess vanishes.  A continuation ceiling remains a
reported numerical guard and is a failed run if reached.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable
import math

import numpy as np
from scipy import ndimage, optimize, special


@dataclass(frozen=True)
class CrossPredictiveResolution:
    """Numerical safety controls that do not select physical relation scale."""

    maximum_continuations: int = 128


def _validate_line(observation: np.ndarray) -> np.ndarray:
    line = np.asarray(observation, dtype=np.float64)
    if line.ndim != 1 or line.size < 8:
        raise ValueError("cross-predictive transport expects at least 8 samples")
    if not np.all(np.isfinite(line)):
        raise ValueError("cross-predictive transport requires finite samples")
    return line


def _representation_floor(line: np.ndarray) -> float:
    magnitude = max(float(np.max(np.abs(line))), float(np.ptp(line)))
    if magnitude == 0.0:
        return np.finfo(float).tiny
    return math.sqrt(np.finfo(float).eps) * magnitude


def relation_scale_transport(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize the full topological scale measure into one prediction.

    Each midpoint prediction excludes the observation at its own target.  Its
    conductance field is not yet strictly pointwise J-invariant because the
    local cross-predictive action contains the target's validation residual.
    This leakage is exposed in diagnostics and must be removed by the future
    joint particle solver rather than hidden behind a parity setting.
    """
    forms, diagnostic = relation_scale_readout_forms(observation)
    return forms["mean"], diagnostic


def relation_scale_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Expose mean, W1, and branch readouts of one unchanged scale law."""
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 2
    if float(np.ptp(line)) == 0.0:
        lags = np.arange(1, maximum_lag + 1, dtype=np.float64)
        haar = 1.0 / lags
        haar /= np.sum(haar)
        return {
            "mean": line.copy(),
            "median": line.copy(),
            "maximum_branch": line.copy(),
        }, {
            "state": "full-scale first-jet characteristic conductance measure",
            "scale_measure": "discrete ds/s over every topological lag",
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "scale_count": int(maximum_lag),
            "characteristics_per_scale": 3,
            "characteristic_count": int(3 * maximum_lag),
            "mean_barycentric_lag": float(np.dot(haar, lags)),
            "mean_geometric_lag": float(np.exp(np.dot(haar, np.log(lags)))),
            "predictive_action_mean": 0.0,
            "endpoint_dispersion_mean": 0.0,
            "center_prediction_pointwise_j_invariant": True,
            "conductance_pointwise_j_invariant": False,
        }
    padded = np.pad(line, 2 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 2 * maximum_lag
    floor = _representation_floor(line)

    prediction_fields = []
    conductance_fields = []
    lag_fields = []
    action_means = []
    dispersion_means = []

    # This is a Riemann sum on the complete topological interval.  There is no
    # selected lag catalogue and no winning scale.
    for lag in range(1, maximum_lag + 1):
        left_one = padded[index - lag]
        right_one = padded[index + lag]
        left_two = padded[index - 2 * lag]
        right_two = padded[index + 2 * lag]
        characteristics = (
            (0.5 * (left_one + right_one),
             0.5 * np.abs(right_one - left_one)),
            (2.0 * left_one - left_two, np.abs(left_one - left_two)),
            (2.0 * right_one - right_two, np.abs(right_one - right_two)),
        )
        for prediction, dispersion in characteristics:
            point_error = np.abs(prediction - line)
            predictive_action = ndimage.uniform_filter1d(
                point_error, size=2 * lag + 1, mode="reflect")

            # Predictive error and characteristic variation are consecutive
            # pieces of one path action, so their costs add.  Reciprocal action
            # is conductance.  1/lag is the discrete Haar measure ds/s.
            total_action = predictive_action + dispersion
            conductance = 1.0 / (lag * np.maximum(total_action, floor))
            prediction_fields.append(prediction)
            conductance_fields.append(conductance)
            lag_fields.append(float(lag))
            action_means.append(float(np.mean(predictive_action)))
            dispersion_means.append(float(np.mean(dispersion)))

    prediction = np.stack(prediction_fields, axis=-1)
    conductance = np.stack(conductance_fields, axis=-1)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    mean = np.sum(mass * prediction, axis=-1)
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(mass, order, axis=-1)
    median_index = np.argmax(
        np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
    median = np.take_along_axis(
        ordered_prediction, median_index[:, None], axis=-1)[:, 0]
    maximum_branch = np.take_along_axis(
        prediction,
        np.argmax(mass, axis=-1)[:, None],
        axis=-1,
    )[:, 0]
    lag_coordinate = np.asarray(lag_fields, dtype=np.float64)
    mean_scale = mass @ lag_coordinate
    geometric_scale = np.exp(mass @ np.log(lag_coordinate))
    return {
        "mean": mean,
        "median": median,
        "maximum_branch": maximum_branch,
    }, {
        "state": "full-scale first-jet characteristic conductance measure",
        "scale_measure": "discrete ds/s over every topological lag",
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "scale_count": int(maximum_lag),
        "characteristics_per_scale": 3,
        "characteristic_count": int(3 * maximum_lag),
        "mean_barycentric_lag": float(np.mean(mean_scale)),
        "mean_geometric_lag": float(np.mean(geometric_scale)),
        "predictive_action_mean": float(np.mean(action_means)),
        "endpoint_dispersion_mean": float(np.mean(dispersion_means)),
        "center_prediction_pointwise_j_invariant": True,
        "conductance_pointwise_j_invariant": False,
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
    }


def relation_scale_particle_law(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the complete aligned lag/path law before scalar readout."""
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 2
    branch_count = 3 * maximum_lag
    target_identity = np.arange(samples, dtype=np.int64)
    period = 2 * (samples - 1)

    def reflected_identity(raw_index: np.ndarray) -> np.ndarray:
        folded = np.mod(raw_index, period)
        return np.where(folded < samples, folded, period - folded)

    source_fields = []
    for lag_value in range(1, maximum_lag + 1):
        left_one_identity = reflected_identity(target_identity - lag_value)
        right_one_identity = reflected_identity(target_identity + lag_value)
        left_two_identity = reflected_identity(
            target_identity - 2 * lag_value)
        right_two_identity = reflected_identity(
            target_identity + 2 * lag_value)
        source_fields.extend((
            np.stack((left_one_identity, right_one_identity), axis=-1),
            np.stack((left_one_identity, left_two_identity), axis=-1),
            np.stack((right_one_identity, right_two_identity), axis=-1),
        ))
    source_identity = np.stack(source_fields, axis=1)
    if float(np.ptp(line)) == 0.0:
        lag = np.repeat(
            np.arange(1, maximum_lag + 1, dtype=np.float64), 3)
        base = 1.0 / lag
        mass = np.broadcast_to(
            base / np.sum(base), (samples, branch_count)).copy()
        return {
            "prediction": np.broadcast_to(
                line[:, None], (samples, branch_count)).copy(),
            "jet": np.zeros((samples, branch_count), dtype=np.float64),
            "total_action": np.zeros(
                (samples, branch_count), dtype=np.float64),
            "mass": mass,
            "lag": lag,
            "path_family": np.tile(np.arange(3), maximum_lag),
            "source_identity": source_identity,
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(branch_count),
            "mean_collision_population": float(
                1.0 / np.sum(mass[0] * mass[0])),
        }
    padded = np.pad(line, 2 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 2 * maximum_lag
    floor = _representation_floor(line)
    prediction_fields = []
    jet_fields = []
    action_fields = []
    conductance_fields = []
    lag_fields = []
    family_fields = []
    for lag in range(1, maximum_lag + 1):
        left_one = padded[index - lag]
        right_one = padded[index + lag]
        left_two = padded[index - 2 * lag]
        right_two = padded[index + 2 * lag]
        characteristics = (
            (
                0.5 * (left_one + right_one),
                0.5 * np.abs(right_one - left_one),
                (right_one - left_one) / (2.0 * lag),
            ),
            (
                2.0 * left_one - left_two,
                np.abs(left_one - left_two),
                (left_one - left_two) / lag,
            ),
            (
                2.0 * right_one - right_two,
                np.abs(right_one - right_two),
                (right_two - right_one) / lag,
            ),
        )
        for family, (prediction, dispersion, jet) in enumerate(characteristics):
            point_error = np.abs(prediction - line)
            predictive_action = ndimage.uniform_filter1d(
                point_error, size=2 * lag + 1, mode="reflect")
            total_action = predictive_action + dispersion
            prediction_fields.append(prediction)
            jet_fields.append(jet)
            action_fields.append(total_action)
            conductance_fields.append(
                1.0 / (lag * np.maximum(total_action, floor)))
            lag_fields.append(float(lag))
            family_fields.append(family)
    prediction = np.stack(prediction_fields, axis=-1)
    conductance = np.stack(conductance_fields, axis=-1)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "jet": np.stack(jet_fields, axis=-1),
        "total_action": np.stack(action_fields, axis=-1),
        "mass": mass,
        "lag": np.asarray(lag_fields, dtype=np.float64),
        "path_family": np.asarray(family_fields, dtype=np.int64),
        "source_identity": source_identity,
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(branch_count),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
    }


def distinct_ancestry_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score each relation particle only against disjoint source ancestry."""
    line = _validate_line(observation)
    law, _diagnostic = relation_scale_particle_law(line)
    prediction = law["prediction"]
    jet = law["jet"]
    source_identity = law["source_identity"]
    samples, branches = prediction.shape
    lag = law["lag"]
    haar = 1.0 / lag
    haar /= np.sum(haar)
    target = np.arange(samples, dtype=np.int64)
    validity = (
        np.all(source_identity != target[:, None, None], axis=2)
        & (source_identity[:, :, 0] != source_identity[:, :, 1])
    )
    reference = validity * haar[None, :]
    missing = np.sum(reference, axis=1) <= np.finfo(float).tiny
    reference[missing] = haar
    reference /= np.sum(reference, axis=1, keepdims=True)
    if float(np.ptp(line)) == 0.0:
        result = dict(law)
        result.update({
            "total_action": np.zeros_like(prediction),
            "mass": reference.copy(),
            "reference_mass": reference,
        })
        return result, {
            "state": "independent-ancestry value-jet predictive law",
            "target_value_enters_interior_action": False,
            "mean_disjoint_population": float(branches),
            "valid_source_fraction": float(np.mean(validity)),
            "physical_parameters": "none",
        }

    floor = _representation_floor(line)
    action = np.empty_like(prediction)
    disjoint_populations = []
    for index in range(samples):
        source = source_identity[index]
        first_in_other = (
            (source[:, None, 0] == source[None, :, 0])
            | (source[:, None, 0] == source[None, :, 1])
        )
        second_in_other = (
            (source[:, None, 1] == source[None, :, 0])
            | (source[:, None, 1] == source[None, :, 1])
        )
        disjoint = ~(first_in_other | second_in_other)
        state = np.column_stack((prediction[index], jet[index]))
        center = reference[index] @ state
        centered = state - center
        covariance = (
            centered * reference[index, :, None]
        ).T @ centered
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        precision_eigenvalue = 1.0 / np.maximum(
            eigenvalue, floor * floor)
        precision_eigenvalue /= np.sqrt(np.prod(precision_eigenvalue))
        precision = (
            eigenvector * precision_eigenvalue[None, :]
        ) @ eigenvector.T
        defect = state[:, None, :] - state[None, :, :]
        distance = np.sqrt(np.maximum(np.einsum(
            "ija,ab,ijb->ij", defect, precision, defect), 0.0))
        witness = disjoint * reference[index][None, :]
        witness_total = np.sum(witness, axis=1, keepdims=True)
        no_witness = witness_total[:, 0] <= np.finfo(float).tiny
        if np.any(no_witness):
            witness[no_witness] = reference[index]
            witness_total[no_witness] = np.sum(
                witness[no_witness], axis=1, keepdims=True)
        witness /= witness_total
        action[index] = np.sum(witness * distance, axis=1)
        disjoint_populations.append(float(np.mean(
            1.0 / np.sum(witness * witness, axis=1))))
    conductance = reference / np.maximum(action, floor)
    mass = conductance / np.sum(conductance, axis=1, keepdims=True)
    result = dict(law)
    result.update({
        "total_action": action,
        "mass": mass,
        "reference_mass": reference,
        "ancestry_action": action,
    })
    return result, {
        "state": "independent-ancestry value-jet predictive law",
        "action": (
            "conditional expected determinant-one value-jet distance "
            "against particles with disjoint source identity"
        ),
        "target_value_enters_interior_action": False,
        "mean_disjoint_population": float(np.mean(disjoint_populations)),
        "valid_source_fraction": float(np.mean(validity)),
        "boundary_fallback_fraction": float(np.mean(missing)),
        "mean_ancestry_action": float(np.mean(action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=1))),
        "physical_parameters": "none",
    }


def _reflected_scale_reference_1d(
    samples: int,
    maximum_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Haar scale law restricted to four distinct source points."""
    lag = np.arange(1, maximum_lag + 1, dtype=np.float64)
    haar = 1.0 / lag
    haar /= np.sum(haar)
    target_identity = np.arange(samples)
    period = 2 * (samples - 1)

    def reflected_identity(raw_index: np.ndarray) -> np.ndarray:
        folded = np.mod(raw_index, period)
        return np.where(folded < samples, folded, period - folded)

    validity_fields = []
    for scale in range(1, maximum_lag + 1):
        source_identity = np.stack((
            reflected_identity(target_identity - scale),
            reflected_identity(target_identity + scale),
            reflected_identity(target_identity - 2 * scale),
            reflected_identity(target_identity + 2 * scale),
        ), axis=-1)
        excludes_target = np.all(
            source_identity != target_identity[:, None], axis=-1)
        ordered_identity = np.sort(source_identity, axis=-1)
        distinct_sources = np.all(
            np.diff(ordered_identity, axis=-1) != 0, axis=-1)
        validity_fields.append(excludes_target & distinct_sources)
    validity = np.stack(validity_fields, axis=-1)
    reference_mass = validity * haar[None, :]
    reference_total = np.sum(reference_mass, axis=-1, keepdims=True)
    boundary_fallback = reference_total[:, 0] <= np.finfo(float).tiny
    reference_mass[boundary_fallback] = haar
    reference_mass /= np.sum(reference_mass, axis=-1, keepdims=True)
    return reference_mass, validity, boundary_fallback


def _w1_population_action_1d(
    prediction: np.ndarray,
    reference_mass: np.ndarray,
) -> np.ndarray:
    """Evaluate each particle's exact W1 potential against its population."""
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_reference = np.take_along_axis(
        reference_mass, order, axis=-1)
    cumulative_mass = np.cumsum(ordered_reference, axis=-1)
    cumulative_moment = np.cumsum(
        ordered_reference * ordered_prediction, axis=-1)
    total_moment = cumulative_moment[:, -1:]
    population_action_ordered = (
        ordered_prediction * cumulative_mass - cumulative_moment
        + (total_moment - cumulative_moment)
        - ordered_prediction * (1.0 - cumulative_mass)
    )
    return np.take_along_axis(
        population_action_ordered,
        np.argsort(order, axis=-1, kind="stable"),
        axis=-1,
    )


def _minimum_information_coverage_projection(
    baseline_mass: np.ndarray,
    contracted_mass: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restore one transported moment by the unique exponential I-projection."""
    baseline = np.asarray(baseline_mass, dtype=np.float64)
    contracted = np.asarray(contracted_mass, dtype=np.float64)
    moment_coordinate = np.asarray(coordinate, dtype=np.float64)
    if baseline.shape != contracted.shape or baseline.shape != moment_coordinate.shape:
        raise ValueError("coverage laws and coordinate must align")
    projected = contracted.copy()
    baseline_moment = np.sum(baseline * moment_coordinate, axis=1)
    contracted_moment = np.sum(contracted * moment_coordinate, axis=1)
    for index in range(baseline.shape[0]):
        target = baseline_moment[index]
        if contracted_moment[index] >= target:
            continue
        positive = contracted[index] > 0.0
        local_coordinate = moment_coordinate[index, positive]
        if local_coordinate.size == 0 or float(np.ptp(local_coordinate)) == 0.0:
            continue
        log_base = np.log(contracted[index, positive])

        def tilted(parameter: float) -> tuple[np.ndarray, float]:
            log_weight = log_base + parameter * local_coordinate
            log_weight -= np.max(log_weight)
            weight = np.exp(log_weight)
            weight /= np.sum(weight)
            return weight, float(np.dot(weight, local_coordinate))

        lower = 0.0
        upper = 1.0
        _weight, upper_moment = tilted(upper)
        for _ in range(64):
            if upper_moment >= target:
                break
            upper *= 2.0
            _weight, upper_moment = tilted(upper)
        for _ in range(64):
            middle = 0.5 * (lower + upper)
            _weight, middle_moment = tilted(middle)
            if middle_moment < target:
                lower = middle
            else:
                upper = middle
        weight, _moment = tilted(upper)
        projected[index] = 0.0
        projected[index, positive] = weight
    restored_moment = np.sum(projected * moment_coordinate, axis=1)
    return projected, baseline_moment, contracted_moment, restored_moment


def paired_side_collision_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Form full-scale particles from disjoint left/right affine arrivals.

    Each scale contributes one particle whose two parents lie strictly on
    opposite sides of the target in the interior. Their midpoint supplies the
    latent value and jet; their disagreement is retained in the action before
    any spatial lineage or scalar projection. The target observation enters
    only through the exact residual coordinate used later by the joint metric.
    """
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 2
    lag = np.arange(1, maximum_lag + 1, dtype=np.float64)
    haar = 1.0 / lag
    haar /= np.sum(haar)
    if float(np.ptp(line)) == 0.0:
        prediction = np.broadcast_to(
            line[:, None], (samples, maximum_lag)).copy()
        mass = np.broadcast_to(haar, prediction.shape).copy()
        return {
            "prediction": prediction,
            "jet": np.zeros_like(prediction),
            "total_action": np.zeros_like(prediction),
            "mass": mass,
            "lag": lag,
            "side_disagreement": np.zeros_like(prediction),
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(maximum_lag),
            "target_value_enters_interior_action": False,
            "mean_side_disagreement": 0.0,
        }

    padded = np.pad(line, 2 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 2 * maximum_lag
    floor = _representation_floor(line)
    prediction_fields = []
    jet_fields = []
    action_fields = []
    disagreement_fields = []
    for scale in range(1, maximum_lag + 1):
        left_one = padded[index - scale]
        right_one = padded[index + scale]
        left_two = padded[index - 2 * scale]
        right_two = padded[index + 2 * scale]
        left_prediction = 2.0 * left_one - left_two
        right_prediction = 2.0 * right_one - right_two
        left_jet = (left_one - left_two) / scale
        right_jet = (right_two - right_one) / scale
        prediction_fields.append(
            0.5 * (left_prediction + right_prediction))
        jet_fields.append(0.5 * (left_jet + right_jet))
        disagreement = np.abs(left_prediction - right_prediction)
        path_variation = (
            np.abs(left_one - left_two)
            + np.abs(right_two - right_one)
        )
        disagreement_fields.append(disagreement)
        action_fields.append(disagreement + path_variation)

    prediction = np.stack(prediction_fields, axis=-1)
    jet = np.stack(jet_fields, axis=-1)
    local_action = np.stack(action_fields, axis=-1)
    disagreement = np.stack(disagreement_fields, axis=-1)
    reference_mass, validity, boundary_fallback = (
        _reflected_scale_reference_1d(samples, maximum_lag))

    # The arrival population is the exact W1 potential of the complete scale
    # law. This prevents an isolated zero-action scale from monopolizing mass
    # without introducing a kernel width or accepted scale catalogue.
    population_action = _w1_population_action_1d(
        prediction, reference_mass)
    total_action = local_action + population_action
    conductance = reference_mass / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "jet": jet,
        "total_action": total_action,
        "mass": mass,
        "lag": lag,
        "reference_mass": reference_mass,
        "side_disagreement": disagreement,
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(maximum_lag),
        "target_value_enters_interior_action": False,
        "valid_source_fraction": float(np.mean(validity)),
        "boundary_fallback_fraction": float(np.mean(boundary_fallback)),
        "mean_side_disagreement": float(np.mean(disagreement)),
        "mean_population_action": float(np.mean(population_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "boundary_closure": "reflection; not yet continuum-normalized",
    }


def nested_midpoint_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Form affine-exact particles from nested, disjoint secant shells.

    At scale ``s`` the latent value is the midpoint of the two observations at
    ``x-s`` and ``x+s`` and the jet is their secant.  The conjugate shell at
    ``2s`` supplies the same two coordinates.  Their value--jet discrepancy is
    zero on every affine signal and contains no target sample; it is therefore
    a transported consistency action rather than a fitted noise rule.
    """
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 2
    lag = np.arange(1, maximum_lag + 1, dtype=np.float64)
    haar = 1.0 / lag
    haar /= np.sum(haar)
    if float(np.ptp(line)) == 0.0:
        prediction = np.broadcast_to(
            line[:, None], (samples, maximum_lag)).copy()
        mass = np.broadcast_to(haar, prediction.shape).copy()
        return {
            "prediction": prediction,
            "jet": np.zeros_like(prediction),
            "total_action": np.zeros_like(prediction),
            "mass": mass,
            "lag": lag,
            "value_shell_defect": np.zeros_like(prediction),
            "jet_shell_defect": np.zeros_like(prediction),
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(maximum_lag),
            "target_value_enters_interior_action": False,
            "affine_interior_action": 0.0,
        }

    padded = np.pad(line, 2 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 2 * maximum_lag
    floor = _representation_floor(line)
    prediction_fields = []
    jet_fields = []
    value_defect_fields = []
    jet_defect_fields = []
    action_fields = []
    for scale in range(1, maximum_lag + 1):
        left = padded[index - scale]
        right = padded[index + scale]
        outer_left = padded[index - 2 * scale]
        outer_right = padded[index + 2 * scale]
        prediction = 0.5 * (left + right)
        jet = (right - left) / (2.0 * scale)
        outer_prediction = 0.5 * (outer_left + outer_right)
        outer_jet = (outer_right - outer_left) / (4.0 * scale)
        value_defect = np.abs(prediction - outer_prediction)
        jet_defect = scale * np.abs(jet - outer_jet)
        prediction_fields.append(prediction)
        jet_fields.append(jet)
        value_defect_fields.append(value_defect)
        jet_defect_fields.append(jet_defect)
        action_fields.append(value_defect + jet_defect)

    prediction = np.stack(prediction_fields, axis=-1)
    jet = np.stack(jet_fields, axis=-1)
    local_action = np.stack(action_fields, axis=-1)
    value_defect = np.stack(value_defect_fields, axis=-1)
    jet_defect = np.stack(jet_defect_fields, axis=-1)
    reference_mass, validity, boundary_fallback = (
        _reflected_scale_reference_1d(samples, maximum_lag))
    population_action = _w1_population_action_1d(
        prediction, reference_mass)
    total_action = local_action + population_action
    conductance = reference_mass / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "jet": jet,
        "total_action": total_action,
        "mass": mass,
        "lag": lag,
        "reference_mass": reference_mass,
        "value_shell_defect": value_defect,
        "jet_shell_defect": jet_defect,
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(maximum_lag),
        "target_value_enters_interior_action": False,
        "valid_source_fraction": float(np.mean(validity)),
        "boundary_fallback_fraction": float(np.mean(boundary_fallback)),
        "mean_value_shell_defect": float(np.mean(value_defect)),
        "mean_jet_shell_defect": float(np.mean(jet_defect)),
        "mean_population_action": float(np.mean(population_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "boundary_closure": "reflection; not yet continuum-normalized",
    }


def root_context_collision_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Join the observed root and nested context as a two-source particle law.

    The root is lifted through the contextual jet distribution rather than
    being assigned a fabricated derivative.  Its action is the inherited
    contextual path action plus the root-to-context collision edge, so both
    branch roles are compared after traversing the same causal depth.  Root
    The context receives two thirds of the source measure because its value
    has two opposite-side parents, while the observed root receives one third.
    These simplex weights are causal multiplicities, not denoising parameters.
    """
    line = _validate_line(observation)
    context, context_diagnostic = nested_midpoint_particle_law_1d(line)
    prediction = context["prediction"]
    jet = context["jet"]
    samples, scales = prediction.shape
    context_reference = np.asarray(
        context.get("reference_mass", 1.0 / context["lag"]),
        dtype=np.float64,
    )
    if context_reference.ndim == 1:
        context_reference = np.broadcast_to(
            context_reference / np.sum(context_reference), prediction.shape)
    root_collision_action = np.abs(line[:, None] - prediction)
    root_action = context["total_action"] + root_collision_action
    combined_prediction = np.concatenate((
        prediction,
        np.broadcast_to(line[:, None], prediction.shape),
    ), axis=-1)
    combined_jet = np.concatenate((jet, jet), axis=-1)
    combined_action = np.concatenate((
        context["total_action"], root_action), axis=-1)
    combined_reference = np.concatenate((
        (2.0 / 3.0) * context_reference,
        (1.0 / 3.0) * context_reference,
    ), axis=-1)
    floor = _representation_floor(line)
    conductance = combined_reference / np.maximum(combined_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": combined_prediction,
        "jet": combined_jet,
        "total_action": combined_action,
        "mass": mass,
        "lag": np.concatenate((context["lag"], context["lag"])),
        "reference_mass": combined_reference,
        "branch_role": np.concatenate((
            np.zeros(scales, dtype=np.int64),
            np.ones(scales, dtype=np.int64),
        )),
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(context_diagnostic["maximum_lag"]),
        "characteristic_count": int(2 * scales),
        "target_value_enters_interior_action": True,
        "root_action": (
            "contextual path action plus root-to-context collision edge"
        ),
        "source_measure": (
            "two contextual parents plus one observed root on the causal simplex"
        ),
        "mean_root_action": float(np.mean(root_action)),
        "mean_root_collision_action": float(np.mean(
            root_collision_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "context_law": context_diagnostic,
    }


def independent_side_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build target-free left/right particles with independent ancestry.

    A side predicts the target affinely from its samples at ``s`` and ``2s``.
    The same side's outer ``2s,4s`` shell supplies an affine-exact value--jet
    consistency action.  Left and right roles remain distinct during lineage;
    they are allowed to meet only in the terminal collision measure.
    """
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 4
    lag = np.arange(1, maximum_lag + 1, dtype=np.float64)
    haar = 1.0 / lag
    haar /= np.sum(haar)
    if float(np.ptp(line)) == 0.0:
        side_prediction = np.broadcast_to(
            line[:, None], (samples, maximum_lag)).copy()
        prediction = np.concatenate((side_prediction, side_prediction), axis=-1)
        reference = np.broadcast_to(
            0.5 * np.concatenate((haar, haar)), prediction.shape).copy()
        return {
            "prediction": prediction,
            "jet": np.zeros_like(prediction),
            "total_action": np.zeros_like(prediction),
            "mass": reference.copy(),
            "lag": np.concatenate((lag, lag)),
            "reference_mass": reference,
            "branch_role": np.concatenate((
                np.zeros(maximum_lag, dtype=np.int64),
                np.ones(maximum_lag, dtype=np.int64),
            )),
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(2 * maximum_lag),
            "target_value_enters_interior_action": False,
            "affine_interior_action": 0.0,
        }

    padded = np.pad(line, 4 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 4 * maximum_lag
    floor = _representation_floor(line)
    target_identity = np.arange(samples)
    period = 2 * (samples - 1)

    def reflected_identity(raw_index: np.ndarray) -> np.ndarray:
        folded = np.mod(raw_index, period)
        return np.where(folded < samples, folded, period - folded)

    role_predictions = []
    role_jets = []
    role_actions = []
    role_references = []
    validity_records = []
    for direction in (-1, 1):
        predictions = []
        jets = []
        actions = []
        validity_fields = []
        for scale in range(1, maximum_lag + 1):
            inner = padded[index + direction * scale]
            middle = padded[index + direction * 2 * scale]
            outer = padded[index + direction * 4 * scale]
            prediction = 2.0 * inner - middle
            outer_prediction = 2.0 * middle - outer
            if direction < 0:
                jet = (inner - middle) / scale
                outer_jet = (middle - outer) / (2.0 * scale)
            else:
                jet = (middle - inner) / scale
                outer_jet = (outer - middle) / (2.0 * scale)
            predictions.append(prediction)
            jets.append(jet)
            actions.append(
                np.abs(prediction - outer_prediction)
                + scale * np.abs(jet - outer_jet)
            )
            raw_identities = np.stack((
                target_identity + direction * scale,
                target_identity + direction * 2 * scale,
                target_identity + direction * 4 * scale,
            ), axis=-1)
            identities = reflected_identity(raw_identities)
            in_domain = np.all(
                (raw_identities >= 0) & (raw_identities < samples), axis=-1)
            excludes_target = np.all(
                identities != target_identity[:, None], axis=-1)
            distinct = np.all(
                np.diff(np.sort(identities, axis=-1), axis=-1) != 0,
                axis=-1,
            )
            validity_fields.append(in_domain & excludes_target & distinct)
        role_prediction = np.stack(predictions, axis=-1)
        role_jet = np.stack(jets, axis=-1)
        local_action = np.stack(actions, axis=-1)
        validity = np.stack(validity_fields, axis=-1)
        role_reference = validity * haar[None, :]
        missing = np.sum(role_reference, axis=-1) <= np.finfo(float).tiny
        role_reference[missing] = haar
        role_reference /= np.sum(role_reference, axis=-1, keepdims=True)
        population_action = _w1_population_action_1d(
            role_prediction, role_reference)
        role_predictions.append(role_prediction)
        role_jets.append(role_jet)
        role_actions.append(local_action + population_action)
        role_references.append(role_reference)
        validity_records.append(validity)

    prediction = np.concatenate(role_predictions, axis=-1)
    jet = np.concatenate(role_jets, axis=-1)
    total_action = np.concatenate(role_actions, axis=-1)
    reference_mass = 0.5 * np.concatenate(role_references, axis=-1)
    conductance = reference_mass / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "jet": jet,
        "total_action": total_action,
        "mass": mass,
        "lag": np.concatenate((lag, lag)),
        "reference_mass": reference_mass,
        "branch_role": np.concatenate((
            np.zeros(maximum_lag, dtype=np.int64),
            np.ones(maximum_lag, dtype=np.int64),
        )),
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(2 * maximum_lag),
        "target_value_enters_interior_action": False,
        "valid_source_fraction": float(np.mean(validity_records)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "boundary_closure": "reflection; not yet continuum-normalized",
    }


def symmetric_second_jet_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build target-free symmetric particles exact on quadratic structure."""
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 4
    lag = np.arange(1, maximum_lag + 1, dtype=np.float64)
    haar = 1.0 / lag
    haar /= np.sum(haar)
    if float(np.ptp(line)) == 0.0:
        prediction = np.broadcast_to(
            line[:, None], (samples, maximum_lag)).copy()
        mass = np.broadcast_to(haar, prediction.shape).copy()
        return {
            "prediction": prediction,
            "jet": np.zeros_like(prediction),
            "curvature": np.zeros_like(prediction),
            "total_action": np.zeros_like(prediction),
            "mass": mass,
            "lag": lag,
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(maximum_lag),
            "target_value_enters_interior_action": False,
            "quadratic_interior_action": 0.0,
        }

    padded = np.pad(line, 4 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 4 * maximum_lag
    target = np.arange(samples)
    floor = _representation_floor(line)
    prediction_fields = []
    jet_fields = []
    curvature_fields = []
    action_fields = []
    validity_fields = []
    for scale in range(1, maximum_lag + 1):
        left_one = padded[index - scale]
        right_one = padded[index + scale]
        left_two = padded[index - 2 * scale]
        right_two = padded[index + 2 * scale]
        left_four = padded[index - 4 * scale]
        right_four = padded[index + 4 * scale]
        midpoint_one = 0.5 * (left_one + right_one)
        midpoint_two = 0.5 * (left_two + right_two)
        midpoint_four = 0.5 * (left_four + right_four)
        jet_one = (right_one - left_one) / (2.0 * scale)
        jet_two = (right_two - left_two) / (4.0 * scale)
        jet_four = (right_four - left_four) / (8.0 * scale)
        prediction = (4.0 * midpoint_one - midpoint_two) / 3.0
        outer_prediction = (4.0 * midpoint_two - midpoint_four) / 3.0
        jet = (4.0 * jet_one - jet_two) / 3.0
        outer_jet = (4.0 * jet_two - jet_four) / 3.0
        curvature = (
            (2.0 / 3.0) * (midpoint_two - midpoint_one) / (scale * scale)
        )
        outer_curvature = (
            (2.0 / 3.0) * (midpoint_four - midpoint_two)
            / (4.0 * scale * scale)
        )
        prediction_fields.append(prediction)
        jet_fields.append(jet)
        curvature_fields.append(curvature)
        action_fields.append(
            np.abs(prediction - outer_prediction)
            + scale * np.abs(jet - outer_jet)
            + scale * scale * np.abs(curvature - outer_curvature)
        )
        validity_fields.append(
            (target - 4 * scale >= 0) & (target + 4 * scale < samples))

    prediction = np.stack(prediction_fields, axis=-1)
    jet = np.stack(jet_fields, axis=-1)
    curvature = np.stack(curvature_fields, axis=-1)
    local_action = np.stack(action_fields, axis=-1)
    validity = np.stack(validity_fields, axis=-1)
    reference_mass = validity * haar[None, :]
    missing = np.sum(reference_mass, axis=-1) <= np.finfo(float).tiny
    reference_mass[missing] = haar
    reference_mass /= np.sum(reference_mass, axis=-1, keepdims=True)
    population_action = _w1_population_action_1d(
        prediction, reference_mass)
    total_action = local_action + population_action
    conductance = reference_mass / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "jet": jet,
        "curvature": curvature,
        "total_action": total_action,
        "mass": mass,
        "lag": lag,
        "reference_mass": reference_mass,
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(maximum_lag),
        "target_value_enters_interior_action": False,
        "valid_source_fraction": float(np.mean(validity)),
        "boundary_fallback_fraction": float(np.mean(missing)),
        "mean_value_jet2_action": float(np.mean(local_action)),
        "mean_population_action": float(np.mean(population_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "boundary_closure": "reflection fallback only outside admissible shells",
    }


def continuous_curvature_particle_law_1d(
    observation: np.ndarray,
    *,
    curvature_intervals: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport the full midpoint-to-quadratic curvature simplex."""
    line = _validate_line(observation)
    if curvature_intervals < 1:
        raise ValueError("curvature representation needs at least one interval")
    samples = line.size
    maximum_lag = samples // 4
    scales = np.arange(1, maximum_lag + 1, dtype=np.float64)
    haar = 1.0 / scales
    haar /= np.sum(haar)
    curvature_coordinate = np.linspace(
        0.0, 1.0, curvature_intervals + 1)
    curvature_weight = np.ones(curvature_intervals + 1, dtype=np.float64)
    curvature_weight[[0, -1]] = 0.5
    curvature_weight /= np.sum(curvature_weight)
    reference_fibre = (
        haar[:, None] * curvature_weight[None, :]).reshape(-1)
    branches = reference_fibre.size
    if float(np.ptp(line)) == 0.0:
        prediction = np.broadcast_to(line[:, None], (samples, branches)).copy()
        mass = np.broadcast_to(reference_fibre, prediction.shape).copy()
        return {
            "prediction": prediction,
            "midpoint_prediction": prediction.copy(),
            "jet": np.zeros_like(prediction),
            "curvature": np.zeros_like(prediction),
            "local_action": np.zeros_like(prediction),
            "total_action": np.zeros_like(prediction),
            "mass": mass,
            "lag": np.repeat(scales, curvature_coordinate.size),
            "reference_mass": reference_fibre,
            "curvature_coordinate": np.tile(
                curvature_coordinate, maximum_lag),
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(branches),
            "curvature_intervals": int(curvature_intervals),
            "target_value_enters_interior_action": False,
        }

    padded = np.pad(line, 4 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 4 * maximum_lag
    target = np.arange(samples)
    floor = _representation_floor(line)
    predictions = []
    midpoint_predictions = []
    jets = []
    curvatures = []
    actions = []
    validities = []
    for scale in range(1, maximum_lag + 1):
        left_one = padded[index - scale]
        right_one = padded[index + scale]
        left_two = padded[index - 2 * scale]
        right_two = padded[index + 2 * scale]
        left_four = padded[index - 4 * scale]
        right_four = padded[index + 4 * scale]
        z_one = 0.5 * (left_one + right_one)
        z_two = 0.5 * (left_two + right_two)
        z_four = 0.5 * (left_four + right_four)
        j_one = (right_one - left_one) / (2.0 * scale)
        j_two = (right_two - left_two) / (4.0 * scale)
        j_four = (right_four - left_four) / (8.0 * scale)
        z_richardson = (4.0 * z_one - z_two) / 3.0
        z_outer_richardson = (4.0 * z_two - z_four) / 3.0
        j_richardson = (4.0 * j_one - j_two) / 3.0
        j_outer_richardson = (4.0 * j_two - j_four) / 3.0
        curvature = (
            (2.0 / 3.0) * (z_two - z_one) / (scale * scale))
        outer_curvature = (
            (2.0 / 3.0) * (z_four - z_two)
            / (4.0 * scale * scale))
        t = curvature_coordinate[None, :]
        prediction = (
            z_one[:, None]
            + t * (z_richardson - z_one)[:, None])
        outer_prediction = (
            z_two[:, None]
            + t * (z_outer_richardson - z_two)[:, None])
        jet = j_one[:, None] + t * (j_richardson - j_one)[:, None]
        outer_jet = (
            j_two[:, None]
            + t * (j_outer_richardson - j_two)[:, None])
        lifted_curvature = t * curvature[:, None]
        lifted_outer_curvature = t * outer_curvature[:, None]
        predictions.append(prediction)
        midpoint_predictions.append(np.broadcast_to(
            z_one[:, None], prediction.shape))
        jets.append(jet)
        curvatures.append(lifted_curvature)
        actions.append(
            np.abs(prediction - outer_prediction)
            + scale * np.abs(jet - outer_jet)
            + scale * scale * np.abs(
                lifted_curvature - lifted_outer_curvature))
        valid = (target - 4 * scale >= 0) & (target + 4 * scale < samples)
        validities.append(np.broadcast_to(
            valid[:, None], prediction.shape))

    prediction = np.concatenate(predictions, axis=-1)
    midpoint_prediction = np.concatenate(midpoint_predictions, axis=-1)
    jet = np.concatenate(jets, axis=-1)
    curvature = np.concatenate(curvatures, axis=-1)
    local_action = np.concatenate(actions, axis=-1)
    validity = np.concatenate(validities, axis=-1)
    reference_mass = validity * reference_fibre[None, :]
    missing = np.sum(reference_mass, axis=-1) <= np.finfo(float).tiny
    reference_mass[missing] = reference_fibre
    reference_mass /= np.sum(reference_mass, axis=-1, keepdims=True)
    population_action = _w1_population_action_1d(
        prediction, reference_mass)
    total_action = local_action + population_action
    conductance = reference_mass / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "midpoint_prediction": midpoint_prediction,
        "jet": jet,
        "curvature": curvature,
        "local_action": local_action,
        "total_action": total_action,
        "mass": mass,
        "lag": np.repeat(scales, curvature_coordinate.size),
        "reference_mass": reference_mass,
        "curvature_coordinate": np.tile(
            curvature_coordinate, maximum_lag),
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(branches),
        "curvature_intervals": int(curvature_intervals),
        "curvature_measure": "Lebesgue measure on the reconstruction simplex",
        "target_value_enters_interior_action": False,
        "valid_source_fraction": float(np.mean(validity)),
        "boundary_fallback_fraction": float(np.mean(missing)),
        "mean_local_action": float(np.mean(local_action)),
        "mean_population_action": float(np.mean(population_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
    }


def curvature_consensus_particle_law_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Add the exact full-scale W1 curvature potential to jet-two particles."""
    line = _validate_line(observation)
    law, diagnostic = symmetric_second_jet_particle_law_1d(line)
    prediction = law["prediction"]
    if float(np.ptp(line)) == 0.0:
        diagnostic = dict(diagnostic)
        diagnostic["curvature_population_action"] = "zero constant law"
        return law, diagnostic
    reference = np.asarray(law["reference_mass"], dtype=np.float64)
    curvature = law["curvature"]
    curvature_population_action = (
        law["lag"][None, :] ** 2
        * np.sum(
            reference[:, None, :]
            * np.abs(curvature[:, :, None] - curvature[:, None, :]),
            axis=-1,
        )
    )
    total_action = law["total_action"] + curvature_population_action
    floor = _representation_floor(line)
    conductance = reference / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    result = dict(law)
    result["total_action"] = total_action
    result["mass"] = mass
    result["curvature_population_action"] = curvature_population_action
    output_diagnostic = dict(diagnostic)
    output_diagnostic.update({
        "curvature_population_action": (
            "s^2 times exact W1 potential against the complete curvature law"
        ),
        "mean_curvature_population_action": float(np.mean(
            curvature_population_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
    })
    return result, output_diagnostic


def _lineage_branch_transport_1d(
    observation: np.ndarray,
    *,
    bundle_metric: str,
    transition_normalization: str = "markov",
    preserve_branch_role: bool = False,
    connection_mean_multiplier: np.ndarray | None = None,
    connection_transport_covariance: np.ndarray | None = None,
    particle_law_builder: Callable[
        [np.ndarray], tuple[dict[str, np.ndarray], dict[str, Any]]
    ] = relation_scale_particle_law,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport positive branch mass through the one-dimensional jet bundle.

    Local branch action supplies observation likelihood.  Between adjacent
    base points, atoms are parallel transported to their common midpoint and
    coupled by reciprocal Sasaki distance.  Forward/backward messages are
    positive and mass-conserving; branch identity is the geometric jet state,
    not its quadrature index.
    """
    line = _validate_line(observation)
    law, local_diagnostic = particle_law_builder(line)
    prediction = law["prediction"]
    jet = law["jet"]
    action = law["total_action"]
    samples, branches = prediction.shape
    if connection_mean_multiplier is not None:
        connection_mean_multiplier = np.asarray(
            connection_mean_multiplier, dtype=np.float64)
        if connection_mean_multiplier.shape != (samples - 1,):
            raise ValueError(
                "connection mean multiplier must align with edges")
        if (
            not np.all(np.isfinite(connection_mean_multiplier))
            or np.any(connection_mean_multiplier < 0.0)
        ):
            raise ValueError(
                "connection mean multiplier must be finite and nonnegative")
    if connection_transport_covariance is not None:
        connection_transport_covariance = np.asarray(
            connection_transport_covariance, dtype=np.float64)
        if connection_transport_covariance.shape != (samples - 1, 3, 3):
            raise ValueError(
                "connection transport covariance must align with edges")
        if not np.all(np.isfinite(connection_transport_covariance)):
            raise ValueError("connection transport covariance must be finite")
    uses_curvature = bundle_metric == "joint_information_curvature"
    curvature = np.asarray(
        law.get("curvature", np.zeros_like(prediction)), dtype=np.float64)
    if uses_curvature and curvature.shape != prediction.shape:
        raise ValueError("curvature must align with second-order particles")
    bundle_dimension = 4 if uses_curvature else 3
    floor = _representation_floor(line)
    lag = law["lag"]
    haar = 1.0 / lag
    haar /= np.sum(haar)
    reference_law = np.asarray(
        law.get("reference_mass", haar), dtype=np.float64)
    reference = (
        np.broadcast_to(reference_law, prediction.shape)
        if reference_law.ndim == 1
        else reference_law
    )
    if reference.shape != prediction.shape:
        raise ValueError("branch reference measure must align with particles")
    branch_role = np.asarray(
        law.get("branch_role", np.zeros(branches, dtype=np.int64)))
    if branch_role.shape != (branches,):
        raise ValueError("branch role must align with particle fibre")
    path_family = np.asarray(
        law.get("path_family", np.zeros(branches, dtype=np.int64)))
    if path_family.shape != (branches,):
        raise ValueError("path family must align with particle fibre")
    if transition_normalization not in {"markov", "action_density"}:
        raise ValueError("unknown transition normalization")

    if float(np.ptp(line)) == 0.0:
        mass = law["mass"].copy()
        return {
            "mass": mass,
            "forward_mass": mass.copy(),
            "backward_mass": mass.copy(),
            "symmetric_parent_mass": mass.copy(),
            "hj_joint_mass": mass.copy(),
            "hj_joint_collision_mass": mass.copy(),
            "hj_phase_collision_mass": mass.copy(),
            "hj_coupled_phase_mass": mass.copy(),
            "hj_coupled_phase_collision_mass": mass.copy(),
            "hj_coupled_phase_coverage_mass": mass.copy(),
            "hj_coupled_phase_bundle_coverage_mass": mass.copy(),
            "hj_viterbi_section": line.copy(),
            "hj_viterbi_branch": np.zeros(samples, dtype=np.int64),
            "hj_viterbi_predecessor": np.zeros(
                (samples, branches), dtype=np.int64),
            "posterior_characteristic_section": line.copy(),
            "posterior_characteristic_branch": np.zeros(
                samples, dtype=np.int64),
            "posterior_characteristic_predecessor": np.zeros(
                (samples, branches), dtype=np.int64),
            "path_collision_mass": mass.copy(),
            "path_affinity_mass": mass.copy(),
            "path_fidelity_mass": mass.copy(),
            "transport_fidelity_mass": mass.copy(),
            "transport_plan_history_mass": mass.copy(),
            "self_consistent_transport_mass": mass.copy(),
            "distributed_transport_mass": mass.copy(),
            "action_contracting_transport_mass": mass.copy(),
            "two_history_action_transport_mass": mass.copy(),
            "transport_edge_fidelity": np.ones(samples - 1),
            "transport_vertex_fidelity": np.ones(samples),
            "path_fidelity_participation": np.ones(samples),
            "coupled_phase_action": np.zeros_like(mass),
            "coupled_phase_population": np.ones_like(mass),
            "prediction": prediction,
            "jet": jet,
            "curvature": curvature,
            "curvature_coordinate": np.asarray(
                law.get("curvature_coordinate", np.zeros(branches)),
                dtype=np.float64,
            ),
            "midpoint_prediction": np.asarray(
                law.get("midpoint_prediction", prediction),
                dtype=np.float64,
            ),
            "reference_mass": reference_law,
            "edge_precision": np.broadcast_to(
                np.eye(bundle_dimension),
                (samples - 1, bundle_dimension, bundle_dimension),
            ).copy(),
        }, {
            "state": "bidirectional positive lineage on the jet bundle",
            "mean_lineage_population": float(
                1.0 / np.sum(mass[0] * mass[0])),
            "mean_transition_population": float(branches),
            "log_path_evidence": 0.0,
            "transition_normalization": transition_normalization,
            "preserve_branch_role": bool(preserve_branch_role),
            "target_value_enters_local_action": bool(local_diagnostic.get(
                "target_value_enters_interior_action", True)),
            "boundary_condition": "symmetric forward/backward messages",
            "transport_plan_fidelity": {
                "state": (
                    "Hellinger fidelity between ordinary and contracted "
                    "posterior transport plans"
                ),
                "minimum_edge_fidelity": 1.0,
                "mean_edge_fidelity": 1.0,
                "maximum_edge_fidelity": 1.0,
                "minimum_vertex_survival": 1.0,
                "mean_vertex_survival": 1.0,
                "maximum_vertex_survival": 1.0,
                "physical_parameters": "none",
            },
            "local_law": local_diagnostic,
        }

    likelihood = 1.0 / np.maximum(action, floor)
    likelihood /= np.max(likelihood, axis=1, keepdims=True)
    forward = np.empty_like(prediction)
    backward = np.empty_like(prediction)
    forward_score = np.full_like(prediction, -np.inf)
    backward_score = np.full_like(prediction, -np.inf)
    viterbi_predecessor = np.full(
        (samples, branches), -1, dtype=np.int64)
    forward[0] = reference[0] * likelihood[0]
    initial_evidence = float(np.sum(forward[0]))
    log_path_evidence = math.log(max(
        initial_evidence, np.finfo(float).tiny))
    forward[0] /= initial_evidence
    forward_score[0] = np.log(np.maximum(
        forward[0] / np.maximum(reference[0], np.finfo(float).tiny),
        np.finfo(float).tiny))
    forward_score[0] -= np.max(forward_score[0])
    transition_populations = []
    forward_kernels = []

    residual = line[:, None] - prediction
    information_anisotropies = []
    edge_precisions = []
    connection_authorities = []
    connection_mean_defects = []
    connection_covariances = []
    connection_family_disagreements = []
    connection_family_fidelities = []

    def transition(left: int, right: int, *, record: bool) -> np.ndarray:
        complete_similarity = None
        if uses_curvature:
            left_value = (
                prediction[left] + 0.5 * jet[left] + 0.125 * curvature[left])
            right_value = (
                prediction[right] - 0.5 * jet[right] + 0.125 * curvature[right])
            left_jet = jet[left] + 0.5 * curvature[left]
            right_jet = jet[right] - 0.5 * curvature[right]
        else:
            left_value = prediction[left] + 0.5 * jet[left]
            right_value = prediction[right] - 0.5 * jet[right]
            left_jet = jet[left]
            right_jet = jet[right]
        value_defect = left_value[:, None] - right_value[None, :]
        jet_defect = left_jet[:, None] - right_jet[None, :]
        if bundle_metric == "signal_euclidean":
            distance = np.hypot(value_defect, jet_defect)
        else:
            residual_defect = (
                residual[left][:, None] - residual[right][None, :])
            if bundle_metric == "joint_euclidean":
                distance = np.hypot(
                    np.hypot(value_defect, jet_defect), residual_defect)
            elif bundle_metric == "transport_ancestry_connection":
                left_coordinates = np.column_stack((
                    left_value, left_jet, residual[left]))
                right_coordinates = np.column_stack((
                    right_value, right_jet, residual[right]))
                family_mean = []
                family_precision = []
                family_authority = []
                family_covariances = []
                for family in np.unique(path_family):
                    family_index = path_family == family
                    left_weight = law["mass"][left, family_index]
                    right_weight = law["mass"][right, family_index]
                    left_weight = left_weight / np.sum(left_weight)
                    right_weight = right_weight / np.sum(right_weight)
                    family_left = left_coordinates[family_index]
                    family_right = right_coordinates[family_index]
                    left_center = left_weight @ family_left
                    right_center = right_weight @ family_right
                    left_centered = family_left - left_center
                    right_centered = family_right - right_center
                    local_covariance = (
                        (left_centered * left_weight[:, None]).T @ left_centered
                        + (right_centered * right_weight[:, None]).T
                        @ right_centered
                    )
                    eigenvalue, eigenvector = np.linalg.eigh(
                        local_covariance)
                    regularized_eigenvalue = np.maximum(
                        eigenvalue, floor * floor)
                    regularized_covariance = (
                        eigenvector * regularized_eigenvalue[None, :]
                    ) @ eigenvector.T
                    raw_family_precision = (
                        eigenvector
                        * (1.0 / regularized_eigenvalue)[None, :]
                    ) @ eigenvector.T
                    family_mean.append(left_center - right_center)
                    family_precision.append(raw_family_precision)
                    family_covariances.append(regularized_covariance)
                    local_mean_defect = left_center - right_center
                    local_action = float(
                        local_mean_defect
                        @ raw_family_precision
                        @ local_mean_defect)
                    family_authority.append(
                        local_action / (1.0 + local_action))
                family_mean_array = np.stack(family_mean, axis=0)
                pair_fidelity = []
                for first in range(len(family_covariances)):
                    for second in range(first + 1, len(family_covariances)):
                        covariance_first = family_covariances[first]
                        covariance_second = family_covariances[second]
                        pooled_covariance = 0.5 * (
                            covariance_first + covariance_second)
                        delta = (
                            family_mean_array[first]
                            - family_mean_array[second])
                        _, logdet_first = np.linalg.slogdet(covariance_first)
                        _, logdet_second = np.linalg.slogdet(covariance_second)
                        _, logdet_pooled = np.linalg.slogdet(
                            pooled_covariance)
                        log_affinity = (
                            0.25 * (logdet_first + logdet_second)
                            - 0.5 * logdet_pooled
                            - 0.125 * delta
                            @ np.linalg.solve(pooled_covariance, delta)
                        )
                        pair_fidelity.append(float(np.clip(
                            np.exp(2.0 * min(log_affinity, 0.0)),
                            0.0,
                            1.0,
                        )))
                information = np.sum(family_precision, axis=0)
                within_covariance = np.linalg.inv(information)
                information_moment = np.sum([
                    family_precision[index] @ family_mean_array[index]
                    for index in range(len(family_precision))
                ], axis=0)
                mean_defect = within_covariance @ information_moment
                family_offset = family_mean_array - mean_defect
                between_covariance = (
                    family_offset.T @ family_offset
                    / family_mean_array.shape[0]
                )
                left_weight = law["mass"][left]
                right_weight = law["mass"][right]
                left_center = left_weight @ left_coordinates
                right_center = right_weight @ right_coordinates
                left_centered = left_coordinates - left_center
                right_centered = right_coordinates - right_center
                population_covariance = (
                    (left_centered * left_weight[:, None]).T @ left_centered
                    + (right_centered * right_weight[:, None]).T
                    @ right_centered
                )
                defect_covariance = (
                    population_covariance + between_covariance)
                eigenvalue, eigenvector = np.linalg.eigh(defect_covariance)
                raw_precision_eigenvalue = 1.0 / np.maximum(
                    eigenvalue, floor * floor)
                raw_precision = (
                    eigenvector * raw_precision_eigenvalue[None, :]
                ) @ eigenvector.T
                family_fidelity = float(np.prod(pair_fidelity))
                connection_authority = float(
                    np.prod(family_authority) * family_fidelity)
                precision_eigenvalue = raw_precision_eigenvalue.copy()
                precision_eigenvalue /= np.exp(np.mean(np.log(
                    precision_eigenvalue)))
                precision = (
                    eigenvector * precision_eigenvalue[None, :]
                ) @ eigenvector.T
                defect = np.stack(np.broadcast_arrays(
                    value_defect, jet_defect, residual_defect), axis=-1)
                centered_defect = (
                    defect
                    - connection_authority * mean_defect[None, None, :])
                distance_squared = np.einsum(
                    "...a,ab,...b->...",
                    centered_defect,
                    precision,
                    centered_defect,
                )
                distance = np.sqrt(np.maximum(distance_squared, 0.0))
                if record:
                    information_anisotropies.append(float(
                        np.max(precision_eigenvalue)
                        / np.min(precision_eigenvalue)))
                    edge_precisions.append(precision)
                    connection_authorities.append(connection_authority)
                    connection_mean_defects.append(mean_defect)
                    connection_covariances.append(defect_covariance)
                    connection_family_disagreements.append(float(np.trace(
                        raw_precision @ between_covariance)))
                    connection_family_fidelities.append(family_fidelity)
            elif bundle_metric in {
                "transport_covariance",
                "transport_bidirectional_collision",
                "transport_connection_collision",
                "transport_distribution",
                "transport_external_connection",
                "transport_external_action_marginal",
                "transport_external_gaussian_marginal",
                "transport_gaussian_potential",
                "transport_self_consistent",
            }:
                left_coordinates = np.column_stack((
                    left_value, left_jet, residual[left]))
                right_coordinates = np.column_stack((
                    right_value, right_jet, residual[right]))
                left_weight = law["mass"][left]
                right_weight = law["mass"][right]
                left_center = left_weight @ left_coordinates
                right_center = right_weight @ right_coordinates
                left_centered = left_coordinates - left_center
                right_centered = right_coordinates - right_center
                defect_covariance = (
                    (left_centered * left_weight[:, None]).T @ left_centered
                    + (right_centered * right_weight[:, None]).T
                    @ right_centered
                )
                if (
                    bundle_metric == "transport_external_gaussian_marginal"
                    and connection_transport_covariance is not None
                ):
                    defect_covariance = (
                        defect_covariance
                        + connection_transport_covariance[left])
                eigenvalue, eigenvector = np.linalg.eigh(defect_covariance)
                raw_precision_eigenvalue = 1.0 / np.maximum(
                    eigenvalue, floor * floor)
                precision_eigenvalue = raw_precision_eigenvalue.copy()
                precision_eigenvalue /= np.exp(np.mean(np.log(
                    precision_eigenvalue)))
                precision = (
                    eigenvector * precision_eigenvalue[None, :]
                ) @ eigenvector.T
                mean_defect = left_center - right_center
                raw_precision = (
                    eigenvector * raw_precision_eigenvalue[None, :]
                ) @ eigenvector.T
                connection_action = float(
                    mean_defect @ raw_precision @ mean_defect)
                connection_authority = (
                    connection_action / (1.0 + connection_action))
                defect = np.stack(np.broadcast_arrays(
                    value_defect, jet_defect, residual_defect), axis=-1)
                if bundle_metric == "transport_distribution":
                    transported_mean_defect = mean_defect
                elif bundle_metric in {
                    "transport_external_connection",
                    "transport_external_action_marginal",
                    "transport_external_gaussian_marginal",
                }:
                    if connection_mean_multiplier is None:
                        raise ValueError(
                            "external connection metric requires a multiplier")
                    transported_mean_defect = (
                        connection_mean_multiplier[left] * mean_defect)
                elif bundle_metric == "transport_self_consistent":
                    transported_mean_defect = (
                        connection_authority * mean_defect)
                elif bundle_metric == "transport_connection_collision":
                    transported_mean_defect = (
                        connection_authority ** 2 * mean_defect)
                elif bundle_metric == "transport_bidirectional_collision":
                    transported_mean_defect = (
                        connection_authority ** 4 * mean_defect)
                else:
                    transported_mean_defect = np.zeros_like(mean_defect)
                centered_defect = (
                    defect - transported_mean_defect[None, None, :])
                if bundle_metric == "transport_external_gaussian_marginal":
                    gaussian_action = np.einsum(
                        "...a,ab,...b->...",
                        centered_defect,
                        raw_precision,
                        centered_defect,
                    )
                    log_similarity = -0.5 * gaussian_action
                    log_similarity -= np.max(
                        log_similarity, axis=1, keepdims=True)
                    complete_similarity = np.exp(log_similarity)
                    distance = np.ones_like(complete_similarity)
                elif bundle_metric == "transport_external_action_marginal":
                    direction = transported_mean_defect
                    quadratic = float(
                        direction @ raw_precision @ direction)
                    uncentered_action = np.einsum(
                        "...a,ab,...b->...", defect, raw_precision, defect)
                    if quadratic > np.finfo(float).tiny:
                        linear = np.einsum(
                            "...a,ab,b->...",
                            defect,
                            raw_precision,
                            direction,
                        )
                        root_quadratic = math.sqrt(quadratic)
                        lower = -linear / root_quadratic
                        upper = (quadratic - linear) / root_quadratic
                        interval_mass = (
                            special.ndtr(upper) - special.ndtr(lower))
                        log_similarity = (
                            -0.5 * (
                                uncentered_action
                                - linear * linear / quadratic)
                            + 0.5 * math.log(2.0 * math.pi / quadratic)
                            + np.log(np.maximum(
                                interval_mass, np.finfo(float).tiny))
                        )
                    else:
                        log_similarity = -0.5 * uncentered_action
                    # Row constants cancel in Markov normalization and this
                    # representation prevents underflow of exact marginal
                    # action evidence.
                    log_similarity -= np.max(
                        log_similarity, axis=1, keepdims=True)
                    complete_similarity = np.exp(log_similarity)
                    distance = np.ones_like(complete_similarity)
                elif bundle_metric == "transport_gaussian_potential":
                    centered_defect = defect - mean_defect[None, None, :]
                    radius_squared = np.einsum(
                        "...a,ab,...b->...",
                        centered_defect,
                        raw_precision,
                        centered_defect,
                    )
                    radius = np.sqrt(np.maximum(radius_squared, 0.0))
                    gaussian_potential = np.divide(
                        special.erf(radius / math.sqrt(2.0)),
                        radius,
                        out=np.full_like(radius, math.sqrt(2.0 / math.pi)),
                        where=radius > math.sqrt(np.finfo(float).eps),
                    )
                    complete_similarity = gaussian_potential
                    distance = np.ones_like(radius)
                else:
                    distance_squared = np.einsum(
                        "...a,ab,...b->...",
                        centered_defect,
                        precision,
                        centered_defect,
                    )
                    if (
                        bundle_metric == "transport_external_connection"
                        and connection_transport_covariance is not None
                    ):
                        distance_squared = distance_squared + float(np.trace(
                            precision @ connection_transport_covariance[left]
                        ))
                    distance = np.sqrt(np.maximum(distance_squared, 0.0))
                if record:
                    information_anisotropies.append(float(
                        np.max(precision_eigenvalue)
                        / np.min(precision_eigenvalue)))
                    edge_precisions.append(precision)
                    connection_authorities.append(connection_authority)
                    connection_mean_defects.append(mean_defect)
                    connection_covariances.append(defect_covariance)
            elif bundle_metric == "complete_participation":
                left_coordinates = np.column_stack((
                    left_value, left_jet, residual[left]))
                right_coordinates = np.column_stack((
                    right_value, right_jet, residual[right]))
                coordinates = np.concatenate(
                    (left_coordinates, right_coordinates), axis=0)
                weights = 0.5 * np.concatenate(
                    (law["mass"][left], law["mass"][right]))
                center = weights @ coordinates
                centered = coordinates - center
                variance = np.sum(
                    weights[:, None] * centered * centered, axis=0)
                precision_coordinate = 1.0 / np.maximum(
                    variance, floor * floor)
                precision_coordinate /= np.exp(np.mean(np.log(
                    precision_coordinate)))
                defects = np.stack(np.broadcast_arrays(
                    value_defect, jet_defect, residual_defect), axis=-1)
                primitive = 1.0 / np.sqrt(
                    1.0
                    + precision_coordinate[None, None, :] * defects ** 2)
                complete_similarity = np.prod(1.0 + primitive, axis=-1) - 1.0
                distance = 1.0 / np.maximum(
                    complete_similarity, np.finfo(float).tiny)
                if record:
                    information_anisotropies.append(float(
                        np.max(precision_coordinate)
                        / np.min(precision_coordinate)))
                    edge_precisions.append(np.diag(precision_coordinate))
            elif bundle_metric in {
                "joint_information", "joint_information_curvature"
            }:
                if uses_curvature:
                    left_coordinates = np.column_stack((
                        left_value,
                        left_jet,
                        curvature[left],
                        residual[left],
                    ))
                    right_coordinates = np.column_stack((
                        right_value,
                        right_jet,
                        curvature[right],
                        residual[right],
                    ))
                    defect_components = (
                        value_defect,
                        jet_defect,
                        curvature[left][:, None] - curvature[right][None, :],
                        residual_defect,
                    )
                else:
                    left_coordinates = np.column_stack((
                        left_value, left_jet, residual[left]))
                    right_coordinates = np.column_stack((
                        right_value, right_jet, residual[right]))
                    defect_components = (
                        value_defect, jet_defect, residual_defect)
                coordinates = np.concatenate(
                    (left_coordinates, right_coordinates), axis=0)
                weights = 0.5 * np.concatenate(
                    (law["mass"][left], law["mass"][right]))
                center = weights @ coordinates
                centered = coordinates - center
                covariance = (centered * weights[:, None]).T @ centered
                eigenvalue, eigenvector = np.linalg.eigh(covariance)
                covariance_floor = floor * floor
                precision_eigenvalue = 1.0 / np.maximum(
                    eigenvalue, covariance_floor)
                geometric_precision = float(np.exp(np.mean(
                    np.log(precision_eigenvalue))))
                precision_eigenvalue /= geometric_precision
                precision = (
                    eigenvector * precision_eigenvalue[None, :]
                ) @ eigenvector.T
                defect = np.stack(
                    np.broadcast_arrays(*defect_components),
                    axis=-1,
                )
                distance_squared = np.einsum(
                    "...a,ab,...b->...", defect, precision, defect)
                distance = np.sqrt(np.maximum(distance_squared, 0.0))
                if record:
                    information_anisotropies.append(float(
                        np.max(precision_eigenvalue)
                        / np.min(precision_eigenvalue)))
                    edge_precisions.append(precision)
            else:
                raise ValueError(f"unknown branch bundle metric {bundle_metric!r}")
        kernel = (
            reference[right][None, :] * complete_similarity
            if complete_similarity is not None
            else reference[right][None, :] / np.maximum(distance, floor)
        )
        if preserve_branch_role:
            kernel = kernel * (
                branch_role[:, None] == branch_role[None, :])
        if transition_normalization == "markov":
            kernel /= np.sum(kernel, axis=1, keepdims=True)
        if record:
            conditional_kernel = kernel / np.sum(
                kernel, axis=1, keepdims=True)
            transition_populations.append(float(np.mean(
                1.0 / np.sum(conditional_kernel * conditional_kernel, axis=1))))
        return kernel

    for index in range(1, samples):
        kernel = transition(index - 1, index, record=True)
        forward_kernels.append(kernel)
        predicted_mass = forward[index - 1] @ kernel
        forward[index] = predicted_mass * likelihood[index]
        step_evidence = float(np.sum(forward[index]))
        log_path_evidence += math.log(max(
            step_evidence, np.finfo(float).tiny))
        forward[index] /= step_evidence
        density_kernel = kernel / np.maximum(
            reference[index][None, :], np.finfo(float).tiny)
        predecessor_score = (
            forward_score[index - 1][:, None]
            + np.log(np.maximum(
                density_kernel, np.finfo(float).tiny))
        )
        viterbi_predecessor[index] = np.argmax(
            predecessor_score, axis=0)
        forward_score[index] = np.max(
            predecessor_score, axis=0
        ) + np.log(np.maximum(likelihood[index], np.finfo(float).tiny))
        forward_score[index] -= np.max(forward_score[index])

    viterbi_branch = np.empty(samples, dtype=np.int64)
    viterbi_branch[-1] = int(np.argmax(forward_score[-1]))
    for index in range(samples - 1, 0, -1):
        viterbi_branch[index - 1] = viterbi_predecessor[
            index, viterbi_branch[index]]
    viterbi_section = prediction[np.arange(samples), viterbi_branch]

    backward[-1] = 1.0
    backward_score[-1] = 0.0
    for index in range(samples - 2, -1, -1):
        kernel = forward_kernels[index]
        backward[index] = kernel @ (
            likelihood[index + 1] * backward[index + 1])
        maximum = float(np.max(backward[index]))
        if maximum > 0.0:
            backward[index] /= maximum
        density_kernel = kernel / np.maximum(
            reference[index + 1][None, :], np.finfo(float).tiny)
        backward_score[index] = np.max(
            np.log(np.maximum(density_kernel, np.finfo(float).tiny))
            + np.log(np.maximum(
                likelihood[index + 1], np.finfo(float).tiny))[None, :]
            + backward_score[index + 1][None, :],
            axis=1,
        )
        backward_score[index] -= np.max(backward_score[index])

    mass = forward * backward
    mass /= np.sum(mass, axis=1, keepdims=True)

    # A single raw-likelihood Viterbi history is structurally sharp but lets a
    # corrupted local branch seize the whole path.  Retain the complete
    # sum-product uncertainty first, then remarch its representation-invariant
    # density as the source action.  This is a transport of the posterior over
    # transports: the second characteristic is chosen by evidence already
    # marginalized over every competing history.
    posterior_score = np.full_like(prediction, -np.inf)
    posterior_predecessor = np.full(
        (samples, branches), -1, dtype=np.int64)
    posterior_density = mass / np.maximum(
        reference, np.finfo(float).tiny)
    posterior_score[0] = np.log(np.maximum(
        posterior_density[0], np.finfo(float).tiny))
    posterior_score[0] -= np.max(posterior_score[0])
    for index in range(1, samples):
        density_kernel = forward_kernels[index - 1] / np.maximum(
            reference[index][None, :], np.finfo(float).tiny)
        candidate_score = (
            posterior_score[index - 1][:, None]
            + np.log(np.maximum(
                density_kernel, np.finfo(float).tiny))
        )
        posterior_predecessor[index] = np.argmax(candidate_score, axis=0)
        posterior_score[index] = (
            np.max(candidate_score, axis=0)
            + np.log(np.maximum(
                posterior_density[index], np.finfo(float).tiny))
        )
        posterior_score[index] -= np.max(posterior_score[index])
    posterior_branch = np.empty(samples, dtype=np.int64)
    posterior_branch[-1] = int(np.argmax(posterior_score[-1]))
    for index in range(samples - 1, 0, -1):
        posterior_branch[index - 1] = posterior_predecessor[
            index, posterior_branch[index]]
    posterior_characteristic_section = prediction[
        np.arange(samples), posterior_branch]

    # Collide complete histories before marginalizing them.  If the original
    # path density is written relative to the product Haar law, the order-two
    # collision squares every likelihood density and every transition density.
    # The induced child transition measure is therefore
    #
    #   h_child * (K / h_child)^2 = K^2 / h_child.
    #
    # A second sum-product pass computes the exact branch marginals of this
    # global path-collision law.  It retains all coherent histories, unlike a
    # Viterbi section, but collision occurs before any pointwise barycenter.
    path_collision_forward = np.empty_like(prediction)
    path_collision_backward = np.empty_like(prediction)
    path_collision_forward[0] = reference[0] * likelihood[0] ** 2
    path_collision_forward[0] /= np.sum(path_collision_forward[0])
    collision_kernels = []
    for index in range(1, samples):
        child_reference = np.maximum(
            reference[index], np.finfo(float).tiny)
        density_kernel = forward_kernels[index - 1] / child_reference[None, :]
        collision_kernel = (
            reference[index][None, :] * density_kernel * density_kernel)
        collision_kernels.append(collision_kernel)
        predicted_collision = (
            path_collision_forward[index - 1] @ collision_kernel)
        path_collision_forward[index] = (
            predicted_collision * likelihood[index] ** 2)
        path_collision_forward[index] /= np.sum(
            path_collision_forward[index])
    path_collision_backward[-1] = 1.0
    for index in range(samples - 2, -1, -1):
        path_collision_backward[index] = collision_kernels[index] @ (
            likelihood[index + 1] ** 2
            * path_collision_backward[index + 1]
        )
        maximum = float(np.max(path_collision_backward[index]))
        if maximum > 0.0:
            path_collision_backward[index] /= maximum
    path_collision_mass = (
        path_collision_forward * path_collision_backward)
    path_collision_mass /= np.sum(
        path_collision_mass, axis=1, keepdims=True)

    # The ordinary and path-collision laws are two complete estimates of the
    # same transport history measure.  Their Hellinger affinity is an exact
    # uncertainty coordinate in [0,1].  Use it as arclength on the
    # Fisher--Rao density geodesic between those laws: a collision law that
    # disagrees with the full posterior is automatically weakened, while
    # identical laws retain the complete order-two collision.
    path_affinity = np.sum(np.sqrt(
        mass * path_collision_mass), axis=1)
    base_log_density = np.log(np.maximum(
        mass / np.maximum(reference, np.finfo(float).tiny),
        np.finfo(float).tiny,
    ))
    collision_log_density = np.log(np.maximum(
        path_collision_mass
        / np.maximum(reference, np.finfo(float).tiny),
        np.finfo(float).tiny,
    ))
    path_affinity_log_density = (
        (1.0 - path_affinity[:, None]) * base_log_density
        + path_affinity[:, None] * collision_log_density
    )
    path_affinity_log_density -= np.max(
        path_affinity_log_density, axis=1, keepdims=True)
    path_affinity_mass = reference * np.exp(path_affinity_log_density)
    path_affinity_mass /= np.sum(
        path_affinity_mass, axis=1, keepdims=True)
    path_fidelity = path_affinity * path_affinity
    path_fidelity_log_density = (
        (1.0 - path_fidelity[:, None]) * base_log_density
        + path_fidelity[:, None] * collision_log_density
    )
    path_fidelity_log_density -= np.max(
        path_fidelity_log_density, axis=1, keepdims=True)
    path_fidelity_mass = reference * np.exp(path_fidelity_log_density)
    path_fidelity_mass /= np.sum(
        path_fidelity_mass, axis=1, keepdims=True)

    # State-marginal agreement does not measure uncertainty in the transport
    # map: two path laws can have nearly identical vertex marginals while
    # assigning their mass to different predecessor-successor couplings.  The
    # exact posterior edge plans retain that missing random object,
    #
    #   Xi_t(i,j) propto alpha_t(i) K_t(i,j) L_{t+1}(j) beta_{t+1}(j).
    #
    # Compare the ordinary and order-two-contracted edge plans by Hellinger
    # fidelity.  A vertex history uses both incident transports, so its
    # survival probability is the product of the two incident fidelities
    # (Hellinger fidelity is multiplicative on product laws).  This produces a
    # parameter-free density-geodesic coordinate that includes uncertainty
    # about transport itself, not merely uncertainty about the current state.
    transport_edge_fidelity = np.empty(samples - 1, dtype=np.float64)
    for index in range(samples - 1):
        edge_plan = (
            forward[index][:, None]
            * forward_kernels[index]
            * likelihood[index + 1][None, :]
            * backward[index + 1][None, :]
        )
        edge_plan /= np.sum(edge_plan)
        collision_edge_plan = (
            path_collision_forward[index][:, None]
            * collision_kernels[index]
            * likelihood[index + 1][None, :] ** 2
            * path_collision_backward[index + 1][None, :]
        )
        collision_edge_plan /= np.sum(collision_edge_plan)
        edge_affinity = float(np.sum(np.sqrt(
            edge_plan * collision_edge_plan)))
        transport_edge_fidelity[index] = np.clip(
            edge_affinity * edge_affinity, 0.0, 1.0)
    transport_vertex_fidelity = np.ones(samples, dtype=np.float64)
    transport_vertex_fidelity[0] = transport_edge_fidelity[0]
    transport_vertex_fidelity[-1] = transport_edge_fidelity[-1]
    if samples > 2:
        transport_vertex_fidelity[1:-1] = (
            transport_edge_fidelity[:-1] * transport_edge_fidelity[1:])
    transport_fidelity_log_density = (
        (1.0 - transport_vertex_fidelity[:, None]) * base_log_density
        + transport_vertex_fidelity[:, None] * collision_log_density
    )
    transport_fidelity_log_density -= np.max(
        transport_fidelity_log_density, axis=1, keepdims=True)
    transport_fidelity_mass = reference * np.exp(
        transport_fidelity_log_density)
    transport_fidelity_mass /= np.sum(
        transport_fidelity_mass, axis=1, keepdims=True)

    # The estimator's two legitimate histories are both already contracted:
    # the pointwise collision of the smoothed marginal and the collision of
    # complete paths before marginalization.  Interpolate between those laws,
    # not between an uncontracted posterior and a contracted path law.
    local_collision_log_density = 2.0 * base_log_density
    path_fidelity_order = np.argsort(prediction, axis=1, kind="stable")
    ordered_path_fidelity_mass = np.take_along_axis(
        path_fidelity_mass, path_fidelity_order, axis=1)
    ordered_prediction = np.take_along_axis(
        prediction, path_fidelity_order, axis=1)
    path_fidelity_median_index = np.argmax(
        np.cumsum(ordered_path_fidelity_mass, axis=1) >= 0.5, axis=1)
    path_fidelity_median = np.take_along_axis(
        ordered_prediction,
        path_fidelity_median_index[:, None],
        axis=1,
    )[:, 0]
    path_fidelity_deviation = prediction - path_fidelity_median[:, None]
    path_fidelity_first_moment = np.sum(
        path_fidelity_mass * np.abs(path_fidelity_deviation), axis=1)
    path_fidelity_second_moment = np.sum(
        path_fidelity_mass * path_fidelity_deviation ** 2, axis=1)
    path_fidelity_participation = np.divide(
        path_fidelity_first_moment ** 2,
        path_fidelity_second_moment,
        out=np.ones_like(path_fidelity_first_moment),
        where=path_fidelity_second_moment > np.finfo(float).tiny,
    )
    path_fidelity_participation = np.clip(
        path_fidelity_participation, 0.0, 1.0)

    def contracted_history_geodesic(coordinate: np.ndarray) -> np.ndarray:
        log_density = (
            (1.0 - coordinate[:, None]) * local_collision_log_density
            + coordinate[:, None] * collision_log_density
        )
        log_density -= np.max(log_density, axis=1, keepdims=True)
        history_mass = reference * np.exp(log_density)
        history_mass /= np.sum(history_mass, axis=1, keepdims=True)
        return history_mass

    transport_plan_history_mass = contracted_history_geodesic(
        transport_vertex_fidelity)
    coherent_history_survival = (
        transport_vertex_fidelity * (1.0 - path_fidelity_participation))
    distributed_history_survival = (
        transport_vertex_fidelity * path_fidelity_participation)
    self_consistent_transport_mass = contracted_history_geodesic(
        coherent_history_survival)
    distributed_transport_mass = contracted_history_geodesic(
        distributed_history_survival)
    self_consistent_log_density = np.log(np.maximum(
        self_consistent_transport_mass
        / np.maximum(reference, np.finfo(float).tiny),
        np.finfo(float).tiny,
    ))
    transport_fidelity_density = np.log(np.maximum(
        transport_fidelity_mass
        / np.maximum(reference, np.finfo(float).tiny),
        np.finfo(float).tiny,
    ))
    action_contracting_log_density = (
        (1.0 - path_fidelity_participation[:, None])
        * self_consistent_log_density
        + path_fidelity_participation[:, None]
        * transport_fidelity_density
    )
    action_contracting_log_density -= np.max(
        action_contracting_log_density, axis=1, keepdims=True)
    action_contracting_transport_mass = reference * np.exp(
        action_contracting_log_density)
    action_contracting_transport_mass /= np.sum(
        action_contracting_transport_mass, axis=1, keepdims=True)
    two_history_participation = path_fidelity_participation ** 2
    two_history_action_log_density = (
        (1.0 - two_history_participation[:, None])
        * self_consistent_log_density
        + two_history_participation[:, None]
        * transport_fidelity_density
    )
    two_history_action_log_density -= np.max(
        two_history_action_log_density, axis=1, keepdims=True)
    two_history_action_transport_mass = reference * np.exp(
        two_history_action_log_density)
    two_history_action_transport_mass /= np.sum(
        two_history_action_transport_mass, axis=1, keepdims=True)
    right_mass = backward * likelihood
    right_mass /= np.sum(right_mass, axis=1, keepdims=True)
    symmetric_parent_mass = np.sqrt(forward * right_mass)
    symmetric_parent_mass /= np.sum(
        symmetric_parent_mass, axis=1, keepdims=True)
    joint_score = forward_score + backward_score
    joint_score -= np.max(joint_score, axis=1, keepdims=True)
    hj_joint_mass = reference * np.exp(joint_score)
    hj_joint_mass /= np.sum(hj_joint_mass, axis=1, keepdims=True)
    hj_joint_collision_mass = reference * np.exp(2.0 * joint_score)
    hj_joint_collision_mass /= np.sum(
        hj_joint_collision_mass, axis=1, keepdims=True)

    # Retain uncertainty about the transport itself.  The ordinary smoothed
    # branch law above marginalizes predecessor and successor identities at
    # the current vertex.  Here they meet through the shared current branch.
    # For branch j, the exact chain posterior factors as
    #
    #   p(i,k | j,y) = p(i | j,y) p(k | j,y),
    #
    # where i and k are the predecessor and successor branches.  Transport
    # both outer phase states to the current point, measure their expected
    # determinant-one Sasaki distance, and use its reciprocal as an action
    # likelihood.  No scalar balance of the forward/backward marginals can
    # substitute for this coupled path law: equally diffuse contradictory
    # histories have large phase action here.
    coupled_phase_action = np.ones_like(prediction)
    coupled_phase_population = np.ones_like(prediction)
    for index in range(1, samples - 1):
        previous_kernel = forward_kernels[index - 1]
        next_kernel = forward_kernels[index]

        predecessor_joint = forward[index - 1][:, None] * previous_kernel
        predecessor_total = np.sum(
            predecessor_joint, axis=0, keepdims=True)
        predecessor_conditional = np.divide(
            predecessor_joint,
            predecessor_total,
            out=np.full_like(predecessor_joint, 1.0 / branches),
            where=predecessor_total > np.finfo(float).tiny,
        )

        successor_evidence = likelihood[index + 1] * backward[index + 1]
        successor_joint = next_kernel * successor_evidence[None, :]
        successor_total = np.sum(
            successor_joint, axis=1, keepdims=True)
        successor_conditional = np.divide(
            successor_joint,
            successor_total,
            out=np.full_like(successor_joint, 1.0 / branches),
            where=successor_total > np.finfo(float).tiny,
        )

        left_phase = np.column_stack((
            prediction[index - 1] + jet[index - 1],
            jet[index - 1],
        ))
        right_phase = np.column_stack((
            prediction[index + 1] - jet[index + 1],
            jet[index + 1],
        ))
        left_weight = forward[index - 1] / np.sum(forward[index - 1])
        right_weight = successor_evidence / np.sum(successor_evidence)
        phase_state = np.concatenate((left_phase, right_phase), axis=0)
        phase_weight = 0.5 * np.concatenate((left_weight, right_weight))
        phase_center = phase_weight @ phase_state
        centered_phase = phase_state - phase_center
        phase_covariance = (
            centered_phase * phase_weight[:, None]
        ).T @ centered_phase
        phase_eigenvalue, phase_eigenvector = np.linalg.eigh(
            phase_covariance)
        phase_precision_eigenvalue = 1.0 / np.maximum(
            phase_eigenvalue, floor * floor)
        phase_precision_eigenvalue /= np.sqrt(
            np.prod(phase_precision_eigenvalue))
        phase_precision = (
            phase_eigenvector * phase_precision_eigenvalue[None, :]
        ) @ phase_eigenvector.T
        phase_defect = left_phase[:, None, :] - right_phase[None, :, :]
        phase_distance = np.sqrt(np.maximum(np.einsum(
            "ika,ab,ikb->ik", phase_defect, phase_precision, phase_defect
        ), 0.0))

        # (pred^T D)[j,k] followed by the successor conditional for the
        # same j evaluates E[d(Q^-,Q^+) | current branch j].
        conditional_distance = predecessor_conditional.T @ phase_distance
        coupled_phase_action[index] = np.sum(
            conditional_distance * successor_conditional, axis=1)
        predecessor_population = 1.0 / np.sum(
            predecessor_conditional * predecessor_conditional, axis=0)
        successor_population = 1.0 / np.sum(
            successor_conditional * successor_conditional, axis=1)
        coupled_phase_population[index] = np.sqrt(
            predecessor_population * successor_population)

    coupled_score = joint_score - np.log(np.maximum(
        coupled_phase_action, floor))
    coupled_score -= np.max(coupled_score, axis=1, keepdims=True)
    hj_coupled_phase_mass = reference * np.exp(coupled_score)
    hj_coupled_phase_mass /= np.sum(
        hj_coupled_phase_mass, axis=1, keepdims=True)
    hj_coupled_phase_collision_mass = reference * np.exp(
        2.0 * coupled_score)
    hj_coupled_phase_collision_mass /= np.sum(
        hj_coupled_phase_collision_mass, axis=1, keepdims=True)

    # Reciprocal phase action has a false flat-history null direction.  Apply
    # the minimum-information correction that restores predictive coverage:
    # among exponential tilts of the contracted law, choose the unique one
    # whose second moment about the *transported baseline center* equals the
    # pre-contraction moment.  This is a numerical I-projection onto a moment
    # half-space, not an amplitude blend or a selected strength.  If coverage
    # did not contract, the identity projection is retained.
    baseline_center = np.sum(
        hj_joint_mass * prediction, axis=1, keepdims=True)
    coverage_coordinate = (prediction - baseline_center) ** 2
    (
        hj_coupled_phase_coverage_mass,
        baseline_coverage,
        contracted_coverage,
        restored_coverage,
    ) = _minimum_information_coverage_projection(
        hj_joint_mass, hj_coupled_phase_mass, coverage_coordinate)

    phase_state = np.stack((prediction, jet), axis=-1)
    phase_center = np.sum(
        hj_joint_mass[..., None] * phase_state, axis=1, keepdims=True)
    centered_phase = phase_state - phase_center
    phase_coverage_coordinate = np.empty_like(prediction)
    for index in range(samples):
        covariance = np.einsum(
            "k,ka,kb->ab",
            hj_joint_mass[index],
            centered_phase[index],
            centered_phase[index],
        )
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        precision_eigenvalue = 1.0 / np.maximum(
            eigenvalue, floor * floor)
        precision_eigenvalue /= np.sqrt(np.prod(precision_eigenvalue))
        precision = (
            eigenvector * precision_eigenvalue[None, :]
        ) @ eigenvector.T
        phase_coverage_coordinate[index] = np.einsum(
            "ka,ab,kb->k",
            centered_phase[index],
            precision,
            centered_phase[index],
        )
    (
        hj_coupled_phase_bundle_coverage_mass,
        baseline_phase_coverage,
        contracted_phase_coverage,
        restored_phase_coverage,
    ) = _minimum_information_coverage_projection(
        hj_joint_mass,
        hj_coupled_phase_mass,
        phase_coverage_coordinate,
    )

    baseline_phase_action = np.sum(
        hj_joint_mass * coupled_phase_action, axis=1)
    contracted_phase_action = np.sum(
        hj_coupled_phase_mass * coupled_phase_action, axis=1)
    phase_action_tolerance = math.sqrt(np.finfo(float).eps) * np.maximum(
        baseline_phase_action, floor)
    if np.any(
        contracted_phase_action > baseline_phase_action + phase_action_tolerance
    ):
        raise RuntimeError("coupled transport posterior increased phase action")
    forward_density = forward / np.maximum(
        reference, np.finfo(float).tiny)
    backward_density = right_mass / np.maximum(
        reference, np.finfo(float).tiny)
    parent_fraction = np.divide(
        forward_density,
        forward_density + backward_density,
        out=np.full_like(forward_density, 0.5),
        where=(forward_density + backward_density) > np.finfo(float).tiny,
    )
    effective_phase_histories = 1.0 / (
        parent_fraction * parent_fraction
        + (1.0 - parent_fraction) * (1.0 - parent_fraction))
    hj_phase_collision_mass = reference * np.exp(
        effective_phase_histories * joint_score)
    hj_phase_collision_mass /= np.sum(
        hj_phase_collision_mass, axis=1, keepdims=True)
    return {
        "mass": mass,
        "forward_mass": forward,
        "backward_mass": right_mass,
        "symmetric_parent_mass": symmetric_parent_mass,
        "hj_joint_mass": hj_joint_mass,
        "hj_joint_collision_mass": hj_joint_collision_mass,
        "hj_phase_collision_mass": hj_phase_collision_mass,
        "hj_coupled_phase_mass": hj_coupled_phase_mass,
        "hj_coupled_phase_collision_mass": hj_coupled_phase_collision_mass,
        "hj_coupled_phase_coverage_mass": hj_coupled_phase_coverage_mass,
        "hj_coupled_phase_bundle_coverage_mass": (
            hj_coupled_phase_bundle_coverage_mass),
        "hj_viterbi_section": viterbi_section,
        "hj_viterbi_branch": viterbi_branch,
        "hj_viterbi_predecessor": viterbi_predecessor,
        "posterior_characteristic_section": posterior_characteristic_section,
        "posterior_characteristic_branch": posterior_branch,
        "posterior_characteristic_predecessor": posterior_predecessor,
        "path_collision_mass": path_collision_mass,
        "path_affinity_mass": path_affinity_mass,
        "path_fidelity_mass": path_fidelity_mass,
        "transport_fidelity_mass": transport_fidelity_mass,
        "transport_plan_history_mass": transport_plan_history_mass,
        "self_consistent_transport_mass": self_consistent_transport_mass,
        "distributed_transport_mass": distributed_transport_mass,
        "action_contracting_transport_mass": (
            action_contracting_transport_mass),
        "two_history_action_transport_mass": (
            two_history_action_transport_mass),
        "transport_edge_fidelity": transport_edge_fidelity,
        "transport_vertex_fidelity": transport_vertex_fidelity,
        "path_fidelity_participation": path_fidelity_participation,
        "coupled_phase_action": coupled_phase_action,
        "coupled_phase_population": coupled_phase_population,
        "prediction": prediction,
        "jet": jet,
        "curvature": curvature,
        "curvature_coordinate": np.asarray(
            law.get("curvature_coordinate", np.zeros(branches)),
            dtype=np.float64,
        ),
        "midpoint_prediction": np.asarray(
            law.get("midpoint_prediction", prediction),
            dtype=np.float64,
        ),
        "reference_mass": reference_law,
        "edge_precision": (
            np.stack(edge_precisions, axis=0)
            if edge_precisions
            else np.broadcast_to(
                np.eye(bundle_dimension),
                (samples - 1, bundle_dimension, bundle_dimension),
            ).copy()
        ),
        "likelihood": likelihood,
        "forward_kernels": np.stack(forward_kernels, axis=0),
        "connection_authority": np.asarray(
            connection_authorities, dtype=np.float64),
        "connection_mean_defect": (
            np.stack(connection_mean_defects, axis=0)
            if connection_mean_defects
            else np.empty((0, bundle_dimension), dtype=np.float64)
        ),
        "connection_covariance": (
            np.stack(connection_covariances, axis=0)
            if connection_covariances
            else np.empty(
                (0, bundle_dimension, bundle_dimension), dtype=np.float64)
        ),
        "source_identity": np.asarray(
            law.get("source_identity", np.empty(
                (samples, branches, 0), dtype=np.int64)),
            dtype=np.int64,
        ),
    }, {
        "state": "bidirectional positive lineage on the jet bundle",
        "bundle_coordinates": (
            "transported value, first jet, curvature, and conserved residual"
            if uses_curvature
            else (
                "transported value, jet, and conserved residual"
                if bundle_metric != "signal_euclidean"
                else "transported value and jet"
            )
        ),
        "bundle_metric": bundle_metric,
        "transition_normalization": transition_normalization,
        "preserve_branch_role": bool(preserve_branch_role),
        "mean_information_anisotropy": (
            float(np.mean(information_anisotropies))
            if information_anisotropies else 1.0
        ),
        "mean_lineage_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=1))),
        "mean_transition_population": float(np.mean(transition_populations)),
        "log_path_evidence": float(log_path_evidence),
        "mean_connection_authority": (
            float(np.mean(connection_authorities))
            if connection_authorities else 0.0
        ),
        "mean_connection_covariance_trace": (
            float(np.mean([
                np.trace(value) for value in connection_covariances
            ]))
            if connection_covariances else 0.0
        ),
        "mean_connection_family_disagreement": (
            float(np.mean(connection_family_disagreements))
            if connection_family_disagreements else 0.0
        ),
        "mean_connection_family_fidelity": (
            float(np.mean(connection_family_fidelities))
            if connection_family_fidelities else 1.0
        ),
        "phase_collision_order": {
            "identity": "1 / (t^2 + (1-t)^2)",
            "minimum": float(np.min(effective_phase_histories)),
            "mean": float(np.mean(effective_phase_histories)),
            "maximum": float(np.max(effective_phase_histories)),
            "parent_fraction": (
                "relative forward/backward phase-arrival density"
            ),
        },
        "coupled_transport_posterior": {
            "state": (
                "predecessor-current-successor branch coupling after "
                "parallel phase transport"
            ),
            "action": (
                "conditional expected determinant-one value-jet distance "
                "between two-sided transported histories"
            ),
            "update": "joint path density divided by coupled phase action",
            "mean_baseline_phase_action": float(np.mean(
                baseline_phase_action[1:-1])),
            "mean_contracted_phase_action": float(np.mean(
                contracted_phase_action[1:-1])),
            "maximum_action_increase": float(np.max(
                contracted_phase_action - baseline_phase_action)),
            "mean_conditional_transport_population": float(np.mean(
                coupled_phase_population[1:-1])),
            "duration": "none",
            "noise_model": "none",
        },
        "coupled_transport_coverage": {
            "state": (
                "minimum-information projection of the contracted path law "
                "onto transported predictive second-moment coverage"
            ),
            "mean_baseline_coverage": float(np.mean(baseline_coverage)),
            "mean_contracted_coverage": float(np.mean(contracted_coverage)),
            "mean_restored_coverage": float(np.mean(restored_coverage)),
            "maximum_coverage_deficit": float(np.max(
                baseline_coverage - restored_coverage)),
            "physical_parameters": "none",
        },
        "coupled_transport_phase_coverage": {
            "state": (
                "minimum-information projection onto determinant-one "
                "transported value-jet second-moment coverage"
            ),
            "mean_baseline_coverage": float(np.mean(
                baseline_phase_coverage)),
            "mean_contracted_coverage": float(np.mean(
                contracted_phase_coverage)),
            "mean_restored_coverage": float(np.mean(
                restored_phase_coverage)),
            "maximum_coverage_deficit": float(np.max(
                baseline_phase_coverage - restored_phase_coverage)),
            "physical_parameters": "none",
        },
        "global_characteristic_section": {
            "state": (
                "single representation-density-invariant maximum-action "
                "history traced through every base point"
            ),
            "readout_order": (
                "global branch traceback before scalar projection"
            ),
            "mean_absolute_jet_defect": float(np.mean(np.abs(
                np.diff(viterbi_section)
                - jet[np.arange(samples), viterbi_branch][:-1]
            ))),
            "branch_changes": int(np.count_nonzero(np.diff(
                viterbi_branch))),
            "physical_parameters": "none",
        },
        "posterior_characteristic_section": {
            "state": (
                "global characteristic remarched through the complete "
                "sum-product transport posterior density"
            ),
            "readout_order": (
                "marginalize transport uncertainty, remarch its density, "
                "then trace one coherent section"
            ),
            "mean_absolute_jet_defect": float(np.mean(np.abs(
                np.diff(posterior_characteristic_section)
                - jet[np.arange(samples), posterior_branch][:-1]
            ))),
            "branch_changes": int(np.count_nonzero(np.diff(
                posterior_branch))),
            "physical_parameters": "none",
        },
        "global_path_collision": {
            "state": (
                "order-two collision of complete representation-invariant "
                "transport histories before branch marginalization"
            ),
            "transition_measure": "h_child * (K / h_child)^2",
            "likelihood_density_order": 2,
            "mean_terminal_population": float(np.mean(
                1.0 / np.sum(
                    path_collision_mass * path_collision_mass, axis=1))),
            "physical_parameters": "none",
        },
        "path_affinity_geodesic": {
            "state": (
                "Fisher--Rao density geodesic from the full path posterior "
                "to its order-two collision"
            ),
            "coordinate": (
                "Hellinger affinity of ordinary and collided path marginals"
            ),
            "minimum_affinity": float(np.min(path_affinity)),
            "mean_affinity": float(np.mean(path_affinity)),
            "maximum_affinity": float(np.max(path_affinity)),
            "physical_parameters": "none",
        },
        "path_fidelity_geodesic": {
            "state": (
                "Fisher--Rao density geodesic with squared Hellinger "
                "affinity as two-history survival probability"
            ),
            "coordinate": "F = (integral sqrt(p q))^2",
            "minimum_fidelity": float(np.min(path_fidelity)),
            "mean_fidelity": float(np.mean(path_fidelity)),
            "maximum_fidelity": float(np.max(path_fidelity)),
            "physical_parameters": "none",
        },
        "transport_plan_fidelity": {
            "state": (
                "Hellinger fidelity between ordinary and contracted "
                "posterior transport plans"
            ),
            "edge_plan": (
                "Xi_t(i,j) proportional to alpha_t(i) K_t(i,j) "
                "L_{t+1}(j) beta_{t+1}(j)"
            ),
            "vertex_survival": (
                "product of incident edge fidelities; boundary has one edge"
            ),
            "minimum_edge_fidelity": float(np.min(
                transport_edge_fidelity)),
            "mean_edge_fidelity": float(np.mean(transport_edge_fidelity)),
            "maximum_edge_fidelity": float(np.max(
                transport_edge_fidelity)),
            "minimum_vertex_survival": float(np.min(
                transport_vertex_fidelity)),
            "mean_vertex_survival": float(np.mean(
                transport_vertex_fidelity)),
            "maximum_vertex_survival": float(np.max(
                transport_vertex_fidelity)),
            "physical_parameters": "none",
        },
        "self_consistent_transport": {
            "state": (
                "density geodesic between local collision and complete-path "
                "collision, including uncertainty about the edge plans"
            ),
            "local_history": "h * (posterior / h)^2",
            "global_history": "complete path collision before marginalization",
            "plan_coordinate": "product of incident edge fidelities",
            "coherent_history_coordinate": (
                "plan coordinate times one minus barycentric participation"
            ),
            "distributed_control_coordinate": (
                "plan coordinate times barycentric participation"
            ),
            "minimum_coherent_history_survival": float(np.min(
                coherent_history_survival)),
            "mean_coherent_history_survival": float(np.mean(
                coherent_history_survival)),
            "maximum_coherent_history_survival": float(np.max(
                coherent_history_survival)),
            "physical_parameters": "none",
        },
        "action_contracting_transport": {
            "state": (
                "fibre-participation geodesic between coherent-history and "
                "transport-uncertainty density laws"
            ),
            "coordinate": (
                "(E|Z-median|)^2 / E|Z-median|^2 under path fidelity"
            ),
            "mean_transport_uncertainty_participation": float(np.mean(
                path_fidelity_participation)),
            "physical_parameters": "none",
            "status": (
                "vertexwise marginal experiment; joint path-law lift pending"
            ),
        },
        "two_history_action_transport": {
            "state": (
                "order-two survival of distributed fibre participation on "
                "the action-contracting transport-history geodesic"
            ),
            "coordinate": (
                "((E|Z-median|)^2 / E|Z-median|^2)^2"
            ),
            "mean_two_history_participation": float(np.mean(
                two_history_participation)),
            "physical_parameters": "none",
            "status": (
                "vertexwise marginal experiment; joint path-law lift pending"
            ),
        },
        "mean_hj_joint_population": float(np.mean(
            1.0 / np.sum(hj_joint_mass * hj_joint_mass, axis=1))),
        "target_value_enters_local_action": bool(local_diagnostic.get(
            "target_value_enters_interior_action", True)),
        "boundary_condition": "symmetric forward/backward messages",
        "local_law": local_diagnostic,
    }


def signal_lineage_branch_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rejected control transporting value/jet without the residual graph."""
    return _lineage_branch_transport_1d(
        observation, bundle_metric="signal_euclidean")


def complete_participation_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport the complete algebra of value/jet/residual participation."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="complete_participation",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "complete unit-multiplicity value/jet/residual participation algebra"
    )
    diagnostic["participation_algebra"] = {
        "identity": "(1 + K_value)(1 + K_jet)(1 + K_residual) - 1",
        "primitive": "1 / sqrt(1 + precision_coordinate * defect^2)",
        "coordinate_precision": (
            "marginal covariance precision normalized to determinant one"
        ),
        "selected_coordinates": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def distinct_ancestry_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport branches whose support is reproduced by disjoint ancestry."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        transition_normalization="markov",
        particle_law_builder=distinct_ancestry_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "target-free determinant-one consensus across disjoint source sets"
    )
    return law, diagnostic


def transport_distribution_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport relative to the inferred mean and covariance of each edge."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_distribution",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "posterior mean-and-covariance connection on value/jet/residual phase"
    )
    diagnostic["transport_distribution"] = {
        "mean": "E[q_left] - E[q_right]",
        "covariance": "Cov(q_left) + Cov(q_right)",
        "precision_normalization": "determinant one",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def transport_covariance_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Keep the zero connection while transporting its inferred covariance."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_covariance",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "zero-mean connection with posterior value/jet/residual defect metric"
    )
    diagnostic["transport_distribution"] = {
        "mean": "zero exact parallel-transport defect",
        "covariance": "Cov(q_left) + Cov(q_right)",
        "precision_normalization": "determinant one",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def ancestry_connection_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Infer connection drift from agreement among causal ancestry families."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_ancestry_connection",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "precision-fused bilateral/left/right connection posterior"
    )
    diagnostic["transport_distribution"] = {
        "family_states": "bilateral, left-causal, and right-causal",
        "family_mean": "E_f[q_left] - E_f[q_right]",
        "mean_fusion": "inverse sum of family precisions",
        "covariance": (
            "complete particle defect covariance plus covariance of family "
            "connection means"
        ),
        "family_authority": "rho_f = s_f / (1 + s_f)",
        "family_fidelity": "Gaussian Hellinger fidelity for each family pair",
        "authority": (
            "product of three family authorities and three pair fidelities"
        ),
        "mean": "rho * precision-fused family connection",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def self_consistent_connection_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Infer connection drift only to the degree certified by its covariance."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_self_consistent",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "Hotelling-authorized connection on value/jet/residual phase"
    )
    diagnostic["transport_distribution"] = {
        "mean": "rho * (E[q_left] - E[q_right])",
        "covariance": "Cov(q_left) + Cov(q_right)",
        "authority_action": "s = mu^T covariance^-1 mu",
        "authority": "rho = s / (1 + s)",
        "precision_normalization": "determinant one after authority action",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def collision_consistent_connection_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Require two independent connection histories to authorize edge drift."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_connection_collision",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "order-two Hotelling-authorized value/jet/residual connection"
    )
    diagnostic["transport_distribution"] = {
        "mean": "rho^2 * (E[q_left] - E[q_right])",
        "covariance": "Cov(q_left) + Cov(q_right)",
        "authority_action": "s = mu^T covariance^-1 mu",
        "single_history_authority": "rho = s / (1 + s)",
        "connection_survival": "rho^2 for two independent histories",
        "precision_normalization": "determinant one after authority action",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def bidirectional_collision_connection_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Certify two transport histories independently at both edge endpoints."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_bidirectional_collision",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "bidirectional order-two Hotelling-authorized phase connection"
    )
    diagnostic["transport_distribution"] = {
        "mean": "rho^4 * (E[q_left] - E[q_right])",
        "covariance": "Cov(q_left) + Cov(q_right)",
        "single_history_authority": "rho = s / (1 + s)",
        "path_survival": (
            "two independent histories at each of two edge endpoints"
        ),
        "connection_survival": "rho^(2 histories * 2 endpoints)",
        "precision_normalization": "determinant one after authority action",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def gaussian_connection_potential_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Marginalize a continuous Gaussian law of transport connections.

    In the three-coordinate phase bundle, integrating reciprocal Mahalanobis
    action over the inferred Gaussian connection law is the exact Newton
    potential ``erf(r / sqrt(2)) / r``.  The removable origin singularity is
    represented by its analytic limit, not a selected physical scale.
    """
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="transport_gaussian_potential",
        transition_normalization="markov",
    )
    diagnostic["particle_law"] = (
        "continuous Gaussian posterior-predictive connection potential"
    )
    diagnostic["transport_distribution"] = {
        "mean": "E[q_left] - E[q_right]",
        "covariance": "Cov(q_left) + Cov(q_right)",
        "conductance": "erf(r / sqrt(2)) / r",
        "radius": "Mahalanobis distance from the inferred connection mean",
        "origin": "analytic limit sqrt(2 / pi)",
        "selected_noise_model": "none",
        "physical_parameters": "none",
    }
    return law, diagnostic


def gaussian_connection_potential_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the continuous connection law after bidirectional transport."""
    law, diagnostic = gaussian_connection_potential_lineage_transport_1d(
        observation)
    prediction = law["prediction"]
    mass = law["mass"]
    reference_mass = law["reference_mass"]
    reference = (
        np.broadcast_to(reference_mass, prediction.shape)
        if reference_mass.ndim == 1 else reference_mass)
    collision_mass = mass * mass / np.maximum(
        reference, np.finfo(float).tiny)
    collision_mass /= np.sum(collision_mass, axis=1, keepdims=True)
    path_mass = law["path_collision_mass"]
    return {
        "mean": np.sum(mass * prediction, axis=1),
        "collision_mean": np.sum(collision_mass * prediction, axis=1),
        "path_collision_mean": np.sum(path_mass * prediction, axis=1),
    }, diagnostic


def _connection_defect_ownership_1d(
    observation: np.ndarray,
    connection_mean: np.ndarray,
    connection_covariance: np.ndarray,
    *,
    include_covariance: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull adjacent connection-law disagreement back onto signal vertices.

    The Hotelling form compares adjacent connection means in their pooled
    covariance metric.  The Hellinger form additionally treats covariance as
    part of the transported random object.  Both are dimensionless and
    invariant under a common nonsingular change of bundle coordinates, apart
    from the representation-scale eigensafety floor.
    """
    line = _validate_line(observation)
    edges = line.size - 1
    if (
        connection_mean.shape != (edges, 3)
        or connection_covariance.shape != (edges, 3, 3)
    ):
        raise ValueError("connection law must align with signal edges")
    floor_squared = _representation_floor(line) ** 2

    regularized_covariance = np.empty_like(connection_covariance)
    log_determinant = np.empty(edges, dtype=np.float64)
    for edge in range(edges):
        eigenvalue, eigenvector = np.linalg.eigh(
            connection_covariance[edge])
        eigenvalue = np.maximum(eigenvalue, floor_squared)
        regularized_covariance[edge] = (
            eigenvector * eigenvalue[None, :]) @ eigenvector.T
        log_determinant[edge] = float(np.sum(np.log(eigenvalue)))

    survival = np.zeros(line.size, dtype=np.float64)
    action = np.zeros(line.size, dtype=np.float64)
    for vertex in range(1, line.size - 1):
        left = vertex - 1
        right = vertex
        pooled = 0.5 * (
            regularized_covariance[left]
            + regularized_covariance[right])
        delta = connection_mean[left] - connection_mean[right]
        mean_action = float(delta @ np.linalg.solve(pooled, delta))
        if include_covariance:
            _, pooled_log_determinant = np.linalg.slogdet(pooled)
            log_affinity = (
                0.25 * (
                    log_determinant[left] + log_determinant[right])
                - 0.5 * pooled_log_determinant
                - 0.125 * mean_action
            )
            probability_affinity = float(np.exp(min(log_affinity, 0.0)))
            survival[vertex] = np.clip(
                1.0 - probability_affinity, 0.0, 1.0)
            action[vertex] = -min(log_affinity, 0.0)
        else:
            action[vertex] = mean_action
            survival[vertex] = mean_action / (1.0 + mean_action)
    return survival, action


def _spd_geometric_midpoint(
    first: np.ndarray,
    second: np.ndarray,
    floor_squared: float,
) -> np.ndarray:
    """Return the affine-invariant midpoint of two positive covariances."""
    eigenvalue, eigenvector = np.linalg.eigh(first)
    eigenvalue = np.maximum(eigenvalue, floor_squared)
    square_root = (
        eigenvector * np.sqrt(eigenvalue)[None, :]) @ eigenvector.T
    inverse_square_root = (
        eigenvector * (1.0 / np.sqrt(eigenvalue))[None, :]) @ eigenvector.T
    relative = inverse_square_root @ second @ inverse_square_root
    relative_eigenvalue, relative_eigenvector = np.linalg.eigh(relative)
    relative_square_root = (
        relative_eigenvector
        * np.sqrt(np.maximum(
            relative_eigenvalue, floor_squared))[None, :]
    ) @ relative_eigenvector.T
    midpoint = square_root @ relative_square_root @ square_root
    return 0.5 * (midpoint + midpoint.T)


def _transported_gaussian_connection_contrast_1d(
    observation: np.ndarray,
    edge_mean: np.ndarray,
    edge_covariance: np.ndarray,
    source_identity: np.ndarray,
    branch_mass: np.ndarray,
    *,
    parallel_transport_connection: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare a vertex connection law with complete transported source laws.

    Mean and covariance remain coupled Gaussian state variables until every
    branch has transported its two exact source laws.  Only the resulting
    Gaussian Hellinger defects are marginalized over branch probability.
    """
    line = _validate_line(observation)
    samples, branches = branch_mass.shape
    if samples != line.size:
        raise ValueError("branch mass must align with the observation")
    if edge_mean.shape != (samples - 1, 3):
        raise ValueError("connection mean must align with edges")
    if edge_covariance.shape != (samples - 1, 3, 3):
        raise ValueError("connection covariance must align with edges")
    if source_identity.shape != (samples, branches, 2):
        raise ValueError("source identity must align with branch mass")
    floor_squared = _representation_floor(line) ** 2

    regularized_edge_covariance = np.empty_like(edge_covariance)
    for edge in range(samples - 1):
        eigenvalue, eigenvector = np.linalg.eigh(edge_covariance[edge])
        eigenvalue = np.maximum(eigenvalue, floor_squared)
        regularized_edge_covariance[edge] = (
            eigenvector * eigenvalue[None, :]) @ eigenvector.T

    vertex_mean = np.empty((samples, 3), dtype=np.float64)
    vertex_covariance = np.empty((samples, 3, 3), dtype=np.float64)
    vertex_mean[0] = edge_mean[0]
    vertex_mean[-1] = edge_mean[-1]
    vertex_covariance[0] = regularized_edge_covariance[0]
    vertex_covariance[-1] = regularized_edge_covariance[-1]
    for vertex in range(1, samples - 1):
        vertex_mean[vertex] = 0.5 * (
            edge_mean[vertex - 1] + edge_mean[vertex])
        vertex_covariance[vertex] = _spd_geometric_midpoint(
            regularized_edge_covariance[vertex - 1],
            regularized_edge_covariance[vertex],
            floor_squared,
        )

    survival = np.zeros(samples, dtype=np.float64)
    action = np.zeros(samples, dtype=np.float64)
    branch_survival_field = np.empty(
        (samples, branches), dtype=np.float64)
    for vertex in range(samples):
        local_covariance = vertex_covariance[vertex]
        _, local_logdet = np.linalg.slogdet(local_covariance)
        branch_survival = np.empty(branches, dtype=np.float64)
        branch_action = np.empty(branches, dtype=np.float64)
        for branch in range(branches):
            first_source, second_source = source_identity[vertex, branch]
            first_mean = vertex_mean[first_source]
            second_mean = vertex_mean[second_source]
            first_covariance = vertex_covariance[first_source]
            second_covariance = vertex_covariance[second_source]
            if parallel_transport_connection:
                first_transport = np.array([
                    [1.0, float(vertex - first_source), 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
                second_transport = np.array([
                    [1.0, float(vertex - second_source), 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
                first_mean = first_transport @ first_mean
                second_mean = second_transport @ second_mean
                first_covariance = (
                    first_transport @ first_covariance
                    @ first_transport.T)
                second_covariance = (
                    second_transport @ second_covariance
                    @ second_transport.T)
            context_mean = 0.5 * (
                first_mean + second_mean)
            context_covariance = _spd_geometric_midpoint(
                first_covariance,
                second_covariance,
                floor_squared,
            )
            pooled_covariance = 0.5 * (
                local_covariance + context_covariance)
            delta = vertex_mean[vertex] - context_mean
            mean_action = float(
                delta @ np.linalg.solve(pooled_covariance, delta))
            _, context_logdet = np.linalg.slogdet(context_covariance)
            _, pooled_logdet = np.linalg.slogdet(pooled_covariance)
            log_affinity = (
                0.25 * (local_logdet + context_logdet)
                - 0.5 * pooled_logdet
                - 0.125 * mean_action
            )
            branch_action[branch] = -min(log_affinity, 0.0)
            branch_survival[branch] = np.clip(
                1.0 - np.exp(min(log_affinity, 0.0)), 0.0, 1.0)
        survival[vertex] = branch_mass[vertex] @ branch_survival
        action[vertex] = branch_mass[vertex] @ branch_action
        branch_survival_field[vertex] = branch_survival
    return survival, action, branch_survival_field


def _connection_geodesic_acceleration_1d(
    observation: np.ndarray,
    edge_mean: np.ndarray,
    edge_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Hellinger defect from the connection geodesic equation.

    The connection law at an interior vertex is compared with the product
    geodesic midpoint of its neighboring laws: Euclidean midpoint for the
    mean and the affine-invariant SPD midpoint for covariance.  This is a
    parameter-free second covariant difference of the random connection.
    """
    line = _validate_line(observation)
    samples = line.size
    if edge_mean.shape != (samples - 1, 3):
        raise ValueError("connection mean must align with edges")
    if edge_covariance.shape != (samples - 1, 3, 3):
        raise ValueError("connection covariance must align with edges")
    floor_squared = _representation_floor(line) ** 2
    regularized_edge_covariance = np.empty_like(edge_covariance)
    for edge in range(samples - 1):
        eigenvalue, eigenvector = np.linalg.eigh(edge_covariance[edge])
        eigenvalue = np.maximum(eigenvalue, floor_squared)
        regularized_edge_covariance[edge] = (
            eigenvector * eigenvalue[None, :]) @ eigenvector.T

    vertex_mean = np.empty((samples, 3), dtype=np.float64)
    vertex_covariance = np.empty((samples, 3, 3), dtype=np.float64)
    vertex_mean[0] = edge_mean[0]
    vertex_mean[-1] = edge_mean[-1]
    vertex_covariance[0] = regularized_edge_covariance[0]
    vertex_covariance[-1] = regularized_edge_covariance[-1]
    for vertex in range(1, samples - 1):
        vertex_mean[vertex] = 0.5 * (
            edge_mean[vertex - 1] + edge_mean[vertex])
        vertex_covariance[vertex] = _spd_geometric_midpoint(
            regularized_edge_covariance[vertex - 1],
            regularized_edge_covariance[vertex],
            floor_squared,
        )

    survival = np.zeros(samples, dtype=np.float64)
    action = np.zeros(samples, dtype=np.float64)
    for vertex in range(1, samples - 1):
        predicted_mean = 0.5 * (
            vertex_mean[vertex - 1] + vertex_mean[vertex + 1])
        predicted_covariance = _spd_geometric_midpoint(
            vertex_covariance[vertex - 1],
            vertex_covariance[vertex + 1],
            floor_squared,
        )
        pooled_covariance = 0.5 * (
            vertex_covariance[vertex] + predicted_covariance)
        delta = vertex_mean[vertex] - predicted_mean
        mean_action = float(
            delta @ np.linalg.solve(pooled_covariance, delta))
        _, local_logdet = np.linalg.slogdet(vertex_covariance[vertex])
        _, predicted_logdet = np.linalg.slogdet(predicted_covariance)
        _, pooled_logdet = np.linalg.slogdet(pooled_covariance)
        log_affinity = (
            0.25 * (local_logdet + predicted_logdet)
            - 0.5 * pooled_logdet
            - 0.125 * mean_action
        )
        action[vertex] = -min(log_affinity, 0.0)
        survival[vertex] = np.clip(
            1.0 - np.exp(min(log_affinity, 0.0)), 0.0, 1.0)
    return survival, action


def _bernoulli_geodesic_residual_1d(
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Hellinger residual from a Bernoulli midpoint equation."""
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 1 or value.size < 3:
        raise ValueError("Bernoulli phase field must be a one-dimensional path")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("Bernoulli phase field must lie in the unit interval")
    context = np.zeros_like(value)
    context[1:-1] = 0.5 * (value[:-2] + value[2:])
    affinity = np.ones_like(value)
    affinity[1:-1] = (
        np.sqrt(value[1:-1] * context[1:-1])
        + np.sqrt((1.0 - value[1:-1]) * (1.0 - context[1:-1]))
    )
    affinity = np.clip(affinity, 0.0, 1.0)
    return 1.0 - affinity, -np.log(np.maximum(
        affinity, np.finfo(float).tiny))


def _spd_square_root_pair(
    covariance: np.ndarray,
    floor_squared: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric square-root and inverse square-root matrices."""
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvalue = np.maximum(eigenvalue, floor_squared)
    square_root = (
        eigenvector * np.sqrt(eigenvalue)[None, :]) @ eigenvector.T
    inverse_square_root = (
        eigenvector * (1.0 / np.sqrt(eigenvalue))[None, :]
    ) @ eigenvector.T
    return square_root, inverse_square_root


def _symmetric_matrix_logarithm(
    matrix: np.ndarray,
    floor_squared: float,
) -> np.ndarray:
    eigenvalue, eigenvector = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (
        eigenvector
        * np.log(np.maximum(eigenvalue, floor_squared))[None, :]
    ) @ eigenvector.T


def _symmetric_tangent_coordinates(matrix: np.ndarray) -> np.ndarray:
    """Vectorize Sym(3) isometrically for the Frobenius inner product."""
    root_two = math.sqrt(2.0)
    return np.array((
        matrix[0, 0],
        matrix[1, 1],
        matrix[2, 2],
        root_two * matrix[0, 1],
        root_two * matrix[0, 2],
        root_two * matrix[1, 2],
    ))


def _transported_connection_tangent_acceleration_1d(
    observation: np.ndarray,
    edge_mean: np.ndarray,
    edge_covariance: np.ndarray,
    source_identity: np.ndarray,
    branch_mass: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Transport the full connection-acceleration tangent through ancestry."""
    line = _validate_line(observation)
    samples, branches = branch_mass.shape
    if samples != line.size:
        raise ValueError("branch mass must align with the observation")
    if edge_mean.shape != (samples - 1, 3):
        raise ValueError("connection mean must align with edges")
    if edge_covariance.shape != (samples - 1, 3, 3):
        raise ValueError("connection covariance must align with edges")
    if source_identity.shape != (samples, branches, 2):
        raise ValueError("source identity must align with branch mass")
    floor_squared = _representation_floor(line) ** 2

    edge_covariance_regularized = np.empty_like(edge_covariance)
    for edge in range(samples - 1):
        eigenvalue, eigenvector = np.linalg.eigh(edge_covariance[edge])
        eigenvalue = np.maximum(eigenvalue, floor_squared)
        edge_covariance_regularized[edge] = (
            eigenvector * eigenvalue[None, :]) @ eigenvector.T
    vertex_mean = np.empty((samples, 3), dtype=np.float64)
    vertex_covariance = np.empty((samples, 3, 3), dtype=np.float64)
    vertex_mean[0] = edge_mean[0]
    vertex_mean[-1] = edge_mean[-1]
    vertex_covariance[0] = edge_covariance_regularized[0]
    vertex_covariance[-1] = edge_covariance_regularized[-1]
    for vertex in range(1, samples - 1):
        vertex_mean[vertex] = 0.5 * (
            edge_mean[vertex - 1] + edge_mean[vertex])
        vertex_covariance[vertex] = _spd_geometric_midpoint(
            edge_covariance_regularized[vertex - 1],
            edge_covariance_regularized[vertex],
            floor_squared,
        )

    square_root = np.empty_like(vertex_covariance)
    inverse_square_root = np.empty_like(vertex_covariance)
    for vertex in range(samples):
        square_root[vertex], inverse_square_root[vertex] = (
            _spd_square_root_pair(
                vertex_covariance[vertex], floor_squared)
        )
    mean_acceleration = np.zeros((samples, 3), dtype=np.float64)
    covariance_acceleration = np.zeros(
        (samples, 3, 3), dtype=np.float64)
    for vertex in range(1, samples - 1):
        mean_acceleration[vertex] = (
            vertex_mean[vertex - 1]
            - 2.0 * vertex_mean[vertex]
            + vertex_mean[vertex + 1])
        local_inverse = inverse_square_root[vertex]
        previous_relative = (
            local_inverse @ vertex_covariance[vertex - 1]
            @ local_inverse)
        next_relative = (
            local_inverse @ vertex_covariance[vertex + 1]
            @ local_inverse)
        whitened_acceleration = (
            _symmetric_matrix_logarithm(
                previous_relative, floor_squared)
            + _symmetric_matrix_logarithm(
                next_relative, floor_squared)
        )
        covariance_acceleration[vertex] = (
            square_root[vertex] @ whitened_acceleration
            @ square_root[vertex])

    branch_survival = np.zeros(
        (samples, branches), dtype=np.float64)
    valid_fraction = []
    mean_population_action = []
    for target in range(1, samples - 1):
        target_inverse = inverse_square_root[target]

        def transported_coordinate(source: int) -> np.ndarray:
            relative = (
                target_inverse @ vertex_covariance[source]
                @ target_inverse)
            relative_inverse_square_root = _spd_square_root_pair(
                relative, floor_squared)[1]
            transport = (
                square_root[target] @ relative_inverse_square_root
                @ target_inverse)
            transported_mean = (
                target_inverse @ transport @ mean_acceleration[source])
            transported_covariance = (
                target_inverse @ transport
                @ covariance_acceleration[source]
                @ transport.T @ target_inverse)
            return np.concatenate((
                transported_mean,
                _symmetric_tangent_coordinates(transported_covariance),
            ))

        source_cache = {
            int(source): transported_coordinate(int(source))
            for source in np.unique(source_identity[target])
        }
        context = np.empty((branches, 9), dtype=np.float64)
        for branch in range(branches):
            first, second = source_identity[target, branch]
            context[branch] = 0.5 * (
                source_cache[int(first)] + source_cache[int(second)])
        local_coordinate = np.concatenate((
            target_inverse @ mean_acceleration[target],
            _symmetric_tangent_coordinates(
                target_inverse @ covariance_acceleration[target]
                @ target_inverse),
        ))
        valid = (
            np.all(source_identity[target] != target, axis=1)
            & (source_identity[target, :, 0]
               != source_identity[target, :, 1])
        )
        weight = branch_mass[target] * valid
        if np.sum(weight) <= np.finfo(float).tiny:
            continue
        weight /= np.sum(weight)
        context_mean = weight @ context
        centered = context - context_mean
        population_covariance = (
            (centered * weight[:, None]).T @ centered)
        # These coordinates are already whitened by the local Gaussian
        # connection.  The identity is therefore its intrinsic Fisher metric,
        # not a ridge parameter.  Transport-population covariance adds
        # epistemic uncertainty about the tangent itself; it must broaden the
        # law rather than turn missing population directions into infinite
        # precision.
        tangent_uncertainty = np.eye(9) + population_covariance
        precision = np.linalg.inv(tangent_uncertainty)
        defect = context - local_coordinate[None, :]
        tangent_action = np.einsum(
            "...a,ab,...b->...", defect, precision, defect)
        # Equal-covariance Gaussian affinity on the whitened tangent bundle.
        # The factor one eighth is fixed by the Bhattacharyya/Hellinger
        # identity, rather than exposed as a phase threshold.
        branch_survival[target] = 1.0 - np.exp(-0.125 * tangent_action)
        branch_survival[target, ~valid] = 0.0
        valid_fraction.append(float(np.mean(valid)))
        mean_population_action.append(float(weight @ tangent_action))
    return branch_survival, {
        "state": (
            "nine-dimensional mean/SPD connection-acceleration tangent "
            "transported through exact ancestry"
        ),
        "tangent_uncertainty": (
            "intrinsic whitened Gaussian Fisher metric plus transported "
            "ancestry-population covariance"
        ),
        "mean_valid_ancestry_fraction": float(np.mean(valid_fraction)),
        "mean_tangent_population_action": float(np.mean(
            mean_population_action)),
        "physical_parameters": "none",
    }


def _transported_connection_spherical_phase_1d(
    observation: np.ndarray,
    edge_mean: np.ndarray,
    edge_covariance: np.ndarray,
    source_identity: np.ndarray,
    branch_mass: np.ndarray,
    *,
    weight_by_connection_confidence: bool = True,
    population_resultant_only: bool = False,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Transport connection phase along every branch's exact ancestry.

    Gaussian means are whitened in the target covariance frame, so direction
    lies on a coordinate-invariant sphere.  The two exact source directions
    define a great-circle phase path; spherical interpolation or extrapolation
    evaluates that path at the target's actual base coordinate.  The complete
    ancestry population, rather than a selected scale, supplies phase
    concentration.
    """
    line = _validate_line(observation)
    samples, branches = branch_mass.shape
    if edge_mean.shape != (samples - 1, 3):
        raise ValueError("connection mean must align with edges")
    if edge_covariance.shape != (samples - 1, 3, 3):
        raise ValueError("connection covariance must align with edges")
    if source_identity.shape != (samples, branches, 2):
        raise ValueError("source identity must align with branch mass")
    floor_squared = _representation_floor(line) ** 2
    floor = math.sqrt(floor_squared)

    regularized = np.empty_like(edge_covariance)
    for edge in range(samples - 1):
        eigenvalue, eigenvector = np.linalg.eigh(edge_covariance[edge])
        regularized[edge] = (
            eigenvector * np.maximum(eigenvalue, floor_squared)[None, :]
        ) @ eigenvector.T
    vertex_mean = np.empty((samples, 3), dtype=np.float64)
    vertex_covariance = np.empty((samples, 3, 3), dtype=np.float64)
    vertex_mean[0] = edge_mean[0]
    vertex_mean[-1] = edge_mean[-1]
    vertex_covariance[0] = regularized[0]
    vertex_covariance[-1] = regularized[-1]
    for vertex in range(1, samples - 1):
        vertex_mean[vertex] = 0.5 * (
            edge_mean[vertex - 1] + edge_mean[vertex])
        vertex_covariance[vertex] = _spd_geometric_midpoint(
            regularized[vertex - 1], regularized[vertex], floor_squared)
    square_root = np.empty_like(vertex_covariance)
    inverse_square_root = np.empty_like(vertex_covariance)
    for vertex in range(samples):
        square_root[vertex], inverse_square_root[vertex] = (
            _spd_square_root_pair(vertex_covariance[vertex], floor_squared)
        )

    phase = np.zeros((samples, branches), dtype=np.float64)
    resultant_records: list[float] = []
    valid_records: list[float] = []
    for target in range(1, samples - 1):
        target_inverse = inverse_square_root[target]

        def target_coordinate(source: int) -> np.ndarray:
            relative = (
                target_inverse @ vertex_covariance[source] @ target_inverse)
            relative_inverse = _spd_square_root_pair(
                relative, floor_squared)[1]
            transport = (
                square_root[target] @ relative_inverse @ target_inverse)
            return target_inverse @ transport @ vertex_mean[source]

        source_cache = {
            int(source): target_coordinate(int(source))
            for source in np.unique(source_identity[target])
        }
        local = target_inverse @ vertex_mean[target]
        local_radius = float(np.linalg.norm(local))
        if local_radius <= floor:
            continue
        local_direction = local / local_radius
        local_confidence = local_radius * local_radius / (
            1.0 + local_radius * local_radius)
        prediction = np.zeros((branches, 3), dtype=np.float64)
        confidence = np.zeros(branches, dtype=np.float64)
        valid = np.zeros(branches, dtype=bool)
        for branch in range(branches):
            first, second = source_identity[target, branch]
            if first == target or second == target or first == second:
                continue
            first_vector = source_cache[int(first)]
            second_vector = source_cache[int(second)]
            first_radius = float(np.linalg.norm(first_vector))
            second_radius = float(np.linalg.norm(second_vector))
            if first_radius <= floor or second_radius <= floor:
                continue
            first_direction = first_vector / first_radius
            second_direction = second_vector / second_radius
            cosine = float(np.clip(
                first_direction @ second_direction, -1.0, 1.0))
            angle = math.acos(cosine)
            sine = math.sin(angle)
            coordinate = float(target - first) / float(second - first)
            if abs(sine) <= floor:
                spherical = (
                    (1.0 - coordinate) * first_direction
                    + coordinate * second_direction)
            else:
                spherical = (
                    math.sin((1.0 - coordinate) * angle) / sine
                    * first_direction
                    + math.sin(coordinate * angle) / sine
                    * second_direction)
            radius = float(np.linalg.norm(spherical))
            if radius <= floor:
                continue
            prediction[branch] = spherical / radius
            first_confidence = first_radius * first_radius / (
                1.0 + first_radius * first_radius)
            second_confidence = second_radius * second_radius / (
                1.0 + second_radius * second_radius)
            confidence[branch] = (
                local_confidence * first_confidence * second_confidence
            ) ** (1.0 / 3.0)
            valid[branch] = True
        weight = branch_mass[target] * valid
        if float(np.sum(weight)) <= np.finfo(float).tiny:
            continue
        weight /= np.sum(weight)
        resultant = weight @ prediction
        resultant_length = float(np.linalg.norm(resultant))
        fidelity = 0.5 * (
            1.0 + prediction @ local_direction)
        if population_resultant_only:
            phase[target, valid] = resultant_length
        else:
            phase[target] = np.clip(
                resultant_length
                * (confidence if weight_by_connection_confidence else 1.0)
                * fidelity,
                0.0,
                1.0,
            )
        phase[target, ~valid] = 0.0
        resultant_records.append(resultant_length)
        valid_records.append(float(np.mean(valid)))
    return phase, {
        "state": (
            "target-whitened spherical connection phase transported along "
            "each exact ancestry geodesic"
        ),
        "population_phase": (
            "complete posterior ancestry resultant; no selected scale"
        ),
        "connection_confidence_weighted": bool(
            weight_by_connection_confidence),
        "population_resultant_only": bool(population_resultant_only),
        "mean_phase_resultant": float(
            np.mean(resultant_records) if resultant_records else 0.0),
        "mean_valid_ancestry_fraction": float(
            np.mean(valid_records) if valid_records else 0.0),
        "physical_parameters": "none",
    }


def connection_ownership_lineage_transport_1d(
    observation: np.ndarray,
    *,
    ownership_measure: str = "root_context",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evolve signal branches and ownership of the connection jointly.

    One connection atom is the fused zero-defect information law; the other
    carries the fully inferred drift.  Their mode transition is extracted from
    continuity of the connection state itself.  Sum-product and complete-path
    collision are performed on ``(connection mode, signal branch)`` before
    scalar readout.
    """
    line = _validate_line(observation)
    if ownership_measure not in {
        "root_context",
        "connection_hotelling",
        "connection_hellinger",
        "transported_hellinger_contrast",
        "transported_covariance_contrast",
        "transported_gaussian_law_contrast",
    }:
        raise ValueError(f"unknown ownership measure {ownership_measure!r}")
    samples = line.size
    if float(np.ptp(line)) == 0.0:
        mass = np.zeros((samples, 2, 1), dtype=np.float64)
        mass[:, 0, 0] = 1.0
        prediction = line[:, None]
        return {
            "mass": mass,
            "path_collision_mass": mass.copy(),
            "branch_mass": np.ones((samples, 1), dtype=np.float64),
            "baseline_mass": np.ones((samples, 1), dtype=np.float64),
            "mode_mass": mass[:, :, 0].copy(),
            "mode_reference": mass[:, :, 0].copy(),
            "prediction": prediction,
            "reference_mass": np.ones((samples, 1), dtype=np.float64),
            "mode_transition": np.full(
                (samples - 1, 2, 2), 0.5, dtype=np.float64),
            "connection_state": np.zeros(
                (samples - 1, 2, 3), dtype=np.float64),
        }, {
            "state": "joint signal-branch and connection-ownership path law",
            "mean_drift_ownership": 0.0,
            "mean_mode_population": 1.0,
            "mean_mode_transition_population": 1.0,
            "log_path_evidence": 0.0,
            "physical_parameters": "none",
        }

    zero_law, zero_diagnostic = _lineage_branch_transport_1d(
        line, bundle_metric="joint_information")
    drift_law, drift_diagnostic = (
        bidirectional_collision_connection_lineage_transport_1d(line))
    prediction = zero_law["prediction"]
    if not np.array_equal(prediction, drift_law["prediction"]):
        raise RuntimeError("connection modes do not share a particle fibre")
    likelihood = zero_law["likelihood"]
    if not np.allclose(
        likelihood, drift_law["likelihood"], atol=0.0, rtol=0.0
    ):
        raise RuntimeError("connection modes do not share local evidence")
    reference_mass = zero_law["reference_mass"]
    reference = (
        np.broadcast_to(reference_mass, prediction.shape)
        if reference_mass.ndim == 1 else reference_mass)
    zero_kernel = zero_law["forward_kernels"]
    drift_kernel = drift_law["forward_kernels"].copy()
    edges = samples - 1
    branches = prediction.shape[1]
    if zero_kernel.shape != (edges, branches, branches):
        raise RuntimeError("invalid zero-connection transition field")
    if drift_kernel.shape != zero_kernel.shape:
        raise RuntimeError("connection transition fields do not align")

    authority = drift_law["connection_authority"]
    mean_defect = drift_law["connection_mean_defect"]
    covariance = drift_law["connection_covariance"]
    if (
        authority.shape != (edges,)
        or mean_defect.shape != (edges, 3)
        or covariance.shape != (edges, 3, 3)
    ):
        raise RuntimeError("connection posterior does not align with edges")
    if ownership_measure == "root_context":
        root_law, root_diagnostic = nested_midpoint_lineage_transport_1d(line)
        root_authority, _root_action, _population_action = (
            _energy_root_authority(
                line, root_law["prediction"], root_law["mass"])
        )
        source_identity = zero_law["source_identity"]
        if source_identity.shape != (samples, branches, 2):
            raise RuntimeError(
                "connection ownership requires two-source ancestry")
        transported_source_authority = 0.5 * (
            root_authority[source_identity[:, :, 0]]
            + root_authority[source_identity[:, :, 1]])
        context_authority = np.sum(
            zero_law["mass"] * transported_source_authority, axis=1)
        authority_competition = (
            root_authority * root_authority
            + context_authority * context_authority)
        ownership_survival = np.divide(
            (1.0 - root_authority)
            * (context_authority - root_authority) ** 2,
            authority_competition,
            out=np.zeros_like(root_authority),
            where=authority_competition > np.finfo(float).tiny,
        )
        ownership_action = ownership_survival / np.maximum(
            1.0 - ownership_survival, np.finfo(float).tiny)
        ownership_evidence = {
            "state": "energy-distance authority under target-free transport",
            "identity": "E|Z-Z'| / (2 E|y-Z|)",
            "context": (
                "branch-posterior barycenter of authority transported from "
                "the two exact particle sources"
            ),
            "survival": "(1-a) (c-a)^2 / (a^2 + c^2)",
            "target_value_enters_context_action": bool(
                root_diagnostic["target_value_enters_local_action"]),
        }
    elif ownership_measure == "transported_gaussian_law_contrast":
        source_identity = zero_law["source_identity"]
        ownership_survival, ownership_action, _branch_survival = (
            _transported_gaussian_connection_contrast_1d(
                line,
                authority[:, None] ** 4 * mean_defect,
                covariance,
                source_identity,
                zero_law["mass"],
            )
        )
        root_authority = np.zeros(samples, dtype=np.float64)
        context_authority = 1.0 - ownership_survival
        ownership_evidence = {
            "state": (
                "complete Gaussian connection laws transported from exact "
                "branch ancestry before scalarization"
            ),
            "identity": (
                "branch-expected Hellinger distance after affine-invariant "
                "Gaussian covariance transport"
            ),
            "connection_mean": "rho^4 times inferred edge defect",
            "connection_covariance": (
                "affine-invariant SPD geodesic midpoint"
            ),
            "target_value_enters_context_action": True,
        }
    elif ownership_measure in {
        "connection_hotelling",
        "connection_hellinger",
        "transported_hellinger_contrast",
        "transported_covariance_contrast",
    }:
        connection_mean_for_uncertainty = (
            np.zeros_like(mean_defect)
            if ownership_measure == "transported_covariance_contrast"
            else authority[:, None] ** 4 * mean_defect
        )
        local_connection_uncertainty, local_connection_action = (
            _connection_defect_ownership_1d(
                line,
                connection_mean_for_uncertainty,
                covariance,
                include_covariance=(
                    ownership_measure != "connection_hotelling"),
            )
        )
        if ownership_measure in {
            "transported_hellinger_contrast",
            "transported_covariance_contrast",
        }:
            source_identity = zero_law["source_identity"]
            if source_identity.shape != (samples, branches, 2):
                raise RuntimeError(
                    "connection phase contrast requires exact ancestry")
            transported_source_uncertainty = 0.5 * (
                local_connection_uncertainty[source_identity[:, :, 0]]
                + local_connection_uncertainty[source_identity[:, :, 1]])
            context_authority = np.sum(
                zero_law["mass"] * transported_source_uncertainty, axis=1)
            probability_affinity = (
                np.sqrt(local_connection_uncertainty * context_authority)
                + np.sqrt(
                    (1.0 - local_connection_uncertainty)
                    * (1.0 - context_authority))
            )
            ownership_survival = np.clip(
                1.0 - probability_affinity, 0.0, 1.0)
            ownership_action = -np.log(np.maximum(
                probability_affinity, np.finfo(float).tiny))
        else:
            ownership_survival = local_connection_uncertainty
            ownership_action = local_connection_action
            context_authority = np.zeros(samples, dtype=np.float64)
        root_authority = np.zeros(samples, dtype=np.float64)
        ownership_evidence = {
            "state": (
                "transported phase contrast of adjacent Gaussian "
                "connection-posterior covariance"
                if ownership_measure == "transported_covariance_contrast"
                else "transported phase contrast of adjacent Gaussian "
                "connection-posterior defect"
                if ownership_measure == "transported_hellinger_contrast"
                else (
                    "adjacent Gaussian connection-posterior defect "
                    "transported to vertices"
                )
            ),
            "identity": (
                "Bernoulli Hellinger distance from transported source context"
                if ownership_measure in {
                    "transported_hellinger_contrast",
                    "transported_covariance_contrast",
                }
                else (
                    "squared Hellinger distance"
                    if ownership_measure == "connection_hellinger"
                    else "Hotelling action divided by one plus action"
                )
            ),
            "connection_mean": (
                "held at zero while uncertainty ownership is inferred"
                if ownership_measure == "transported_covariance_contrast"
                else "rho^4 times inferred edge defect"
            ),
            "connection_covariance": (
                "jointly participates"
                if ownership_measure != "connection_hotelling"
                else "supplies the pooled inverse metric"
            ),
            "target_value_enters_context_action": True,
        }
    else:  # pragma: no cover - validated before the constant exact branch
        raise AssertionError("unreachable ownership measure")
    two_history_bypass_survival = ownership_survival ** 2
    mode_reference = np.column_stack((
        1.0 - ownership_survival,
        ownership_survival,
    ))
    mode_likelihood = np.broadcast_to(
        likelihood[:, None, :], (samples, 2, branches))

    connection_state = np.zeros((edges, 2, 3), dtype=np.float64)
    connection_state[:, 1] = authority[:, None] ** 4 * mean_defect
    mode_transition = np.empty((edges, 2, 2), dtype=np.float64)
    for edge in range(edges):
        source_mode = mode_reference[edge]
        target_mode = mode_reference[edge + 1]
        coupling = np.zeros((2, 2), dtype=np.float64)
        coupling[0, 0] = min(source_mode[0], target_mode[0])
        coupling[1, 1] = min(source_mode[1], target_mode[1])
        if target_mode[1] > source_mode[1]:
            coupling[0, 1] = target_mode[1] - source_mode[1]
        else:
            coupling[1, 0] = source_mode[1] - target_mode[1]
        for mode in range(2):
            if source_mode[mode] > np.finfo(float).tiny:
                mode_transition[edge, mode] = (
                    coupling[mode] / source_mode[mode])
            else:
                mode_transition[edge, mode] = target_mode

    forward = np.empty((samples, 2, branches), dtype=np.float64)
    backward = np.empty_like(forward)
    forward[0] = (
        mode_reference[0, :, None]
        * reference[0][None, :]
        * mode_likelihood[0]
    )
    evidence = float(np.sum(forward[0]))
    log_path_evidence = math.log(max(evidence, np.finfo(float).tiny))
    forward[0] /= evidence
    kernels = (zero_kernel, drift_kernel)
    for index in range(1, samples):
        edge = index - 1
        for next_mode in range(2):
            predicted = np.zeros(branches, dtype=np.float64)
            for previous_mode in range(2):
                predicted += mode_transition[edge, previous_mode, next_mode] * (
                    forward[index - 1, previous_mode]
                    @ kernels[next_mode][edge]
                )
            forward[index, next_mode] = (
                predicted * mode_likelihood[index, next_mode])
        evidence = float(np.sum(forward[index]))
        log_path_evidence += math.log(max(
            evidence, np.finfo(float).tiny))
        forward[index] /= evidence

    backward[-1] = 1.0
    for index in range(samples - 2, -1, -1):
        edge = index
        for previous_mode in range(2):
            message = np.zeros(branches, dtype=np.float64)
            for next_mode in range(2):
                message += mode_transition[
                    edge, previous_mode, next_mode] * (
                    kernels[next_mode][edge] @ (
                        mode_likelihood[index + 1, next_mode]
                        * backward[index + 1, next_mode]
                    )
                )
            backward[index, previous_mode] = message
        maximum = float(np.max(backward[index]))
        if maximum > 0.0:
            backward[index] /= maximum

    mass = forward * backward
    mass /= np.sum(mass, axis=(1, 2), keepdims=True)

    # Collide the complete joint histories, including connection ownership,
    # before either mode or signal branch is marginalized.
    path_forward = np.empty_like(forward)
    path_backward = np.empty_like(backward)
    joint_reference = mode_reference[:, :, None] * reference[:, None, :]
    path_forward[0] = joint_reference[0] * mode_likelihood[0] ** 2
    path_forward[0] /= np.sum(path_forward[0])
    for index in range(1, samples):
        edge = index - 1
        for next_mode in range(2):
            child_reference = np.maximum(
                joint_reference[index, next_mode], np.finfo(float).tiny)
            predicted = np.zeros(branches, dtype=np.float64)
            for previous_mode in range(2):
                joint_kernel = (
                    mode_transition[edge, previous_mode, next_mode]
                    * kernels[next_mode][edge]
                )
                collision_kernel = joint_kernel * joint_kernel / child_reference
                predicted += (
                    path_forward[index - 1, previous_mode]
                    @ collision_kernel
                )
            path_forward[index, next_mode] = (
                predicted * mode_likelihood[index, next_mode] ** 2)
        path_forward[index] /= np.sum(path_forward[index])
    path_backward[-1] = 1.0
    for index in range(samples - 2, -1, -1):
        edge = index
        for previous_mode in range(2):
            message = np.zeros(branches, dtype=np.float64)
            for next_mode in range(2):
                child_reference = np.maximum(
                    joint_reference[index + 1, next_mode],
                    np.finfo(float).tiny,
                )
                joint_kernel = (
                    mode_transition[edge, previous_mode, next_mode]
                    * kernels[next_mode][edge]
                )
                collision_kernel = joint_kernel * joint_kernel / child_reference
                message += collision_kernel @ (
                    mode_likelihood[index + 1, next_mode] ** 2
                    * path_backward[index + 1, next_mode]
                )
            path_backward[index, previous_mode] = message
        maximum = float(np.max(path_backward[index]))
        if maximum > 0.0:
            path_backward[index] /= maximum
    path_collision_mass = path_forward * path_backward
    path_collision_mass /= np.sum(
        path_collision_mass, axis=(1, 2), keepdims=True)

    mode_mass = np.sum(mass, axis=2)
    mode_transition_population = 1.0 / np.sum(
        mode_transition * mode_transition, axis=2)
    return {
        "mass": mass,
        "path_collision_mass": path_collision_mass,
        "branch_mass": np.sum(mass, axis=1),
        "baseline_mass": zero_law["mass"],
        "mode_mass": mode_mass,
        "mode_reference": mode_reference,
        "prediction": prediction,
        "reference_mass": reference,
        "mode_transition": mode_transition,
        "connection_state": connection_state,
        "ownership_emission": mode_reference,
    }, {
        "state": "joint signal-branch and connection-ownership path law",
        "connection_modes": (
            "fused zero-defect information connection",
            "bidirectional two-history certified connection drift",
        ),
        "mode_transition": (
            "minimum-switching optimal transport of adjacent Bernoulli "
            "ownership reference measures"
        ),
        "collision": (
            "order-two complete joint path collision before mode or branch "
            "marginalization"
        ),
        "mean_drift_ownership": float(np.mean(mode_mass[:, 1])),
        "minimum_drift_ownership": float(np.min(mode_mass[:, 1])),
        "maximum_drift_ownership": float(np.max(mode_mass[:, 1])),
        "mean_mode_population": float(np.mean(
            1.0 / np.sum(mode_mass * mode_mass, axis=1))),
        "mean_mode_transition_population": float(np.mean(
            mode_transition_population)),
        "mean_root_authority": float(np.mean(root_authority)),
        "mean_context_authority": float(np.mean(context_authority)),
        "mean_sparse_bypass_survival": float(np.mean(
            ownership_survival)),
        "mean_two_history_bypass_survival": float(np.mean(
            two_history_bypass_survival)),
        "mean_ownership_action": float(np.mean(ownership_action)),
        "ownership_measure": ownership_measure,
        "log_path_evidence": float(log_path_evidence),
        "zero_connection": {
            "state": zero_diagnostic["state"],
            "bundle_metric": zero_diagnostic["bundle_metric"],
        },
        "drift_connection": drift_diagnostic["transport_distribution"],
        "root_ownership_evidence": {
            **ownership_evidence,
            "two_history_survival": "square of sparse bypass survival",
            "mode_reference": "(one minus sparse survival, sparse survival)",
            "order_two": "supplied by joint path collision, not the reference",
        },
        "physical_parameters": "none",
    }


def connection_ownership_readout_forms(
    observation: np.ndarray,
    *,
    ownership_measure: str = "root_context",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project the joint connection-ownership path only at its endpoint."""
    law, diagnostic = connection_ownership_lineage_transport_1d(
        observation, ownership_measure=ownership_measure)
    prediction = law["prediction"]
    mass = law["mass"]
    path_mass = law["path_collision_mass"]
    reference = law["reference_mass"]
    joint_reference = law["mode_reference"][:, :, None] * reference[:, None, :]
    collision_mass = mass * mass / np.maximum(
        joint_reference, np.finfo(float).tiny)
    collision_mass /= np.sum(
        collision_mass, axis=(1, 2), keepdims=True)
    baseline_mass = law["baseline_mass"]
    baseline_collision_mass = baseline_mass * baseline_mass / np.maximum(
        reference, np.finfo(float).tiny)
    baseline_collision_mass /= np.sum(
        baseline_collision_mass, axis=1, keepdims=True)
    return {
        "baseline_collision_mean": np.sum(
            baseline_collision_mass * prediction, axis=1),
        "mean": np.sum(mass * prediction[:, None, :], axis=(1, 2)),
        "collision_mean": np.sum(
            collision_mass * prediction[:, None, :], axis=(1, 2)),
        "path_collision_mean": np.sum(
            path_mass * prediction[:, None, :], axis=(1, 2)),
    }, diagnostic


def _connection_action_quadratic_1d(
    observation: np.ndarray,
    zero_law: dict[str, np.ndarray],
    connection_law: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear and quadratic action terms on the drift geodesic."""
    line = _validate_line(observation)
    prediction = zero_law["prediction"]
    jet = zero_law["jet"]
    residual = line[:, None] - prediction
    samples, branches = prediction.shape
    mean_defect = connection_law["connection_mean_defect"]
    covariance = connection_law["connection_covariance"]
    if mean_defect.shape != (samples - 1, 3):
        raise ValueError("connection mean must align with edges")
    if covariance.shape != (samples - 1, 3, 3):
        raise ValueError("connection covariance must align with edges")
    linear = np.zeros(samples - 1, dtype=np.float64)
    quadratic = np.zeros(samples - 1, dtype=np.float64)
    floor_squared = _representation_floor(line) ** 2
    for edge in range(samples - 1):
        kernel = zero_law["forward_kernels"][edge]
        plan = (
            zero_law["forward_mass"][edge, :, None]
            * kernel
            * (
                zero_law["likelihood"][edge + 1]
                * zero_law["backward_mass"][edge + 1]
            )[None, :]
        )
        plan /= np.sum(plan)
        left_value = prediction[edge] + 0.5 * jet[edge]
        right_value = prediction[edge + 1] - 0.5 * jet[edge + 1]
        value_defect = left_value[:, None] - right_value[None, :]
        jet_defect = jet[edge, :, None] - jet[edge + 1, None, :]
        residual_defect = (
            residual[edge, :, None] - residual[edge + 1, None, :])
        defect = np.stack(np.broadcast_arrays(
            value_defect, jet_defect, residual_defect), axis=-1)
        plan_mean = np.einsum("ij,ija->a", plan, defect)
        eigenvalue, eigenvector = np.linalg.eigh(covariance[edge])
        precision = (
            eigenvector
            * (1.0 / np.maximum(eigenvalue, floor_squared))[None, :]
        ) @ eigenvector.T
        direction = mean_defect[edge]
        linear[edge] = float(direction @ precision @ plan_mean)
        quadratic[edge] = float(direction @ precision @ direction)
    return linear, quadratic


def _connection_newton_coefficients_1d(
    observation: np.ndarray,
    zero_law: dict[str, np.ndarray],
    connection_law: dict[str, np.ndarray],
) -> np.ndarray:
    """Minimize connection action along the inferred drift direction."""
    linear, quadratic = _connection_action_quadratic_1d(
        observation, zero_law, connection_law)
    coefficient = np.divide(
        linear,
        quadratic,
        out=np.zeros_like(linear),
        where=quadratic > np.finfo(float).tiny,
    )
    return np.clip(coefficient, 0.0, 1.0)


def _connection_action_posterior_1d(
    observation: np.ndarray,
    zero_law: dict[str, np.ndarray],
    connection_law: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginalize the complete quadratic action on zero--drift transport.

    The scalar connection coordinate lies on the geodesic segment from the
    fused zero connection to the fully inferred drift.  A uniform volume law
    on that segment and the exact quadratic action induce a truncated-normal
    posterior.  Its first two moments retain uncertainty about transport
    instead of replacing the law by its Newton minimizer.
    """
    linear, quadratic = _connection_action_quadratic_1d(
        observation, zero_law, connection_law)
    mean = np.full_like(linear, 0.5)
    variance = np.full_like(linear, 1.0 / 12.0)
    mode = np.zeros_like(linear)
    informative = quadratic > math.sqrt(np.finfo(float).eps)
    mode[informative] = linear[informative] / quadratic[informative]
    if np.any(informative):
        precision = quadratic[informative]
        sigma = 1.0 / np.sqrt(precision)
        center = mode[informative]
        lower = -center / sigma
        upper = (1.0 - center) / sigma
        normalization = special.ndtr(upper) - special.ndtr(lower)
        stable = normalization > math.sqrt(np.finfo(float).eps)
        if np.any(stable):
            density_lower = np.exp(-0.5 * lower[stable] ** 2) / math.sqrt(
                2.0 * math.pi)
            density_upper = np.exp(-0.5 * upper[stable] ** 2) / math.sqrt(
                2.0 * math.pi)
            ratio = (
                density_lower - density_upper
            ) / normalization[stable]
            posterior_mean = center[stable] + sigma[stable] * ratio
            posterior_variance = sigma[stable] ** 2 * (
                1.0
                + (
                    lower[stable] * density_lower
                    - upper[stable] * density_upper
                ) / normalization[stable]
                - ratio ** 2
            )
            location = np.flatnonzero(informative)[stable]
            mean[location] = np.clip(posterior_mean, 0.0, 1.0)
            variance[location] = np.maximum(posterior_variance, 0.0)
        # Vanishing interval probability is a numerical boundary case.  The
        # action then concentrates at the nearest geodesic endpoint.
        if np.any(~stable):
            location = np.flatnonzero(informative)[~stable]
            mean[location] = np.clip(mode[location], 0.0, 1.0)
            variance[location] = 0.0
    mode = np.clip(mode, 0.0, 1.0)
    return mean, variance, mode


def _observation_cavity_hellinger_surprise_1d(
    lineage_law: dict[str, np.ndarray],
) -> np.ndarray:
    """Return local observation-induced displacement of the path posterior.

    A smoothed branch posterior contains the local likelihood exactly once.
    Dividing it out and renormalizing therefore gives the target-free cavity
    law without rerunning transport.  Hellinger defect measures how strongly
    the current observation moves that predictive ancestry distribution.
    """
    mass = np.asarray(lineage_law["mass"], dtype=np.float64)
    likelihood = np.asarray(lineage_law["likelihood"], dtype=np.float64)
    if mass.shape != likelihood.shape or mass.ndim != 2:
        raise ValueError("lineage mass and likelihood must align")
    cavity = mass / np.maximum(likelihood, np.finfo(float).tiny)
    cavity /= np.sum(cavity, axis=1, keepdims=True)
    affinity = np.sum(np.sqrt(mass * cavity), axis=1)
    return np.clip(1.0 - affinity, 0.0, 1.0)


def _transported_source_support_fidelity_1d(
    source_identity: np.ndarray,
    posterior_mass: np.ndarray,
    reference_mass: np.ndarray,
) -> np.ndarray:
    """Compare transported source incidence with topological opportunity.

    Every posterior ancestry branch contributes half its mass to each of its
    two exact sources.  Repeating the same pushforward with the branch
    reference measure gives the opportunity degree induced by the complete
    proposal topology.  Their symmetric scale commensurability is one only
    when posterior support and opportunity agree; no support is selected.
    """
    source = np.asarray(source_identity, dtype=np.int64)
    posterior = np.asarray(posterior_mass, dtype=np.float64)
    reference = np.asarray(reference_mass, dtype=np.float64)
    if source.shape != posterior.shape + (2,):
        raise ValueError("source identity must align with posterior mass")
    if reference.ndim == 1:
        reference = np.broadcast_to(reference, posterior.shape)
    if reference.shape != posterior.shape:
        raise ValueError("reference mass must align with posterior mass")
    samples = posterior.shape[0]
    support_degree = np.zeros(samples, dtype=np.float64)
    opportunity_degree = np.zeros(samples, dtype=np.float64)
    for endpoint in range(2):
        np.add.at(
            support_degree,
            source[:, :, endpoint].ravel(),
            (0.5 * posterior).ravel(),
        )
        np.add.at(
            opportunity_degree,
            source[:, :, endpoint].ravel(),
            (0.5 * reference).ravel(),
        )
    numerator = 2.0 * support_degree * opportunity_degree
    denominator = support_degree ** 2 + opportunity_degree ** 2
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > np.finfo(float).tiny,
    )


def action_contracting_connection_transport_1d(
    observation: np.ndarray,
    *,
    parallel_transport_connection: bool = False,
    require_population_phase_collision: bool = False,
    fuse_population_phase_odds: bool = False,
    fuse_connection_acceleration_odds: bool = False,
    fuse_connection_jerk_odds: bool = False,
    fuse_connection_tangent_odds: bool = False,
    fuse_connection_spherical_phase_odds: bool = False,
    fuse_connection_spherical_phase_union: bool = False,
    suppress_connection_on_spherical_phase: bool = False,
    fuse_phase_defect_spherical_odds: bool = False,
    newton_optimize_connection: bool = False,
    marginalize_connection_action: bool = False,
    marginalize_gaussian_connection: bool = False,
    phase_coherent_connection_posterior: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Contract connection action continuously after ancestry transport.

    Each branch retains the Hellinger defect between its local Gaussian
    connection law and the complete law transported from its exact sources.
    At an edge, the endpoint defects define the midpoint mass that mixes zero
    and inferred-connection conductances.  The corresponding effective action
    is their weighted harmonic mean, so contraction follows analytically from
    the arithmetic--harmonic inequality rather than from an iteration count.
    """
    line = _validate_line(observation)
    zero_law, zero_diagnostic = _lineage_branch_transport_1d(
        line, bundle_metric="joint_information")
    prediction = zero_law["prediction"]
    samples, branches = prediction.shape
    reference_mass = zero_law["reference_mass"]
    reference = (
        np.broadcast_to(reference_mass, prediction.shape)
        if reference_mass.ndim == 1 else reference_mass)
    if float(np.ptp(line)) == 0.0:
        mass = zero_law["mass"].copy()
        return {
            "mass": mass,
            "path_collision_mass": mass.copy(),
            "baseline_mass": mass.copy(),
            "prediction": prediction,
            "reference_mass": reference,
            "branch_connection_contrast": np.zeros_like(mass),
            "effective_kernels": np.empty(
                (samples - 1, branches, branches), dtype=np.float64),
        }, {
            "state": "continuous action-contracting connection transport",
            "mean_branch_connection_contrast": 0.0,
            "maximum_harmonic_action_violation": 0.0,
            "mean_lineage_population": float(
                1.0 / np.sum(mass[0] * mass[0])),
            "physical_parameters": "none",
        }

    drift_law, drift_diagnostic = (
        bidirectional_collision_connection_lineage_transport_1d(line))
    likelihood = zero_law["likelihood"]
    authority = drift_law["connection_authority"]
    mean_defect = drift_law["connection_mean_defect"]
    covariance = drift_law["connection_covariance"]
    connection_estimators = sum((
        bool(newton_optimize_connection),
        bool(marginalize_connection_action),
        bool(marginalize_gaussian_connection),
        bool(phase_coherent_connection_posterior),
    ))
    if connection_estimators > 1:
        raise ValueError(
            "connection estimator must be uniquely selected")
    newton_coefficient = np.ones(samples - 1, dtype=np.float64)
    posterior_connection_mean = np.ones(samples - 1, dtype=np.float64)
    posterior_connection_variance = np.zeros(
        samples - 1, dtype=np.float64)
    connection_mean_multiplier = authority ** 4
    effective_connection_covariance = covariance
    source_identity = zero_law["source_identity"]
    transported_connection_mean: np.ndarray | None = None
    phase_connection_order = np.zeros(samples, dtype=np.float64)
    phase_connection_diagnostic: dict[str, float | str] | None = None
    observation_cavity_surprise = np.zeros(samples, dtype=np.float64)
    transport_support_fidelity = np.ones(samples, dtype=np.float64)
    if connection_estimators:
        if phase_coherent_connection_posterior:
            newton_coefficient = _connection_newton_coefficients_1d(
                line, zero_law, drift_law)
            (
                posterior_connection_mean,
                posterior_connection_variance,
                _posterior_mode,
            ) = _connection_action_posterior_1d(
                line, zero_law, drift_law)
            newton_multiplier = authority ** 4 * newton_coefficient
            posterior_multiplier = (
                authority ** 4 * posterior_connection_mean)
            newton_drift_law, _newton_diagnostic = (
                _lineage_branch_transport_1d(
                    line,
                    bundle_metric="transport_external_connection",
                    transition_normalization="markov",
                    connection_mean_multiplier=newton_multiplier,
                )
            )
            posterior_drift_law, _posterior_diagnostic = (
                _lineage_branch_transport_1d(
                    line,
                    bundle_metric="transport_external_action_marginal",
                    transition_normalization="markov",
                    connection_mean_multiplier=authority ** 4,
                )
            )
            phase_field, phase_connection_diagnostic = (
                _transported_connection_spherical_phase_1d(
                    line,
                    authority[:, None] ** 4 * mean_defect,
                    covariance,
                    source_identity,
                    zero_law["mass"],
                    weight_by_connection_confidence=False,
                    population_resultant_only=True,
                )
            )
            base_connection_mean = authority[:, None] ** 4 * mean_defect
            _, _, base_branch_defect = (
                _transported_gaussian_connection_contrast_1d(
                    line,
                    base_connection_mean,
                    covariance,
                    source_identity,
                    zero_law["mass"],
                    parallel_transport_connection=(
                        parallel_transport_connection),
                )
            )
            target_identity = np.arange(samples)[:, None]
            base_valid = (
                np.all(
                    source_identity != target_identity[:, :, None], axis=2)
                & (source_identity[:, :, 0] != source_identity[:, :, 1])
            )
            base_branch_defect = np.where(
                base_valid, base_branch_defect, 0.0)
            base_local_uncertainty, _base_local_action = (
                _connection_defect_ownership_1d(
                    line,
                    base_connection_mean,
                    covariance,
                    include_covariance=True,
                )
            )
            base_source_uncertainty = 0.5 * (
                base_local_uncertainty[source_identity[:, :, 0]]
                + base_local_uncertainty[source_identity[:, :, 1]])
            base_context_uncertainty = np.sum(
                zero_law["mass"] * base_source_uncertainty, axis=1)
            base_phase_affinity = (
                np.sqrt(
                    base_local_uncertainty * base_context_uncertainty)
                + np.sqrt(
                    (1.0 - base_local_uncertainty)
                    * (1.0 - base_context_uncertainty))
            )
            base_phase_defect = np.clip(
                1.0 - base_phase_affinity, 0.0, 1.0)
            observation_cavity_surprise = (
                _observation_cavity_hellinger_surprise_1d(zero_law))
            transport_support_fidelity = (
                _transported_source_support_fidelity_1d(
                    source_identity,
                    zero_law["mass"],
                    reference,
                )
            )
            coherent_posterior_field = (
                phase_field ** 2
                * (1.0 - base_phase_defect[:, None])
                * (1.0 - base_branch_defect))
            phase_connection_order = np.sum(
                zero_law["mass"] * coherent_posterior_field, axis=1)
            edge_phase_order = np.divide(
                2.0
                * phase_connection_order[:-1]
                * phase_connection_order[1:],
                phase_connection_order[:-1]
                + phase_connection_order[1:],
                out=np.zeros(samples - 1, dtype=np.float64),
                where=(
                    phase_connection_order[:-1]
                    + phase_connection_order[1:]
                    > np.finfo(float).tiny),
            )
            drift_kernel = (
                (1.0 - edge_phase_order)[:, None, None]
                * newton_drift_law["forward_kernels"]
                + edge_phase_order[:, None, None]
                * posterior_drift_law["forward_kernels"]
            )
            drift_kernel /= np.sum(drift_kernel, axis=2, keepdims=True)
            newton_mean = newton_multiplier[:, None] * mean_defect
            posterior_mean = posterior_multiplier[:, None] * mean_defect
            transported_connection_mean = (
                (1.0 - edge_phase_order)[:, None] * newton_mean
                + edge_phase_order[:, None] * posterior_mean)
            posterior_covariance_addition = (
                (authority ** 8 * posterior_connection_variance)[
                    :, None, None]
                * np.einsum("ea,eb->eab", mean_defect, mean_defect)
            )
            mean_separation = newton_mean - posterior_mean
            effective_connection_covariance = (
                covariance
                + edge_phase_order[:, None, None]
                * posterior_covariance_addition
                + (
                    edge_phase_order * (1.0 - edge_phase_order)
                )[:, None, None]
                * np.einsum(
                    "ea,eb->eab", mean_separation, mean_separation)
            )
            connection_mean_multiplier = np.divide(
                np.linalg.norm(transported_connection_mean, axis=1),
                np.maximum(
                    np.linalg.norm(mean_defect, axis=1),
                    np.finfo(float).tiny,
                ),
            )
        elif marginalize_connection_action:
            (
                posterior_connection_mean,
                posterior_connection_variance,
                newton_coefficient,
            ) = _connection_action_posterior_1d(
                line, zero_law, drift_law)
            connection_mean_multiplier *= posterior_connection_mean
            covariance_addition = (
                (authority ** 8 * posterior_connection_variance)[:, None, None]
                * np.einsum("ea,eb->eab", mean_defect, mean_defect)
            )
            effective_connection_covariance = covariance + covariance_addition
        elif marginalize_gaussian_connection:
            covariance_addition = (
                authority ** 8)[:, None, None] * covariance
            effective_connection_covariance = covariance + covariance_addition
        else:
            newton_coefficient = _connection_newton_coefficients_1d(
                line, zero_law, drift_law)
            connection_mean_multiplier *= newton_coefficient
            covariance_addition = None
        if not phase_coherent_connection_posterior:
            optimized_drift_law, _optimized_drift_diagnostic = (
                _lineage_branch_transport_1d(
                    line,
                    bundle_metric=(
                        "transport_external_action_marginal"
                        if marginalize_connection_action
                        else (
                            "transport_external_gaussian_marginal"
                            if marginalize_gaussian_connection
                            else "transport_external_connection"
                        )
                    ),
                    transition_normalization="markov",
                    connection_mean_multiplier=(
                        authority ** 4
                        if (
                            marginalize_connection_action
                            or marginalize_gaussian_connection
                        )
                        else connection_mean_multiplier
                    ),
                    connection_transport_covariance=(
                        covariance_addition
                        if marginalize_gaussian_connection
                        else None
                    ),
                )
            )
            drift_kernel = optimized_drift_law["forward_kernels"]
    else:
        drift_kernel = drift_law["forward_kernels"]
    if transported_connection_mean is None:
        transported_connection_mean = (
            connection_mean_multiplier[:, None] * mean_defect)
    _, _, branch_contrast = _transported_gaussian_connection_contrast_1d(
        line,
        transported_connection_mean,
        effective_connection_covariance,
        source_identity,
        zero_law["mass"],
        parallel_transport_connection=parallel_transport_connection,
    )
    target_identity = np.arange(samples)[:, None]
    distinct_target_free_ancestry = (
        np.all(source_identity != target_identity[:, :, None], axis=2)
        & (source_identity[:, :, 0] != source_identity[:, :, 1])
    )
    branch_contrast = np.where(
        distinct_target_free_ancestry, branch_contrast, 0.0)
    population_combination_count = sum((
        bool(require_population_phase_collision),
        bool(fuse_population_phase_odds),
        bool(fuse_connection_acceleration_odds),
        bool(fuse_connection_jerk_odds),
        bool(fuse_connection_tangent_odds),
        bool(fuse_connection_spherical_phase_odds),
        bool(fuse_connection_spherical_phase_union),
        bool(suppress_connection_on_spherical_phase),
        bool(fuse_phase_defect_spherical_odds),
    ))
    if population_combination_count > 1:
        raise ValueError("population phase combination must be unique")
    tangent_diagnostic: dict[str, float | str] | None = None
    spherical_phase_diagnostic: dict[str, float | str] | None = None
    if fuse_phase_defect_spherical_odds:
        spherical_coherence, spherical_phase_diagnostic = (
            _transported_connection_spherical_phase_1d(
                line,
                transported_connection_mean,
                effective_connection_covariance,
                source_identity,
                zero_law["mass"],
                weight_by_connection_confidence=False,
                population_resultant_only=True,
            )
        )
        local_connection_uncertainty, _local_action = (
            _connection_defect_ownership_1d(
                line,
                transported_connection_mean,
                effective_connection_covariance,
                include_covariance=True,
            )
        )
        transported_source_uncertainty = 0.5 * (
            local_connection_uncertainty[source_identity[:, :, 0]]
            + local_connection_uncertainty[source_identity[:, :, 1]])
        context_uncertainty = np.sum(
            zero_law["mass"] * transported_source_uncertainty, axis=1)
        population_affinity = (
            np.sqrt(local_connection_uncertainty * context_uncertainty)
            + np.sqrt(
                (1.0 - local_connection_uncertainty)
                * (1.0 - context_uncertainty))
        )
        phase_defect = np.clip(
            1.0 - population_affinity, 0.0, 1.0)
        coherent_phase = (
            spherical_coherence
            * (1.0 - phase_defect[:, None])
            * (1.0 - branch_contrast))
        tangent_phase_survival = (
            phase_defect[:, None] * (1.0 - coherent_phase))
        population_phase_survival = np.sum(
            zero_law["mass"] * tangent_phase_survival, axis=1)
    elif (
        fuse_connection_spherical_phase_odds
        or fuse_connection_spherical_phase_union
        or suppress_connection_on_spherical_phase
    ):
        tangent_phase_survival, spherical_phase_diagnostic = (
            _transported_connection_spherical_phase_1d(
                line,
                transported_connection_mean,
                effective_connection_covariance,
                source_identity,
                zero_law["mass"],
                weight_by_connection_confidence=(
                    not suppress_connection_on_spherical_phase),
                population_resultant_only=(
                    suppress_connection_on_spherical_phase),
            )
        )
        population_phase_survival = np.sum(
            zero_law["mass"] * tangent_phase_survival, axis=1)
    elif fuse_connection_tangent_odds:
        tangent_phase_survival, tangent_diagnostic = (
            _transported_connection_tangent_acceleration_1d(
                line,
                transported_connection_mean,
                effective_connection_covariance,
                source_identity,
                zero_law["mass"],
            )
        )
        population_phase_survival = np.sum(
            zero_law["mass"] * tangent_phase_survival, axis=1)
    elif fuse_connection_acceleration_odds or fuse_connection_jerk_odds:
        connection_acceleration_survival, _acceleration_action = (
            _connection_geodesic_acceleration_1d(
                line,
                transported_connection_mean,
                effective_connection_covariance,
            )
        )
        if fuse_connection_jerk_odds:
            population_phase_survival, _jerk_action = (
                _bernoulli_geodesic_residual_1d(
                    connection_acceleration_survival)
            )
        else:
            population_phase_survival = connection_acceleration_survival
    elif require_population_phase_collision or fuse_population_phase_odds:
        local_connection_uncertainty, _local_action = (
            _connection_defect_ownership_1d(
                line,
                transported_connection_mean,
                effective_connection_covariance,
                include_covariance=True,
            )
        )
        transported_source_uncertainty = 0.5 * (
            local_connection_uncertainty[source_identity[:, :, 0]]
            + local_connection_uncertainty[source_identity[:, :, 1]])
        context_uncertainty = np.sum(
            zero_law["mass"] * transported_source_uncertainty, axis=1)
        population_affinity = (
            np.sqrt(local_connection_uncertainty * context_uncertainty)
            + np.sqrt(
                (1.0 - local_connection_uncertainty)
                * (1.0 - context_uncertainty))
        )
        population_phase_survival = np.clip(
            1.0 - population_affinity, 0.0, 1.0)
    else:
        population_phase_survival = np.ones(samples, dtype=np.float64)
    if require_population_phase_collision:
        branch_contrast *= population_phase_survival[:, None]
    elif suppress_connection_on_spherical_phase:
        branch_contrast *= 1.0 - tangent_phase_survival
    elif fuse_connection_spherical_phase_union:
        branch_contrast = 1.0 - (
            (1.0 - branch_contrast) * (1.0 - tangent_phase_survival))
    elif (
        fuse_population_phase_odds
        or fuse_connection_acceleration_odds
        or fuse_connection_jerk_odds
        or fuse_connection_tangent_odds
        or fuse_connection_spherical_phase_odds
        or fuse_phase_defect_spherical_odds
    ):
        phase_survival = (
            tangent_phase_survival
            if (
                fuse_connection_tangent_odds
                or fuse_connection_spherical_phase_odds
                or fuse_phase_defect_spherical_odds
            )
            else population_phase_survival[:, None]
        )
        positive_agreement = branch_contrast * phase_survival
        negative_agreement = (
            (1.0 - branch_contrast) * (1.0 - phase_survival))
        branch_contrast = np.divide(
            positive_agreement,
            positive_agreement + negative_agreement,
            out=np.zeros_like(branch_contrast),
            where=(
                positive_agreement + negative_agreement
                > np.finfo(float).tiny),
        )
    zero_kernel = zero_law["forward_kernels"]
    effective_kernels = np.empty_like(zero_kernel)
    maximum_harmonic_action_violation = 0.0
    tiny = np.finfo(float).tiny
    for edge in range(samples - 1):
        midpoint_contrast = 0.5 * (
            branch_contrast[edge, :, None]
            + branch_contrast[edge + 1, None, :])
        effective_conductance = (
            (1.0 - midpoint_contrast) * zero_kernel[edge]
            + midpoint_contrast * drift_kernel[edge])
        arithmetic_action = (
            (1.0 - midpoint_contrast)
            / np.maximum(zero_kernel[edge], tiny)
            + midpoint_contrast / np.maximum(drift_kernel[edge], tiny))
        arithmetic_bound_conductance = 1.0 / np.maximum(
            arithmetic_action, tiny)
        maximum_harmonic_action_violation = max(
            maximum_harmonic_action_violation,
            float(np.max(
                arithmetic_bound_conductance - effective_conductance)),
        )
        effective_kernels[edge] = effective_conductance / np.sum(
            effective_conductance, axis=1, keepdims=True)

    forward = np.empty_like(prediction)
    backward = np.empty_like(prediction)
    forward[0] = reference[0] * likelihood[0]
    forward[0] /= np.sum(forward[0])
    for index in range(1, samples):
        forward[index] = (
            forward[index - 1] @ effective_kernels[index - 1]
        ) * likelihood[index]
        forward[index] /= np.sum(forward[index])
    backward[-1] = 1.0
    for index in range(samples - 2, -1, -1):
        backward[index] = effective_kernels[index] @ (
            likelihood[index + 1] * backward[index + 1])
        backward[index] /= np.max(backward[index])
    mass = forward * backward
    mass /= np.sum(mass, axis=1, keepdims=True)

    path_forward = np.empty_like(forward)
    path_backward = np.empty_like(backward)
    path_forward[0] = reference[0] * likelihood[0] ** 2
    path_forward[0] /= np.sum(path_forward[0])
    for index in range(1, samples):
        collision_kernel = (
            effective_kernels[index - 1] ** 2
            / np.maximum(reference[index][None, :], tiny))
        path_forward[index] = (
            path_forward[index - 1] @ collision_kernel
        ) * likelihood[index] ** 2
        path_forward[index] /= np.sum(path_forward[index])
    path_backward[-1] = 1.0
    for index in range(samples - 2, -1, -1):
        collision_kernel = (
            effective_kernels[index] ** 2
            / np.maximum(reference[index + 1][None, :], tiny))
        path_backward[index] = collision_kernel @ (
            likelihood[index + 1] ** 2 * path_backward[index + 1])
        path_backward[index] /= np.max(path_backward[index])
    path_collision_mass = path_forward * path_backward
    path_collision_mass /= np.sum(
        path_collision_mass, axis=1, keepdims=True)
    return {
        "mass": mass,
        "path_collision_mass": path_collision_mass,
        "baseline_mass": zero_law["mass"],
        "prediction": prediction,
        "reference_mass": reference,
        "branch_connection_contrast": branch_contrast,
        "effective_kernels": effective_kernels,
        "phase_connection_order": phase_connection_order,
        "observation_cavity_surprise": observation_cavity_surprise,
        "transport_support_fidelity": transport_support_fidelity,
    }, {
        "state": "continuous action-contracting connection transport",
        "connection_state": (
            "complete Gaussian mean/covariance law through exact ancestry"
        ),
        "conductance_action": (
            "weighted conductance sum; equivalently weighted harmonic action"
        ),
        "edge_ownership": (
            "topological midpoint of source and target branch contrasts"
        ),
        "mean_branch_connection_contrast": float(np.mean(branch_contrast)),
        "valid_distinct_ancestry_fraction": float(np.mean(
            distinct_target_free_ancestry)),
        "parallel_transport_connection": bool(
            parallel_transport_connection),
        "population_phase_collision": bool(
            require_population_phase_collision),
        "population_phase_odds_fusion": bool(fuse_population_phase_odds),
        "connection_acceleration_odds_fusion": bool(
            fuse_connection_acceleration_odds),
        "connection_jerk_odds_fusion": bool(fuse_connection_jerk_odds),
        "connection_tangent_odds_fusion": bool(
            fuse_connection_tangent_odds),
        "connection_spherical_phase_odds_fusion": bool(
            fuse_connection_spherical_phase_odds),
        "connection_spherical_phase_union_fusion": bool(
            fuse_connection_spherical_phase_union),
        "connection_spherical_phase_suppression": bool(
            suppress_connection_on_spherical_phase),
        "phase_defect_spherical_odds_fusion": bool(
            fuse_phase_defect_spherical_odds),
        "newton_optimized_connection": bool(newton_optimize_connection),
        "marginalized_connection_action": bool(
            marginalize_connection_action),
        "marginalized_gaussian_connection": bool(
            marginalize_gaussian_connection),
        "phase_coherent_connection_posterior": bool(
            phase_coherent_connection_posterior),
        "mean_phase_connection_order": float(np.mean(
            phase_connection_order)),
        "mean_observation_cavity_surprise": float(np.mean(
            observation_cavity_surprise)),
        "mean_transport_support_fidelity": float(np.mean(
            transport_support_fidelity)),
        "phase_connection": phase_connection_diagnostic,
        "mean_connection_newton_coefficient": float(np.mean(
            newton_coefficient)),
        "mean_connection_action_posterior": float(np.mean(
            posterior_connection_mean)),
        "mean_connection_action_posterior_variance": float(np.mean(
            posterior_connection_variance)),
        "mean_population_phase_survival": float(np.mean(
            population_phase_survival)),
        "connection_tangent": tangent_diagnostic,
        "connection_spherical_phase": spherical_phase_diagnostic,
        "maximum_harmonic_action_violation": float(
            maximum_harmonic_action_violation),
        "mean_lineage_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=1))),
        "zero_connection": zero_diagnostic["bundle_metric"],
        "drift_connection": drift_diagnostic["transport_distribution"],
        "physical_parameters": "none",
    }


def action_contracting_connection_readout_forms(
    observation: np.ndarray,
    *,
    parallel_transport_connection: bool = False,
    require_population_phase_collision: bool = False,
    fuse_population_phase_odds: bool = False,
    fuse_connection_acceleration_odds: bool = False,
    fuse_connection_jerk_odds: bool = False,
    fuse_connection_tangent_odds: bool = False,
    fuse_connection_spherical_phase_odds: bool = False,
    fuse_connection_spherical_phase_union: bool = False,
    suppress_connection_on_spherical_phase: bool = False,
    fuse_phase_defect_spherical_odds: bool = False,
    newton_optimize_connection: bool = False,
    marginalize_connection_action: bool = False,
    marginalize_gaussian_connection: bool = False,
    phase_coherent_connection_posterior: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read continuous action-contracted connection transport at the end."""
    law, diagnostic = action_contracting_connection_transport_1d(
        observation,
        parallel_transport_connection=parallel_transport_connection,
        require_population_phase_collision=(
            require_population_phase_collision),
        fuse_population_phase_odds=fuse_population_phase_odds,
        fuse_connection_acceleration_odds=(
            fuse_connection_acceleration_odds),
        fuse_connection_jerk_odds=fuse_connection_jerk_odds,
        fuse_connection_tangent_odds=fuse_connection_tangent_odds,
        fuse_connection_spherical_phase_odds=(
            fuse_connection_spherical_phase_odds),
        fuse_connection_spherical_phase_union=(
            fuse_connection_spherical_phase_union),
        suppress_connection_on_spherical_phase=(
            suppress_connection_on_spherical_phase),
        fuse_phase_defect_spherical_odds=(
            fuse_phase_defect_spherical_odds),
        newton_optimize_connection=newton_optimize_connection,
        marginalize_connection_action=marginalize_connection_action,
        marginalize_gaussian_connection=marginalize_gaussian_connection,
        phase_coherent_connection_posterior=(
            phase_coherent_connection_posterior),
    )
    prediction = law["prediction"]
    reference = law["reference_mass"]
    mass = law["mass"]
    collision_mass = mass * mass / np.maximum(
        reference, np.finfo(float).tiny)
    collision_mass /= np.sum(collision_mass, axis=1, keepdims=True)
    baseline_mass = law["baseline_mass"]
    baseline_collision_mass = baseline_mass * baseline_mass / np.maximum(
        reference, np.finfo(float).tiny)
    baseline_collision_mass /= np.sum(
        baseline_collision_mass, axis=1, keepdims=True)
    return {
        "baseline_collision_mean": np.sum(
            baseline_collision_mass * prediction, axis=1),
        "mean": np.sum(mass * prediction, axis=1),
        "collision_mean": np.sum(collision_mass * prediction, axis=1),
        "path_collision_mean": np.sum(
            law["path_collision_mass"] * prediction, axis=1),
    }, diagnostic


def direct_sum_lineage_branch_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Control using an unwhitened Euclidean value/jet/residual metric."""
    return _lineage_branch_transport_1d(
        observation, bundle_metric="joint_euclidean")


def lineage_branch_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport the fused law in its determinant-one information metric."""
    return _lineage_branch_transport_1d(
        observation, bundle_metric="joint_information")


def paired_side_collision_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport disjoint-side collision particles in the joint metric."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        particle_law_builder=paired_side_collision_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "full-scale disjoint left/right affine collision before lineage"
    )
    return law, diagnostic


def nested_midpoint_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport nested midpoint/secant particles in the joint metric."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        particle_law_builder=nested_midpoint_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "full-scale nested midpoint/secant collision before lineage"
    )
    return law, diagnostic


def root_context_collision_lineage_transport_1d(
    observation: np.ndarray,
    *,
    transition_normalization: str = "action_density",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport the observed-root/context collision in the joint metric."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        transition_normalization=transition_normalization,
        particle_law_builder=root_context_collision_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "two-source observed-root and nested-context collision"
    )
    return law, diagnostic


def independent_side_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport left/right causal roles without allowing role transmutation."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        transition_normalization="action_density",
        preserve_branch_role=True,
        particle_law_builder=independent_side_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "independent nested left/right affine lineages"
    )
    return law, diagnostic


def symmetric_second_jet_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport quadratic-exact symmetric particles in the joint metric."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        transition_normalization="action_density",
        particle_law_builder=symmetric_second_jet_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "symmetric quadratic-exact value/first-jet/curvature shells"
    )
    return law, diagnostic


def symmetric_second_jet_curvature_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport quadratic particles through the second-order jet connection."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information_curvature",
        transition_normalization="action_density",
        particle_law_builder=symmetric_second_jet_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "quadratic-exact particles with parallel-transported curvature"
    )
    return law, diagnostic


def curvature_consensus_lineage_transport_1d(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport curvature-consensus particles through the jet-two bundle."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information_curvature",
        transition_normalization="action_density",
        particle_law_builder=curvature_consensus_particle_law_1d,
    )
    diagnostic["particle_law"] = (
        "quadratic particles with full W1 curvature consensus"
    )
    return law, diagnostic


def continuous_curvature_lineage_transport_1d(
    observation: np.ndarray,
    *,
    curvature_intervals: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport the continuous curvature simplex in the joint metric."""
    law, diagnostic = _lineage_branch_transport_1d(
        observation,
        bundle_metric="joint_information",
        transition_normalization="action_density",
        particle_law_builder=lambda value: continuous_curvature_particle_law_1d(
            value, curvature_intervals=curvature_intervals),
    )
    diagnostic["particle_law"] = (
        "continuous midpoint-to-quadratic reconstruction simplex"
    )
    return law, diagnostic


def _scalar_lineage_readouts(
    law: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Project a transported particle law only at its scalar endpoint."""
    prediction = law["prediction"]
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    reference_mass = law["reference_mass"]
    reference = (
        reference_mass[None, :]
        if reference_mass.ndim == 1 else reference_mass)

    def endpoints(prefix: str, mass: np.ndarray) -> dict[str, np.ndarray]:
        mean = np.sum(mass * prediction, axis=-1)
        ordered_mass = np.take_along_axis(mass, order, axis=-1)
        median_index = np.argmax(
            np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
        median = np.take_along_axis(
            ordered_prediction, median_index[:, None], axis=-1)[:, 0]
        return {f"{prefix}mean": mean, f"{prefix}median": median}

    mass = law["mass"]
    collision_mass = mass * mass / np.maximum(
        reference, np.finfo(float).tiny)
    collision_mass /= np.sum(collision_mass, axis=-1, keepdims=True)
    readouts = endpoints("", mass)
    readouts["collision_mean"] = np.sum(
        collision_mass * prediction, axis=-1)

    hj_mass = law["hj_joint_mass"]
    hj_collision_mass = law["hj_joint_collision_mass"]
    readouts.update(endpoints("hj_", hj_mass))
    readouts.update(endpoints("hj_collision_", hj_collision_mass))
    phase_collision_mass = law["hj_phase_collision_mass"]
    readouts.update(endpoints("phase_collision_", phase_collision_mass))
    coupled_phase_mass = law["hj_coupled_phase_mass"]
    coupled_phase_collision_mass = law["hj_coupled_phase_collision_mass"]
    readouts.update(endpoints("coupled_phase_", coupled_phase_mass))
    readouts.update(endpoints(
        "coupled_phase_collision_", coupled_phase_collision_mass))
    readouts.update(endpoints(
        "coupled_phase_coverage_", law["hj_coupled_phase_coverage_mass"]))
    readouts.update(endpoints(
        "coupled_phase_bundle_coverage_",
        law["hj_coupled_phase_bundle_coverage_mass"],
    ))
    readouts["global_characteristic_section"] = law["hj_viterbi_section"]
    readouts["posterior_characteristic_section"] = law[
        "posterior_characteristic_section"]
    readouts.update(endpoints("path_collision_", law["path_collision_mass"]))
    readouts.update(endpoints("path_affinity_", law["path_affinity_mass"]))
    readouts.update(endpoints("path_fidelity_", law["path_fidelity_mass"]))
    readouts.update(endpoints(
        "transport_fidelity_", law["transport_fidelity_mass"]))
    readouts.update(endpoints(
        "transport_plan_history_", law["transport_plan_history_mass"]))
    readouts.update(endpoints(
        "self_consistent_transport_", law["self_consistent_transport_mass"]))
    readouts.update(endpoints(
        "distributed_transport_", law["distributed_transport_mass"]))
    readouts.update(endpoints(
        "action_contracting_transport_",
        law["action_contracting_transport_mass"],
    ))
    readouts.update(endpoints(
        "two_history_action_transport_",
        law["two_history_action_transport_mass"],
    ))
    path_fidelity_mass = law["path_fidelity_mass"]
    path_fidelity_mean = readouts["path_fidelity_mean"]
    path_fidelity_median = readouts["path_fidelity_median"]
    deviation = prediction - path_fidelity_median[:, None]
    first_moment = np.sum(
        path_fidelity_mass * np.abs(deviation), axis=1)
    second_moment = np.sum(
        path_fidelity_mass * deviation * deviation, axis=1)
    participation = np.divide(
        first_moment * first_moment,
        second_moment,
        out=np.ones_like(first_moment),
        where=second_moment > np.finfo(float).tiny,
    )
    participation = np.clip(participation, 0.0, 1.0)
    readouts["path_fidelity_participation_section"] = (
        participation * path_fidelity_mean
        + (1.0 - participation) * path_fidelity_median
    )
    return readouts


def _energy_root_authority(
    observation: np.ndarray,
    prediction: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return root participation from the exact scalar energy distance."""
    root_action = np.sum(
        mass * np.abs(observation[:, None] - prediction), axis=-1)
    population_action = np.sum(
        mass[:, :, None]
        * mass[:, None, :]
        * np.abs(prediction[:, :, None] - prediction[:, None, :]),
        axis=(1, 2),
    )
    authority = np.divide(
        population_action,
        2.0 * root_action,
        out=np.ones_like(root_action),
        where=root_action > np.finfo(float).tiny,
    )
    return np.clip(authority, 0.0, 1.0), root_action, population_action


def _value_jet_hilbert_section(
    law: dict[str, np.ndarray],
    mass: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Return the exact Riesz section of the value/first-jet Hilbert law."""
    prediction = law["prediction"]
    jet = law["jet"]
    samples = prediction.shape[0]
    value_mean = np.sum(mass * prediction, axis=-1)
    jet_mean = np.sum(mass * jet, axis=-1)
    edge_jet = 0.5 * (jet_mean[:-1] + jet_mean[1:])
    matrix = np.eye(samples, dtype=np.float64)
    diagonal = np.full(samples, 2.0, dtype=np.float64)
    if samples > 2:
        diagonal[1:-1] = 3.0
    matrix[np.arange(samples), np.arange(samples)] = diagonal
    edge = np.arange(samples - 1)
    matrix[edge, edge + 1] = -1.0
    matrix[edge + 1, edge] = -1.0
    adjoint_jet = np.zeros(samples, dtype=np.float64)
    adjoint_jet[0] = -edge_jet[0]
    adjoint_jet[-1] = edge_jet[-1]
    if samples > 2:
        adjoint_jet[1:-1] = edge_jet[:-1] - edge_jet[1:]
    right_hand_side = value_mean + adjoint_jet
    section = np.linalg.solve(matrix, right_hand_side)

    def energy(value: np.ndarray) -> float:
        vertex = np.sum(mass * (value[:, None] - prediction) ** 2)
        edge_action = np.sum(
            0.5 * mass[:-1] * (np.diff(value)[:, None] - jet[:-1]) ** 2
            + 0.5 * mass[1:] * (np.diff(value)[:, None] - jet[1:]) ** 2
        )
        return float(vertex + edge_action)

    return section, {
        "state": "exact value/transported-first-jet Hilbert section",
        "equation": "(I + D*D)u = E[z] + D*E[j]",
        "initial_mean_energy": energy(value_mean),
        "section_energy": energy(section),
        "continuation": "none",
    }


def _value_jet_phase_sasaki_section(
    observation: np.ndarray,
    law: dict[str, np.ndarray],
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the exact scalar section of the joint value--jet phase law.

    Each endpoint particle is parallel transported to its edge midpoint.  The
    resulting ``(value, jet)`` cloud supplies its own determinant-one
    information metric, so neither coordinate receives a tunable relative
    weight.  A scalar section induces the exact midpoint state

        ((u_i + u_{i+1}) / 2, u_{i+1} - u_i).

    The global minimizer of the summed phase-space action is one symmetric
    positive-definite linear solve.  Crucially, value and jet remain a joint
    law until after this solve; their correlation is not destroyed by taking
    two marginal expectations first.
    """
    prediction = np.asarray(law["prediction"], dtype=np.float64)
    jet = np.asarray(law["jet"], dtype=np.float64)
    probability = np.asarray(mass, dtype=np.float64)
    line = _validate_line(observation)
    samples, branches = prediction.shape
    if line.size != samples:
        raise ValueError("phase root must align with particle base space")
    if jet.shape != prediction.shape or probability.shape != prediction.shape:
        raise ValueError("phase section laws must share one particle fibre")
    if float(np.ptp(prediction)) == 0.0:
        constant = prediction[:, 0].copy()
        return constant, np.ones(samples, dtype=np.float64), {
            "state": "exact determinant-one value-jet phase section",
            "initial_action": 0.0,
            "section_action": 0.0,
            "action_decrease": 0.0,
            "mean_phase_anisotropy": 1.0,
            "maximum_phase_anisotropy": 1.0,
            "phase_orientation_variation": 0.0,
            "continuation": "none",
        }

    matrix = np.zeros((samples, samples), dtype=np.float64)
    right_hand_side = np.zeros(samples, dtype=np.float64)
    edge_means = np.empty((samples - 1, 2), dtype=np.float64)
    edge_precisions = np.empty((samples - 1, 2, 2), dtype=np.float64)
    anisotropies = np.empty(samples - 1, dtype=np.float64)
    orientations = np.empty(samples - 1, dtype=np.float64)
    edge_authority = np.empty(samples - 1, dtype=np.float64)
    graph = np.asarray(((0.5, 0.5), (-1.0, 1.0)), dtype=np.float64)
    numerical_floor = _representation_floor(prediction)

    for edge in range(samples - 1):
        state = np.concatenate((
            np.stack((
                prediction[edge] + 0.5 * jet[edge],
                jet[edge],
            ), axis=-1),
            np.stack((
                prediction[edge + 1] - 0.5 * jet[edge + 1],
                jet[edge + 1],
            ), axis=-1),
        ), axis=0)
        weight = 0.5 * np.concatenate((
            probability[edge], probability[edge + 1]))
        weight /= np.sum(weight)
        mean = np.sum(weight[:, None] * state, axis=0)
        centered = state - mean
        covariance = np.einsum(
            "k,ka,kb->ab", weight, centered, centered)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        eigenvalue = np.maximum(
            eigenvalue, numerical_floor * numerical_floor)
        precision_eigenvalue = 1.0 / eigenvalue
        precision_eigenvalue /= np.sqrt(np.prod(precision_eigenvalue))
        precision = (
            eigenvector * precision_eigenvalue[None, :]
        ) @ eigenvector.T
        local_matrix = graph.T @ precision @ graph
        local_rhs = graph.T @ precision @ mean
        index = np.asarray((edge, edge + 1))
        matrix[np.ix_(index, index)] += local_matrix
        right_hand_side[index] += local_rhs
        edge_means[edge] = mean
        edge_precisions[edge] = precision
        anisotropies[edge] = float(
            np.max(precision_eigenvalue) / np.min(precision_eigenvalue))
        principal = eigenvector[:, np.argmax(eigenvalue)]
        orientations[edge] = math.atan2(principal[1], principal[0])
        particle_defect = state[:, None, :] - state[None, :, :]
        particle_distance = np.sqrt(np.maximum(np.einsum(
            "kla,ab,klb->kl", particle_defect, precision, particle_defect
        ), 0.0))
        population_action = float(np.sum(
            weight[:, None] * weight[None, :] * particle_distance))
        root_state = np.asarray((
            0.5 * (line[edge] + line[edge + 1]),
            line[edge + 1] - line[edge],
        ))
        root_defect = root_state - state
        root_distance = np.sqrt(np.maximum(np.einsum(
            "ka,ab,kb->k", root_defect, precision, root_defect), 0.0))
        root_action = float(np.sum(weight * root_distance))
        edge_authority[edge] = np.clip(
            population_action / (2.0 * root_action)
            if root_action > np.finfo(float).tiny else 1.0,
            0.0,
            1.0,
        )

    section = np.linalg.solve(matrix, right_hand_side)
    value_mean = np.sum(probability * prediction, axis=-1)

    def action(value: np.ndarray) -> float:
        state = np.stack((
            0.5 * (value[:-1] + value[1:]), np.diff(value)), axis=-1)
        defect = state - edge_means
        return float(np.sum(np.einsum(
            "ea,eab,eb->e", defect, edge_precisions, defect)))

    # Principal directions are unoriented lines.  Doubling the angle removes
    # the sign gauge before unwrapping; halving returns a continuous phase.
    continuous_orientation = 0.5 * np.unwrap(2.0 * orientations)
    initial_action = action(value_mean)
    section_action = action(section)
    vertex_authority = np.empty(samples, dtype=np.float64)
    vertex_authority[0] = edge_authority[0]
    vertex_authority[-1] = edge_authority[-1]
    if samples > 2:
        vertex_authority[1:-1] = 0.5 * (
            edge_authority[:-1] + edge_authority[1:])
    return section, vertex_authority, {
        "state": "exact determinant-one value-jet phase section",
        "equation": "sum_e B_e^T P_e B_e u = sum_e B_e^T P_e E[q_e]",
        "phase_state": "transported midpoint value and discrete first jet",
        "phase_gauge": "principal orientation modulo pi",
        "initial_action": initial_action,
        "section_action": section_action,
        "action_decrease": initial_action - section_action,
        "mean_phase_anisotropy": float(np.mean(anisotropies)),
        "maximum_phase_anisotropy": float(np.max(anisotropies)),
        "phase_orientation_variation": float(np.sum(np.abs(
            np.diff(continuous_orientation)))),
        "root_participation": "E d(Q,Q') / (2 E d(q_root,Q))",
        "minimum_root_authority": float(np.min(vertex_authority)),
        "mean_root_authority": float(np.mean(vertex_authority)),
        "maximum_root_authority": float(np.max(vertex_authority)),
        "continuation": "none",
    }


def paired_side_collision_lineage_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read continuous scalar forms after paired-side lineage transport."""
    law, diagnostic = paired_side_collision_lineage_transport_1d(observation)
    return _scalar_lineage_readouts(law), diagnostic


def nested_midpoint_lineage_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar forms after nested midpoint/secant lineage transport."""
    line = _validate_line(observation)
    law, diagnostic = nested_midpoint_lineage_transport_1d(line)
    readouts = _scalar_lineage_readouts(law)
    prediction = law["prediction"]
    mass = law["mass"]
    authority, _root_action, _population_action = _energy_root_authority(
        line, prediction, mass)
    readouts.update({
        "energy_root_mean": (
            authority * line + (1.0 - authority) * readouts["mean"]),
        "energy_root_median": (
            authority * line + (1.0 - authority) * readouts["median"]),
        "energy_root_collision_mean": (
            authority * line
            + (1.0 - authority) * readouts["collision_mean"]),
    })
    reference = law["reference_mass"]
    if reference.ndim == 1:
        reference = np.broadcast_to(reference, mass.shape)
    collision_density = np.sum(
        mass * mass / np.maximum(reference, np.finfo(float).tiny), axis=-1)
    concentration = np.clip(
        1.0 - 1.0 / np.maximum(collision_density, 1.0), 0.0, 1.0)
    supported = authority * concentration
    unsupported = (1.0 - authority) * (1.0 - concentration)
    denominator = supported + unsupported
    transported_authority = np.divide(
        supported,
        denominator,
        out=np.ones_like(supported),
        where=denominator > np.finfo(float).tiny,
    )
    readouts.update({
        "transport_energy_root_mean": (
            transported_authority * line
            + (1.0 - transported_authority) * readouts["mean"]),
        "transport_energy_root_median": (
            transported_authority * line
            + (1.0 - transported_authority) * readouts["median"]),
        "transport_energy_root_collision_mean": (
            transported_authority * line
            + (1.0 - transported_authority) * readouts["collision_mean"]),
    })
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(mass, order, axis=-1)
    cdf_midpoint = (
        np.cumsum(ordered_mass, axis=-1) - 0.5 * ordered_mass)

    def transported_quantile(section_authority: np.ndarray) -> np.ndarray:
        result = np.empty_like(line)
        for index in range(line.size):
            root_rank = np.interp(
                line[index],
                ordered_prediction[index],
                cdf_midpoint[index],
                left=0.0,
                right=1.0,
            )
            section_rank = 0.5 + section_authority[index] * (
                root_rank - 0.5)
            result[index] = np.interp(
                section_rank,
                cdf_midpoint[index],
                ordered_prediction[index],
                left=ordered_prediction[index, 0],
                right=ordered_prediction[index, -1],
            )
        return result

    readouts["energy_root_quantile"] = transported_quantile(authority)
    readouts["transport_energy_root_quantile"] = transported_quantile(
        transported_authority)
    hilbert_mean, hilbert_mean_diagnostic = _value_jet_hilbert_section(
        law, mass)
    collision_mass = mass * mass / np.maximum(
        reference, np.finfo(float).tiny)
    collision_mass /= np.sum(collision_mass, axis=-1, keepdims=True)
    hilbert_collision, hilbert_collision_diagnostic = (
        _value_jet_hilbert_section(law, collision_mass))
    phase_sasaki_mean, phase_mean_authority, phase_sasaki_mean_diagnostic = (
        _value_jet_phase_sasaki_section(line, law, mass))
    phase_sasaki_collision, phase_collision_authority, phase_sasaki_collision_diagnostic = (
        _value_jet_phase_sasaki_section(line, law, collision_mass))
    readouts["hilbert_value_jet_mean"] = hilbert_mean
    readouts["hilbert_value_jet_collision"] = hilbert_collision
    readouts["phase_sasaki_mean"] = phase_sasaki_mean
    readouts["phase_sasaki_collision"] = phase_sasaki_collision
    readouts["phase_sasaki_energy_root_mean"] = (
        phase_mean_authority * line
        + (1.0 - phase_mean_authority) * phase_sasaki_mean)
    readouts["phase_sasaki_energy_root_collision"] = (
        phase_collision_authority * line
        + (1.0 - phase_collision_authority) * phase_sasaki_collision)
    diagnostic["energy_root_participation"] = {
        "identity": "E|Z-Z'| / (2 E|y-Z|)",
        "mean_authority": float(np.mean(authority)),
        "minimum_authority": float(np.min(authority)),
        "maximum_authority": float(np.max(authority)),
        "root_enters_context_action": False,
        "mean_collision_concentration": float(np.mean(concentration)),
        "mean_transported_authority": float(np.mean(transported_authority)),
        "fusion": "addition of energy and collision-concentration log odds",
    }
    diagnostic["hilbert_value_jet_mean"] = hilbert_mean_diagnostic
    diagnostic["hilbert_value_jet_collision"] = hilbert_collision_diagnostic
    diagnostic["phase_sasaki_mean"] = phase_sasaki_mean_diagnostic
    diagnostic["phase_sasaki_collision"] = phase_sasaki_collision_diagnostic
    return readouts, diagnostic


def root_context_collision_lineage_readout_forms(
    observation: np.ndarray,
    *,
    transition_normalization: str = "action_density",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar forms after root/context collision lineage transport."""
    line = _validate_line(observation)
    law, diagnostic = root_context_collision_lineage_transport_1d(
        observation,
        transition_normalization=transition_normalization,
    )
    readouts = _scalar_lineage_readouts(law)

    # Root and context share the same Haar scale fibre.  Their normalized
    # role-conditional densities determine a Bhattacharyya/Hellinger overlap
    # O in [0,1].  The exponential density geodesic
    #     f = f_context * f_root**O
    # moves continuously from context-only transport (O=0) to the exact
    # independent-role collision density (O=1).  The observation affects the
    # support law but never enters the contextual scalar particle itself.
    scales = law["prediction"].shape[1] // 2
    context_prediction = law["prediction"][:, :scales]
    context_mass = law["mass"][:, :scales]
    root_mass = law["mass"][:, scales:]
    context_reference = law["reference_mass"][:, :scales]
    scale_reference = context_reference / (2.0 / 3.0)
    context_conditional = context_mass / np.sum(
        context_mass, axis=-1, keepdims=True)
    root_conditional = root_mass / np.sum(
        root_mass, axis=-1, keepdims=True)
    overlap = np.sum(
        np.sqrt(context_conditional * root_conditional), axis=-1)
    tiny = np.finfo(float).tiny
    context_density = context_conditional / np.maximum(
        scale_reference, tiny)
    root_density = root_conditional / np.maximum(scale_reference, tiny)
    log_density = (
        np.log(np.maximum(context_density, tiny))
        + overlap[:, None] * np.log(np.maximum(root_density, tiny))
    )
    log_density -= np.max(log_density, axis=-1, keepdims=True)
    ancestry_mass = scale_reference * np.exp(log_density)
    ancestry_mass /= np.sum(ancestry_mass, axis=-1, keepdims=True)
    ancestry_mean = np.sum(
        ancestry_mass * context_prediction, axis=-1)
    order = np.argsort(context_prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(
        context_prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(ancestry_mass, order, axis=-1)
    median_index = np.argmax(
        np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
    ancestry_median = np.take_along_axis(
        ordered_prediction, median_index[:, None], axis=-1)[:, 0]
    simplex_prediction = (
        (2.0 / 3.0) * context_prediction + (1.0 / 3.0) * line[:, None]
    )
    readouts.update({
        "ancestry_geodesic_mean": ancestry_mean,
        "ancestry_geodesic_median": ancestry_median,
        "ancestry_geodesic_simplex_mean": np.sum(
            ancestry_mass * simplex_prediction, axis=-1),
    })
    diagnostic["effective_ancestry"] = {
        "coordinate": "Hellinger affinity of role-conditional scale laws",
        "mean_overlap": float(np.mean(overlap)),
        "minimum_overlap": float(np.min(overlap)),
        "maximum_overlap": float(np.max(overlap)),
        "terminal_density": "f_context * f_root ** overlap",
        "root_enters_scalar_particle": False,
        "maximum_mass_error": float(np.max(np.abs(
            np.sum(ancestry_mass, axis=-1) - 1.0))),
    }
    return readouts, diagnostic


def independent_side_collision_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collide independently transported side densities at the endpoint."""
    law, diagnostic = independent_side_lineage_transport_1d(observation)
    branches = law["prediction"].shape[1]
    scales = branches // 2
    left_prediction = law["prediction"][:, :scales]
    right_prediction = law["prediction"][:, scales:]
    left_mass = law["mass"][:, :scales]
    right_mass = law["mass"][:, scales:]
    left_conditional = left_mass / np.sum(
        left_mass, axis=-1, keepdims=True)
    right_conditional = right_mass / np.sum(
        right_mass, axis=-1, keepdims=True)
    left_reference = 2.0 * law["reference_mass"][:, :scales]
    right_reference = 2.0 * law["reference_mass"][:, scales:]
    shared_reference = np.sqrt(left_reference * right_reference)
    shared_total = np.sum(shared_reference, axis=-1, keepdims=True)
    missing = shared_total[:, 0] <= np.finfo(float).tiny
    if np.any(missing):
        haar = 1.0 / np.arange(1, scales + 1, dtype=np.float64)
        haar /= np.sum(haar)
        shared_reference[missing] = haar
    shared_reference /= np.sum(
        shared_reference, axis=-1, keepdims=True)
    tiny = np.finfo(float).tiny
    left_density = left_conditional / np.maximum(left_reference, tiny)
    right_density = right_conditional / np.maximum(right_reference, tiny)
    collision_mass = (
        shared_reference * left_density * right_density
    )
    collision_mass /= np.sum(collision_mass, axis=-1, keepdims=True)
    prediction = 0.5 * (left_prediction + right_prediction)
    mean = np.sum(collision_mass * prediction, axis=-1)
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(collision_mass, order, axis=-1)
    median_index = np.argmax(
        np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
    median = np.take_along_axis(
        ordered_prediction, median_index[:, None], axis=-1)[:, 0]
    overlap = np.sum(
        np.sqrt(left_conditional * right_conditional), axis=-1)
    diagnostic["terminal_collision"] = {
        "density": "f_left * f_right against shared Haar scale measure",
        "scalar_particle": "midpoint of independent affine arrivals",
        "mean_hellinger_overlap": float(np.mean(overlap)),
        "minimum_hellinger_overlap": float(np.min(overlap)),
        "maximum_hellinger_overlap": float(np.max(overlap)),
        "maximum_mass_error": float(np.max(np.abs(
            np.sum(collision_mass, axis=-1) - 1.0))),
    }
    # The matched-scale form above is a falsification control: equal scale
    # labels do not imply equal transported arrivals.  The physical terminal
    # collision lives on the complete left x right product measure and couples
    # particles by reciprocal determinant-one value--jet distance.
    pooled_coordinates = np.concatenate((
        np.stack((left_prediction, law["jet"][:, :scales]), axis=-1),
        np.stack((right_prediction, law["jet"][:, scales:]), axis=-1),
    ), axis=1)
    pooled_weight = 0.5 * np.concatenate((
        left_conditional, right_conditional), axis=-1)
    center = np.einsum("nb,nba->na", pooled_weight, pooled_coordinates)
    centered = pooled_coordinates - center[:, None, :]
    covariance = np.einsum(
        "nb,nba,nbc->nac", pooled_weight, centered, centered)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    floor = _representation_floor(_validate_line(observation))
    precision_eigenvalue = 1.0 / np.maximum(eigenvalue, floor * floor)
    precision_eigenvalue /= np.sqrt(np.prod(
        precision_eigenvalue, axis=-1, keepdims=True))
    precision = np.einsum(
        "nai,ni,nbi->nab",
        eigenvector,
        precision_eigenvalue,
        eigenvector,
    )
    value_defect = (
        left_prediction[:, :, None] - right_prediction[:, None, :])
    jet_defect = (
        law["jet"][:, :scales, None]
        - law["jet"][:, None, scales:])
    defect = np.stack((value_defect, jet_defect), axis=-1)
    distance = np.sqrt(np.maximum(np.einsum(
        "nsta,nab,nstb->nst", defect, precision, defect), 0.0))
    joint_collision_mass = (
        left_conditional[:, :, None]
        * right_conditional[:, None, :]
        / np.maximum(distance, floor)
    )
    joint_collision_mass /= np.sum(
        joint_collision_mass, axis=(1, 2), keepdims=True)
    joint_prediction = 0.5 * (
        left_prediction[:, :, None] + right_prediction[:, None, :])
    joint_mean = np.sum(
        joint_collision_mass * joint_prediction, axis=(1, 2))
    flat_prediction = joint_prediction.reshape(joint_prediction.shape[0], -1)
    flat_mass = joint_collision_mass.reshape(joint_collision_mass.shape[0], -1)
    flat_order = np.argsort(flat_prediction, axis=-1, kind="stable")
    ordered_joint_prediction = np.take_along_axis(
        flat_prediction, flat_order, axis=-1)
    ordered_joint_mass = np.take_along_axis(flat_mass, flat_order, axis=-1)
    joint_median_index = np.argmax(
        np.cumsum(ordered_joint_mass, axis=-1) >= 0.5, axis=-1)
    joint_median = np.take_along_axis(
        ordered_joint_prediction,
        joint_median_index[:, None],
        axis=-1,
    )[:, 0]
    diagnostic["joint_terminal_collision"] = {
        "measure": "complete left-scale x right-scale product law",
        "kernel": "reciprocal determinant-one value-jet arrival distance",
        "mean_information_anisotropy": float(np.mean(
            np.max(precision_eigenvalue, axis=-1)
            / np.min(precision_eigenvalue, axis=-1))),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(
                joint_collision_mass * joint_collision_mass, axis=(1, 2)))),
        "maximum_mass_error": float(np.max(np.abs(
            np.sum(joint_collision_mass, axis=(1, 2)) - 1.0))),
    }
    return {
        "mean": mean,
        "median": median,
        "joint_mean": joint_mean,
        "joint_median": joint_median,
    }, diagnostic


def symmetric_second_jet_lineage_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read continuous scalar forms after symmetric second-jet lineage."""
    law, diagnostic = symmetric_second_jet_lineage_transport_1d(observation)
    return _scalar_lineage_readouts(law), diagnostic


def symmetric_second_jet_curvature_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar forms only after second-order bundle transport."""
    law, diagnostic = symmetric_second_jet_curvature_transport_1d(observation)
    return _scalar_lineage_readouts(law), diagnostic


def curvature_consensus_lineage_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar forms after full curvature-population lineage."""
    law, diagnostic = curvature_consensus_lineage_transport_1d(observation)
    return _scalar_lineage_readouts(law), diagnostic


def continuous_curvature_lineage_readout_forms(
    observation: np.ndarray,
    *,
    curvature_intervals: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar and energy-root forms after curvature-simplex lineage."""
    line = _validate_line(observation)
    law, diagnostic = continuous_curvature_lineage_transport_1d(
        line, curvature_intervals=curvature_intervals)
    readouts = _scalar_lineage_readouts(law)
    authority, _root_action, _population_action = _energy_root_authority(
        line, law["prediction"], law["mass"])
    readouts.update({
        "energy_root_mean": (
            authority * line + (1.0 - authority) * readouts["mean"]),
        "energy_root_median": (
            authority * line + (1.0 - authority) * readouts["median"]),
        "energy_root_collision_mean": (
            authority * line
            + (1.0 - authority) * readouts["collision_mean"]),
    })
    effective_prediction = (
        law["midpoint_prediction"]
        + authority[:, None]
        * (law["prediction"] - law["midpoint_prediction"])
    )
    effective_law = dict(law)
    effective_law["prediction"] = effective_prediction
    authority_curvature = _scalar_lineage_readouts(effective_law)
    readouts.update({
        "authority_curvature_mean": authority_curvature["mean"],
        "authority_curvature_median": authority_curvature["median"],
        "authority_curvature_collision_mean": authority_curvature[
            "collision_mean"],
    })
    curvature_coordinate = law["curvature_coordinate"]
    diagnostic["energy_root_participation"] = {
        "identity": "E|Z-Z'| / (2 E|y-Z|)",
        "mean_authority": float(np.mean(authority)),
        "minimum_authority": float(np.min(authority)),
        "maximum_authority": float(np.max(authority)),
        "root_enters_context_action": False,
    }
    diagnostic["transported_curvature_coordinate"] = {
        "mean": float(np.mean(np.sum(
            law["mass"] * curvature_coordinate[None, :], axis=-1))),
        "mean_after_root_authority": float(np.mean(np.sum(
            law["mass"]
            * curvature_coordinate[None, :]
            * authority[:, None],
            axis=-1,
        ))),
        "representation_intervals": int(curvature_intervals),
    }
    return readouts, diagnostic


def _weighted_scalar_median(
    value: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return the lower W1 barycenter of one finite positive measure."""
    order = np.argsort(value, kind="stable")
    ordered_value = value[order]
    ordered_weight = weight[order]
    index = int(np.argmax(
        np.cumsum(ordered_weight) >= 0.5 * np.sum(ordered_weight)))
    return float(ordered_value[index])


def _joint_w1_value_jet_section(
    prediction: np.ndarray,
    jet: np.ndarray,
    mass: np.ndarray,
    initial: np.ndarray,
    numerical_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize one joint W1 value/transported-jet action on the line.

    Every vertex contributes its complete latent-value law.  Every edge pools
    the two endpoint jet laws with equal measure, because both are parallel
    descriptions of the same topological edge.  With neighboring coordinates
    fixed, the exact minimizer is a weighted median; symmetric coordinate
    sweeps therefore descend the stated nonsmooth action without a step size.
    """
    samples, branches = prediction.shape
    edge_jet = np.concatenate((jet[:-1], jet[1:]), axis=1)
    edge_mass = 0.5 * np.concatenate((mass[:-1], mass[1:]), axis=1)
    section = np.asarray(initial, dtype=np.float64).copy()

    def action(value: np.ndarray) -> float:
        vertex = np.sum(mass * np.abs(value[:, None] - prediction))
        edge = np.sum(edge_mass * np.abs(
            np.diff(value)[:, None] - edge_jet))
        return float(vertex + edge)

    initial_action = action(section)
    previous_action = initial_action
    maximum_sweeps = 8 * samples
    resolved = False
    sweeps = 0
    for sweeps in range(1, maximum_sweeps + 1):
        changed = 0.0
        for traversal in (range(samples), range(samples - 1, -1, -1)):
            for index in traversal:
                values = [prediction[index]]
                weights = [mass[index]]
                if index > 0:
                    values.append(section[index - 1] + edge_jet[index - 1])
                    weights.append(edge_mass[index - 1])
                if index + 1 < samples:
                    values.append(section[index + 1] - edge_jet[index])
                    weights.append(edge_mass[index])
                update = _weighted_scalar_median(
                    np.concatenate(values), np.concatenate(weights))
                changed = max(changed, abs(update - section[index]))
                section[index] = update
        current_action = action(section)
        tolerance = math.sqrt(np.finfo(float).eps) * max(
            previous_action, numerical_floor)
        if current_action > previous_action + tolerance:
            raise RuntimeError("joint W1 coordinate action increased")
        if changed <= numerical_floor or previous_action - current_action <= tolerance:
            resolved = True
            previous_action = current_action
            break
        previous_action = current_action
    return section, {
        "state": "joint W1 value/transported-jet coordinate equilibrium",
        "initial_action": float(initial_action),
        "final_action": float(previous_action),
        "action_decrease": float(initial_action - previous_action),
        "sweeps": int(sweeps),
        "resolved": bool(resolved),
        "ceiling": int(maximum_sweeps),
        "edge_measure": "equal pool of both transported endpoint jet laws",
    }


def _joint_information_field_section(
    observation: np.ndarray,
    law: dict[str, np.ndarray],
    initial: np.ndarray,
    numerical_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a scalar field by one determinant-one joint bundle action.

    At each edge the candidate scalar field induces two exact-graph particles:
    the same midpoint latent value and discrete jet, with the residual carried
    from each observation endpoint. Their distances to the corresponding
    transported branch particles are measured in the edge's determinant-one
    information precision. This keeps value, jet, and residual coupled in one
    convex action instead of assigning them a relative penalty.
    """
    line = np.asarray(observation, dtype=np.float64)
    prediction = law["prediction"]
    jet = law["jet"]
    mass = law["mass"]
    precision = law["edge_precision"]
    samples = line.size
    if float(np.ptp(line)) == 0.0:
        return line.copy(), {
            "state": "determinant-one joint information field equilibrium",
            "initial_action": 0.0,
            "final_action": 0.0,
            "action_decrease": 0.0,
            "iterations": 0,
            "resolved": True,
            "message": "constant exact state",
        }

    left_value = prediction[:-1] + 0.5 * jet[:-1]
    right_value = prediction[1:] - 0.5 * jet[1:]
    half_mass_left = 0.5 * mass[:-1]
    half_mass_right = 0.5 * mass[1:]
    smooth_floor_squared = numerical_floor * numerical_floor

    def objective_and_gradient(value: np.ndarray):
        midpoint = 0.5 * (value[:-1] + value[1:])
        difference = value[1:] - value[:-1]
        left_defect = np.stack((
            midpoint[:, None] - left_value,
            difference[:, None] - jet[:-1],
            prediction[:-1] - value[:-1, None],
        ), axis=-1)
        right_defect = np.stack((
            midpoint[:, None] - right_value,
            difference[:, None] - jet[1:],
            prediction[1:] - value[1:, None],
        ), axis=-1)
        left_covector = np.einsum(
            "eab,ekb->eka", precision, left_defect)
        right_covector = np.einsum(
            "eab,ekb->eka", precision, right_defect)
        left_norm = np.sqrt(np.maximum(
            np.einsum("eka,eka->ek", left_defect, left_covector),
            0.0,
        ) + smooth_floor_squared)
        right_norm = np.sqrt(np.maximum(
            np.einsum("eka,eka->ek", right_defect, right_covector),
            0.0,
        ) + smooth_floor_squared)
        action = float(np.sum(
            half_mass_left * left_norm + half_mass_right * right_norm))
        left_unit = half_mass_left[..., None] * (
            left_covector / left_norm[..., None])
        right_unit = half_mass_right[..., None] * (
            right_covector / right_norm[..., None])
        gradient = np.zeros(samples, dtype=np.float64)
        gradient[:-1] += np.sum(
            0.5 * left_unit[..., 0]
            - left_unit[..., 1]
            - left_unit[..., 2],
            axis=1,
        )
        gradient[1:] += np.sum(
            0.5 * left_unit[..., 0] + left_unit[..., 1],
            axis=1,
        )
        gradient[:-1] += np.sum(
            0.5 * right_unit[..., 0] - right_unit[..., 1],
            axis=1,
        )
        gradient[1:] += np.sum(
            0.5 * right_unit[..., 0]
            + right_unit[..., 1]
            - right_unit[..., 2],
            axis=1,
        )
        return action, gradient

    initial_value = np.asarray(initial, dtype=np.float64)
    initial_action = objective_and_gradient(initial_value)[0]
    result = optimize.minimize(
        objective_and_gradient,
        initial_value,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 8 * samples,
            "ftol": np.finfo(float).eps,
            "gtol": math.sqrt(np.finfo(float).eps),
        },
    )
    final_action = objective_and_gradient(result.x)[0]
    tolerance = math.sqrt(np.finfo(float).eps) * max(
        initial_action, numerical_floor)
    if final_action > initial_action + tolerance:
        raise RuntimeError("joint information field action increased")
    return np.asarray(result.x, dtype=np.float64), {
        "state": "determinant-one joint information field equilibrium",
        "initial_action": float(initial_action),
        "final_action": float(final_action),
        "action_decrease": float(initial_action - final_action),
        "iterations": int(result.nit),
        "resolved": bool(result.success),
        "message": str(result.message),
        "ceiling": int(8 * samples),
        "joint_coordinates": (
            "midpoint latent value, discrete jet, exact endpoint residual"
        ),
    }


def lineage_branch_readout_forms(
    observation: np.ndarray,
    *,
    include_experimental: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar sections only after positive branch-lineage transport."""
    law, diagnostic = lineage_branch_transport_1d(observation)
    prediction = law["prediction"]
    mass = law["mass"]
    reference_mass = law["reference_mass"]
    reference = (
        reference_mass[None, :]
        if reference_mass.ndim == 1 else reference_mass)
    mean = np.sum(mass * prediction, axis=-1)
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(mass, order, axis=-1)
    median_index = np.argmax(
        np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
    median = np.take_along_axis(
        ordered_prediction, median_index[:, None], axis=-1)[:, 0]
    maximum_branch = np.take_along_axis(
        prediction, np.argmax(mass, axis=-1)[:, None], axis=-1)[:, 0]
    collision_mass = mass * mass / np.maximum(
        reference, np.finfo(float).tiny)
    collision_mass /= np.sum(collision_mass, axis=-1, keepdims=True)
    collision_mean = np.sum(collision_mass * prediction, axis=-1)
    collision_ordered_mass = np.take_along_axis(
        collision_mass, order, axis=-1)
    collision_median_index = np.argmax(
        np.cumsum(collision_ordered_mass, axis=-1) >= 0.5, axis=-1)
    collision_median = np.take_along_axis(
        ordered_prediction,
        collision_median_index[:, None],
        axis=-1,
    )[:, 0]
    oriented_sections = []
    for oriented_mass in (law["forward_mass"], law["backward_mass"]):
        oriented_collision = (
            oriented_mass * oriented_mass
            / np.maximum(reference, np.finfo(float).tiny))
        oriented_sections.append(np.take_along_axis(
            prediction,
            np.argmax(oriented_collision, axis=-1)[:, None],
            axis=-1,
        )[:, 0])
    oriented_collision_section = 0.5 * (
        oriented_sections[0] + oriented_sections[1])
    symmetric_parent_mass = law["symmetric_parent_mass"]
    symmetric_parent_mean = np.sum(
        symmetric_parent_mass * prediction, axis=-1)
    symmetric_ordered_mass = np.take_along_axis(
        symmetric_parent_mass, order, axis=-1)
    symmetric_median_index = np.argmax(
        np.cumsum(symmetric_ordered_mass, axis=-1) >= 0.5, axis=-1)
    symmetric_parent_median = np.take_along_axis(
        ordered_prediction,
        symmetric_median_index[:, None],
        axis=-1,
    )[:, 0]
    hj_joint_mass = law["hj_joint_mass"]
    hj_joint_mean = np.sum(hj_joint_mass * prediction, axis=-1)
    hj_ordered_mass = np.take_along_axis(hj_joint_mass, order, axis=-1)
    hj_median_index = np.argmax(
        np.cumsum(hj_ordered_mass, axis=-1) >= 0.5, axis=-1)
    hj_joint_median = np.take_along_axis(
        ordered_prediction, hj_median_index[:, None], axis=-1)[:, 0]
    hj_joint_collision_mass = law["hj_joint_collision_mass"]
    hj_joint_collision_mean = np.sum(
        hj_joint_collision_mass * prediction, axis=-1)
    hj_collision_ordered_mass = np.take_along_axis(
        hj_joint_collision_mass, order, axis=-1)
    hj_collision_median_index = np.argmax(
        np.cumsum(hj_collision_ordered_mass, axis=-1) >= 0.5, axis=-1)
    hj_joint_collision_median = np.take_along_axis(
        ordered_prediction,
        hj_collision_median_index[:, None],
        axis=-1,
    )[:, 0]
    hj_phase_collision_mass = law["hj_phase_collision_mass"]
    hj_phase_collision_mean = np.sum(
        hj_phase_collision_mass * prediction, axis=-1)
    hj_phase_collision_ordered_mass = np.take_along_axis(
        hj_phase_collision_mass, order, axis=-1)
    hj_phase_collision_median_index = np.argmax(
        np.cumsum(hj_phase_collision_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    hj_phase_collision_median = np.take_along_axis(
        ordered_prediction,
        hj_phase_collision_median_index[:, None],
        axis=-1,
    )[:, 0]
    hj_coupled_phase_mass = law["hj_coupled_phase_mass"]
    hj_coupled_phase_mean = np.sum(
        hj_coupled_phase_mass * prediction, axis=-1)
    hj_coupled_phase_ordered_mass = np.take_along_axis(
        hj_coupled_phase_mass, order, axis=-1)
    hj_coupled_phase_median_index = np.argmax(
        np.cumsum(hj_coupled_phase_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    hj_coupled_phase_median = np.take_along_axis(
        ordered_prediction,
        hj_coupled_phase_median_index[:, None],
        axis=-1,
    )[:, 0]
    hj_coupled_phase_collision_mass = law[
        "hj_coupled_phase_collision_mass"]
    hj_coupled_phase_collision_mean = np.sum(
        hj_coupled_phase_collision_mass * prediction, axis=-1)
    hj_coupled_phase_collision_ordered_mass = np.take_along_axis(
        hj_coupled_phase_collision_mass, order, axis=-1)
    hj_coupled_phase_collision_median_index = np.argmax(
        np.cumsum(hj_coupled_phase_collision_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    hj_coupled_phase_collision_median = np.take_along_axis(
        ordered_prediction,
        hj_coupled_phase_collision_median_index[:, None],
        axis=-1,
    )[:, 0]
    hj_coupled_phase_coverage_mass = law[
        "hj_coupled_phase_coverage_mass"]
    hj_coupled_phase_coverage_mean = np.sum(
        hj_coupled_phase_coverage_mass * prediction, axis=-1)
    hj_coupled_phase_coverage_ordered_mass = np.take_along_axis(
        hj_coupled_phase_coverage_mass, order, axis=-1)
    hj_coupled_phase_coverage_median_index = np.argmax(
        np.cumsum(hj_coupled_phase_coverage_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    hj_coupled_phase_coverage_median = np.take_along_axis(
        ordered_prediction,
        hj_coupled_phase_coverage_median_index[:, None],
        axis=-1,
    )[:, 0]
    hj_coupled_phase_bundle_coverage_mass = law[
        "hj_coupled_phase_bundle_coverage_mass"]
    hj_coupled_phase_bundle_coverage_mean = np.sum(
        hj_coupled_phase_bundle_coverage_mass * prediction, axis=-1)
    hj_coupled_phase_bundle_coverage_ordered_mass = np.take_along_axis(
        hj_coupled_phase_bundle_coverage_mass, order, axis=-1)
    hj_coupled_phase_bundle_coverage_median_index = np.argmax(
        np.cumsum(
            hj_coupled_phase_bundle_coverage_ordered_mass, axis=-1
        ) >= 0.5,
        axis=-1,
    )
    hj_coupled_phase_bundle_coverage_median = np.take_along_axis(
        ordered_prediction,
        hj_coupled_phase_bundle_coverage_median_index[:, None],
        axis=-1,
    )[:, 0]
    path_collision_mass = law["path_collision_mass"]
    path_collision_mean = np.sum(
        path_collision_mass * prediction, axis=-1)
    path_collision_ordered_mass = np.take_along_axis(
        path_collision_mass, order, axis=-1)
    path_collision_median_index = np.argmax(
        np.cumsum(path_collision_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    path_collision_median = np.take_along_axis(
        ordered_prediction,
        path_collision_median_index[:, None],
        axis=-1,
    )[:, 0]
    path_affinity_mass = law["path_affinity_mass"]
    path_affinity_mean = np.sum(path_affinity_mass * prediction, axis=-1)
    path_affinity_ordered_mass = np.take_along_axis(
        path_affinity_mass, order, axis=-1)
    path_affinity_median_index = np.argmax(
        np.cumsum(path_affinity_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    path_affinity_median = np.take_along_axis(
        ordered_prediction,
        path_affinity_median_index[:, None],
        axis=-1,
    )[:, 0]
    path_fidelity_mass = law["path_fidelity_mass"]
    path_fidelity_mean = np.sum(path_fidelity_mass * prediction, axis=-1)
    path_fidelity_ordered_mass = np.take_along_axis(
        path_fidelity_mass, order, axis=-1)
    path_fidelity_median_index = np.argmax(
        np.cumsum(path_fidelity_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    path_fidelity_median = np.take_along_axis(
        ordered_prediction,
        path_fidelity_median_index[:, None],
        axis=-1,
    )[:, 0]
    transport_fidelity_mass = law["transport_fidelity_mass"]
    transport_fidelity_mean = np.sum(
        transport_fidelity_mass * prediction, axis=-1)
    transport_fidelity_ordered_mass = np.take_along_axis(
        transport_fidelity_mass, order, axis=-1)
    transport_fidelity_median_index = np.argmax(
        np.cumsum(transport_fidelity_ordered_mass, axis=-1) >= 0.5,
        axis=-1,
    )
    transport_fidelity_median = np.take_along_axis(
        ordered_prediction,
        transport_fidelity_median_index[:, None],
        axis=-1,
    )[:, 0]
    transport_plan_history_mass = law["transport_plan_history_mass"]
    transport_plan_history_mean = np.sum(
        transport_plan_history_mass * prediction, axis=-1)
    self_consistent_transport_mass = law["self_consistent_transport_mass"]
    self_consistent_transport_mean = np.sum(
        self_consistent_transport_mass * prediction, axis=-1)
    distributed_transport_mass = law["distributed_transport_mass"]
    distributed_transport_mean = np.sum(
        distributed_transport_mass * prediction, axis=-1)
    action_contracting_transport_mass = law[
        "action_contracting_transport_mass"]
    action_contracting_transport_mean = np.sum(
        action_contracting_transport_mass * prediction, axis=-1)
    two_history_action_transport_mass = law[
        "two_history_action_transport_mass"]
    two_history_action_transport_mean = np.sum(
        two_history_action_transport_mass * prediction, axis=-1)
    path_fidelity_deviation = prediction - path_fidelity_median[:, None]
    path_fidelity_first_moment = np.sum(
        path_fidelity_mass * np.abs(path_fidelity_deviation), axis=1)
    path_fidelity_second_moment = np.sum(
        path_fidelity_mass * path_fidelity_deviation ** 2, axis=1)
    path_fidelity_participation = np.divide(
        path_fidelity_first_moment ** 2,
        path_fidelity_second_moment,
        out=np.ones_like(path_fidelity_first_moment),
        where=path_fidelity_second_moment > np.finfo(float).tiny,
    )
    path_fidelity_participation = np.clip(
        path_fidelity_participation, 0.0, 1.0)
    path_fidelity_participation_section = (
        path_fidelity_participation * path_fidelity_mean
        + (1.0 - path_fidelity_participation) * path_fidelity_median
    )
    transport_history_participation_mean = (
        path_fidelity_participation * collision_mean
        + (1.0 - path_fidelity_participation) * path_fidelity_mean
    )
    transport_history_participation_median = (
        path_fidelity_participation * collision_mean
        + (1.0 - path_fidelity_participation) * path_fidelity_median
    )
    result = {
        "mean": mean,
        "median": median,
        "maximum_branch": maximum_branch,
        "collision_mean": collision_mean,
        "collision_median": collision_median,
        "oriented_collision_section": oriented_collision_section,
        "symmetric_parent_mean": symmetric_parent_mean,
        "symmetric_parent_median": symmetric_parent_median,
        "hj_joint_mean": hj_joint_mean,
        "hj_joint_median": hj_joint_median,
        "hj_joint_collision_mean": hj_joint_collision_mean,
        "hj_joint_collision_median": hj_joint_collision_median,
        "hj_phase_collision_mean": hj_phase_collision_mean,
        "hj_phase_collision_median": hj_phase_collision_median,
        "hj_coupled_phase_mean": hj_coupled_phase_mean,
        "hj_coupled_phase_median": hj_coupled_phase_median,
        "hj_coupled_phase_collision_mean": hj_coupled_phase_collision_mean,
        "hj_coupled_phase_collision_median": hj_coupled_phase_collision_median,
        "hj_coupled_phase_coverage_mean": hj_coupled_phase_coverage_mean,
        "hj_coupled_phase_coverage_median": hj_coupled_phase_coverage_median,
        "hj_coupled_phase_bundle_coverage_mean": (
            hj_coupled_phase_bundle_coverage_mean),
        "hj_coupled_phase_bundle_coverage_median": (
            hj_coupled_phase_bundle_coverage_median),
        "hj_global_characteristic_section": law["hj_viterbi_section"],
        "posterior_characteristic_section": law[
            "posterior_characteristic_section"],
        "path_collision_mean": path_collision_mean,
        "path_collision_median": path_collision_median,
        "path_affinity_mean": path_affinity_mean,
        "path_affinity_median": path_affinity_median,
        "path_fidelity_mean": path_fidelity_mean,
        "path_fidelity_median": path_fidelity_median,
        "transport_fidelity_mean": transport_fidelity_mean,
        "transport_fidelity_median": transport_fidelity_median,
        "transport_plan_history_mean": transport_plan_history_mean,
        "self_consistent_transport_mean": self_consistent_transport_mean,
        "distributed_transport_mean": distributed_transport_mean,
        "action_contracting_transport_mean": (
            action_contracting_transport_mean),
        "two_history_action_transport_mean": (
            two_history_action_transport_mean),
        "path_fidelity_participation_section": (
            path_fidelity_participation_section),
        "transport_history_participation_mean": (
            transport_history_participation_mean),
        "transport_history_participation_median": (
            transport_history_participation_median),
    }
    line = _validate_line(observation)
    energy_authority, _root_action, _population_action = (
        _energy_root_authority(line, prediction, mass))
    result.update({
        "energy_root_mean": (
            energy_authority * line + (1.0 - energy_authority) * mean),
        "energy_root_median": (
            energy_authority * line + (1.0 - energy_authority) * median),
        "energy_root_collision_mean": (
            energy_authority * line
            + (1.0 - energy_authority) * collision_mean),
    })
    diagnostic["energy_root_participation"] = {
        "identity": "E|Z-Z'| / (2 E|y-Z|)",
        "mean_authority": float(np.mean(energy_authority)),
        "minimum_authority": float(np.min(energy_authority)),
        "maximum_authority": float(np.max(energy_authority)),
        "particle_action_target_excluded": False,
        "status": "empirical endpoint control; local action leakage remains",
    }
    diagnostic["path_fidelity_participation"] = {
        "identity": "(E|Z-median|)^2 / E|Z-median|^2",
        "mean_barycentric_participation": float(np.mean(
            path_fidelity_participation)),
        "minimum_barycentric_participation": float(np.min(
            path_fidelity_participation)),
        "maximum_barycentric_participation": float(np.max(
            path_fidelity_participation)),
        "complement": "W1-median participation",
        "physical_parameters": "none",
    }
    if include_experimental:
        joint_w1_value_jet, joint_w1_diagnostic = _joint_w1_value_jet_section(
            prediction,
            law["jet"],
            mass,
            median,
            _representation_floor(_validate_line(observation)),
        )
        diagnostic["joint_w1_value_jet"] = joint_w1_diagnostic
        joint_information_field, joint_information_diagnostic = (
            _joint_information_field_section(
                _validate_line(observation),
                law,
                median,
                _representation_floor(_validate_line(observation)),
            )
        )
        diagnostic["joint_information_field"] = joint_information_diagnostic
        result["joint_w1_value_jet"] = joint_w1_value_jet
        result["joint_information_field"] = joint_information_field
    return result, diagnostic


def denoise_information_lineage_transport(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One-pass W1 readout after fused information-lineage transport."""
    forms, lineage = lineage_branch_readout_forms(observation)
    return np.clip(forms["median"], 0.0, 1.0), {
        "status": "one-pass fused information-lineage section",
        "theory_status": "positive 1-D research candidate; not GUI-promoted",
        "readout": "W1 median after positive lineage transport",
        "continuation": "none",
        "lineage": lineage,
        "rejected_alternatives": [
            "signal-only Euclidean lineage deflates structure",
            "unwhitened joint lineage gives back replacement recovery",
            "residual continuation restores corruption",
            "maximum and collision-conditioned branches are unstable",
        ],
    }


def causal_collision_particle_law(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a target-excluded law from collisions of affine continuations.

    At each scale, two independently based copies of the same characteristic
    are prolonged to the target.  Their arrival discrepancy is the action.
    The observed value at the target is therefore absent from both the
    prediction and its conductance (apart from the still-provisional reflected
    boundary closure).  Support is agreement produced by transport, not a
    rule applied to an observed residual.
    """
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 2
    branch_count = 3 * maximum_lag
    if float(np.ptp(line)) == 0.0:
        lag = np.repeat(
            np.arange(1, maximum_lag + 1, dtype=np.float64), 3)
        base = 1.0 / lag
        mass = np.broadcast_to(
            base / np.sum(base), (samples, branch_count)).copy()
        prediction = np.broadcast_to(
            line[:, None], (samples, branch_count)).copy()
        return {
            "prediction": prediction,
            "conjugate_prediction": prediction.copy(),
            "mass": mass,
            "lag": lag,
            "path_family": np.tile(np.arange(3), maximum_lag),
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(branch_count),
            "mean_collision_population": float(
                1.0 / np.sum(mass[0] * mass[0])),
            "target_value_enters_interior_action": False,
        }

    # Three shells are required so the one-sided characteristics have two
    # disjoint affine germs.  Reflection is only a numerical boundary closure;
    # the interior law is pointwise observation-excluded.
    padded = np.pad(line, 3 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 3 * maximum_lag
    floor = _representation_floor(line)
    prediction_fields = []
    conjugate_fields = []
    action_fields = []
    lag_fields = []
    family_fields = []
    collision_means = []
    for lag in range(1, maximum_lag + 1):
        left_one = padded[index - lag]
        right_one = padded[index + lag]
        left_two = padded[index - 2 * lag]
        right_two = padded[index + 2 * lag]
        left_three = padded[index - 3 * lag]
        right_three = padded[index + 3 * lag]

        # Opposite arrivals use nested symmetric shells.  One-sided arrivals
        # prolong two adjacent, non-identical affine germs to the same target.
        characteristics = (
            (
                0.5 * (left_one + right_one),
                0.5 * (left_two + right_two),
                0.5 * np.abs(right_one - left_one)
                + 0.5 * np.abs(right_two - left_two),
            ),
            (
                2.0 * left_one - left_two,
                3.0 * left_two - 2.0 * left_three,
                np.abs(left_one - left_two)
                + np.abs(left_two - left_three),
            ),
            (
                2.0 * right_one - right_two,
                3.0 * right_two - 2.0 * right_three,
                np.abs(right_one - right_two)
                + np.abs(right_two - right_three),
            ),
        )
        for family, (prediction, conjugate, path_variation) in enumerate(
                characteristics):
            collision_action = np.abs(prediction - conjugate)
            total_action = collision_action + path_variation
            prediction_fields.append(prediction)
            conjugate_fields.append(conjugate)
            action_fields.append(total_action)
            lag_fields.append(float(lag))
            family_fields.append(family)
            collision_means.append(float(np.mean(collision_action)))

    prediction = np.stack(prediction_fields, axis=-1)
    conjugate = np.stack(conjugate_fields, axis=-1)
    local_action = np.stack(action_fields, axis=-1)
    lag_coordinate = np.asarray(lag_fields, dtype=np.float64)
    haar = 1.0 / lag_coordinate
    haar /= np.sum(haar)

    # A branch has support only to the extent that it belongs to the arrival
    # population.  This is the exact weighted absolute-deviation potential of
    # the empirical W1 law; unlike a kernel density it introduces no bandwidth.
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_haar = np.take_along_axis(
        np.broadcast_to(haar, prediction.shape), order, axis=-1)
    cumulative_mass = np.cumsum(ordered_haar, axis=-1)
    cumulative_moment = np.cumsum(
        ordered_haar * ordered_prediction, axis=-1)
    total_moment = cumulative_moment[:, -1:]
    population_action_ordered = (
        ordered_prediction * cumulative_mass - cumulative_moment
        + (total_moment - cumulative_moment)
        - ordered_prediction * (1.0 - cumulative_mass)
    )
    inverse_order = np.argsort(order, axis=-1, kind="stable")
    population_action = np.take_along_axis(
        population_action_ordered, inverse_order, axis=-1)
    total_action = local_action + population_action
    conductance = haar[None, :] / np.maximum(total_action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "conjugate_prediction": conjugate,
        "mass": mass,
        "lag": np.asarray(lag_fields, dtype=np.float64),
        "path_family": np.asarray(family_fields, dtype=np.int64),
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(branch_count),
        "mean_collision_action": float(np.mean(collision_means)),
        "mean_population_action": float(np.mean(population_action)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "target_value_enters_interior_action": False,
        "boundary_closure": "reflection; not yet continuum-normalized",
    }


def causal_collision_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read mean, W1 barycenter, and maximum branch from the collision law."""
    law, diagnostic = causal_collision_particle_law(observation)
    prediction = law["prediction"]
    mass = law["mass"]
    mean = np.sum(mass * prediction, axis=-1)
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(mass, order, axis=-1)
    median_index = np.argmax(
        np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
    median = np.take_along_axis(
        ordered_prediction, median_index[:, None], axis=-1)[:, 0]
    maximum_branch = np.take_along_axis(
        prediction, np.argmax(mass, axis=-1)[:, None], axis=-1)[:, 0]
    return {
        "mean": mean,
        "median": median,
        "maximum_branch": maximum_branch,
    }, diagnostic


def causal_crossfit_particle_law(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score each characteristic outside the target's dependency ancestry.

    A lag-s characteristic owns a two-shell domain.  Predictive errors are
    integrated on that entire domain after removing the target and every site
    whose predictor reads the target.  Thus the action remains local while
    the target cannot certify its own reconstruction.
    """
    line = _validate_line(observation)
    samples = line.size
    maximum_lag = samples // 2
    branch_count = 3 * maximum_lag
    if float(np.ptp(line)) == 0.0:
        lag = np.repeat(
            np.arange(1, maximum_lag + 1, dtype=np.float64), 3)
        base = 1.0 / lag
        mass = np.broadcast_to(
            base / np.sum(base), (samples, branch_count)).copy()
        return {
            "prediction": np.broadcast_to(
                line[:, None], (samples, branch_count)).copy(),
            "mass": mass,
            "lag": lag,
            "path_family": np.tile(np.arange(3), maximum_lag),
        }, {
            "minimum_lag": 1,
            "maximum_lag": int(maximum_lag),
            "characteristic_count": int(branch_count),
            "target_value_enters_interior_action": False,
        }

    padded = np.pad(line, 2 * maximum_lag, mode="reflect")
    index = np.arange(samples) + 2 * maximum_lag
    floor = _representation_floor(line)
    prediction_fields = []
    conductance_fields = []
    total_action_fields = []
    lag_fields = []
    family_fields = []
    action_means = []
    for lag in range(1, maximum_lag + 1):
        left_one = padded[index - lag]
        right_one = padded[index + lag]
        left_two = padded[index - 2 * lag]
        right_two = padded[index + 2 * lag]
        characteristics = (
            (
                0.5 * (left_one + right_one),
                0.5 * np.abs(right_one - left_one),
                (-lag, 0, lag),
            ),
            (
                2.0 * left_one - left_two,
                np.abs(left_one - left_two),
                (0, lag, 2 * lag),
            ),
            (
                2.0 * right_one - right_two,
                np.abs(right_one - right_two),
                (-2 * lag, -lag, 0),
            ),
        )
        for family, (prediction, dispersion, excluded) in enumerate(
                characteristics):
            point_error = np.abs(prediction - line)
            domain_sum = ndimage.uniform_filter1d(
                point_error, size=4 * lag + 1, mode="reflect")
            domain_sum *= 4 * lag + 1
            error_pad = np.pad(point_error, 2 * maximum_lag, mode="reflect")
            excluded_sum = sum(error_pad[index + offset] for offset in excluded)
            predictive_action = (domain_sum - excluded_sum) / (4 * lag - 2)
            total_action = predictive_action + dispersion
            prediction_fields.append(prediction)
            total_action_fields.append(total_action)
            conductance_fields.append(
                1.0 / (lag * np.maximum(total_action, floor)))
            lag_fields.append(float(lag))
            family_fields.append(family)
            action_means.append(float(np.mean(predictive_action)))

    prediction = np.stack(prediction_fields, axis=-1)
    conductance = np.stack(conductance_fields, axis=-1)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": prediction,
        "total_action": np.stack(total_action_fields, axis=-1),
        "mass": mass,
        "lag": np.asarray(lag_fields, dtype=np.float64),
        "path_family": np.asarray(family_fields, dtype=np.int64),
    }, {
        "minimum_lag": 1,
        "maximum_lag": int(maximum_lag),
        "characteristic_count": int(branch_count),
        "mean_predictive_action": float(np.mean(action_means)),
        "mean_collision_population": float(np.mean(
            1.0 / np.sum(mass * mass, axis=-1))),
        "target_value_enters_interior_action": False,
        "validation_domain": "complete two-shell characteristic footprint",
        "boundary_closure": "reflection; not yet continuum-normalized",
    }


def causal_crossfit_readout_forms(
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar sections from the ancestry-excluded predictive law."""
    law, diagnostic = causal_crossfit_particle_law(observation)
    prediction = law["prediction"]
    mass = law["mass"]
    mean = np.sum(mass * prediction, axis=-1)
    order = np.argsort(prediction, axis=-1, kind="stable")
    ordered_prediction = np.take_along_axis(prediction, order, axis=-1)
    ordered_mass = np.take_along_axis(mass, order, axis=-1)
    median_index = np.argmax(
        np.cumsum(ordered_mass, axis=-1) >= 0.5, axis=-1)
    median = np.take_along_axis(
        ordered_prediction, median_index[:, None], axis=-1)[:, 0]
    maximum_branch = np.take_along_axis(
        prediction, np.argmax(mass, axis=-1)[:, None], axis=-1)[:, 0]
    return {
        "mean": mean,
        "median": median,
        "maximum_branch": maximum_branch,
    }, diagnostic


def denoise_causal_collision_transport(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One-pass W1 section of target-excluded characteristic collisions."""
    forms, diagnostic = causal_collision_readout_forms(observation)
    return np.clip(forms["median"], 0.0, 1.0), {
        "status": "one-pass target-excluded characteristic collision section",
        "theory_status": "causal collision experiment; not promoted",
        "readout": "W1 median of the full relation-scale law",
        "collision_law": diagnostic,
    }


def debiased_transport_coefficient(
    residual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Return the mass-conserving coefficient owned by excess covariance."""
    product = residual * prediction
    covariance = float(np.mean(product))
    covariance_variance = float(np.var(product, ddof=1) / residual.size)
    prediction_energy = float(np.mean(prediction * prediction))
    excess = covariance * covariance - covariance_variance
    if covariance <= 0.0 or excess <= 0.0 or prediction_energy <= 0.0:
        coefficient = 0.0
    else:
        # c / E[q^2] is the line-search minimizer.  The second factor removes
        # the covariance energy expected from finite-sample chance.  A residual
        # atom cannot transport more than its full mass in one continuation.
        coefficient = min(
            1.0,
            (covariance / prediction_energy)
            * (excess / (covariance * covariance)),
        )
    return {
        "coefficient": float(coefficient),
        "residual_prediction_covariance": covariance,
        "covariance_variance": covariance_variance,
        "excess_covariance_energy": float(max(excess, 0.0)),
        "prediction_energy": prediction_energy,
    }


def denoise_lineage_branch_transport(
    observation: np.ndarray,
    resolution: CrossPredictiveResolution = CrossPredictiveResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue the information-metric lineage section to residual equilibrium."""
    line = _validate_line(observation)
    ceiling = int(resolution.maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    initial_forms, initial_lineage = lineage_branch_readout_forms(line)
    state = initial_forms["median"]
    action = float(np.mean((line - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = line - state
        residual_forms, residual_lineage = lineage_branch_readout_forms(residual)
        prediction = residual_forms["median"]
        transport = debiased_transport_coefficient(residual, prediction)
        coefficient = transport["coefficient"]
        candidate = state + coefficient * prediction
        candidate_action = float(np.mean((line - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        accepted = bool(
            coefficient > 0.0 and candidate_action < action - numerical)
        records.append({
            "continuation": continuation,
            "accepted": accepted,
            "residual_action_before": action,
            "residual_action_after": candidate_action if accepted else action,
            **transport,
            "residual_lineage": residual_lineage,
        })
        if not accepted:
            equilibrium = True
            break
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "joint-lineage residual covariance equilibrium"
            if equilibrium
            else "lineage continuation ceiling reached; unresolved"
        ),
        "theory_status": (
            "information-metric positive-lineage experiment; not promoted"
        ),
        "initial_lineage": initial_lineage,
        "initial_readout": "W1 median after bidirectional positive lineage",
        "residual_readout": "same transported W1 lineage law",
        "accepted_continuations": int(sum(
            record["accepted"] for record in records)),
        "continuation_attempts": len(records),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "numerical_resolution": asdict(resolution),
    }


def denoise_cross_predictive_transport(
    observation: np.ndarray,
    resolution: CrossPredictiveResolution = CrossPredictiveResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the last broad-gate-passing relation transport candidate."""
    line = _validate_line(observation)
    ceiling = int(resolution.maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")

    initial_forms, initial_measure = relation_scale_readout_forms(line)
    state = initial_forms["mean"]
    return _continue_relation_state(
        line,
        state,
        initial_measure,
        resolution,
        initial_barycenter="mean",
        residual_barycenter="mean",
    )


def denoise_cross_predictive_w1_transport(
    observation: np.ndarray,
    resolution: CrossPredictiveResolution = CrossPredictiveResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Experimental W1 section and W1 residual continuation."""
    line = _validate_line(observation)
    initial_forms, initial_measure = relation_scale_readout_forms(line)
    return _continue_relation_state(
        line,
        initial_forms["median"],
        initial_measure,
        resolution,
        initial_barycenter="W1 median",
        residual_barycenter="median",
    )


def denoise_cross_predictive_mixed_transport(
    observation: np.ndarray,
    resolution: CrossPredictiveResolution = CrossPredictiveResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rejected control: W1 initial section followed by W2 residual means."""
    line = _validate_line(observation)
    initial_forms, initial_measure = relation_scale_readout_forms(line)
    return _continue_relation_state(
        line,
        initial_forms["median"],
        initial_measure,
        resolution,
        initial_barycenter="W1 median",
        residual_barycenter="mean",
    )


def _particle_covariance_update(
    residual: np.ndarray,
    prediction: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transport only branch particles with positive excess covariance."""
    r = np.asarray(residual, dtype=np.float64)
    q = np.asarray(prediction, dtype=np.float64)
    probability = np.asarray(mass, dtype=np.float64)
    if q.shape != probability.shape or q.shape[0] != r.size:
        raise ValueError("residual particle law must align with the line")
    product = r[:, None] * q
    covariance = np.mean(product, axis=0)
    covariance_variance = np.var(product, axis=0, ddof=1) / r.size
    prediction_energy = np.mean(q * q, axis=0)
    excess = np.maximum(covariance * covariance - covariance_variance, 0.0)
    coefficient = np.zeros_like(covariance)
    valid = (
        (covariance > 0.0)
        & (excess > 0.0)
        & (prediction_energy > 0.0)
    )
    coefficient[valid] = np.minimum(
        1.0,
        (covariance[valid] / prediction_energy[valid])
        * (excess[valid] / (covariance[valid] * covariance[valid])),
    )
    update = np.sum(probability * coefficient[None, :] * q, axis=1)
    update_energy = float(np.mean(update * update))
    projection = float(np.mean(r * update))
    global_step = (
        float(np.clip(projection / update_energy, 0.0, 1.0))
        if update_energy > 0.0 else 0.0
    )
    return global_step * update, {
        "positive_particle_fraction": float(np.mean(coefficient > 0.0)),
        "mean_particle_coefficient": float(np.mean(coefficient)),
        "maximum_particle_coefficient": float(np.max(coefficient)),
        "mean_particle_covariance": float(np.mean(covariance)),
        "mean_particle_covariance_variance": float(np.mean(
            covariance_variance)),
        "global_descent_step": global_step,
        "raw_update_energy": update_energy,
        "raw_update_projection": projection,
    }


def denoise_cross_predictive_particle_transport(
    observation: np.ndarray,
    resolution: CrossPredictiveResolution = CrossPredictiveResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue a W1 section with covariance carried by branch particles."""
    line = _validate_line(observation)
    ceiling = int(resolution.maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    initial_forms, initial_measure = relation_scale_readout_forms(line)
    state = initial_forms["median"]
    action = float(np.mean((line - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = line - state
        law, law_diagnostic = relation_scale_particle_law(residual)
        update, transport = _particle_covariance_update(
            residual, law["prediction"], law["mass"])
        candidate = state + update
        candidate_action = float(np.mean((line - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        accepted = (
            transport["global_descent_step"] > 0.0
            and candidate_action < action - numerical
        )
        records.append({
            "continuation": continuation,
            "accepted": accepted,
            "residual_action_before": action,
            "residual_action_after": candidate_action if accepted else action,
            **transport,
            "particle_law": law_diagnostic,
        })
        if not accepted:
            equilibrium = True
            break
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "branch-particle covariance equilibrium"
            if equilibrium else "particle continuation ceiling reached; unresolved"
        ),
        "theory_status": (
            "W1 initial section with residual covariance retained on lag/path particles"
        ),
        "initial_scale_measure": initial_measure,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "numerical_resolution": asdict(resolution),
    }


def _continue_relation_state(
    line: np.ndarray,
    state: np.ndarray,
    initial_measure: dict[str, Any],
    resolution: CrossPredictiveResolution,
    *,
    initial_barycenter: str,
    residual_barycenter: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue a fixed initial relation section to covariance equilibrium."""
    ceiling = int(resolution.maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    records = []
    ceiling_hit = True
    previous_action = float(np.mean((line - state) ** 2))
    for continuation in range(ceiling):
        residual = line - state
        residual_forms, scale_measure = relation_scale_readout_forms(residual)
        prediction = residual_forms[residual_barycenter]
        transport = debiased_transport_coefficient(residual, prediction)
        coefficient = transport["coefficient"]
        if coefficient == 0.0:
            ceiling_hit = False
            records.append({
                "continuation": continuation,
                "residual_action_before": previous_action,
                "residual_action_after": previous_action,
                "accepted": False,
                **transport,
                "scale_measure": scale_measure,
            })
            break
        candidate = state + coefficient * prediction
        action = float(np.mean((line - candidate) ** 2))
        # The coefficient is no greater than the quadratic line-search
        # minimizer, so this should be an identity up to roundoff.
        if action > previous_action + np.finfo(float).eps * max(previous_action, 1.0):
            raise RuntimeError("debiased transport violated action descent")
        records.append({
            "continuation": continuation,
            "residual_action_before": previous_action,
            "residual_action_after": action,
            "accepted": True,
            **transport,
            "scale_measure": scale_measure,
        })
        state = candidate
        previous_action = action

    if ceiling_hit:
        status = "numerical continuation ceiling reached; experiment unresolved"
    else:
        status = "debiased residual-transport equilibrium"
    return np.clip(state, 0.0, 1.0), {
        "status": status,
        "theory_status": "one-dimensional foundation experiment; not promoted",
        "observation_model": "unnamed residual on exact y = z + r graph",
        "initial_scale_measure": initial_measure,
        "initial_barycenter": initial_barycenter,
        "residual_barycenter": residual_barycenter,
        "accepted_continuations": int(sum(
            bool(record["accepted"]) for record in records)),
        "continuation_attempts": len(records),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": previous_action,
        "continuations": records,
        "numerical_resolution": asdict(resolution),
        "unresolved": [
            "scale conductance still contains local validation leakage",
            "midpoint state carries values but not explicit higher jets",
            "reflecting boundary relation mass is not continuum-normalized",
        ],
    }
