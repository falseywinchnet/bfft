"""Full symmetric fourth-cumulant transport by positive directional mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .quartic_shape_transport import _pooled_relative_log_magnitudes


@dataclass(frozen=True)
class FullQuarticTransport:
    standardized_cumulants: np.ndarray
    raw_standardized_cumulants: np.ndarray
    dictionary_weights: np.ndarray
    residual_displacements: tuple[np.ndarray, ...]
    residual_weights: tuple[np.ndarray, ...]
    authority: float
    diagnostics: dict[str, object]


def _sigma_measure(
    side_weight: float,
    angle_radians: float,
    orthogonal_side_weight: float = 1.0 / 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    first_weight = float(side_weight)
    second_weight = float(orthogonal_side_weight)
    first_mass = np.asarray((
        first_weight, 1.0 - 2.0 * first_weight, first_weight))
    second_mass = np.asarray((
        second_weight, 1.0 - 2.0 * second_weight, second_weight))
    first_extent = np.sqrt(1.0 / (2.0 * first_weight))
    second_extent = np.sqrt(1.0 / (2.0 * second_weight))
    coordinates = (-1.0, 0.0, 1.0)
    points = np.asarray([
        (first * first_extent, second * second_extent)
        for first in coordinates for second in coordinates
    ], dtype=np.float64)
    mass = np.asarray([
        first_mass[first] * second_mass[second]
        for first in range(3) for second in range(3)
    ], dtype=np.float64)
    cosine = np.cos(float(angle_radians))
    sine = np.sin(float(angle_radians))
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    return points @ rotation.T, mass


def _standardized_cumulant(
    points: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return np.asarray((
        np.sum(weights * x ** 4) - 3.0,
        np.sum(weights * x ** 3 * y),
        np.sum(weights * x ** 2 * y ** 2) - 1.0,
        np.sum(weights * x * y ** 3),
        np.sum(weights * y ** 4) - 3.0,
    ))


def directional_quartic_dictionary(
    direction_count: int = 8,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], np.ndarray, list[str]]:
    """Return positive identity-covariance measures and their full K4 tensors."""
    measures = [_sigma_measure(1.0 / 6.0, 0.0)]
    labels = ["gaussian_matched_zero_cumulant"]
    angles = np.linspace(
        0.0, np.pi, max(int(direction_count), 4), endpoint=False)
    for side_weight, role in ((0.34, "bounded_axis"), (0.085, "tailed_axis")):
        for angle in angles:
            measures.append(_sigma_measure(side_weight, float(angle)))
            labels.append(f"{role}_angle_{float(np.rad2deg(angle)):.3f}")
    tensors = np.stack([
        _standardized_cumulant(points, weights)
        for points, weights in measures
    ])
    return tuple(measures), tensors, labels


def _covariance_square_root(covariance: np.ndarray) -> np.ndarray:
    """Return the canonical principal-axis factor ``A`` with ``A A.T = C``.

    The factor deliberately is not the symmetric matrix square root.  At zero
    quartic authority, applying the dictionary's nine-point baseline through
    this factor is then exactly the established positive covariance measure,
    including its otherwise-unmodelled sixth and higher moments.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]


def _baseline_relative_transfer(
    pooled_x: np.ndarray,
    pooled_y: np.ndarray,
    covariances: np.ndarray,
    transfer_floor: float = 0.015,
) -> np.ndarray:
    records = []
    for covariance in covariances:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        log_transfer = np.zeros_like(pooled_x)
        for axis in range(2):
            extent = np.sqrt(3.0 * max(eigenvalues[axis], 0.0))
            projection = (
                pooled_x * eigenvectors[0, axis]
                + pooled_y * eigenvectors[1, axis])
            factor = (
                2.0 / 3.0
                + 1.0 / 3.0 * np.cos(
                    2.0 * np.pi * extent * projection))
            log_transfer += 0.5 * np.log(
                factor * factor + transfer_floor ** 2)
        records.append(log_transfer)
    stack = np.stack(records)
    return stack - np.mean(stack, axis=0, keepdims=True)


def _project_positive_dictionary(
    target: np.ndarray,
    dictionary_tensors: np.ndarray,
    regularization: float,
    prior_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    count = len(dictionary_tensors)
    baseline = np.zeros(count, dtype=np.float64)
    baseline[0] = 1.0
    prior = (
        baseline if prior_weight is None
        else np.asarray(prior_weight, dtype=np.float64))
    if prior.shape != (count,) or np.any(prior < 0.0):
        raise ValueError("dictionary prior must be a positive simplex weight")
    prior = prior / max(float(np.sum(prior)), np.finfo(float).tiny)
    multiplicity = np.sqrt(np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)))

    def objective(weight: np.ndarray) -> float:
        residual = (weight @ dictionary_tensors - target) * multiplicity
        return float(
            residual @ residual
            + float(regularization) * np.sum((weight - prior) ** 2))

    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weight: np.sum(weight) - 1.0},
        options={"maxiter": 300, "ftol": 1e-12},
    )
    weight = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    weight /= max(float(np.sum(weight)), np.finfo(float).tiny)
    projected = weight @ dictionary_tensors
    residual = float(np.sqrt(np.sum(
        multiplicity ** 2 * (projected - target) ** 2)))
    return weight, residual, bool(result.success)


def _project_positive_dictionary_joint(
    targets: np.ndarray,
    dictionary_tensors: np.ndarray,
    regularization: float,
    prior_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Realize relative K4 tensors with one shared positive common gauge."""
    target = np.asarray(targets, dtype=np.float64)
    capture_count = len(target)
    component_count = len(dictionary_tensors)
    baseline = np.zeros(component_count, dtype=np.float64)
    baseline[0] = 1.0
    prior = (
        baseline if prior_weight is None
        else np.asarray(prior_weight, dtype=np.float64))
    if prior.shape != (component_count,) or np.any(prior < 0.0):
        raise ValueError("joint dictionary prior must be a positive simplex")
    prior = prior / max(float(np.sum(prior)), np.finfo(float).tiny)
    multiplicity = np.sqrt(np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)))
    multiplicity_squared = multiplicity * multiplicity
    initial = np.stack([
        _project_positive_dictionary(
            item + prior @ dictionary_tensors,
            dictionary_tensors,
            regularization,
            prior,
        )[0]
        for item in target
    ])

    def objective(flat_weight: np.ndarray) -> float:
        weight = flat_weight.reshape(capture_count, component_count)
        realized = weight @ dictionary_tensors
        relative = realized - np.mean(realized, axis=0, keepdims=True)
        residual = (relative - target) * multiplicity[None, :]
        return float(
            np.sum(residual * residual)
            + float(regularization) * np.sum((weight - prior[None, :]) ** 2))

    def gradient(flat_weight: np.ndarray) -> np.ndarray:
        weight = flat_weight.reshape(capture_count, component_count)
        realized = weight @ dictionary_tensors
        relative = realized - np.mean(realized, axis=0, keepdims=True)
        residual = relative - target
        derivative = (
            2.0 * (residual * multiplicity_squared[None, :])
            @ dictionary_tensors.T
            + 2.0 * float(regularization)
            * (weight - prior[None, :])
        )
        return derivative.ravel()

    simplex_jacobian = np.zeros(
        (capture_count, capture_count * component_count), dtype=np.float64)
    for capture in range(capture_count):
        simplex_jacobian[
            capture,
            capture * component_count:(capture + 1) * component_count,
        ] = 1.0
    constraints = {
        "type": "eq",
        "fun": lambda flat_weight: np.sum(flat_weight.reshape(
            capture_count, component_count), axis=1) - 1.0,
        "jac": lambda flat_weight: simplex_jacobian,
    }
    result = minimize(
        objective,
        initial.ravel(),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (capture_count * component_count),
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    weight = np.maximum(
        np.asarray(result.x, dtype=np.float64).reshape(
            capture_count, component_count),
        0.0,
    )
    weight /= np.maximum(
        np.sum(weight, axis=1, keepdims=True), np.finfo(float).tiny)
    realized = weight @ dictionary_tensors
    common_gauge = np.mean(realized, axis=0)
    relative = realized - common_gauge[None, :]
    residual = np.sqrt(np.sum(
        multiplicity[None, :] ** 2 * (relative - target) ** 2,
        axis=1,
    ))
    return weight, residual, common_gauge, bool(result.success)


def _dictionary_relative_transfer(
    dictionary_weights: np.ndarray,
    dictionary: tuple[tuple[np.ndarray, np.ndarray], ...],
    covariances: np.ndarray,
    frequency_x: np.ndarray,
    frequency_y: np.ndarray,
    *,
    transfer_floor: float = 0.015,
) -> np.ndarray:
    """Evaluate the exact relative characteristic-function magnitude."""
    capture_count = len(dictionary_weights)
    frequency_count = len(frequency_x)
    log_transfer = np.empty((capture_count, frequency_count), dtype=np.float64)
    for capture in range(capture_count):
        factor = _covariance_square_root(covariances[capture])
        transformed_x = (
            factor[0, 0] * frequency_x + factor[1, 0] * frequency_y)
        transformed_y = (
            factor[0, 1] * frequency_x + factor[1, 1] * frequency_y)
        component_transfer = []
        for points, weights in dictionary:
            phase = 2.0 * np.pi * (
                points[:, 0, None] * transformed_x[None, :]
                + points[:, 1, None] * transformed_y[None, :])
            component_transfer.append(np.sum(
                weights[:, None] * np.cos(phase), axis=0))
        transfer = dictionary_weights[capture] @ np.stack(component_transfer)
        log_transfer[capture] = 0.5 * np.log(
            transfer * transfer + transfer_floor ** 2)
    return log_transfer - np.mean(log_transfer, axis=0, keepdims=True)


def _common_gauge_prior_mass(
    dictionary_tensors: np.ndarray,
    baseline_mass: float = 0.5,
) -> np.ndarray:
    """A positive zero-mean prior over baseline/bounded/tailed gauges."""
    count = len(dictionary_tensors)
    if count < 3 or (count - 1) % 2:
        raise ValueError("directional dictionary needs paired shape branches")
    direction_count = (count - 1) // 2
    bounded_scale = abs(float(dictionary_tensors[1, 0]))
    tailed_scale = abs(float(dictionary_tensors[1 + direction_count, 0]))
    bounded_fraction = tailed_scale / max(
        bounded_scale + tailed_scale, np.finfo(float).tiny)
    remaining = 1.0 - float(baseline_mass)
    prior = np.empty(count, dtype=np.float64)
    prior[0] = float(baseline_mass)
    prior[1:1 + direction_count] = (
        remaining * bounded_fraction / direction_count)
    prior[1 + direction_count:] = (
        remaining * (1.0 - bounded_fraction) / direction_count)
    return prior / np.sum(prior)


def estimate_full_quartic_transport(
    observations: Sequence[np.ndarray],
    covariances: np.ndarray,
    *,
    minimum_frequency: float = 0.02,
    maximum_frequency: float = 0.14,
    radial_bins: int = 20,
    angular_bins: int = 24,
    linear_regularization: float = 0.05,
    projection_regularization: float = 1e-4,
    direction_count: int = 8,
    tensor_signal_floor: float = 0.20,
    gauge_posterior_temperature: float = 0.05,
    evaluate_gauge_catalog: bool = False,
) -> FullQuarticTransport:
    """Fit all five K4 components and project them into positive measures."""
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    covariance = np.asarray(covariances, dtype=np.float64)
    capture_count = len(images)
    if capture_count < 3 or covariance.shape != (capture_count, 2, 2):
        raise ValueError("full quartic transport needs N>=3 covariance fields")
    pooled_x, pooled_y, observed, pool_record = (
        _pooled_relative_log_magnitudes(
            images,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            radial_bins=radial_bins,
            angular_bins=angular_bins,
        ))
    pooled_mass = np.asarray(pool_record.pop("pooled_mass"))
    crossfit_fold = np.asarray(pool_record.pop("crossfit_fold"), dtype=np.int8)
    root_mass = np.sqrt(pooled_mass / max(float(np.mean(pooled_mass)), 1e-12))
    baseline_prediction = _baseline_relative_transfer(
        pooled_x, pooled_y, covariance)
    target = observed - baseline_prediction
    baseline_residual = (target * root_mass[None, :]).ravel()
    cumulant_coefficient = (2.0 * np.pi) ** 4 / 24.0
    component_count = 5
    basis = np.empty(
        (capture_count, component_count, len(pooled_x)), dtype=np.float64)
    for capture in range(capture_count):
        root = _covariance_square_root(covariance[capture])
        transformed_x = root[0, 0] * pooled_x + root[1, 0] * pooled_y
        transformed_y = root[0, 1] * pooled_x + root[1, 1] * pooled_y
        basis[capture] = cumulant_coefficient * np.stack((
            transformed_x ** 4,
            4.0 * transformed_x ** 3 * transformed_y,
            6.0 * transformed_x ** 2 * transformed_y ** 2,
            4.0 * transformed_x * transformed_y ** 3,
            transformed_y ** 4,
        ))
    columns = capture_count * component_count
    design = np.empty((capture_count * len(pooled_x), columns))
    for capture in range(capture_count):
        for component in range(component_count):
            effect = np.zeros((capture_count, len(pooled_x)))
            effect[capture] = basis[capture, component]
            effect -= np.mean(effect, axis=0, keepdims=True)
            design[:, component_count * capture + component] = effect.ravel()
    row_weight = np.tile(root_mass, capture_count)
    weighted_design = design * row_weight[:, None]
    weighted_target = target.ravel() * row_weight
    regularizer = np.sqrt(max(float(linear_regularization), 0.0)) * np.eye(columns)
    fold_tensors = []
    fold_ranks = []
    for fold in (0, 1):
        training_cells = crossfit_fold == fold
        training_rows = np.tile(training_cells, capture_count)
        system = np.vstack((weighted_design[training_rows], regularizer))
        right = np.concatenate((
            weighted_target[training_rows], np.zeros(columns)))
        solution, _, rank, _ = np.linalg.lstsq(system, right, rcond=1e-8)
        tensor = solution.reshape(capture_count, component_count)
        tensor -= np.mean(tensor, axis=0, keepdims=True)
        fold_tensors.append(tensor)
        fold_ranks.append(int(rank))
    fold_tensors = np.stack(fold_tensors)
    raw_tensor = np.mean(fold_tensors, axis=0)
    held_out_residuals = []
    for training_fold, tensor in enumerate(fold_tensors):
        held_out_cells = crossfit_fold != training_fold
        held_out_rows = np.tile(held_out_cells, capture_count)
        predicted = design @ tensor.ravel()
        held_out_residuals.append(
            (predicted[held_out_rows] - target.ravel()[held_out_rows])
            * row_weight[held_out_rows])
    fitted_residual = np.concatenate(held_out_residuals)
    baseline_rms = float(np.sqrt(np.mean(baseline_residual ** 2)))
    fitted_rms = float(np.sqrt(np.mean(fitted_residual ** 2)))
    crossfit_authority = float(np.clip(
        1.0 - fitted_rms / max(baseline_rms, 1e-12), 0.0, 1.0))
    disagreement = 0.5 * np.abs(fold_tensors[0] - fold_tensors[1])
    coherence = raw_tensor ** 2 / (
        raw_tensor ** 2 + disagreement ** 2 + 0.10 ** 2)
    # Reflection boundaries and unmodelled sixth-order terms can create a
    # small, stable K4 fit even for an exact covariance measure.  A shared,
    # dimensionless fourth-power evidence taper suppresses that null floor
    # continuously; it neither chooses captures nor assigns a blur family.
    tensor_signal_rms = float(np.sqrt(
        np.sum(raw_tensor * raw_tensor) / capture_count))
    signal_authority = float(1.0 - np.exp(-(
        tensor_signal_rms / max(float(tensor_signal_floor), 1e-12)) ** 4))
    parameter_authority = crossfit_authority * signal_authority * coherence
    transported_tensor = parameter_authority * raw_tensor
    authority = float(np.mean(parameter_authority))
    dictionary, dictionary_tensors, dictionary_labels = (
        directional_quartic_dictionary(direction_count))
    catalog_prior_mass = _common_gauge_prior_mass(dictionary_tensors)
    branch_indices = (
        np.arange(len(dictionary_tensors), dtype=np.int64)
        if evaluate_gauge_catalog else np.asarray((0,), dtype=np.int64))
    branch_prior_mass = catalog_prior_mass[branch_indices]
    branch_prior_mass /= np.sum(branch_prior_mass)
    branch_dictionary_weights = []
    branch_projection_residuals = []
    branch_common_gauges = []
    branch_projection_success = []
    branch_exact_rms = []
    for branch in branch_indices:
        gauge_prior = np.zeros(len(dictionary_tensors), dtype=np.float64)
        gauge_prior[branch] = 1.0
        branch_weight, branch_residual, branch_gauge, branch_success = (
            _project_positive_dictionary_joint(
                transported_tensor,
                dictionary_tensors,
                projection_regularization,
                gauge_prior,
            ))
        exact_prediction = _dictionary_relative_transfer(
            branch_weight,
            dictionary,
            covariance,
            pooled_x,
            pooled_y,
        )
        exact_residual = (
            (exact_prediction - observed) * root_mass[None, :])
        branch_dictionary_weights.append(branch_weight)
        branch_projection_residuals.append(branch_residual)
        branch_common_gauges.append(branch_gauge)
        branch_projection_success.append(branch_success)
        branch_exact_rms.append(float(np.sqrt(np.mean(
            exact_residual * exact_residual))))
    branch_dictionary_weights = np.stack(branch_dictionary_weights)
    branch_projection_residuals = np.stack(branch_projection_residuals)
    branch_common_gauges = np.stack(branch_common_gauges)
    branch_exact_rms_array = np.asarray(branch_exact_rms)
    normalized_branch_energy = (
        branch_exact_rms_array / max(baseline_rms, 1e-12)) ** 2
    log_posterior = (
        np.log(np.maximum(branch_prior_mass, np.finfo(float).tiny))
        - 0.5 * normalized_branch_energy
        / max(float(gauge_posterior_temperature), 1e-8)
    )
    log_posterior -= np.max(log_posterior)
    raw_branch_posterior = np.exp(log_posterior)
    raw_branch_posterior /= np.sum(raw_branch_posterior)
    gauge_closure_improvement = float(np.clip(
        1.0 - np.min(branch_exact_rms_array)
        / max(branch_exact_rms_array[0], 1e-12),
        0.0,
        1.0,
    ))
    gauge_excursion_authority = float(
        authority * gauge_closure_improvement)
    branch_posterior = gauge_excursion_authority * raw_branch_posterior
    branch_posterior[int(np.flatnonzero(branch_indices == 0)[0])] += (
        1.0 - gauge_excursion_authority)
    dictionary_weight_array = np.einsum(
        "b,bnk->nk", branch_posterior, branch_dictionary_weights)
    common_shape_gauge = np.einsum(
        "b,bk->k", branch_posterior, branch_common_gauges)
    realized_tensor = dictionary_weight_array @ dictionary_tensors
    realized_relative = realized_tensor - np.mean(
        realized_tensor, axis=0, keepdims=True)
    multiplicity = np.sqrt(np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)))
    projection_residual_array = np.sqrt(np.sum(
        ((realized_relative - transported_tensor)
         * multiplicity[None, :]) ** 2,
        axis=1,
    ))
    joint_projection_success = bool(all(branch_projection_success))
    measures = []
    for capture in range(capture_count):
        mixture = dictionary_weight_array[capture]
        root = _covariance_square_root(covariance[capture])
        points = []
        weights = []
        for component_weight, (standard_points, standard_weights) in zip(
            mixture, dictionary
        ):
            if component_weight <= 1e-8:
                continue
            points.append(standard_points @ root.T)
            weights.append(component_weight * standard_weights)
        point_array = np.concatenate(points)
        weight_array = np.concatenate(weights)
        weight_array /= max(float(np.sum(weight_array)), np.finfo(float).tiny)
        measures.append((point_array, weight_array))
    return FullQuarticTransport(
        standardized_cumulants=transported_tensor,
        raw_standardized_cumulants=raw_tensor,
        dictionary_weights=dictionary_weight_array,
        residual_displacements=tuple(item[0] for item in measures),
        residual_weights=tuple(item[1] for item in measures),
        authority=authority,
        diagnostics={
            "method": (
                "crossfit_full_symmetric_quartic_positive_directional_transport"),
            "tensor_component_order": [
                "xxxx", "xxxy", "xxyy", "xyyy", "yyyy"],
            "shape_gauge": (
                "exact_zero_mean_standardized_full_fourth_cumulant"),
            "standardized_cumulants": transported_tensor.tolist(),
            "raw_standardized_cumulants": raw_tensor.tolist(),
            "fold_tensor_disagreement": disagreement.tolist(),
            "parameter_authority": parameter_authority.tolist(),
            "crossfit_predictive_authority": crossfit_authority,
            "tensor_signal_rms": tensor_signal_rms,
            "tensor_signal_floor": float(tensor_signal_floor),
            "tensor_signal_authority": signal_authority,
            "shape_authority": authority,
            "baseline_relative_log_magnitude_rms": baseline_rms,
            "fitted_relative_log_magnitude_rms": fitted_rms,
            "linear_fold_ranks": fold_ranks,
            "dictionary_labels": dictionary_labels,
            "dictionary_weights": dictionary_weight_array.tolist(),
            "projection_method": (
                "positive_common_gauge_posterior_exact_transfer_closure"),
            "projection_residuals": projection_residual_array.tolist(),
            "projection_success": joint_projection_success,
            "common_standardized_cumulant_gauge": common_shape_gauge.tolist(),
            "gauge_catalog_evaluated": bool(evaluate_gauge_catalog),
            "gauge_branch_dictionary_indices": branch_indices.tolist(),
            "gauge_branch_labels": [
                dictionary_labels[index] for index in branch_indices],
            "gauge_branch_prior_mass": branch_prior_mass.tolist(),
            "gauge_branch_raw_posterior_mass": raw_branch_posterior.tolist(),
            "gauge_branch_posterior_mass": branch_posterior.tolist(),
            "gauge_branch_exact_transfer_rms": branch_exact_rms,
            "gauge_branch_projection_residual_rms": [
                float(np.sqrt(np.mean(item * item)))
                for item in branch_projection_residuals],
            "gauge_branch_projection_success": branch_projection_success,
            "gauge_branch_dictionary_weights": (
                branch_dictionary_weights.tolist()),
            "gauge_posterior_temperature": float(
                gauge_posterior_temperature),
            "gauge_closure_improvement": gauge_closure_improvement,
            "gauge_excursion_authority": gauge_excursion_authority,
            "gauge_posterior_entropy": float(-np.sum(
                branch_posterior * np.log(np.maximum(
                    branch_posterior, np.finfo(float).tiny)))),
            "gauge_posterior_effective_branch_count": float(
                1.0 / np.sum(branch_posterior * branch_posterior)),
            "active_dictionary_components": [
                int(np.count_nonzero(item > 1e-8))
                for item in dictionary_weight_array],
            "maximum_atom_count": max(len(item[0]) for item in measures),
            **pool_record,
            "capture_role": (
                "all_directional_measures_have_positive_mass_no_shape_selection"),
        },
    )
