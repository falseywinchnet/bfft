"""One shared-latent descent for any trusted spatial exposure fields.

Estimation may supply rotation, affine, or dense displacement evidence, but it
does not select the reconstruction law. Every observation is pulled through
its barycentric map and every centered positive exposure enters one normalized
forward/adjoint descent on the same latent image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curvilinear import residual_discrepancy
from .spatial_transport import (
    CompactGlobalExposureField,
    CompactGlobalExposureOperatorBatch,
    CompactGlobalReflectedExposureOperator,
    CovarianceExposureField,
    CovarianceExposureOperatorBatch,
    CovarianceReflectedExposureOperator,
    SpatialExposureField,
    SpatialReflectedExposureOperator,
    pullback_barycentric_coordinates,
    pullback_barycentric_values,
    pullback_compact_global_values,
    pullback_covariance_coordinates,
)


@dataclass(frozen=True)
class SpatialFieldConsensusSolution:
    image: np.ndarray
    predicted_transport_gauge_observations: np.ndarray
    uncertainty: np.ndarray
    diagnostics: dict[str, object]


def solve_spatial_field_consensus(
    observations: list[np.ndarray] | tuple[np.ndarray, ...],
    fields: (
        list[
            SpatialExposureField
            | CompactGlobalExposureField
            | CovarianceExposureField]
        | tuple[
            SpatialExposureField
            | CompactGlobalExposureField
            | CovarianceExposureField, ...]
    ),
    *,
    frame_weights: np.ndarray | None = None,
    geometric_uncertainty: np.ndarray | None = None,
    passes: int = 64,
    ratio_limit: float = 4.0,
    discrepancy_ratio: float = 1.1,
    descent_method: str = "multiplicative",
) -> SpatialFieldConsensusSolution:
    """Transport all fields into one latent state without a family branch."""
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    exposure_fields = tuple(fields)
    if len(images) < 2 or len(images) != len(exposure_fields):
        raise ValueError("spatial consensus needs matching observation and field lists")
    if any(item.shape != images[0].shape for item in images[1:]):
        raise ValueError("spatial consensus observations must share one raster")
    if any(field.shape != images[0].shape[:2] for field in exposure_fields):
        raise ValueError("spatial consensus fields must match the observations")
    fold_fractions = np.asarray([
        field.diagnostics()["fold_fraction"] for field in exposure_fields
    ], dtype=np.float64)
    folded_geometry = bool(np.any(fold_fractions > 0.0))

    spatial_shape = images[0].shape[:2]
    spatial_precision = False
    if frame_weights is None:
        sensor_weights = np.ones(
            (len(images), *spatial_shape), dtype=np.float64)
        scalar_weights = np.ones(len(images), dtype=np.float64)
    else:
        supplied_weights = np.asarray(frame_weights, dtype=np.float64)
        if supplied_weights.shape == (len(images),):
            scalar_weights = supplied_weights.copy()
            sensor_weights = np.broadcast_to(
                scalar_weights[:, None, None],
                (len(images), *spatial_shape),
            ).copy()
        elif supplied_weights.shape == (len(images), *spatial_shape):
            spatial_precision = True
            scalar_weights = np.mean(supplied_weights, axis=(1, 2))
            sensor_weights = supplied_weights.copy()
        else:
            raise ValueError(
                "frame weights must be N scalars or an NxHxW positive measure")
        if np.any(~np.isfinite(sensor_weights)) or np.any(sensor_weights < 0.0):
            raise ValueError("frame weights must be finite and non-negative")
    if not np.any(sensor_weights > 0.0):
        raise ValueError("spatial consensus needs positive frame weight")
    if descent_method not in ("multiplicative", "optimal_positive_line"):
        raise ValueError("unknown spatial consensus descent method")

    pulled: list[np.ndarray] = []
    operators: list[
        SpatialReflectedExposureOperator
        | CompactGlobalReflectedExposureOperator
        | CovarianceReflectedExposureOperator
    ] = []
    pullback_records: list[dict[str, object]] = []
    pulled_weights: list[np.ndarray] = []
    for image, field, sensor_weight in zip(
        images, exposure_fields, sensor_weights
    ):
        if folded_geometry:
            if isinstance(field, CovarianceExposureField):
                raise ValueError(
                    "folded covariance flow needs direct generated-flow support")
            pulled.append(image)
            pulled_weights.append(np.maximum(sensor_weight, 0.0))
            operators.append(
                CompactGlobalReflectedExposureOperator(field)
                if isinstance(field, CompactGlobalExposureField)
                else SpatialReflectedExposureOperator(field))
            pullback_records.append({
                "method": "direct_joint_operator_no_single_valued_pullback",
                "fold_fraction": field.diagnostics()["fold_fraction"],
            })
        else:
            if isinstance(field, CompactGlobalExposureField):
                observation, record = pullback_compact_global_values(
                    image, field)
                operator = CompactGlobalReflectedExposureOperator(
                    field.centered_field())
                precision_field = field
            elif isinstance(field, CovarianceExposureField):
                observation, centered_field, record = (
                    pullback_covariance_coordinates(image, field))
                operator = CovarianceReflectedExposureOperator(
                    centered_field.covariance_components,
                    centered_field.axis_side_weights,
                )
                precision_field = field.barycentric_field()
            else:
                observation, centered_field, record = (
                    pullback_barycentric_coordinates(image, field))
                operator = SpatialReflectedExposureOperator(centered_field)
                precision_field = field
            if spatial_precision:
                if isinstance(
                    precision_field, CompactGlobalExposureField
                ):
                    latent_weight, weight_record = (
                        pullback_compact_global_values(
                            sensor_weight, precision_field))
                else:
                    latent_weight, weight_record = pullback_barycentric_values(
                        sensor_weight, precision_field)
            else:
                latent_weight = sensor_weight.copy()
                weight_record = {
                    "method": "constant_precision_is_coordinate_invariant",
                    "iterations_used": 0,
                }
            pulled.append(observation)
            pulled_weights.append(np.maximum(latent_weight, 0.0))
            operators.append(operator)
            record = {**record, "weight_pullback": weight_record}
            pullback_records.append(record)

    stack = np.stack(pulled, axis=0)
    weight_stack = np.stack(pulled_weights, axis=0)
    support_mass = np.sum(weight_stack, axis=0)
    sensor_unsupported = support_mass <= 1e-12
    safe_weight_stack = weight_stack.copy()
    safe_weight_stack[:, sensor_unsupported] = 1.0
    ownership = safe_weight_stack / np.sum(
        safe_weight_stack, axis=0, keepdims=True)
    ownership_for_image = (
        ownership if images[0].ndim == 2 else ownership[..., None])
    channels = None if images[0].ndim == 2 else images[0].shape[2]
    operator_weight_images = [
        weight if channels is None else np.repeat(
            weight[..., None], channels, axis=2)
        for weight in ownership
    ]
    operator_batch = (
        CovarianceExposureOperatorBatch(tuple(operators))
        if all(isinstance(
            operator, CovarianceReflectedExposureOperator)
            for operator in operators)
        else CompactGlobalExposureOperatorBatch(tuple(operators))
        if all(isinstance(
            operator, CompactGlobalReflectedExposureOperator)
            for operator in operators)
        else None)

    def forward_all(values: np.ndarray) -> np.ndarray:
        if operator_batch is not None:
            return operator_batch.forward(values)
        return np.stack([
            operator.forward(value)
            for operator, value in zip(operators, values)
        ])

    def adjoint_all(values: np.ndarray) -> np.ndarray:
        if operator_batch is not None:
            return operator_batch.adjoint(values)
        return np.stack([
            operator.adjoint(value)
            for operator, value in zip(operators, values)
        ])

    normalizations = adjoint_all(np.stack(operator_weight_images))
    raw_denominator = np.sum(normalizations, axis=0)
    coverage = (
        raw_denominator if channels is None else np.mean(raw_denominator, axis=2))
    latent_coverage_stack = np.stack([
        normalization
        if channels is None else np.mean(normalization, axis=2)
        for normalization in normalizations
    ], axis=0)
    latent_coverage_mass = np.sum(latent_coverage_stack, axis=0)
    safe_latent_coverage = latent_coverage_stack.copy()
    safe_latent_coverage[:, latent_coverage_mass <= 1e-12] = 1.0
    latent_ownership = safe_latent_coverage / np.sum(
        safe_latent_coverage, axis=0, keepdims=True)
    ownership_entropy = -np.sum(
        latent_ownership * np.log(np.maximum(
            latent_ownership, np.finfo(float).tiny)),
        axis=0,
    ) / np.log(float(len(images)))
    coverage_scale = max(float(np.median(coverage)), 1e-8)
    unsupported = coverage <= 1e-6 * coverage_scale
    denominator = np.maximum(raw_denominator, 1e-8)
    if folded_geometry:
        initial_numerator = np.sum(np.stack([
            operator.adjoint(weight_image * observation)
            for operator, weight_image, observation in zip(
                operators, operator_weight_images, pulled)
        ]), axis=0)
        latent = np.clip(initial_numerator / denominator, 1e-8, 1.0)
    else:
        latent = np.clip(
            np.sum(stack * ownership_for_image, axis=0), 1e-8, 1.0)
    predictions = forward_all(np.broadcast_to(
        latent, (len(operators), *latent.shape)))
    residual_trace: list[float] = []
    discrepancy_trace: list[float] = []
    step_trace: list[float] = []
    stopped_by = "maximum_passes"
    target = max(float(discrepancy_ratio), 1.0)

    def discrepancy_value() -> float:
        ratios = np.asarray([
            residual_discrepancy(observation, prediction)[
                "total_to_read_ratio"]
            for observation, prediction in zip(pulled, predictions)
        ])
        frame_mass = np.mean(ownership, axis=(1, 2))
        return float(np.sum(frame_mass * ratios) / np.sum(frame_mass))

    initial_discrepancy = discrepancy_value()
    if initial_discrepancy <= target:
        stopped_by = "noise_discrepancy"
    limit = max(float(ratio_limit), 1.0)
    for _ in range(
        0 if stopped_by == "noise_discrepancy" else max(int(passes), 0)
    ):
        ratios = np.stack([
            np.clip(
                observation / np.maximum(prediction, 1e-8),
                1.0 / limit,
                limit,
            )
            for observation, prediction in zip(pulled, predictions)
        ])
        numerator = np.sum(adjoint_all(
            np.stack(operator_weight_images) * ratios), axis=0)
        correction = numerator / denominator
        if descent_method == "optimal_positive_line":
            direction = latent * (correction - 1.0)
            direction_predictions = forward_all(np.broadcast_to(
                direction, (len(operators), *direction.shape)))
            step_numerator = 0.0
            step_denominator = 0.0
            for weight_image, observation, prediction, direction_prediction in zip(
                operator_weight_images, pulled, predictions,
                direction_predictions,
            ):
                step_numerator += float(np.sum(
                    weight_image * (observation - prediction)
                    * direction_prediction))
                step_denominator += float(np.sum(
                    weight_image * direction_prediction * direction_prediction))
            step = max(step_numerator / max(step_denominator, 1e-20), 0.0)
            negative = direction < 0.0
            if np.any(negative):
                positive_limit = float(np.min(
                    -latent[negative] / direction[negative]))
                step = min(step, 0.999 * positive_limit)
            # The upper signal bound is a physical radiance constraint. Find
            # its corresponding exact line limit rather than clipping after a
            # step, which would leave the one-dimensional optimum.
            positive = direction > 0.0
            if np.any(positive):
                upper_limit = float(np.min(
                    (1.0 - latent[positive]) / direction[positive]))
                step = min(step, max(0.999 * upper_limit, 0.0))
            latent = latent + step * direction
            predictions = predictions + step * direction_predictions
        else:
            step = 1.0
            latent = np.clip(latent * correction, 0.0, 1.0)
            predictions = forward_all(np.broadcast_to(
                latent, (len(operators), *latent.shape)))
        step_trace.append(float(step))
        residual_trace.append(float(np.sqrt(np.mean(np.sum(
            ownership_for_image * (
                np.stack(predictions, axis=0) - stack) ** 2,
            axis=0,
        )))))
        discrepancy = discrepancy_value()
        discrepancy_trace.append(discrepancy)
        if discrepancy <= target:
            stopped_by = "noise_discrepancy"
            break
        if descent_method == "optimal_positive_line" and step <= 1e-5:
            stopped_by = "optimal_positive_line_stationarity"
            break

    branch_mean = np.sum(stack * ownership_for_image, axis=0)
    if folded_geometry:
        branch_variance = np.sum(np.stack([
            operator.adjoint(
                weight_image * (prediction - observation) ** 2)
            for weight_image, observation, prediction, operator in zip(
                operator_weight_images, pulled, predictions, operators)
        ]), axis=0) / denominator
    else:
        branch_variance = np.sum(
            ownership_for_image * (stack - branch_mean[None, ...]) ** 2,
            axis=0,
        )
    transported_residual = np.sum(adjoint_all(np.stack([
        weight_image * np.abs(prediction - observation)
        for weight_image, observation, prediction in zip(
            operator_weight_images, pulled, predictions)
    ])), axis=0) / denominator
    entropy_for_image = (
        ownership_entropy
        if latent.ndim == 2 else ownership_entropy[..., None])
    visibility_uncertainty = (
        (1.0 - entropy_for_image) * np.sqrt(np.maximum(
            branch_variance, 0.0)))
    unsupported_for_image = (
        unsupported if latent.ndim == 2 else unsupported[..., None])
    visibility_uncertainty = np.where(
        unsupported_for_image, 1.0, visibility_uncertainty)
    if geometric_uncertainty is None:
        geometry = np.zeros_like(latent)
    else:
        geometry = np.asarray(geometric_uncertainty, dtype=np.float64)
        if geometry.shape == latent.shape[:2] and latent.ndim == 3:
            geometry = np.repeat(geometry[..., None], latent.shape[2], axis=2)
        if geometry.shape != latent.shape:
            raise ValueError("geometric uncertainty must match latent shape")
    uncertainty = np.sqrt(
        branch_variance
        + transported_residual * transported_residual
        + geometry * geometry
        + visibility_uncertainty * visibility_uncertainty
    )
    terminal_discrepancy = (
        discrepancy_trace[-1] if discrepancy_trace else initial_discrepancy)
    return SpatialFieldConsensusSolution(
        image=latent,
        predicted_transport_gauge_observations=np.stack(predictions, axis=0),
        uncertainty=uncertainty,
        diagnostics={
            "method": "shared_latent_spatial_positive_exposure_transport",
            "execution_chart": (
                "direct_joint_operator"
                if folded_geometry else "invertible_barycentric_pullback"),
            "estimation_decision": (
                "joint_coverage_direct_transport_over_individual_folds"
                if folded_geometry else "invertible_barycentric_transport"),
            "field_names": [field.name for field in exposure_fields],
            "operator_backends": [operator.backend for operator in operators],
            "operator_batch_backend": (
                None if operator_batch is None else operator_batch.backend),
            "compact_global_operator_count": int(sum(
                isinstance(operator, CompactGlobalReflectedExposureOperator)
                or getattr(operator, "_scalar_coefficients", None) is not None
                for operator in operators)),
            "generated_global_fft_operator_count": int(sum(
                isinstance(operator, CompactGlobalReflectedExposureOperator)
                for operator in operators)),
            "generated_global_fft_storage_bytes": int(sum(
                operator.storage_bytes
                for operator in operators
                if isinstance(operator, CompactGlobalReflectedExposureOperator)
            )),
            "generated_global_materialized_bytes_avoided": int(sum(
                max(
                    7 * operator.field.atom_count
                    * operator.shape[0] * operator.shape[1] * 8
                    - operator.storage_bytes,
                    0,
                )
                for operator in operators
                if isinstance(operator, CompactGlobalReflectedExposureOperator)
            )),
            "compact_spatial_coefficient_bytes_avoided": int(sum(
                operator._source_indices.shape[0]
                * operator.shape[0] * operator.shape[1] * 8
                for operator in operators
                if getattr(operator, "_scalar_coefficients", None) is not None
            ) + sum(
                max(
                    7 * operator.field.atom_count
                    * operator.shape[0] * operator.shape[1] * 8
                    - operator.storage_bytes,
                    0,
                )
                for operator in operators
                if isinstance(operator, CompactGlobalReflectedExposureOperator)
            )),
            "generated_covariance_operator_count": int(sum(
                isinstance(operator, CovarianceReflectedExposureOperator)
                for operator in operators)),
            "generated_covariance_storage_bytes": int(sum(
                operator.storage_bytes
                for operator in operators
                if isinstance(operator, CovarianceReflectedExposureOperator)
            )),
            "frame_weights": (
                scalar_weights.tolist() if not spatial_precision else None),
            "spatial_precision": spatial_precision,
            "sensor_weight_mean": np.mean(
                sensor_weights, axis=(1, 2)).tolist(),
            "sensor_weight_min": np.min(
                sensor_weights, axis=(1, 2)).tolist(),
            "sensor_weight_max": np.max(
                sensor_weights, axis=(1, 2)).tolist(),
            "unsupported_visibility_fraction": float(np.mean(unsupported)),
            "joint_coverage_min": float(np.min(coverage)),
            "joint_coverage_median": float(np.median(coverage)),
            "ownership_entropy_mean": float(np.mean(ownership_entropy)),
            "ownership_entropy_min": float(np.min(ownership_entropy)),
            "ownership_entropy_max": float(np.max(ownership_entropy)),
            "ownership_role": (
                "positive_latent_measure_induced_by_adjoint_joint_coverage"),
            "fold_fractions": fold_fractions.tolist(),
            "pullback_records": pullback_records,
            "passes_used": len(residual_trace),
            "stopped_by": stopped_by,
            "initial_discrepancy_ratio": initial_discrepancy,
            "terminal_discrepancy_ratio": terminal_discrepancy,
            "residual_trace": residual_trace,
            "descent_method": descent_method,
            "optimal_step_trace": step_trace,
            "descent_role": (
                "exact_weighted_quadratic_line_on_positive_multiplicative_"
                "transport_direction"
                if descent_method == "optimal_positive_line" else
                "normalized_positive_multiplicative_transport"),
            "uncertainty_rms": float(np.sqrt(np.mean(
                uncertainty * uncertainty))),
            "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
            "uncertainty_role": (
                "positive_visibility_ownership_plus_cross_observation_plus_"
                "residual_plus_supplied_geometry_"
                "not_calibrated_interval"
            ),
        },
    )
