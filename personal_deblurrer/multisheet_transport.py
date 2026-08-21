"""Permutation-symmetric positive transport of distinct latent appearances.

The single-sheet operator cannot explain two appearances crossing the same
sensor pixel under different motion.  This module extends the same positive
forward/adjoint descent without selecting a blur family.  For observation i,

    y_i = sum_s pi_{i,s} A_{i,s} x_s,

where every ``A`` is a spatial positive-exposure operator and ``pi`` is a
non-negative simplex measure over motion sheets.  Sheet names and ordering
have no mathematical role.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .spatial_transport import (
    SpatialExposureOperatorBatch,
    SpatialExposureField,
    SpatialReflectedExposureOperator,
)


@dataclass(frozen=True)
class MultiSheetConsensusSolution:
    image: np.ndarray
    sheet_images: np.ndarray
    sensor_ownership: np.ndarray
    reference_ownership: np.ndarray
    predicted_observations: np.ndarray
    uncertainty: np.ndarray
    diagnostics: dict[str, object]


def _as_image_weight(weight: np.ndarray, ndim: int) -> np.ndarray:
    return weight if ndim == 2 else weight[..., None]


def _normalise_simplex(measure: np.ndarray, axis: int) -> np.ndarray:
    positive = np.maximum(np.asarray(measure, dtype=np.float64), 0.0)
    mass = np.sum(positive, axis=axis, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every ownership simplex needs positive mass")
    return positive / mass


def solve_multisheet_consensus(
    observations: Sequence[np.ndarray],
    sheet_fields: Sequence[Sequence[SpatialExposureField]],
    *,
    operator_plans: Sequence[
        Sequence[SpatialReflectedExposureOperator]
    ] | None = None,
    sensor_ownership: np.ndarray | None = None,
    reference_ownership: np.ndarray | None = None,
    frame_weights: np.ndarray | None = None,
    appearance_coupling: float = 0.0,
    passes: int = 64,
    ratio_limit: float = 4.0,
) -> MultiSheetConsensusSolution:
    """Recover distinct appearances under a positive motion-sheet measure.

    ``sheet_fields[i][s]`` transports latent appearance ``s`` into frame ``i``.
    ``sensor_ownership[i, s]`` is a continuous simplex measure, not a class.
    When reference ownership is omitted it is induced by matched adjoint
    coverage, so the output remains permutation invariant.
    """
    images = tuple(np.asarray(image, dtype=np.float64) for image in observations)
    if len(images) < 2:
        raise ValueError("multi-sheet consensus needs at least two observations")
    if any(image.shape != images[0].shape for image in images[1:]):
        raise ValueError("multi-sheet observations must share one raster")
    frame_fields = tuple(tuple(fields) for fields in sheet_fields)
    if len(frame_fields) != len(images) or not frame_fields[0]:
        raise ValueError("sheet fields must provide every frame and one sheet")
    sheet_count = len(frame_fields[0])
    if any(len(fields) != sheet_count for fields in frame_fields):
        raise ValueError("every frame must carry the same sheet count")
    spatial_shape = images[0].shape[:2]
    if any(
        field.shape != spatial_shape
        for fields in frame_fields for field in fields
    ):
        raise ValueError("sheet fields must match the observation raster")

    if sensor_ownership is None:
        ownership = np.full(
            (len(images), sheet_count, *spatial_shape),
            1.0 / float(sheet_count),
            dtype=np.float64,
        )
        ownership_source = "uniform_positive_simplex"
    else:
        supplied = np.asarray(sensor_ownership, dtype=np.float64)
        expected = (len(images), sheet_count, *spatial_shape)
        if supplied.shape != expected or np.any(~np.isfinite(supplied)):
            raise ValueError("sensor ownership must have shape NxSxHxW and be finite")
        ownership = _normalise_simplex(supplied, axis=1)
        ownership_source = "supplied_continuous_positive_simplex"

    if frame_weights is None:
        precision = np.ones((len(images), *spatial_shape), dtype=np.float64)
    else:
        supplied_weights = np.asarray(frame_weights, dtype=np.float64)
        if supplied_weights.shape == (len(images),):
            precision = np.broadcast_to(
                supplied_weights[:, None, None],
                (len(images), *spatial_shape),
            ).copy()
        elif supplied_weights.shape == (len(images), *spatial_shape):
            precision = supplied_weights.copy()
        else:
            raise ValueError("frame weights must be N scalars or NxHxW")
        if np.any(~np.isfinite(precision)) or np.any(precision < 0.0):
            raise ValueError("frame weights must be finite and non-negative")
    if not np.any(precision > 0.0):
        raise ValueError("multi-sheet consensus needs positive precision")

    if reference_ownership is None:
        supplied_reference_measure = None
    else:
        supplied_reference = np.asarray(reference_ownership, dtype=np.float64)
        if supplied_reference.shape != (sheet_count, *spatial_shape):
            raise ValueError("reference ownership must have shape SxHxW")
        supplied_reference_measure = _normalise_simplex(
            supplied_reference, axis=0)
    coupling = np.clip(float(appearance_coupling), 0.0, 1.0)

    if operator_plans is None:
        operators = tuple(tuple(
            SpatialReflectedExposureOperator(field) for field in fields
        ) for fields in frame_fields)
    else:
        operators = tuple(tuple(plans) for plans in operator_plans)
        if (
            len(operators) != len(images)
            or any(len(plans) != sheet_count for plans in operators)
            or any(
                operator.shape != spatial_shape
                for plans in operators for operator in plans
            )
        ):
            raise ValueError(
                "precomputed operator plans must match every frame and sheet")
    operator_batches = tuple(
        SpatialExposureOperatorBatch(frame_operators)
        for frame_operators in operators)
    image_ndim = images[0].ndim
    denominators: list[np.ndarray] = []
    appearances: list[np.ndarray] = []
    latent_coverage = []
    for sheet in range(sheet_count):
        denominator = np.zeros_like(images[0])
        numerator = np.zeros_like(images[0])
        for frame, image in enumerate(images):
            weight = precision[frame] * ownership[frame, sheet]
            image_weight = _as_image_weight(weight, image_ndim)
            denominator += operators[frame][sheet].adjoint(image_weight)
            numerator += operators[frame][sheet].adjoint(image_weight * image)
        safe_denominator = np.maximum(denominator, 1e-8)
        denominators.append(safe_denominator)
        appearances.append(np.clip(numerator / safe_denominator, 1e-8, 1.0))
        latent_coverage.append(
            denominator if image_ndim == 2 else np.mean(denominator, axis=2))

    residual_trace: list[float] = []
    limit = max(float(ratio_limit), 1.0)
    predictions: list[np.ndarray] = []
    components: list[list[np.ndarray]] = []
    for _ in range(max(int(passes), 0)):
        components = [list(operator_batches[frame].forward(
            np.stack(appearances, axis=0)
        )) for frame in range(len(images))]
        predictions = [np.sum(np.stack([
            _as_image_weight(ownership[frame, sheet], image_ndim)
            * components[frame][sheet]
            for sheet in range(sheet_count)
        ], axis=0), axis=0) for frame in range(len(images))]
        ratio = [np.clip(
            image / np.maximum(prediction, 1e-8), 1.0 / limit, limit
        ) for image, prediction in zip(images, predictions)]
        numerators = [np.zeros_like(images[0]) for _ in range(sheet_count)]
        for frame in range(len(images)):
            weighted_ratios = []
            for sheet in range(sheet_count):
                weight = precision[frame] * ownership[frame, sheet]
                weighted_ratios.append(
                    _as_image_weight(weight, image_ndim) * ratio[frame])
            transported = operator_batches[frame].adjoint(
                np.stack(weighted_ratios, axis=0))
            for sheet in range(sheet_count):
                numerators[sheet] += transported[sheet]
        for sheet in range(sheet_count):
            appearances[sheet] = np.clip(
                appearances[sheet] * numerators[sheet] / denominators[sheet],
                0.0,
                1.0,
            )
        if coupling > 0.0 and supplied_reference_measure is not None:
            appearance_stack = np.stack(appearances, axis=0)
            reference_for_image = (
                supplied_reference_measure
                if image_ndim == 2 else supplied_reference_measure[..., None])
            common = np.sum(reference_for_image * appearance_stack, axis=0)
            for sheet in range(sheet_count):
                evidence = _as_image_weight(
                    supplied_reference_measure[sheet], image_ndim)
                retention = evidence / np.maximum(
                    evidence + coupling * (1.0 - evidence), 1e-8)
                appearances[sheet] = np.clip(
                    retention * appearances[sheet]
                    + (1.0 - retention) * common,
                    0.0,
                    1.0,
                )
        residual_trace.append(float(np.sqrt(np.mean(np.stack([
            (prediction - image) ** 2
            for prediction, image in zip(predictions, images)
        ], axis=0)))))

    components = [list(operator_batches[frame].forward(
        np.stack(appearances, axis=0)
    )) for frame in range(len(images))]
    predictions = [np.sum(np.stack([
        _as_image_weight(ownership[frame, sheet], image_ndim)
        * components[frame][sheet]
        for sheet in range(sheet_count)
    ], axis=0), axis=0) for frame in range(len(images))]

    coverage_stack = np.stack(latent_coverage, axis=0)
    if supplied_reference_measure is None:
        reference = _normalise_simplex(coverage_stack, axis=0)
        reference_source = "adjoint_coverage_induced_positive_simplex"
    else:
        reference = supplied_reference_measure
        reference_source = "supplied_continuous_positive_simplex"
    sheet_stack = np.stack(appearances, axis=0)
    reference_for_image = (
        reference if image_ndim == 2 else reference[..., None])
    reconstruction = np.sum(reference_for_image * sheet_stack, axis=0)

    residual_energy = np.zeros_like(images[0])
    total_coverage = np.zeros_like(images[0])
    for frame in range(len(images)):
        frame_residual = (predictions[frame] - images[frame]) ** 2
        for sheet in range(sheet_count):
            weight = precision[frame] * ownership[frame, sheet]
            image_weight = _as_image_weight(weight, image_ndim)
            residual_energy += operators[frame][sheet].adjoint(
                image_weight * frame_residual)
            total_coverage += operators[frame][sheet].adjoint(image_weight)
    uncertainty = np.sqrt(residual_energy / np.maximum(total_coverage, 1e-8))
    spatial_coverage = (
        total_coverage if image_ndim == 2 else np.mean(total_coverage, axis=2))
    coverage_scale = max(float(np.median(spatial_coverage)), 1e-8)
    unsupported = spatial_coverage <= 1e-6 * coverage_scale
    entropy = -np.sum(
        reference * np.log(np.maximum(reference, np.finfo(float).tiny)),
        axis=0,
    ) / np.log(float(sheet_count)) if sheet_count > 1 else np.zeros(spatial_shape)
    return MultiSheetConsensusSolution(
        image=reconstruction,
        sheet_images=sheet_stack,
        sensor_ownership=ownership,
        reference_ownership=reference,
        predicted_observations=np.stack(predictions, axis=0),
        uncertainty=uncertainty,
        diagnostics={
            "method": "permutation_symmetric_positive_multisheet_transport",
            "formation": "sum_s_pi_is_A_is_x_s",
            "sheet_count": sheet_count,
            "operator_batch_backends": [
                batch.backend for batch in operator_batches],
            "ownership_source": ownership_source,
            "reference_ownership_source": reference_source,
            "passes_used": max(int(passes), 0),
            "stopped_by": "maximum_passes",
            "appearance_coupling": coupling,
            "appearance_coupling_role": (
                "unsupported_sheet_appearance_returns_to_common_latent_gauge"),
            "residual_trace": residual_trace,
            "terminal_residual_rms": float(np.sqrt(np.mean(np.stack([
                (prediction - image) ** 2
                for prediction, image in zip(predictions, images)
            ], axis=0)))),
            "reference_ownership_entropy_mean": float(np.mean(entropy)),
            "reference_ownership_entropy_min": float(np.min(entropy)),
            "minimum_latent_sheet_coverage": float(np.min(coverage_stack)),
            "unsupported_visibility_fraction": float(np.mean(unsupported)),
            "uncertainty_rms": float(np.sqrt(np.mean(uncertainty ** 2))),
            "permutation_role": (
                "sheet_indices_are_exchangeable_without_argmax_routing"),
        },
    )
