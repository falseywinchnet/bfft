"""Multi-observation consensus for continuous rotational exposure fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .spatial_consensus import solve_spatial_field_consensus
from .spatial_transport import (
    SpatialExposureField,
    rotational_exposure,
)


def _luminance(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[2] >= 3:
        return (
            0.2126 * value[..., 0]
            + 0.7152 * value[..., 1]
            + 0.0722 * value[..., 2]
        )
    raise ValueError("rotation consensus images must be HxW or RGB")


@dataclass(frozen=True)
class RotationPairEvidence:
    first_index: int
    second_index: int
    relative_angle_degrees: float
    inverse_consistency_degrees: float
    robust_loss: float
    curvature: float
    confidence: float


@dataclass(frozen=True)
class RotationConsensusEstimate:
    relative_mean_angles_degrees: np.ndarray
    exposure_extents_degrees: np.ndarray
    angle_standard_deviation_degrees: np.ndarray
    fields: tuple[SpatialExposureField, ...]
    pair_evidence: tuple[RotationPairEvidence, ...]
    reference_index: int
    cycle_rms_degrees: float
    confidence: float
    relative_motion_observable: bool
    common_rotation_gauge_unidentifiable: bool


@dataclass(frozen=True)
class SpatialConsensusResult:
    image: np.ndarray
    uncertainty: np.ndarray
    estimate: RotationConsensusEstimate
    diagnostics: dict[str, object]


def _rotation_registration_objective(
    reference: np.ndarray,
    moving: np.ndarray,
):
    from scipy.ndimage import gaussian_filter, rotate

    first = gaussian_filter(_luminance(reference), 0.8, mode="reflect")
    second = gaussian_filter(_luminance(moving), 0.8, mode="reflect")
    if first.shape != second.shape:
        raise ValueError("rotation consensus observations must share one shape")
    margin = max(4, int(round(0.12 * min(first.shape))))
    crop = (slice(margin, -margin), slice(margin, -margin))
    target = first[crop]
    target = (target - np.mean(target)) / max(float(np.std(target)), 1e-8)

    def objective(angle_degrees: float) -> float:
        candidate = rotate(
            second,
            float(angle_degrees),
            reshape=False,
            order=1,
            mode="reflect",
            prefilter=False,
        )[crop]
        candidate = (
            (candidate - np.mean(candidate))
            / max(float(np.std(candidate)), 1e-8)
        )
        difference = candidate - target
        scale = 0.02
        return float(np.mean(
            np.sqrt(difference * difference + scale * scale) - scale))

    return objective


def _estimate_pair_rotation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_index: int,
    second_index: int,
    maximum_angle_degrees: float,
) -> RotationPairEvidence:
    from scipy.optimize import minimize_scalar

    bound = max(float(maximum_angle_degrees), 0.25)

    def solve(reference: np.ndarray, moving: np.ndarray):
        objective = _rotation_registration_objective(reference, moving)
        result = minimize_scalar(
            objective,
            bounds=(-bound, bound),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 96},
        )
        angle = float(result.x)
        loss = float(result.fun)
        step = 0.05
        curvature = max(
            (objective(angle - step) + objective(angle + step) - 2.0 * loss)
            / (step * step),
            0.0,
        )
        return angle, loss, curvature

    forward, forward_loss, forward_curvature = solve(first, second)
    reverse, reverse_loss, reverse_curvature = solve(second, first)
    relative = 0.5 * (forward - reverse)
    inconsistency = abs(forward + reverse)
    loss = 0.5 * (forward_loss + reverse_loss)
    curvature = 0.5 * (forward_curvature + reverse_curvature)
    confidence = (
        (curvature + 1e-4)
        / ((loss + 1e-4) * (1.0 + inconsistency / 0.05))
    )
    return RotationPairEvidence(
        first_index=int(first_index),
        second_index=int(second_index),
        relative_angle_degrees=float(relative),
        inverse_consistency_degrees=float(inconsistency),
        robust_loss=float(loss),
        curvature=float(curvature),
        confidence=float(confidence),
    )


def estimate_rotation_consensus(
    observations: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    reference_index: int | None = None,
    maximum_angle_degrees: float = 12.0,
    duty_cycle: float = 1.0,
) -> RotationConsensusEstimate:
    """Fit one cycle-consistent continuous camera-rotation trajectory."""
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    if len(images) < 2:
        raise ValueError("rotation consensus needs at least two observations")
    if any(item.shape != images[0].shape for item in images[1:]):
        raise ValueError("rotation consensus observations must share one raster")
    count = len(images)
    reference = count // 2 if reference_index is None else int(reference_index)
    if not 0 <= reference < count:
        raise ValueError("rotation consensus reference index is out of range")
    evidence = tuple(
        _estimate_pair_rotation(
            images[first],
            images[second],
            first_index=first,
            second_index=second,
            maximum_angle_degrees=maximum_angle_degrees,
        )
        for first in range(count)
        for second in range(first + 1, count)
    )
    raw_confidence = np.asarray(
        [item.confidence for item in evidence], dtype=np.float64)
    confidence_scale = max(float(np.median(raw_confidence)), 1e-12)
    edge_weight = np.clip(raw_confidence / confidence_scale, 0.05, 20.0)
    matrix = np.zeros((len(evidence) + 1, count), dtype=np.float64)
    target = np.zeros(len(evidence) + 1, dtype=np.float64)
    weights = np.ones(len(evidence) + 1, dtype=np.float64)
    for row, item in enumerate(evidence):
        # Registration rotates observation j into i; therefore theta_j-theta_i.
        matrix[row, item.first_index] = -1.0
        matrix[row, item.second_index] = 1.0
        target[row] = item.relative_angle_degrees
        weights[row] = edge_weight[row]
    matrix[-1, reference] = 1.0
    weights[-1] = 1000.0
    weighted_matrix = np.sqrt(weights)[:, None] * matrix
    weighted_target = np.sqrt(weights) * target
    angles = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)[0]
    angles -= angles[reference]
    edge_residual = np.asarray([
        angles[item.second_index] - angles[item.first_index]
        - item.relative_angle_degrees
        for item in evidence
    ])
    cycle_rms = float(np.sqrt(np.mean(edge_residual * edge_residual)))

    normal = weighted_matrix.T @ weighted_matrix
    covariance = np.linalg.pinv(normal)
    measurement_scale = max(
        cycle_rms,
        float(np.median([
            item.inverse_consistency_degrees for item in evidence
        ])),
        0.005,
    )
    angle_sigma = measurement_scale * np.sqrt(np.maximum(
        np.diag(covariance), 0.0))
    angle_sigma[reference] = 0.0

    extent = np.empty(count, dtype=np.float64)
    if count == 2:
        extent[:] = abs(angles[1] - angles[0])
    else:
        extent[0] = abs(angles[1] - angles[0])
        extent[-1] = abs(angles[-1] - angles[-2])
        for index in range(1, count - 1):
            extent[index] = 0.5 * abs(angles[index + 1] - angles[index - 1])
    extent *= max(float(duty_cycle), 0.0)
    fields = tuple(
        rotational_exposure(
            images[0].shape[:2],
            mean_angle_degrees=float(angles[index]),
            exposure_degrees=float(extent[index]),
            atoms=9,
        )
        for index in range(count)
    )
    consensus_confidence = float(
        math.exp(-cycle_rms / 0.05)
        * math.exp(-float(np.mean([
            item.inverse_consistency_degrees for item in evidence
        ])) / 0.1)
    )
    relative_motion_observable = bool(
        float(np.ptp(angles)) >= 0.1 and consensus_confidence >= 0.1)
    return RotationConsensusEstimate(
        relative_mean_angles_degrees=angles,
        exposure_extents_degrees=extent,
        angle_standard_deviation_degrees=angle_sigma,
        fields=fields,
        pair_evidence=evidence,
        reference_index=reference,
        cycle_rms_degrees=cycle_rms,
        confidence=consensus_confidence,
        relative_motion_observable=relative_motion_observable,
        common_rotation_gauge_unidentifiable=True,
    )


def deblur_rotation_consensus(
    observations: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    reference_index: int | None = None,
    maximum_angle_degrees: float = 12.0,
    duty_cycle: float = 1.0,
    passes: int = 64,
    ratio_limit: float = 4.0,
    discrepancy_ratio: float = 1.1,
) -> SpatialConsensusResult:
    """Estimate rotation consensus and solve one shared latent transport state."""
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    estimate = estimate_rotation_consensus(
        images,
        reference_index=reference_index,
        maximum_angle_degrees=maximum_angle_degrees,
        duty_cycle=duty_cycle,
    )
    frame_sigma = np.asarray(
        estimate.angle_standard_deviation_degrees, dtype=np.float64)
    frame_weight = 1.0 / (1.0 + (frame_sigma / 0.05) ** 2)
    frame_weight /= np.sum(frame_weight)
    solution = solve_spatial_field_consensus(
        images,
        estimate.fields,
        frame_weights=frame_weight,
        passes=passes,
        ratio_limit=ratio_limit,
        discrepancy_ratio=discrepancy_ratio,
    )
    latent = solution.image
    height, width = latent.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt(
        (xx - 0.5 * (width - 1)) ** 2
        + (yy - 0.5 * (height - 1)) ** 2)
    angular_sigma = float(np.sum(frame_weight * frame_sigma))
    position_sigma = radius * np.deg2rad(angular_sigma)
    if latent.ndim == 2:
        gradient = np.sqrt(sum(item * item for item in np.gradient(latent)))
        geometric_uncertainty = position_sigma * gradient
    else:
        geometric_uncertainty = np.stack([
            position_sigma * np.sqrt(sum(
                item * item for item in np.gradient(latent[..., channel])))
            for channel in range(latent.shape[2])
        ], axis=2)
    uncertainty = np.sqrt(
        solution.uncertainty * solution.uncertainty
        + geometric_uncertainty * geometric_uncertainty)
    return SpatialConsensusResult(
        image=latent,
        uncertainty=uncertainty,
        estimate=estimate,
        diagnostics={
            **solution.diagnostics,
            "method": "multi_observation_rotation_flow_consensus_transport",
            "reference_index": estimate.reference_index,
            "common_rotation_gauge_unidentifiable": True,
            "relative_mean_angles_degrees": (
                estimate.relative_mean_angles_degrees.tolist()),
            "exposure_extents_degrees": (
                estimate.exposure_extents_degrees.tolist()),
            "angle_standard_deviation_degrees": (
                estimate.angle_standard_deviation_degrees.tolist()),
            "cycle_rms_degrees": estimate.cycle_rms_degrees,
            "consensus_confidence": estimate.confidence,
            "relative_motion_observable": estimate.relative_motion_observable,
            "estimation_decision": (
                "relative_rotation_trajectory_supported"
                if estimate.relative_motion_observable
                else "abstain_common_rotation_and_exposure_gauge"
            ),
            "uncertainty_rms": float(np.sqrt(np.mean(
                uncertainty * uncertainty))),
            "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
            "uncertainty_role": (
                "cross_observation_plus_residual_plus_rotation_geometry_"
                "not_calibrated_interval"
            ),
        },
    )
