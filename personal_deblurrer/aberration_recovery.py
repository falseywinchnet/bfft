"""Relative affine-aberration recovery from same-scene observations.

The estimator does not choose radial, astigmatic, shear, or ghost families.
It first recovers the existing positive local covariance atlas from pairwise
Fourier-circle quotients, then expresses that atlas in the complete quadratic
coordinate jet induced by affine observation maps.  Clean truth never enters
estimation.  A common aberration remains a declared gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .multicapture_transport import (
    MultiCaptureTransportResult,
    deblur_multicapture_consensus,
)
from .spatial_transport import CovarianceExposureField


@dataclass(frozen=True)
class AffineAberrationJet:
    coefficients: np.ndarray
    fitted_covariance_fields: np.ndarray
    fitted_relative_covariance_fields: np.ndarray
    stationary_points_xy: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class AberrationRecoveryResult:
    image: np.ndarray
    uncertainty: np.ndarray
    transport_result: MultiCaptureTransportResult
    aberration_jet: AffineAberrationJet
    diagnostics: dict[str, object]


def covariance_field_matrices(
    fields: Sequence[CovarianceExposureField],
) -> np.ndarray:
    """Expand compact xx,xy,yy fields into symmetric matrices."""
    if not fields:
        raise ValueError("aberration covariance fields cannot be empty")
    shape = fields[0].shape
    if any(field.shape != shape for field in fields):
        raise ValueError("aberration fields must share one raster")
    output = np.empty((len(fields), *shape, 2, 2), dtype=np.float64)
    for index, field in enumerate(fields):
        components = field.covariance_components
        output[index, ..., 0, 0] = components[..., 0]
        output[index, ..., 0, 1] = components[..., 1]
        output[index, ..., 1, 0] = components[..., 1]
        output[index, ..., 1, 1] = components[..., 2]
    return output


def _quadratic_basis(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    scale_x = max(0.5 * (width - 1), 1.0)
    scale_y = max(0.5 * (height - 1), 1.0)
    x = (xx - 0.5 * (width - 1)) / scale_x
    y = (yy - 0.5 * (height - 1)) / scale_y
    basis = np.stack(
        (x * x, x * y, y * y, x, y, np.ones_like(x)), axis=-1)
    return basis.reshape((-1, 6)), x, y


def _fit_components(
    design: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    selected = np.ones(len(design), dtype=bool) if mask is None else mask
    return np.linalg.lstsq(
        design[selected], values[selected], rcond=None)[0]


def fit_affine_aberration_jet(
    fields: Sequence[CovarianceExposureField],
) -> AffineAberrationJet:
    """Fit the full quadratic covariance jet with checkerboard crossfit."""
    covariance = covariance_field_matrices(fields)
    count, height, width = covariance.shape[:3]
    components = np.stack((
        covariance[..., 0, 0],
        covariance[..., 0, 1],
        covariance[..., 1, 1],
    ), axis=-1)
    design, normalized_x, normalized_y = _quadratic_basis((height, width))
    yy, xx = np.mgrid[:height, :width]
    parity = ((xx + yy) & 1).ravel().astype(bool)
    coefficients = np.empty((count, 3, 6), dtype=np.float64)
    crossfit = np.empty_like(components)
    for capture in range(count):
        for component in range(3):
            values = components[capture, ..., component].ravel()
            coefficients[capture, component] = _fit_components(
                design, values)
            first = _fit_components(design, values, ~parity)
            second = _fit_components(design, values, parity)
            prediction = np.empty_like(values)
            prediction[parity] = design[parity] @ first
            prediction[~parity] = design[~parity] @ second
            crossfit[capture, ..., component] = prediction.reshape(
                (height, width))
    fitted_components = np.einsum(
        "pc,nkc->npk", design, coefficients, optimize=True).reshape(
            (count, height, width, 3))
    fitted = np.empty_like(covariance)
    fitted[..., 0, 0] = fitted_components[..., 0]
    fitted[..., 0, 1] = fitted_components[..., 1]
    fitted[..., 1, 0] = fitted_components[..., 1]
    fitted[..., 1, 1] = fitted_components[..., 2]
    relative = fitted - np.mean(fitted, axis=0, keepdims=True)

    raw_values = components.ravel()
    crossfit_values = crossfit.ravel()
    centered = raw_values - np.mean(raw_values)
    crossfit_error = float(np.mean((crossfit_values - raw_values) ** 2))
    zero_error = float(np.mean(centered * centered))
    predictive_authority = float(np.clip(
        1.0 - crossfit_error / max(zero_error, np.finfo(float).tiny),
        0.0,
        1.0,
    ))
    fit_rms = float(np.sqrt(np.mean(
        (fitted_components - components) ** 2)))
    signal_rms = float(np.sqrt(np.mean(
        (components - np.mean(components, axis=0, keepdims=True)) ** 2)))

    stationary = np.full((count, 2), np.nan, dtype=np.float64)
    stationary_authority = np.zeros(count, dtype=np.float64)
    center_pixel = np.asarray((0.5 * (width - 1), 0.5 * (height - 1)))
    scale_pixel = np.asarray((0.5 * (width - 1), 0.5 * (height - 1)))
    for capture in range(count):
        trace_coefficients = (
            coefficients[capture, 0] + coefficients[capture, 2])
        quadratic = np.asarray((
            (trace_coefficients[0], 0.5 * trace_coefficients[1]),
            (0.5 * trace_coefficients[1], trace_coefficients[2]),
        ))
        linear = trace_coefficients[3:5]
        eigenvalues = np.linalg.eigvalsh(quadratic)
        definite = bool(np.prod(eigenvalues) > 0.0)
        conditioning = float(
            np.min(np.abs(eigenvalues))
            / max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny))
        if np.linalg.norm(quadratic) > 1e-12 and definite:
            normalized_stationary = -0.5 * np.linalg.solve(
                quadratic, linear)
            radius = float(np.max(np.abs(normalized_stationary)))
            support_authority = float(np.clip(2.0 - radius, 0.0, 1.0))
            stationary_authority[capture] = conditioning * support_authority
            if stationary_authority[capture] > 0.05:
                stationary[capture] = (
                    center_pixel + scale_pixel * normalized_stationary)

    return AffineAberrationJet(
        coefficients=coefficients,
        fitted_covariance_fields=fitted,
        fitted_relative_covariance_fields=relative,
        stationary_points_xy=stationary,
        diagnostics={
            "method": "crossfit_complete_quadratic_affine_aberration_jet",
            "basis": ["x2", "xy", "y2", "x", "y", "one"],
            "family_classification": False,
            "coefficient_count_per_covariance_component": 6,
            "crossfit_predictive_authority": predictive_authority,
            "crossfit_error_rms": float(np.sqrt(crossfit_error)),
            "full_fit_rms": fit_rms,
            "relative_atlas_signal_rms": signal_rms,
            "stationary_points_xy": stationary.tolist(),
            "stationary_point_authority": stationary_authority.tolist(),
            "stationary_point_policy": (
                "definite_trace_action_with_conditioning_and_in_raster_"
                "support_taper"),
            "common_aberration_identifiable": False,
            "identifiability_boundary": (
                "only_relative_quadratic_aberration_is_observed; common_"
                "positive_transport_remains_a_gauge"),
        },
    )


def recover_affine_aberration_multicapture(
    observations: Sequence[np.ndarray],
    *,
    passes: int = 64,
    patch_size: int = 32,
    stride: int = 16,
    quartic_shape: bool = True,
) -> AberrationRecoveryResult:
    """Estimate and invert a relative aberration atlas without clean truth."""
    transport = deblur_multicapture_consensus(
        observations,
        passes=passes,
        descent_method="optimal_positive_line",
        mixing_patch_size=patch_size,
        mixing_stride=stride,
        quartic_shape=quartic_shape,
    )
    if not all(
        isinstance(field, CovarianceExposureField)
        for field in transport.fields
    ):
        raise RuntimeError("affine aberration recovery requires a spatial atlas")
    fields = tuple(transport.fields)
    jet = fit_affine_aberration_jet(fields)
    diagnostics = {
        **transport.diagnostics,
        "method": "relative_affine_aberration_transport_recovery",
        "aberration_jet": jet.diagnostics,
        "truth_used_for_estimation": False,
        "family_classification": False,
        "common_aberration_identifiable": False,
    }
    return AberrationRecoveryResult(
        image=transport.image,
        uncertainty=transport.uncertainty,
        transport_result=transport,
        aberration_jet=jet,
        diagnostics=diagnostics,
    )
