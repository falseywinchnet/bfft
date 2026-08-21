"""Synthetic controls for blur anomalies in one observation-transport law.

Generator names exist only to construct falsifiable observations.  Every
geometric or mixing case becomes an ``AffineObservationMeasure`` and consumes
the same consolidated inverse.  Sampling anomalies instead become admissible
intervals and continuous precision; they are never fitted as displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .composed_transport import AffineObservationMeasure, ObservationBounds


@dataclass(frozen=True)
class BoundedSensorObservation:
    measured: np.ndarray
    transport_center: np.ndarray
    bounds: ObservationBounds
    transport_observation: np.ndarray
    diagnostics: dict[str, object]


def _center(shape: tuple[int, int], center_xy) -> np.ndarray:
    height, width = map(int, shape)
    value = np.asarray(
        (0.5 * (width - 1), 0.5 * (height - 1))
        if center_xy is None else center_xy,
        dtype=np.float64,
    )
    if value.shape != (2,) or np.any(~np.isfinite(value)):
        raise ValueError("affine exposure center must contain two finite values")
    return value


def translation_mixture_measure(
    source_offsets_xy: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    name: str = "translation_mixture_observation",
) -> AffineObservationMeasure:
    """Positive mixture of translated source samples ``q=p+offset``."""
    offsets = np.asarray(source_offsets_xy, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("translation offsets must have shape Kx2")
    mass = (
        np.ones(len(offsets), dtype=np.float64)
        if weights is None else np.asarray(weights, dtype=np.float64))
    return AffineObservationMeasure(
        name,
        np.broadcast_to(np.eye(2), (len(offsets), 2, 2)).copy(),
        offsets,
        mass,
    )


def ghost_measure(
    source_offset_xy: tuple[float, float],
    *,
    ghost_mass: float = 0.12,
) -> AffineObservationMeasure:
    """Identity plus one faint displaced positive copy."""
    mass = float(ghost_mass)
    if not np.isfinite(mass) or mass <= 0.0 or mass >= 1.0:
        raise ValueError("ghost mass must lie in (0,1)")
    return translation_mixture_measure(
        np.asarray(((0.0, 0.0), source_offset_xy), dtype=np.float64),
        weights=np.asarray((1.0 - mass, mass)),
        name=f"identity_plus_ghost_mass_{mass:g}",
    )


def rotation_exposure_measure(
    shape: tuple[int, int],
    *,
    exposure_degrees: float = 4.0,
    mean_degrees: float = 0.0,
    center_xy: tuple[float, float] | None = None,
) -> AffineObservationMeasure:
    """Three-node positive angular exposure around an arbitrary center."""
    center = _center(shape, center_xy)
    extent = float(exposure_degrees)
    mean = float(mean_degrees)
    if not np.isfinite(extent) or extent < 0.0 or not np.isfinite(mean):
        raise ValueError("rotation exposure parameters must be finite")
    angles = np.deg2rad(mean + np.asarray((-0.5 * extent, 0.0, 0.5 * extent)))
    cosine = np.cos(angles)
    sine = np.sin(angles)
    matrices = np.stack([
        np.asarray(((c, -s), (s, c))) for c, s in zip(cosine, sine)
    ])
    offsets = center[None, :] - np.einsum(
        "kij,j->ki", matrices, center)
    return AffineObservationMeasure(
        f"rotation_exposure_{mean:g}_{extent:g}_degrees",
        matrices,
        offsets,
        np.asarray((0.25, 0.5, 0.25)),
    )


def shear_exposure_measure(
    shape: tuple[int, int],
    *,
    fractional_extent: float = 0.04,
    center_xy: tuple[float, float] | None = None,
) -> AffineObservationMeasure:
    """Three-node x-by-y shear exposure, centered in the image chart."""
    center = _center(shape, center_xy)
    extent = float(fractional_extent)
    if not np.isfinite(extent) or extent < 0.0:
        raise ValueError("shear exposure extent must be finite and nonnegative")
    shears = np.asarray((-extent, 0.0, extent))
    matrices = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    matrices[:, 0, 1] = shears
    offsets = center[None, :] - np.einsum(
        "kij,j->ki", matrices, center)
    return AffineObservationMeasure(
        f"shear_exposure_extent_{extent:g}",
        matrices,
        offsets,
        np.asarray((0.25, 0.5, 0.25)),
    )


def astigmatic_scale_measure(
    shape: tuple[int, int],
    *,
    fractional_extent: float = 0.045,
    angle_degrees: float = 0.0,
    center_xy: tuple[float, float] | None = None,
) -> AffineObservationMeasure:
    """Positive anisotropic scale exposure in a rotated optical chart."""
    center = _center(shape, center_xy)
    extent = float(fractional_extent)
    if not np.isfinite(extent) or extent <= 0.0 or extent >= 1.0:
        raise ValueError("astigmatic extent must lie in (0,1)")
    angle = np.deg2rad(float(angle_degrees))
    rotation = np.asarray((
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    ))
    scales = (1.0 - extent, 1.0, 1.0 + extent)
    matrices = np.stack([
        rotation @ np.diag((scale, 1.0)) @ rotation.T
        for scale in scales
    ])
    offsets = center[None, :] - np.einsum(
        "kij,j->ki", matrices, center)
    return AffineObservationMeasure(
        f"astigmatic_scale_{extent:g}_angle_{angle_degrees:g}",
        matrices,
        offsets,
        np.asarray((0.25, 0.5, 0.25)),
    )


def bounded_linear_sensor_observation(
    transport_observation: np.ndarray,
    *,
    exposure_gain: float = 1.0,
    quantization_levels: int = 256,
    invalid_mask: np.ndarray | None = None,
    invalid_value: float = 0.0,
) -> BoundedSensorObservation:
    """Apply gain, clipping, quantization, and optional missing samples.

    Bounds are returned in the linear transport-output domain. Maximum-code
    samples become lower bounds, minimum-code samples become upper bounds, and
    invalid samples receive zero authority. No clipped or missing value is
    treated as ordinary equality evidence.
    """
    clean = np.clip(np.asarray(transport_observation, dtype=np.float64), 0.0, 1.0)
    gain = float(exposure_gain)
    levels = int(quantization_levels)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("exposure gain must be positive and finite")
    if levels < 2:
        raise ValueError("quantization levels must be at least two")
    scale = levels - 1
    sensor = np.clip(gain * clean, 0.0, 1.0)
    measured = np.round(sensor * scale) / scale
    half_step = 0.5 / scale
    sensor_lower = np.clip(measured - half_step, 0.0, 1.0)
    sensor_upper = np.clip(measured + half_step, 0.0, 1.0)
    lower = np.clip(sensor_lower / gain, 0.0, 1.0)
    upper = np.clip(sensor_upper / gain, 0.0, 1.0)
    maximum_code = measured >= 1.0 - np.finfo(float).eps
    minimum_code = measured <= np.finfo(float).eps
    upper[maximum_code] = 1.0
    lower[minimum_code] = 0.0
    precision = np.ones_like(clean)
    if invalid_mask is not None:
        invalid = np.asarray(invalid_mask, dtype=bool)
        if invalid.shape == clean.shape[:2] and clean.ndim == 3:
            invalid = np.broadcast_to(invalid[..., None], clean.shape)
        if invalid.shape != clean.shape:
            raise ValueError("invalid mask must match pixels or samples")
        measured = measured.copy()
        measured[invalid] = float(invalid_value)
        lower[invalid] = 0.0
        upper[invalid] = 1.0
        precision[invalid] = 0.0
    else:
        invalid = np.zeros_like(clean, dtype=bool)
    bounds = ObservationBounds(
        lower=lower,
        upper=upper,
        precision=precision,
        diagnostics={
            "method": "linear_gain_clip_quantize_interval_transport",
            "exposure_gain": gain,
            "quantization_levels": levels,
            "quantization_step": 1.0 / scale,
            "maximum_code_fraction": float(np.mean(maximum_code)),
            "minimum_code_fraction": float(np.mean(minimum_code)),
            "invalid_fraction": float(np.mean(invalid)),
            "invalid_samples_have_zero_authority": True,
        },
    )
    return BoundedSensorObservation(
        measured=measured,
        transport_center=np.clip(measured / gain, 0.0, 1.0),
        bounds=bounds,
        transport_observation=clean,
        diagnostics=bounds.diagnostics,
    )
