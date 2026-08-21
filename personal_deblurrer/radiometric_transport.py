"""Symmetric exposure gauge and continuous sensor-censoring precision.

The geometric transport operates on radiance-like observations. Real capture
pairs can instead differ by exposure gain and clipping. This module estimates
one relative gain from their full quantile transport, moves both observations
to the symmetric geometric-mean gauge, and lowers precision continuously near
sensor bounds. It never chooses a preferred frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dense_estimation import _luminance


@dataclass(frozen=True)
class RadiometricPairTransport:
    images: tuple[np.ndarray, np.ndarray]
    precision: np.ndarray
    relative_gain_second_over_first: float
    authority: float
    diagnostics: dict[str, object]


def _soft_sensor_precision(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 3:
        upper_value = np.max(value, axis=2)
        lower_value = np.min(value, axis=2)
    else:
        upper_value = value
        lower_value = value
    upper_headroom = 1.0 / (
        1.0 + np.exp(np.clip((upper_value - 0.985) / 0.008, -60.0, 60.0)))
    lower_headroom = 1.0 / (
        1.0 + np.exp(np.clip((0.003 - lower_value) / 0.002, -60.0, 60.0)))
    return 0.01 + 0.99 * upper_headroom * lower_headroom


def _quantile_log_gain(first: np.ndarray, second: np.ndarray) -> tuple[float, int]:
    a = _luminance(first)
    b = _luminance(second)
    quantiles = np.linspace(0.08, 0.82, 38, dtype=np.float64)
    first_quantiles = np.quantile(a, quantiles)
    second_quantiles = np.quantile(b, quantiles)
    supported = (
        (first_quantiles > 0.02)
        & (first_quantiles < 0.85)
        & (second_quantiles > 0.02)
        & (second_quantiles < 0.95)
    )
    if not np.any(supported):
        return 0.0, 0
    log_ratios = (
        np.log(second_quantiles[supported])
        - np.log(first_quantiles[supported])
    )
    return float(np.median(log_ratios)), int(np.sum(supported))


def transport_radiometric_pair(
    first: np.ndarray,
    second: np.ndarray,
) -> RadiometricPairTransport:
    """Return a symmetric radiometric gauge and positive sensor precision."""
    images = (
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    if images[0].shape != images[1].shape:
        raise ValueError("radiometric pair must share one raster")
    log_gain, supported_quantiles = _quantile_log_gain(*images)
    log_gain = float(np.clip(log_gain, -np.log(8.0), np.log(8.0)))
    relative_gain = float(np.exp(log_gain))
    half_gain = float(np.exp(0.5 * log_gain))
    normalized = (
        images[0] * half_gain,
        images[1] / half_gain,
    )
    mismatch = abs(log_gain)
    authority = float(1.0 - np.exp(-((mismatch / 0.12) ** 4)))
    transported = tuple(
        (1.0 - authority) * image + authority * gauge_image
        for image, gauge_image in zip(images, normalized)
    )
    raw_precision = np.stack([
        _soft_sensor_precision(image) for image in images
    ], axis=0)
    precision = (
        (1.0 - authority) * np.ones_like(raw_precision)
        + authority * raw_precision
    )
    clipped_fraction = [
        float(np.mean(np.max(image, axis=2) >= 1.0 - 1e-8))
        if image.ndim == 3 else float(np.mean(image >= 1.0 - 1e-8))
        for image in images
    ]
    return RadiometricPairTransport(
        images=(transported[0], transported[1]),
        precision=np.ascontiguousarray(precision),
        relative_gain_second_over_first=relative_gain,
        authority=authority,
        diagnostics={
            "radiometric_method": (
                "symmetric_quantile_gain_and_continuous_sensor_censoring"),
            "radiometric_gauge": "geometric_mean_exposure",
            "relative_gain_second_over_first": relative_gain,
            "radiometric_log_gain_magnitude": mismatch,
            "radiometric_authority": authority,
            "supported_quantile_count": supported_quantiles,
            "sensor_precision_mean": [
                float(np.mean(precision[index])) for index in range(2)
            ],
            "sensor_precision_min": [
                float(np.min(precision[index])) for index in range(2)
            ],
            "hard_upper_clip_fraction": clipped_fraction,
            "radiometric_role": (
                "continuous_precision_measure_not_frame_selection"),
        },
    )
