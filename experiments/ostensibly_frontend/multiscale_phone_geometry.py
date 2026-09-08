"""Raw-retaining multiscale and Gabor geometry for whole phone rasters."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve


RAW_WEIGHT = 0.1
SMOOTH2_WEIGHT = 4.0
SMOOTH4_WEIGHT = 8.0
GABOR_WEIGHT = 2.0
GABOR_POOL_SHAPE = (16, 6)
DURATION_LOG_WEIGHT = 0.08


def _unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array / max(float(np.linalg.norm(array)), 1e-30)


@lru_cache(maxsize=1)
def gabor_kernels() -> tuple[np.ndarray, ...]:
    kernels = []
    for wavelength in (4.0, 8.0, 16.0):
        sigma = 0.55 * wavelength
        radius = int(min(15, max(4, round(2.5 * sigma))))
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
            along = xx * np.cos(theta) + yy * np.sin(theta)
            across = -xx * np.sin(theta) + yy * np.cos(theta)
            kernel = np.exp(
                -(along * along + across * across) / (2.0 * sigma * sigma)
            ) * np.exp(2j * np.pi * along / wavelength)
            kernels.append(kernel - np.mean(kernel))
    return tuple(kernels)


def _pooled_gabor(field: np.ndarray) -> np.ndarray:
    shape = GABOR_POOL_SHAPE
    responses = []
    for kernel in gabor_kernels():
        magnitude = np.abs(fftconvolve(field, kernel, mode="same"))
        pooled = ndi.zoom(
            magnitude,
            (shape[0] / field.shape[0], shape[1] / field.shape[1]),
            order=1,
            mode="nearest",
            prefilter=False,
        )[:shape[0], :shape[1]]
        responses.append(pooled)
    return _unit(np.stack(responses))


def phone_geometry_components(field: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return normalized raw, coarse-scale, and oriented-ridge components."""

    source = np.asarray(field, dtype=np.float64)
    if source.ndim != 2 or not source.size:
        raise ValueError("phone geometry requires a nonempty 2-D field")
    return (
        _unit(source),
        _unit(ndi.gaussian_filter(source, sigma=(2.0, 1.2), mode="nearest")),
        _unit(ndi.gaussian_filter(source, sigma=(4.0, 2.4), mode="nearest")),
        _pooled_gabor(source),
    )


def multiscale_phone_descriptor(
    field: np.ndarray,
    weights: tuple[float, float, float, float] = (
        RAW_WEIGHT,
        SMOOTH2_WEIGHT,
        SMOOTH4_WEIGHT,
        GABOR_WEIGHT,
    ),
) -> np.ndarray:
    """Embed one complete raster in a single weighted scale-space geometry."""

    if len(weights) != 4 or not any(weight > 0.0 for weight in weights):
        raise ValueError("four nonnegative weights with positive mass are required")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("scale-space weights must be nonnegative")
    components = phone_geometry_components(field)
    return _unit(np.concatenate([
        weight * component
        for weight, component in zip(weights, components)
        if weight > 0.0
    ]))


def multiscale_phone_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = multiscale_phone_descriptor(left)
    second = multiscale_phone_descriptor(right)
    return float(1.0 - np.clip(np.dot(first, second), -1.0, 1.0))


def duration_coupled_similarity(
    query_descriptor: np.ndarray,
    query_duration_frames: int,
    witness_descriptor: np.ndarray,
    witness_duration_frames: int,
    *,
    duration_weight: float = DURATION_LOG_WEIGHT,
) -> float:
    """Couple scale-space agreement to a weak relative-duration witness.

    The fixed 0.08 coefficient maximized mean top-5 recall on the BDL↔SLT
    calibration folds before the disjoint CLB holdout was measured.
    """

    if duration_weight < 0.0:
        raise ValueError("duration weight must be nonnegative")
    query = _unit(query_descriptor)
    witness = _unit(witness_descriptor)
    acoustic = float(np.clip(np.dot(query, witness), -1.0, 1.0))
    duration = abs(np.log(
        max(int(query_duration_frames), 1)
        / max(int(witness_duration_frames), 1)
    ))
    return acoustic - duration_weight * float(duration)
