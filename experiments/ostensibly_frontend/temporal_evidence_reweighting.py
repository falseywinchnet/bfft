"""Cleanup/SHARK-supported temporal reweighting of occupation measures."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def cleanup_shark_extraction_support(
    cleanup_probability: np.ndarray,
    fused_score: np.ndarray,
    unvoiced_score: np.ndarray,
) -> np.ndarray:
    """Return a conservative union envelope for voiced or unvoiced content."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (cleanup_probability, fused_score, unvoiced_score)
    )
    if (
        not arrays[0].size
        or any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays)
        or not all(np.all(np.isfinite(value)) for value in arrays)
    ):
        raise ValueError("Cleanup/SHARK extraction evidence is incompatible")
    cleanup = np.clip(arrays[0], 0.0, 1.0)
    voiced = expit((arrays[1] - 0.95) / 0.20)
    unvoiced = np.clip(arrays[2], 0.0, 1.0)
    return np.maximum.reduce((cleanup, voiced, unvoiced))


def temporal_reweight_cloud(
    points: np.ndarray,
    frame_support: np.ndarray,
    *,
    support_floor: float,
) -> np.ndarray:
    """Resample cloud mass by measured frame support without moving points."""

    cloud = np.asarray(points, dtype=np.float64)
    support = np.asarray(frame_support, dtype=np.float64)
    if (
        cloud.ndim != 2
        or cloud.shape[1] != 3
        or not cloud.size
        or not np.all(np.isfinite(cloud))
        or support.ndim != 1
        or not support.size
        or not np.all(np.isfinite(support))
        or not 0.0 <= support_floor <= 1.0
    ):
        raise ValueError("temporal reweighting received invalid evidence")
    if support_floor == 1.0:
        return cloud.copy()

    interpolated = np.interp(
        cloud[:, 1],
        np.arange(support.size, dtype=np.float64),
        np.clip(support, 0.0, 1.0),
        left=float(np.clip(support[0], 0.0, 1.0)),
        right=float(np.clip(support[-1], 0.0, 1.0)),
    )
    weights = support_floor + (1.0 - support_floor) * interpolated
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("temporal support removes the complete occupation measure")
    order = np.lexsort((cloud[:, 2], cloud[:, 1], cloud[:, 0]))
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights / np.sum(ordered_weights))
    cumulative[-1] = 1.0
    quantiles = (np.arange(cloud.shape[0], dtype=np.float64) + 0.5) / cloud.shape[0]
    selected = order[np.searchsorted(cumulative, quantiles, side="left")]
    return cloud[selected]
