"""PORT 01: one frozen Meyer support measure and physical metric.

C++ input
    contiguous RGB image plus Meyer/support scalar parameters.
C++ output
    float32 measure, Qxx, Qxy, Qyy, cartoon, texture, glass and energy.

The target is decomposed once.  Restriction to the allocation grid samples
that finished geometry; it must never decompose a resized or changed target.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.wasserstein_allocation_tree import (
    pyramid_geometry as _pyramid_geometry,
    single_decomposition_geometry as _single_decomposition_geometry,
)
from .fast_image_ops import gaussian_filter, sobel


def build_frozen_geometry(rgb: np.ndarray, **parameters) -> dict:
    parameters.setdefault("meyer_solver", 1)
    geometry = _single_decomposition_geometry(rgb, **parameters)
    # The transport walk is bandwidth-bound.  Preserve float64 scalars, but
    # make image fields explicit float32 port targets.
    for key, value in tuple(geometry.items()):
        if isinstance(value, np.ndarray):
            geometry[key] = np.ascontiguousarray(value, dtype=np.float32)
    return geometry


def restrict_geometry(geometry: dict, maximum_side: int) -> dict:
    return _pyramid_geometry(geometry, int(maximum_side))


def reweight_frozen_support(
    geometry: dict,
    *,
    texture_support_weight: float,
    glass_support_weight: float,
    null_evidence_strength: float | None = None,
) -> dict:
    """Rebuild only the support tensor from an existing Meyer decomposition.

    ``cartoon``, ``texture`` and ``glass`` are already frozen fields.  This
    function changes which of those fields commands population without
    repeating Meyer, ROF, source-evidence, or boundary measurements.
    """
    cartoon = np.asarray(geometry["cartoon"], dtype=np.float64)
    texture = np.asarray(geometry["texture"], dtype=np.float64)
    glass = np.asarray(geometry["glass"], dtype=np.float64)
    smooth_cartoon = gaussian_filter(cartoon, 3.0)
    channels = np.ascontiguousarray(np.stack((
        cartoon - smooth_cartoon,
        math.sqrt(max(float(texture_support_weight), 0.0)) * texture,
        math.sqrt(max(float(glass_support_weight), 0.0)) * glass,
    )))
    channel_gx, channel_gy = sobel(channels)
    energy = np.sum(channels * channels, axis=0)
    jxx = np.sum(channel_gx * channel_gx, axis=0)
    jxy = np.sum(channel_gx * channel_gy, axis=0)
    jyy = np.sum(channel_gy * channel_gy, axis=0)
    energy, jxx, jxy, jyy = gaussian_filter(
        np.stack((energy, jxx, jxy, jyy)), 1.5)

    scale = max(float(np.percentile(energy, 99.5)), 1e-20)
    transport_reliability = energy / (energy + 1e-5 * scale)
    source_reliability = np.asarray(
        geometry["source_reliability"], dtype=np.float64)
    null_attenuation = np.asarray(
        geometry["null_attenuation"], dtype=np.float64)
    null_strength = float(geometry.get("null_evidence_strength", 0.0))
    if null_evidence_strength is not None and null_strength > 1e-30:
        weak_null = np.clip(
            (1.0 - null_attenuation) / null_strength, 0.0, 1.0)
        null_strength = float(np.clip(
            null_evidence_strength, 0.0, 1.0))
        null_attenuation = 1.0 - null_strength * weak_null
        source_reliability = (
            np.asarray(
                geometry["amplitude_reliability"], dtype=np.float64)
            * null_attenuation
        )
    reliability = transport_reliability * source_reliability
    denominator = energy + 1e-5 * scale
    qxx = reliability * jxx / denominator
    qxy = reliability * jxy / denominator
    qyy = reliability * jyy / denominator

    max_length = max(float(geometry["max_support_px"]), 1.0)
    frequency_floor = 1.0 / (max_length * max_length)
    qxx += frequency_floor
    qyy += frequency_floor
    trace = qxx + qyy
    discriminant = np.hypot(qxx - qyy, 2.0 * qxy)
    coherence = discriminant / np.maximum(trace, 1e-30)
    exact_high = 0.5 * (trace + discriminant)
    exact_low = 0.5 * (trace - discriminant)
    high = np.maximum(exact_high, frequency_floor)
    low = np.maximum(exact_low, frequency_floor)
    tangent_fraction = float(np.clip(
        geometry.get("coherent_tangent_fraction", 0.02), 0.0, 1.0))
    low_factor = 1.0 - (1.0 - tangent_fraction) * coherence
    low = frequency_floor + low_factor * (low - frequency_floor)

    safe_discriminant = np.maximum(discriminant, 1e-30)
    alpha = (high - low) / safe_discriminant
    beta = (
        low * exact_high - high * exact_low
    ) / safe_discriminant
    degenerate = discriminant < 1e-18
    alpha = np.where(degenerate, 1.0, alpha)
    beta = np.where(degenerate, 0.0, beta)
    qxx = alpha * qxx + beta
    qxy = alpha * qxy
    qyy = alpha * qyy + beta

    determinant = np.maximum(qxx * qyy - qxy * qxy, 0.0)
    raw_measure = np.sqrt(determinant) / math.pi
    implied_cells = max(float(np.sum(raw_measure)), 1e-30)

    result = dict(geometry)
    result.update({
        "measure": np.ascontiguousarray(
            raw_measure / implied_cells, dtype=np.float32),
        "precision_xx": np.ascontiguousarray(qxx, dtype=np.float32),
        "precision_xy": np.ascontiguousarray(qxy, dtype=np.float32),
        "precision_yy": np.ascontiguousarray(qyy, dtype=np.float32),
        "energy": np.ascontiguousarray(energy, dtype=np.float32),
        "implied_cells": implied_cells,
        "metric_trace_p90": max(
            float(np.percentile(qxx + qyy, 90.0)), 1e-12),
        "texture_support_weight": float(texture_support_weight),
        "glass_support_weight": float(glass_support_weight),
        "source_reliability": np.ascontiguousarray(
            source_reliability, dtype=np.float32),
        "null_attenuation": np.ascontiguousarray(
            null_attenuation, dtype=np.float32),
        "null_evidence_strength": null_strength,
        "support_reweighted_from_frozen_decomposition": True,
    })
    return result
