#!/usr/bin/env python3
"""Forensic focus measures: active autofocus versus one-frame defocus.

An ordinary sharpness score is a valid *active* autofocus objective because
the same scene is observed at several lens positions.  It is not an absolute
focus measurement: a sharp low-frequency surface can have less gradient
energy than a blurred high-frequency surface.

This module keeps those two questions separate and adds a one-frame model for
textured support.  Locally approximate the latent two-dimensional power
spectrum by a power law

    P(k) = C |k|^{-alpha}.

Gaussian optical blur of width ``sigma`` and a Gaussian derivative aperture
of width ``s`` give local gradient energy

    E(s) = K (s**2 + sigma**2)^{-q},
    q = (4 - alpha) / 2.

Thus ``log E(s)`` is affine in ``log(s**2 + sigma**2)``.  Contrast is absorbed
by the intercept, the unknown natural-image spectral slope is absorbed by
``q``, and ``sigma`` is selected from a short deterministic scale ladder.
The residual of that fit is an explicit identifiability confidence: periodic
patterns and flat support are returned as unknown rather than assigned a
fabricated focal depth.

The existing calibrated edge-reblur estimator remains the correct observable
for coherent step-like boundaries.  The scale-space model is its textured
counterpart, not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from experiments.transport_focus_forensics import (
    relative_defocus_evidence,
)


@dataclass(frozen=True)
class AutofocusMetricConfig:
    """Scale ladder and acceptance gates for textured focus evidence."""

    derivative_scales: tuple[float, ...] = (
        0.70, 1.00, 1.40, 2.00, 2.80, 4.00,
    )
    blur_candidates: tuple[float, ...] = (
        0.00, 0.35, 0.50, 0.70, 1.00, 1.40, 2.00, 2.80, 4.00, 5.60,
    )
    pooling_scale: float = 4.0
    minimum_spectral_exponent: float = 0.10
    maximum_spectral_exponent: float = 2.50
    minimum_log_energy_span: float = 0.35
    strength_percentile: float = 85.0


def _rgb01(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = value[..., :3]
    if value.max(initial=0.0) > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _linear_rgb(image: np.ndarray) -> np.ndarray:
    rgb = _rgb01(image)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_lightness(image: np.ndarray) -> np.ndarray:
    rgb = _linear_rgb(image)
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def active_autofocus_scores(image: np.ndarray) -> dict[str, float]:
    """Return conventional scores for comparing *the same scene* in a sweep.

    These numbers deliberately have no single-frame depth interpretation.
    Their only valid comparison is between lens states of the same support.
    """
    lightness = _linear_lightness(image)
    gx = ndi.gaussian_filter(
        lightness, 0.7, order=(0, 1), mode="reflect")
    gy = ndi.gaussian_filter(
        lightness, 0.7, order=(1, 0), mode="reflect")
    dxx = ndi.gaussian_filter(
        lightness, 0.7, order=(0, 2), mode="reflect")
    dyy = ndi.gaussian_filter(
        lightness, 0.7, order=(2, 0), mode="reflect")
    high = lightness - ndi.gaussian_filter(
        lightness, 1.0, mode="reflect")
    ac = lightness - ndi.gaussian_filter(
        lightness, 4.0, mode="reflect")
    variance = float(np.mean(ac * ac))
    return {
        "tenengrad": float(np.mean(gx * gx + gy * gy)),
        "modified_laplacian": float(np.mean(
            (np.abs(dxx) + np.abs(dyy)) ** 2)),
        "reblur_loss": float(np.mean(high * high)),
        "normalized_tenengrad": float(
            np.mean(gx * gx + gy * gy) / max(variance, 1e-30)),
        "normalized_reblur_loss": float(
            np.mean(high * high) / max(variance, 1e-30)),
    }


def texture_scale_space_focus(
    image: np.ndarray,
    config: AutofocusMetricConfig = AutofocusMetricConfig(),
) -> dict[str, np.ndarray | tuple[float, ...]]:
    """Estimate one-frame Gaussian blur over locally power-law texture.

    The fit is vectorized over pixels.  Only the small candidate blur ladder
    is looped.  ``confidence`` is zero when the local measurements do not
    support the model.
    """
    lightness = _linear_lightness(image)
    scales = np.asarray(config.derivative_scales, dtype=np.float64)
    candidates = np.asarray(config.blur_candidates, dtype=np.float64)
    if scales.ndim != 1 or scales.size < 4 or np.any(scales <= 0):
        raise ValueError("at least four positive derivative scales required")
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("blur candidate ladder cannot be empty")

    pooling = max(float(config.pooling_scale), 0.0)
    energy = []
    base_gx = None
    base_gy = None
    for scale in scales:
        gx = ndi.gaussian_filter(
            lightness, float(scale), order=(0, 1), mode="reflect")
        gy = ndi.gaussian_filter(
            lightness, float(scale), order=(1, 0), mode="reflect")
        if base_gx is None:
            base_gx = gx
            base_gy = gy
        local = gx * gx + gy * gy
        if pooling > 0.0:
            local = ndi.gaussian_filter(local, pooling, mode="reflect")
        energy.append(local)
    energy_stack = np.stack(energy, axis=0)
    positive = energy_stack[energy_stack > 0.0]
    floor = (
        max(float(np.percentile(positive, 0.5)) * 1e-3, 1e-30)
        if positive.size else 1e-30
    )
    log_energy = np.log(np.maximum(energy_stack, floor))
    mean_y = np.mean(log_energy, axis=0)
    centered_y = log_energy - mean_y
    total_y = np.sum(centered_y * centered_y, axis=0)

    best_error = np.full(lightness.shape, np.inf, dtype=np.float64)
    best_sigma = np.zeros(lightness.shape, dtype=np.float64)
    best_exponent = np.zeros(lightness.shape, dtype=np.float64)
    for sigma in candidates:
        x = np.log(scales * scales + sigma * sigma + 1e-12)
        centered_x = x - np.mean(x)
        xx = float(np.dot(centered_x, centered_x))
        covariance = np.tensordot(
            centered_x, centered_y, axes=(0, 0))
        slope = covariance / max(xx, 1e-30)
        error = np.maximum(
            total_y - covariance * covariance / max(xx, 1e-30),
            0.0,
        )
        improve = error < best_error
        best_error[improve] = error[improve]
        best_sigma[improve] = sigma
        best_exponent[improve] = -slope[improve]

    explained = np.clip(
        1.0 - best_error / np.maximum(total_y, 1e-30),
        0.0,
        1.0,
    )
    span = np.maximum(log_energy[0] - log_energy[-1], 0.0)
    span_gate = np.clip(
        span / max(float(config.minimum_log_energy_span), 1e-6),
        0.0,
        1.0,
    )
    exponent_gate = (
        (best_exponent >= float(config.minimum_spectral_exponent))
        & (best_exponent <= float(config.maximum_spectral_exponent))
    )
    base_energy = energy_stack[0]
    base_positive = base_energy[base_energy > 0.0]
    strength_scale = (
        float(np.percentile(
            base_positive, float(config.strength_percentile)))
        if base_positive.size else 1.0
    )
    strength = base_energy / np.maximum(
        base_energy + max(strength_scale, 1e-30), 1e-30)
    # The power-law derivation integrates over angular frequency. A single
    # sinusoid or a coherent step has a narrow orientation distribution and
    # can mimic the same scale curve with the wrong sigma. Route coherent
    # structure to the calibrated edge estimator instead.
    jxx = ndi.gaussian_filter(
        base_gx * base_gx, pooling, mode="reflect")
    jxy = ndi.gaussian_filter(
        base_gx * base_gy, pooling, mode="reflect")
    jyy = ndi.gaussian_filter(
        base_gy * base_gy, pooling, mode="reflect")
    coherence = np.hypot(
        jxx - jyy, 2.0 * jxy) / np.maximum(jxx + jyy, 1e-30)
    texture_isotropy = np.clip(1.0 - coherence, 0.0, 1.0)
    # A fit with no curvature preference is unidentifiable.  Compare the
    # selected model against the zero-blur endpoint, which is nested in the
    # same family and therefore a proper likelihood-style control.
    x0 = np.log(scales * scales + 1e-12)
    x0c = x0 - np.mean(x0)
    x0x0 = float(np.dot(x0c, x0c))
    covariance0 = np.tensordot(x0c, centered_y, axes=(0, 0))
    error0 = np.maximum(
        total_y - covariance0 * covariance0 / max(x0x0, 1e-30),
        0.0,
    )
    curvature_gain = np.clip(
        (error0 - best_error) / np.maximum(total_y, 1e-30),
        0.0,
        1.0,
    )
    # Sharp support legitimately selects sigma=0, so model confidence must
    # not vanish there.  Curvature gain only boosts a non-zero estimate.
    identifiability = np.where(
        best_sigma > candidates.min() + 1e-12,
        np.sqrt(curvature_gain),
        explained,
    )
    confidence = (
        explained
        * identifiability
        * span_gate
        * strength
        * texture_isotropy
        * exponent_gate
    )
    return {
        "blur_sigma": best_sigma,
        "spectral_exponent": best_exponent,
        "fit_r2": explained,
        "curvature_gain": curvature_gain,
        "energy_span": span,
        "coherence": coherence,
        "texture_isotropy": texture_isotropy,
        "confidence": confidence,
        "gradient_energy_ladder": energy_stack,
        "derivative_scales": tuple(float(x) for x in scales),
        "blur_candidates": tuple(float(x) for x in candidates),
    }


def chromatic_focus_evidence(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure signed channel focus ordering on common RGB structure.

    Longitudinal chromatic aberration can encode the sign of defocus, but
    only where all three channels observe the same physical edge.  Coloured
    material boundaries are rejected by the common-edge gate.
    """
    linear = _linear_rgb(image)
    channel_scale = []
    channel_confidence = []
    channel_gradient = []
    for index in range(3):
        gray = np.repeat(linear[..., index, None], 3, axis=2)
        # relative_defocus_evidence expects sRGB and linearizes it.  Encode
        # the already-linear channel before using the shared calibrated path.
        srgb = np.where(
            gray <= 0.0031308,
            12.92 * gray,
            1.055 * np.maximum(gray, 0.0) ** (1.0 / 2.4) - 0.055,
        )
        result = relative_defocus_evidence(srgb)
        channel_scale.append(np.log1p(result["effective_scale"]))
        channel_confidence.append(result["confidence"])
        channel_gradient.append(result["original_gradient"])
    scale = np.stack(channel_scale, axis=-1)
    confidence = np.stack(channel_confidence, axis=-1)
    gradient = np.stack(channel_gradient, axis=-1)

    # A material colour edge can be present almost entirely in one channel.
    # Require comparable normalized channel gradients before interpreting
    # channel-width differences optically.
    gradient_fraction = gradient / np.maximum(
        np.max(gradient, axis=-1, keepdims=True), 1e-30)
    common_edge = np.min(gradient_fraction, axis=-1)
    common_confidence = np.min(confidence, axis=-1) * common_edge
    signed_red_minus_blue = scale[..., 0] - scale[..., 2]
    spread = np.std(scale, axis=-1)
    return {
        "channel_log_effective_scale": scale,
        "common_edge_confidence": common_confidence,
        "red_minus_blue_scale": signed_red_minus_blue,
        "channel_scale_spread": spread,
        "signed_evidence": signed_red_minus_blue * common_confidence,
        "magnitude_evidence": spread * common_confidence,
    }
