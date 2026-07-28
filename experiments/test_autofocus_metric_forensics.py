#!/usr/bin/env python3
"""Controlled identifiability tests for autofocus observables."""

import numpy as np
from scipy import ndimage as ndi

from experiments.autofocus_metric_forensics import (
    active_autofocus_scores,
    chromatic_focus_evidence,
    texture_scale_space_focus,
)


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def _fractal_texture(
    size: int = 257,
    exponent: float = 1.2,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.hypot(fx, fy)
    amplitude = np.maximum(radius, 1.0 / size) ** (-0.5 * exponent)
    phase = rng.normal(size=(size, size)) + 1j * rng.normal(
        size=(size, size))
    texture = np.fft.ifft2(phase * amplitude).real
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    linear = 0.15 + 0.70 * texture
    return np.repeat(_encode_srgb(linear)[..., None], 3, axis=2)


def _weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value.ravel())
    values = value.ravel()[order]
    weights = weight.ravel()[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return 0.0
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def test_active_scores_peak_on_sharpest_view_of_same_scene() -> None:
    source = _fractal_texture()
    curves = {}
    for sigma in (0.0, 0.8, 1.6, 3.2):
        image = ndi.gaussian_filter(source, (sigma, sigma, 0.0))
        for name, value in active_autofocus_scores(image).items():
            curves.setdefault(name, []).append(value)
    for name in (
        "tenengrad",
        "modified_laplacian",
        "reblur_loss",
        "normalized_reblur_loss",
    ):
        values = curves[name]
        assert np.all(np.diff(values) < 0.0)


def test_variance_normalized_tenengrad_is_rejected_as_focus_objective() -> None:
    """Its denominator can contract faster than gradient energy under blur."""
    source = _fractal_texture()
    values = []
    for sigma in (0.0, 0.8, 1.6, 3.2):
        image = ndi.gaussian_filter(source, (sigma, sigma, 0.0))
        values.append(active_autofocus_scores(
            image)["normalized_tenengrad"])
    assert values[1] > values[0]


def test_texture_model_is_monotone_under_known_blur() -> None:
    source = _fractal_texture(exponent=1.4)
    estimates = []
    for sigma in (0.0, 0.7, 1.4, 2.8):
        image = ndi.gaussian_filter(source, (sigma, sigma, 0.0))
        result = texture_scale_space_focus(image)
        estimates.append(_weighted_median(
            result["blur_sigma"], result["confidence"]))
    assert np.all(np.diff(estimates) >= 0.0)
    assert estimates[-1] >= 2.0


def test_texture_model_absorbs_contrast_in_intercept() -> None:
    source = _fractal_texture(exponent=1.0)
    linear = np.where(
        source <= 0.04045,
        source / 12.92,
        ((source + 0.055) / 1.055) ** 2.4,
    )
    estimates = []
    for contrast in (0.25, 0.50, 0.80):
        adjusted = _encode_srgb(
            np.clip(0.5 + contrast * (linear - 0.5), 0.0, 1.0))
        blurred = ndi.gaussian_filter(adjusted, (2.0, 2.0, 0.0))
        result = texture_scale_space_focus(blurred)
        estimates.append(_weighted_median(
            result["blur_sigma"], result["confidence"]))
    assert np.ptp(estimates) <= 0.5


def test_texture_model_rejects_single_orientation_as_unidentifiable() -> None:
    size = 257
    _, xx = np.mgrid[:size, :size]
    linear = 0.5 + 0.35 * np.sin(2.0 * np.pi * 8.0 * xx / size)
    image = np.repeat(_encode_srgb(linear)[..., None], 3, axis=2)
    result = texture_scale_space_focus(image)
    assert float(np.mean(result["confidence"])) < 0.01
    assert float(np.percentile(result["texture_isotropy"], 90.0)) < 0.05


def test_chromatic_focus_is_zero_for_achromatic_blur() -> None:
    source = _fractal_texture()
    blurred = ndi.gaussian_filter(source, (1.8, 1.8, 0.0))
    result = chromatic_focus_evidence(blurred)
    weight = result["common_edge_confidence"]
    signed = result["red_minus_blue_scale"]
    mean = float(np.sum(weight * signed) / np.maximum(np.sum(weight), 1e-30))
    assert abs(mean) < 1e-8


def test_chromatic_focus_detects_channel_blur_order() -> None:
    source = _fractal_texture()
    linear = np.where(
        source <= 0.04045,
        source / 12.92,
        ((source + 0.055) / 1.055) ** 2.4,
    )
    red = ndi.gaussian_filter(linear[..., 0], 2.4)
    green = ndi.gaussian_filter(linear[..., 1], 1.5)
    blue = ndi.gaussian_filter(linear[..., 2], 0.8)
    image = _encode_srgb(np.stack((red, green, blue), axis=-1))
    result = chromatic_focus_evidence(image)
    weight = result["common_edge_confidence"]
    signed = result["red_minus_blue_scale"]
    mean = float(np.sum(weight * signed) / np.maximum(np.sum(weight), 1e-30))
    assert mean > 0.15
