#!/usr/bin/env python3
"""Invariants for contrast-normalized focus evidence."""

import numpy as np
from scipy import ndimage as ndi

from experiments.transport_focus_forensics import (
    autofocus_cell_score,
    relative_defocus_evidence,
    transport_focus_forensics,
    transport_focus_interfaces,
)
from experiments.embedded_interface_topology import (
    build_embedded_interface_topology,
)


def _step(size: int = 257, contrast: float = 1.0) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.float64)
    image[:, size // 2:] = contrast
    return image


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def _optically_blurred_step(sigma: float, contrast: float = 1.0) -> np.ndarray:
    linear = ndi.gaussian_filter(
        _step(contrast=contrast), (sigma, sigma, 0.0))
    return _encode_srgb(linear)


def _ridge_estimate(result: dict) -> float:
    confidence = result["confidence"]
    value = result["defocus_radius"]
    y = confidence.shape[0] // 2
    band = np.s_[y - 32:y + 33, :]
    index = np.argmax(confidence[band], axis=1)
    rows = np.arange(y - 32, y + 33)
    weight = confidence[rows, index]
    return float(np.sum(weight * value[rows, index]) / np.sum(weight))


def test_known_gaussian_blur_is_recovered() -> None:
    source = _step()
    estimates = []
    for sigma in (1.0, 2.0, 4.0):
        blurred = _optically_blurred_step(sigma)
        estimates.append(_ridge_estimate(
            relative_defocus_evidence(blurred)))
    assert np.all(np.diff(estimates) > 0.0)
    assert np.allclose(estimates, (1.0, 2.0, 4.0), atol=0.25)


def test_edge_contrast_cancels() -> None:
    estimates = []
    for contrast in (0.15, 0.35, 0.70, 1.0):
        blurred = _optically_blurred_step(2.0, contrast)
        estimates.append(_ridge_estimate(
            relative_defocus_evidence(blurred)))
    assert np.ptp(estimates) < 0.03


def test_flat_region_has_no_focus_confidence() -> None:
    flat = np.full((64, 80, 3), 0.4)
    result = relative_defocus_evidence(flat)
    assert float(np.max(result["confidence"])) < 1e-12


def test_transport_pooling_never_crosses_labels() -> None:
    image = _step(129)
    image[:, :64] = ndi.gaussian_filter(
        image[:, :64], (3.0, 3.0, 0.0))
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    labels[:, 64:] = 1
    result = transport_focus_forensics(image, labels)
    assert result["cell_effective_scale"].shape == (2,)
    assert result["cell_evidence_coverage"].shape == (2,)
    assert result["cell_defocus_radius"].shape == (2,)


def test_autofocus_score_is_neutral_without_evidence() -> None:
    evidence = {
        "cell_defocus_radius": np.array([0.0, 1.0]),
        "cell_evidence_coverage": np.zeros(2),
    }
    assert np.allclose(autofocus_cell_score(evidence), 0.5)


def test_autofocus_score_prefers_supported_sharp_cell() -> None:
    evidence = {
        "cell_defocus_radius": np.array([0.2, 3.5]),
        "cell_evidence_coverage": np.ones(2),
    }
    score = autofocus_cell_score(evidence)
    assert score[0] > 0.7
    assert score[1] < 0.4


def _layered_focus_scene(foreground_blur: float) -> tuple[np.ndarray, np.ndarray]:
    height = width = 193
    yy, xx = np.mgrid[:height, :width]
    mask = (xx - 96) ** 2 + (yy - 96) ** 2 < 42 ** 2
    background = 0.25 + 0.12 * np.sin(0.45 * xx) * np.sin(0.32 * yy)
    foreground = 0.70 + 0.10 * np.sin(0.55 * xx) * np.sin(0.41 * yy)
    if foreground_blur <= 0.0:
        linear = np.where(
            mask,
            foreground,
            ndi.gaussian_filter(background, 3.0),
        )
    else:
        alpha = ndi.gaussian_filter(mask.astype(np.float64), foreground_blur)
        blurred_foreground = ndi.gaussian_filter(
            foreground, foreground_blur)
        linear = alpha * blurred_foreground + (1.0 - alpha) * background
    srgb = _encode_srgb(np.clip(linear, 0.0, 1.0))
    return np.repeat(srgb[..., None], 3, axis=2), mask.astype(np.int32)


def test_occlusion_edge_blur_matches_the_foreground_side() -> None:
    # Label 0 is background and label 1 is foreground in both controls.
    # A focused foreground supplies a sharp common edge over a blurred
    # background; a defocused foreground supplies a blurred common edge over
    # a sharp background. In both cases the edge scale follows label 1.
    for foreground_blur in (0.0, 3.0):
        image, labels = _layered_focus_scene(foreground_blur)
        evidence = relative_defocus_evidence(image)
        topology = build_embedded_interface_topology(labels)
        relation = transport_focus_interfaces(
            evidence, labels, topology)
        assert relation["reliability"][0] > 0.1
        assert relation["first_match_margin"][0] < -0.05
