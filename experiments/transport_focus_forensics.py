#!/usr/bin/env python3
"""Contrast-normalized defocus evidence on frozen transport support.

This module does not infer objects or alter the transport cells.  It measures
one asymmetric scene observable that the reversible transport metric cannot
contain by itself: the width of image structure under defocus.

For an ideal step edge of contrast ``c`` blurred by a Gaussian point-spread
function of width ``sigma``, the peak gradient after differentiation at scale
``s`` is proportional to

    c / sqrt(sigma**2 + s**2).

Apply one *known* additional Gaussian blur ``delta`` and form the ratio

    r = gradient_reblurred / gradient_original
      = a / sqrt(a**2 + delta**2),

where ``a**2 = sigma**2 + s**2``.  Therefore

    a = delta * r / sqrt(1 - r**2).

The unknown edge contrast cancels.  The estimate is only trusted on coherent
gradient ridges; flat interiors carry no focus evidence.  Sparse observations
are pooled onto the already-existing connected transport fragments without
crossing or creating an interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

@dataclass(frozen=True)
class FocusEvidenceConfig:
    """Sampling scales for the calibrated reblur experiment."""

    derivative_scale: float = 0.8
    reblur_scale: float = 1.0
    tensor_scale: float = 1.2
    strength_percentile: float = 90.0
    maximum_effective_scale: float = 12.0


def _rgb01(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = value[..., :3]
    if value.max(initial=0.0) > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _linear_lightness(image: np.ndarray) -> np.ndarray:
    """Linear-light Rec. 709 luminance, required for contrast cancellation."""
    rgb = _rgb01(image)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def _gradient(
    lightness: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = ndi.gaussian_filter(
        lightness, sigma, order=(0, 1), mode="reflect")
    gy = ndi.gaussian_filter(
        lightness, sigma, order=(1, 0), mode="reflect")
    return np.hypot(gx, gy), gx, gy


def relative_defocus_evidence(
    image: np.ndarray,
    config: FocusEvidenceConfig = FocusEvidenceConfig(),
) -> dict[str, np.ndarray | float]:
    """Measure effective edge width and its observation confidence.

    ``effective_scale`` includes the derivative aperture and is the directly
    observed, best-conditioned quantity. ``defocus_radius`` removes that
    known aperture in quadrature. Values below the pixel-aperture floor are
    consequently zero and should be interpreted as "sharper than resolved,"
    not as literal zero optical blur.
    """
    derivative_scale = max(float(config.derivative_scale), 1e-3)
    reblur_scale = max(float(config.reblur_scale), 1e-3)
    tensor_scale = max(float(config.tensor_scale), 1e-3)
    lightness = _linear_lightness(image)

    original, gx, gy = _gradient(lightness, derivative_scale)
    combined_scale = np.hypot(derivative_scale, reblur_scale)
    reblurred, _, _ = _gradient(lightness, combined_scale)
    raw_ratio = reblurred / np.maximum(original, 1e-30)
    ratio = np.clip(raw_ratio, 1e-6, 1.0 - 1e-6)
    effective = (
        reblur_scale * ratio / np.sqrt(1.0 - ratio * ratio))
    effective = np.minimum(
        effective, max(float(config.maximum_effective_scale), 1e-3))
    defocus = np.sqrt(np.maximum(
        effective * effective - derivative_scale * derivative_scale,
        0.0,
    ))

    # The structure tensor distinguishes a coherent edge from isotropic
    # texture. A soft ridge score suppresses pixels beside an edge where
    # reblurring can increase rather than decrease the local response.
    jxx = ndi.gaussian_filter(
        gx * gx, tensor_scale, mode="reflect")
    jxy = ndi.gaussian_filter(
        gx * gy, tensor_scale, mode="reflect")
    jyy = ndi.gaussian_filter(
        gy * gy, tensor_scale, mode="reflect")
    coherence = (
        np.hypot(jxx - jyy, 2.0 * jxy)
        / np.maximum(jxx + jyy, 1e-30)
    )
    positive = original[original > 0.0]
    strength_scale = (
        float(np.percentile(
            positive, float(config.strength_percentile)))
        if positive.size else 1.0
    )
    strength_scale = max(strength_scale, 1e-30)
    strength = original / (original + strength_scale)
    ridge = (
        original
        / np.maximum(ndi.maximum_filter(original, size=3), 1e-30)
    )
    valid_loss = raw_ratio < 1.0
    confidence = (
        strength
        * np.clip(coherence, 0.0, 1.0)
        * np.clip(ridge, 0.0, 1.0)
        * valid_loss
    )
    relative_focus = 1.0 / (1.0 + effective)
    return {
        "lightness": lightness,
        "original_gradient": original,
        "reblurred_gradient": reblurred,
        "reblur_ratio": raw_ratio,
        "effective_scale": effective,
        "defocus_radius": defocus,
        "relative_focus": relative_focus,
        "coherence": coherence,
        "confidence": confidence,
        "derivative_scale": derivative_scale,
        "reblur_scale": reblur_scale,
    }


def _weighted_cell_mean(
    labels: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cell = np.asarray(labels, dtype=np.int32)
    if cell.shape != np.asarray(value).shape:
        raise ValueError("focus values and transport labels must align")
    count = int(cell.max(initial=-1)) + 1
    flat = cell.ravel()
    w = np.asarray(weight, dtype=np.float64).ravel()
    total_weight = np.bincount(flat, weights=w, minlength=count)
    total_value = np.bincount(
        flat,
        weights=w * np.asarray(value, dtype=np.float64).ravel(),
        minlength=count,
    )
    mean = total_value / np.maximum(total_weight, 1e-30)
    return mean, total_weight


def transport_focus_forensics(
    image: np.ndarray,
    labels: np.ndarray,
    config: FocusEvidenceConfig = FocusEvidenceConfig(),
) -> dict[str, np.ndarray | float]:
    """Pool complementary edge and texture focus onto transport fragments.

    Coherent boundaries use the calibrated step-edge reblur estimate. Local
    isotropic texture uses the power-law scale-space estimate. Their
    orientation gates are complementary, and their confidences remain
    observable separately in the returned record.
    """
    edge = relative_defocus_evidence(image, config)
    # Lazy import avoids making the standalone calibrated edge estimator
    # depend on the larger autofocus-forensics module.
    from experiments.autofocus_metric_forensics import (
        chromatic_focus_evidence,
        texture_scale_space_focus,
    )
    texture = texture_scale_space_focus(image)
    chromatic = chromatic_focus_evidence(image)
    edge_weight = np.asarray(edge["confidence"], dtype=np.float64)
    texture_weight = np.asarray(texture["confidence"], dtype=np.float64)
    total_weight = edge_weight + texture_weight
    defocus = (
        edge_weight * np.asarray(edge["defocus_radius"], dtype=np.float64)
        + texture_weight * np.asarray(texture["blur_sigma"], dtype=np.float64)
    ) / np.maximum(total_weight, 1e-30)
    derivative_scale = float(edge["derivative_scale"])
    effective = np.hypot(defocus, derivative_scale)
    confidence = 1.0 - (
        1.0 - np.clip(edge_weight, 0.0, 1.0)
    ) * (
        1.0 - np.clip(texture_weight, 0.0, 1.0)
    )
    evidence = {
        **edge,
        "edge_effective_scale": edge["effective_scale"],
        "edge_defocus_radius": edge["defocus_radius"],
        "edge_confidence": edge["confidence"],
        "texture_blur_sigma": texture["blur_sigma"],
        "texture_spectral_exponent": texture["spectral_exponent"],
        "texture_fit_r2": texture["fit_r2"],
        "texture_curvature_gain": texture["curvature_gain"],
        "texture_isotropy": texture["texture_isotropy"],
        "texture_focus_confidence": texture["confidence"],
        "chromatic_focus_sign": chromatic["signed_evidence"],
        "chromatic_focus_confidence":
            chromatic["common_edge_confidence"],
        "chromatic_focus_spread": chromatic["magnitude_evidence"],
        "effective_scale": effective,
        "defocus_radius": defocus,
        "relative_focus": 1.0 / (1.0 + effective),
        "confidence": confidence,
    }
    cell_scale, cell_mass = _weighted_cell_mean(
        labels,
        np.log1p(evidence["effective_scale"]),
        evidence["confidence"],
    )
    cell_scale = np.expm1(cell_scale)
    cell_defocus, _ = _weighted_cell_mean(
        labels,
        np.log1p(evidence["defocus_radius"]),
        evidence["confidence"],
    )
    cell_defocus = np.expm1(cell_defocus)
    cell_focus = 1.0 / (1.0 + cell_scale)
    area = np.bincount(
        np.asarray(labels, dtype=np.int32).ravel(),
        minlength=len(cell_scale),
    ).astype(np.float64)
    cell_coverage = cell_mass / np.maximum(area, 1.0)
    return {
        **evidence,
        "cell_effective_scale": cell_scale,
        "cell_defocus_radius": cell_defocus,
        "cell_relative_focus": cell_focus,
        "cell_evidence_mass": cell_mass,
        "cell_evidence_coverage": cell_coverage,
    }


def autofocus_cell_score(
    forensics: dict[str, np.ndarray | float],
    *,
    resolved_blur_scale: float = 1.5,
) -> np.ndarray:
    """Return a unary focal-plane selection score for every support cell.

    This is intentionally not an interface barrier. It answers the narrower
    question "is this observed support close to the resolved focal plane?"
    Unknown support is neutral (0.5), sharp supported structure approaches
    one, and confidently defocused support approaches zero.
    """
    blur = np.asarray(
        forensics["cell_defocus_radius"], dtype=np.float64)
    coverage = np.asarray(
        forensics["cell_evidence_coverage"], dtype=np.float64)
    observed = coverage[coverage > 1e-12]
    coverage_scale = (
        float(np.median(observed)) if observed.size else 1.0)
    reliability = coverage / np.maximum(
        coverage + max(coverage_scale, 1e-12), 1e-30)
    resolved = np.exp(
        -0.5 * (blur / max(float(resolved_blur_scale), 1e-6)) ** 2)
    return np.clip(
        0.5 + reliability * (resolved - 0.5),
        0.0,
        1.0,
    )


def focus_likeness(
    forensics: dict[str, np.ndarray | float],
    anchor_cell: int,
) -> np.ndarray:
    """Return a confidence-gated likeness to one cell's focus distribution."""
    scale = np.log1p(np.asarray(
        forensics["cell_effective_scale"], dtype=np.float64))
    coverage = np.asarray(
        forensics["cell_evidence_coverage"], dtype=np.float64)
    anchor = int(np.clip(anchor_cell, 0, max(len(scale) - 1, 0)))
    observed = scale[coverage > 1e-8]
    shoulder = (
        1.4826 * float(np.median(np.abs(observed - np.median(observed))))
        if observed.size else 1.0
    )
    shoulder = max(shoulder, 0.05)
    likeness = np.exp(-0.5 * ((scale - scale[anchor]) / shoulder) ** 2)
    confidence = np.sqrt(
        np.clip(coverage / max(float(np.percentile(
            coverage, 90.0)), 1e-30), 0.0, 1.0)
        * np.clip(
            coverage[anchor] / max(float(np.percentile(
                coverage, 90.0)), 1e-30),
            0.0,
            1.0,
        )
    )
    return likeness * confidence


def transport_focus_interfaces(
    forensics: dict[str, np.ndarray | float],
    labels: np.ndarray,
    topology: dict,
    *,
    interior_margin: float = 2.0,
    side_offset_min: int = 6,
    side_offset_max: int = 14,
) -> dict[str, np.ndarray]:
    """Compare each interface's blur with the blur inside its two sides.

    This is a measurement, not a border-ownership decision.  A positive
    ``first_match_margin`` says the observed boundary scale is closer to the
    nearby half-strip scale on the side of ``arc.cell_first``; a negative
    value says it is closer to the side of ``arc.cell_second``.
    ``reliability`` is kept separate so an absent focus observation cannot
    become a direction. The half-strips deliberately cross reconstruction
    cell seams: optical focus is a property of the imaged surface, not of the
    tessellation used to represent it.
    """
    cell = np.asarray(labels, dtype=np.int32)
    height, width = cell.shape
    count = int(cell.max(initial=-1)) + 1
    confidence = np.asarray(
        forensics["confidence"], dtype=np.float64)
    log_scale = np.log1p(np.asarray(
        forensics["effective_scale"], dtype=np.float64))

    boundary = np.zeros(cell.shape, dtype=bool)
    boundary[:, 1:] |= cell[:, 1:] != cell[:, :-1]
    boundary[:, :-1] |= cell[:, 1:] != cell[:, :-1]
    boundary[1:] |= cell[1:] != cell[:-1]
    boundary[:-1] |= cell[1:] != cell[:-1]
    distance = ndi.distance_transform_edt(~boundary)
    interior_gate = np.clip(
        (distance - 0.5) / max(float(interior_margin), 1e-6),
        0.0,
        1.0,
    )
    interior_weight = confidence * interior_gate
    node_scale, node_mass = _weighted_cell_mean(
        cell, log_scale, interior_weight)
    area = np.bincount(
        cell.ravel(), minlength=count).astype(np.float64)
    node_coverage = node_mass / np.maximum(area, 1.0)

    edgel = topology["edgel"]
    arc = np.asarray(edgel["arc"], dtype=np.int32)
    orientation = np.asarray(edgel["orientation"], dtype=np.int8)
    vertex = np.asarray(edgel["vertex_first"], dtype=np.int64)
    grid_x = vertex % (width + 1)
    grid_y = vertex // (width + 1)
    vertical = orientation == 1
    y0 = np.where(vertical, grid_y, grid_y - 1)
    x0 = np.where(vertical, grid_x - 1, grid_x)
    y1 = np.where(vertical, grid_y, grid_y)
    x1 = np.where(vertical, grid_x, grid_x)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    edgel_weight = 0.5 * (
        confidence[y0, x0] + confidence[y1, x1])
    edgel_scale = (
        confidence[y0, x0] * log_scale[y0, x0]
        + confidence[y1, x1] * log_scale[y1, x1]
    ) / np.maximum(
        confidence[y0, x0] + confidence[y1, x1],
        1e-30,
    )

    arc_count = int(topology["arc"]["count"])
    arc_mass = np.bincount(
        arc, weights=edgel_weight, minlength=arc_count)
    arc_scale = np.bincount(
        arc,
        weights=edgel_weight * edgel_scale,
        minlength=arc_count,
    ) / np.maximum(arc_mass, 1e-30)
    arc_length = np.asarray(
        topology["arc"]["length"], dtype=np.float64)
    first = np.asarray(
        topology["arc"]["cell_first"], dtype=np.int32)
    second = np.asarray(
        topology["arc"]["cell_second"], dtype=np.int32)

    # Measure the two physical sides of each embedded edgel. Tiny transport
    # fragments often contain no pixel two samples from their own boundary;
    # requiring a fragment interior would therefore erase the cue precisely
    # where the representation is densest.
    side0_numerator = np.zeros(len(arc), dtype=np.float64)
    side0_mass = np.zeros(len(arc), dtype=np.float64)
    side1_numerator = np.zeros(len(arc), dtype=np.float64)
    side1_mass = np.zeros(len(arc), dtype=np.float64)
    offset_first = max(int(side_offset_min), 1)
    offset_last = max(int(side_offset_max), offset_first)
    kernel_mass = 0.0
    for offset in range(offset_first, offset_last + 1):
        kernel = 1.0 / float(offset)
        kernel_mass += kernel
        side0_y = np.where(vertical, grid_y, grid_y - offset)
        side0_x = np.where(vertical, grid_x - offset, grid_x)
        side1_y = np.where(vertical, grid_y, grid_y + offset - 1)
        side1_x = np.where(vertical, grid_x + offset - 1, grid_x)
        valid0 = (
            (side0_y >= 0) & (side0_y < height)
            & (side0_x >= 0) & (side0_x < width)
        )
        valid1 = (
            (side1_y >= 0) & (side1_y < height)
            & (side1_x >= 0) & (side1_x < width)
        )
        weight0 = np.zeros(len(arc), dtype=np.float64)
        weight1 = np.zeros(len(arc), dtype=np.float64)
        value0 = np.zeros(len(arc), dtype=np.float64)
        value1 = np.zeros(len(arc), dtype=np.float64)
        weight0[valid0] = kernel * confidence[
            side0_y[valid0], side0_x[valid0]]
        weight1[valid1] = kernel * confidence[
            side1_y[valid1], side1_x[valid1]]
        value0[valid0] = log_scale[
            side0_y[valid0], side0_x[valid0]]
        value1[valid1] = log_scale[
            side1_y[valid1], side1_x[valid1]]
        side0_mass += weight0
        side1_mass += weight1
        side0_numerator += weight0 * value0
        side1_numerator += weight1 * value1

    edgel_first_is_side0 = cell[y0, x0] == first[arc]
    first_edgel_mass = np.where(
        edgel_first_is_side0, side0_mass, side1_mass)
    second_edgel_mass = np.where(
        edgel_first_is_side0, side1_mass, side0_mass)
    first_edgel_numerator = np.where(
        edgel_first_is_side0, side0_numerator, side1_numerator)
    second_edgel_numerator = np.where(
        edgel_first_is_side0, side1_numerator, side0_numerator)
    first_mass = np.bincount(
        arc, weights=first_edgel_mass, minlength=arc_count)
    second_mass = np.bincount(
        arc, weights=second_edgel_mass, minlength=arc_count)
    first_scale = np.bincount(
        arc, weights=first_edgel_numerator, minlength=arc_count
    ) / np.maximum(first_mass, 1e-30)
    second_scale = np.bincount(
        arc, weights=second_edgel_numerator, minlength=arc_count
    ) / np.maximum(second_mass, 1e-30)
    first_distance = np.abs(arc_scale - first_scale)
    second_distance = np.abs(arc_scale - second_scale)
    first_match_margin = second_distance - first_distance

    first_coverage = (
        first_mass
        / np.maximum(arc_length * kernel_mass, 1e-30)
    )
    second_coverage = (
        second_mass
        / np.maximum(arc_length * kernel_mass, 1e-30)
    )
    boundary_coverage = arc_mass / np.maximum(arc_length, 1.0)

    def normalized_coverage(value: np.ndarray) -> np.ndarray:
        positive = value[value > 0.0]
        scale = (
            float(np.percentile(positive, 80.0))
            if positive.size else 1.0
        )
        return np.clip(value / max(scale, 1e-30), 0.0, 1.0)

    reliability = np.cbrt(
        normalized_coverage(first_coverage)
        * normalized_coverage(second_coverage)
        * normalized_coverage(boundary_coverage)
    )
    return {
        "node_log_effective_scale": node_scale,
        "node_evidence_mass": node_mass,
        "node_evidence_coverage": node_coverage,
        "arc_log_effective_scale": arc_scale,
        "arc_evidence_mass": arc_mass,
        "arc_first_log_effective_scale": first_scale,
        "arc_second_log_effective_scale": second_scale,
        "arc_first_evidence_coverage": first_coverage,
        "arc_second_evidence_coverage": second_coverage,
        "first_distance": first_distance,
        "second_distance": second_distance,
        "first_match_margin": first_match_margin,
        "reliability": reliability,
    }
