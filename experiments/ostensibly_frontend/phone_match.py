"""Frequency-anchored Eikonal fingerprints and top-k template matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from experiments.pdf_optimizer.src.pdf_optimizer.glyph_residual import (
    fourier_circle_descriptor,
)
from experiments.ostensibly_frontend.trace_geometry import ridge_score


@dataclass(frozen=True)
class PhoneFingerprint:
    anchor_row: int
    duration_frames: int
    radial: np.ndarray
    spatial: np.ndarray
    intensity: np.ndarray
    row_profile: np.ndarray


@dataclass(frozen=True)
class WholePatchFingerprint:
    """Complete registered phone raster plus a harmonic routing landmark."""

    anchor_row: int
    source_centroid_row: float
    source_centroid_frame: float
    centroid_row: float
    centroid_frame: float
    duration_frames: int
    harmonicity: float
    field: np.ndarray


def _resize(values: np.ndarray, shape: tuple[int, int], order: int) -> np.ndarray:
    zoom = (shape[0] / values.shape[0], shape[1] / values.shape[1])
    resized = ndi.zoom(values, zoom, order=order, mode="nearest", prefilter=order > 1)
    return np.asarray(resized[: shape[0], : shape[1]], dtype=np.float64)


def lowest_persistent_anchor(score: np.ndarray) -> int:
    """Find the lowest non-DC ridge with persistence and prominence evidence."""
    if score.ndim != 2 or not score.size:
        raise ValueError("anchor score must be a nonempty rows-by-frames array")
    profile = np.percentile(score, 70.0, axis=1)
    profile[:3] = 0.0
    smooth = ndi.gaussian_filter1d(profile, sigma=1.0, mode="nearest")
    peak = float(np.max(smooth))
    if peak <= 0.0:
        return 0
    peaks, properties = find_peaks(
        smooth,
        distance=3,
        prominence=max(0.05 * peak, 1e-12),
        height=0.12 * peak,
    )
    if peaks.size:
        return int(peaks[0])
    return int(np.argmax(smooth))


def fingerprint_phone_patch(
    cartoon_patch: np.ndarray,
    *,
    band_rows: int = 128,
    canonical_shape: tuple[int, int] = (64, 32),
) -> PhoneFingerprint:
    """Build an offset-normalized multi-ridge phone fingerprint."""
    source = np.asarray(cartoon_patch, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError("phone patch must be rows-by-frames")
    score = ridge_score(source)
    anchor = lowest_persistent_anchor(score)
    row0 = max(anchor - 3, 0)
    row1 = min(row0 + band_rows, source.shape[0])
    band = score[row0:row1]
    if band.shape[0] < band_rows:
        band = np.pad(band, ((0, band_rows - band.shape[0]), (0, 0)))
    intensity = _resize(band, canonical_shape, order=1)
    intensity -= float(np.min(intensity))
    intensity /= max(float(np.linalg.norm(intensity)), 1e-30)
    row_profile = np.sqrt(np.mean(intensity * intensity, axis=1))
    row_profile = ndi.gaussian_filter1d(row_profile, sigma=2.0, mode="nearest")
    row_profile /= max(float(np.linalg.norm(row_profile)), 1e-30)
    positive = intensity[intensity > 0.0]
    threshold = float(np.percentile(positive, 70.0)) if positive.size else 0.0
    support = intensity > threshold
    if not support.any():
        support[0, 0] = True
    radial, spatial = fourier_circle_descriptor(
        support,
        size=32,
        bins=10,
    )
    return PhoneFingerprint(
        anchor_row=anchor,
        duration_frames=int(source.shape[1]),
        radial=radial,
        spatial=spatial,
        intensity=intensity.ravel(),
        row_profile=row_profile,
    )


def registered_row_profile_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Best ordered-profile cosine distance over modest tract affine maps."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("row profiles must be equal-length vectors")
    coordinates = np.arange(first.size, dtype=np.float64)
    best = float("inf")
    for scale in np.linspace(0.75, 1.35, 13):
        warped = np.interp(
            coordinates / scale,
            coordinates,
            second,
            left=0.0,
            right=0.0,
        )
        for shift in range(-4, 5):
            candidate = ndi.shift(
                warped,
                shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            candidate /= max(float(np.linalg.norm(candidate)), 1e-30)
            best = min(best, 1.0 - float(np.dot(first, candidate)))
    return best


def phone_distance(left: PhoneFingerprint, right: PhoneFingerprint) -> float:
    """Compare registered geometry; absolute frequency offset is excluded."""
    radial = float(np.linalg.norm(left.radial - right.radial))
    spatial = float(np.sqrt(np.mean((left.spatial - right.spatial) ** 2)))
    intensity = float(np.sqrt(np.mean((left.intensity - right.intensity) ** 2)))
    duration = abs(np.log(
        max(left.duration_frames, 1) / max(right.duration_frames, 1)
    ))
    ordered_profile = registered_row_profile_distance(
        left.row_profile, right.row_profile
    )
    return (
        radial
        + 2.0 * spatial
        + 1.5 * intensity
        + 0.12 * duration
        + 4.0 * ordered_profile
    )


def rank_phone_templates(
    query: PhoneFingerprint,
    templates: dict[str, tuple[PhoneFingerprint, ...]],
    *,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Rank phone classes by their best synthetic-voice witness."""
    scored = [
        (label, min(phone_distance(query, reference) for reference in references))
        for label, references in templates.items()
        if references
    ]
    return sorted(scored, key=lambda item: (item[1], item[0]))[:top_k]


def fingerprint_whole_phone_patch(
    fused_patch: np.ndarray,
    *,
    canonical_shape: tuple[int, int] = (128, 24),
) -> WholePatchFingerprint:
    """Locate harmonics, but retain every fused pixel for terminal matching.

    Ridge evidence supplies only the anchor and centroid used by registration.
    The stored field is the complete positive raster, including background and
    nuisance texture.  It is linearly resized in time/row and L2-normalized;
    no cartoon, mask, band selection, or evidence-channel concatenation occurs.
    """
    source = np.maximum(np.asarray(fused_patch, dtype=np.float64), 0.0)
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError("phone patch must be rows-by-frames")
    harmonic = np.maximum(ridge_score(source), 0.0)
    anchor = lowest_persistent_anchor(harmonic)
    threshold = float(np.quantile(harmonic, 0.75))
    landmark_weight = np.maximum(harmonic - threshold, 0.0)
    if float(np.sum(landmark_weight)) <= 1e-30:
        landmark_weight = source
    yy, xx = np.mgrid[: source.shape[0], : source.shape[1]]
    mass = max(float(np.sum(landmark_weight)), 1e-30)
    centroid_row = float(np.sum(landmark_weight * yy) / mass)
    centroid_frame = float(np.sum(landmark_weight * xx) / mass)
    total_energy = max(float(np.sum(source * source)), 1e-30)
    harmonicity = float(np.sum(landmark_weight * landmark_weight) / total_energy)

    # Translate in a doubled canvas before reduction so centroid alignment
    # never achieves invariance by clipping away the opposite side of the
    # observed band.  Every original pixel remains represented.
    canvas = np.zeros(
        (2 * source.shape[0], 2 * source.shape[1]), dtype=np.float64)
    canvas[: source.shape[0], : source.shape[1]] = source
    centered = ndi.shift(
        canvas,
        (source.shape[0] - centroid_row, source.shape[1] - centroid_frame),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    field = _resize(centered, canonical_shape, order=1)
    field /= max(float(np.linalg.norm(field)), 1e-30)
    return WholePatchFingerprint(
        anchor_row=anchor,
        source_centroid_row=centroid_row,
        source_centroid_frame=centroid_frame,
        centroid_row=0.5 * (canonical_shape[0] - 1),
        centroid_frame=0.5 * (canonical_shape[1] - 1),
        duration_frames=int(source.shape[1]),
        harmonicity=harmonicity,
        field=field,
    )


def _warp_about_centroids(
    moving: WholePatchFingerprint,
    fixed: WholePatchFingerprint,
    *,
    row_scale: float,
    time_scale: float,
) -> np.ndarray:
    target = fixed.field
    yy, xx = np.mgrid[: target.shape[0], : target.shape[1]]
    source_y = (
        (yy - fixed.centroid_row) / row_scale + moving.centroid_row
    )
    source_x = (
        (xx - fixed.centroid_frame) / time_scale + moving.centroid_frame
    )
    return ndi.map_coordinates(
        moving.field,
        (source_y, source_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def whole_patch_distance(
    left: WholePatchFingerprint,
    right: WholePatchFingerprint,
) -> float:
    """One terminal geometry: best full-raster L2 after centroid registration."""
    if left.field.shape != right.field.shape:
        raise ValueError("whole-patch fields must share one canonical raster")
    best = float("inf")
    # Vocal-tract and phone-duration nuisance are coordinates of the same
    # affine registration, not separately scored evidence channels.
    for row_scale in np.linspace(0.80, 1.25, 10):
        for time_scale in np.linspace(0.84, 1.20, 7):
            candidate = _warp_about_centroids(
                right,
                left,
                row_scale=float(row_scale),
                time_scale=float(time_scale),
            )
            candidate /= max(float(np.linalg.norm(candidate)), 1e-30)
            distance = float(np.sqrt(np.mean(
                (left.field - candidate) ** 2)))
            best = min(best, distance)
    return best


def rank_whole_patch_templates(
    query: WholePatchFingerprint,
    templates: dict[str, tuple[WholePatchFingerprint, ...]],
    *,
    candidate_labels: set[str] | None = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Rank a routed candidate set with only whole-patch registered geometry."""
    labels = set(templates) if candidate_labels is None else candidate_labels
    scored = [
        (
            label,
            min(whole_patch_distance(query, reference)
                for reference in references),
        )
        for label, references in templates.items()
        if label in labels and references
    ]
    return sorted(scored, key=lambda item: (item[1], item[0]))[:top_k]
