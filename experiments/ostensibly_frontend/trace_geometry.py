"""Geometry extraction and Eikonal descriptors for Meyer speech cartoons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from experiments.pdf_optimizer.src.pdf_optimizer.glyph_residual import (
    fourier_circle_descriptor,
)


@dataclass(frozen=True)
class TraceGeometryConfig:
    opening_rows: int = 15
    opening_frames: int = 3
    score_percentile: float = 90.0
    closing_frames: int = 3
    minimum_area: int = 12
    minimum_frame_span: int = 4
    descriptor_size: int = 32
    radial_bins: int = 10


@dataclass(frozen=True)
class TraceComponent:
    label: int
    row0: int
    frame0: int
    row1: int
    frame1: int
    area: int
    centroid_row: float
    centroid_frame: float
    radial: np.ndarray
    spatial: np.ndarray

    @property
    def row_span(self) -> int:
        return self.row1 - self.row0

    @property
    def frame_span(self) -> int:
        return self.frame1 - self.frame0


def ridge_score(
    cartoon: np.ndarray,
    config: TraceGeometryConfig = TraceGeometryConfig(),
) -> np.ndarray:
    """Positive vertical top-hat response of a speech-cartoon field."""
    source = np.asarray(cartoon, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("ridge_score expects a rows-by-frames array")
    opened = ndi.grey_opening(
        source,
        size=(config.opening_rows, config.opening_frames),
    )
    return np.maximum(source - opened, 0.0)


def extract_trace_components(
    cartoon: np.ndarray,
    config: TraceGeometryConfig = TraceGeometryConfig(),
) -> tuple[np.ndarray, np.ndarray, tuple[TraceComponent, ...]]:
    """Threshold, connect, and describe persistent cartoon ridges.

    Labels are compact and contain only components passing the declared area
    and time-span gates.  Descriptors are the signed-distance/Fourier-circle
    pair used by the glyph residual project.
    """
    score = ridge_score(cartoon, config)
    positive = score[score > 0.0]
    if positive.size == 0:
        return score, np.zeros(score.shape, dtype=np.int32), ()
    threshold = float(np.percentile(positive, config.score_percentile))
    support = score >= threshold
    if config.closing_frames > 1:
        support = ndi.binary_closing(
            support,
            structure=np.ones((1, config.closing_frames), dtype=bool),
        )
    raw_labels, count = ndi.label(
        support, structure=np.ones((3, 3), dtype=np.uint8)
    )
    objects = ndi.find_objects(raw_labels, max_label=count)
    labels = np.zeros(raw_labels.shape, dtype=np.int32)
    components: list[TraceComponent] = []
    for raw_label, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        rows, frames = bounds
        patch = raw_labels[rows, frames] == raw_label
        area = int(np.count_nonzero(patch))
        frame_span = int(frames.stop - frames.start)
        if area < config.minimum_area or frame_span < config.minimum_frame_span:
            continue
        compact_label = len(components) + 1
        local_labels = labels[rows, frames]
        local_labels[patch] = compact_label
        centroid = ndi.center_of_mass(patch)
        radial, spatial = fourier_circle_descriptor(
            patch,
            size=config.descriptor_size,
            bins=config.radial_bins,
        )
        components.append(
            TraceComponent(
                label=compact_label,
                row0=int(rows.start),
                frame0=int(frames.start),
                row1=int(rows.stop),
                frame1=int(frames.stop),
                area=area,
                centroid_row=float(rows.start + centroid[0]),
                centroid_frame=float(frames.start + centroid[1]),
                radial=radial,
                spatial=spatial,
            )
        )
    return score, labels, tuple(components)


def descriptor_distance(left: TraceComponent, right: TraceComponent) -> float:
    """Glyph-project registration distance with an explicit aspect penalty."""
    radial = float(np.linalg.norm(left.radial - right.radial))
    spatial = float(np.sqrt(np.mean((left.spatial - right.spatial) ** 2)))
    aspect_left = left.frame_span / max(left.row_span, 1)
    aspect_right = right.frame_span / max(right.row_span, 1)
    aspect = abs(np.log(max(aspect_left, 1e-12) / max(aspect_right, 1e-12)))
    return radial + 2.0 * spatial + 0.25 * aspect
