"""Continuous morphology and lifted-extrema geometry for the step-3 trace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DepthmapGeometryConfig:
    """Small, explicit geometry controls in native step-3 pixels."""

    opening_sizes: tuple[tuple[int, int], ...] = ((3, 3), (5, 3), (9, 5))
    seed_count: int = 4096
    random_seed: int = 7341
    row_distance_scale: float = 4.0
    connection_radius: float = 5.0
    neighbors_per_extremum: int = 3
    path_support_ratio: float = 0.35
    backfill_spacing: float = 0.5

    def __post_init__(self) -> None:
        if not self.opening_sizes or any(
            rows < 1 or frames < 1 for rows, frames in self.opening_sizes
        ):
            raise ValueError("opening sizes must be positive")
        if self.seed_count < 1 or self.row_distance_scale <= 0.0:
            raise ValueError("invalid seed geometry")
        if self.connection_radius <= 0.0 or self.neighbors_per_extremum < 1:
            raise ValueError("invalid proximity geometry")
        if not 0.0 <= self.path_support_ratio <= 1.0:
            raise ValueError("invalid continuous path-support ratio")
        if self.backfill_spacing <= 0.0:
            raise ValueError("invalid backfill spacing")


@dataclass(frozen=True)
class LiftedGeometry:
    """Extrema, supported connections, and the resulting lifted point cloud."""

    extrema: np.ndarray
    hit_counts: np.ndarray
    segments: np.ndarray
    points: np.ndarray


def multiscale_morphological_residual(
    depthmap: np.ndarray,
    config: DepthmapGeometryConfig = DepthmapGeometryConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Remove broad bright mass continuously with grayscale white top-hats.

    No binary support is formed.  Each scale subtracts a grayscale opening;
    the median retains relief that recurs across feature sizes.
    """

    source = np.asarray(depthmap, dtype=np.float64)
    if source.ndim != 2 or not source.size or not np.all(np.isfinite(source)):
        raise ValueError("depthmap must be one finite, nonempty 2-D field")
    source = np.maximum(source, 0.0)
    residuals = []
    for size in config.opening_sizes:
        opened = ndi.grey_opening(source, size=size, mode="nearest")
        residuals.append(np.maximum(source - opened, 0.0))
    scale_stack = np.stack(residuals)
    return np.median(scale_stack, axis=0), scale_stack


def _climb_one(field: np.ndarray, row: int, frame: int) -> tuple[int, int]:
    """Move one seed monotonically to an eight-neighbor local maximum."""

    rows, frames = field.shape
    while True:
        row0, row1 = max(0, row - 1), min(rows, row + 2)
        frame0, frame1 = max(0, frame - 1), min(frames, frame + 2)
        patch = field[row0:row1, frame0:frame1]
        best_value = float(np.max(patch))
        if best_value <= float(field[row, frame]):
            return row, frame
        best = np.argwhere(patch == best_value)
        # A stable lexicographic choice makes the lift reproducible on plateaus.
        next_row, next_frame = best[0]
        row, frame = row0 + int(next_row), frame0 + int(next_frame)


def lift_random_seeds_to_extrema(
    residual: np.ndarray,
    config: DepthmapGeometryConfig = DepthmapGeometryConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the continuous relief densely and lift every seed uphill."""

    field = np.asarray(residual, dtype=np.float64)
    if field.ndim != 2 or not field.size or not np.all(np.isfinite(field)):
        raise ValueError("residual must be one finite, nonempty 2-D field")
    weights = np.maximum(field, 0.0).ravel()
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(config.random_seed)
    seeds = rng.choice(
        weights.size,
        size=config.seed_count,
        replace=True,
        p=weights / total,
    )
    lifted = np.empty((config.seed_count, 2), dtype=np.int64)
    for index, flat in enumerate(seeds):
        lifted[index] = _climb_one(field, *np.unravel_index(int(flat), field.shape))
    positions, counts = np.unique(lifted, axis=0, return_counts=True)
    peak = max(float(np.percentile(field[field > 0.0], 99.5)), 1e-30)
    heights = np.clip(field[positions[:, 0], positions[:, 1]] / peak, 0.0, 1.0)
    extrema = np.column_stack((positions.astype(np.float64), heights))
    return extrema, counts.astype(np.int64)


def _sample_line(
    field: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance = float(np.linalg.norm(end[:2] - start[:2]))
    count = max(2, int(np.ceil(distance / spacing)) + 1)
    alpha = np.linspace(0.0, 1.0, count)
    coordinates = start[None, :2] * (1.0 - alpha[:, None]) + end[None, :2] * alpha[:, None]
    values = ndi.map_coordinates(
        field,
        (coordinates[:, 0], coordinates[:, 1]),
        order=1,
        mode="nearest",
    )
    return coordinates, values


def backfill_extrema_geometry(
    residual: np.ndarray,
    extrema: np.ndarray,
    hit_counts: np.ndarray,
    config: DepthmapGeometryConfig = DepthmapGeometryConfig(),
) -> LiftedGeometry:
    """Connect nearby extrema only when continuous relief supports the path."""

    field = np.asarray(residual, dtype=np.float64)
    peaks = np.asarray(extrema, dtype=np.float64)
    hits = np.asarray(hit_counts, dtype=np.int64)
    if peaks.ndim != 2 or peaks.shape[1] != 3 or hits.shape != (peaks.shape[0],):
        raise ValueError("invalid extrema geometry")
    if not peaks.size:
        return LiftedGeometry(peaks, hits, np.empty((0, 4)), np.empty((0, 3)))

    metric_points = np.column_stack(
        (peaks[:, 0] / config.row_distance_scale, peaks[:, 1])
    )
    tree = cKDTree(metric_points)
    candidate_pairs = sorted(tree.query_pairs(config.connection_radius))
    by_source: dict[int, list[tuple[float, int]]] = {}
    for left, right in candidate_pairs:
        distance = float(np.linalg.norm(metric_points[left] - metric_points[right]))
        by_source.setdefault(left, []).append((distance, right))
        by_source.setdefault(right, []).append((distance, left))

    peak_gauge = max(float(np.percentile(field[field > 0.0], 99.5)), 1e-30)
    accepted_segments: list[tuple[float, float, float, float]] = []
    point_chunks = [peaks]
    accepted: set[tuple[int, int]] = set()
    for left, neighbors in by_source.items():
        for _distance, right in sorted(neighbors)[: config.neighbors_per_extremum]:
            pair = (min(left, right), max(left, right))
            if pair in accepted:
                continue
            coordinates, values = _sample_line(
                field, peaks[left], peaks[right], config.backfill_spacing
            )
            endpoint_floor = min(
                float(field[int(peaks[left, 0]), int(peaks[left, 1])]),
                float(field[int(peaks[right, 0]), int(peaks[right, 1])]),
            )
            if float(np.mean(values)) < config.path_support_ratio * endpoint_floor:
                continue
            accepted.add(pair)
            accepted_segments.append(
                (peaks[left, 0], peaks[left, 1], peaks[right, 0], peaks[right, 1])
            )
            heights = np.clip(values / peak_gauge, 0.0, 1.0)
            point_chunks.append(np.column_stack((coordinates, heights)))

    segments = np.asarray(accepted_segments, dtype=np.float64).reshape(-1, 4)
    points = np.concatenate(point_chunks, axis=0)
    # The continuous coordinates are quantized only for exact duplicate removal.
    quantized = np.round(points[:, :2] * 2.0).astype(np.int64)
    _, unique = np.unique(quantized, axis=0, return_index=True)
    points = points[np.sort(unique)]
    return LiftedGeometry(peaks, hits, segments, points)


def depthmap_geometry(
    depthmap: np.ndarray,
    config: DepthmapGeometryConfig = DepthmapGeometryConfig(),
) -> tuple[np.ndarray, np.ndarray, LiftedGeometry]:
    """Run the complete continuous residual, lift, and backfill construction."""

    residual, scale_stack = multiscale_morphological_residual(depthmap, config)
    extrema, hit_counts = lift_random_seeds_to_extrema(residual, config)
    geometry = backfill_extrema_geometry(residual, extrema, hit_counts, config)
    return residual, scale_stack, geometry
