"""One-shot frozen-Meyer-density Voronoi amplitude decomposition.

The operator has exactly three image-scale stages:

1. two fixed stages of the shipped C++ Meyer kernel measure a bounded
   inverse-support tensor;
2. ``sqrt(det(Q)) / pi`` emits all germs simultaneously and one analytic
   curvature law corrects supports that cannot remain locally straight;
3. one anisotropic first-arrival solve produces every Voronoi cell.

There is no extrema spacing, candidate search, birth loop, Lloyd motion,
support diffusion, or convergence test.  Conditional on the frozen cells,
the baseline is a bounded convex map of cell amplitudes and the rotation is
its exact additive complement.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from port_needed.continuous_eikonal_transport import (
    continuous_first_partition_prepared,
    prepare_continuous_metric,
)
from port_needed.density_population import (
    curvature_limited_geometry,
    emit_density_population,
)
from port_needed.frozen_meyer_geometry import (
    build_frozen_geometry,
    restrict_geometry,
)
from port_needed.wide_stencil_transport import _metric_fields


@dataclass(frozen=True)
class VoronoiITDConfig:
    """Controls that change measured geometry, never allocation iterations."""

    levels: int = 1
    alpha: float = 0.5
    tgfd_sweeps: int = 2
    flow_sweeps: int = 2
    null_evidence_strength: float = 1.0
    coherent_tangent_fraction: float = 0.08
    texture_support_weight: float = 0.65
    curvature_limited_density: bool = True
    metric_strength: float = 1.5
    boundary_jump_strength: float = 48.0
    allocation_max_side: int = 256
    safety_cells: int = 32768


@dataclass
class IntrinsicVoronoiSupport:
    """Frozen support geometry with no amplitude reconstruction attached."""

    centers_xy: np.ndarray
    sites_yx: np.ndarray
    owner: np.ndarray
    support_confidence: np.ndarray
    delaunay_edges: np.ndarray
    eikonal_distance: np.ndarray
    support_measure: np.ndarray
    geometry_milliseconds: float
    allocation_milliseconds: float


@dataclass
class VoronoiITDLevel:
    """The single frozen partition and its bounded amplitude readout."""

    source: np.ndarray
    rotation: np.ndarray
    baseline: np.ndarray
    sites_yx: np.ndarray
    site_values: np.ndarray
    knot_values: np.ndarray
    polarity: np.ndarray
    owner: np.ndarray
    support_confidence: np.ndarray
    delaunay_edges: np.ndarray
    partition_error: float
    eikonal_distance: np.ndarray
    support_measure: np.ndarray
    geometry_milliseconds: float
    allocation_milliseconds: float


@dataclass
class VoronoiITDResult:
    """One rotation plus its residual baseline."""

    levels: list[VoronoiITDLevel]
    residual: np.ndarray

    @property
    def rotations(self) -> np.ndarray:
        if not self.levels:
            return np.empty((0,) + self.residual.shape, dtype=np.float64)
        return np.stack([level.rotation for level in self.levels])

    @property
    def reconstruction(self) -> np.ndarray:
        if not self.levels:
            return self.residual.copy()
        return self.residual + self.levels[0].rotation


def _validate_image(image: np.ndarray) -> np.ndarray:
    field = np.ascontiguousarray(image, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("Voronoi amplitude decomposition expects a 2-D field")
    if min(field.shape) < 2:
        raise ValueError("both image dimensions must be at least two")
    if not np.all(np.isfinite(field)):
        raise ValueError("image contains non-finite samples")
    return field


def _guidance_rgb(
    field: np.ndarray,
    guidance: np.ndarray | None,
) -> np.ndarray:
    if guidance is None:
        rgb = np.repeat(field[..., None], 3, axis=2)
    else:
        rgb = np.asarray(guidance, dtype=np.float64)
        if rgb.shape[:2] != field.shape or rgb.ndim != 3:
            raise ValueError("Meyer guidance must have shape HxWxC")
        rgb = (
            np.repeat(rgb[..., :1], 3, axis=2)
            if rgb.shape[2] < 3
            else rgb[..., :3]
        )
    if not np.all(np.isfinite(rgb)):
        raise ValueError("guidance contains non-finite samples")
    return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0))


def _prolong_nearest(
    field: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Sample a restricted allocation field at full pixel centres."""

    if field.shape == shape:
        return field
    source_height, source_width = field.shape
    target_height, target_width = shape
    sample_y = np.clip(
        ((np.arange(target_height) + 0.5)
         * source_height / target_height).astype(np.int32),
        0,
        source_height - 1,
    )
    sample_x = np.clip(
        ((np.arange(target_width) + 0.5)
         * source_width / target_width).astype(np.int32),
        0,
        source_width - 1,
    )
    return field[sample_y[:, None], sample_x[None, :]]


def eikonal_adjacency(owner: np.ndarray) -> np.ndarray:
    """Return exactly the site pairs sharing a hard cell interface."""

    labels = np.asarray(owner, dtype=np.int32)
    vertical = np.column_stack((
        labels[:-1, :].ravel(), labels[1:, :].ravel()))
    horizontal = np.column_stack((
        labels[:, :-1].ravel(), labels[:, 1:].ravel()))
    candidates = np.vstack((vertical, horizontal))
    candidates = candidates[candidates[:, 0] != candidates[:, 1]]
    if not len(candidates):
        return np.empty((0, 2), dtype=np.int32)
    candidates.sort(axis=1)
    return np.unique(candidates, axis=0).astype(np.int32)


def update_knots(
    site_values: np.ndarray,
    centers_xy: np.ndarray,
    edges: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """One bounded convex average over the measured interface graph."""

    values = np.asarray(site_values, dtype=np.float64)
    points = np.asarray(centers_xy, dtype=np.float64)
    count = len(values)
    prediction = values.copy()
    if len(edges):
        first = edges[:, 0].astype(np.intp)
        second = edges[:, 1].astype(np.intp)
        distance = np.linalg.norm(
            points[first] - points[second], axis=1)
        weight = 1.0 / np.maximum(distance, 1e-12)
        numerator = (
            np.bincount(
                first,
                weights=weight * values[second],
                minlength=count,
            )
            + np.bincount(
                second,
                weights=weight * values[first],
                minlength=count,
            )
        )
        denominator = (
            np.bincount(first, weights=weight, minlength=count)
            + np.bincount(second, weights=weight, minlength=count)
        )
        valid = denominator > 0.0
        prediction[valid] = numerator[valid] / denominator[valid]
    blend = np.clip(float(alpha), 0.0, 1.0)
    return (1.0 - blend) * values + blend * prediction


def extract_intrinsic_voronoi_support(
    image: np.ndarray,
    config: VoronoiITDConfig = VoronoiITDConfig(),
    guidance: np.ndarray | None = None,
) -> IntrinsicVoronoiSupport:
    """Measure, populate, and march one frozen support partition only."""

    field = _validate_image(image)
    rgb = _guidance_rgb(field, guidance)

    started = time.perf_counter()
    geometry = build_frozen_geometry(
        rgb,
        tgfd_sweeps=max(int(config.tgfd_sweeps), 1),
        flow_sweeps=max(int(config.flow_sweeps), 1),
        null_evidence_strength=config.null_evidence_strength,
        coherent_tangent_fraction=config.coherent_tangent_fraction,
        texture_support_weight=config.texture_support_weight,
    )
    if config.curvature_limited_density:
        geometry = curvature_limited_geometry(geometry)
    geometry_ms = 1000.0 * (time.perf_counter() - started)

    allocation_started = time.perf_counter()
    allocation_geometry = restrict_geometry(
        geometry, max(int(config.allocation_max_side), 2))
    centers, _ = emit_density_population(
        allocation_geometry,
        safety_cells=max(int(config.safety_cells), 1),
    )
    metric = prepare_continuous_metric(*_metric_fields(
        allocation_geometry,
        config.metric_strength,
        config.boundary_jump_strength,
    ))
    partition = continuous_first_partition_prepared(centers, metric)
    owner = _prolong_nearest(
        np.asarray(partition["labels"], dtype=np.int32), field.shape)
    distance = _prolong_nearest(
        np.asarray(partition["distance"], dtype=np.float64), field.shape)
    allocation_ms = 1000.0 * (
        time.perf_counter() - allocation_started)
    edges = eikonal_adjacency(owner)

    height, width = field.shape
    sites_yx = np.column_stack((
        np.clip(
            np.rint(centers[:, 1] * height - 0.5),
            0,
            height - 1,
        ),
        np.clip(
            np.rint(centers[:, 0] * width - 0.5),
            0,
            width - 1,
        ),
    )).astype(np.int32)
    positive = distance[np.isfinite(distance) & (distance > 0.0)]
    scale = (
        max(float(np.median(positive)), 1e-12)
        if positive.size
        else 1.0
    )
    confidence = 1.0 / (1.0 + distance / scale)
    return IntrinsicVoronoiSupport(
        centers_xy=np.asarray(centers, dtype=np.float64),
        sites_yx=sites_yx,
        owner=owner,
        support_confidence=confidence,
        delaunay_edges=edges,
        eikonal_distance=distance,
        support_measure=np.asarray(
            geometry["measure"], dtype=np.float64),
        geometry_milliseconds=geometry_ms,
        allocation_milliseconds=allocation_ms,
    )


def extract_voronoi_baseline(
    image: np.ndarray,
    config: VoronoiITDConfig = VoronoiITDConfig(),
    guidance: np.ndarray | None = None,
) -> VoronoiITDLevel:
    """Attach the bounded amplitude readout to one frozen support."""

    field = _validate_image(image)
    support = extract_intrinsic_voronoi_support(
        field, config, guidance)
    cell_total = len(support.centers_xy)
    flat_owner = support.owner.ravel()
    cell_count = np.maximum(
        np.bincount(flat_owner, minlength=cell_total), 1)
    site_values = np.bincount(
        flat_owner,
        weights=field.ravel(),
        minlength=cell_total,
    ) / cell_count
    knots = update_knots(
        site_values,
        support.centers_xy,
        support.delaunay_edges,
        config.alpha,
    )
    baseline = knots[support.owner]
    return VoronoiITDLevel(
        source=field,
        rotation=np.ascontiguousarray(field - baseline),
        baseline=np.ascontiguousarray(baseline),
        sites_yx=support.sites_yx,
        site_values=site_values,
        knot_values=knots,
        # These are tensor-density germs, not extrema.  The value only gives
        # the viewer a distinct diagnostic colour.
        polarity=np.full(cell_total, 3, dtype=np.int8),
        owner=support.owner,
        support_confidence=support.support_confidence,
        delaunay_edges=support.delaunay_edges,
        partition_error=0.0,
        eikonal_distance=support.eikonal_distance,
        support_measure=support.support_measure,
        geometry_milliseconds=support.geometry_milliseconds,
        allocation_milliseconds=support.allocation_milliseconds,
    )


def voronoi_itd(
    image: np.ndarray,
    config: VoronoiITDConfig = VoronoiITDConfig(),
    guidance: np.ndarray | None = None,
) -> VoronoiITDResult:
    """Return at most one frozen baseline; ``levels`` is a compatibility cap."""

    field = _validate_image(image)
    if int(config.levels) <= 0 or float(np.ptp(field)) <= 1e-14:
        return VoronoiITDResult(levels=[], residual=field.copy())
    level = extract_voronoi_baseline(field, config, guidance)
    return VoronoiITDResult(levels=[level], residual=level.baseline)


def support_boundary(owner: np.ndarray) -> np.ndarray:
    """Return the one-pixel boundary of a hard ownership image."""

    labels = np.asarray(owner)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def level_statistics(level: VoronoiITDLevel) -> dict[str, float | int]:
    """Diagnostics used by the viewer and tests."""

    source_energy = float(np.sum(level.source * level.source))
    rotation_energy = float(np.sum(level.rotation * level.rotation))
    return {
        "sites": int(len(level.sites_yx)),
        "extrema": 0,
        "delaunay_edges": int(len(level.delaunay_edges)),
        "rotation_energy_fraction": (
            rotation_energy / max(source_energy, 1e-30)),
        "partition_error": float(level.partition_error),
        "support_confidence_mean": float(
            np.mean(level.support_confidence)),
        "geometry_ms": float(level.geometry_milliseconds),
        "allocation_ms": float(level.allocation_milliseconds),
    }
