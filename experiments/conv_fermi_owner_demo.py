"""Experimental V3-owner Fermi-chart extension for the image resize demo.

This module intentionally lives outside the copy-isolated standalone demo.
It measures V3 structural owners, gives each owner the straight Fermi chart
induced by its variation-support inertia, reconstructs one scalar potential
in that chart with CONV, and samples the potential back on the target grid.
The normal and companion currents are therefore derived from the same scalar
surface and satisfy the commuting-current law identically.

Curved owners are not flattened into one global direction: the V3 population
already subdivides support where its straight chart loses validity.  A future
native implementation can replace the chart rasterization without changing
the coordinate or current laws tested here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import (  # noqa: E402
    _basin_average_axis,
    conv_evaluate_lines,
    conv_evaluate_profile,
    conv_resize,
)
from bfft.effects import srgb_to_lab  # noqa: E402
from experiments.segmenting_v3 import (  # noqa: E402
    SegmentingV3Config,
    _half_cartoon_scaffold,
    _resize,
)
from port_needed.continuous_eikonal_transport import (  # noqa: E402
    continuous_first_partition_prepared,
    prepare_continuous_metric,
)
from port_needed.wide_stencil_transport import _metric_fields  # noqa: E402
from experiments.self_geometric_harmonic_interpolation import (  # noqa: E402
    _local_fourth_order_2jet,
)


Array = np.ndarray

DISTANCE_RESPONSES = (
    "inverse_power_2",
    "softmax",
    "scaled_softmax",
    "cauchy_kernel",
    "normalized_softplus",
)


@dataclass(frozen=True)
class OwnerFrame:
    owner: int
    population: int
    center_x: float
    center_y: float
    normal_x: float
    normal_y: float
    tangent_x: float
    tangent_y: float
    transport_compatibility: float
    minimum_phi: float
    maximum_phi: float
    minimum_psi: float
    maximum_psi: float


@dataclass(frozen=True)
class OwnerAtlas:
    labels: Array
    frames: tuple[OwnerFrame, ...]
    source_shape: tuple[int, int]


@dataclass(frozen=True)
class DistanceAtlas:
    """Source-conditioned Eikonal actions with no winning-owner field."""

    distances: Array
    frames: tuple[OwnerFrame, ...]
    source_shape: tuple[int, int]


def _as_rgb(values: Array) -> Array:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        return np.repeat(field[..., None], 3, axis=2)
    if field.ndim != 3:
        raise ValueError("owner geometry requires an HxW or HxWxC raster")
    if field.shape[2] == 1:
        return np.repeat(field, 3, axis=2)
    return field[..., :3]


def _variation_jet(values: Array) -> tuple[Array, Array]:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        field = field[..., None]
    _, gx, _ = _local_fourth_order_2jet(np.swapaxes(field, 0, 1))
    gx = np.swapaxes(gx, 0, 1)
    _, gy, _ = _local_fourth_order_2jet(field)
    return gx, gy


def _variation_measure(values: Array) -> Array:
    gx, gy = _variation_jet(values)
    return np.sum(gx * gx + gy * gy, axis=2)


def _fix_frame_sign(normal: Array) -> Array:
    result = np.asarray(normal, dtype=np.float64)
    if result[0] < 0.0 or (result[0] == 0.0 and result[1] < 0.0):
        result = -result
    return result


def build_owner_atlas(values: Array) -> OwnerAtlas:
    """Measure V3 structural owners and their variation-support frames."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5:
        raise ValueError("Fermi-owner geometry requires an image at least 5x5")
    rgb = np.ascontiguousarray(np.clip(_as_rgb(field), 0.0, 1.0))
    config = SegmentingV3Config(
        owner_upgrade=False,
        cartoon_full_refit=False,
        texture_cleanup=False,
        structural_characteristic_passes=0,
        threads=4,
    )
    scaffold = _half_cartoon_scaffold(rgb, srgb_to_lab(rgb), config)
    labels = np.asarray(scaffold["labels"], dtype=np.int32)
    gx, gy = _variation_jet(field)
    measure = np.sum(gx * gx + gy * gy, axis=2)
    y, x = np.indices(labels.shape, dtype=np.float64)
    frames: list[OwnerFrame] = []
    for owner in range(int(np.max(labels)) + 1):
        selected = labels == owner
        population = int(np.count_nonzero(selected))
        if population == 0:
            continue
        weight = np.where(selected, measure, 0.0)
        total = float(np.sum(weight))
        if total <= np.finfo(np.float64).tiny:
            weight = selected.astype(np.float64)
            total = float(population)
        center_x = float(np.sum(weight * x) / total)
        center_y = float(np.sum(weight * y) / total)
        dx = x - center_x
        dy = y - center_y
        covariance = np.array((
            (float(np.sum(weight * dx * dx)), float(np.sum(weight * dx * dy))),
            (float(np.sum(weight * dx * dy)), float(np.sum(weight * dy * dy))),
        )) / total
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = _fix_frame_sign(np.array((-tangent[1], tangent[0])))
        tangent = np.array((-normal[1], normal[0]))
        jxx = float(np.sum(np.where(selected[..., None], gx * gx, 0.0)))
        jxy = float(np.sum(np.where(selected[..., None], gx * gy, 0.0)))
        jyy = float(np.sum(np.where(selected[..., None], gy * gy, 0.0)))
        current_eigenvalues, current_eigenvectors = np.linalg.eigh(
            np.array(((jxx, jxy), (jxy, jyy)))
        )
        current_normal = current_eigenvectors[:, 1]
        current_tangent = np.array((-current_normal[1], current_normal[0]))
        current_total = float(np.sum(current_eigenvalues))
        coherence = (
            float(current_eigenvalues[1] - current_eigenvalues[0]) / current_total
            if current_total > np.finfo(np.float64).tiny else 0.0
        )
        compatibility = coherence * float(np.dot(tangent, current_tangent) ** 2)
        owner_dx = dx[selected]
        owner_dy = dy[selected]
        phi = normal[0] * owner_dx + normal[1] * owner_dy
        psi = tangent[0] * owner_dx + tangent[1] * owner_dy
        frames.append(OwnerFrame(
            owner=owner,
            population=population,
            center_x=center_x,
            center_y=center_y,
            normal_x=float(normal[0]),
            normal_y=float(normal[1]),
            tangent_x=float(tangent[0]),
            tangent_y=float(tangent[1]),
            transport_compatibility=compatibility,
            minimum_phi=float(np.min(phi)),
            maximum_phi=float(np.max(phi)),
            minimum_psi=float(np.min(psi)),
            maximum_psi=float(np.max(psi)),
        ))
    return OwnerAtlas(labels, tuple(frames), labels.shape)


def build_distance_atlas(values: Array) -> DistanceAtlas:
    """Measure every germ's Eikonal action without retaining an argmin map."""

    field = np.asarray(values, dtype=np.float64)
    owner_atlas = build_owner_atlas(field)
    rgb = np.ascontiguousarray(np.clip(_as_rgb(field), 0.0, 1.0))
    config = SegmentingV3Config(
        owner_upgrade=False,
        cartoon_full_refit=False,
        texture_cleanup=False,
        structural_characteristic_passes=0,
        threads=4,
    )
    scaffold = _half_cartoon_scaffold(rgb, srgb_to_lab(rgb), config)
    low_metric = _metric_fields(
        scaffold["geometry"],
        config.metric_strength,
        config.boundary_jump_strength,
    )
    full_metric = tuple(
        _resize(component, owner_atlas.source_shape, order=1)
        for component in low_metric
    )
    prepared = prepare_continuous_metric(*full_metric)
    height, width = owner_atlas.source_shape
    actions = []
    for frame in owner_atlas.frames:
        center = np.array([[
            (frame.center_x + 0.5) / width,
            (frame.center_y + 0.5) / height,
        ]], dtype=np.float64)
        arrival = continuous_first_partition_prepared(
            center, prepared, compact=True
        )
        actions.append(np.asarray(arrival["distance"], dtype=np.float64))
    return DistanceAtlas(
        np.stack(actions, axis=0), owner_atlas.frames, owner_atlas.source_shape
    )


def build_direct_distance_atlas(
    values: Array,
    response: str = "inverse_power_2",
) -> DistanceAtlas:
    """Derive source frames from action weights, without an owner partition."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5:
        raise ValueError("direct Eikonal geometry requires an image at least 5x5")
    rgb = np.ascontiguousarray(np.clip(_as_rgb(field), 0.0, 1.0))
    config = SegmentingV3Config(
        owner_upgrade=False,
        cartoon_full_refit=False,
        texture_cleanup=False,
        structural_characteristic_passes=0,
        threads=4,
    )
    scaffold = _half_cartoon_scaffold(rgb, srgb_to_lab(rgb), config)
    centers = np.asarray(scaffold["centers"], dtype=np.float64)
    low_metric = _metric_fields(
        scaffold["geometry"],
        config.metric_strength,
        config.boundary_jump_strength,
    )
    prepared = prepare_continuous_metric(*tuple(
        _resize(component, field.shape[:2], order=1)
        for component in low_metric
    ))
    height, width = field.shape[:2]
    actions = []
    for center in centers:
        arrival = continuous_first_partition_prepared(
            np.asarray(center, dtype=np.float64)[None, :],
            prepared,
            compact=True,
        )
        actions.append(np.asarray(arrival["distance"], dtype=np.float64))
    distances = np.stack(actions, axis=0)

    # The normalized action response is the support measure itself.  Frames
    # therefore depend on no argmin labels or Voronoi-cell statistics.
    if len(centers) <= 1:
        scales = np.ones(len(centers), dtype=np.float64)
    else:
        center_x = centers[:, 0] * width - 0.5
        center_y = centers[:, 1] * height - 0.5
        scales = np.empty(len(centers), dtype=np.float64)
        for index, distance in enumerate(distances):
            values_at_centers = map_coordinates(
                distance, (center_y, center_x), order=1,
                mode="nearest", prefilter=False,
            )
            values_at_centers[index] = np.inf
            scales[index] = float(np.min(values_at_centers))
        positive = scales[np.isfinite(scales) & (scales > 0.0)]
        fallback = float(np.median(positive)) if positive.size else 1.0
        scales = np.where(
            np.isfinite(scales) & (scales > 0.0), scales, fallback
        )
    raw = np.stack([
        _distance_response(distance, scales[index], response)
        for index, distance in enumerate(distances)
    ], axis=0)
    support = raw / np.sum(raw, axis=0, keepdims=True)

    gx, gy = _variation_jet(field)
    measure = np.sum(gx * gx + gy * gy, axis=2)
    y, x = np.indices((height, width), dtype=np.float64)
    corners = np.array((
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (0.0, height - 1.0),
        (width - 1.0, height - 1.0),
    ))
    frames: list[OwnerFrame] = []
    for index, center in enumerate(centers):
        center_x = float(center[0] * width - 0.5)
        center_y = float(center[1] * height - 0.5)
        weight = support[index] * measure
        total = float(np.sum(weight))
        if total <= np.finfo(np.float64).tiny:
            weight = support[index]
            total = float(np.sum(weight))
        dx = x - center_x
        dy = y - center_y
        covariance = np.array((
            (float(np.sum(weight * dx * dx)), float(np.sum(weight * dx * dy))),
            (float(np.sum(weight * dx * dy)), float(np.sum(weight * dy * dy))),
        )) / total
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = _fix_frame_sign(np.array((-tangent[1], tangent[0])))
        tangent = np.array((-normal[1], normal[0]))
        action_weight = support[index][..., None]
        jxx = float(np.sum(action_weight * gx * gx))
        jxy = float(np.sum(action_weight * gx * gy))
        jyy = float(np.sum(action_weight * gy * gy))
        current_eigenvalues, current_eigenvectors = np.linalg.eigh(
            np.array(((jxx, jxy), (jxy, jyy)))
        )
        current_normal = current_eigenvectors[:, 1]
        current_tangent = np.array((-current_normal[1], current_normal[0]))
        current_total = float(np.sum(current_eigenvalues))
        coherence = (
            float(current_eigenvalues[1] - current_eigenvalues[0])
            / current_total
            if current_total > np.finfo(np.float64).tiny else 0.0
        )
        compatibility = coherence * float(np.dot(tangent, current_tangent) ** 2)
        corner_dx = corners[:, 0] - center_x
        corner_dy = corners[:, 1] - center_y
        phi = normal[0] * corner_dx + normal[1] * corner_dy
        psi = tangent[0] * corner_dx + tangent[1] * corner_dy
        frames.append(OwnerFrame(
            owner=index,
            population=int(np.count_nonzero(support[index] > 0.0)),
            center_x=center_x,
            center_y=center_y,
            normal_x=float(normal[0]),
            normal_y=float(normal[1]),
            tangent_x=float(tangent[0]),
            tangent_y=float(tangent[1]),
            transport_compatibility=compatibility,
            minimum_phi=float(np.min(phi)),
            maximum_phi=float(np.max(phi)),
            minimum_psi=float(np.min(psi)),
            maximum_psi=float(np.max(psi)),
        ))
    return DistanceAtlas(distances, tuple(frames), (height, width))


def _sample_field(values: Array, y: Array, x: Array) -> Array:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        return map_coordinates(
            field, (y, x), order=1, mode="nearest", prefilter=False
        )
    return np.stack([
        map_coordinates(
            field[..., channel], (y, x), order=1,
            mode="nearest", prefilter=False,
        )
        for channel in range(field.shape[2])
    ], axis=-1)


def _phase_aligned_chart(
    values: Array,
    frame: OwnerFrame,
    phi_axis: Array,
    psi_axis: Array,
    atlas_shape: tuple[int, int],
) -> Array:
    """Construct the owner potential directly on its Fermi chart.

    For a fixed signed-Eikonal level ``phi``, each Cartesian source line is
    intersected with that level and evaluated through its admitted CONV
    current.  Those intersections are ordered by the companion coordinate
    ``psi`` and admitted once more along the level.  The horizontal and
    vertical factorizations are combined with the coordinate-energy weights
    n_x^2 and n_y^2.  The result is one scalar potential; neither component
    current is subsequently edited.
    """

    field = np.asarray(values, dtype=np.float64)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    height, width, channels = field.shape
    atlas_height, atlas_width = atlas_shape
    step_x = (atlas_width - 1.0) / max(width - 1.0, 1.0)
    step_y = (atlas_height - 1.0) / max(height - 1.0, 1.0)
    nx, ny = frame.normal_x, frame.normal_y
    tolerance = 128.0 * np.finfo(np.float64).eps

    row_lines = np.ascontiguousarray(
        np.moveaxis(field, 1, 0).reshape(width, height * channels),
        dtype=np.float32,
    )
    column_lines = np.ascontiguousarray(
        field.reshape(height, width * channels), dtype=np.float32
    )
    source_y = step_y * np.arange(height, dtype=np.float64)
    source_x = step_x * np.arange(width, dtype=np.float64)
    row_chart = np.zeros(
        (psi_axis.size, phi_axis.size, channels), dtype=np.float64
    )
    column_chart = np.zeros_like(row_chart)
    row_mass = nx * nx if abs(nx) > tolerance else 0.0
    column_mass = ny * ny if abs(ny) > tolerance else 0.0

    for phi_index, phi in enumerate(phi_axis):
        if abs(nx) > tolerance:
            crossing_x = frame.center_x + (
                phi - ny * (source_y - frame.center_y)
            ) / nx
            crossing_source_x = np.clip(
                crossing_x / step_x, 0.0, width - 1.0
            )
            level_trace = conv_evaluate_lines(
                row_lines, np.repeat(crossing_source_x, channels)
            ).reshape(height, channels)
            query_y = frame.center_y + ny * phi + nx * psi_axis
            route = conv_evaluate_profile(
                level_trace, np.clip(query_y / step_y, 0.0, height - 1.0)
            )
            row_chart[:, phi_index] = route
        if abs(ny) > tolerance:
            crossing_y = frame.center_y + (
                phi - nx * (source_x - frame.center_x)
            ) / ny
            crossing_source_y = np.clip(
                crossing_y / step_y, 0.0, height - 1.0
            )
            level_trace = conv_evaluate_lines(
                column_lines, np.repeat(crossing_source_y, channels)
            ).reshape(width, channels)
            query_x = frame.center_x + nx * phi - ny * psi_axis
            route = conv_evaluate_profile(
                level_trace, np.clip(query_x / step_x, 0.0, width - 1.0)
            )
            column_chart[:, phi_index] = route
        if row_mass + column_mass <= tolerance:
            raise RuntimeError("owner frame has no resolvable Cartesian route")

    chart = (
        row_mass * row_chart + column_mass * column_chart
    ) / (row_mass + column_mass)

    return chart[..., 0] if scalar else chart


def _target_owner_weights(
    atlas: OwnerAtlas, target: tuple[int, int]
) -> dict[int, Array]:
    """Bilinear partition of unity subordinate to the coarse owner atlas."""

    height, width = target
    source_height, source_width = atlas.source_shape
    y = np.linspace(0.0, source_height - 1.0, height)
    x = np.linspace(0.0, source_width - 1.0, width)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    fy = y - y0
    fx = x - x0
    weights: dict[int, Array] = {}
    for yy, wy in ((y0, 1.0 - fy), (y1, fy)):
        for xx, wx in ((x0, 1.0 - fx), (x1, fx)):
            labels = atlas.labels[yy[:, None], xx[None, :]]
            contribution = wy[:, None] * wx[None, :]
            for owner in np.unique(labels):
                owner_id = int(owner)
                if owner_id not in weights:
                    weights[owner_id] = np.zeros(target, dtype=np.float64)
                weights[owner_id] += np.where(
                    labels == owner_id, contribution, 0.0
                )
    return weights


def _target_distance_weights(
    atlas: DistanceAtlas, target: tuple[int, int]
) -> Array:
    """Parameter-free Shepard partition in source-conditioned action."""

    height, width = target
    source_height, source_width = atlas.source_shape
    y = np.linspace(0.0, source_height - 1.0, height)
    x = np.linspace(0.0, source_width - 1.0, width)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    fy = y - y0
    fx = x - x0
    action = (
        atlas.distances[:, y0[:, None], x0[None, :]]
        * (1.0 - fy)[None, :, None] * (1.0 - fx)[None, None, :]
        + atlas.distances[:, y0[:, None], x1[None, :]]
        * (1.0 - fy)[None, :, None] * fx[None, None, :]
        + atlas.distances[:, y1[:, None], x0[None, :]]
        * fy[None, :, None] * (1.0 - fx)[None, None, :]
        + atlas.distances[:, y1[:, None], x1[None, :]]
        * fy[None, :, None] * fx[None, None, :]
    )
    zero = action <= 64.0 * np.finfo(np.float64).eps
    inverse = 1.0 / np.maximum(action * action, np.finfo(np.float64).tiny)
    weights = inverse / np.sum(inverse, axis=0, keepdims=True)
    any_zero = np.any(zero, axis=0)
    if np.any(any_zero):
        exact = zero[:, any_zero].astype(np.float64)
        weights[:, any_zero] = exact / np.sum(exact, axis=0, keepdims=True)
    return weights


def _target_action(distance: Array, target: tuple[int, int]) -> Array:
    """Bilinearly evaluate one coarse action field on the target lattice."""

    height, width = target
    source_height, source_width = distance.shape
    y = np.linspace(0.0, source_height - 1.0, height)
    x = np.linspace(0.0, source_width - 1.0, width)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    fy = y - y0
    fx = x - x0
    return (
        distance[y0[:, None], x0[None, :]]
        * (1.0 - fy)[:, None] * (1.0 - fx)[None, :]
        + distance[y0[:, None], x1[None, :]]
        * (1.0 - fy)[:, None] * fx[None, :]
        + distance[y1[:, None], x0[None, :]]
        * fy[:, None] * (1.0 - fx)[None, :]
        + distance[y1[:, None], x1[None, :]]
        * fy[:, None] * fx[None, :]
    )


def _distance_scales(atlas: DistanceAtlas) -> Array:
    """Nearest-germ action supplies one intrinsic scale per source."""

    count = len(atlas.frames)
    if count <= 1:
        return np.ones(count, dtype=np.float64)
    height, width = atlas.source_shape
    scales = np.empty(count, dtype=np.float64)
    centers_x = np.array([frame.center_x for frame in atlas.frames])
    centers_y = np.array([frame.center_y for frame in atlas.frames])
    for index, distance in enumerate(atlas.distances):
        values = map_coordinates(
            distance, (centers_y, centers_x), order=1,
            mode="nearest", prefilter=False,
        )
        values[index] = np.inf
        scales[index] = float(np.min(values))
    positive = scales[np.isfinite(scales) & (scales > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(np.isfinite(scales) & (scales > 0.0), scales, fallback)


def _distance_response(action: Array, scale: float, response: str) -> Array:
    """Evaluate one of the fixed positive decreasing response laws."""

    distance = np.asarray(action, dtype=np.float64)
    if response == "inverse_power_2":
        epsilon = 64.0 * np.finfo(np.float64).eps
        return 1.0 / np.maximum(distance * distance, epsilon * epsilon)
    if response == "softmax":
        return np.exp(-distance)
    normalized = distance / max(float(scale), np.finfo(np.float64).tiny)
    if response == "scaled_softmax":
        return np.exp(-normalized)
    if response == "cauchy_kernel":
        return 1.0 / (1.0 + normalized * normalized)
    if response == "normalized_softplus":
        return np.logaddexp(0.0, -normalized)
    raise ValueError(f"unknown distance response {response!r}")


def synthesize_owner_charts(
    values: Array,
    target_shape: tuple[int, int],
    atlas: OwnerAtlas,
) -> Array:
    """Reconstruct one shared scalar potential in every owner Fermi chart."""

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5:
        raise ValueError("owner synthesis requires an HxW or HxWxC raster")
    baseline = conv_resize(field, target)
    baseline64 = np.asarray(baseline, dtype=np.float64)
    output = baseline64.copy()
    target_weights = _target_owner_weights(atlas, target)
    atlas_height, atlas_width = atlas.source_shape
    source_height, source_width = field.shape[:2]
    target_height, target_width = target

    source_step_x = (
        (atlas_width - 1.0) / (source_width - 1.0)
        if source_width > 1 else 1.0
    )
    source_step_y = (
        (atlas_height - 1.0) / (source_height - 1.0)
        if source_height > 1 else 1.0
    )
    target_step_x = (
        (atlas_width - 1.0) / (target_width - 1.0)
        if target_width > 1 else source_step_x
    )
    target_step_y = (
        (atlas_height - 1.0) / (target_height - 1.0)
        if target_height > 1 else source_step_y
    )
    fine_step = min(target_step_x, target_step_y)
    # A bilinear owner indicator reaches one full coarse cell beyond its
    # owner's nodes.  This is exactly the overlap required by the partition.
    halo = max(source_step_x, source_step_y)

    target_y, target_x = np.indices(target, dtype=np.float64)
    atlas_x = target_x * (atlas_width - 1.0) / max(target_width - 1.0, 1.0)
    atlas_y = target_y * (atlas_height - 1.0) / max(target_height - 1.0, 1.0)

    for frame in atlas.frames:
        owner_weight = target_weights.get(frame.owner)
        if owner_weight is None:
            continue
        selected = owner_weight > 0.0
        if not np.any(selected):
            continue
        phi0 = math.floor((frame.minimum_phi - halo) / fine_step) * fine_step
        phi1 = math.ceil((frame.maximum_phi + halo) / fine_step) * fine_step
        psi0 = math.floor((frame.minimum_psi - halo) / fine_step) * fine_step
        psi1 = math.ceil((frame.maximum_psi + halo) / fine_step) * fine_step
        phi_count = int(round((phi1 - phi0) / fine_step)) + 1
        psi_count = int(round((psi1 - psi0) / fine_step)) + 1
        if phi_count < 5 or psi_count < 5:
            continue
        phi_axis = phi0 + fine_step * np.arange(phi_count)
        psi_axis = psi0 + fine_step * np.arange(psi_count)
        potential = _phase_aligned_chart(
            field, frame, phi_axis, psi_axis, atlas.source_shape
        )

        dx = atlas_x[selected] - frame.center_x
        dy = atlas_y[selected] - frame.center_y
        query_phi = frame.normal_x * dx + frame.normal_y * dy
        query_psi = frame.tangent_x * dx + frame.tangent_y * dy
        fine_x = (query_phi - phi0) / fine_step
        fine_y = (query_psi - psi0) / fine_step
        sampled = _sample_field(potential, fine_y, fine_x)
        # The squared membership is the same-owner product mass.  The
        # complement remains in the common Cartesian chart, preventing two
        # incompatible owner frames from replacing their shared coordinate
        # system at an overlap.
        correction_weight = (
            frame.transport_compatibility * owner_weight[selected] ** 2
        )
        if output.ndim == 2:
            output[selected] += correction_weight * (
                sampled - baseline64[selected]
            )
        else:
            output[selected] += correction_weight[:, None] * (
                sampled - baseline64[selected]
            )

    # Remove only floating-point accumulation drift from convex route fusion.
    # In exact arithmetic the chart is already inside this interval.
    return np.clip(output, float(np.min(field)), float(np.max(field))).astype(
        np.float32
    )


def synthesize_distance_charts(
    values: Array,
    target_shape: tuple[int, int],
    atlas: DistanceAtlas,
    response: str = "inverse_power_2",
    support_power: float = 2.0,
) -> Array:
    """Fuse chart transports by continuous source-conditioned action."""

    return synthesize_distance_responses(
        values, target_shape, atlas, (response,), support_power=support_power
    )[response]


def synthesize_distance_responses(
    values: Array,
    target_shape: tuple[int, int],
    atlas: DistanceAtlas,
    responses: tuple[str, ...] = DISTANCE_RESPONSES,
    support_power: float = 2.0,
) -> dict[str, Array]:
    """Evaluate several response laws while sharing every chart proposal."""

    field = np.asarray(values, dtype=np.float32)
    if not np.isfinite(support_power) or support_power <= 0.0:
        raise ValueError("support_power must be positive and finite")
    target = tuple(map(int, target_shape))
    baseline = np.asarray(conv_resize(field, target), dtype=np.float64)
    selected_responses = tuple(dict.fromkeys(responses))
    for name in selected_responses:
        _distance_response(np.zeros(1), 1.0, name)
    outputs = {name: baseline.copy() for name in selected_responses}
    scales = _distance_scales(atlas)
    denominators = {
        name: np.zeros(target, dtype=np.float64)
        for name in selected_responses
    }
    for index, distance in enumerate(atlas.distances):
        action = _target_action(distance, target)
        for name in selected_responses:
            denominators[name] += _distance_response(
                action, scales[index], name
            )
    atlas_height, atlas_width = atlas.source_shape
    source_height, source_width = field.shape[:2]
    target_height, target_width = target
    source_step_x = (atlas_width - 1.0) / max(source_width - 1.0, 1.0)
    source_step_y = (atlas_height - 1.0) / max(source_height - 1.0, 1.0)
    target_step_x = (atlas_width - 1.0) / max(target_width - 1.0, 1.0)
    target_step_y = (atlas_height - 1.0) / max(target_height - 1.0, 1.0)
    fine_step = min(target_step_x, target_step_y)
    halo = max(source_step_x, source_step_y)
    target_y, target_x = np.indices(target, dtype=np.float64)
    atlas_x = target_x * (atlas_width - 1.0) / max(target_width - 1.0, 1.0)
    atlas_y = target_y * (atlas_height - 1.0) / max(target_height - 1.0, 1.0)

    for index, frame in enumerate(atlas.frames):
        action = _target_action(atlas.distances[index], target)
        correction = {}
        for name in selected_responses:
            mass = _distance_response(
                action, scales[index], name
            ) / denominators[name]
            correction[name] = (
                frame.transport_compatibility * mass ** support_power
            )
        selected = np.ones(target, dtype=bool)
        phi0 = math.floor((frame.minimum_phi - halo) / fine_step) * fine_step
        phi1 = math.ceil((frame.maximum_phi + halo) / fine_step) * fine_step
        psi0 = math.floor((frame.minimum_psi - halo) / fine_step) * fine_step
        psi1 = math.ceil((frame.maximum_psi + halo) / fine_step) * fine_step
        phi_count = int(round((phi1 - phi0) / fine_step)) + 1
        psi_count = int(round((psi1 - psi0) / fine_step)) + 1
        if phi_count < 5 or psi_count < 5:
            continue
        phi_axis = phi0 + fine_step * np.arange(phi_count)
        psi_axis = psi0 + fine_step * np.arange(psi_count)
        potential = _phase_aligned_chart(
            field, frame, phi_axis, psi_axis, atlas.source_shape
        )
        dx = atlas_x[selected] - frame.center_x
        dy = atlas_y[selected] - frame.center_y
        fine_x = (
            frame.normal_x * dx + frame.normal_y * dy - phi0
        ) / fine_step
        fine_y = (
            frame.tangent_x * dx + frame.tangent_y * dy - psi0
        ) / fine_step
        sampled = _sample_field(potential, fine_y, fine_x)
        delta = sampled - baseline[selected]
        for name in selected_responses:
            output = outputs[name]
            if output.ndim == 2:
                output[selected] += correction[name][selected] * delta
            else:
                output[selected] += correction[name][selected, None] * delta
    minimum = float(np.min(field))
    maximum = float(np.max(field))
    return {
        name: np.clip(output, minimum, maximum).astype(np.float32)
        for name, output in outputs.items()
    }


def eikonal_basin_conv_resize(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """EB-CONV synthesis with geometry measured only from its input raster.

    The owner population, signed-Eikonal normal coordinate, and companion
    tangent coordinate are determined from ``values`` before target
    evaluation.  No reference or target-resolution raster enters the atlas.
    """

    field = np.asarray(values, dtype=np.float32)
    return synthesize_owner_charts(field, target_shape, build_owner_atlas(field))


def distance_eikonal_conv_resize(
    values: Array,
    target_shape: tuple[int, int],
    response: str = "inverse_power_2",
) -> Array:
    """Experimental owner-free, source-conditioned Eikonal synthesis."""

    field = np.asarray(values, dtype=np.float32)
    return synthesize_distance_charts(
        field, target_shape, build_distance_atlas(field), response=response
    )


def direct_distance_eikonal_conv_resize(
    values: Array,
    target_shape: tuple[int, int],
    response: str = "inverse_power_2",
    support_power: float = 1.0,
) -> Array:
    """Owner-free action frames with direct barycentric chart fusion."""

    field = np.asarray(values, dtype=np.float32)
    atlas = build_direct_distance_atlas(field, response=response)
    return synthesize_distance_charts(
        field, target_shape, atlas, response=response,
        support_power=support_power,
    )


def v3_germ_direct_resize(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Experimental V3-germ cycle retained only for diagnostic comparison.

    Reduction is exact basin analysis of the admitted source profile.  Any
    remaining enlargement is synthesized by the direct, inverse-square
    source-action partition with linear partition mass.  No owner field,
    response-law selector, or support exponent is exposed by this operator.
    """

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    if field.ndim not in (2, 3) or len(target) != 2:
        raise ValueError("V3-germ transfer requires an HxW or HxWxC raster")
    if min(field.shape[:2]) < 5 or min(target) < 5:
        raise ValueError("V3-germ transfer requires at least five sites per axis")

    output = field
    # Analysis is the reverse of the declared horizontal-then-vertical
    # synthesis order.
    if target[0] < output.shape[0]:
        output = _basin_average_axis(output, target[0], 0)
    if target[1] < output.shape[1]:
        output = _basin_average_axis(output, target[1], 1)

    if output.shape[:2] == target:
        return np.asarray(output, dtype=np.float32)
    return direct_distance_eikonal_conv_resize(
        output,
        target,
        response="inverse_power_2",
        support_power=1.0,
    )
