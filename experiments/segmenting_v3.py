"""Version 3.0: half-scale cartoon transport, full-scale local texture.

The hierarchy is deliberately strict:

1. Downsample the unchanged source by two and perform one Meyer split.
2. Build and transport *cartoon-only* geometry at that scale.
3. Fit the cartoon there and lift its field and owner IDs.
4. Upgrade only the lifted owner interfaces against full-resolution evidence.
5. Define full-resolution texture exactly as ``target - refined_cartoon``.
6. Emit a full-resolution texture population, give every texture germ exactly
   one cartoon parent, and transport only among siblings.
7. Fit two paired one-sided coordinates per texture microcell.

This is an experimental representation, not a compatibility layer around the
canonical segmenter.  It intentionally omits candidate Meyer scoring,
interface proposals, owner-free diffusion, characteristic relaxation, and
global texture/cartoon intersection products.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "viewer", ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from bfft.effects import lab_to_srgb, srgb_to_lab
from bfft.vision import (
    hard_affine_fit_native,
    hard_basis_refit_native,
    measure_paired_offsets,
)
from port_needed.continuous_eikonal_transport import (
    continuous_first_partition_prepared,
    prepare_continuous_metric,
)
from port_needed.anisotropic_edge_cost import (
    build_edge_costs,
    build_geometry_edge_costs,
)
from port_needed.density_population import (
    curvature_limited_geometry,
    emit_density_population,
)
from port_needed.frozen_meyer_geometry import (
    build_frozen_geometry,
    reweight_frozen_support,
    restrict_geometry,
)
from port_needed.first_arrival_site_force import safe_characteristic_site_step
from port_needed.fast_image_ops import (
    binary_dilation,
    gaussian_filter,
    resize,
    sobel,
)
from port_needed.two_label_transport import (
    hard_first_partition_with_forest,
    hard_partition_with_forest,
    local_hard_partition_with_forest,
    restrict_costs_to_partition,
)
from port_needed.reverse_residual_flow import reverse_residual_refill
from port_needed.wide_stencil_transport import _metric_fields


@dataclass(frozen=True)
class SegmentingV3Config:
    structural_topology: str = "half_cartoon"
    structural_allocation_side: int = 512
    structural_safety_cells: int = 32768
    structural_glass_weight: float = 0.70
    structural_flow_sweeps: int = 24
    structural_null_evidence_strength: float = 0.5
    structural_population_scale: float = 0.8
    structural_characteristic_passes: int = 1
    structural_characteristic_trust_fraction: float = 0.5
    structural_characteristic_core_radius: float = 3.0
    structural_full_transport: str = "auto"
    structural_ridges: int = 1
    cartoon_scale: float = 0.5
    meyer_sweeps: int = 1
    meyer_operator: str = "jump_measure"
    metric_strength: float = 1.5
    boundary_jump_strength: float = 24.0
    safety_cells: int = 32768
    owner_upgrade: bool = True
    owner_upgrade_mode: str = "boundary_band"
    owner_upgrade_radius: int = 8
    owner_upgrade_sweeps: int = 2
    owner_upgrade_strength: float = 64.0
    owner_upgrade_cartoon_strength: float = 0.0
    cartoon_full_refit: bool = True
    cartoon_refit_strength: float = 0.5
    texture_model: str = "nested_population"
    nested_texture_ridges: int = 3
    texture_graph_phase: bool = True
    texture_dirichlet_envelope: bool = True
    texture_support_weight: float = 0.65
    texture_population_scale: float = 1.0
    texture_population_phase: float = 0.125
    texture_curvature_population: bool = True
    texture_safety_cells: int = 131072
    texture_cleanup: bool = True
    texture_split_error_ratio: float = 2.5
    texture_split_peak_error_ratio: float = 9.0
    texture_split_return_extent: float = 2.0
    texture_split_minimum_pixels: int = 12
    texture_split_transport: str = "paired_metric"
    texture_split_metric_strength: float = 0.25
    texture_merge_penalty: float = 4.0
    texture_merge_rounds: int = 1
    texture_merge_topology: str = "best_graph"
    texture_cross_structural_merges: bool = False
    joint_leaf_collapse: bool = True
    joint_leaf_penalty: float = 4.0
    texture_interface_refresh: bool = False
    texture_interface_radius: int = 1
    texture_interface_sweeps: int = 1
    texture_interface_strength: float = 64.0
    texture_interface_confidence: float = 0.15
    texture_interface_error_ratio: float = 2.0
    texture_coordinates: int = 8
    texture_tensor_sigma: float = 1.0
    coordinate_axes: str = "paired"
    coordinate_geometry: str = "straight"
    eikonal_sweeps: int = 2
    eikonal_metric_strength: float = 2.0
    offset_bins: int = 161
    ridge_kappa: float = 16.0
    compound_segmentation: bool = False
    compound_segment_ratio: float = 1.0
    region_posterization: bool = False
    region_posterization_depth: int = 6
    region_posterization_histogram_side: int = 32
    region_family_fusion: bool = False
    threads: int = 4
    diagnostic_return_basis: bool = False


def _replace_population_measure(
    geometry: dict,
    raw_measure: np.ndarray,
) -> dict:
    """Replace only a geometry's population measure in physical cell units."""
    raw = np.maximum(np.asarray(raw_measure, dtype=np.float64), 0.0)
    implied = max(float(np.sum(raw)), 1e-30)
    result = dict(geometry)
    result["measure"] = np.ascontiguousarray(
        raw / implied, dtype=np.float32)
    result["implied_cells"] = implied
    return result


def _structural_measure_below_full(
    structural: dict,
    full: dict,
) -> dict:
    """Make structural mass a literal submeasure of full support demand."""
    structural_raw = (
        np.asarray(structural["measure"], dtype=np.float64)
        * float(structural["implied_cells"])
    )
    full_raw = (
        np.asarray(full["measure"], dtype=np.float64)
        * float(full["implied_cells"])
    )
    return _replace_population_measure(
        structural, np.minimum(structural_raw, full_raw))


def _detail_surplus_geometry(
    full: dict,
    structural: dict,
    scale: float,
) -> dict:
    """Return the nonnegative population remainder ``full - structural``."""
    full_raw = (
        np.asarray(full["measure"], dtype=np.float64)
        * float(full["implied_cells"])
    )
    structural_raw = (
        np.asarray(structural["measure"], dtype=np.float64)
        * float(structural["implied_cells"])
    )
    detail_raw = np.maximum(full_raw - structural_raw, 0.0)
    return _replace_population_measure(
        full, detail_raw * max(float(scale), 1e-6))


def _resize(value: np.ndarray, shape: tuple[int, int], *, order: int) -> np.ndarray:
    return resize(value, shape, order=order, anti_aliasing=order != 0)


def _lift_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    low_height, low_width = labels.shape
    y = np.minimum(
        ((np.arange(height) + 0.5) * low_height / height).astype(np.intp),
        low_height - 1,
    )
    x = np.minimum(
        ((np.arange(width) + 0.5) * low_width / width).astype(np.intp),
        low_width - 1,
    )
    return np.ascontiguousarray(labels[y[:, None], x[None, :]], dtype=np.int32)


def _residual_tensor(
    residual_lab: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx = np.zeros(residual_lab.shape[:2], dtype=np.float64)
    xy = np.zeros_like(xx)
    yy = np.zeros_like(xx)
    channels = np.ascontiguousarray(np.moveaxis(residual_lab, -1, 0))
    gradient_x, gradient_y = sobel(channels)
    for gx, gy in zip(gradient_x, gradient_y):
        xx += gx * gx
        xy += gx * gy
        yy += gy * gy
    smoothing = max(float(sigma), 0.0)
    if smoothing > 0.0:
        xx, xy, yy = gaussian_filter(
            np.stack((xx, xy, yy)), smoothing)
    return xx, xy, yy, xx + yy


def _cell_frame(
    labels: np.ndarray,
    centers: np.ndarray,
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    normalization_cells: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flat = labels.ravel()
    cells = len(centers)

    def reduce(value):
        return np.bincount(
            flat,
            weights=np.asarray(value, dtype=np.float64).ravel(),
            minlength=cells,
        )

    xx, xy, yy = tensor
    doubled_x = reduce(xx - yy)
    doubled_y = reduce(2.0 * xy)
    discriminant = np.hypot(doubled_x, doubled_y)
    cosine_double = np.divide(
        doubled_x,
        discriminant,
        out=np.ones_like(discriminant),
        where=discriminant > 1e-30,
    )
    sine_double = np.divide(
        doubled_y,
        discriminant,
        out=np.zeros_like(discriminant),
        where=discriminant > 1e-30,
    )
    normal_x = np.sqrt(np.maximum(0.5 * (1.0 + cosine_double), 0.0))
    normal_y = np.copysign(
        np.sqrt(np.maximum(0.5 * (1.0 - cosine_double), 0.0)),
        np.where(np.abs(sine_double) > 1e-30, sine_double, 1.0),
    )

    height, width = labels.shape
    y, x = np.mgrid[:height, :width]
    dx = x.ravel() - (centers[:, 0] * width - 0.5)[flat]
    dy = y.ravel() - (centers[:, 1] * height - 0.5)[flat]
    scale_cells = (
        cells if normalization_cells is None else int(normalization_cells))
    scale = max(
        math.sqrt(height * width / max(scale_cells, 1)), 1e-30)
    normal = (
        dx * normal_x[flat] + dy * normal_y[flat]
    ) / scale
    tangent = (
        -dx * normal_y[flat] + dy * normal_x[flat]
    ) / scale
    return normal, tangent


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True, fastmath=False) if njit is not None else _identity
_compile_parallel = (
    njit(cache=True, fastmath=False, parallel=True)
    if njit is not None else _identity
)


@_compile
def _maximum_spanning_tree_csr(cells, pairs, order):
    """Return a deterministic maximum-confidence spanning forest as CSR."""
    union = np.arange(cells, dtype=np.int32)
    rank = np.zeros(cells, dtype=np.uint8)
    selected = np.empty((max(cells - 1, 0), 2), dtype=np.int32)
    used = 0
    for position in range(len(order)):
        edge = order[position]
        first = pairs[edge, 0]
        while union[first] != first:
            union[first] = union[union[first]]
            first = union[first]
        second = pairs[edge, 1]
        while union[second] != second:
            union[second] = union[union[second]]
            second = union[second]
        if first == second:
            continue
        if rank[first] < rank[second]:
            swap = first
            first = second
            second = swap
        union[second] = first
        if rank[first] == rank[second]:
            rank[first] += 1
        selected[used] = pairs[edge]
        used += 1

    degree = np.zeros(cells, dtype=np.int32)
    for edge in range(used):
        degree[selected[edge, 0]] += 1
        degree[selected[edge, 1]] += 1
    offset = np.empty(cells + 1, dtype=np.int64)
    offset[0] = 0
    for cell in range(cells):
        offset[cell + 1] = offset[cell] + degree[cell]
    neighbour = np.empty(2 * used, dtype=np.int32)
    cursor = offset[:-1].copy()
    for edge in range(used):
        first = selected[edge, 0]
        second = selected[edge, 1]
        neighbour[cursor[first]] = second
        cursor[first] += 1
        neighbour[cursor[second]] = first
        cursor[second] += 1
    return offset, neighbour, used


@_compile
def _unroll_phase_tree(
    offset,
    neighbour,
    root_order,
    center_x,
    center_y,
    wave_x,
    wave_y,
    local_phase,
):
    """Make adjacent cell phases agree at each spanning-tree interface."""
    cells = len(center_x)
    correction = np.zeros(cells, dtype=np.float64)
    seen = np.zeros(cells, dtype=np.uint8)
    queue = np.empty(cells, dtype=np.int32)
    for root_position in range(len(root_order)):
        root = root_order[root_position]
        if seen[root]:
            continue
        seen[root] = 1
        head = 0
        tail = 1
        queue[0] = root
        while head < tail:
            first = queue[head]
            head += 1
            for edge in range(offset[first], offset[first + 1]):
                second = neighbour[edge]
                if seen[second]:
                    continue
                midpoint_x = 0.5 * (
                    center_x[first] + center_x[second])
                midpoint_y = 0.5 * (
                    center_y[first] + center_y[second])
                first_phase = (
                    local_phase[first]
                    + correction[first]
                    + wave_x[first] * (
                        midpoint_x - center_x[first])
                    + wave_y[first] * (
                        midpoint_y - center_y[first])
                )
                second_phase = (
                    local_phase[second]
                    + wave_x[second] * (
                        midpoint_x - center_x[second])
                    + wave_y[second] * (
                        midpoint_y - center_y[second])
                )
                difference = first_phase - second_phase
                correction[second] = math.atan2(
                    math.sin(difference), math.cos(difference))
                seen[second] = 1
                queue[tail] = second
                tail += 1
    return correction


def _same_cell_lag_correlation(
    signal: np.ndarray,
    labels: np.ndarray,
    cells: int,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized paired one-sided correlation for one lattice lag."""
    height, width = labels.shape
    source_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_y = slice(max(0, dy), min(height, height + dy))
    target_x = slice(max(0, dx), min(width, width + dx))
    first = signal[source_y, source_x]
    second = signal[target_y, target_x]
    owner = labels[source_y, source_x]
    same = owner == labels[target_y, target_x]
    selected_owner = owner[same]
    numerator = np.bincount(
        selected_owner,
        weights=(first * second)[same],
        minlength=cells,
    )
    denominator = np.bincount(
        selected_owner,
        weights=(0.5 * (first * first + second * second))[same],
        minlength=cells,
    )
    correlation = np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
    )
    return np.clip(correlation, -1.0, 1.0), denominator


@_compile
def _cell_pixel_csr(flat_labels, cells):
    """Group raster indices by immutable cell in two linear passes."""
    count = np.zeros(cells, dtype=np.int64)
    for pixel in range(len(flat_labels)):
        count[flat_labels[pixel]] += 1
    offset = np.empty(cells + 1, dtype=np.int64)
    offset[0] = 0
    for cell in range(cells):
        offset[cell + 1] = offset[cell] + count[cell]
    cursor = offset[:-1].copy()
    pixel_order = np.empty(len(flat_labels), dtype=np.int64)
    for pixel in range(len(flat_labels)):
        cell = flat_labels[pixel]
        pixel_order[cursor[cell]] = pixel
        cursor[cell] += 1
    return offset, pixel_order


@_compile_parallel
def _fused_cell_phase_measurements(
    flat_labels,
    pixel_offset,
    pixel_order,
    signal,
    center_x,
    center_y,
    height,
    width,
):
    """Recover both wave covectors and local phases without image temporaries."""
    cells = len(center_x)
    mean = np.zeros(cells, dtype=np.float64)
    for cell in prange(cells):
        total = 0.0
        for position in range(pixel_offset[cell], pixel_offset[cell + 1]):
            total += signal[pixel_order[position]]
        mean[cell] = total / max(
            pixel_offset[cell + 1] - pixel_offset[cell], 1)

    correlation_numerator = np.zeros((cells, 4), dtype=np.float64)
    correlation_denominator = np.zeros((cells, 4), dtype=np.float64)
    for cell in prange(cells):
        cell_mean = mean[cell]
        for position in range(pixel_offset[cell], pixel_offset[cell + 1]):
            pixel = pixel_order[position]
            y = pixel // width
            x = pixel - y * width
            first = signal[pixel] - cell_mean
            if x + 1 < width and flat_labels[pixel + 1] == cell:
                second = signal[pixel + 1] - cell_mean
                correlation_numerator[cell, 0] += first * second
                correlation_denominator[cell, 0] += 0.5 * (
                    first * first + second * second)
            if y + 1 < height:
                down = pixel + width
                if flat_labels[down] == cell:
                    second = signal[down] - cell_mean
                    correlation_numerator[cell, 1] += first * second
                    correlation_denominator[cell, 1] += 0.5 * (
                        first * first + second * second)
                if x + 1 < width and flat_labels[down + 1] == cell:
                    second = signal[down + 1] - cell_mean
                    correlation_numerator[cell, 2] += first * second
                    correlation_denominator[cell, 2] += 0.5 * (
                        first * first + second * second)
                if x > 0 and flat_labels[down - 1] == cell:
                    second = signal[down - 1] - cell_mean
                    correlation_numerator[cell, 3] += first * second
                    correlation_denominator[cell, 3] += 0.5 * (
                        first * first + second * second)

    wave_x = np.empty(cells, dtype=np.float64)
    wave_y = np.empty(cells, dtype=np.float64)
    confidence = np.empty(cells, dtype=np.float64)
    for cell in prange(cells):
        correlation = np.empty(4, dtype=np.float64)
        for direction in range(4):
            denominator = correlation_denominator[cell, direction]
            if denominator > 1e-30:
                value = (
                    correlation_numerator[cell, direction] / denominator)
                correlation[direction] = min(max(value, -1.0), 1.0)
            else:
                correlation[direction] = 0.0
        wave_x[cell] = math.acos(correlation[0])
        wave_y[cell] = math.acos(correlation[1])
        if correlation[2] >= correlation[3]:
            wave_y[cell] = -wave_y[cell]
        confidence[cell] = math.sqrt(max(
            correlation_denominator[cell, 0]
            * correlation_denominator[cell, 1],
            0.0,
        ))

    local_phase = np.empty((cells, 2), dtype=np.float64)
    for cell in prange(cells):
        cc0 = 0.0
        ss0 = 0.0
        cs0 = 0.0
        vc0 = 0.0
        vs0 = 0.0
        cc1 = 0.0
        ss1 = 0.0
        cs1 = 0.0
        vc1 = 0.0
        vs1 = 0.0
        axis_x0 = wave_x[cell]
        axis_y0 = wave_y[cell]
        axis_x1 = -axis_y0
        axis_y1 = axis_x0
        cell_mean = mean[cell]
        for position in range(pixel_offset[cell], pixel_offset[cell + 1]):
            pixel = pixel_order[position]
            y = pixel // width
            x = pixel - y * width
            dx = x - center_x[cell]
            dy = y - center_y[cell]
            value = signal[pixel] - cell_mean
            argument = axis_x0 * dx + axis_y0 * dy
            cosine0 = math.cos(argument)
            sine0 = math.sin(argument)
            argument = axis_x1 * dx + axis_y1 * dy
            cosine1 = math.cos(argument)
            sine1 = math.sin(argument)
            cc0 += cosine0 * cosine0
            ss0 += sine0 * sine0
            cs0 += cosine0 * sine0
            vc0 += value * cosine0
            vs0 += value * sine0
            cc1 += cosine1 * cosine1
            ss1 += sine1 * sine1
            cs1 += cosine1 * sine1
            vc1 += value * cosine1
            vs1 += value * sine1

        determinant = cc0 * ss0 - cs0 * cs0
        if abs(determinant) < 1e-30:
            determinant = 1e-30
        coefficient_cosine = (vc0 * ss0 - vs0 * cs0) / determinant
        coefficient_sine = (vs0 * cc0 - vc0 * cs0) / determinant
        local_phase[cell, 0] = math.atan2(
            -coefficient_sine, coefficient_cosine)

        determinant = cc1 * ss1 - cs1 * cs1
        if abs(determinant) < 1e-30:
            determinant = 1e-30
        coefficient_cosine = (vc1 * ss1 - vs1 * cs1) / determinant
        coefficient_sine = (vs1 * cc1 - vc1 * cs1) / determinant
        local_phase[cell, 1] = math.atan2(
            -coefficient_sine, coefficient_cosine)

    return wave_x, wave_y, confidence, local_phase


@_compile_parallel
def _render_graph_phase_fields(
    flat_labels,
    center_x,
    center_y,
    wave_x,
    wave_y,
    local_phase,
    correction_normal,
    correction_second,
    width,
    cosine_only,
    quadrature,
):
    """Render both synchronized phase fields in one raster traversal."""
    columns = 4 if quadrature else 2
    output = np.empty((len(flat_labels), columns), dtype=np.float64)
    for pixel in prange(len(flat_labels)):
        cell = flat_labels[pixel]
        y = pixel // width
        x = pixel - y * width
        dx = x - center_x[cell]
        dy = y - center_y[cell]
        first = (
            local_phase[cell, 0]
            + correction_normal[cell]
            + wave_x[cell] * dx
            + wave_y[cell] * dy
        )
        second = (
            local_phase[cell, 1]
            + correction_second[cell]
            - wave_y[cell] * dx
            + wave_x[cell] * dy
        )
        if quadrature:
            output[pixel, 0] = math.cos(first)
            output[pixel, 1] = math.sin(first)
            output[pixel, 2] = math.cos(second)
            output[pixel, 3] = math.sin(second)
        elif cosine_only:
            output[pixel, 0] = math.cos(first)
            output[pixel, 1] = math.cos(second)
        else:
            output[pixel, 0] = first
            output[pixel, 1] = second
    return output


def _graph_unrolled_texture_fields(
    labels: np.ndarray,
    centers: np.ndarray,
    texture_signal: np.ndarray,
    *,
    cosine_only: bool,
    quadrature: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Shared fused construction for phase diagnostics and cosine columns."""
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    centers = np.asarray(centers, dtype=np.float64)
    signal = np.ascontiguousarray(
        texture_signal, dtype=np.float64).ravel()
    height, width = labels.shape
    cells = len(centers)
    flat = labels.ravel()
    center_x = centers[:, 0] * width - 0.5
    center_y = centers[:, 1] * height - 0.5
    pixel_offset, pixel_order = _cell_pixel_csr(flat, cells)
    wave_x, wave_y, confidence, local_phase = (
        _fused_cell_phase_measurements(
            flat,
            pixel_offset,
            pixel_order,
            signal,
            center_x,
            center_y,
            height,
            width,
        )
    )

    pairs = _adjacent_cell_pairs(labels, cells)
    if pairs.size:
        distance = np.hypot(
            center_x[pairs[:, 0]] - center_x[pairs[:, 1]],
            center_y[pairs[:, 0]] - center_y[pairs[:, 1]],
        )
        edge_weight = np.minimum(
            confidence[pairs[:, 0]], confidence[pairs[:, 1]],
        ) / np.maximum(distance, 1.0)
        order = np.lexsort((
            pairs[:, 1],
            pairs[:, 0],
            -edge_weight,
        )).astype(np.int64)
        tree_offset, tree_neighbour, tree_edges = (
            _maximum_spanning_tree_csr(cells, pairs, order))
    else:
        tree_offset = np.zeros(cells + 1, dtype=np.int64)
        tree_neighbour = np.empty(0, dtype=np.int32)
        tree_edges = 0
    root_order = np.argsort(-confidence, kind="stable").astype(np.int32)
    correction_normal = _unroll_phase_tree(
        tree_offset,
        tree_neighbour,
        root_order,
        center_x,
        center_y,
        wave_x,
        wave_y,
        local_phase[:, 0],
    )
    correction_second = _unroll_phase_tree(
        tree_offset,
        tree_neighbour,
        root_order,
        center_x,
        center_y,
        -wave_y,
        wave_x,
        local_phase[:, 1],
    )
    fields = _render_graph_phase_fields(
        flat,
        center_x,
        center_y,
        wave_x,
        wave_y,
        local_phase,
        correction_normal,
        correction_second,
        width,
        cosine_only,
        quadrature,
    )
    return tuple(
        fields[:, index] for index in range(fields.shape[1])
    ), {
        "enabled": True,
        "graph_edges": int(len(pairs)),
        "tree_edges": int(tree_edges),
        "median_wave_x": float(np.median(np.abs(wave_x))),
        "median_wave_y": float(np.median(np.abs(wave_y))),
        "confidence_cells": int(np.count_nonzero(confidence > 1e-12)),
        "full_band_one_sided": True,
        "quadrature_complete": bool(quadrature),
        "fused_cell_rasters": True,
    }


def _graph_unrolled_texture_phases(
    labels: np.ndarray,
    centers: np.ndarray,
    texture_signal: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Build two full-band phase fields on one synchronized cell graph.

    Central differences measure ``sin(omega)`` and fold frequencies above
    pi/2. Paired one-sided correlations measure ``cos(omega)``, allowing
    ``arccos`` to recover the complete discrete band [0, pi]. The two
    orthogonal wave covectors are synchronized over one maximum-confidence
    spanning forest; no carrier or object components are enumerated.
    """
    return _graph_unrolled_texture_fields(
        labels,
        centers,
        texture_signal,
        cosine_only=False,
    )


def _graph_unrolled_texture_columns(
    labels: np.ndarray,
    centers: np.ndarray,
    texture_signal: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Return gauge-invariant paired carrier quadratures directly."""
    return _graph_unrolled_texture_fields(
        labels,
        centers,
        texture_signal,
        cosine_only=True,
        quadrature=True,
    )


@_compile
def _apply_texture_dirichlet_envelope(
    owner,
    target,
    fitted,
    cells,
):
    """Fused sufficient statistics and channelwise coefficient contraction."""
    height, width = owner.shape
    count = np.zeros(cells, dtype=np.int64)
    mean = np.zeros((cells, 3), dtype=np.float64)
    target_energy = np.zeros((cells, 3), dtype=np.float64)
    fitted_energy = np.zeros((cells, 3), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            cell = owner[y, x]
            count[cell] += 1
            for channel in range(3):
                mean[cell, channel] += fitted[y, x, channel]
            if x + 1 < width:
                second_cell = owner[y, x + 1]
                for channel in range(3):
                    target_delta = (
                        target[y, x + 1, channel]
                        - target[y, x, channel]
                    )
                    fitted_delta = (
                        fitted[y, x + 1, channel]
                        - fitted[y, x, channel]
                    )
                    target_edge = target_delta * target_delta
                    fitted_edge = fitted_delta * fitted_delta
                    if second_cell == cell:
                        target_energy[cell, channel] += target_edge
                        fitted_energy[cell, channel] += fitted_edge
                    else:
                        target_energy[cell, channel] += 0.5 * target_edge
                        fitted_energy[cell, channel] += 0.5 * fitted_edge
                        target_energy[second_cell, channel] += (
                            0.5 * target_edge)
                        fitted_energy[second_cell, channel] += (
                            0.5 * fitted_edge)
            if y + 1 < height:
                second_cell = owner[y + 1, x]
                for channel in range(3):
                    target_delta = (
                        target[y + 1, x, channel]
                        - target[y, x, channel]
                    )
                    fitted_delta = (
                        fitted[y + 1, x, channel]
                        - fitted[y, x, channel]
                    )
                    target_edge = target_delta * target_delta
                    fitted_edge = fitted_delta * fitted_delta
                    if second_cell == cell:
                        target_energy[cell, channel] += target_edge
                        fitted_energy[cell, channel] += fitted_edge
                    else:
                        target_energy[cell, channel] += 0.5 * target_edge
                        fitted_energy[cell, channel] += 0.5 * fitted_edge
                        target_energy[second_cell, channel] += (
                            0.5 * target_edge)
                        fitted_energy[second_cell, channel] += (
                            0.5 * fitted_edge)
    gain = np.ones((cells, 3), dtype=np.float64)
    for cell in range(cells):
        inverse_count = 1.0 / max(count[cell], 1)
        for channel in range(3):
            mean[cell, channel] *= inverse_count
            if (
                fitted_energy[cell, channel]
                > target_energy[cell, channel]
                and fitted_energy[cell, channel] > 1e-30
            ):
                gain[cell, channel] = math.sqrt(
                    max(target_energy[cell, channel], 0.0)
                    / fitted_energy[cell, channel]
                )
    output = np.empty_like(fitted)
    for y in range(height):
        for x in range(width):
            cell = owner[y, x]
            for channel in range(3):
                center = mean[cell, channel]
                output[y, x, channel] = (
                    center
                    + gain[cell, channel]
                    * (fitted[y, x, channel] - center)
                )
    after_energy = 0.0
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                if x + 1 < width:
                    delta = (
                        output[y, x + 1, channel]
                        - output[y, x, channel]
                    )
                    after_energy += delta * delta
                if y + 1 < height:
                    delta = (
                        output[y + 1, x, channel]
                        - output[y, x, channel]
                    )
                    after_energy += delta * delta
    return (
        output,
        gain,
        count,
        target_energy,
        fitted_energy,
        after_energy,
    )


def _texture_dirichlet_envelope(
    labels: np.ndarray,
    target: np.ndarray,
    fitted: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Prevent a texture fit from increasing within-cell gradient energy.

    A least-squares projection is contractive in sample-space L2, but that
    does not make it contractive in gradient energy: it may discard a smooth
    residual component while concentrating the retained energy in a carrier.
    That is the precise mechanism behind a reconstruction which has less
    total texture energy yet looks oversharpened.

    For each immutable texture cell, measure the four-neighbour Dirichlet
    energy of the exact residual target and of its fit. Cross-cell edges are
    split equally between their incident owners, so phase-aligned but
    independently fitted amplitudes cannot evade the measurement at cell
    interfaces. Preserve the fitted cell mean and, in each Lab channel, apply
    the unique largest scalar contraction indicated by that incident energy
    budget. The operation is one-shot and can be represented by scaling only
    the nonconstant cell coefficients; cells which already satisfy the bound
    are bitwise unchanged.
    """
    owner = np.ascontiguousarray(labels, dtype=np.int32)
    target_value = np.asarray(target, dtype=np.float64)
    fitted_value = np.asarray(fitted, dtype=np.float64)
    if target_value.shape != fitted_value.shape:
        raise ValueError("texture target and fit must have equal shapes")
    if target_value.shape[:2] != owner.shape or target_value.shape[2:] != (3,):
        raise ValueError("texture fields must have shape HxWx3")

    cells = int(np.max(owner)) + 1
    (
        contracted,
        gain,
        count,
        target_energy,
        fitted_energy,
        after_energy,
    ) = (
        _apply_texture_dirichlet_envelope(
            owner,
            np.ascontiguousarray(target_value),
            np.ascontiguousarray(fitted_value),
            cells,
        )
    )
    contracted_components = gain < 1.0
    contracted_cells = np.any(contracted_components, axis=1)
    if not np.any(contracted_cells):
        return fitted_value, {
            "enabled": True,
            "contracted_cells": 0,
            "contracted_pixels": 0,
            "minimum_gain": 1.0,
            "mean_contracted_gain": 1.0,
            "target_energy": float(np.sum(target_energy)),
            "before_energy": float(np.sum(fitted_energy)),
            "after_energy": float(np.sum(fitted_energy)),
        }

    return np.ascontiguousarray(contracted), {
        "enabled": True,
        "contracted_cells": int(np.count_nonzero(contracted_cells)),
        "contracted_pixels": int(np.sum(count[contracted_cells])),
        "minimum_gain": float(np.min(gain)),
        "mean_contracted_gain": float(np.mean(
            gain[contracted_components])),
        "target_energy": float(np.sum(target_energy)),
        "before_energy": float(np.sum(fitted_energy)),
        "after_energy": float(after_energy),
    }


@_compile
def _owner_masked_distance(
    labels,
    seed,
    horizontal,
    vertical,
    diagonal_down_right,
    diagonal_down_left,
    sweep_count,
):
    """Fixed causal raster sweeps for all immutable owners simultaneously."""
    height, width = labels.shape
    distance = np.full((height, width), np.inf)
    # Numba does not implement multidimensional boolean assignment.  The
    # explicit initialization is also deterministic and touches each pixel
    # exactly once before the causal passes.
    for y in range(height):
        for x in range(width):
            if seed[y, x]:
                distance[y, x] = 0.0
    sweeps = max(sweep_count, 1)
    for _ in range(sweeps):
        # Four causal scan orders cover all eight local directions. Owners
        # are absolute barriers: texture paths cannot cross a cartoon cell.
        for mode in range(4):
            if mode < 2:
                y_start, y_stop, y_step = 0, height, 1
            else:
                y_start, y_stop, y_step = height - 1, -1, -1
            if mode == 0 or mode == 3:
                x_start, x_stop, x_step = 0, width, 1
            else:
                x_start, x_stop, x_step = width - 1, -1, -1
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    owner = labels[y, x]
                    best = distance[y, x]
                    if x_step > 0:
                        if x > 0 and labels[y, x - 1] == owner:
                            best = min(
                                best,
                                distance[y, x - 1] + horizontal[y, x - 1],
                            )
                    elif x + 1 < width and labels[y, x + 1] == owner:
                        best = min(
                            best,
                            distance[y, x + 1] + horizontal[y, x],
                        )
                    previous_y = y - y_step
                    if 0 <= previous_y < height:
                        edge_y = min(y, previous_y)
                        if labels[previous_y, x] == owner:
                            best = min(
                                best,
                                distance[previous_y, x] + vertical[edge_y, x],
                            )
                        if x > 0 and labels[previous_y, x - 1] == owner:
                            diagonal = (
                                diagonal_down_right[previous_y, x - 1]
                                if previous_y < y
                                else diagonal_down_left[y, x - 1]
                            )
                            best = min(
                                best,
                                distance[previous_y, x - 1] + diagonal,
                            )
                        if (
                            x + 1 < width
                            and labels[previous_y, x + 1] == owner
                        ):
                            diagonal = (
                                diagonal_down_left[previous_y, x]
                                if previous_y < y
                                else diagonal_down_right[y, x]
                            )
                            best = min(
                                best,
                                distance[previous_y, x + 1] + diagonal,
                            )
                    distance[y, x] = best
    return distance


@_compile
def _upgrade_owner_band(
    lifted,
    fixed,
    band,
    horizontal,
    vertical,
    diagonal_down_right,
    diagonal_down_left,
    sweep_count,
):
    """Compete existing owners only inside a narrow full-resolution band."""
    height, width = lifted.shape
    owner = np.full((height, width), -1, dtype=np.int32)
    distance = np.full((height, width), np.inf)
    for y in range(height):
        for x in range(width):
            if fixed[y, x]:
                owner[y, x] = lifted[y, x]
                distance[y, x] = 0.0

    for _ in range(max(sweep_count, 1)):
        for mode in range(4):
            if mode < 2:
                y_start, y_stop, y_step = 0, height, 1
            else:
                y_start, y_stop, y_step = height - 1, -1, -1
            if mode == 0 or mode == 3:
                x_start, x_stop, x_step = 0, width, 1
            else:
                x_start, x_stop, x_step = width - 1, -1, -1
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    if not band[y, x] or fixed[y, x]:
                        continue
                    best = distance[y, x]
                    best_owner = owner[y, x]
                    original = lifted[y, x]

                    if x_step > 0:
                        nx = x - 1
                        edge = horizontal[y, x - 1] if x > 0 else np.inf
                    else:
                        nx = x + 1
                        edge = horizontal[y, x] if x + 1 < width else np.inf
                    if 0 <= nx < width and owner[y, nx] >= 0:
                        candidate = distance[y, nx] + edge
                        candidate_owner = owner[y, nx]
                        if (
                            candidate < best - 1e-12
                            or (
                                abs(candidate - best) <= 1e-12
                                and candidate_owner == original
                                and best_owner != original
                            )
                        ):
                            best, best_owner = candidate, candidate_owner

                    previous_y = y - y_step
                    if 0 <= previous_y < height:
                        edge_y = min(y, previous_y)
                        for offset in range(-1, 2):
                            nx = x + offset
                            if nx < 0 or nx >= width:
                                continue
                            candidate_owner = owner[previous_y, nx]
                            if candidate_owner < 0:
                                continue
                            if offset == 0:
                                edge = vertical[edge_y, x]
                            elif offset < 0:
                                edge = (
                                    diagonal_down_right[previous_y, x - 1]
                                    if previous_y < y
                                    else diagonal_down_left[y, x - 1]
                                )
                            else:
                                edge = (
                                    diagonal_down_left[previous_y, x]
                                    if previous_y < y
                                    else diagonal_down_right[y, x]
                                )
                            candidate = distance[previous_y, nx] + edge
                            if (
                                candidate < best - 1e-12
                                or (
                                    abs(candidate - best) <= 1e-12
                                    and candidate_owner == original
                                    and best_owner != original
                                )
                            ):
                                best, best_owner = candidate, candidate_owner
                    distance[y, x] = best
                    owner[y, x] = best_owner

    for y in range(height):
        for x in range(width):
            if owner[y, x] < 0:
                owner[y, x] = lifted[y, x]
    return owner, distance


@_compile
def _upgrade_owner_band_tensor(
    lifted,
    fixed,
    band,
    mxx,
    mxy,
    myy,
    sweep_count,
):
    """Band competition with incident metric costs evaluated on demand."""
    height, width = lifted.shape
    owner = np.full((height, width), -1, dtype=np.int32)
    distance = np.full((height, width), np.inf)
    for y in range(height):
        for x in range(width):
            if fixed[y, x]:
                owner[y, x] = lifted[y, x]
                distance[y, x] = 0.0

    for _ in range(max(sweep_count, 1)):
        for mode in range(4):
            if mode < 2:
                y_start, y_stop, y_step = 0, height, 1
            else:
                y_start, y_stop, y_step = height - 1, -1, -1
            if mode == 0 or mode == 3:
                x_start, x_stop, x_step = 0, width, 1
            else:
                x_start, x_stop, x_step = width - 1, -1, -1
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    if not band[y, x] or fixed[y, x]:
                        continue
                    best = distance[y, x]
                    best_owner = owner[y, x]
                    original = lifted[y, x]

                    nx = x - x_step
                    if 0 <= nx < width and owner[y, nx] >= 0:
                        dx = x - nx
                        a = 0.5 * (mxx[y, x] + mxx[y, nx])
                        b = 0.5 * (mxy[y, x] + mxy[y, nx])
                        c = 0.5 * (myy[y, x] + myy[y, nx])
                        edge = np.float32(math.sqrt(max(
                            dx * dx * a,
                            1e-30,
                        )))
                        candidate = distance[y, nx] + edge
                        candidate_owner = owner[y, nx]
                        if (
                            candidate < best - 1e-12
                            or (
                                abs(candidate - best) <= 1e-12
                                and candidate_owner == original
                                and best_owner != original
                            )
                        ):
                            best, best_owner = candidate, candidate_owner

                    previous_y = y - y_step
                    if 0 <= previous_y < height:
                        for offset in range(-1, 2):
                            nx = x + offset
                            if nx < 0 or nx >= width:
                                continue
                            candidate_owner = owner[previous_y, nx]
                            if candidate_owner < 0:
                                continue
                            dx = x - nx
                            dy = y - previous_y
                            a = 0.5 * (
                                mxx[y, x] + mxx[previous_y, nx])
                            b = 0.5 * (
                                mxy[y, x] + mxy[previous_y, nx])
                            c = 0.5 * (
                                myy[y, x] + myy[previous_y, nx])
                            edge = np.float32(math.sqrt(max(
                                dx * dx * a
                                + 2.0 * dx * dy * b
                                + dy * dy * c,
                                1e-30,
                            )))
                            candidate = distance[previous_y, nx] + edge
                            if (
                                candidate < best - 1e-12
                                or (
                                    abs(candidate - best) <= 1e-12
                                    and candidate_owner == original
                                    and best_owner != original
                                )
                            ):
                                best, best_owner = candidate, candidate_owner
                    distance[y, x] = best
                    owner[y, x] = best_owner

    for y in range(height):
        for x in range(width):
            if owner[y, x] < 0:
                owner[y, x] = lifted[y, x]
    return owner, distance


def _metric_coefficients(
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, xy, yy = tensor
    scale = max(float(np.percentile(xx + yy, 90.0)), 1e-30)
    gain = max(float(strength), 0.0) / scale
    return 1.0 + gain * xx, gain * xy, 1.0 + gain * yy


def _metric_edge_costs_from_coefficients(
    metric: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mxx, mxy, myy = metric
    def edge(first, second, dx: float, dy: float):
        a = 0.5 * (mxx[first] + mxx[second])
        b = 0.5 * (mxy[first] + mxy[second])
        c = 0.5 * (myy[first] + myy[second])
        return np.ascontiguousarray(np.sqrt(np.maximum(
            dx * dx * a + 2.0 * dx * dy * b + dy * dy * c,
            1e-30,
        )), dtype=np.float32)

    return (
        edge(
            (slice(None), slice(0, -1)),
            (slice(None), slice(1, None)),
            1.0, 0.0,
        ),
        edge(
            (slice(0, -1), slice(None)),
            (slice(1, None), slice(None)),
            0.0, 1.0,
        ),
        edge(
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(1, None)),
            1.0, 1.0,
        ),
        edge(
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(0, -1)),
            -1.0, 1.0,
        ),
    )


def _metric_edge_costs(
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _metric_edge_costs_from_coefficients(
        _metric_coefficients(tensor, strength))


def _upgrade_full_resolution_owners(
    lifted: np.ndarray,
    centers: np.ndarray,
    target_lab: np.ndarray,
    cartoon_metric: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    mode: str,
    radius: int,
    sweeps: int,
    strength: float,
    cartoon_strength: float,
) -> tuple[np.ndarray, dict]:
    """Move only lifted interfaces using unchanged full-resolution evidence."""
    flat = lifted.ravel()
    cells = len(centers)
    height, width = lifted.shape
    y, x = np.mgrid[:height, :width]
    dx = x.ravel() - (centers[:, 0] * width - 0.5)[flat]
    dy = y.ravel() - (centers[:, 1] * height - 0.5)[flat]
    squared = dx * dx + dy * dy
    closest = np.full(cells, np.inf)
    np.minimum.at(closest, flat, squared)
    closest_seed = (
        squared <= closest[flat] + 1e-12
    ).reshape(lifted.shape)

    selected_mode = str(mode).strip().lower()
    if selected_mode == "full_map":
        band = np.ones(lifted.shape, dtype=bool)
        fixed = closest_seed
        forced_owner_seeds = cells
    elif selected_mode == "boundary_band":
        boundary = np.zeros(lifted.shape, dtype=bool)
        boundary[1:] |= lifted[1:] != lifted[:-1]
        boundary[:-1] |= lifted[:-1] != lifted[1:]
        boundary[:, 1:] |= lifted[:, 1:] != lifted[:, :-1]
        boundary[:, :-1] |= lifted[:, :-1] != lifted[:, 1:]
        band = binary_dilation(boundary, max(int(radius), 1))
        fixed = ~band
        core_count = np.bincount(
            flat[fixed.ravel()], minlength=cells)
        missing = core_count == 0
        fixed |= closest_seed & missing[flat].reshape(lifted.shape)
        forced_owner_seeds = int(np.count_nonzero(missing))
    else:
        raise ValueError(
            "owner_upgrade_mode must be 'full_map' or 'boundary_band'")

    xx, xy, yy, _ = _residual_tensor(target_lab, 0.8)
    edge_mxx, edge_mxy, edge_myy = _metric_coefficients(
        (xx, xy, yy), strength)
    cartoon_mxx, cartoon_mxy, cartoon_myy = cartoon_metric
    cartoon_trace = (
        np.maximum(cartoon_mxx - 1.0, 0.0)
        + np.maximum(cartoon_myy - 1.0, 0.0)
    )
    cartoon_scale = max(
        float(np.percentile(cartoon_trace, 90.0)), 1e-30)
    cartoon_gain = max(float(cartoon_strength), 0.0) / cartoon_scale
    combined_metric = (
        edge_mxx + cartoon_gain * (cartoon_mxx - 1.0),
        edge_mxy + cartoon_gain * cartoon_mxy,
        edge_myy + cartoon_gain * (cartoon_myy - 1.0),
    )
    upgraded, distance = _upgrade_owner_band(
        np.ascontiguousarray(lifted, dtype=np.int32),
        np.ascontiguousarray(fixed),
        np.ascontiguousarray(band),
        *_metric_edge_costs_from_coefficients(combined_metric),
        max(int(sweeps), 1),
    )
    changed = upgraded != lifted
    return upgraded, {
        "band_pixels": int(np.count_nonzero(band)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "forced_owner_seeds": int(forced_owner_seeds),
        "unreached_pixels": int(np.count_nonzero(~np.isfinite(distance))),
        "mode": selected_mode,
    }


def _nested_texture_partition(
    rgb: np.ndarray,
    target_lab: np.ndarray,
    parent_labels: np.ndarray,
    parent_centers: np.ndarray,
    config: SegmentingV3Config,
    prepared_geometry: dict | None = None,
    structural_population_geometry: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    """Emit one full-resolution support population nested in cartoon owners."""
    geometry_started = time.perf_counter()
    if prepared_geometry is None:
        geometry = build_frozen_geometry(
            rgb,
            target_lab=target_lab,
            tgfd_sweeps=max(int(config.meyer_sweeps), 1),
            meyer_operator=config.meyer_operator,
            flow_sweeps=1,
            texture_support_weight=max(
                float(config.texture_support_weight), 0.0),
            glass_support_weight=0.0,
            null_evidence_strength=0.5,
            threads=max(int(config.threads), 0),
        )
        if bool(config.texture_curvature_population):
            geometry = curvature_limited_geometry(geometry)
    else:
        geometry = prepared_geometry
    population_scale = max(float(config.texture_population_scale), 1e-6)
    separable_population = structural_population_geometry is not None
    if separable_population:
        detail_geometry = _detail_surplus_geometry(
            geometry,
            structural_population_geometry,
            population_scale,
        )
        remaining = max(
            int(config.texture_safety_cells) - len(parent_centers), 1)
        detail_centers, detail_population = emit_density_population(
            detail_geometry,
            safety_cells=remaining,
            phase_shift=float(config.texture_population_phase),
        )
        centers = np.vstack((
            np.asarray(parent_centers, dtype=np.float64),
            detail_centers,
        ))
        population = {
            **detail_population,
            "structural_sites": int(len(parent_centers)),
            "surplus_sites": int(len(detail_centers)),
            "surplus_implied_cells": float(
                detail_geometry["implied_cells"]),
            "combined_realized_cells": int(len(centers)),
            "separable_population": True,
        }
    else:
        if population_scale != 1.0:
            geometry = dict(geometry)
            geometry["implied_cells"] = (
                float(geometry["implied_cells"]) * population_scale)
        centers, population = emit_density_population(
            geometry,
            safety_cells=max(int(config.texture_safety_cells), 1),
            phase_shift=0.0,
        )
        population["separable_population"] = False
    geometry_ms = 1000.0 * (time.perf_counter() - geometry_started)

    height, width = parent_labels.shape
    seed_x = np.clip(
        np.rint(centers[:, 0] * width - 0.5).astype(np.intp),
        0,
        width - 1,
    )
    seed_y = np.clip(
        np.rint(centers[:, 1] * height - 0.5).astype(np.intp),
        0,
        height - 1,
    )
    parent_count = len(parent_centers)
    if separable_population:
        seed_pixel = seed_y * width + seed_x
        parent_seed_count = min(parent_count, len(centers))
        occupied = set(seed_pixel[:parent_seed_count].tolist())
        keep_extra = []
        for site in range(parent_seed_count, len(centers)):
            pixel_id = int(seed_pixel[site])
            if pixel_id not in occupied:
                occupied.add(pixel_id)
                keep_extra.append(site)
        keep = np.concatenate((
            np.arange(parent_seed_count, dtype=np.intp),
            np.asarray(keep_extra, dtype=np.intp),
        ))
        centers = centers[keep]
        seed_x, seed_y = seed_x[keep], seed_y[keep]
        parent = np.concatenate((
            np.arange(parent_seed_count, dtype=np.int32),
            parent_labels[
                seed_y[parent_seed_count:],
                seed_x[parent_seed_count:],
            ],
        ))
        represented = np.ones(parent_count, dtype=bool)
        missing = np.empty(0, dtype=np.int32)
    else:
        seed_pixel = seed_y * width + seed_x
        _, first = np.unique(seed_pixel, return_index=True)
        first.sort()
        centers = centers[first]
        seed_x, seed_y = seed_x[first], seed_y[first]
        parent = parent_labels[seed_y, seed_x]
        represented = np.bincount(parent, minlength=parent_count) > 0
        missing = np.flatnonzero(~represented)
    if missing.size:
        flat_parent = parent_labels.ravel()
        pixel = np.arange(flat_parent.size, dtype=np.int64)
        x = pixel % width
        y = pixel // width
        dx = x - (parent_centers[:, 0] * width - 0.5)[flat_parent]
        dy = y - (parent_centers[:, 1] * height - 0.5)[flat_parent]
        squared = dx * dx + dy * dy
        closest = np.full(parent_count, np.inf)
        np.minimum.at(closest, flat_parent, squared)
        candidate = (
            (~represented)[flat_parent]
            & (squared <= closest[flat_parent] + 1e-12)
        )
        # Ties are harmless geometrically but would create redundant sites.
        candidate_pixel = pixel[candidate]
        candidate_parent = flat_parent[candidate]
        _, one = np.unique(candidate_parent, return_index=True)
        candidate_pixel = candidate_pixel[one]
        candidate_parent = candidate_parent[one]
        extra = np.column_stack((
            (candidate_pixel % width + 0.5) / width,
            (candidate_pixel // width + 0.5) / height,
        ))
        centers = np.vstack((centers, extra))
        parent = np.concatenate((parent, candidate_parent))

    transport_started = time.perf_counter()
    costs = restrict_costs_to_partition(
        build_edge_costs(geometry, config.metric_strength),
        parent_labels,
    )
    forest = hard_first_partition_with_forest(centers, costs)
    labels = forest["labels"]
    unreachable = labels < 0
    if np.any(unreachable):
        # Each parent has a compulsory seed. Raster-disconnected islands use
        # that parent's first texture site rather than crossing a parent edge.
        first_site = np.full(parent_count, -1, dtype=np.int32)
        for site, parent_id in enumerate(parent):
            if first_site[parent_id] < 0:
                first_site[parent_id] = site
        labels[unreachable] = first_site[parent_labels[unreachable]]
        forest["labels"] = labels
        forest["distance"][unreachable] = 0.0
        forest["parent"][unreachable] = -1

    surviving = np.unique(labels)
    remap = np.full(len(centers), -1, dtype=np.int32)
    remap[surviving] = np.arange(len(surviving), dtype=np.int32)
    labels = remap[labels]
    centers = centers[surviving]
    parent = parent[surviving]
    forest["labels"] = labels
    transport_ms = 1000.0 * (time.perf_counter() - transport_started)
    diagnostic = {
        "emitted_sites": int(population.get(
            "combined_realized_cells",
            population["realized_cells"],
        )),
        "surplus_sites": int(population.get("surplus_sites", 0)),
        "surplus_implied_cells": float(
            population.get("surplus_implied_cells", 0.0)),
        "separable_population": bool(separable_population),
        "nested_sites": int(len(centers)),
        "missing_parent_seeds": int(len(missing)),
        "surviving_parent_ids": int(len(np.unique(parent))),
        "geometry_ms": geometry_ms,
        "transport_ms": transport_ms,
    }
    return labels, centers, geometry, forest, diagnostic


@_compile
def _centers_from_labels_kernel(
    labels: np.ndarray,
    measure: np.ndarray,
    cells: int,
) -> np.ndarray:
    """Fuse weighted mass and both moments into one row-major traversal."""
    height, width = labels.shape
    mass = np.zeros(cells, dtype=np.float64)
    center_x = np.zeros(cells, dtype=np.float64)
    center_y = np.zeros(cells, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            cell = labels[y, x]
            weight = max(measure[y, x], 0.0) + 1e-12
            mass[cell] += weight
            center_x[cell] += weight * (x + 0.5)
            center_y[cell] += weight * (y + 0.5)
    output = np.empty((cells, 2), dtype=np.float64)
    for cell in range(cells):
        denominator = max(mass[cell], 1e-30)
        output[cell, 0] = center_x[cell] / denominator / width
        output[cell, 1] = center_y[cell] / denominator / height
    return output


def _centers_from_labels(
    labels: np.ndarray,
    measure: np.ndarray,
) -> np.ndarray:
    """Return one support-weighted, normalized center for every compact ID."""
    label_field = np.ascontiguousarray(labels, dtype=np.int32)
    cells = int(np.max(label_field)) + 1
    return _centers_from_labels_kernel(
        label_field,
        np.ascontiguousarray(measure, dtype=np.float64),
        cells,
    )


def _centers_after_collapse(
    previous_labels: np.ndarray,
    collapsed_labels: np.ndarray,
    previous_centers: np.ndarray,
    measure: np.ndarray,
) -> np.ndarray:
    """Retain the transport gauge of every uncontracted model.

    A quotient changes the support center only for components containing more
    than one previous cell. Replacing every surviving site's center by its
    raster centroid moves untouched anisotropic cells, thereby rotating their
    normal frame and changing a ridge model which the quotient never merged.
    """
    previous = np.asarray(previous_labels, dtype=np.int32)
    collapsed = np.asarray(collapsed_labels, dtype=np.int32)
    centers = np.asarray(previous_centers, dtype=np.float64)
    if previous.shape != collapsed.shape:
        raise ValueError("collapse label maps must have identical shapes")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("previous centers must have shape (cells, 2)")

    pooled = _centers_from_labels(collapsed, measure)
    previous_flat = previous.ravel()
    if previous_flat.size and int(np.max(previous_flat)) >= len(centers):
        raise ValueError("collapse labels exceed the previous center table")
    # A first-arrival population may contain a duplicate seed which owns no
    # raster pixel.  Such a dead site remains in the canonical center table so
    # that the coordinate gauge is unchanged, but it is not a collapse cell
    # and must not prevent the represented quotient from being constructed.
    represented = np.unique(previous_flat)
    remap = np.full(len(centers), -1, dtype=np.int32)
    remap[previous_flat] = collapsed.ravel()
    if not np.array_equal(remap[previous.ravel()], collapsed.ravel()):
        raise ValueError("a previous cell cannot split during contraction")
    group_size = np.bincount(remap[represented], minlength=len(pooled))
    singleton = group_size[remap[represented]] == 1
    singleton_cell = represented[singleton]
    pooled[remap[singleton_cell]] = centers[singleton_cell]
    return np.ascontiguousarray(pooled)


@_compile
def _adjacent_cell_pairs(labels, cells):
    """Hash the undirected four-neighbour cell graph in one raster pass."""
    capacity = 1024
    target = max(16 * cells, 1024)
    while capacity < target:
        capacity *= 2
    table = np.full(capacity, -1, dtype=np.int64)
    mask = capacity - 1
    used = 0
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            first = labels[y, x]
            if x + 1 < width:
                second = labels[y, x + 1]
                if first != second:
                    low = min(first, second)
                    high = max(first, second)
                    key = np.int64(low) * cells + high
                    slot = key & mask
                    while table[slot] != -1 and table[slot] != key:
                        slot = (slot + 1) & mask
                    if table[slot] == -1:
                        if (used + 1) * 10 >= capacity * 7:
                            raise RuntimeError(
                                "texture adjacency hash exceeded its "
                                "fragmentation ceiling")
                        table[slot] = key
                        used += 1
            if y + 1 < height:
                second = labels[y + 1, x]
                if first != second:
                    low = min(first, second)
                    high = max(first, second)
                    key = np.int64(low) * cells + high
                    slot = key & mask
                    while table[slot] != -1 and table[slot] != key:
                        slot = (slot + 1) & mask
                    if table[slot] == -1:
                        if (used + 1) * 10 >= capacity * 7:
                            raise RuntimeError(
                                "texture adjacency hash exceeded its "
                                "fragmentation ceiling")
                        table[slot] = key
                        used += 1
    pairs = np.empty((used, 2), dtype=np.int32)
    index = 0
    for slot in range(capacity):
        key = table[slot]
        if key >= 0:
            pairs[index, 0] = key // cells
            pairs[index, 1] = key % cells
            index += 1
    return pairs


@_compile
def _union_components(cells, pairs):
    parent = np.arange(cells, dtype=np.int32)
    for edge in range(len(pairs)):
        first = pairs[edge, 0]
        while parent[first] != first:
            parent[first] = parent[parent[first]]
            first = parent[first]
        second = pairs[edge, 1]
        while parent[second] != second:
            parent[second] = parent[parent[second]]
            second = parent[second]
        if first == second:
            continue
        if first < second:
            parent[second] = first
        else:
            parent[first] = second
    for cell in range(cells):
        root = cell
        while parent[root] != root:
            root = parent[root]
        parent[cell] = root
    return parent


@_compile
def _compact_label_image(labels):
    """Compact nonnegative integer IDs in linear time, preserving ID order."""
    flat = labels.ravel()
    maximum = 0
    for index in range(len(flat)):
        maximum = max(maximum, flat[index])
    present = np.zeros(maximum + 1, dtype=np.uint8)
    for index in range(len(flat)):
        present[flat[index]] = 1
    remap = np.empty(maximum + 1, dtype=np.int32)
    cells = 0
    for label in range(maximum + 1):
        if present[label]:
            remap[label] = cells
            cells += 1
        else:
            remap[label] = -1
    output = np.empty(labels.shape, dtype=np.int32)
    output_flat = output.ravel()
    for index in range(len(flat)):
        output_flat[index] = remap[flat[index]]
    return output, cells


@_compile
def _affine_sufficient_statistics(labels, target, cells):
    """Fuse all pooled-affine moments into one raster traversal."""
    height, width = labels.shape
    count = np.zeros(cells, dtype=np.float64)
    gram = np.zeros((cells, 3, 3), dtype=np.float64)
    cross = np.zeros((cells, 3, 3), dtype=np.float64)
    square = np.zeros(cells, dtype=np.float64)
    for y in range(height):
        coordinate_y = (y + 0.5) / height - 0.5
        for x in range(width):
            coordinate_x = (x + 0.5) / width - 0.5
            cell = labels[y, x]
            count[cell] += 1.0
            gram[cell, 0, 0] += 1.0
            gram[cell, 0, 1] += coordinate_x
            gram[cell, 0, 2] += coordinate_y
            gram[cell, 1, 1] += coordinate_x * coordinate_x
            gram[cell, 1, 2] += coordinate_x * coordinate_y
            gram[cell, 2, 2] += coordinate_y * coordinate_y
            for channel in range(3):
                value = target[y, x, channel]
                cross[cell, 0, channel] += value
                cross[cell, 1, channel] += coordinate_x * value
                cross[cell, 2, channel] += coordinate_y * value
                square[cell] += value * value
    for cell in range(cells):
        gram[cell, 1, 0] = gram[cell, 0, 1]
        gram[cell, 2, 0] = gram[cell, 0, 2]
        gram[cell, 2, 1] = gram[cell, 1, 2]
    return count, gram, cross, square


def _mutual_model_cost_merges(
    labels: np.ndarray,
    target: np.ndarray,
    residual_energy: np.ndarray,
    cartoon_labels: np.ndarray,
    *,
    basis_terms: int,
    penalty: float,
    topology: str,
    preserve_cartoon_parent: bool = False,
) -> tuple[np.ndarray, dict]:
    """Choose disjoint flat-cell merges by a one-shot affine MDL exchange."""
    flat = np.asarray(labels, dtype=np.int32).ravel()
    cells = int(np.max(flat)) + 1
    target_value = np.ascontiguousarray(target, dtype=np.float64)
    count, gram, cross, square = _affine_sufficient_statistics(
        np.ascontiguousarray(labels, dtype=np.int32),
        target_value,
        cells,
    )
    pairs = _adjacent_cell_pairs(labels, cells)
    cartoon_parent = np.full(cells, -1, dtype=np.int32)
    cartoon_flat = np.asarray(cartoon_labels, dtype=np.int32).ravel()
    cartoon_parent[flat] = cartoon_flat
    if preserve_cartoon_parent and pairs.size:
        same_parent = (
            cartoon_parent[pairs[:, 0]] == cartoon_parent[pairs[:, 1]])
        pairs = pairs[same_parent]
    scalar_variance = float(np.median(
        np.asarray(residual_energy, dtype=np.float64)))
    allowance = (
        max(float(penalty), 0.0)
        * max(scalar_variance, 1e-30)
        * 3.0
        * max(int(basis_terms), 1)
    )
    if pairs.size:
        first, second = pairs[:, 0], pairs[:, 1]
        union_gram = gram[first] + gram[second]
        union_cross = cross[first] + cross[second]
        ridge = (
            1e-12 * np.maximum(count[first] + count[second], 1.0)
        )
        union_gram[:, 0, 0] += ridge
        union_gram[:, 1, 1] += ridge
        union_gram[:, 2, 2] += ridge
        coefficients = np.linalg.solve(union_gram, union_cross)
        cell_gram = gram.copy()
        cell_ridge = 1e-12 * np.maximum(count, 1.0)
        cell_gram[:, 0, 0] += cell_ridge
        cell_gram[:, 1, 1] += cell_ridge
        cell_gram[:, 2, 2] += cell_ridge
        cell_coefficients = np.linalg.solve(cell_gram, cross)
        merged_sse = (
            square[first] + square[second]
            - np.einsum(
                "eic,eic->e",
                union_cross,
                coefficients,
                optimize=True,
            )
        )
        independent_sse = np.maximum(
            square - np.einsum(
                "kij,kij->k",
                cross,
                cell_coefficients,
                optimize=True,
            ),
            0.0,
        )
        increase = np.maximum(
            merged_sse - independent_sse[first] - independent_sse[second],
            0.0,
        )
        score = increase / max(allowance, 1e-30)
    else:
        score = np.empty(0, dtype=np.float64)

    best_peer = np.full(cells, -1, dtype=np.int32)
    best_score = np.full(cells, np.inf)
    # Two scatter reductions replace a Python edge loop. The second reduction
    # resolves exact score ties by the smaller peer ID.
    if pairs.size:
        first, second = pairs[:, 0], pairs[:, 1]
        np.minimum.at(best_score, first, score)
        np.minimum.at(best_score, second, score)
        largest = np.iinfo(np.int32).max
        peer_for_first = np.where(
            score == best_score[first], second, largest)
        peer_for_second = np.where(
            score == best_score[second], first, largest)
        tie_peer = np.full(cells, largest, dtype=np.int32)
        np.minimum.at(tie_peer, first, peer_for_first)
        np.minimum.at(tie_peer, second, peer_for_second)
        best_peer[tie_peer != largest] = tie_peer[tie_peer != largest]
        selected_topology = str(topology).strip().lower()
        if selected_topology == "mutual":
            accepted_mask = (
                (score <= 1.0)
                & (best_peer[first] == second)
                & (best_peer[second] == first)
            )
            accepted_array = pairs[accepted_mask]
            representative = np.arange(cells, dtype=np.int32)
            representative[accepted_array[:, 1]] = accepted_array[:, 0]
        elif selected_topology == "best_graph":
            cell = np.arange(cells, dtype=np.int32)
            selected = (best_peer >= 0) & (best_score <= 1.0)
            selected_pairs = np.column_stack((
                np.minimum(cell[selected], best_peer[selected]),
                np.maximum(cell[selected], best_peer[selected]),
            ))
            accepted_array = np.unique(selected_pairs, axis=0)
            representative = _union_components(cells, accepted_array)
        else:
            raise ValueError(
                "texture_merge_topology must be 'mutual' or 'best_graph'")
    else:
        accepted_array = np.empty((0, 2), dtype=np.int32)
        representative = np.arange(cells, dtype=np.int32)
    crossed = (
        int(np.count_nonzero(
            cartoon_parent[accepted_array[:, 0]]
            != cartoon_parent[accepted_array[:, 1]]
        ))
        if accepted_array.size else 0
    )
    return representative, {
        "merge_count": int(cells - len(np.unique(representative))),
        "merge_candidate_count": int(len(pairs)),
        "cross_parent_merge_count": crossed,
        "merge_allowance": allowance,
        "merge_model": "pooled_affine",
        "robust_scalar_variance": scalar_variance,
        "merge_pairs": accepted_array,
    }


@_compile
def _joint_leaf_statistics(
    labels,
    cartoon_target,
    cartoon_fit,
    texture_target,
    texture_fit,
    cells,
):
    """Accumulate both leaf objectives in one raster traversal."""
    height, width = labels.shape
    count = np.zeros(cells, dtype=np.float64)
    gram = np.zeros((cells, 3, 3), dtype=np.float64)
    cartoon_cross = np.zeros((cells, 3, 3), dtype=np.float64)
    texture_cross = np.zeros((cells, 3, 3), dtype=np.float64)
    cartoon_square = np.zeros(cells, dtype=np.float64)
    texture_square = np.zeros(cells, dtype=np.float64)
    cartoon_independent = np.zeros(cells, dtype=np.float64)
    texture_independent = np.zeros(cells, dtype=np.float64)
    residual_energy = np.empty(height * width, dtype=np.float64)
    pixel = 0
    for y in range(height):
        coordinate_y = (y + 0.5) / height - 0.5
        for x in range(width):
            coordinate_x = (x + 0.5) / width - 0.5
            cell = labels[y, x]
            count[cell] += 1.0
            gram[cell, 0, 0] += 1.0
            gram[cell, 0, 1] += coordinate_x
            gram[cell, 0, 2] += coordinate_y
            gram[cell, 1, 1] += coordinate_x * coordinate_x
            gram[cell, 1, 2] += coordinate_x * coordinate_y
            gram[cell, 2, 2] += coordinate_y * coordinate_y
            total_error = 0.0
            for channel in range(3):
                cartoon_value = cartoon_target[y, x, channel]
                texture_value = texture_target[y, x, channel]
                cartoon_cross[cell, 0, channel] += cartoon_value
                cartoon_cross[cell, 1, channel] += (
                    coordinate_x * cartoon_value)
                cartoon_cross[cell, 2, channel] += (
                    coordinate_y * cartoon_value)
                texture_cross[cell, 0, channel] += texture_value
                texture_cross[cell, 1, channel] += (
                    coordinate_x * texture_value)
                texture_cross[cell, 2, channel] += (
                    coordinate_y * texture_value)
                cartoon_square[cell] += cartoon_value * cartoon_value
                texture_square[cell] += texture_value * texture_value
                cartoon_error = (
                    cartoon_value - cartoon_fit[y, x, channel])
                texture_error = (
                    texture_value - texture_fit[y, x, channel])
                cartoon_independent[cell] += cartoon_error * cartoon_error
                texture_independent[cell] += texture_error * texture_error
                total_error += (
                    cartoon_value + texture_value
                    - cartoon_fit[y, x, channel]
                    - texture_fit[y, x, channel]
                ) ** 2
            residual_energy[pixel] = total_error / 3.0
            pixel += 1
    for cell in range(cells):
        gram[cell, 1, 0] = gram[cell, 0, 1]
        gram[cell, 2, 0] = gram[cell, 0, 2]
        gram[cell, 2, 1] = gram[cell, 1, 2]
    return (
        count,
        gram,
        cartoon_cross,
        texture_cross,
        cartoon_square,
        texture_square,
        cartoon_independent,
        texture_independent,
        residual_energy,
    )


def _joint_leaf_collapse(
    structural_labels: np.ndarray,
    texture_labels: np.ndarray,
    target_lab: np.ndarray,
    cartoon_lab: np.ndarray,
    texture_target: np.ndarray,
    texture_fit: np.ndarray,
    *,
    penalty: float,
    basis_terms: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collapse compatible one-child structural and texture leaves together.

    Eligibility is topological: both incident structural owners must have
    exactly one final texture child, and that child must not cross a parent
    boundary. Compatibility is then the increase in the *joint* pooled
    cartoon and texture affine objectives. The accepted best-edge graph is
    contracted once; no region descent or repeated image scan is performed.
    """
    structural = np.ascontiguousarray(structural_labels, dtype=np.int32)
    texture = np.ascontiguousarray(texture_labels, dtype=np.int32)
    structural_cells = int(np.max(structural)) + 1
    texture_cells = int(np.max(texture)) + 1
    flat_structural = structural.ravel()
    flat_texture = texture.ravel()

    parent_min = np.full(texture_cells, structural_cells, dtype=np.int32)
    parent_max = np.full(texture_cells, -1, dtype=np.int32)
    np.minimum.at(parent_min, flat_texture, flat_structural)
    np.maximum.at(parent_max, flat_texture, flat_structural)
    pure = parent_min == parent_max
    pure_texture = np.flatnonzero(pure)
    child_count = np.bincount(
        parent_min[pure_texture], minlength=structural_cells)
    only_child = np.full(structural_cells, -1, dtype=np.int32)
    only_child[parent_min[pure_texture]] = pure_texture
    eligible = child_count == 1

    all_pairs = _adjacent_cell_pairs(structural, structural_cells)
    pairs = all_pairs
    if pairs.size:
        pairs = pairs[eligible[pairs[:, 0]] & eligible[pairs[:, 1]]]

    (
        count,
        gram,
        cartoon_cross,
        texture_cross,
        cartoon_square,
        texture_square,
        cartoon_independent,
        texture_independent,
        residual_energy,
    ) = _joint_leaf_statistics(
        structural,
        np.ascontiguousarray(target_lab, dtype=np.float64),
        np.ascontiguousarray(cartoon_lab, dtype=np.float64),
        np.ascontiguousarray(texture_target, dtype=np.float64),
        np.ascontiguousarray(texture_fit, dtype=np.float64),
        structural_cells,
    )
    robust_variance = float(np.median(residual_energy))
    allowance = (
        max(float(penalty), 0.0)
        * max(robust_variance, 1e-30)
        * 3.0
        * max(int(basis_terms), 1)
    )

    if pairs.size:
        first, second = pairs[:, 0], pairs[:, 1]
        union_gram = gram[first] + gram[second]
        # These are symmetric positive-semidefinite Gram blocks. A general
        # batched SVD computes two unused singular-vector matrices for every
        # candidate edge. Their absolute eigenvalues are the same singular
        # values, so the Hermitian route is the identical 2-norm condition
        # number without that cubic baggage.
        eigenvalue = np.abs(np.linalg.eigvalsh(union_gram))
        condition = (
            np.max(eigenvalue, axis=1)
            / np.maximum(np.min(eigenvalue, axis=1), 1e-300)
        )
        ridge = (
            np.finfo(np.float64).eps
            * np.maximum(count[first] + count[second], 1.0)
        )
        union_gram[:, 0, 0] += ridge
        union_gram[:, 1, 1] += ridge
        union_gram[:, 2, 2] += ridge
        union_cartoon_cross = cartoon_cross[first] + cartoon_cross[second]
        union_texture_cross = texture_cross[first] + texture_cross[second]
        cartoon_coefficients = np.linalg.solve(
            union_gram, union_cartoon_cross)
        texture_coefficients = np.linalg.solve(
            union_gram, union_texture_cross)
        pooled_cartoon = (
            cartoon_square[first] + cartoon_square[second]
            - np.einsum(
                "eic,eic->e",
                union_cartoon_cross,
                cartoon_coefficients,
                optimize=True,
            )
        )
        pooled_texture = (
            texture_square[first] + texture_square[second]
            - np.einsum(
                "eic,eic->e",
                union_texture_cross,
                texture_coefficients,
                optimize=True,
            )
        )
        independent = (
            cartoon_independent[first]
            + cartoon_independent[second]
            + texture_independent[first]
            + texture_independent[second]
        )
        increase = np.maximum(
            pooled_cartoon + pooled_texture - independent,
            0.0,
        )
        represented_energy = (
            cartoon_square[first] + cartoon_square[second]
            + texture_square[first] + texture_square[second]
        )
        numerical_allowance = (
            128.0 * np.finfo(np.float64).eps
            * np.maximum(condition, 1.0)
            * np.maximum(represented_energy, 1.0)
        )
        score = increase / np.maximum(
            allowance + numerical_allowance, 1e-300)
        score[condition > 1.0 / np.sqrt(np.finfo(np.float64).eps)] = np.inf

        best_score = np.full(structural_cells, np.inf)
        best_peer = np.full(structural_cells, -1, dtype=np.int32)
        np.minimum.at(best_score, first, score)
        np.minimum.at(best_score, second, score)
        largest = np.iinfo(np.int32).max
        peer_for_first = np.where(
            score == best_score[first], second, largest)
        peer_for_second = np.where(
            score == best_score[second], first, largest)
        tie_peer = np.full(structural_cells, largest, dtype=np.int32)
        np.minimum.at(tie_peer, first, peer_for_first)
        np.minimum.at(tie_peer, second, peer_for_second)
        best_peer[tie_peer != largest] = tie_peer[tie_peer != largest]
        cell = np.arange(structural_cells, dtype=np.int32)
        selected = (best_peer >= 0) & (best_score <= 1.0)
        # Pairwise compatibility is not transitive. A best-edge forest can
        # join dozens of acceptable pairs into one uncertified model (the
        # coffee-cup penumbra previously formed a 32-leaf component). Keep
        # only reciprocal best edges. Every resulting component is therefore
        # exactly the two-cell model measured above, with no extra raster
        # scan, iterative merge, or unmeasured transitive closure.
        peer = np.where(selected, best_peer, cell)
        mutual = selected & selected[peer] & (best_peer[peer] == cell)
        accepted = np.unique(np.column_stack((
            np.minimum(cell[mutual], best_peer[mutual]),
            np.maximum(cell[mutual], best_peer[mutual]),
        )), axis=0)
    else:
        score = np.empty(0, dtype=np.float64)
        accepted = np.empty((0, 2), dtype=np.int32)

    fit_compatible = int(np.count_nonzero(score <= 1.0))
    ill_conditioned = int(np.count_nonzero(~np.isfinite(score)))

    if accepted.size:
        structural_representative = _union_components(
            structural_cells, accepted)
        collapsed_structural, final_structural_cells = _compact_label_image(
            np.ascontiguousarray(
                structural_representative[structural], dtype=np.int32))
        texture_pairs = np.column_stack((
            only_child[accepted[:, 0]],
            only_child[accepted[:, 1]],
        ))
        texture_pairs.sort(axis=1)
        texture_representative = _union_components(
            texture_cells, texture_pairs)
        collapsed_texture, final_texture_cells = _compact_label_image(
            np.ascontiguousarray(
                texture_representative[texture], dtype=np.int32))
    else:
        collapsed_structural = structural.copy()
        collapsed_texture = texture.copy()
        final_structural_cells = structural_cells
        final_texture_cells = texture_cells

    return collapsed_structural, collapsed_texture, {
        "enabled": True,
        "total_structural_cells": int(structural_cells),
        "structural_cells_without_pure_child": int(np.count_nonzero(
            child_count == 0)),
        "structural_cells_with_multiple_children": int(np.count_nonzero(
            child_count > 1)),
        "eligible_structural_cells": int(np.count_nonzero(eligible)),
        "pure_texture_cells": int(len(pure_texture)),
        "cross_parent_texture_cells": int(texture_cells - len(pure_texture)),
        "total_adjacent_pairs": int(len(all_pairs)),
        "candidate_pairs": int(len(pairs)),
        "fit_compatible_pairs": fit_compatible,
        "ill_conditioned_pairs": ill_conditioned,
        "accepted_pairs": int(len(accepted)),
        "initial_structural_cells": int(structural_cells),
        "final_structural_cells": int(final_structural_cells),
        "structural_cells_removed": int(
            structural_cells - final_structural_cells),
        "initial_texture_cells": int(texture_cells),
        "final_texture_cells": int(final_texture_cells),
        "texture_cells_removed": int(texture_cells - final_texture_cells),
        "robust_residual_variance": robust_variance,
        "model_allowance": allowance,
        "maximum_accepted_score": (
            float(np.max(score[score <= 1.0]))
            if np.any(score <= 1.0) else 0.0
        ),
        "collapse_topology": "mutual_best_edge",
        "accepted_pairs_array": accepted,
    }


def _paired_metric_split_partition(
    parent_labels: np.ndarray,
    centers: np.ndarray,
    parent_of_centers: np.ndarray,
    geometry: dict,
    metric_strength: float,
    *,
    freeze_pair_metric: bool = False,
) -> np.ndarray:
    """Assign all two-child splits by closed-form metric bisectors.

    For two sites ``a,b`` and one frozen positive metric ``M``, equality of
    ``(x-a)'M(x-a)`` and ``(x-b)'M(x-b)`` is a half-space.  Reduced Meyer's
    first resolvent can leave a coherent transverse tail beside a thin
    structure.  Letting ``M`` vary at every incident pixel then bends this
    nominal bisector toward that tail and may leave almost all unexplained
    energy in one child.

    ``freeze_pair_metric`` integrates the BFFT tensor along the literal site
    chord with two-point Gauss--Legendre quadrature and uses that one tensor
    for the pair.  This is the deterministic geodesic-line approximation: it
    preserves anisotropy, restores one affine bisector, and needs two tensor
    samples per split rather than one tensor fetch per incident pixel.  The
    full jump-measure path retains the spatially varying control, whose
    established quotient is intentionally unchanged.
    """
    labels = np.asarray(parent_labels, dtype=np.int32)
    flat = labels.ravel()
    initial_cells = int(np.max(flat)) + 1
    parent_of = np.asarray(parent_of_centers, dtype=np.int32)
    child_site = np.full(initial_cells, -1, dtype=np.int32)
    additions = np.arange(initial_cells, len(centers), dtype=np.int32)
    child_site[parent_of[additions]] = additions
    child = child_site[flat]
    active = child >= 0
    result = flat.copy()
    if not np.any(active):
        return result.reshape(labels.shape)

    height, width = labels.shape
    pixel = np.flatnonzero(active)
    parent = flat[pixel]
    child_at_pixel = child[pixel]
    x = pixel % width + 0.5
    y = pixel // width + 0.5
    center = np.asarray(centers, dtype=np.float64)
    minus_x = x - center[parent, 0] * width
    minus_y = y - center[parent, 1] * height
    plus_x = x - center[child_at_pixel, 0] * width
    plus_y = y - center[child_at_pixel, 1] * height

    trace_scale = (
        float(geometry["metric_trace_p90"])
        if "metric_trace_p90" in geometry
        else max(float(np.percentile(
            np.asarray(geometry["precision_xx"])
            + np.asarray(geometry["precision_yy"]),
            90.0,
        )), 1e-12)
    )
    strength = (
        max(float(metric_strength), 0.0)
        * float(geometry["max_support_px"]) ** 2
        / max(trace_scale, 1e-30)
    )
    if freeze_pair_metric:
        parent_site = np.flatnonzero(child_site >= 0)
        child_for_parent = child_site[parent_site]
        half_inverse_sqrt_three = 0.5 / math.sqrt(3.0)
        quadrature = np.array((
            0.5 - half_inverse_sqrt_three,
            0.5 + half_inverse_sqrt_three,
        ))[:, None]
        parent_x = center[parent_site, 0] * width
        parent_y = center[parent_site, 1] * height
        chord_x = center[child_for_parent, 0] * width - parent_x
        chord_y = center[child_for_parent, 1] * height - parent_y
        pair_x = parent_x + quadrature * chord_x
        pair_y = parent_y + quadrature * chord_y
        sample_x = np.clip(pair_x.astype(np.int64), 0, width - 1)
        sample_y = np.clip(pair_y.astype(np.int64), 0, height - 1)

        def chord_average(name):
            field = np.asarray(geometry[name], dtype=np.float64)
            return 0.5 * np.sum(field[sample_y, sample_x], axis=0)

        pair_qxx = np.zeros(initial_cells, dtype=np.float64)
        pair_qxy = np.zeros(initial_cells, dtype=np.float64)
        pair_qyy = np.zeros(initial_cells, dtype=np.float64)
        pair_qxx[parent_site] = chord_average("precision_xx")
        pair_qxy[parent_site] = chord_average("precision_xy")
        pair_qyy[parent_site] = chord_average("precision_yy")
        qxx = pair_qxx[parent]
        qxy = pair_qxy[parent]
        qyy = pair_qyy[parent]
    else:
        qxx = np.asarray(
            geometry["precision_xx"], dtype=np.float64).ravel()[pixel]
        qxy = np.asarray(
            geometry["precision_xy"], dtype=np.float64).ravel()[pixel]
        qyy = np.asarray(
            geometry["precision_yy"], dtype=np.float64).ravel()[pixel]
    mxx = 1.0 + strength * qxx
    mxy = strength * qxy
    myy = 1.0 + strength * qyy
    minus_cost = (
        mxx * minus_x * minus_x
        + 2.0 * mxy * minus_x * minus_y
        + myy * minus_y * minus_y
    )
    plus_cost = (
        mxx * plus_x * plus_x
        + 2.0 * mxy * plus_x * plus_y
        + myy * plus_y * plus_y
    )
    result[pixel[plus_cost < minus_cost]] = child_at_pixel[
        plus_cost < minus_cost]
    return result.reshape(labels.shape)


def _refresh_flat_texture_interfaces(
    labels: np.ndarray,
    centers: np.ndarray,
    geometry: dict,
    *,
    radius: int,
    sweeps: int,
    strength: float,
    confidence_threshold: float,
    error_ratio: np.ndarray,
    error_ratio_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Move final texture interfaces in one fixed full-resolution band."""
    flat = np.asarray(labels, dtype=np.int32).ravel()
    cells = len(centers)
    height, width = labels.shape
    pixel = np.arange(flat.size, dtype=np.int64)
    x = pixel % width
    y = pixel // width
    dx = x - (centers[:, 0] * width - 0.5)[flat]
    dy = y - (centers[:, 1] * height - 0.5)[flat]
    squared = dx * dx + dy * dy
    closest = np.full(cells, np.inf)
    np.minimum.at(closest, flat, squared)
    closest_seed = (
        squared <= closest[flat] + 1e-12
    ).reshape(labels.shape)

    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    confidence = np.asarray(
        geometry["boundary_confidence"], dtype=np.float64)
    boundary &= confidence >= max(float(confidence_threshold), 0.0)
    boundary &= (
        np.asarray(error_ratio, dtype=np.float64)
        >= max(float(error_ratio_threshold), 0.0)
    )
    band = binary_dilation(boundary, max(int(radius), 1))
    fixed = ~band
    core_count = np.bincount(flat[fixed.ravel()], minlength=cells)
    missing = core_count == 0
    fixed |= closest_seed & missing[flat].reshape(labels.shape)
    tensor = (
        np.asarray(geometry["boundary_xx"], dtype=np.float64),
        np.asarray(geometry["boundary_xy"], dtype=np.float64),
        np.asarray(geometry["boundary_yy"], dtype=np.float64),
    )
    refreshed, distance = _upgrade_owner_band_tensor(
        np.ascontiguousarray(labels, dtype=np.int32),
        np.ascontiguousarray(fixed),
        np.ascontiguousarray(band),
        *_metric_coefficients(tensor, strength),
        max(int(sweeps), 1),
    )
    changed = refreshed != labels
    return refreshed, {
        "interface_band_pixels": int(np.count_nonzero(band)),
        "interface_changed_pixels": int(np.count_nonzero(changed)),
        "interface_forced_seeds": int(np.count_nonzero(missing)),
        "interface_unreached_pixels": int(
            np.count_nonzero(~np.isfinite(distance))),
    }


def _flat_texture_cleanup(
    labels: np.ndarray,
    centers: np.ndarray,
    forest: dict,
    geometry: dict,
    target: np.ndarray,
    affine_fit: np.ndarray,
    cartoon_labels: np.ndarray,
    config: SegmentingV3Config,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Forget cartoon parents, then split hot cells and merge cold neighbours."""
    initial_cells = len(centers)
    residual_energy = np.mean(np.square(target - affine_fit), axis=2)
    proposed_centers, split = reverse_residual_refill(
        labels,
        centers,
        forest,
        residual_energy,
        geometry["measure"],
        error_ratio_threshold=float(config.texture_split_error_ratio),
        peak_error_ratio_threshold=float(
            config.texture_split_peak_error_ratio),
        return_distance_threshold=float(config.texture_split_return_extent),
        minimum_region_pixels=int(config.texture_split_minimum_pixels),
        safety_cells=max(int(config.texture_safety_cells), 1),
    )
    split_transport = str(config.texture_split_transport).strip().lower()
    if split["split_count"] and split_transport == "paired_metric":
        split_labels = _paired_metric_split_partition(
            labels,
            proposed_centers,
            split["parent_of_centers"],
            geometry,
            config.texture_split_metric_strength,
            freeze_pair_metric=(
                str(config.meyer_operator).strip().lower()
                == "legacy_one_pass"
            ),
        )
    elif split["split_count"] and split_transport == "local_eikonal":
        proposed_centers, split_forest = local_hard_partition_with_forest(
            proposed_centers,
            split["parent_of_centers"],
            labels,
            build_edge_costs(geometry, config.metric_strength),
        )
        split_labels = np.asarray(split_forest["labels"], dtype=np.int32)
    elif split["split_count"]:
        raise ValueError(
            "texture_split_transport must be 'paired_metric' or "
            "'local_eikonal'")
    else:
        split_labels = np.asarray(labels, dtype=np.int32)

    parent_of_site = np.asarray(split["parent_of_centers"], dtype=np.int32)
    parent_was_split = np.zeros(initial_cells, dtype=bool)
    parent_was_split[split["split_ids"]] = True
    key = parent_of_site.astype(np.int64)
    # Begin with both children distinct. The common affine objective may
    # subsequently reunite them, but only if the split bought no model cost.
    child = parent_was_split[parent_of_site]
    key[child] = initial_cells + np.flatnonzero(child)
    _, site_remap = np.unique(key, return_inverse=True)
    cleaned = site_remap[split_labels]
    cleaned, _ = _compact_label_image(
        np.ascontiguousarray(cleaned.reshape(labels.shape), dtype=np.int32))

    merge_rounds = []
    merge_count = 0
    merge_candidates = 0
    cross_parent_merges = 0
    merge_allowance = 0.0
    robust_variance = float(np.median(residual_energy))
    for _ in range(max(int(config.texture_merge_rounds), 0)):
        representative, round_diagnostic = _mutual_model_cost_merges(
            cleaned,
            target,
            residual_energy,
            cartoon_labels,
            basis_terms=3 + max(int(config.nested_texture_ridges), 0),
            penalty=float(config.texture_merge_penalty),
            topology=config.texture_merge_topology,
            preserve_cartoon_parent=(
                str(config.structural_topology).strip().lower()
                == "canonical_v2"
                and not bool(config.texture_cross_structural_merges)
            ),
        )
        accepted = int(round_diagnostic["merge_count"])
        merge_rounds.append(round_diagnostic)
        merge_count += accepted
        merge_candidates += int(
            round_diagnostic["merge_candidate_count"])
        cross_parent_merges += int(
            round_diagnostic["cross_parent_merge_count"])
        merge_allowance = float(round_diagnostic["merge_allowance"])
        robust_variance = float(
            round_diagnostic["robust_scalar_variance"])
        if accepted == 0:
            break
        merged = representative[cleaned]
        cleaned, _ = _compact_label_image(
            np.ascontiguousarray(merged, dtype=np.int32))

    merge = {
        "merge_count": int(merge_count),
        "merge_candidate_count": int(merge_candidates),
        "cross_parent_merge_count": int(cross_parent_merges),
        "merge_allowance": merge_allowance,
        "robust_scalar_variance": robust_variance,
        "merge_model": "fixed_depth_pooled_affine",
        "merge_rounds": merge_rounds,
    }
    cleaned_centers = _centers_from_labels(cleaned, geometry["measure"])
    interface = {
        "interface_band_pixels": 0,
        "interface_changed_pixels": 0,
        "interface_forced_seeds": 0,
        "interface_unreached_pixels": 0,
    }
    if (
        bool(config.texture_interface_refresh)
    ):
        cleaned, interface = _refresh_flat_texture_interfaces(
            cleaned,
            cleaned_centers,
            geometry,
            radius=config.texture_interface_radius,
            sweeps=config.texture_interface_sweeps,
            strength=config.texture_interface_strength,
            confidence_threshold=config.texture_interface_confidence,
            error_ratio=split["error_ratio_map"],
            error_ratio_threshold=config.texture_interface_error_ratio,
        )
        cleaned_centers = _centers_from_labels(cleaned, geometry["measure"])
    diagnostic = {
        "enabled": True,
        "split_transport": split_transport,
        "initial_cells": int(initial_cells),
        "final_cells": int(len(cleaned_centers)),
        **split,
        **merge,
        **interface,
    }
    return cleaned, cleaned_centers, diagnostic


def _owner_eikonal_coordinates(
    labels: np.ndarray,
    straight_coordinates: tuple[np.ndarray, ...],
    coordinate_names: tuple[str, ...],
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    sweeps: int,
    strength: float,
) -> tuple[tuple[np.ndarray, ...], dict]:
    flat = labels.ravel()
    cells = int(np.max(flat)) + 1
    edges = _metric_edge_costs(tensor, strength)
    spacing = max(math.sqrt(labels.size / max(cells, 1)), 1e-30)
    coordinates = []
    fallback_counts = []
    for straight in straight_coordinates:
        absolute = np.abs(straight)
        closest = np.full(cells, np.inf)
        np.minimum.at(closest, flat, absolute)
        seed = (
            absolute <= closest[flat] + 1e-12
        ).reshape(labels.shape)
        distance = _owner_masked_distance(
            labels,
            np.ascontiguousarray(seed),
            *edges,
            max(int(sweeps), 1),
        ).ravel()
        # A transported owner can contain raster-disconnected islands (most
        # commonly diagonal one-pixel pieces after lifting).  They have no
        # legal path to the owner's selected zero-line.  Preserve the exact
        # straight coordinate there instead of inventing a cross-owner path.
        unreachable = ~np.isfinite(distance)
        fallback_counts.append(int(np.count_nonzero(unreachable)))
        distance[unreachable] = (
            np.abs(straight[unreachable]) * spacing)
        signed = np.where(straight < 0.0, -distance, distance) / spacing
        coordinates.append(np.ascontiguousarray(signed))
    diagnostics = {
        f"{name}_fallback_pixels": fallback
        for name, fallback in zip(coordinate_names, fallback_counts)
    }
    diagnostics["total_pixels"] = int(labels.size)
    return tuple(coordinates), diagnostics


def _half_cartoon_scaffold(
    rgb: np.ndarray,
    target_lab: np.ndarray,
    config: SegmentingV3Config,
) -> dict:
    height, width = rgb.shape[:2]
    low_height = max(int(round(height * config.cartoon_scale)), 2)
    low_width = max(int(round(width * config.cartoon_scale)), 2)
    low_rgb = _resize(rgb, (low_height, low_width), order=1)
    low_lab = srgb_to_lab(low_rgb)

    phase = time.perf_counter()
    geometry = build_frozen_geometry(
        low_rgb,
        target_lab=low_lab,
        tgfd_sweeps=max(int(config.meyer_sweeps), 1),
        meyer_operator=config.meyer_operator,
        flow_sweeps=1,
        texture_support_weight=0.0,
        glass_support_weight=0.0,
        null_evidence_strength=1.0,
        threads=max(int(config.threads), 0),
    )
    geometry_ms = 1000.0 * (time.perf_counter() - phase)

    phase = time.perf_counter()
    centers, population = emit_density_population(
        geometry,
        safety_cells=max(int(config.safety_cells), 1),
    )
    metric = prepare_continuous_metric(*_metric_fields(
        geometry,
        config.metric_strength,
        config.boundary_jump_strength,
    ))
    forest = continuous_first_partition_prepared(centers, metric)
    transport_ms = 1000.0 * (time.perf_counter() - phase)
    low_labels = forest["labels"]

    phase = time.perf_counter()
    cartoon_target = low_lab.copy()
    cartoon_target[..., 0] = geometry["cartoon"]
    native = hard_affine_fit_native(low_labels, cartoon_target)
    if native is None:
        raise RuntimeError("version 3.0 requires the native hard affine fitter")
    low_cartoon_lab = native[-1]
    fit_ms = 1000.0 * (time.perf_counter() - phase)
    lifted_cartoon_lab = _resize(low_cartoon_lab, (height, width), order=1)
    lifted_labels = _lift_labels(low_labels, (height, width))
    low_metric = _metric_fields(
        geometry,
        config.metric_strength,
        config.boundary_jump_strength,
    )
    full_metric = tuple(
        _resize(field, (height, width), order=1)
        for field in low_metric
    )

    phase = time.perf_counter()
    if bool(config.owner_upgrade):
        labels, owner_upgrade = _upgrade_full_resolution_owners(
            lifted_labels,
            centers,
            target_lab,
            full_metric,
            mode=config.owner_upgrade_mode,
            radius=config.owner_upgrade_radius,
            sweeps=config.owner_upgrade_sweeps,
            strength=config.owner_upgrade_strength,
            cartoon_strength=config.owner_upgrade_cartoon_strength,
        )
    else:
        labels = lifted_labels
        owner_upgrade = {
            "band_pixels": 0,
            "changed_pixels": 0,
            "forced_owner_seeds": 0,
            "unreached_pixels": 0,
            "mode": "disabled",
        }
    owner_upgrade_ms = 1000.0 * (time.perf_counter() - phase)

    phase = time.perf_counter()
    if bool(config.cartoon_full_refit):
        full_native = hard_affine_fit_native(labels, target_lab)
        if full_native is None:
            raise RuntimeError(
                "version 3.0 requires the native hard affine fitter")
        strength = np.clip(
            float(config.cartoon_refit_strength), 0.0, 1.0)
        cartoon_lab = (
            (1.0 - strength) * lifted_cartoon_lab
            + strength * full_native[-1]
        )
    else:
        cartoon_lab = lifted_cartoon_lab
    refit_ms = 1000.0 * (time.perf_counter() - phase)
    return {
        "labels": labels,
        "centers": centers,
        "geometry": geometry,
        "forest": forest,
        "cartoon_lab": cartoon_lab,
        "lifted_cartoon_lab": lifted_cartoon_lab,
        "lifted_labels": lifted_labels,
        "low_labels": low_labels,
        "population": population,
        "owner_upgrade": owner_upgrade,
        "geometry_ms": geometry_ms,
        "transport_ms": transport_ms,
        "fit_ms": fit_ms,
        "owner_upgrade_ms": owner_upgrade_ms,
        "refit_ms": refit_ms,
        "prepared_texture_geometry": None,
        "structural_population_geometry": None,
        "characteristic": {
            "trace": [],
            "initial_centers": centers.copy(),
            "final_centers": centers.copy(),
            "requested_passes": 0,
            "effective_passes": 0,
            "resolved_core": True,
            "mean_pixels_per_site": float(
                low_labels.size / max(len(centers), 1)),
        },
        "transport_model": "half_continuous",
        "topology": "half_cartoon",
    }


def _fit_canonical_structural_model(
    labels: np.ndarray,
    centers: np.ndarray,
    target_lab: np.ndarray,
    geometry: dict,
    config: SegmentingV3Config,
    *,
    frame_cells: int | None = None,
) -> np.ndarray:
    """Fit the canonical affine-plus-ridge model on one fixed quotient."""
    native = hard_affine_fit_native(labels, target_lab)
    if native is None:
        raise RuntimeError("version 3.0 requires the native hard affine fitter")
    flat, basis, count, radius, _centroid, cartoon_lab = native
    active_basis = basis
    normal, _tangent = _cell_frame(
        labels,
        centers,
        (
            np.asarray(geometry["boundary_xx"], dtype=np.float64),
            np.asarray(geometry["boundary_xy"], dtype=np.float64),
            np.asarray(geometry["boundary_yy"], dtype=np.float64),
        ),
        normalization_cells=frame_cells,
    )
    target_flat = target_lab.reshape(-1, 3)
    ones = np.ones(flat.size, dtype=np.float64)
    for _ in range(max(int(config.structural_ridges), 0)):
        residual = target_flat - cartoon_lab.reshape(-1, 3)
        _score, offset = measure_paired_offsets(
            flat,
            ones,
            residual,
            normal,
            len(centers),
            bins=max(int(config.offset_bins), 3),
        )
        ridge = np.clip(
            float(config.ridge_kappa) * (normal - offset[flat]),
            -1.0,
            1.0,
        )
        active_basis = np.column_stack((active_basis, ridge))
        refit = hard_basis_refit_native(
            flat, active_basis, target_flat, count, radius)
        if refit is None:
            raise RuntimeError(
                "version 3.0 requires the native basis refitter")
        cartoon_lab = refit.reshape(target_lab.shape)
    return cartoon_lab


def _canonical_v2_scaffold(
    rgb: np.ndarray,
    target_lab: np.ndarray,
    config: SegmentingV3Config,
) -> dict:
    """Canonical sparse structural quotient with a full-resolution readout."""
    phase = time.perf_counter()
    geometry = build_frozen_geometry(
        rgb,
        target_lab=target_lab,
        tgfd_sweeps=max(int(config.meyer_sweeps), 1),
        meyer_operator=config.meyer_operator,
        flow_sweeps=max(int(config.structural_flow_sweeps), 1),
        texture_support_weight=max(
            float(config.texture_support_weight), 0.0),
        glass_support_weight=max(
            float(config.structural_glass_weight), 0.0),
        null_evidence_strength=0.5,
        threads=max(int(config.threads), 0),
    )
    if bool(config.texture_curvature_population):
        geometry = curvature_limited_geometry(geometry)
    structural_geometry = reweight_frozen_support(
        geometry,
        texture_support_weight=0.0,
        glass_support_weight=0.0,
        null_evidence_strength=(
            config.structural_null_evidence_strength),
    )
    if bool(config.texture_curvature_population):
        structural_geometry = curvature_limited_geometry(
            structural_geometry)
    structural_geometry = dict(structural_geometry)
    structural_geometry["implied_cells"] = (
        float(structural_geometry["implied_cells"])
        * max(float(config.structural_population_scale), 1e-6)
    )
    structural_geometry = _structural_measure_below_full(
        structural_geometry, geometry)
    geometry_ms = 1000.0 * (time.perf_counter() - phase)

    phase = time.perf_counter()
    allocation_geometry = restrict_geometry(
        structural_geometry,
        max(int(config.structural_allocation_side), 2),
    )
    centers, population = emit_density_population(
        allocation_geometry,
        safety_cells=max(int(config.structural_safety_cells), 1),
    )
    represented_structural_geometry = dict(structural_geometry)
    represented_structural_geometry["implied_cells"] = float(
        population["commanded_cells"])
    initial_centers = centers.copy()

    same_grid = (
        allocation_geometry["measure"].shape == geometry["measure"].shape
    )
    requested_transport = str(
        config.structural_full_transport).strip().lower()
    if requested_transport == "auto":
        full_transport = "continuous" if same_grid else "bucket_graph"
    elif requested_transport in ("continuous", "bucket_graph"):
        full_transport = requested_transport
    else:
        raise ValueError(
            "structural_full_transport must be 'auto', 'continuous', or "
            "'bucket_graph'")

    requested_passes = max(
        int(config.structural_characteristic_passes), 0)
    mean_pixels_per_site = float(
        allocation_geometry["measure"].size / max(len(centers), 1))
    characteristic_core_area = (
        math.pi
        * max(float(config.structural_characteristic_core_radius), 0.0) ** 2
    )
    resolved_core = mean_pixels_per_site >= characteristic_core_area
    effective_passes = requested_passes if resolved_core else 0

    # A characteristic update and a same-grid continuous readout both need
    # the allocation-grid forest.  A full-grid bucket readout does not.  In
    # particular, do not prepare and march a coarse continuous metric merely
    # to reject a core whose footprint is larger than its average cell.
    allocation_transport_geometry = None
    allocation_metric = None
    allocation_forest = None
    need_allocation_forest = (
        effective_passes > 0
        or (full_transport == "continuous" and same_grid)
    )
    if need_allocation_forest:
        allocation_transport_geometry = restrict_geometry(
            geometry,
            max(int(config.structural_allocation_side), 2),
        )
        allocation_metric = prepare_continuous_metric(*_metric_fields(
            allocation_transport_geometry,
            config.metric_strength,
            config.boundary_jump_strength,
        ))
        allocation_forest = continuous_first_partition_prepared(
            centers,
            allocation_metric,
            compact=(effective_passes == 0),
            source_gradients=False,
        )

    characteristic_trace = []
    for iteration in range(effective_passes):
        centers, allocation_forest, diagnostic = (
            safe_characteristic_site_step(
                centers,
                allocation_forest,
                allocation_metric,
                allocation_geometry["measure"],
                trust_fraction=(
                    config.structural_characteristic_trust_fraction),
                core_radius_px=(
                    config.structural_characteristic_core_radius),
                compact_result=(iteration + 1 >= effective_passes),
            )
        )
        diagnostic["iteration"] = iteration + 1
        characteristic_trace.append(diagnostic)
        if not diagnostic["accepted"]:
            break
    if full_transport == "continuous" and same_grid:
        forest = allocation_forest
    elif full_transport == "continuous":
        metric = prepare_continuous_metric(*_metric_fields(
            geometry,
            config.metric_strength,
            config.boundary_jump_strength,
        ))
        forest = continuous_first_partition_prepared(
            centers, metric, compact=True)
    elif full_transport == "bucket_graph":
        forest = hard_first_partition_with_forest(
            centers,
            build_geometry_edge_costs(
                geometry,
                config.metric_strength,
                config.boundary_jump_strength,
            ),
        )
    else:
        raise AssertionError("unreachable structural transport selection")
    labels = forest["labels"]
    transport_ms = 1000.0 * (time.perf_counter() - phase)

    phase = time.perf_counter()
    cartoon_lab = _fit_canonical_structural_model(
        labels, centers, target_lab, geometry, config)
    fit_ms = 1000.0 * (time.perf_counter() - phase)
    owner_upgrade = {
        "band_pixels": 0,
        "changed_pixels": 0,
        "forced_owner_seeds": 0,
        "unreached_pixels": 0,
        "mode": "canonical_v2_full_resolution",
    }
    return {
        "labels": labels,
        "centers": centers,
        "geometry": geometry,
        "forest": forest,
        "cartoon_lab": cartoon_lab,
        "lifted_cartoon_lab": cartoon_lab,
        "lifted_labels": labels,
        "low_labels": labels,
        "population": population,
        "owner_upgrade": owner_upgrade,
        "geometry_ms": geometry_ms,
        "transport_ms": transport_ms,
        "fit_ms": fit_ms,
        "owner_upgrade_ms": 0.0,
        "refit_ms": 0.0,
        "prepared_texture_geometry": geometry,
        "structural_population_geometry": represented_structural_geometry,
        "characteristic": {
            "trace": characteristic_trace,
            "initial_centers": initial_centers,
            "final_centers": centers.copy(),
            "requested_passes": requested_passes,
            "effective_passes": effective_passes,
            "resolved_core": resolved_core,
            "mean_pixels_per_site": mean_pixels_per_site,
        },
        "transport_model": full_transport,
        "topology": "canonical_v2",
    }


def build_segmenting_v3(
    source_rgb: np.ndarray,
    config: SegmentingV3Config = SegmentingV3Config(),
) -> dict:
    """Build the strict two-scale representation and expose every stage."""
    started = time.perf_counter()
    rgb = np.ascontiguousarray(
        np.clip(source_rgb, 0.0, 1.0), dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("version 3.0 expects one HxWx3 RGB image")
    height, width = rgb.shape[:2]
    target_lab = srgb_to_lab(rgb)
    topology = str(config.structural_topology).strip().lower()
    if topology == "canonical_v2":
        scaffold = _canonical_v2_scaffold(rgb, target_lab, config)
    elif topology == "half_cartoon":
        scaffold = _half_cartoon_scaffold(rgb, target_lab, config)
    else:
        raise ValueError(
            "structural_topology must be 'half_cartoon' or 'canonical_v2'")
    labels = scaffold["labels"]
    centers = scaffold["centers"]
    cartoon_geometry = scaffold["geometry"]
    forest = scaffold["forest"]
    cartoon_lab = scaffold["cartoon_lab"]
    lifted_cartoon_lab = scaffold["lifted_cartoon_lab"]
    lifted_labels = scaffold["lifted_labels"]
    low_labels = scaffold["low_labels"]
    population = scaffold["population"]
    owner_upgrade = scaffold["owner_upgrade"]
    cartoon_geometry_ms = scaffold["geometry_ms"]
    cartoon_transport_ms = scaffold["transport_ms"]
    cartoon_fit_ms = scaffold["fit_ms"]
    owner_upgrade_ms = scaffold["owner_upgrade_ms"]
    cartoon_refit_ms = scaffold["refit_ms"]
    characteristic = scaffold["characteristic"]
    structural_initial_labels = labels
    structural_initial_centers = centers

    texture_target = target_lab - cartoon_lab

    texture_model = str(config.texture_model).strip().lower()
    texture_geometry = None
    texture_forest = None
    texture_population = {
        "emitted_sites": 0,
        "nested_sites": len(centers),
        "missing_parent_seeds": 0,
        "surviving_parent_ids": len(centers),
        "geometry_ms": 0.0,
        "transport_ms": 0.0,
    }
    if texture_model == "nested_population":
        prepared_texture_geometry = scaffold["prepared_texture_geometry"]
        (
            texture_labels,
            texture_centers,
            texture_geometry,
            texture_forest,
            texture_population,
        ) = _nested_texture_partition(
            rgb,
            target_lab,
            labels,
            centers,
            config,
            prepared_geometry=prepared_texture_geometry,
            structural_population_geometry=(
                scaffold["structural_population_geometry"]),
        )
        frame_tensor = (
            np.asarray(texture_geometry["boundary_xx"], dtype=np.float64),
            np.asarray(texture_geometry["boundary_xy"], dtype=np.float64),
            np.asarray(texture_geometry["boundary_yy"], dtype=np.float64),
        )
        tensor_xx, tensor_xy, tensor_yy = frame_tensor
        texture_energy = tensor_xx + tensor_yy
        tensor_ms = 0.0
        coordinate_limit = max(int(config.nested_texture_ridges), 0)
    elif texture_model == "parent_ridges":
        phase = time.perf_counter()
        tensor_xx, tensor_xy, tensor_yy, texture_energy = _residual_tensor(
            texture_target,
            config.texture_tensor_sigma,
        )
        tensor_ms = 1000.0 * (time.perf_counter() - phase)
        texture_labels = labels
        texture_centers = centers
        frame_tensor = (tensor_xx, tensor_xy, tensor_yy)
        coordinate_limit = max(int(config.texture_coordinates), 0)
    else:
        raise ValueError(
            "texture_model must be 'nested_population' or 'parent_ridges'")

    phase = time.perf_counter()
    native = hard_affine_fit_native(texture_labels, texture_target)
    if native is None:
        raise RuntimeError("version 3.0 requires the native hard affine fitter")
    flat, basis, count, radius, texture_centroid, texture_fit = native
    texture_affine_ms = 1000.0 * (time.perf_counter() - phase)
    texture_initial_labels = texture_labels
    texture_initial_centers = texture_centers
    texture_cleanup = {
        "enabled": False,
        "initial_cells": int(len(texture_centers)),
        "final_cells": int(len(texture_centers)),
        "split_count": 0,
        "merge_count": 0,
        "merge_candidate_count": 0,
        "cross_parent_merge_count": 0,
    }
    texture_cleanup_ms = 0.0
    if (
        texture_model == "nested_population"
        and bool(config.texture_cleanup)
    ):
        phase = time.perf_counter()
        texture_labels, texture_centers, texture_cleanup = (
            _flat_texture_cleanup(
                texture_labels,
                texture_centers,
                texture_forest,
                texture_geometry,
                texture_target,
                texture_fit,
                labels,
                config,
            )
        )
        texture_cleanup_ms = 1000.0 * (time.perf_counter() - phase)
        if (
            texture_cleanup["split_count"]
            or texture_cleanup["merge_count"]
        ):
            phase = time.perf_counter()
            native = hard_affine_fit_native(texture_labels, texture_target)
            if native is None:
                raise RuntimeError(
                    "version 3.0 requires the native hard affine fitter")
            (
                flat,
                basis,
                count,
                radius,
                texture_centroid,
                texture_fit,
            ) = native
            texture_affine_ms += 1000.0 * (
                time.perf_counter() - phase)

    # Phase synchronization belongs to the measured texture incidence graph.
    # A later model quotient may contract compatible leaves, but quotienting
    # first changes spanning-tree paths globally and can move the phase of an
    # untouched remote cell. Preserve the pre-quotient graph as the gauge on
    # which phase is measured; its pixel columns can still be fitted by the
    # collapsed model below.
    phase_incidence_labels = texture_labels
    phase_incidence_centers = texture_centers
    phase_incidence_signal = np.ascontiguousarray(texture_target[..., 0])

    joint_leaf_collapse = {
        "enabled": False,
        "total_structural_cells": int(len(centers)),
        "structural_cells_without_pure_child": 0,
        "structural_cells_with_multiple_children": 0,
        "eligible_structural_cells": 0,
        "pure_texture_cells": 0,
        "cross_parent_texture_cells": 0,
        "total_adjacent_pairs": 0,
        "candidate_pairs": 0,
        "fit_compatible_pairs": 0,
        "ill_conditioned_pairs": 0,
        "accepted_pairs": 0,
        "initial_structural_cells": int(len(centers)),
        "final_structural_cells": int(len(centers)),
        "structural_cells_removed": 0,
        "initial_texture_cells": int(len(texture_centers)),
        "final_texture_cells": int(len(texture_centers)),
        "texture_cells_removed": 0,
        "model_refit_ms": 0.0,
    }
    joint_leaf_collapse_ms = 0.0
    if (
        scaffold["topology"] == "canonical_v2"
        and texture_model == "nested_population"
        and bool(config.joint_leaf_collapse)
    ):
        phase = time.perf_counter()
        pre_collapse_labels = labels
        pre_collapse_texture_labels = texture_labels
        pre_collapse_centers = centers
        pre_collapse_texture_centers = texture_centers
        labels, texture_labels, joint_leaf_collapse = _joint_leaf_collapse(
            labels,
            texture_labels,
            target_lab,
            cartoon_lab,
            texture_target,
            texture_fit,
            penalty=float(config.joint_leaf_penalty),
            basis_terms=(
                6
                + max(int(config.structural_ridges), 0)
                + max(int(config.nested_texture_ridges), 0)
            ),
        )
        if joint_leaf_collapse["structural_cells_removed"]:
            centers = _centers_after_collapse(
                pre_collapse_labels,
                labels,
                pre_collapse_centers,
                cartoon_geometry["measure"],
            )
            texture_centers = _centers_after_collapse(
                pre_collapse_texture_labels,
                texture_labels,
                pre_collapse_texture_centers,
                texture_geometry["measure"],
            )
            refit_started = time.perf_counter()
            cartoon_lab = _fit_canonical_structural_model(
                labels,
                centers,
                target_lab,
                cartoon_geometry,
                config,
                frame_cells=len(pre_collapse_centers),
            )
            texture_target = target_lab - cartoon_lab
            native = hard_affine_fit_native(texture_labels, texture_target)
            if native is None:
                raise RuntimeError(
                    "version 3.0 requires the native hard affine fitter")
            (
                flat,
                basis,
                count,
                radius,
                texture_centroid,
                texture_fit,
            ) = native
            refit_ms = 1000.0 * (time.perf_counter() - refit_started)
            joint_leaf_collapse["model_refit_ms"] = refit_ms
            texture_affine_ms += refit_ms
        else:
            joint_leaf_collapse["model_refit_ms"] = 0.0
        joint_leaf_collapse_ms = 1000.0 * (
            time.perf_counter() - phase)

    straight_normal, straight_tangent = _cell_frame(
        texture_labels,
        texture_centers,
        frame_tensor,
        normalization_cells=len(phase_incidence_centers),
    )
    axes_mode = str(config.coordinate_axes).strip().lower()
    if texture_model == "nested_population":
        # In synthesis the tangent is the normal of the orthogonal feature
        # family. Treat both coordinates symmetrically; "tangent" is only a
        # geometric name, not a reason to discard the second split direction.
        straight_coordinates = (straight_normal, straight_tangent)
        coordinate_names = ("normal", "second-normal")
        axes_mode = "nested_paired_normals"
    elif axes_mode == "paired":
        straight_coordinates = (straight_normal, straight_tangent)
        coordinate_names = ("normal", "tangent")
    elif axes_mode == "four_axes":
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        straight_coordinates = (
            straight_normal,
            straight_tangent,
            (straight_normal + straight_tangent) * inverse_sqrt_two,
            (straight_normal - straight_tangent) * inverse_sqrt_two,
        )
        coordinate_names = (
            "normal",
            "tangent",
            "normal+tangent",
            "normal-tangent",
        )
    else:
        raise ValueError("coordinate_axes must be 'paired' or 'four_axes'")
    phase = time.perf_counter()
    geometry_mode = str(config.coordinate_geometry).strip().lower()
    coordinate_geometry = {
        f"{name}_fallback_pixels": 0 for name in coordinate_names}
    coordinate_geometry["total_pixels"] = int(labels.size)
    if texture_model == "nested_population":
        coordinates = straight_coordinates
        geometry_mode = "straight"
    elif geometry_mode == "straight":
        coordinates = straight_coordinates
    elif geometry_mode == "owner_eikonal":
        coordinates, coordinate_geometry = _owner_eikonal_coordinates(
            labels,
            straight_coordinates,
            coordinate_names,
            (tensor_xx, tensor_xy, tensor_yy),
            sweeps=config.eikonal_sweeps,
            strength=config.eikonal_metric_strength,
        )
    else:
        raise ValueError(
            "coordinate_geometry must be 'straight' or 'owner_eikonal'")
    coordinate_geometry_ms = 1000.0 * (time.perf_counter() - phase)
    active_basis = basis
    coordinate_trace = []
    paired_ridge_columns = []
    ones = np.ones(flat.size, dtype=np.float64)
    target_flat = texture_target.reshape(-1, 3)
    structural_trace = np.zeros(labels.shape, dtype=np.float64)
    structural_trace[1:] += labels[1:] != labels[:-1]
    structural_trace[:-1] += labels[:-1] != labels[1:]
    structural_trace[:, 1:] += labels[:, 1:] != labels[:, :-1]
    structural_trace[:, :-1] += labels[:, :-1] != labels[:, 1:]
    # Exact fraction of the four cardinal incidences which leave the owner.
    structural_trace = np.minimum(0.25 * structural_trace.ravel(), 1.0)
    phase_graph = {
        "enabled": False,
        "graph_edges": 0,
        "tree_edges": 0,
        "confidence_cells": 0,
        "full_band_one_sided": False,
    }
    phase_graph_ms = 0.0
    phase_columns = ()
    if (
        texture_model == "nested_population"
        and bool(config.texture_graph_phase)
    ):
        phase = time.perf_counter()
        phase_columns, phase_graph = _graph_unrolled_texture_columns(
            phase_incidence_labels,
            phase_incidence_centers,
            # Measure the exact post-cartoon residual before the compatible
            # leaf quotient. The quotient refit may alter that residual by
            # roundoff-scale amounts; remeasuring would allow a spanning-tree
            # branch flip to warp an untouched remote cell.
            phase_incidence_signal,
        )
        phase_graph["signal"] = "pre_joint_post_cartoon_residual"
        phase_graph["incidence"] = "pre_joint_leaf_quotient"
        active_basis = np.column_stack((
            active_basis,
            *phase_columns,
        ))
        refit = hard_basis_refit_native(
            flat, active_basis, target_flat, count, radius)
        if refit is None:
            raise RuntimeError(
                "version 3.0 requires the native graph-phase refitter")
        texture_fit = refit.reshape(texture_target.shape)
        phase_graph_ms = 1000.0 * (time.perf_counter() - phase)
        coordinate_trace.append({
            "axis": "graph phase paired normals",
            "milliseconds": phase_graph_ms,
            "active_cells": int(phase_graph["confidence_cells"]),
            "score_mean": 0.0,
        })
    for index in range(coordinate_limit):
        phase = time.perf_counter()
        coordinate_index = index % len(coordinates)
        coordinate = coordinates[coordinate_index]
        residual = target_flat - texture_fit.reshape(-1, 3)
        score, offset = measure_paired_offsets(
            flat,
            ones,
            residual,
            coordinate,
            len(texture_centers),
            bins=max(int(config.offset_bins), 3),
        )
        ridge = np.clip(
            float(config.ridge_kappa) * (
                coordinate - offset[flat]),
            -1.0,
            1.0,
        )
        axis_name = coordinate_names[coordinate_index]
        appended_columns = [ridge]
        if texture_model == "nested_population":
            if (
                index >= len(coordinates)
                and len(paired_ridge_columns) >= 2
            ):
                # The paired ridges already contain the two incident
                # one-sided decisions. Their product is the missing corner
                # parity. Keep it independent of the measured third normal
                # so straight-edge and corner amplitudes do not share a
                # coefficient. Both columns enter this same refit: this adds
                # no offset measurement, cell scan, or fitting stage.
                appended_columns.append(
                    paired_ridge_columns[0] * paired_ridge_columns[1])
                # Phase amplitude may jump across an authentic edge. The
                # synchronized quadrature span and finite-band parity of the
                # two measured normal traces form the smallest gauge-complete
                # tensor product for that modulation: no phase choice or
                # new scan.
                appended_columns.extend(
                    phase * paired_ridge_columns[0] * ridge
                    for phase in phase_columns
                )
                # A long cell follows the first-order tensor frame, but an
                # authentic curved contour moves transversely as its tangent
                # coordinate changes. The mixed and tangent-quadratic
                # monomials are the second-order transverse correction:
                # they model shear and curvature without measuring another
                # offset, selecting a per-cell model, or scanning the image.
                # They enter this existing final refit beside the corner
                # parity, so no additional fitting stage is introduced.
                clipped_normal = np.clip(coordinates[0], -1.0, 1.0)
                clipped_tangent = np.clip(coordinates[1], -1.0, 1.0)
                appended_columns.extend((
                    clipped_normal * clipped_tangent,
                    clipped_tangent * clipped_tangent,
                    # Independent hard-cell fits otherwise truncate a
                    # carrier at the structural interface with no trace
                    # degree of freedom. Cardinal incidence and its normal
                    # moment supply the value/slope trace in this same fit.
                    structural_trace,
                    structural_trace * clipped_normal,
                ))
                axis_name = (
                    "normal + corner + phase envelope + quadratic frame + "
                    "structural trace")
            else:
                paired_ridge_columns.append(ridge)
        active_basis = np.column_stack((active_basis, *appended_columns))
        refit = hard_basis_refit_native(
            flat, active_basis, target_flat, count, radius)
        if refit is None:
            raise RuntimeError("version 3.0 requires the native basis refitter")
        texture_fit = refit.reshape(texture_target.shape)
        trace_item = {
            "axis": axis_name,
            "milliseconds": 1000.0 * (time.perf_counter() - phase),
            "active_cells": int(np.count_nonzero(score > 1e-12)),
            "score_mean": float(np.mean(score)),
        }
        coordinate_trace.append(trace_item)

    phase = time.perf_counter()
    if bool(config.texture_dirichlet_envelope):
        texture_fit, texture_envelope = _texture_dirichlet_envelope(
            texture_labels,
            texture_target,
            texture_fit,
        )
    else:
        texture_envelope = {
            "enabled": False,
            "contracted_cells": 0,
            "contracted_pixels": 0,
            "minimum_gain": 1.0,
            "mean_contracted_gain": 1.0,
            "target_energy": 0.0,
            "before_energy": 0.0,
            "after_energy": 0.0,
        }
    texture_envelope_ms = 1000.0 * (time.perf_counter() - phase)

    reconstruction_lab = cartoon_lab + texture_fit
    reconstruction_rgb = np.clip(
        lab_to_srgb(reconstruction_lab), 0.0, 1.0)
    rgb_error = np.mean(np.square(rgb - reconstruction_rgb), axis=2)
    rgb_mse = float(np.mean(rgb_error))
    compound_segmentation = {
        "enabled": False,
        "atom_count": int(len(texture_centers)),
        "leaf_count": int(len(texture_centers)),
        "compound_count": int(len(texture_centers)),
        "milliseconds": 0.0,
        "reconstruction_changed": False,
    }
    if bool(config.compound_segmentation):
        from experiments.compound_segment_quotient import (
            build_compound_segment_quotient,
        )

        compound_segmentation = build_compound_segment_quotient(
            texture_labels,
            target_lab,
            reconstruction_lab,
            boundary_confidence=(
                None if texture_geometry is None
                else texture_geometry["boundary_confidence"]
            ),
            target_ratio=float(config.compound_segment_ratio),
        )
    region_posterization = {
        "enabled": False,
        "region_count": int(compound_segmentation["compound_count"]),
        "milliseconds": 0.0,
    }
    if (
        (bool(config.region_posterization) or bool(config.region_family_fusion))
        and bool(compound_segmentation["enabled"])
    ):
        from experiments.region_posterization import (
            build_region_posterization,
        )

        region_posterization = build_region_posterization(
            rgb,
            compound_segmentation["labels"],
            max_depth=int(config.region_posterization_depth),
            histogram_side=int(
                config.region_posterization_histogram_side),
            labels_are_compact=True,
            source_lab=target_lab,
        )
        region_posterization["enabled"] = True
    region_family_fusion = {
        "enabled": False,
        "input_regions": int(compound_segmentation["compound_count"]),
        "family_count": int(compound_segmentation["compound_count"]),
        "milliseconds": 0.0,
    }
    if bool(config.region_family_fusion) and bool(region_posterization["enabled"]):
        from experiments.region_family_fusion import (
            build_region_family_fusion,
        )

        region_family_fusion = build_region_family_fusion(
            region_posterization)
    total_ms = 1000.0 * (time.perf_counter() - started)
    model_geometry = (
        "eikonal" if geometry_mode == "owner_eikonal" else "straight")
    result = {
        "source_rgb": rgb,
        "target_lab": target_lab,
        "cartoon_lab": cartoon_lab,
        "lifted_cartoon_lab": lifted_cartoon_lab,
        "reconstruction_rgb": reconstruction_rgb,
        "reconstruction_lab": reconstruction_lab,
        "texture_target_lab": texture_target,
        "texture_fit_lab": texture_fit,
        "texture_energy": texture_energy,
        "normal_coordinate": coordinates[0].reshape(height, width),
        "tangent_coordinate": (
            coordinates[1] if len(coordinates) > 1 else straight_tangent
        ).reshape(height, width),
        "coordinate_fields": {
            name: coordinate.reshape(height, width)
            for name, coordinate in zip(coordinate_names, coordinates)
        },
        "residual_energy": rgb_error,
        "labels": labels,
        "texture_labels": texture_labels,
        "texture_centers": texture_centers,
        "texture_initial_labels": texture_initial_labels,
        "texture_initial_centers": texture_initial_centers,
        "structural_initial_labels": structural_initial_labels,
        "structural_initial_centers": structural_initial_centers,
        "lifted_labels": lifted_labels,
        "low_labels": low_labels,
        "centers": centers,
        "population": population,
        "cartoon_geometry": cartoon_geometry,
        "structural_population_geometry": (
            scaffold["structural_population_geometry"]),
        "forest": forest,
        "texture_geometry": texture_geometry,
        "texture_forest": texture_forest,
        "texture_population": texture_population,
        "texture_cleanup": texture_cleanup,
        "joint_leaf_collapse": joint_leaf_collapse,
        "structural_topology": scaffold["topology"],
        "structural_transport_model": scaffold["transport_model"],
        "coordinate_trace": coordinate_trace,
        "coordinate_geometry": coordinate_geometry,
        "texture_phase_graph": phase_graph,
        "texture_dirichlet_envelope": texture_envelope,
        "compound_segmentation": compound_segmentation,
        "region_posterization": region_posterization,
        "region_family_fusion": region_family_fusion,
        "owner_upgrade": owner_upgrade,
        "structural_characteristic": characteristic,
        "record": {
            "rgb_mse": rgb_mse,
            "psnr": -10.0 * math.log10(max(rgb_mse, 1e-12)),
        },
        "timing": {
            "cartoon_geometry_ms": cartoon_geometry_ms,
            "cartoon_transport_ms": cartoon_transport_ms,
            "cartoon_fit_ms": cartoon_fit_ms,
            "owner_upgrade_ms": owner_upgrade_ms,
            "cartoon_refit_ms": cartoon_refit_ms,
            "texture_tensor_ms": tensor_ms,
            "texture_population_geometry_ms": float(
                texture_population["geometry_ms"]),
            "texture_population_transport_ms": float(
                texture_population["transport_ms"]),
            "texture_affine_ms": texture_affine_ms,
            "texture_cleanup_ms": texture_cleanup_ms,
            "joint_leaf_collapse_ms": joint_leaf_collapse_ms,
            "coordinate_geometry_ms": coordinate_geometry_ms,
            "texture_coordinate_ms": float(sum(
                item["milliseconds"] for item in coordinate_trace)),
            "texture_phase_graph_ms": phase_graph_ms,
            "texture_dirichlet_envelope_ms": texture_envelope_ms,
            "compound_segmentation_ms": float(
                compound_segmentation["milliseconds"]),
            "region_posterization_ms": float(
                region_posterization["milliseconds"]),
            "region_family_fusion_ms": float(
                region_family_fusion["milliseconds"]),
            "total_ms": total_ms,
        },
        "model": (
            (
                "v3_half_cartoon_full_map_owner_upgrade_"
                if scaffold["topology"] == "half_cartoon"
                else "v3_canonical_v2_structural_quotient_"
            )
            + f"{texture_model}_{model_geometry}_"
            f"{axes_mode}_local_texture"
        ),
    }
    if bool(config.diagnostic_return_basis):
        result["texture_active_basis"] = active_basis.reshape(
            height, width, active_basis.shape[1])
        result["texture_basis_count"] = count
        result["texture_basis_radius"] = radius
        result["texture_basis_centroid"] = texture_centroid
    return result


def _main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="camera")
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--texture-coordinates", type=int, default=8)
    parser.add_argument(
        "--texture-model",
        choices=("nested_population", "parent_ridges"),
        default="nested_population",
    )
    parser.add_argument("--nested-texture-ridges", type=int, default=2)
    parser.add_argument("--texture-support-weight", type=float, default=0.65)
    parser.add_argument("--no-texture-cleanup", action="store_true")
    parser.add_argument("--texture-split-error-ratio", type=float, default=2.5)
    parser.add_argument("--texture-split-return-extent", type=float, default=2.0)
    parser.add_argument("--texture-split-minimum-pixels", type=int, default=12)
    parser.add_argument("--texture-split-metric-strength", type=float, default=0.25)
    parser.add_argument("--texture-merge-penalty", type=float, default=4.0)
    parser.add_argument(
        "--no-texture-curvature",
        action="store_true",
    )
    parser.add_argument("--texture-safety-cells", type=int, default=65536)
    parser.add_argument(
        "--no-owner-upgrade",
        action="store_true",
        help="retain the literal nearest-neighbour lift as a control",
    )
    parser.add_argument(
        "--owner-upgrade-mode",
        choices=("full_map", "boundary_band"),
        default="boundary_band",
    )
    parser.add_argument("--owner-upgrade-radius", type=int, default=8)
    parser.add_argument("--owner-upgrade-sweeps", type=int, default=2)
    parser.add_argument("--owner-upgrade-strength", type=float, default=64.0)
    parser.add_argument(
        "--owner-upgrade-cartoon-strength",
        type=float,
        default=0.0,
    )
    parser.add_argument("--no-cartoon-refit", action="store_true")
    parser.add_argument("--cartoon-refit-strength", type=float, default=0.5)
    parser.add_argument(
        "--coordinate-axes",
        choices=("paired", "four_axes"),
        default="paired",
    )
    parser.add_argument(
        "--coordinate-geometry",
        choices=("straight", "owner_eikonal"),
        default="straight",
    )
    parser.add_argument("--eikonal-sweeps", type=int, default=2)
    parser.add_argument("--eikonal-strength", type=float, default=2.0)
    parser.add_argument("--output", default="segmenting_v3_demo.png")
    args = parser.parse_args()

    if args.image:
        from PIL import Image
        image = np.asarray(Image.open(
            Path(args.image).expanduser()).convert("RGB"))
    else:
        import gallery
        image = gallery.load(args.gallery)
    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float64) / np.iinfo(value.dtype).max
    else:
        value = value.astype(np.float64)
        peak = float(np.max(value, initial=0.0))
        if peak > 1.0:
            if peak <= 255.0:
                value /= 255.0
            else:
                raise ValueError(
                    "floating-point image must use either [0, 1] or [0, 255]")
    if args.max_side > 0 and max(value.shape[:2]) > args.max_side:
        scale = args.max_side / max(value.shape[:2])
        value = _resize(
            value,
            (
                max(int(round(value.shape[0] * scale)), 2),
                max(int(round(value.shape[1] * scale)), 2),
            ),
            order=1,
        )
    result = build_segmenting_v3(
        value,
        SegmentingV3Config(
            texture_coordinates=args.texture_coordinates,
            texture_model=args.texture_model,
            nested_texture_ridges=args.nested_texture_ridges,
            texture_support_weight=args.texture_support_weight,
            texture_curvature_population=not args.no_texture_curvature,
            texture_safety_cells=args.texture_safety_cells,
            texture_cleanup=not args.no_texture_cleanup,
            texture_split_error_ratio=args.texture_split_error_ratio,
            texture_split_return_extent=args.texture_split_return_extent,
            texture_split_minimum_pixels=(
                args.texture_split_minimum_pixels),
            texture_split_metric_strength=(
                args.texture_split_metric_strength),
            texture_merge_penalty=args.texture_merge_penalty,
            owner_upgrade=not args.no_owner_upgrade,
            owner_upgrade_mode=args.owner_upgrade_mode,
            owner_upgrade_radius=args.owner_upgrade_radius,
            owner_upgrade_sweeps=args.owner_upgrade_sweeps,
            owner_upgrade_strength=args.owner_upgrade_strength,
            owner_upgrade_cartoon_strength=(
                args.owner_upgrade_cartoon_strength),
            cartoon_full_refit=not args.no_cartoon_refit,
            cartoon_refit_strength=args.cartoon_refit_strength,
            coordinate_axes=args.coordinate_axes,
            coordinate_geometry=args.coordinate_geometry,
            eikonal_sweeps=args.eikonal_sweeps,
            eikonal_metric_strength=args.eikonal_strength,
        ))

    import matplotlib.pyplot as plt
    boundary = np.zeros(result["labels"].shape, dtype=bool)
    boundary[:, 1:] |= (
        result["labels"][:, 1:] != result["labels"][:, :-1])
    boundary[1:] |= (
        result["labels"][1:] != result["labels"][:-1])
    overlay = result["reconstruction_rgb"].copy()
    overlay[boundary] = (1.0, 0.04, 0.01)
    texture = np.sqrt(np.maximum(result["texture_energy"], 0.0))
    residual = np.log1p(500.0 * result["residual_energy"])
    panels = (
        (result["source_rgb"], "source"),
        (
            np.clip(lab_to_srgb(result["cartoon_lab"]), 0.0, 1.0),
            "full-resolution refined cartoon",
        ),
        (texture, "full residual texture"),
        (overlay, "full-map upgraded cartoon owners"),
        (result["reconstruction_rgb"], "local texture reconstruction"),
        (residual, "final residual energy"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=120)
    for axis, (panel, title) in zip(axes.ravel(), panels):
        axis.imshow(panel, cmap="turbo" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    timing = result["timing"]
    figure.suptitle(
        f"Version 3.0 | {len(result['centers']):,} cartoon cells | "
        f"{result['record']['psnr']:.2f} dB | "
        f"cartoon {timing['cartoon_geometry_ms']:.0f} + "
        f"transport {timing['cartoon_transport_ms']:.0f} ms | "
        f"texture mechanics "
        f"{timing['texture_tensor_ms'] + timing['coordinate_geometry_ms'] + timing['texture_affine_ms'] + timing['texture_coordinate_ms']:.0f} ms"
    )
    figure.tight_layout()
    output = Path(args.output).expanduser().resolve()
    figure.savefig(output, bbox_inches="tight")
    print(output)
    print(result["record"])
    print(result["timing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
