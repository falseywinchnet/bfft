"""PORT 08: reverse residual transport and simultaneous local refill.

The current hard partition supplies an exact predecessor forest.  Unexplained
single-stage energy flows from every pixel back toward its owning site:

    F(p) = e(p) + sum_{child q of p} F(q)

and the return work is

    W_i = sum_{p in cell i} F(p) [d(p) - d(parent(p))].

``W_i / E_i`` is the average geodesic distance through which the unaccounted
energy in cell ``i`` must return.  It is large only when meaningful residual
energy lives away from the current site.  Every cell crosses the same robust
local threshold independently; no cells are sorted or ranked.

For a deserving cell, residual covariance supplies the split direction.  A
fixed local histogram divides a mixture of original support mass and residual
mass, keeping the coarse domain covered while placing the new support toward
the missing detail.  All accepted splits are emitted simultaneously.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _reverse_tree_flux(parent, labels, distance, residual, cells):
    pixels = len(parent)
    children = np.zeros(pixels, dtype=np.int32)
    for pixel in range(pixels):
        ancestor = parent[pixel]
        if ancestor >= 0 and labels[ancestor] == labels[pixel]:
            children[ancestor] += 1

    queue = np.empty(pixels, dtype=np.int32)
    tail = 0
    for pixel in range(pixels):
        if children[pixel] == 0:
            queue[tail] = pixel
            tail += 1

    flux = residual.copy()
    work = np.zeros(cells)
    root_mass = np.zeros(cells)
    head = 0
    while head < tail:
        pixel = queue[head]
        head += 1
        label = labels[pixel]
        ancestor = parent[pixel]
        if ancestor >= 0 and labels[ancestor] == label:
            step = max(distance[pixel] - distance[ancestor], 0.0)
            work[label] += flux[pixel] * step
            flux[ancestor] += flux[pixel]
            children[ancestor] -= 1
            if children[ancestor] == 0:
                queue[tail] = ancestor
                tail += 1
        else:
            root_mass[label] += flux[pixel]
    return flux, work, root_mass, head


@_compile
def _residual_moments(labels, residual, width, cells):
    mass = np.zeros(cells)
    sx = np.zeros(cells)
    sy = np.zeros(cells)
    pixels = len(labels)
    for pixel in range(pixels):
        label = labels[pixel]
        value = residual[pixel]
        x = pixel % width + 0.5
        y = pixel // width + 0.5
        mass[label] += value
        sx[label] += value * x
        sy[label] += value * y
    cx = sx / np.maximum(mass, 1e-30)
    cy = sy / np.maximum(mass, 1e-30)
    cxx = np.zeros(cells)
    cxy = np.zeros(cells)
    cyy = np.zeros(cells)
    for pixel in range(pixels):
        label = labels[pixel]
        value = residual[pixel]
        dx = pixel % width + 0.5 - cx[label]
        dy = pixel // width + 0.5 - cy[label]
        cxx[label] += value * dx * dx
        cxy[label] += value * dx * dy
        cyy[label] += value * dy * dy
    safe = np.maximum(mass, 1e-30)
    return mass, cx, cy, cxx / safe, cxy / safe, cyy / safe


def _principal_directions(cxx, cxy, cyy):
    cxx = np.asarray(cxx, dtype=np.float64)
    cxy = np.asarray(cxy, dtype=np.float64)
    cyy = np.asarray(cyy, dtype=np.float64)
    trace = cxx + cyy
    doubled_x = cxx - cyy
    doubled_y = 2.0 * cxy
    disc = np.hypot(doubled_x, doubled_y)
    major = np.maximum(0.5 * (trace + disc), 0.0)
    minor = np.maximum(0.5 * (trace - disc), 0.0)

    # (doubled_x, doubled_y) / disc = (cos(2 theta), sin(2 theta)).
    # Recover its covering-map half angle algebraically.  This is the same
    # projective coordinate used by the BFFT slope/half-angle phase path, but
    # the tensor already supplies doubled phase, so no atan2 or lookup is
    # required here.
    u = np.divide(
        doubled_x, disc, out=np.ones_like(disc), where=disc > 1e-30)
    v = np.divide(
        doubled_y, disc, out=np.zeros_like(disc), where=disc > 1e-30)
    direction_x = np.sqrt(np.maximum(0.5 * (1.0 + u), 0.0))
    direction_y = np.copysign(
        np.sqrt(np.maximum(0.5 * (1.0 - u), 0.0)),
        np.where(np.abs(v) > 1e-30, v, 1.0),
    )
    direction = np.column_stack((direction_x, direction_y))
    return major, minor, direction


@_compile
def _fixed_histogram_refill(
    labels,
    support,
    residual,
    residual_mass,
    support_mass,
    center_x,
    center_y,
    direction,
    split,
    width,
    bins,
    detail_gain,
):
    cells = len(split)
    pixels = len(labels)
    low = np.full(cells, np.inf)
    high = np.full(cells, -np.inf)
    for pixel in range(pixels):
        cell = labels[pixel]
        if not split[cell]:
            continue
        x = pixel % width + 0.5
        y = pixel // width + 0.5
        projection = (
            (x - center_x[cell]) * direction[cell, 0]
            + (y - center_y[cell]) * direction[cell, 1])
        low[cell] = min(low[cell], projection)
        high[cell] = max(high[cell], projection)

    histogram = np.zeros((cells, bins))
    total = np.zeros(cells)
    for pixel in range(pixels):
        cell = labels[pixel]
        if not split[cell]:
            continue
        detail = (
            detail_gain * support_mass[cell] * residual[pixel]
            / max(residual_mass[cell], 1e-30))
        value = support[pixel] + detail
        x = pixel % width + 0.5
        y = pixel // width + 0.5
        projection = (
            (x - center_x[cell]) * direction[cell, 0]
            + (y - center_y[cell]) * direction[cell, 1])
        span = max(high[cell] - low[cell], 1e-30)
        slot = int((projection - low[cell]) / span * bins)
        slot = min(max(slot, 0), bins - 1)
        histogram[cell, slot] += value
        total[cell] += value

    boundary = np.zeros(cells)
    for cell in range(cells):
        if not split[cell]:
            continue
        target = 0.5 * total[cell]
        cumulative = 0.0
        slot = bins - 1
        for candidate in range(bins):
            cumulative += histogram[cell, candidate]
            if cumulative >= target:
                slot = candidate
                break
        boundary[cell] = (
            low[cell] + (slot + 0.5) / bins * (high[cell] - low[cell]))

    minus_mass = np.zeros(cells)
    plus_mass = np.zeros(cells)
    minus_x = np.zeros(cells)
    minus_y = np.zeros(cells)
    plus_x = np.zeros(cells)
    plus_y = np.zeros(cells)
    for pixel in range(pixels):
        cell = labels[pixel]
        if not split[cell]:
            continue
        detail = (
            detail_gain * support_mass[cell] * residual[pixel]
            / max(residual_mass[cell], 1e-30))
        value = support[pixel] + detail
        x = pixel % width + 0.5
        y = pixel // width + 0.5
        projection = (
            (x - center_x[cell]) * direction[cell, 0]
            + (y - center_y[cell]) * direction[cell, 1])
        if projection > boundary[cell]:
            plus_mass[cell] += value
            plus_x[cell] += value * x
            plus_y[cell] += value * y
        else:
            minus_mass[cell] += value
            minus_x[cell] += value * x
            minus_y[cell] += value * y

    minus = np.zeros((cells, 2))
    plus = np.zeros((cells, 2))
    valid = np.zeros(cells, dtype=np.bool_)
    for cell in range(cells):
        if (not split[cell] or minus_mass[cell] <= 1e-20
                or plus_mass[cell] <= 1e-20):
            continue
        minus[cell, 0] = minus_x[cell] / minus_mass[cell]
        minus[cell, 1] = minus_y[cell] / minus_mass[cell]
        plus[cell, 0] = plus_x[cell] / plus_mass[cell]
        plus[cell, 1] = plus_y[cell] / plus_mass[cell]
        valid[cell] = True
    return minus, plus, valid


def reverse_residual_refill(
    labels_2d: np.ndarray,
    centers: np.ndarray,
    forest: dict,
    residual_energy: np.ndarray,
    support_measure: np.ndarray,
    *,
    error_ratio_threshold: float = 1.5,
    return_distance_threshold: float = 2.0,
    minimum_region_pixels: int = 12,
    detail_gain: float = 1.0,
    bins: int = 64,
    safety_cells: int = 32768,
) -> tuple[np.ndarray, dict]:
    labels = np.asarray(labels_2d, dtype=np.int32).ravel()
    parent = np.asarray(forest["parent"], dtype=np.int32).ravel()
    distance = np.asarray(forest["distance"], dtype=np.float64).ravel()
    residual = np.maximum(
        np.asarray(residual_energy, dtype=np.float64).ravel(), 0.0)
    support = np.maximum(
        np.asarray(support_measure, dtype=np.float64).ravel(), 0.0)
    height, width = labels_2d.shape
    cells = len(centers)
    flux, work, root_mass, processed = _reverse_tree_flux(
        parent, labels, distance, residual, cells)
    if processed != labels.size:
        raise RuntimeError("transport predecessor forest is not acyclic")

    count = np.bincount(labels, minlength=cells)
    mass, cx, cy, cxx, cxy, cyy = _residual_moments(
        labels, residual, width, cells)
    major, minor, direction = _principal_directions(cxx, cxy, cyy)
    mean_error = mass / np.maximum(count, 1)
    eligible = count >= 2 * max(int(minimum_region_pixels), 1)
    baseline_values = mean_error[eligible & np.isfinite(mean_error)]
    baseline = (
        float(np.median(baseline_values))
        if baseline_values.size else float(np.median(mean_error)))
    error_ratio = mean_error / max(baseline, 1e-30)
    return_distance = work / np.maximum(mass, 1e-30)
    equivalent_radius = np.sqrt(np.maximum(count, 1) / math.pi)
    return_extent = return_distance / np.maximum(equivalent_radius, 1e-12)
    split = (
        eligible
        & (error_ratio >= float(error_ratio_threshold))
        & (return_extent >= float(return_distance_threshold))
        & (mass > 1e-20)
    )
    if cells + int(np.count_nonzero(split)) > int(safety_cells):
        raise RuntimeError(
            "residual refinement exceeded its safety ceiling; "
            "raise the local thresholds rather than ranking cells")

    support_mass = np.bincount(
        labels, weights=support, minlength=cells)
    center_x = np.asarray(centers)[:, 0] * width
    center_y = np.asarray(centers)[:, 1] * height
    minus, plus, valid = _fixed_histogram_refill(
        labels,
        support,
        residual,
        mass,
        support_mass,
        center_x,
        center_y,
        direction,
        split,
        width,
        max(int(bins), 8),
        max(float(detail_gain), 0.0),
    )
    separation = np.hypot(
        plus[:, 0] - minus[:, 0], plus[:, 1] - minus[:, 1])
    valid &= separation >= 0.75
    split_ids = np.flatnonzero(valid)
    next_centers = np.asarray(centers, dtype=np.float64).copy()
    next_centers[split_ids, 0] = minus[split_ids, 0] / width
    next_centers[split_ids, 1] = minus[split_ids, 1] / height
    additions = np.column_stack((
        plus[split_ids, 0] / width,
        plus[split_ids, 1] / height,
    ))
    next_centers = np.vstack((next_centers, additions))
    parent_of_centers = np.concatenate((
        np.arange(cells, dtype=np.int32),
        split_ids.astype(np.int32),
    ))
    return next_centers, {
        "split_ids": split_ids,
        "split_count": int(len(split_ids)),
        "parent_of_centers": parent_of_centers,
        "baseline_mean_error": baseline,
        "mean_error": mean_error,
        "error_ratio": error_ratio,
        "return_distance": return_distance,
        "return_extent": return_extent,
        "return_work": work,
        "root_mass": root_mass,
        "flux": flux.reshape(height, width),
        "error_ratio_map": error_ratio[labels].reshape(height, width),
        "return_extent_map": return_extent[labels].reshape(height, width),
        "split_map": split[labels].reshape(height, width),
        "residual_major_variance": major,
        "residual_minor_variance": minor,
        "residual_direction": direction,
    }
