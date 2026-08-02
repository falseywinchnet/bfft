"""Deterministic multiscale posterization over immutable segment IDs.

The palette hierarchy is a measurement for a later object-family quotient. It
never changes the reconstruction or the input segmentation.  Palette choices
are region-balanced, while palette colors remain literal pixel-weighted means.
Each segment is represented by its mixture over nested palette families; this
is the analytic average of a dithered realization without generating dots.
"""

from __future__ import annotations

import math
import time

import numpy as np

from bfft.effects import lab_to_srgb, srgb_to_lab

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _compile(function):
    return function if njit is None else njit(cache=True)(function)


@_compile
def _fused_region_histogram(lab, labels, region_population, side):
    """Bin Lab values and accumulate palette statistics in one raster pass."""
    height, width = labels.shape
    dense_count = side * side * side
    dense_bin = np.empty((height, width), dtype=np.int32)
    pixel_count = np.zeros(dense_count, dtype=np.float64)
    pixel_sum = np.zeros((dense_count, 3), dtype=np.float64)
    selection_weight = np.zeros(dense_count, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            first = int(lab[y, x, 0] * side)
            second = int((lab[y, x, 1] + 0.25) * (2.0 * side))
            third = int((lab[y, x, 2] + 0.25) * (2.0 * side))
            first = min(max(first, 0), side - 1)
            second = min(max(second, 0), side - 1)
            third = min(max(third, 0), side - 1)
            index = (first * side + second) * side + third
            dense_bin[y, x] = index
            pixel_count[index] += 1.0
            pixel_sum[index, 0] += lab[y, x, 0]
            pixel_sum[index, 1] += lab[y, x, 1]
            pixel_sum[index, 2] += lab[y, x, 2]
            region = labels[y, x]
            selection_weight[index] += 1.0 / math.sqrt(max(
                region_population[region], 1.0))
    return dense_bin, pixel_count, pixel_sum, selection_weight


def _rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = value[..., :3]
    if float(np.max(value, initial=0.0)) > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _compact_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    value = np.asarray(labels, dtype=np.int64)
    unique, inverse = np.unique(value, return_inverse=True)
    return inverse.reshape(value.shape).astype(np.int32), len(unique)


def _optimal_principal_cut(
    points: np.ndarray,
    selection_weight: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the minimum-SSE cut along a weighted principal direction."""
    if len(indices) < 2:
        return None
    sample = points[indices]
    weight = selection_weight[indices]
    total_weight = float(np.sum(weight))
    if total_weight <= 0.0:
        return None
    mean = np.sum(sample * weight[:, None], axis=0) / total_weight
    centered = sample - mean
    covariance = (centered * weight[:, None]).T @ centered / total_weight
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    axis = eigenvector[:, int(np.argmax(eigenvalue))]
    projection = centered @ axis
    order = np.argsort(projection, kind="stable")
    ordered = sample[order]
    ordered_weight = weight[order]
    cumulative_weight = np.cumsum(ordered_weight)
    cumulative_sum = np.cumsum(
        ordered * ordered_weight[:, None], axis=0)
    cumulative_square = np.cumsum(
        np.sum(ordered * ordered, axis=1) * ordered_weight)
    whole_sum = cumulative_sum[-1]
    whole_square = cumulative_square[-1]
    left_weight = cumulative_weight[:-1]
    right_weight = total_weight - left_weight
    valid = (left_weight > 0.0) & (right_weight > 0.0)
    if not np.any(valid):
        return None
    left_sum = cumulative_sum[:-1]
    right_sum = whole_sum - left_sum
    left_error = cumulative_square[:-1] - (
        np.sum(left_sum * left_sum, axis=1) / left_weight)
    right_error = (whole_square - cumulative_square[:-1]) - (
        np.sum(right_sum * right_sum, axis=1) / right_weight)
    error = np.where(valid, left_error + right_error, np.inf)
    cut = int(np.argmin(error)) + 1
    first = indices[order[:cut]]
    second = indices[order[cut:]]
    if len(first) == 0 or len(second) == 0:
        return None
    return first, second


def _palette_hierarchy(
    points: np.ndarray,
    selection_weight: np.ndarray,
    pixel_count: np.ndarray,
    pixel_sum: np.ndarray,
    max_depth: int,
) -> list[dict]:
    nodes = [np.arange(len(points), dtype=np.int32)]
    levels = []
    for depth in range(max(int(max_depth), 0) + 1):
        family = np.empty(len(points), dtype=np.int32)
        palette = np.empty((len(nodes), 3), dtype=np.float64)
        population = np.empty(len(nodes), dtype=np.float64)
        for identifier, indices in enumerate(nodes):
            family[indices] = identifier
            population[identifier] = np.sum(pixel_count[indices])
            palette[identifier] = (
                np.sum(pixel_sum[indices], axis=0)
                / max(population[identifier], 1.0)
            )
        levels.append({
            "depth": depth,
            "family_count": len(nodes),
            "bin_family": family,
            "palette_lab": palette,
            "palette_population": population,
        })
        next_nodes = []
        split = False
        for indices in nodes:
            children = _optimal_principal_cut(
                points, selection_weight, indices)
            if children is None:
                next_nodes.append(indices)
            else:
                next_nodes.extend(children)
                split = True
        nodes = next_nodes
        if not split:
            break
    return levels


def build_region_posterization(
    source_rgb: np.ndarray,
    region_labels: np.ndarray,
    *,
    max_depth: int = 6,
    histogram_side: int = 32,
    labels_are_compact: bool = False,
    source_lab: np.ndarray | None = None,
) -> dict:
    """Build a nested color-family stack and segment mixture signatures."""
    started = time.perf_counter()
    lab = (
        np.ascontiguousarray(source_lab, dtype=np.float64)
        if source_lab is not None
        else np.ascontiguousarray(srgb_to_lab(_rgb(source_rgb)), dtype=np.float64)
    )
    if bool(labels_are_compact):
        labels = np.ascontiguousarray(region_labels, dtype=np.int32)
        region_count = int(np.max(labels, initial=-1)) + 1
    else:
        labels, region_count = _compact_labels(region_labels)
    if labels.shape != lab.shape[:2]:
        raise ValueError("posterization labels must match the source image")
    flat_labels = labels.ravel()
    region_population = np.bincount(
        flat_labels, minlength=region_count).astype(np.float64)

    side = max(int(histogram_side), 4)
    (
        dense_bin,
        pixel_count,
        dense_pixel_sum,
        dense_selection_weight,
    ) = _fused_region_histogram(lab, labels, region_population, side)
    dense_bin = dense_bin.ravel()
    dense_count = side ** 3
    present = pixel_count > 0.0
    dense_to_sparse = np.full(dense_count, -1, dtype=np.int32)
    dense_to_sparse[present] = np.arange(np.count_nonzero(present))
    sparse_bin = dense_to_sparse[dense_bin]
    sparse_count = pixel_count[present]
    pixel_sum = dense_pixel_sum[present]
    points = pixel_sum / sparse_count[:, None]

    # Each region contributes sqrt(area) rather than area. This retains real
    # support for large regions without allowing a sky or page background to
    # monopolize all palette decisions.
    selection_weight = dense_selection_weight[present]
    levels = _palette_hierarchy(
        points,
        selection_weight,
        sparse_count,
        pixel_sum,
        max_depth,
    )

    finest = levels[-1]
    leaf_count = int(finest["family_count"])
    pixel_leaf_family = finest["bin_family"][sparse_bin].astype(
        np.uint8 if leaf_count <= 256 else np.uint16)
    joint = flat_labels.astype(np.int64) * leaf_count + pixel_leaf_family
    leaf_mixture = np.bincount(
        joint,
        minlength=region_count * leaf_count,
    ).reshape(region_count, leaf_count).astype(np.float32)
    leaf_mixture /= np.maximum(region_population[:, None], 1.0)
    for level in levels:
        families = int(level["family_count"])
        leaf_to_family = np.full(leaf_count, -1, dtype=np.int32)
        for sparse in range(len(points)):
            leaf = int(finest["bin_family"][sparse])
            family = int(level["bin_family"][sparse])
            if leaf_to_family[leaf] < 0:
                leaf_to_family[leaf] = family
            elif leaf_to_family[leaf] != family:
                raise RuntimeError("posterization hierarchy is not nested")
        mixture = np.zeros((region_count, families), dtype=np.float32)
        for leaf, family in enumerate(leaf_to_family):
            mixture[:, family] += leaf_mixture[:, leaf]
        mixed_lab = mixture @ level["palette_lab"]
        level.update({
            "leaf_to_family": leaf_to_family.astype(
                np.uint8 if families <= 256 else np.uint16),
            "region_mixture": mixture,
            "region_dominant_family": np.argmax(mixture, axis=1).astype(
                np.int32),
            "region_average_lab": mixed_lab.astype(np.float32),
        })

    return {
        "labels": labels,
        "region_count": region_count,
        "region_population": region_population,
        "histogram_side": side,
        "occupied_bins": len(points),
        "pixel_leaf_family": pixel_leaf_family.reshape(labels.shape),
        "levels": levels,
        "milliseconds": 1000.0 * (time.perf_counter() - started),
    }


def render_posterization_level(
    posterization: dict,
    depth: int,
    *,
    hard: bool = False,
) -> np.ndarray:
    """Render one palette level without retaining full RGB stacks."""
    levels = posterization["levels"]
    level = levels[int(np.clip(depth, 0, len(levels) - 1))]
    if hard:
        pixel_family = level["leaf_to_family"][
            posterization["pixel_leaf_family"]]
        lab = level["palette_lab"][pixel_family]
    else:
        lab = level["region_average_lab"][posterization["labels"]]
    return np.clip(lab_to_srgb(lab), 0.0, 1.0)


def multiscale_region_affinity(
    posterization: dict,
    region_pairs: np.ndarray,
    *,
    minimum_depth: int = 2,
) -> np.ndarray:
    """Histogram-intersection affinity accumulated over palette scale."""
    pairs = np.asarray(region_pairs, dtype=np.int32)
    affinity = np.zeros(len(pairs), dtype=np.float64)
    weight = 0.0
    # A genuinely two-color image terminates the hierarchy at depth one.
    # Use its deepest non-root evidence rather than returning the accidental
    # all-zero affinity that a fixed depth-two floor would produce.
    available_depth = int(posterization["levels"][-1]["depth"])
    first_depth = min(int(minimum_depth), available_depth)
    for level in posterization["levels"]:
        if int(level["depth"]) < first_depth:
            continue
        mixture = level["region_mixture"]
        # Fine levels carry more discriminative evidence but cannot erase a
        # coarse family relationship which survives posterization.
        level_weight = 1.0 + 0.25 * (
            int(level["depth"]) - first_depth)
        affinity += level_weight * np.sum(np.minimum(
            mixture[pairs[:, 0]], mixture[pairs[:, 1]]), axis=1)
        weight += level_weight
    return affinity / max(weight, 1.0)


def region_adjacency(labels: np.ndarray) -> np.ndarray:
    """Return unique undirected four-neighbour region pairs."""
    value, count = _compact_labels(labels)
    horizontal = value[:, 1:] != value[:, :-1]
    vertical = value[1:] != value[:-1]
    first = np.concatenate((
        value[:, :-1][horizontal],
        value[:-1][vertical],
    )).astype(np.int64)
    second = np.concatenate((
        value[:, 1:][horizontal],
        value[1:][vertical],
    )).astype(np.int64)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    key = np.unique(low * count + high)
    return np.stack((key // count, key % count), axis=1).astype(np.int32)
