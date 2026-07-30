"""PORT 03: exact two-label geodesic transport and final hard refresh.

This is the dominant native port.  It is a monotone shortest-path propagation,
not a candidate search.  The exact local refresh experiment has the same
owner/runner/distance contract and can later replace a global refresh when a
source is inserted or moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bfft.vision import bucket_first_label_native

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "viewer", ROOT / "experiments", ROOT / "experiments" / "sigma_opt"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

def walk_two_labels(
    centers: np.ndarray,
    costs: np.ndarray,
    *,
    queue: str = "bucket",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = costs.shape[1:]
    centers = np.asarray(centers, dtype=np.float64)
    seed_x = np.clip(
        np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
        0,
        width - 1,
    )
    seed_y = np.clip(
        np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
        0,
        height - 1,
    )
    reach = np.zeros(len(centers), dtype=np.float64)
    if queue == "bucket":
        from experiments.sigma_opt.opt_dijkstra_bucket import (
            _dijkstra_bucket,
            queue_geometry,
        )

        seed_pixel = seed_y.astype(np.int64) * width + seed_x.astype(np.int64)
        scale = np.ones(height * width, dtype=np.float64)
        delta, span, shift = queue_geometry(costs, scale, reach)
        result = _dijkstra_bucket(
            seed_pixel, reach, costs, scale, height, width,
            delta, span, shift,
        )
        return result[0], result[1], result[2], result[3]
    if queue != "heap":
        raise ValueError(f"unknown exact queue {queue!r}")
    from transport_voronoi import _dijkstra_two_best_packed

    return _dijkstra_two_best_packed(
        seed_x, seed_y, reach, costs, height, width)


def hard_partition(
    centers: np.ndarray,
    costs: np.ndarray,
    *,
    queue: str = "bucket",
) -> np.ndarray:
    owner, _, _, _ = walk_two_labels(centers, costs, queue=queue)
    return owner.reshape(costs.shape[1:])


def hard_partition_with_forest(
    centers: np.ndarray,
    costs: np.ndarray,
    reach: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Exact bucket refresh plus the achieving first-label predecessor tree."""
    height, width = costs.shape[1:]
    centers = np.asarray(centers, dtype=np.float64)
    seed_x = np.clip(
        np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
        0,
        width - 1,
    )
    seed_y = np.clip(
        np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
        0,
        height - 1,
    )
    seed_pixel = seed_y.astype(np.int64) * width + seed_x.astype(np.int64)
    reach = (
        np.zeros(len(centers), dtype=np.float64)
        if reach is None
        else np.ascontiguousarray(reach, dtype=np.float64)
    )
    if reach.shape != (len(centers),):
        raise ValueError("reach must have one scalar per center")
    scale = np.ones(height * width, dtype=np.float64)
    from experiments.sigma_opt.opt_dijkstra_bucket import (
        _dijkstra_bucket,
        queue_geometry,
    )

    delta, span, shift = queue_geometry(costs, scale, reach)
    result = _dijkstra_bucket(
        seed_pixel, reach, costs, scale, height, width,
        delta, span, shift)
    return {
        "labels": result[0].reshape(height, width),
        "runner": result[1].reshape(height, width),
        "distance": result[2].reshape(height, width),
        "second_distance": result[3].reshape(height, width),
        "parent": result[4].reshape(height, width),
    }


def hard_first_partition_with_forest(
    centers: np.ndarray,
    costs: np.ndarray,
    reach: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Exact first-owner partition without unused runner-up propagation."""
    height, width = costs.shape[1:]
    centers = np.asarray(centers, dtype=np.float64)
    seed_x = np.clip(
        np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
        0,
        width - 1,
    )
    seed_y = np.clip(
        np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
        0,
        height - 1,
    )
    seed_pixel = seed_y.astype(np.int64) * width + seed_x.astype(np.int64)
    reach = (
        np.zeros(len(centers), dtype=np.float64)
        if reach is None
        else np.ascontiguousarray(reach, dtype=np.float64)
    )
    if reach.shape != (len(centers),):
        raise ValueError("reach must have one scalar per center")
    scale = np.ones(height * width, dtype=np.float64)
    from experiments.sigma_opt.opt_dijkstra_bucket import (
        _dijkstra_first_bucket,
        queue_geometry,
    )

    delta, span, shift = queue_geometry(costs, scale, reach)
    native = bucket_first_label_native(
        seed_pixel, reach, costs, delta, span, shift)
    if native is None:
        native = _dijkstra_first_bucket(
            seed_pixel,
            reach,
            costs,
            scale,
            height,
            width,
            delta,
            span,
            shift,
        )
    owner, distance, parent, used = native
    return {
        "labels": owner.reshape(height, width),
        "distance": distance.reshape(height, width),
        "parent": parent.reshape(height, width),
        "queue_pushes": int(used),
    }


def restrict_costs_to_partition(
    costs: np.ndarray,
    parent_labels: np.ndarray,
) -> np.ndarray:
    """Set every edge crossing an established parent domain to infinity."""
    labels = np.asarray(parent_labels, dtype=np.int32)
    height, width = labels.shape
    restricted = np.ascontiguousarray(costs, dtype=np.float32).copy()
    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for direction, (dy, dx) in enumerate(directions):
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        yd = slice(max(0, dy), min(height, height + dy))
        xd = slice(max(0, dx), min(width, width + dx))
        crossing = labels[ys, xs] != labels[yd, xd]
        plane = restricted[direction, ys, xs]
        plane[crossing] = np.inf
    return restricted


def snap_centers_to_parent(
    centers: np.ndarray,
    parent_of_centers: np.ndarray,
    parent_labels: np.ndarray,
) -> np.ndarray:
    """Snap each barycenter to its closest pixel in its established domain.

    A parent has at most two sites after one simultaneous refill, so this is a
    linear image pass in the eventual native form, not a site search.
    """
    labels = np.asarray(parent_labels, dtype=np.int32)
    height, width = labels.shape
    centers = np.asarray(centers, dtype=np.float64)
    parent_of_centers = np.asarray(parent_of_centers, dtype=np.int32)
    best_distance = np.full(len(centers), np.inf, dtype=np.float64)
    best_pixel = np.full(len(centers), -1, dtype=np.int64)
    children = np.full(int(np.max(labels)) + 1, -1, dtype=np.int32)
    for site in range(len(parent_of_centers)):
        parent = int(parent_of_centers[site])
        if site != parent:
            children[parent] = site
    flat = labels.ravel()
    pixel = np.arange(flat.size, dtype=np.int64)
    x = pixel % width + 0.5
    y = pixel // width + 0.5

    def snap_sites(site_at_pixel, valid):
        site = site_at_pixel[valid]
        selected_pixel = pixel[valid]
        dx = x[valid] - centers[site, 0] * width
        dy = y[valid] - centers[site, 1] * height
        distance = dx * dx + dy * dy
        np.minimum.at(best_distance, site, distance)
        winner = distance <= best_distance[site] + 1e-15
        best_pixel[site[winner]] = selected_pixel[winner]

    snap_sites(flat, np.ones(flat.size, dtype=bool))
    child_at_pixel = children[flat]
    snap_sites(child_at_pixel, child_at_pixel >= 0)
    if np.any(best_pixel < 0):
        raise RuntimeError("a refinement site has no pixel in its parent domain")
    snapped = centers.copy()
    snapped[:, 0] = (best_pixel % width + 0.5) / width
    snapped[:, 1] = (best_pixel // width + 0.5) / height
    return snapped


def local_hard_partition_with_forest(
    centers: np.ndarray,
    parent_of_centers: np.ndarray,
    parent_labels: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact child transport constrained to the current parent partition."""
    centers = snap_centers_to_parent(
        centers, parent_of_centers, parent_labels)
    restricted = restrict_costs_to_partition(costs, parent_labels)
    forest = hard_partition_with_forest(centers, restricted)
    unreachable = forest["labels"] < 0
    if np.any(unreachable):
        # A hard Voronoi label may contain a tiny tie-created island whose
        # achieving predecessor belongs to the tied neighbor.  Cutting the
        # parent boundary removes that predecessor.  Preserve the established
        # parent on those isolated pixels; they become additional zero-length
        # roots for reverse accumulation rather than leaking across a domain.
        forest["labels"][unreachable] = parent_labels[unreachable]
        forest["runner"][unreachable] = -1
        forest["distance"][unreachable] = 0.0
        forest["second_distance"][unreachable] = np.inf
        forest["parent"][unreachable] = -1
    escaped = parent_of_centers[forest["labels"]] != parent_labels
    if np.any(escaped):
        raise RuntimeError(
            "local topology refresh escaped a parent domain: "
            f"{int(np.count_nonzero(escaped))}/{escaped.size} pixels, "
            f"label range {int(np.min(forest['labels']))}.."
            f"{int(np.max(forest['labels']))}")
    return centers, forest
