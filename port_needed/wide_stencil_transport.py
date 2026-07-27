"""Rotation-fair control for the anisotropic transport walk.

The production walk uses the eight immediate pixel neighbours.  That graph
approximates a strongly anisotropic Riemannian metric by a crystalline norm:
directions between the lattice axes become artificially expensive.  This
module keeps the same exact two-label shortest-path problem, but connects
every primitive lattice direction inside a small square stencil.

Long edges are not allowed to tunnel through the image.  Their cost is the
line integral of the pointwise BFFT metric, sampled at every crossed pixel.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def primitive_directions(radius: int) -> np.ndarray:
    """Return all visible integer directions in a square of ``radius``."""
    radius = max(int(radius), 1)
    directions = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if math.gcd(abs(dx), abs(dy)) != 1:
                continue
            directions.append((dy, dx))
    directions.sort(key=lambda item: math.atan2(item[0], item[1]))
    return np.ascontiguousarray(directions, dtype=np.int32)


def _metric_fields(
    geometry: dict,
    metric_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    scale = max(float(np.percentile(qxx + qyy, 90.0)), 1e-12)
    strength = (
        max(float(metric_strength), 0.0)
        * float(geometry["max_support_px"]) ** 2
    )
    return (
        1.0 + strength * qxx / scale,
        strength * qxy / scale,
        1.0 + strength * qyy / scale,
    )


def build_wide_edge_costs(
    geometry: dict,
    metric_strength: float,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the physical metric along each primitive lattice edge."""
    mxx, mxy, myy = _metric_fields(geometry, metric_strength)
    height, width = mxx.shape
    directions = primitive_directions(radius)
    costs = np.full(
        (len(directions), height, width), np.inf, dtype=np.float32)

    for index, (dy_value, dx_value) in enumerate(directions):
        dy, dx = int(dy_value), int(dx_value)
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        out = np.zeros(mxx[ys, xs].shape, dtype=np.float64)

        # A primitive edge of Chebyshev length k crosses k+1 pixel centres.
        # Trapezoidal endpoint weights make this a line integral rather than
        # an endpoint shortcut, while retaining a fixed and very small loop.
        steps = max(abs(dx), abs(dy))
        offsets = []
        for step in range(steps + 1):
            oy = int(round(step * dy / steps))
            ox = int(round(step * dx / steps))
            if not offsets or offsets[-1] != (oy, ox):
                offsets.append((oy, ox))
        weights = np.ones(len(offsets), dtype=np.float64)
        if len(weights) > 1:
            weights[0] = weights[-1] = 0.5
        weights /= np.sum(weights)

        source_y0 = max(0, -dy)
        source_y1 = min(height, height - dy)
        source_x0 = max(0, -dx)
        source_x1 = min(width, width - dx)
        for weight, (oy, ox) in zip(weights, offsets):
            sy = slice(source_y0 + oy, source_y1 + oy)
            sx = slice(source_x0 + ox, source_x1 + ox)
            quadratic = (
                dx * dx * mxx[sy, sx]
                + 2.0 * dx * dy * mxy[sy, sx]
                + dy * dy * myy[sy, sx]
            )
            out += weight * np.sqrt(np.maximum(quadratic, 1e-12))
        costs[index, ys, xs] = out.astype(np.float32)
    return costs, directions


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _dijkstra_two_best_wide(
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    reach: np.ndarray,
    costs: np.ndarray,
    directions: np.ndarray,
    height: int,
    width: int,
):
    """Exact two-label packed-heap Dijkstra on an arbitrary fixed stencil."""
    pixels = height * width
    infinity = 1e300
    best = np.full(pixels, infinity)
    second = np.full(pixels, infinity)
    owner = np.full(pixels, -1, dtype=np.int32)
    runner = np.full(pixels, -1, dtype=np.int32)
    parent = np.full(pixels, -1, dtype=np.int64)

    capacity = pixels + 4 * len(seed_x) + 256
    heap_distance = np.empty(capacity, dtype=np.float64)
    heap_pixel = np.empty(capacity, dtype=np.int32)
    heap_site = np.empty(capacity, dtype=np.int32)
    size = 0

    for site in range(len(seed_x)):
        pixel = seed_y[site] * width + seed_x[site]
        distance = -reach[site]
        if distance < best[pixel]:
            second[pixel], runner[pixel] = best[pixel], owner[pixel]
            best[pixel], owner[pixel] = distance, site
            parent[pixel] = -1
        elif site != owner[pixel] and distance < second[pixel]:
            second[pixel], runner[pixel] = distance, site
        if size >= capacity:
            capacity *= 2
            new_distance = np.empty(capacity, dtype=np.float64)
            new_pixel = np.empty(capacity, dtype=np.int32)
            new_site = np.empty(capacity, dtype=np.int32)
            new_distance[:size] = heap_distance[:size]
            new_pixel[:size] = heap_pixel[:size]
            new_site[:size] = heap_site[:size]
            heap_distance, heap_pixel, heap_site = (
                new_distance, new_pixel, new_site)
        heap_distance[size] = distance
        heap_pixel[size] = pixel
        heap_site[size] = site
        child = size
        size += 1
        while child > 0:
            ancestor = (child - 1) // 2
            if heap_distance[ancestor] <= heap_distance[child]:
                break
            heap_distance[ancestor], heap_distance[child] = (
                heap_distance[child], heap_distance[ancestor])
            heap_pixel[ancestor], heap_pixel[child] = (
                heap_pixel[child], heap_pixel[ancestor])
            heap_site[ancestor], heap_site[child] = (
                heap_site[child], heap_site[ancestor])
            child = ancestor

    tolerance = 1e-12
    while size > 0:
        distance = heap_distance[0]
        pixel = heap_pixel[0]
        site = heap_site[0]
        size -= 1
        heap_distance[0] = heap_distance[size]
        heap_pixel[0] = heap_pixel[size]
        heap_site[0] = heap_site[size]
        node = 0
        while True:
            left = 2 * node + 1
            right = left + 1
            smallest = node
            if (
                left < size
                and heap_distance[left] < heap_distance[smallest]
            ):
                smallest = left
            if (
                right < size
                and heap_distance[right] < heap_distance[smallest]
            ):
                smallest = right
            if smallest == node:
                break
            heap_distance[node], heap_distance[smallest] = (
                heap_distance[smallest], heap_distance[node])
            heap_pixel[node], heap_pixel[smallest] = (
                heap_pixel[smallest], heap_pixel[node])
            heap_site[node], heap_site[smallest] = (
                heap_site[smallest], heap_site[node])
            node = smallest

        valid = (
            (owner[pixel] == site and distance <= best[pixel] + tolerance)
            or (
                runner[pixel] == site
                and distance <= second[pixel] + tolerance
            )
        )
        if not valid:
            continue
        y = pixel // width
        x = pixel - y * width
        for direction in range(len(directions)):
            ny = y + directions[direction, 0]
            nx = x + directions[direction, 1]
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            neighbour = ny * width + nx
            candidate = distance + costs[direction, y, x]
            touched = False
            first_changed = False
            if owner[neighbour] == site:
                if candidate + tolerance < best[neighbour]:
                    best[neighbour] = candidate
                    touched = True
                    first_changed = True
            elif runner[neighbour] == site:
                if candidate + tolerance < second[neighbour]:
                    second[neighbour] = candidate
                    if second[neighbour] < best[neighbour]:
                        best[neighbour], second[neighbour] = (
                            second[neighbour], best[neighbour])
                        owner[neighbour], runner[neighbour] = (
                            runner[neighbour], owner[neighbour])
                        first_changed = True
                    touched = True
            elif candidate + tolerance < best[neighbour]:
                second[neighbour], runner[neighbour] = (
                    best[neighbour], owner[neighbour])
                best[neighbour], owner[neighbour] = candidate, site
                touched = True
                first_changed = True
            elif candidate + tolerance < second[neighbour]:
                second[neighbour], runner[neighbour] = candidate, site
                touched = True
            if not touched:
                continue
            if first_changed:
                parent[neighbour] = pixel
            if size >= capacity:
                capacity *= 2
                new_distance = np.empty(capacity, dtype=np.float64)
                new_pixel = np.empty(capacity, dtype=np.int32)
                new_site = np.empty(capacity, dtype=np.int32)
                new_distance[:size] = heap_distance[:size]
                new_pixel[:size] = heap_pixel[:size]
                new_site[:size] = heap_site[:size]
                heap_distance, heap_pixel, heap_site = (
                    new_distance, new_pixel, new_site)
            heap_distance[size] = candidate
            heap_pixel[size] = neighbour
            heap_site[size] = site
            child = size
            size += 1
            while child > 0:
                ancestor = (child - 1) // 2
                if heap_distance[ancestor] <= heap_distance[child]:
                    break
                heap_distance[ancestor], heap_distance[child] = (
                    heap_distance[child], heap_distance[ancestor])
                heap_pixel[ancestor], heap_pixel[child] = (
                    heap_pixel[child], heap_pixel[ancestor])
                heap_site[ancestor], heap_site[child] = (
                    heap_site[child], heap_site[ancestor])
                child = ancestor
    return owner, runner, best, second, parent


def walk_wide_two_labels(
    centers: np.ndarray,
    costs: np.ndarray,
    directions: np.ndarray,
    reach: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Normalized-coordinate wrapper matching the canonical forest contract."""
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
    reach_array = (
        np.zeros(len(centers), dtype=np.float64)
        if reach is None
        else np.ascontiguousarray(reach, dtype=np.float64)
    )
    result = _dijkstra_two_best_wide(
        seed_x,
        seed_y,
        reach_array,
        np.ascontiguousarray(costs, dtype=np.float32),
        np.ascontiguousarray(directions, dtype=np.int32),
        height,
        width,
    )
    return {
        "labels": result[0].reshape(height, width),
        "runner": result[1].reshape(height, width),
        "distance": result[2].reshape(height, width),
        "second_distance": result[3].reshape(height, width),
        "parent": result[4].reshape(height, width),
    }
