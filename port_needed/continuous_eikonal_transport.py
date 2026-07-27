"""Continuous-direction anisotropic front propagation.

Unlike edge Dijkstra, this solver updates an arrival value from a whole
opposing simplex edge.  The barycentric position on that edge is continuous,
so path direction is selected by the local energy rather than enumerated.
Each receiver pixel uses its own metric-reduced six-vector stencil.

This first port returns the winning source only.  It is deliberately isolated
from the production two-label walk until the literal support experiment has
validated the continuum geometry.
"""

from __future__ import annotations

import math

import numpy as np

from .metric_reduced_stencil import metric_reduced_superbase

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


def ordered_local_directions(superbase: np.ndarray) -> np.ndarray:
    """Expand three superbasis vectors to one cyclic six-vector stencil."""
    signed = np.concatenate((superbase, -superbase), axis=2)
    angle = np.arctan2(signed[..., 1], signed[..., 0])
    order = np.argsort(angle, axis=2)
    return np.ascontiguousarray(
        np.take_along_axis(signed, order[..., None], axis=2),
        dtype=np.int32,
    )


def inverse_incidence(
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CSR map from an accepted vertex to receivers whose stencil uses it.

    Four immediate-neighbour incidences are included as a connectivity floor.
    FM-LBR assumes a continuous metric; a measured image tensor can jump
    abruptly enough that independently reduced directed stencils cease to be
    strongly connected.  Cardinal edge updates remain valid upper bounds on
    continuum arrival time and do not quantize a successful simplex update.
    """
    height, width = directions.shape[:2]
    pixels = height * width
    local_receiver = np.repeat(np.arange(pixels, dtype=np.int32), 6)
    cardinal_receiver = np.repeat(np.arange(pixels, dtype=np.int32), 4)
    receiver = np.concatenate((local_receiver, cardinal_receiver))
    y = receiver // width
    x = receiver - y * width
    cardinal = np.tile(
        np.array(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=np.int32),
        (pixels, 1),
    )
    flat_direction = np.concatenate((
        directions.reshape(-1, 2),
        cardinal,
    ))
    nx = x + flat_direction[:, 0]
    ny = y + flat_direction[:, 1]
    valid = (0 <= nx) & (nx < width) & (0 <= ny) & (ny < height)
    vertex = (ny[valid] * width + nx[valid]).astype(np.int32)
    receiver = receiver[valid]
    order = np.argsort(vertex, kind="stable")
    vertex = vertex[order]
    receiver = receiver[order]
    count = np.bincount(vertex, minlength=pixels)
    offset = np.empty(pixels + 1, dtype=np.int64)
    offset[0] = 0
    np.cumsum(count, out=offset[1:])
    return np.ascontiguousarray(offset), np.ascontiguousarray(receiver)


@_compile
def _integrated_segment_geometry(
    directions: np.ndarray,
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
    consistency_limit: float,
):
    """Measure every local vector along its full discrete segment."""
    height, width = mxx.shape
    costs = np.full((height, width, 6), np.inf)
    valid = np.zeros((height, width, 6), dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            for index in range(6):
                dx = directions[y, x, index, 0]
                dy = directions[y, x, index, 1]
                target_x = x + dx
                target_y = y + dy
                if (
                    target_x < 0
                    or target_x >= width
                    or target_y < 0
                    or target_y >= height
                ):
                    continue
                steps = max(abs(dx), abs(dy))
                total = 0.0
                total_weight = 0.0
                previous_x = 1 << 30
                previous_y = 1 << 30
                for step in range(steps + 1):
                    offset_x = int(np.rint(step * dx / steps))
                    offset_y = int(np.rint(step * dy / steps))
                    if offset_x == previous_x and offset_y == previous_y:
                        continue
                    previous_x, previous_y = offset_x, offset_y
                    weight = 0.5 if step == 0 or step == steps else 1.0
                    sample_x = x + offset_x
                    sample_y = y + offset_y
                    value = _metric_norm(
                        dx,
                        dy,
                        mxx[sample_y, sample_x],
                        mxy[sample_y, sample_x],
                        myy[sample_y, sample_x],
                    )
                    total += weight * value
                    total_weight += weight
                integrated = total / max(total_weight, 1e-30)
                local = _metric_norm(
                    dx, dy, mxx[y, x], mxy[y, x], myy[y, x])
                ratio = max(
                    integrated / max(local, 1e-30),
                    local / max(integrated, 1e-30),
                )
                costs[y, x, index] = integrated
                valid[y, x, index] = ratio <= consistency_limit
    return costs, valid


@_compile
def _cardinal_segment_costs(
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
):
    height, width = mxx.shape
    result = np.full((height, width, 4), np.inf)
    dxs = (1, -1, 0, 0)
    dys = (0, 0, 1, -1)
    for y in range(height):
        for x in range(width):
            for index in range(4):
                nx = x + dxs[index]
                ny = y + dys[index]
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                a = 0.5 * (mxx[y, x] + mxx[ny, nx])
                b = 0.5 * (mxy[y, x] + mxy[ny, nx])
                c = 0.5 * (myy[y, x] + myy[ny, nx])
                result[y, x, index] = _metric_norm(
                    dxs[index], dys[index], a, b, c)
    return result


@_compile
def _metric_norm(dx, dy, mxx, mxy, myy):
    return math.sqrt(max(
        mxx * dx * dx + 2.0 * mxy * dx * dy + myy * dy * dy,
        1e-30,
    ))


@_compile
def _simplex_candidate_with_fraction(
    first_value,
    second_value,
    first_x,
    first_y,
    second_x,
    second_y,
    mxx,
    mxy,
    myy,
):
    delta_value = second_value - first_value
    delta_x = second_x - first_x
    delta_y = second_y - first_y

    def derivative(t):
        rx = first_x + t * delta_x
        ry = first_y + t * delta_y
        length = _metric_norm(rx, ry, mxx, mxy, myy)
        return delta_value + (
            delta_x * (mxx * rx + mxy * ry)
            + delta_y * (mxy * rx + myy * ry)
        ) / length

    if derivative(0.0) >= 0.0:
        return (
            first_value + _metric_norm(
                first_x, first_y, mxx, mxy, myy),
            0.0,
        )
    if derivative(1.0) <= 0.0:
        return (
            second_value + _metric_norm(
                second_x, second_y, mxx, mxy, myy),
            1.0,
        )
    low, high = 0.0, 1.0
    for _ in range(18):
        middle = 0.5 * (low + high)
        if derivative(middle) < 0.0:
            low = middle
        else:
            high = middle
    t = 0.5 * (low + high)
    rx = first_x + t * delta_x
    ry = first_y + t * delta_y
    return (
        first_value
        + t * delta_value
        + _metric_norm(rx, ry, mxx, mxy, myy),
        t,
    )


@_compile
def _fast_march_first_label(
    seed_pixel: np.ndarray,
    seed_value: np.ndarray,
    seed_label: np.ndarray,
    seed_gradient_x: np.ndarray,
    seed_gradient_y: np.ndarray,
    directions: np.ndarray,
    direction_costs: np.ndarray,
    direction_valid: np.ndarray,
    cardinal_costs: np.ndarray,
    inverse_offset: np.ndarray,
    inverse_receiver: np.ndarray,
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
):
    height, width = mxx.shape
    pixels = height * width
    infinity = 1e300
    tentative = np.full(pixels, infinity)
    tentative_label = np.full(pixels, -1, dtype=np.int32)
    tentative_gradient_x = np.zeros(pixels, dtype=np.float64)
    tentative_gradient_y = np.zeros(pixels, dtype=np.float64)
    tentative_source_gradient_x = np.zeros(pixels, dtype=np.float64)
    tentative_source_gradient_y = np.zeros(pixels, dtype=np.float64)
    tentative_parent_first = np.full(pixels, -1, dtype=np.int32)
    tentative_parent_second = np.full(pixels, -1, dtype=np.int32)
    tentative_parent_fraction = np.zeros(pixels, dtype=np.float64)
    accepted = np.zeros(pixels, dtype=np.bool_)
    distance = np.full(pixels, infinity)
    owner = np.full(pixels, -1, dtype=np.int32)
    gradient_x = np.zeros(pixels, dtype=np.float64)
    gradient_y = np.zeros(pixels, dtype=np.float64)
    source_gradient_x = np.zeros(pixels, dtype=np.float64)
    source_gradient_y = np.zeros(pixels, dtype=np.float64)
    parent_first = np.full(pixels, -1, dtype=np.int32)
    parent_second = np.full(pixels, -1, dtype=np.int32)
    parent_fraction = np.zeros(pixels, dtype=np.float64)
    acceptance_order = np.empty(pixels, dtype=np.int32)
    accepted_count = 0

    capacity = pixels + 4 * len(seed_pixel) + 256
    heap_value = np.empty(capacity, dtype=np.float64)
    heap_pixel = np.empty(capacity, dtype=np.int32)
    heap_label = np.empty(capacity, dtype=np.int32)
    size = 0

    for seed_index in range(len(seed_pixel)):
        pixel = seed_pixel[seed_index]
        value = seed_value[seed_index]
        label = seed_label[seed_index]
        if value >= tentative[pixel]:
            continue
        tentative[pixel] = value
        tentative_label[pixel] = label
        tentative_gradient_x[pixel] = seed_gradient_x[seed_index]
        tentative_gradient_y[pixel] = seed_gradient_y[seed_index]
        tentative_source_gradient_x[pixel] = -seed_gradient_x[seed_index]
        tentative_source_gradient_y[pixel] = -seed_gradient_y[seed_index]
        heap_value[size] = value
        heap_pixel[size] = pixel
        heap_label[size] = label
        child = size
        size += 1
        while child > 0:
            parent = (child - 1) // 2
            if heap_value[parent] <= heap_value[child]:
                break
            heap_value[parent], heap_value[child] = (
                heap_value[child], heap_value[parent])
            heap_pixel[parent], heap_pixel[child] = (
                heap_pixel[child], heap_pixel[parent])
            heap_label[parent], heap_label[child] = (
                heap_label[child], heap_label[parent])
            child = parent

    while size > 0:
        value = heap_value[0]
        pixel = heap_pixel[0]
        label = heap_label[0]
        size -= 1
        heap_value[0] = heap_value[size]
        heap_pixel[0] = heap_pixel[size]
        heap_label[0] = heap_label[size]
        node = 0
        while True:
            left = 2 * node + 1
            right = left + 1
            smallest = node
            if left < size and heap_value[left] < heap_value[smallest]:
                smallest = left
            if right < size and heap_value[right] < heap_value[smallest]:
                smallest = right
            if smallest == node:
                break
            heap_value[node], heap_value[smallest] = (
                heap_value[smallest], heap_value[node])
            heap_pixel[node], heap_pixel[smallest] = (
                heap_pixel[smallest], heap_pixel[node])
            heap_label[node], heap_label[smallest] = (
                heap_label[smallest], heap_label[node])
            node = smallest

        if accepted[pixel]:
            continue
        if (
            label != tentative_label[pixel]
            or value > tentative[pixel] + 1e-12
        ):
            continue
        accepted[pixel] = True
        distance[pixel] = value
        owner[pixel] = label
        gradient_x[pixel] = tentative_gradient_x[pixel]
        gradient_y[pixel] = tentative_gradient_y[pixel]
        source_gradient_x[pixel] = tentative_source_gradient_x[pixel]
        source_gradient_y[pixel] = tentative_source_gradient_y[pixel]
        parent_first[pixel] = tentative_parent_first[pixel]
        parent_second[pixel] = tentative_parent_second[pixel]
        parent_fraction[pixel] = tentative_parent_fraction[pixel]
        acceptance_order[accepted_count] = pixel
        accepted_count += 1

        for incidence in range(
            inverse_offset[pixel], inverse_offset[pixel + 1]
        ):
            receiver = inverse_receiver[incidence]
            if accepted[receiver]:
                continue
            ry = receiver // width
            rx = receiver - ry * width
            a = mxx[ry, rx]
            b = mxy[ry, rx]
            c = myy[ry, rx]
            best_value = tentative[receiver]
            best_label = tentative_label[receiver]
            best_gradient_x = tentative_gradient_x[receiver]
            best_gradient_y = tentative_gradient_y[receiver]
            best_source_gradient_x = tentative_source_gradient_x[receiver]
            best_source_gradient_y = tentative_source_gradient_y[receiver]
            best_parent_first = tentative_parent_first[receiver]
            best_parent_second = tentative_parent_second[receiver]
            best_parent_fraction = tentative_parent_fraction[receiver]

            cardinal_x = (1, -1, 0, 0)
            cardinal_y = (0, 0, 1, -1)
            for cardinal_index in range(4):
                ux = cardinal_x[cardinal_index]
                uy = cardinal_y[cardinal_index]
                neighbour_x = rx + ux
                neighbour_y = ry + uy
                if (
                    neighbour_x < 0
                    or neighbour_x >= width
                    or neighbour_y < 0
                    or neighbour_y >= height
                ):
                    continue
                neighbour = neighbour_y * width + neighbour_x
                if not accepted[neighbour]:
                    continue
                candidate = (
                    distance[neighbour]
                    + cardinal_costs[ry, rx, cardinal_index]
                )
                if candidate < best_value:
                    best_value = candidate
                    best_label = owner[neighbour]
                    local_length = _metric_norm(ux, uy, a, b, c)
                    best_gradient_x = -(a * ux + b * uy) / local_length
                    best_gradient_y = -(b * ux + c * uy) / local_length
                    best_source_gradient_x = source_gradient_x[neighbour]
                    best_source_gradient_y = source_gradient_y[neighbour]
                    best_parent_first = neighbour
                    best_parent_second = -1
                    best_parent_fraction = 0.0

            for index in range(6):
                ux = directions[ry, rx, index, 0]
                uy = directions[ry, rx, index, 1]
                vx = directions[ry, rx, (index + 1) % 6, 0]
                vy = directions[ry, rx, (index + 1) % 6, 1]
                first_x = rx + ux
                first_y = ry + uy
                second_x = rx + vx
                second_y = ry + vy
                first_inside = (
                    0 <= first_x < width and 0 <= first_y < height)
                second_inside = (
                    0 <= second_x < width and 0 <= second_y < height)
                first_pixel = (
                    first_y * width + first_x if first_inside else -1)
                second_pixel = (
                    second_y * width + second_x if second_inside else -1)

                if first_inside and accepted[first_pixel]:
                    candidate = (
                        distance[first_pixel]
                        + direction_costs[ry, rx, index]
                    )
                    if direction_valid[ry, rx, index] and candidate < best_value:
                        best_value = candidate
                        best_label = owner[first_pixel]
                        local_length = _metric_norm(ux, uy, a, b, c)
                        best_gradient_x = (
                            -(a * ux + b * uy) / local_length)
                        best_gradient_y = (
                            -(b * ux + c * uy) / local_length)
                        best_source_gradient_x = (
                            source_gradient_x[first_pixel])
                        best_source_gradient_y = (
                            source_gradient_y[first_pixel])
                        best_parent_first = first_pixel
                        best_parent_second = -1
                        best_parent_fraction = 0.0
                if second_inside and accepted[second_pixel]:
                    candidate = (
                        distance[second_pixel]
                        + direction_costs[ry, rx, (index + 1) % 6]
                    )
                    if (
                        direction_valid[ry, rx, (index + 1) % 6]
                        and candidate < best_value
                    ):
                        best_value = candidate
                        best_label = owner[second_pixel]
                        local_length = _metric_norm(vx, vy, a, b, c)
                        best_gradient_x = (
                            -(a * vx + b * vy) / local_length)
                        best_gradient_y = (
                            -(b * vx + c * vy) / local_length)
                        best_source_gradient_x = (
                            source_gradient_x[second_pixel])
                        best_source_gradient_y = (
                            source_gradient_y[second_pixel])
                        best_parent_first = second_pixel
                        best_parent_second = -1
                        best_parent_fraction = 0.0
                if (
                    first_inside
                    and second_inside
                    and accepted[first_pixel]
                    and accepted[second_pixel]
                    and owner[first_pixel] == owner[second_pixel]
                    and direction_valid[ry, rx, index]
                    and direction_valid[ry, rx, (index + 1) % 6]
                ):
                    candidate, fraction = _simplex_candidate_with_fraction(
                        distance[first_pixel],
                        distance[second_pixel],
                        ux,
                        uy,
                        vx,
                        vy,
                        a,
                        b,
                        c,
                    )
                    if candidate < best_value:
                        best_value = candidate
                        best_label = owner[first_pixel]
                        foot_x = ux + fraction * (vx - ux)
                        foot_y = uy + fraction * (vy - uy)
                        local_length = _metric_norm(
                            foot_x, foot_y, a, b, c)
                        best_gradient_x = -(
                            a * foot_x + b * foot_y) / local_length
                        best_gradient_y = -(
                            b * foot_x + c * foot_y) / local_length
                        best_source_gradient_x = (
                            (1.0 - fraction)
                            * source_gradient_x[first_pixel]
                            + fraction
                            * source_gradient_x[second_pixel]
                        )
                        best_source_gradient_y = (
                            (1.0 - fraction)
                            * source_gradient_y[first_pixel]
                            + fraction
                            * source_gradient_y[second_pixel]
                        )
                        best_parent_first = first_pixel
                        best_parent_second = second_pixel
                        best_parent_fraction = fraction

            if best_value + 1e-12 >= tentative[receiver]:
                continue
            tentative[receiver] = best_value
            tentative_label[receiver] = best_label
            tentative_gradient_x[receiver] = best_gradient_x
            tentative_gradient_y[receiver] = best_gradient_y
            tentative_source_gradient_x[receiver] = best_source_gradient_x
            tentative_source_gradient_y[receiver] = best_source_gradient_y
            tentative_parent_first[receiver] = best_parent_first
            tentative_parent_second[receiver] = best_parent_second
            tentative_parent_fraction[receiver] = best_parent_fraction
            if size >= capacity:
                capacity *= 2
                new_value = np.empty(capacity, dtype=np.float64)
                new_pixel = np.empty(capacity, dtype=np.int32)
                new_label = np.empty(capacity, dtype=np.int32)
                new_value[:size] = heap_value[:size]
                new_pixel[:size] = heap_pixel[:size]
                new_label[:size] = heap_label[:size]
                heap_value, heap_pixel, heap_label = (
                    new_value, new_pixel, new_label)
            heap_value[size] = best_value
            heap_pixel[size] = receiver
            heap_label[size] = best_label
            child = size
            size += 1
            while child > 0:
                parent = (child - 1) // 2
                if heap_value[parent] <= heap_value[child]:
                    break
                heap_value[parent], heap_value[child] = (
                    heap_value[child], heap_value[parent])
                heap_pixel[parent], heap_pixel[child] = (
                    heap_pixel[child], heap_pixel[parent])
                heap_label[parent], heap_label[child] = (
                    heap_label[child], heap_label[parent])
                child = parent
    return (
        owner,
        distance,
        gradient_x,
        gradient_y,
        source_gradient_x,
        source_gradient_y,
        parent_first,
        parent_second,
        parent_fraction,
        acceptance_order[:accepted_count],
    )


def prepare_continuous_metric(
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
    consistency_limit: float = 1.75,
) -> dict[str, np.ndarray]:
    """Precompute the receiver-adapted causal simplices once per image."""
    mxx = np.ascontiguousarray(mxx, dtype=np.float64)
    mxy = np.ascontiguousarray(mxy, dtype=np.float64)
    myy = np.ascontiguousarray(myy, dtype=np.float64)
    superbase = metric_reduced_superbase(mxx, mxy, myy)
    directions = ordered_local_directions(superbase)
    inverse_offset, inverse_receiver = inverse_incidence(directions)
    direction_costs, direction_valid = _integrated_segment_geometry(
        directions,
        mxx,
        mxy,
        myy,
        max(float(consistency_limit), 1.0),
    )
    cardinal_costs = _cardinal_segment_costs(mxx, mxy, myy)
    return {
        "mxx": mxx,
        "mxy": mxy,
        "myy": myy,
        "superbase": superbase,
        "directions": directions,
        "direction_costs": direction_costs,
        "direction_valid": direction_valid,
        "cardinal_costs": cardinal_costs,
        "inverse_offset": inverse_offset,
        "inverse_receiver": inverse_receiver,
    }


def continuous_first_partition_prepared(
    centers: np.ndarray,
    prepared: dict[str, np.ndarray],
    reach: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Solve a multi-source front using a precomputed local metric mesh."""
    mxx = prepared["mxx"]
    height, width = mxx.shape
    centers = np.asarray(centers, dtype=np.float64)
    center_x = np.clip(
        centers[:, 0] * width - 0.5, 0.0, width - 1.0)
    center_y = np.clip(
        centers[:, 1] * height - 0.5, 0.0, height - 1.0)
    source_reach = (
        np.zeros(len(centers), dtype=np.float64)
        if reach is None
        else np.ascontiguousarray(reach, dtype=np.float64)
    )

    # A source is a continuous point, not a snapped pixel.  Its four enclosing
    # pixel centres receive exact local metric action.  This removes the
    # half-pixel dead zone that otherwise makes centroid/force relaxation
    # discontinuous and axis-biased.
    seed_pixels = []
    seed_values = []
    seed_labels = []
    seed_gradient_x = []
    seed_gradient_y = []
    for label, (sx, sy) in enumerate(zip(center_x, center_y)):
        x0 = int(math.floor(sx))
        x1 = min(x0 + 1, width - 1)
        y0 = int(math.floor(sy))
        y1 = min(y0 + 1, height - 1)
        for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            dx = float(px) - sx
            dy = float(py) - sy
            a = prepared["mxx"][py, px]
            b = prepared["mxy"][py, px]
            c = prepared["myy"][py, px]
            local_length = math.sqrt(max(
                a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
                0.0,
            ))
            if local_length > 1e-15:
                gx = (a * dx + b * dy) / local_length
                gy = (b * dx + c * dy) / local_length
            else:
                gx = 0.0
                gy = 0.0
            seed_pixels.append(py * width + px)
            seed_values.append(-source_reach[label] + local_length)
            seed_labels.append(label)
            seed_gradient_x.append(gx)
            seed_gradient_y.append(gy)
    (
        owner,
        distance,
        gradient_x,
        gradient_y,
        source_gradient_x,
        source_gradient_y,
        parent_first,
        parent_second,
        parent_fraction,
        acceptance_order,
    ) = _fast_march_first_label(
        np.ascontiguousarray(seed_pixels, dtype=np.int32),
        np.ascontiguousarray(seed_values, dtype=np.float64),
        np.ascontiguousarray(seed_labels, dtype=np.int32),
        np.ascontiguousarray(seed_gradient_x, dtype=np.float64),
        np.ascontiguousarray(seed_gradient_y, dtype=np.float64),
        prepared["directions"],
        prepared["direction_costs"],
        prepared["direction_valid"],
        prepared["cardinal_costs"],
        prepared["inverse_offset"],
        prepared["inverse_receiver"],
        prepared["mxx"],
        prepared["mxy"],
        prepared["myy"],
    )
    return {
        "labels": owner.reshape(height, width),
        "distance": distance.reshape(height, width),
        "gradient_x": gradient_x.reshape(height, width),
        "gradient_y": gradient_y.reshape(height, width),
        "source_gradient_x": source_gradient_x.reshape(height, width),
        "source_gradient_y": source_gradient_y.reshape(height, width),
        "parent_first": parent_first.reshape(height, width),
        "parent_second": parent_second.reshape(height, width),
        "parent_fraction": parent_fraction.reshape(height, width),
        "acceptance_order": acceptance_order,
        "superbase": prepared["superbase"],
        "directions": prepared["directions"],
    }


def continuous_first_partition(
    centers: np.ndarray,
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
    reach: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Solve the multi-source anisotropic eikonal equation on the image."""
    prepared = prepare_continuous_metric(mxx, mxy, myy)
    return continuous_first_partition_prepared(centers, prepared, reach)
