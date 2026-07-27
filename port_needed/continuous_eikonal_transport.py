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

from bfft.vision import fast_march_first_label_native

from .metric_reduced_stencil import metric_reduced_superbase

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _ordered_local_directions_kernel(
    superbase: np.ndarray,
) -> np.ndarray:
    """Cyclically order six fixed vectors without angles or a general sort."""
    height, width = superbase.shape[:2]
    result = np.empty((height, width, 6, 2), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            for index in range(3):
                result[y, x, index, 0] = superbase[y, x, index, 0]
                result[y, x, index, 1] = superbase[y, x, index, 1]
                result[y, x, index + 3, 0] = -superbase[y, x, index, 0]
                result[y, x, index + 3, 1] = -superbase[y, x, index, 1]

            # Six elements is a topological constant.  This insertion network
            # compares half-planes and cross products; it does not perform an
            # image-scale ordering or evaluate atan2.
            for index in range(1, 6):
                key_x = result[y, x, index, 0]
                key_y = result[y, x, index, 1]
                key_half = (
                    0 if key_y < 0 or (key_y == 0 and key_x >= 0) else 1)
                position = index
                while position > 0:
                    other_x = result[y, x, position - 1, 0]
                    other_y = result[y, x, position - 1, 1]
                    other_half = (
                        0
                        if other_y < 0 or (other_y == 0 and other_x >= 0)
                        else 1
                    )
                    cross = other_x * key_y - other_y * key_x
                    comes_before = (
                        key_half < other_half
                        or (key_half == other_half and cross < 0)
                    )
                    if not comes_before:
                        break
                    result[y, x, position, 0] = other_x
                    result[y, x, position, 1] = other_y
                    position -= 1
                result[y, x, position, 0] = key_x
                result[y, x, position, 1] = key_y
    return result


def ordered_local_directions(superbase: np.ndarray) -> np.ndarray:
    """Expand three superbasis vectors to one cyclic six-vector stencil."""
    return _ordered_local_directions_kernel(
        np.ascontiguousarray(superbase, dtype=np.int32))


@_compile
def _inverse_incidence_linear(
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build unique vertex-to-receiver CSR incidence in linear time."""
    height, width = directions.shape[:2]
    pixels = height * width
    count = np.zeros(pixels, dtype=np.int64)
    card_x = (1, -1, 0, 0)
    card_y = (0, 0, 1, -1)

    # Local reduced directions first, matching the old stable ordering.
    for receiver in range(pixels):
        y = receiver // width
        x = receiver - y * width
        for index in range(6):
            nx = x + directions[y, x, index, 0]
            ny = y + directions[y, x, index, 1]
            if 0 <= nx < width and 0 <= ny < height:
                count[ny * width + nx] += 1

    # Cardinal edges are a connectivity floor.  Do not duplicate an
    # incidence already supplied by the locally reduced stencil.
    for receiver in range(pixels):
        y = receiver // width
        x = receiver - y * width
        for index in range(4):
            dx = card_x[index]
            dy = card_y[index]
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            duplicate = False
            for local in range(6):
                if (
                    directions[y, x, local, 0] == dx
                    and directions[y, x, local, 1] == dy
                ):
                    duplicate = True
                    break
            if not duplicate:
                count[ny * width + nx] += 1

    offset = np.empty(pixels + 1, dtype=np.int64)
    offset[0] = 0
    for pixel in range(pixels):
        offset[pixel + 1] = offset[pixel] + count[pixel]
    receiver_index = np.empty(offset[pixels], dtype=np.int32)
    cursor = offset[:-1].copy()

    for receiver in range(pixels):
        y = receiver // width
        x = receiver - y * width
        for index in range(6):
            nx = x + directions[y, x, index, 0]
            ny = y + directions[y, x, index, 1]
            if 0 <= nx < width and 0 <= ny < height:
                vertex = ny * width + nx
                receiver_index[cursor[vertex]] = receiver
                cursor[vertex] += 1
    for receiver in range(pixels):
        y = receiver // width
        x = receiver - y * width
        for index in range(4):
            dx = card_x[index]
            dy = card_y[index]
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            duplicate = False
            for local in range(6):
                if (
                    directions[y, x, local, 0] == dx
                    and directions[y, x, local, 1] == dy
                ):
                    duplicate = True
                    break
            if not duplicate:
                vertex = ny * width + nx
                receiver_index[cursor[vertex]] = receiver
                cursor[vertex] += 1
    return offset, receiver_index


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
    return _inverse_incidence_linear(
        np.ascontiguousarray(directions, dtype=np.int32))


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
    quadratic_a = (
        mxx * delta_x * delta_x
        + 2.0 * mxy * delta_x * delta_y
        + myy * delta_y * delta_y
    )
    quadratic_b = (
        delta_x * (mxx * first_x + mxy * first_y)
        + delta_y * (mxy * first_x + myy * first_y)
    )
    quadratic_c = (
        mxx * first_x * first_x
        + 2.0 * mxy * first_x * first_y
        + myy * first_y * first_y
    )
    first_length = math.sqrt(max(quadratic_c, 1e-30))
    derivative_first = delta_value + quadratic_b / first_length
    if derivative_first >= 0.0:
        return (
            first_value + first_length,
            0.0,
        )
    second_quadratic = (
        quadratic_a + 2.0 * quadratic_b + quadratic_c)
    second_length = math.sqrt(max(second_quadratic, 1e-30))
    derivative_second = (
        delta_value
        + (quadratic_a + quadratic_b) / second_length)
    if derivative_second <= 0.0:
        return (
            second_value + second_length,
            1.0,
        )

    # The interior Hopf--Lax minimizer is analytic.  With
    # q(t)=A t^2+2Bt+C and d=second_value-first_value, stationarity is
    # d+(At+B)/sqrt(q)=0.  Solving it without squaring away the sign gives
    #
    #   At+B = -d sqrt((AC-B^2)/(A-d^2)).
    #
    # The endpoint derivative test above implies A-d^2 > 0 for an interior
    # causal update.  This replaces eighteen bisection rounds.
    area = max(
        quadratic_a * quadratic_c - quadratic_b * quadratic_b,
        0.0,
    )
    causal = max(
        quadratic_a - delta_value * delta_value,
        1e-30,
    )
    t = (
        -quadratic_b
        - delta_value * math.sqrt(area / causal)
    ) / max(quadratic_a, 1e-30)
    t = min(max(t, 0.0), 1.0)
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

    # One tentative value exists per unaccepted pixel, so the priority queue
    # also needs exactly one entry per pixel.  Decrease-key relocates that
    # entry in place; stale duplicates and heap growth are impossible.
    heap_value = np.empty(pixels, dtype=np.float64)
    heap_pixel = np.empty(pixels, dtype=np.int32)
    heap_position = np.full(pixels, -1, dtype=np.int32)
    size = 0
    push_count = 0
    maximum_heap_size = 0

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
        child = heap_position[pixel]
        if child < 0:
            child = size
            heap_pixel[child] = pixel
            heap_position[pixel] = child
            size += 1
        heap_value[child] = value
        push_count += 1
        maximum_heap_size = max(maximum_heap_size, size)
        while child > 0:
            parent = (child - 1) // 2
            if heap_value[parent] <= heap_value[child]:
                break
            heap_value[parent], heap_value[child] = (
                heap_value[child], heap_value[parent])
            heap_pixel[parent], heap_pixel[child] = (
                heap_pixel[child], heap_pixel[parent])
            heap_position[heap_pixel[parent]] = parent
            heap_position[heap_pixel[child]] = child
            child = parent

    while size > 0:
        value = heap_value[0]
        pixel = heap_pixel[0]
        size -= 1
        heap_position[pixel] = -2
        if size > 0:
            heap_value[0] = heap_value[size]
            heap_pixel[0] = heap_pixel[size]
            heap_position[heap_pixel[0]] = 0
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
                heap_position[heap_pixel[node]] = node
                heap_position[heap_pixel[smallest]] = smallest
                node = smallest

        label = tentative_label[pixel]
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
            child = heap_position[receiver]
            if child < 0:
                child = size
                heap_pixel[child] = receiver
                heap_position[receiver] = child
                size += 1
            heap_value[child] = best_value
            push_count += 1
            maximum_heap_size = max(maximum_heap_size, size)
            while child > 0:
                parent = (child - 1) // 2
                if heap_value[parent] <= heap_value[child]:
                    break
                heap_value[parent], heap_value[child] = (
                    heap_value[child], heap_value[parent])
                heap_pixel[parent], heap_pixel[child] = (
                    heap_pixel[child], heap_pixel[parent])
                heap_position[heap_pixel[parent]] = parent
                heap_position[heap_pixel[child]] = child
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
        push_count,
        maximum_heap_size,
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
    x0 = np.floor(center_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y0 = np.floor(center_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, height - 1)
    seed_x = np.column_stack((x0, x1, x0, x1)).ravel()
    seed_y = np.column_stack((y0, y0, y1, y1)).ravel()
    seed_labels = np.repeat(
        np.arange(len(centers), dtype=np.int32), 4)
    dx = seed_x.astype(np.float64) - center_x[seed_labels]
    dy = seed_y.astype(np.float64) - center_y[seed_labels]
    a = prepared["mxx"][seed_y, seed_x]
    b = prepared["mxy"][seed_y, seed_x]
    c = prepared["myy"][seed_y, seed_x]
    seed_values = np.sqrt(np.maximum(
        a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
        0.0,
    ))
    nonzero = seed_values > 1e-15
    seed_gradient_x = np.zeros_like(seed_values)
    seed_gradient_y = np.zeros_like(seed_values)
    seed_gradient_x[nonzero] = (
        a[nonzero] * dx[nonzero] + b[nonzero] * dy[nonzero]
    ) / seed_values[nonzero]
    seed_gradient_y[nonzero] = (
        b[nonzero] * dx[nonzero] + c[nonzero] * dy[nonzero]
    ) / seed_values[nonzero]
    seed_values -= source_reach[seed_labels]
    seed_pixels = seed_y * width + seed_x
    native = fast_march_first_label_native(
        np.ascontiguousarray(seed_pixels, dtype=np.int32),
        np.ascontiguousarray(seed_values, dtype=np.float64),
        np.ascontiguousarray(seed_labels, dtype=np.int32),
        np.ascontiguousarray(seed_gradient_x, dtype=np.float64),
        np.ascontiguousarray(seed_gradient_y, dtype=np.float64),
        prepared,
    )
    if native is None:
        native = _fast_march_first_label(
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
        push_count,
        maximum_heap_size,
    ) = native
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
        "front_pushes": int(push_count),
        "front_maximum_heap": int(maximum_heap_size),
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
