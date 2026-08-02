"""Compound segmentation IDs above immutable reconstruction atoms.

The reconstruction partition and the segmentation partition answer different
questions.  A curved glyph or object contour may require several local fit
atoms even when it is one segment, while one expressive atom may carry both
sides of an authentic discontinuity.  This module therefore never refits or
contracts a reconstruction model.

It first resolves every atom into a fixed-depth pair of one-sided amplitude
supports.  The resulting connected leaves are then joined by one Kruskal pass
over their measured interface barrier.  Cutting that tree at the original
atom count gives a literal same-ID-budget A/B against the reconstruction IDs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _compile(function):
    return function if njit is None else njit(cache=True)(function)


def _compile_nogil(function):
    return function if njit is None else njit(cache=True, nogil=True)(function)


@_compile
def _connected_equal_components(keys):
    """Return compact four-connected components of an integer key image."""
    height, width = keys.shape
    pixels = height * width
    parent = np.arange(pixels, dtype=np.int32)
    size = np.ones(pixels, dtype=np.int32)

    for y in range(height):
        for x in range(width):
            pixel = y * width + x
            if x > 0 and keys[y, x - 1] == keys[y, x]:
                first = pixel
                while parent[first] != first:
                    parent[first] = parent[parent[first]]
                    first = parent[first]
                second = pixel - 1
                while parent[second] != second:
                    parent[second] = parent[parent[second]]
                    second = parent[second]
                if first != second:
                    if size[first] < size[second]:
                        first, second = second, first
                    parent[second] = first
                    size[first] += size[second]
            if y > 0 and keys[y - 1, x] == keys[y, x]:
                first = pixel
                while parent[first] != first:
                    parent[first] = parent[parent[first]]
                    first = parent[first]
                second = pixel - width
                while parent[second] != second:
                    parent[second] = parent[parent[second]]
                    second = parent[second]
                if first != second:
                    if size[first] < size[second]:
                        first, second = second, first
                    parent[second] = first
                    size[first] += size[second]

    root_to_label = np.full(pixels, -1, dtype=np.int32)
    labels = np.empty((height, width), dtype=np.int32)
    count = 0
    for pixel in range(pixels):
        root = pixel
        while parent[root] != root:
            root = parent[root]
        parent[pixel] = root
        if root_to_label[root] < 0:
            root_to_label[root] = count
            count += 1
        labels[pixel // width, pixel % width] = root_to_label[root]
    return labels, count


@_compile
def _compact_keys(keys):
    flat = keys.ravel()
    maximum = 0
    for index in range(len(flat)):
        maximum = max(maximum, flat[index])
    present = np.zeros(maximum + 1, dtype=np.uint8)
    for index in range(len(flat)):
        present[flat[index]] = 1
    remap = np.full(maximum + 1, -1, dtype=np.int32)
    count = 0
    for key in range(maximum + 1):
        if present[key]:
            remap[key] = count
            count += 1
    output = np.empty(keys.shape, dtype=np.int32)
    for y in range(keys.shape[0]):
        for x in range(keys.shape[1]):
            output[y, x] = remap[keys[y, x]]
    return output, count


@_compile
def _atom_sufficient_statistics(labels, target, cells):
    """Fuse RGB first and symmetric second moments into one raster pass."""
    count = np.zeros(cells, dtype=np.float64)
    total = np.zeros((cells, 3), dtype=np.float64)
    second = np.zeros((cells, 3, 3), dtype=np.float64)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            cell = labels[y, x]
            count[cell] += 1.0
            first = target[y, x, 0]
            second_value = target[y, x, 1]
            third = target[y, x, 2]
            total[cell, 0] += first
            total[cell, 1] += second_value
            total[cell, 2] += third
            second[cell, 0, 0] += first * first
            second[cell, 0, 1] += first * second_value
            second[cell, 0, 2] += first * third
            second[cell, 1, 1] += second_value * second_value
            second[cell, 1, 2] += second_value * third
            second[cell, 2, 2] += third * third
    for cell in range(cells):
        second[cell, 1, 0] = second[cell, 0, 1]
        second[cell, 2, 0] = second[cell, 0, 2]
        second[cell, 2, 1] = second[cell, 1, 2]
    return count, total, second


@_compile
def _leaf_interface_statistics(labels, target, boundary, cells):
    """Accumulate internal continuation and unique leaf interfaces once."""
    capacity = 1024
    # Retain the established slot layout as part of deterministic equal-edge
    # tie ordering.  A smaller still-safe planar table changed the extraction
    # order and measurably weakened the curved-boundary truth control.
    target_capacity = max(8 * cells, 1024)
    while capacity < target_capacity:
        capacity *= 2
    table = np.full(capacity, -1, dtype=np.int64)
    cross_sum = np.zeros(capacity, dtype=np.float64)
    cross_count = np.zeros(capacity, dtype=np.int64)
    boundary_sum = np.zeros(capacity, dtype=np.float64)
    internal_sum = np.zeros(cells, dtype=np.float64)
    internal_count = np.zeros(cells, dtype=np.int64)
    population = np.zeros(cells, dtype=np.float64)
    leaf_sum = np.zeros((cells, 3), dtype=np.float64)
    leaf_square = np.zeros(cells, dtype=np.float64)
    mask = capacity - 1
    used = 0
    height, width = labels.shape

    for y in range(height):
        for x in range(width):
            first = labels[y, x]
            population[first] += 1.0
            square = 0.0
            for channel in range(3):
                value = target[y, x, channel]
                leaf_sum[first, channel] += value
                square += value * value
            leaf_square[first] += square
            if x + 1 < width:
                second = labels[y, x + 1]
                energy = 0.0
                for channel in range(3):
                    difference = target[y, x, channel] - target[y, x + 1, channel]
                    energy += difference * difference
                if first == second:
                    internal_sum[first] += energy
                    internal_count[first] += 1
                else:
                    low = min(first, second)
                    high = max(first, second)
                    key = np.int64(low) * cells + high
                    slot = key & mask
                    while table[slot] != -1 and table[slot] != key:
                        slot = (slot + 1) & mask
                    if table[slot] == -1:
                        if (used + 1) * 10 >= capacity * 7:
                            raise RuntimeError(
                                "compound interface graph exceeded its "
                                "planar hash ceiling")
                        table[slot] = key
                        used += 1
                    cross_sum[slot] += energy
                    cross_count[slot] += 1
                    boundary_sum[slot] += 0.5 * (
                        boundary[y, x] + boundary[y, x + 1])
            if y + 1 < height:
                second = labels[y + 1, x]
                energy = 0.0
                for channel in range(3):
                    difference = target[y, x, channel] - target[y + 1, x, channel]
                    energy += difference * difference
                if first == second:
                    internal_sum[first] += energy
                    internal_count[first] += 1
                else:
                    low = min(first, second)
                    high = max(first, second)
                    key = np.int64(low) * cells + high
                    slot = key & mask
                    while table[slot] != -1 and table[slot] != key:
                        slot = (slot + 1) & mask
                    if table[slot] == -1:
                        if (used + 1) * 10 >= capacity * 7:
                            raise RuntimeError(
                                "compound interface graph exceeded its "
                                "planar hash ceiling")
                        table[slot] = key
                        used += 1
                    cross_sum[slot] += energy
                    cross_count[slot] += 1
                    boundary_sum[slot] += 0.5 * (
                        boundary[y, x] + boundary[y + 1, x])

    pairs = np.empty((used, 2), dtype=np.int32)
    interface_sum = np.empty(used, dtype=np.float64)
    interface_count = np.empty(used, dtype=np.int64)
    interface_boundary = np.empty(used, dtype=np.float64)
    index = 0
    for slot in range(capacity):
        key = table[slot]
        if key >= 0:
            pairs[index, 0] = key // cells
            pairs[index, 1] = key % cells
            interface_sum[index] = cross_sum[slot]
            interface_count[index] = cross_count[slot]
            interface_boundary[index] = boundary_sum[slot]
            index += 1
    return (
        pairs,
        interface_sum,
        interface_count,
        interface_boundary,
        internal_sum,
        internal_count,
        population,
        leaf_sum,
        leaf_square,
    )


@_compile
def _minimum_barrier_tree(cells, pairs, order, barrier):
    parent = np.arange(cells, dtype=np.int32)
    size = np.ones(cells, dtype=np.int32)
    tree_first = np.empty(max(cells - 1, 0), dtype=np.int32)
    tree_second = np.empty(max(cells - 1, 0), dtype=np.int32)
    tree_barrier = np.empty(max(cells - 1, 0), dtype=np.float64)
    used = 0
    for position in range(len(order)):
        edge = order[position]
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
        tree_first[used] = pairs[edge, 0]
        tree_second[used] = pairs[edge, 1]
        tree_barrier[used] = barrier[edge]
        used += 1
        if size[first] < size[second]:
            first, second = second, first
        parent[second] = first
        size[first] += size[second]
    return tree_first[:used], tree_second[:used], tree_barrier[:used]


@_compile
def _labels_at_count(leaf_labels, cells, first, second, target_count):
    parent = np.arange(cells, dtype=np.int32)
    size = np.ones(cells, dtype=np.int32)
    merges = min(max(cells - target_count, 0), len(first))
    for edge in range(merges):
        a = first[edge]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        b = second[edge]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    root_label = np.full(cells, -1, dtype=np.int32)
    next_label = 0
    output = np.empty(leaf_labels.shape, dtype=np.int32)
    height, width = leaf_labels.shape
    for y in range(height):
        for x in range(width):
            root = leaf_labels[y, x]
            while parent[root] != root:
                root = parent[root]
            if root_label[root] < 0:
                root_label[root] = next_label
                next_label += 1
            output[y, x] = root_label[root]
    return output, next_label


@_compile
def _adaptive_roots(
    cells, pairs, order, barrier, population, scale, stop_count,
):
    """Run one size-conditioned FH pass on a fixed sorted leaf graph."""
    parent = np.arange(cells, dtype=np.int32)
    size = population.copy()
    internal = np.zeros(cells, dtype=np.float64)
    maximum_internal = 0.0
    components = cells
    for position in range(len(order)):
        edge = order[position]
        value = barrier[edge]
        # Edges are sorted.  Every component threshold is bounded above by
        # max(internal) + scale because every population is at least one.  If
        # this edge exceeds that bound, no edge in the remaining suffix can
        # ever be accepted.
        if value > maximum_internal + scale:
            break
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
        first_limit = internal[first] + scale / max(size[first], 1.0)
        second_limit = internal[second] + scale / max(size[second], 1.0)
        if value > min(first_limit, second_limit):
            continue
        if size[first] < size[second]:
            first, second = second, first
        parent[second] = first
        size[first] += size[second]
        internal[first] = max(value, internal[first], internal[second])
        maximum_internal = max(maximum_internal, internal[first])
        components -= 1
        if stop_count > 0 and components <= stop_count:
            break
    for cell in range(cells):
        root = cell
        while parent[root] != root:
            root = parent[root]
        parent[cell] = root
    return parent, components


@_compile_nogil
def _adaptive_component_count(
    cells, pairs, order, barrier, population, scale,
):
    """Calibration-only FH pass: omit the unused final root compression."""
    parent = np.arange(cells, dtype=np.int32)
    size = population.copy()
    internal = np.zeros(cells, dtype=np.float64)
    maximum_internal = 0.0
    components = cells
    for position in range(len(order)):
        edge = order[position]
        value = barrier[edge]
        if value > maximum_internal + scale:
            break
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
        if value > min(
            internal[first] + scale / max(size[first], 1.0),
            internal[second] + scale / max(size[second], 1.0),
        ):
            continue
        if size[first] < size[second]:
            first, second = second, first
        parent[second] = first
        size[first] += size[second]
        internal[first] = max(value, internal[first], internal[second])
        maximum_internal = max(maximum_internal, internal[first])
        components -= 1
    return components


@_compile
def _labels_from_roots(leaf_labels, roots):
    root_label = np.full(len(roots), -1, dtype=np.int32)
    next_label = 0
    output = np.empty(leaf_labels.shape, dtype=np.int32)
    height, width = leaf_labels.shape
    for y in range(height):
        for x in range(width):
            root = roots[leaf_labels[y, x]]
            if root_label[root] < 0:
                root_label[root] = next_label
                next_label += 1
            output[y, x] = root_label[root]
    return output, next_label


def _adaptive_labels_at_count(
    leaf_labels: np.ndarray,
    cells: int,
    pairs: np.ndarray,
    order: np.ndarray,
    barrier: np.ndarray,
    population: np.ndarray,
    target_count: int,
    calibration_steps: int = 6,
    parallel_calibration: bool | None = None,
) -> tuple[np.ndarray, int, float]:
    """Calibrate only the graph scale to a requested comparison budget."""
    target = int(np.clip(target_count, 1, cells))
    if target == cells:
        roots = np.arange(cells, dtype=np.int32)
        labels, count = _labels_from_roots(leaf_labels, roots)
        return labels, count, 0.0

    low = 0.0
    high = max(float(np.median(barrier)) * float(np.mean(population)), 1e-12)
    steps = max(int(calibration_steps), 0)
    parallel = njit is not None and (
        cells >= 16384
        if parallel_calibration is None
        else bool(parallel_calibration)
    )
    # Compile/load the nogil count kernel before any concurrent calls.  This
    # first probe is also the first literal probe in the original bracket.
    count = _adaptive_component_count(
        cells, pairs, order, barrier, population, high)
    if count > target:
        low = high
        if parallel:
            tested = 1
            with ThreadPoolExecutor(max_workers=3) as executor:
                while tested < 32:
                    batch_size = min(3, 32 - tested)
                    scales = [high * (2.0 ** (index + 1))
                              for index in range(batch_size)]
                    futures = [executor.submit(
                        _adaptive_component_count,
                        cells, pairs, order, barrier, population, scale,
                    ) for scale in scales]
                    counts = [future.result() for future in futures]
                    found = False
                    for scale, candidate_count in zip(scales, counts):
                        tested += 1
                        if candidate_count <= target:
                            high = scale
                            found = True
                            break
                        low = scale
                    if found:
                        break
                    high = scales[-1]
                else:
                    high *= 2.0

                # Three monotone checkpoints produce exactly the same bounds
                # as two ordinary bisection levels, but are independent.
                paired_steps = steps // 2
                for _ in range(paired_steps):
                    middle = 0.5 * (low + high)
                    scales = (
                        0.5 * (low + middle),
                        middle,
                        0.5 * (middle + high),
                    )
                    futures = [executor.submit(
                        _adaptive_component_count,
                        cells, pairs, order, barrier, population, scale,
                    ) for scale in scales]
                    counts = [future.result() for future in futures]
                    if counts[0] <= target:
                        high = scales[0]
                    elif counts[1] <= target:
                        low, high = scales[0], scales[1]
                    elif counts[2] <= target:
                        low, high = scales[1], scales[2]
                    else:
                        low = scales[2]
                if steps % 2:
                    middle = 0.5 * (low + high)
                    count = _adaptive_component_count(
                        cells, pairs, order, barrier, population, middle)
                    if count > target:
                        low = middle
                    else:
                        high = middle
        else:
            # Small graphs are cheaper than thread-pool dispatch.
            for _ in range(1, 32):
                high *= 2.0
                count = _adaptive_component_count(
                    cells, pairs, order, barrier, population, high)
                if count <= target:
                    break
                low = high
            for _ in range(steps):
                middle = 0.5 * (low + high)
                count = _adaptive_component_count(
                    cells, pairs, order, barrier, population, middle)
                if count > target:
                    low = middle
                else:
                    high = middle
    else:
        # The first scale already brackets the target.  Preserve the original
        # six-bit calibration in the uncommon zero-to-first-scale interval.
        for _ in range(steps):
            middle = 0.5 * (low + high)
            count = _adaptive_component_count(
                cells, pairs, order, barrier, population, middle)
            if count > target:
                low = middle
            else:
                high = middle
    # A scale threshold can activate many independent edges at once. Stop the
    # same ordered pass at the literal requested budget so the A/B never wins
    # merely by spending fewer IDs than the reconstruction partition.
    final_roots, _ = _adaptive_roots(
        cells, pairs, order, barrier, population, high, target)
    labels, count = _labels_from_roots(leaf_labels, final_roots)
    return labels, count, high


def labels_at_ratio(compound: dict, ratio: float) -> np.ndarray:
    """Recut an existing compound hierarchy without rebuilding its graph."""
    atoms = int(compound["atom_count"])
    target = max(int(round(atoms * max(float(ratio), 0.0))), 1)
    labels, _, _ = _adaptive_labels_at_count(
        np.asarray(compound["leaf_labels"], dtype=np.int32),
        int(compound["leaf_count"]),
        np.asarray(compound["graph_pairs"], dtype=np.int32),
        np.asarray(compound["graph_order"], dtype=np.int64),
        np.asarray(compound["graph_barrier"], dtype=np.float64),
        np.asarray(compound["leaf_population"], dtype=np.float64),
        target,
    )
    return labels


def build_compound_segment_quotient(
    texture_labels: np.ndarray,
    target_lab: np.ndarray,
    reconstruction_lab: np.ndarray,
    *,
    boundary_confidence: np.ndarray | None = None,
    paired_depth: int = 2,
    target_ratio: float = 1.0,
    connected_leaves: bool = True,
) -> dict:
    """Build same-budget compound IDs without changing any fitted atom."""
    started = time.perf_counter()
    atoms = np.ascontiguousarray(texture_labels, dtype=np.int32)
    target = np.ascontiguousarray(target_lab, dtype=np.float64)
    reconstruction = np.asarray(reconstruction_lab, dtype=np.float64)
    boundary = np.ascontiguousarray(
        np.zeros(atoms.shape, dtype=np.float64)
        if boundary_confidence is None
        else np.clip(boundary_confidence, 0.0, 1.0),
        dtype=np.float64,
    )
    phase_started = time.perf_counter()
    flat = atoms.ravel()
    atom_count = int(np.max(flat)) + 1
    samples = target.reshape(-1, 3)
    count, atom_sum, atom_second = _atom_sufficient_statistics(
        atoms, target, atom_count)
    mean = atom_sum / np.maximum(count[:, None], 1.0)
    covariance = (
        atom_second / np.maximum(count[:, None, None], 1.0)
        - mean[:, :, None] * mean[:, None, :]
    )
    _, eigenvectors = np.linalg.eigh(covariance)
    amplitude_axis = eigenvectors[:, :, -1]
    projection = np.einsum(
        "pc,pc->p",
        samples - mean[flat],
        amplitude_axis[flat],
        optimize=True,
    )
    amplitude_moments_ms = 1000.0 * (
        time.perf_counter() - phase_started)

    phase_started = time.perf_counter()
    amplitude_group = flat.copy()
    for _ in range(max(int(paired_depth), 0)):
        groups = int(np.max(amplitude_group)) + 1
        group_count = np.bincount(
            amplitude_group, minlength=groups).astype(np.float64)
        threshold = np.bincount(
            amplitude_group,
            weights=projection,
            minlength=groups,
        ) / np.maximum(group_count, 1.0)
        amplitude_group = (
            2 * amplitude_group
            + (projection >= threshold[amplitude_group]).astype(np.int32)
        )
    keyed = amplitude_group.reshape(atoms.shape)
    if bool(connected_leaves):
        leaf_labels, leaf_count = _connected_equal_components(keyed)
    else:
        leaf_labels, leaf_count = _compact_keys(keyed)
    leaf_construction_ms = 1000.0 * (
        time.perf_counter() - phase_started)

    phase_started = time.perf_counter()
    (
        pairs,
        interface_sum,
        interface_count,
        interface_boundary,
        internal_sum,
        internal_count,
        leaf_population,
        leaf_sum,
        leaf_square,
    ) = _leaf_interface_statistics(
        leaf_labels, target, boundary, leaf_count)
    region_variance = np.maximum(
        leaf_square
        - np.sum(leaf_sum * leaf_sum, axis=1)
        / np.maximum(leaf_population, 1.0),
        0.0,
    ) / np.maximum(leaf_population - 1.0, 1.0)
    continuation = np.where(
        internal_count > 0,
        internal_sum / np.maximum(internal_count, 1),
        region_variance,
    )
    noise = float(np.median(np.sum(
        (target - reconstruction) ** 2, axis=2)))
    cross = interface_sum / np.maximum(interface_count, 1)
    permission = (
        0.5 * (
            continuation[pairs[:, 0]]
            + continuation[pairs[:, 1]]
        )
        + noise
    )
    normalized_interface_barrier = cross / np.maximum(
        cross + permission, 1e-30)
    # The paired leaves have already materialized the two sides of a local
    # discontinuity.  Their compound merge therefore compares the Gaussian
    # amplitude prototypes carried by adjacent sides.  Using the normalized
    # local jump directly creates a single-linkage chain through antialias
    # ramps (the page background then absorbs a glyph).  Prototype distance,
    # combined with the size-conditioned graph pass below, keeps that global
    # region evidence without imposing one reconstruction model on the group.
    leaf_mean = leaf_sum / np.maximum(leaf_population[:, None], 1.0)
    amplitude_distance = np.sum(
        (leaf_mean[pairs[:, 0]] - leaf_mean[pairs[:, 1]]) ** 2,
        axis=1,
    )
    positive_amplitude = amplitude_distance[amplitude_distance > 0.0]
    amplitude_scale = max(
        float(np.median(positive_amplitude))
        if positive_amplitude.size else 0.0,
        1e-30,
    )
    amplitude_barrier = amplitude_distance / (
        amplitude_distance + amplitude_scale)
    boundary_barrier = np.clip(
        interface_boundary / np.maximum(interface_count, 1), 0.0, 1.0)
    # Keep the Gaussian distance in its native squared-Lab scale.  Normalizing
    # it to [0, 1] compresses the gap between a real material change and the
    # several antialias levels inside one glyph, reintroducing the very
    # single-linkage leak this adaptive quotient is intended to prevent.
    boundary_hazard = -np.log1p(-np.minimum(
        boundary_barrier, 1.0 - 1e-12))
    barrier = amplitude_distance + amplitude_scale * boundary_hazard
    graph_construction_ms = 1000.0 * (
        time.perf_counter() - phase_started)
    phase_started = time.perf_counter()
    order = np.argsort(barrier, kind="stable")
    graph_sort_ms = 1000.0 * (time.perf_counter() - phase_started)
    requested = max(
        int(round(atom_count * max(float(target_ratio), 0.0))), 1)
    phase_started = time.perf_counter()
    labels, compound_count, adaptive_scale = _adaptive_labels_at_count(
        leaf_labels,
        leaf_count,
        pairs,
        np.asarray(order, dtype=np.int64),
        np.ascontiguousarray(barrier),
        leaf_population,
        requested,
    )
    adaptive_merge_ms = 1000.0 * (
        time.perf_counter() - phase_started)
    return {
        "enabled": True,
        "labels": labels,
        "leaf_labels": leaf_labels,
        "atom_count": atom_count,
        "leaf_count": int(leaf_count),
        "compound_count": int(compound_count),
        "requested_count": int(requested),
        "paired_depth": int(paired_depth),
        "connected_leaves": bool(connected_leaves),
        "graph_pairs": pairs,
        "graph_order": np.asarray(order, dtype=np.int64),
        "graph_barrier": barrier,
        "normalized_interface_barrier": normalized_interface_barrier,
        "amplitude_barrier": amplitude_barrier,
        "boundary_barrier": boundary_barrier,
        "boundary_hazard": boundary_hazard,
        "amplitude_scale": amplitude_scale,
        "leaf_population": leaf_population,
        "graph_edges": int(len(pairs)),
        "adaptive_scale": float(adaptive_scale),
        "noise_floor": noise,
        "amplitude_moments_ms": amplitude_moments_ms,
        "leaf_construction_ms": leaf_construction_ms,
        "graph_construction_ms": graph_construction_ms,
        "graph_sort_ms": graph_sort_ms,
        "adaptive_merge_ms": adaptive_merge_ms,
        "milliseconds": 1000.0 * (time.perf_counter() - started),
        "reconstruction_changed": False,
    }
