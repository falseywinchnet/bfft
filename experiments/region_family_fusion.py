"""Speculative third-order fusion over immutable compound segment IDs.

This experiment combines multiscale palette-family evidence with graph
relationships.  Subpixel-thickness regions are transparent to relationship
discovery but remain represented in the output.  The reconstruction and the
compound segmentation are never modified.
"""

from __future__ import annotations

import time

import numpy as np

from experiments.region_posterization import multiscale_region_affinity

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _compile(function):
    return function if njit is None else njit(cache=True)(function)


def _boundary_graph(labels: np.ndarray, regions: int):
    perimeter = np.zeros(regions, dtype=np.int64)
    perimeter += np.bincount(labels[0], minlength=regions)
    perimeter += np.bincount(labels[-1], minlength=regions)
    perimeter += np.bincount(labels[:, 0], minlength=regions)
    perimeter += np.bincount(labels[:, -1], minlength=regions)
    keys = []
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1], labels[1:]),
    ):
        crossing = first != second
        a = first[crossing].astype(np.int64)
        b = second[crossing].astype(np.int64)
        perimeter += np.bincount(a, minlength=regions)
        perimeter += np.bincount(b, minlength=regions)
        low = np.minimum(a, b)
        high = np.maximum(a, b)
        keys.append(low * regions + high)
    key, interface = np.unique(
        np.concatenate(keys), return_counts=True)
    pairs = np.stack((key // regions, key % regions), axis=1).astype(np.int32)
    return pairs, interface.astype(np.int64), perimeter


def _find(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = int(parent[root])
    while parent[item] != item:
        next_item = int(parent[item])
        parent[item] = root
        item = next_item
    return root


def _union(parent: np.ndarray, size: np.ndarray, first: int, second: int):
    first = _find(parent, first)
    second = _find(parent, second)
    if first == second:
        return
    if size[first] < size[second]:
        first, second = second, first
    parent[second] = first
    size[first] += size[second]


@_compile
def _thin_components(pairs, thin, regions):
    parent = np.arange(regions, dtype=np.int32)
    size = np.ones(regions, dtype=np.int32)
    for edge in range(len(pairs)):
        first = int(pairs[edge, 0])
        second = int(pairs[edge, 1])
        if not thin[first] or not thin[second]:
            continue
        root_first = first
        while parent[root_first] != root_first:
            root_first = parent[root_first]
        root_second = second
        while parent[root_second] != root_second:
            root_second = parent[root_second]
        if root_first == root_second:
            continue
        if size[root_first] < size[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
    for region in range(regions):
        root = region
        while parent[root] != root:
            root = parent[root]
        parent[region] = root
    return parent


@_compile
def _component_pair_records(
    neighbor, contact, starts, ends, regions, count,
):
    keys = np.empty(count, dtype=np.int64)
    first_contact = np.empty(count, dtype=np.float64)
    second_contact = np.empty(count, dtype=np.float64)
    cursor = 0
    for group in range(len(starts)):
        start = int(starts[group])
        end = int(ends[group])
        members = end - start
        if members < 2 or members > 64:
            continue
        for left_index in range(start, end - 1):
            for right_index in range(left_index + 1, end):
                left = int(neighbor[left_index])
                right = int(neighbor[right_index])
                if left < right:
                    keys[cursor] = left * regions + right
                    first_contact[cursor] = contact[left_index]
                    second_contact[cursor] = contact[right_index]
                else:
                    keys[cursor] = right * regions + left
                    first_contact[cursor] = contact[right_index]
                    second_contact[cursor] = contact[left_index]
                cursor += 1
    return keys, first_contact, second_contact


@_compile
def _group_pair_keys(neighbor, starts, ends, regions, count):
    keys = np.empty(count, dtype=np.int64)
    cursor = 0
    for group in range(len(starts)):
        start = int(starts[group])
        end = int(ends[group])
        members = end - start
        if members < 2 or members > 64:
            continue
        for left_index in range(start, end - 1):
            for right_index in range(left_index + 1, end):
                left = int(neighbor[left_index])
                right = int(neighbor[right_index])
                if left < right:
                    keys[cursor] = left * regions + right
                else:
                    keys[cursor] = right * regions + left
                cursor += 1
    return keys


def _otsu_threshold(value: np.ndarray, bins: int = 256) -> float:
    sample = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    if sample.size == 0:
        return 1.0
    histogram, edges = np.histogram(sample, bins=bins, range=(0.0, 1.0))
    probability = histogram.astype(np.float64)
    probability /= max(float(np.sum(probability)), 1.0)
    center = 0.5 * (edges[:-1] + edges[1:])
    weight_left = np.cumsum(probability)
    mean_left_sum = np.cumsum(probability * center)
    whole_mean = mean_left_sum[-1]
    denominator = weight_left * (1.0 - weight_left)
    between = np.full_like(denominator, -1.0)
    valid = denominator > 0.0
    between[valid] = (
        np.square(whole_mean * weight_left[valid] - mean_left_sum[valid])
        / denominator[valid]
    )
    return float(edges[int(np.argmax(between)) + 1])


def _effective_graph(
    pairs: np.ndarray,
    interface: np.ndarray,
    perimeter: np.ndarray,
    thin: np.ndarray,
    regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridge substantive regions across connected subpixel supports."""
    first = pairs[:, 0]
    second = pairs[:, 1]
    direct = ~thin[first] & ~thin[second]
    direct_key = (
        first[direct].astype(np.int64) * regions + second[direct])
    direct_contact = interface[direct].astype(np.float64)

    roots = _thin_components(pairs, thin, regions)
    crossing = thin[first] ^ thin[second]
    thin_region = np.where(thin[first[crossing]],
                           first[crossing], second[crossing])
    neighbor = np.where(thin[first[crossing]],
                        second[crossing], first[crossing])
    contact_key = (
        roots[thin_region].astype(np.int64) * regions + neighbor)
    if len(contact_key):
        order = np.argsort(contact_key, kind="stable")
        contact_key = contact_key[order]
        contact_length = interface[crossing][order].astype(np.float64)
        starts = np.flatnonzero(np.r_[True, contact_key[1:] != contact_key[:-1]])
        contact_key = contact_key[starts]
        contact_length = np.add.reduceat(contact_length, starts)
        component = contact_key // regions
        neighbor = (contact_key % regions).astype(np.int32)
        group_starts = np.flatnonzero(
            np.r_[True, component[1:] != component[:-1]])
        group_ends = np.r_[group_starts[1:], len(component)]
        group_size = group_ends - group_starts
        pair_count = int(np.sum(
            np.where(group_size <= 64, group_size * (group_size - 1) // 2, 0)))
        bridge_key, bridge_first, bridge_second = _component_pair_records(
            neighbor, contact_length, group_starts, group_ends,
            regions, pair_count)
    else:
        bridge_key = np.empty(0, dtype=np.int64)
        bridge_first = np.empty(0, dtype=np.float64)
        bridge_second = np.empty(0, dtype=np.float64)

    key = np.concatenate((direct_key, bridge_key))
    first_contact = np.concatenate((direct_contact, bridge_first))
    second_contact = np.concatenate((direct_contact, bridge_second))
    if not len(key):
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2))
    order = np.argsort(key, kind="stable")
    key = key[order]
    first_contact = first_contact[order]
    second_contact = second_contact[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    key = key[starts]
    first_contact = np.add.reduceat(first_contact, starts)
    second_contact = np.add.reduceat(second_contact, starts)
    result_pairs = np.column_stack((key // regions, key % regions)).astype(
        np.int32)
    fraction = np.column_stack((
        first_contact / np.maximum(perimeter[result_pairs[:, 0]], 1),
        second_contact / np.maximum(perimeter[result_pairs[:, 1]], 1),
    ))
    return result_pairs, fraction


def _mediated_candidates(
    effective_pairs: np.ndarray,
    coarse_family: np.ndarray,
    regions: int,
) -> np.ndarray:
    if not len(effective_pairs):
        return np.empty((0, 2), dtype=np.int32)
    source = np.concatenate((effective_pairs[:, 0], effective_pairs[:, 1]))
    neighbor = np.concatenate((effective_pairs[:, 1], effective_pairs[:, 0]))
    family_count = int(np.max(coarse_family, initial=-1)) + 1
    group_key = source.astype(np.int64) * family_count + coarse_family[neighbor]
    order = np.argsort(group_key, kind="stable")
    group_key = group_key[order]
    neighbor = neighbor[order]
    starts = np.flatnonzero(np.r_[True, group_key[1:] != group_key[:-1]])
    ends = np.r_[starts[1:], len(group_key)]
    size = ends - starts
    pair_count = int(np.sum(
        np.where(size <= 64, size * (size - 1) // 2, 0)))
    mediated = _group_pair_keys(
        neighbor, starts, ends, regions, pair_count)
    direct = (
        effective_pairs[:, 0].astype(np.int64) * regions
        + effective_pairs[:, 1])
    key = np.unique(np.concatenate((direct, mediated)))
    return np.column_stack((key // regions, key % regions)).astype(np.int32)


def _labels_from_parent(labels: np.ndarray, parent: np.ndarray):
    root_label = np.full(len(parent), -1, dtype=np.int32)
    region_label = np.empty(len(parent), dtype=np.int32)
    count = 0
    for region in range(len(parent)):
        root = _find(parent, region)
        if root_label[root] < 0:
            root_label[root] = count
            count += 1
        region_label[region] = root_label[root]
    return region_label[labels], region_label, count


def build_region_family_fusion(posterization: dict) -> dict:
    """Build color siblings and terminal sibling-to-host hyperedges."""
    started = time.perf_counter()
    labels = np.asarray(posterization["labels"], dtype=np.int32)
    regions = int(posterization["region_count"])
    population = np.asarray(
        posterization["region_population"], dtype=np.float64)
    pairs, interface, perimeter = _boundary_graph(labels, regions)
    thickness = 2.0 * population / np.maximum(perimeter, 1)
    thin = thickness <= 1.0
    effective_pairs, relation_fraction = _effective_graph(
        pairs, interface, perimeter, thin, regions)
    effective_key = (
        effective_pairs[:, 0].astype(np.int64) * regions
        + effective_pairs[:, 1])
    coarse_level = posterization["levels"][min(
        3, len(posterization["levels"]) - 1)]
    coarse_family = coarse_level["region_dominant_family"]
    candidates = _mediated_candidates(
        effective_pairs, coarse_family, regions)
    affinity = multiscale_region_affinity(posterization, candidates)
    scale_similarity = (
        2.0 * np.sqrt(
            population[candidates[:, 0]] * population[candidates[:, 1]])
        / np.maximum(
            population[candidates[:, 0]] + population[candidates[:, 1]],
            1.0,
        )
    )
    score = affinity * scale_similarity
    threshold = _otsu_threshold(score)

    best = np.zeros(regions, dtype=np.float64)
    for edge, (first, second) in enumerate(candidates):
        value = score[edge]
        best[first] = max(best[first], value)
        best[second] = max(best[second], value)
    parent = np.arange(regions, dtype=np.int32)
    size = population.copy()
    sibling_edges = 0
    sibling_merges = []
    for edge, (first, second) in enumerate(candidates):
        value = score[edge]
        if value < threshold or value <= 0.0:
            continue
        if value + 1e-12 < 0.90 * best[first]:
            continue
        if value + 1e-12 < 0.90 * best[second]:
            continue
        _union(parent, size, int(first), int(second))
        sibling_edges += 1
        sibling_merges.append((int(first), int(second), float(value)))

    # Index direct attachment fractions and effective neighborhoods.
    effective_adjacency = [set() for _ in range(regions)]
    for first, second in effective_pairs:
        effective_adjacency[int(first)].add(int(second))
        effective_adjacency[int(second)].add(int(first))
    exterior = np.zeros(regions, dtype=np.bool_)
    exterior[np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    )))] = True

    members_by_root: dict[int, list[int]] = {}
    for region in range(regions):
        members_by_root.setdefault(_find(parent, region), []).append(region)
    hyperedges = 0
    host_merges = []
    for members in members_by_root.values():
        if len(members) < 2:
            continue
        host_votes: dict[int, list[int]] = {}
        for member in members:
            for host in effective_adjacency[member]:
                if host not in members:
                    host_votes.setdefault(host, []).append(member)
        for host, attached in sorted(host_votes.items()):
            if len(attached) < 2:
                continue
            # Border-touching regions are the unbounded exterior components
            # of this planar partition. They may mediate sibling discovery,
            # but cannot be the geometric host of enclosed parts.
            if exterior[host]:
                continue
            if population[host] < sum(population[members]):
                continue
            terminal_votes = 0
            for member in attached:
                low, high = sorted((member, host))
                key = low * regions + high
                relation = int(np.searchsorted(effective_key, key))
                fraction = relation_fraction[relation][
                    0 if member == low else 1]
                if fraction < 0.65:
                    terminal_votes += 1
            if terminal_votes >= 2:
                _union(parent, size, members[0], host)
                hyperedges += 1
                host_merges.append({
                    "members": tuple(members),
                    "host": int(host),
                    "terminal_votes": int(terminal_votes),
                })
                break

    fused_labels, region_family, family_count = _labels_from_parent(
        labels, parent)
    return {
        "enabled": True,
        "labels": fused_labels,
        "region_family": region_family,
        "input_regions": regions,
        "family_count": family_count,
        "boundary_pairs": len(pairs),
        "effective_pairs": len(effective_pairs),
        "candidate_pairs": len(candidates),
        "thin_regions": int(np.count_nonzero(thin)),
        "affinity_threshold": threshold,
        "mean_scale_similarity": (
            float(np.mean(scale_similarity)) if len(scale_similarity) else 0.0),
        "sibling_edges": sibling_edges,
        "sibling_merges": sibling_merges,
        "host_hyperedges": hyperedges,
        "host_merges": host_merges,
        "milliseconds": 1000.0 * (time.perf_counter() - started),
    }
