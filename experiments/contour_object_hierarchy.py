"""Rule-free object membership from a closed-contour ultrametric.

The leaf partition is supplied by ``transport_object_support``.  This module
does not inspect colour names, junction types, containment cases, or semantic
classes.  It quotients the closed-contour barrier onto the leaf adjacency
graph and computes its minimum-barrier spanning tree.

For leaves ``i`` and ``j`` the cophenetic distance is

    d(i, j) = min_P max_{e in P} b(e),

the lowest contour waterline at which they become connected.  This is an
ultrametric.  Consequently the complete part-to-object hierarchy, every hard
waterline cut, and a soft co-membership display all come from one Kruskal
pass.  No criteria engine or convergence iteration is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContourHierarchyConfig:
    waterline: float = 0.50
    soft_temperature: float = 0.12


def _stable_colours(count: int) -> np.ndarray:
    value = np.arange(count, dtype=np.uint32)
    value = value * np.uint32(747796405) + np.uint32(2891336453)
    value = (
        ((value >> ((value >> 28) + 4)) ^ value)
        * np.uint32(277803737)
    )
    value = (value >> 22) ^ value
    return 0.10 + 0.88 * np.column_stack((
        value & 255,
        (value >> 8) & 255,
        (value >> 16) & 255,
    )).astype(np.float64) / 255.0


def quotient_part_interfaces(objects: dict) -> dict[str, np.ndarray | int]:
    """Aggregate canonical arcs into one boundary per adjacent leaf pair."""

    graph = objects["graph"]
    edge = graph["edge"]
    part = np.asarray(objects["object_id_per_cell"], dtype=np.int32)
    first_part = part[np.asarray(edge["first"], dtype=np.int32)]
    second_part = part[np.asarray(edge["second"], dtype=np.int32)]
    crossing = first_part != second_part
    first_part = first_part[crossing]
    second_part = second_part[crossing]
    count = int(part.max(initial=-1)) + 1
    low = np.minimum(first_part, second_part)
    high = np.maximum(first_part, second_part)
    key = low.astype(np.int64) * max(count, 1) + high
    unique, inverse = np.unique(key, return_inverse=True)
    length = np.asarray(edge["length"], dtype=np.float64)[crossing]
    local_barrier = np.clip(
        np.asarray(objects["evidence"]["barrier"], dtype=np.float64)[
            crossing
        ],
        0.0,
        1.0 - 1e-12,
    )
    total_length = np.bincount(inverse, weights=length)
    # Independent contour failure probabilities compose multiplicatively.
    # The normalized log-survival is resolution independent and is stricter
    # than an arithmetic average when a real boundary contains strong arcs.
    log_survival = np.bincount(
        inverse,
        weights=length * np.log1p(-local_barrier),
    ) / np.maximum(total_length, 1e-12)
    barrier = 1.0 - np.exp(log_survival)
    return {
        "count": count,
        "first": (unique // max(count, 1)).astype(np.int32),
        "second": (unique % max(count, 1)).astype(np.int32),
        "length": total_length,
        "barrier": np.clip(barrier, 0.0, 1.0),
    }


def minimum_barrier_tree(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
    barrier: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the minimum spanning forest of a barrier graph."""

    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            previous = int(parent[node])
            parent[node] = root
            node = previous
        return root

    tree_first: list[int] = []
    tree_second: list[int] = []
    tree_barrier: list[float] = []
    for edge_index in np.argsort(barrier, kind="stable"):
        a = int(first[edge_index])
        b = int(second[edge_index])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        tree_first.append(a)
        tree_second.append(b)
        tree_barrier.append(float(barrier[edge_index]))
    return {
        "first": np.asarray(tree_first, dtype=np.int32),
        "second": np.asarray(tree_second, dtype=np.int32),
        "barrier": np.asarray(tree_barrier, dtype=np.float64),
    }


def cophenetic_ultrametric(
    count: int,
    tree: dict[str, np.ndarray],
) -> np.ndarray:
    """Return minimax merge altitude between every pair of leaves."""

    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(count)
    ]
    for a, b, value in zip(
        tree["first"], tree["second"], tree["barrier"]
    ):
        adjacency[int(a)].append((int(b), float(value)))
        adjacency[int(b)].append((int(a), float(value)))
    distance = np.ones((count, count), dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    for source in range(count):
        seen = np.zeros(count, dtype=bool)
        seen[source] = True
        stack = [(source, 0.0)]
        while stack:
            node, altitude = stack.pop()
            distance[source, node] = altitude
            for target, edge_barrier in adjacency[node]:
                if seen[target]:
                    continue
                seen[target] = True
                stack.append((target, max(altitude, edge_barrier)))
    return distance


def labels_at_waterline(
    count: int,
    tree: dict[str, np.ndarray],
    waterline: float,
) -> np.ndarray:
    """Cut an ultrametric hierarchy at one displayed contour waterline."""

    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            previous = int(parent[node])
            parent[node] = root
            node = previous
        return root

    level = float(np.clip(waterline, 0.0, 1.0))
    for a, b, barrier in zip(
        tree["first"], tree["second"], tree["barrier"]
    ):
        if barrier >= level:
            continue
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    root = np.asarray([find(node) for node in range(count)], dtype=np.int32)
    _, label = np.unique(root, return_inverse=True)
    return label.astype(np.int32)


def infer_contour_object_hierarchy(
    objects: dict,
    config: ContourHierarchyConfig = ContourHierarchyConfig(),
) -> dict:
    """Build hard and soft object membership from the contour ultrametric."""

    quotient = quotient_part_interfaces(objects)
    count = int(quotient["count"])
    tree = minimum_barrier_tree(
        count,
        quotient["first"],
        quotient["second"],
        quotient["barrier"],
    )
    ultrametric = cophenetic_ultrametric(count, tree)
    parent = labels_at_waterline(
        count, tree, config.waterline)
    parent_count = int(parent.max(initial=-1)) + 1
    leaf_area = np.bincount(
        np.asarray(objects["object_id_per_cell"], dtype=np.int32),
        weights=np.asarray(objects["graph"]["area"], dtype=np.float64),
        minlength=count,
    )
    temperature = max(float(config.soft_temperature), 1e-6)
    kernel = np.exp(-ultrametric / temperature)
    kernel *= np.sqrt(np.maximum(leaf_area, 1.0))[None, :]
    kernel /= np.maximum(np.sum(kernel, axis=1, keepdims=True), 1e-12)
    leaf_colours = _stable_colours(count)
    soft_leaf_ids = np.clip(kernel @ leaf_colours, 0.0, 1.0)
    parent_colours = _stable_colours(parent_count)

    leaf_labels = np.asarray(objects["object_labels"], dtype=np.int32)
    parent_labels = parent[leaf_labels]
    return {
        "config": config,
        "quotient": quotient,
        "tree": tree,
        "ultrametric": ultrametric,
        "leaf_count": count,
        "parent_count": parent_count,
        "parent_id_per_leaf": parent,
        "parent_labels": parent_labels,
        "parent_ids": parent_colours[parent_labels],
        "soft_leaf_ids": soft_leaf_ids,
        "soft_ids": soft_leaf_ids[leaf_labels],
        "leaf_merge_altitude": np.min(
            np.where(
                np.eye(count, dtype=bool),
                np.inf,
                ultrametric,
            ),
            axis=1,
        )[leaf_labels] if count > 1 else np.zeros_like(
            leaf_labels, dtype=np.float64),
    }

