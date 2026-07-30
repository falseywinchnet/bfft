"""Closed-contour persistence on an embedded interface graph.

The object-support experiment measures good *local* cell interfaces, but a
local interface value cannot say whether that interface is part of an object
contour or merely an open material seam.  The planar embedding supplies the
missing invariant.

For an embedded arc ``e`` with local support ``w(e)``, define

    c(e) = max_{cycles C containing e} min_{f in C} w(f).

``c(e)`` is the strongest bottleneck level at which ``e`` belongs to a closed
contour.  Open bridges have value zero.  Image-frame vertices are identified,
so contours ending on the frame are cycles in relative homology rather than
special cases.

All values are obtained from one maximum spanning forest.  Non-tree arcs close
fundamental cycles; a descending disjoint-set path pass assigns every tree arc
its strongest replacement.  There is no relaxation, convergence loop, or
per-object search.
"""

from __future__ import annotations

import numpy as np


def _frame_vertex(vertex: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return the vertices lying on the rectangular image boundary."""

    height, width = shape
    stride = width + 1
    value = np.asarray(vertex, dtype=np.int64)
    x = value % stride
    y = value // stride
    return (x == 0) | (x == width) | (y == 0) | (y == height)


def _arc_endpoints(
    topology: dict,
    *,
    collapse_frame: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map two-ended embedded arcs to a compact undirected multigraph."""

    arc = topology["arc"]
    count = int(arc["count"])
    offset = np.asarray(arc["endpoint_offset"], dtype=np.int64)
    endpoint = np.asarray(arc["endpoint_vertex"], dtype=np.int64)
    endpoint_count = np.diff(offset)
    ordinary = np.flatnonzero(endpoint_count == 2).astype(np.int32)
    first = endpoint[offset[ordinary]].copy()
    second = endpoint[offset[ordinary] + 1].copy()

    if collapse_frame and ordinary.size:
        frame = np.concatenate((
            first[_frame_vertex(first, topology["shape"])],
            second[_frame_vertex(second, topology["shape"])],
        ))
        if frame.size:
            frame_vertex = int(np.min(frame))
            first[_frame_vertex(first, topology["shape"])] = frame_vertex
            second[_frame_vertex(second, topology["shape"])] = frame_vertex

    if ordinary.size:
        vertices, inverse = np.unique(
            np.concatenate((first, second)),
            return_inverse=True,
        )
        del vertices
        first_compact = inverse[:len(first)].astype(np.int32, copy=False)
        second_compact = inverse[len(first):].astype(np.int32, copy=False)
    else:
        first_compact = np.empty(0, dtype=np.int32)
        second_compact = np.empty(0, dtype=np.int32)

    closed = np.flatnonzero(endpoint_count == 0).astype(np.int32)
    represented = np.zeros(count, dtype=bool)
    represented[ordinary] = True
    represented[closed] = True
    if not np.all(represented):
        raise ValueError("embedded arcs must have zero or two endpoints")
    return ordinary, first_compact, second_compact


def maximum_bottleneck_cycle_support(
    topology: dict,
    arc_support: np.ndarray,
    *,
    collapse_frame: bool = True,
) -> np.ndarray:
    """Return the maximum bottleneck closed-contour value of every arc.

    The result never exceeds the supplied local support.  An arc not belonging
    to any absolute or frame-relative cycle receives zero.
    """

    arc = topology["arc"]
    count = int(arc["count"])
    weight = np.clip(
        np.asarray(arc_support, dtype=np.float64),
        0.0,
        1.0,
    )
    if weight.shape != (count,):
        raise ValueError("arc support must contain one value per embedded arc")

    ordinary, first, second = _arc_endpoints(
        topology,
        collapse_frame=collapse_frame,
    )
    result = np.zeros(count, dtype=np.float64)
    endpoint_offset = np.asarray(
        arc["endpoint_offset"], dtype=np.int64)
    closed = np.flatnonzero(np.diff(endpoint_offset) == 0)
    result[closed] = weight[closed]
    if ordinary.size == 0:
        return result

    node_count = int(max(
        first.max(initial=-1),
        second.max(initial=-1),
    )) + 1
    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            previous = int(parent[node])
            parent[node] = root
            node = previous
        return root

    order = np.argsort(-weight[ordinary], kind="stable")
    tree_mask = np.zeros(len(ordinary), dtype=bool)
    non_tree: list[int] = []
    for local_index in order:
        a = int(first[local_index])
        b = int(second[local_index])
        if a == b:
            non_tree.append(int(local_index))
            result[ordinary[local_index]] = weight[ordinary[local_index]]
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            non_tree.append(int(local_index))
            result[ordinary[local_index]] = weight[ordinary[local_index]]
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        tree_mask[local_index] = True

    tree_local = np.flatnonzero(tree_mask)
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(node_count)
    ]
    for local_index in tree_local:
        a = int(first[local_index])
        b = int(second[local_index])
        adjacency[a].append((b, int(local_index)))
        adjacency[b].append((a, int(local_index)))

    tree_parent = np.full(node_count, -1, dtype=np.int32)
    parent_edge = np.full(node_count, -1, dtype=np.int32)
    depth = np.zeros(node_count, dtype=np.int32)
    component = np.full(node_count, -1, dtype=np.int32)
    roots: list[int] = []
    for root in range(node_count):
        if component[root] >= 0:
            continue
        roots.append(root)
        component[root] = root
        stack = [root]
        while stack:
            node = stack.pop()
            for target, local_index in adjacency[node]:
                if target == tree_parent[node]:
                    continue
                tree_parent[target] = node
                parent_edge[target] = local_index
                depth[target] = depth[node] + 1
                component[target] = root
                stack.append(target)

    levels = max(1, int(node_count).bit_length())
    jump = np.full((levels, node_count), -1, dtype=np.int32)
    jump[0] = tree_parent
    for level in range(1, levels):
        previous = jump[level - 1]
        valid = previous >= 0
        jump[level, valid] = previous[previous[valid]]

    def lca(first_node: int, second_node: int) -> int:
        a, b = first_node, second_node
        if depth[a] < depth[b]:
            a, b = b, a
        difference = int(depth[a] - depth[b])
        bit = 0
        while difference:
            if difference & 1:
                a = int(jump[bit, a])
            difference >>= 1
            bit += 1
        if a == b:
            return a
        for level in range(levels - 1, -1, -1):
            ja, jb = int(jump[level, a]), int(jump[level, b])
            if ja != jb:
                a, b = ja, jb
        return int(tree_parent[a])

    # Descending path painting: the first non-tree replacement to reach a
    # tree edge is its strongest possible replacement.
    skip = np.arange(node_count, dtype=np.int32)

    def skip_find(node: int) -> int:
        root = node
        while skip[root] != root:
            root = int(skip[root])
        while skip[node] != node:
            previous = int(skip[node])
            skip[node] = root
            node = previous
        return root

    def paint_to_ancestor(node: int, ancestor: int, value: float) -> None:
        current = skip_find(node)
        while depth[current] > depth[ancestor]:
            local_edge = int(parent_edge[current])
            result[ordinary[local_edge]] = value
            parent_node = int(tree_parent[current])
            skip[current] = skip_find(parent_node)
            current = skip_find(current)

    non_tree.sort(
        key=lambda index: float(weight[ordinary[index]]),
        reverse=True,
    )
    for local_index in non_tree:
        a = int(first[local_index])
        b = int(second[local_index])
        if a == b or component[a] != component[b]:
            continue
        ancestor = lca(a, b)
        value = float(weight[ordinary[local_index]])
        paint_to_ancestor(a, ancestor, value)
        paint_to_ancestor(b, ancestor, value)

    return np.minimum(result, weight)


def intrinsic_boundary_alignment(
    graph: dict,
    intrinsic_owner: np.ndarray,
) -> np.ndarray:
    """Measure intrinsic-support boundary coincidence on each embedded arc."""

    labels = np.asarray(graph["labels"], dtype=np.int32)
    owner = np.asarray(intrinsic_owner, dtype=np.int32)
    if owner.shape != labels.shape:
        raise ValueError(
            "intrinsic support and canonical graph must share a shape")
    crossing_horizontal = labels[:, :-1] != labels[:, 1:]
    crossing_vertical = labels[:-1] != labels[1:]
    sample = np.concatenate((
        (owner[:, :-1] != owner[:, 1:])[crossing_horizontal],
        (owner[:-1] != owner[1:])[crossing_vertical],
    )).astype(np.float64)
    sample_arc = np.asarray(
        graph["interface_topology"]["edgel"]["arc"],
        dtype=np.int32,
    )
    if len(sample) != len(sample_arc):
        raise RuntimeError(
            "intrinsic boundary samples and embedded edgels disagree")
    arc_count = int(graph["interface_topology"]["arc"]["count"])
    total = np.bincount(
        sample_arc,
        weights=sample,
        minlength=arc_count,
    )
    length = np.bincount(sample_arc, minlength=arc_count)
    return total / np.maximum(length, 1)

