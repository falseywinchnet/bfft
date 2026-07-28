#!/usr/bin/env python3
"""Embedded planar topology of a hard transport-cell representation.

The ordinary region-adjacency graph deliberately collapses every interface
between cell pair ``(i, j)`` into one averaged edge.  Object and part evidence
needs information that quotient destroys: separately connected arcs, arc
endpoints, cyclic junction neighbourhoods, and the actual planar location of
each interface element.

This module preserves that support in one linear-size structure.  It performs
no segmentation and makes no object decision.  It is the substrate on which
closure, good continuation, T-junction, common-surround, and signed transport
experiments can be falsified without returning to an all-pairs cell search.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - project runtime includes numba
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


@njit(cache=True)
def _union_incident_segments(
    ordered_endpoint: np.ndarray,
    endpoint_vertex: np.ndarray,
    endpoint_pair: np.ndarray,
    segment_count: int,
) -> np.ndarray:
    """Union same-pair segments meeting at the same lattice vertex."""
    parent = np.arange(segment_count, dtype=np.int32)
    size = np.ones(segment_count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            previous = parent[node]
            parent[node] = root
            node = previous
        return root

    for cursor in range(1, len(ordered_endpoint)):
        left_index = ordered_endpoint[cursor - 1]
        right_index = ordered_endpoint[cursor]
        if (
            endpoint_vertex[left_index] != endpoint_vertex[right_index]
            or endpoint_pair[left_index] != endpoint_pair[right_index]
        ):
            continue
        left = find(left_index % segment_count)
        right = find(right_index % segment_count)
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    for node in range(segment_count):
        parent[node] = find(node)
    return parent


def _segments(labels: np.ndarray) -> dict[str, np.ndarray]:
    label = np.asarray(labels, dtype=np.int32)
    if label.ndim != 2:
        raise ValueError("labels must be a two-dimensional integer image")
    height, width = label.shape
    stride = width + 1

    horizontal_crossing = label[:, :-1] != label[:, 1:]
    hy, hx = np.nonzero(horizontal_crossing)
    ha = label[hy, hx]
    hb = label[hy, hx + 1]
    # A left/right pixel crossing is a vertical dual-grid segment.
    hv0 = hy.astype(np.int64) * stride + hx + 1
    hv1 = (hy.astype(np.int64) + 1) * stride + hx + 1

    vertical_crossing = label[:-1, :] != label[1:, :]
    vy, vx = np.nonzero(vertical_crossing)
    va = label[vy, vx]
    vb = label[vy + 1, vx]
    # A top/bottom pixel crossing is a horizontal dual-grid segment.
    vv0 = (vy.astype(np.int64) + 1) * stride + vx
    vv1 = (vy.astype(np.int64) + 1) * stride + vx + 1

    cell_a = np.concatenate((ha, va)).astype(np.int32, copy=False)
    cell_b = np.concatenate((hb, vb)).astype(np.int32, copy=False)
    first = np.minimum(cell_a, cell_b)
    second = np.maximum(cell_a, cell_b)
    cell_count = int(label.max(initial=-1)) + 1
    pair = first.astype(np.int64) * max(cell_count, 1) + second
    return {
        "vertex_first": np.concatenate((hv0, vv0)),
        "vertex_second": np.concatenate((hv1, vv1)),
        "cell_first": first,
        "cell_second": second,
        "pair": pair,
        "orientation": np.concatenate((
            np.ones(len(hv0), dtype=np.int8),   # vertical
            np.zeros(len(vv0), dtype=np.int8), # horizontal
        )),
    }


def _incident_groups(
    vertex_first: np.ndarray,
    vertex_second: np.ndarray,
    arc_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """Unique (vertex, arc) incidences and their local segment degree."""
    segment_count = len(arc_id)
    vertex = np.concatenate((vertex_first, vertex_second))
    arc = np.concatenate((arc_id, arc_id))
    occurrence = np.arange(2 * segment_count, dtype=np.int64)
    order = np.lexsort((arc, vertex))
    ordered_vertex = vertex[order]
    ordered_arc = arc[order]
    start = np.r_[0, 1 + np.flatnonzero(
        (ordered_vertex[1:] != ordered_vertex[:-1])
        | (ordered_arc[1:] != ordered_arc[:-1])
    )]
    return {
        "vertex": ordered_vertex[start],
        "arc": ordered_arc[start],
        "degree": np.diff(np.r_[start, len(order)]).astype(np.int32),
        "first_occurrence": occurrence[order[start]],
    }


def build_embedded_interface_topology(labels: np.ndarray) -> dict:
    """Return edgels, separately connected arcs, endpoints, and junctions."""
    label = np.asarray(labels, dtype=np.int32)
    height, width = label.shape
    stride = width + 1
    segment = _segments(label)
    segment_count = len(segment["pair"])
    if segment_count == 0:
        return {
            "shape": label.shape,
            "edgel": {**segment, "arc": np.empty(0, dtype=np.int32)},
            "arc": {
                "count": 0,
                "cell_first": np.empty(0, dtype=np.int32),
                "cell_second": np.empty(0, dtype=np.int32),
                "length": np.empty(0, dtype=np.float64),
                "closed": np.empty(0, dtype=bool),
                "endpoint_offset": np.zeros(1, dtype=np.int64),
                "endpoint_vertex": np.empty(0, dtype=np.int64),
                "endpoint_tangent_x": np.empty(0, dtype=np.float64),
                "endpoint_tangent_y": np.empty(0, dtype=np.float64),
            },
            "junction": {
                "count": 0,
                "vertex": np.empty(0, dtype=np.int64),
                "x": np.empty(0, dtype=np.int32),
                "y": np.empty(0, dtype=np.int32),
                "arc_offset": np.zeros(1, dtype=np.int64),
                "arc": np.empty(0, dtype=np.int32),
            },
        }

    endpoint_vertex = np.concatenate((
        segment["vertex_first"],
        segment["vertex_second"],
    ))
    endpoint_pair = np.concatenate((segment["pair"], segment["pair"]))
    ordered_endpoint = np.lexsort((endpoint_pair, endpoint_vertex))
    root = _union_incident_segments(
        ordered_endpoint,
        endpoint_vertex,
        endpoint_pair,
        segment_count,
    )
    _, arc_id = np.unique(root, return_inverse=True)
    arc_id = arc_id.astype(np.int32)
    arc_count = int(arc_id.max(initial=-1)) + 1
    segment["arc"] = arc_id

    first_segment = np.full(arc_count, segment_count, dtype=np.int64)
    np.minimum.at(first_segment, arc_id, np.arange(segment_count))
    arc_length = np.bincount(
        arc_id, minlength=arc_count).astype(np.float64)

    incidence = _incident_groups(
        segment["vertex_first"],
        segment["vertex_second"],
        arc_id,
    )
    endpoint_mask = incidence["degree"] != 2
    endpoint_arc = incidence["arc"][endpoint_mask]
    endpoint_order = np.argsort(endpoint_arc, kind="stable")
    endpoint_arc = endpoint_arc[endpoint_order]
    endpoint_vertex_unique = incidence["vertex"][endpoint_mask][
        endpoint_order
    ]
    endpoint_occurrence = incidence["first_occurrence"][endpoint_mask][
        endpoint_order
    ]
    endpoint_count = np.bincount(
        endpoint_arc, minlength=arc_count).astype(np.int64)
    endpoint_offset = np.empty(arc_count + 1, dtype=np.int64)
    endpoint_offset[0] = 0
    np.cumsum(endpoint_count, out=endpoint_offset[1:])

    occurrence_is_second = endpoint_occurrence >= segment_count
    endpoint_segment = endpoint_occurrence % segment_count
    other_vertex = np.where(
        occurrence_is_second,
        segment["vertex_first"][endpoint_segment],
        segment["vertex_second"][endpoint_segment],
    )
    ex = endpoint_vertex_unique % stride
    ey = endpoint_vertex_unique // stride
    ox = other_vertex % stride
    oy = other_vertex // stride
    tangent_x = (ox - ex).astype(np.float64)
    tangent_y = (oy - ey).astype(np.float64)
    norm = np.maximum(np.hypot(tangent_x, tangent_y), 1e-12)
    tangent_x /= norm
    tangent_y /= norm

    # A junction is a lattice vertex incident on three or more distinct arcs.
    vertex_order = np.argsort(incidence["vertex"], kind="stable")
    ordered_vertex = incidence["vertex"][vertex_order]
    vertex_start = np.r_[0, 1 + np.flatnonzero(
        ordered_vertex[1:] != ordered_vertex[:-1]
    )]
    vertex_degree = np.diff(
        np.r_[vertex_start, len(vertex_order)]).astype(np.int32)
    junction_group = vertex_degree >= 3
    junction_start = vertex_start[junction_group]
    junction_degree = vertex_degree[junction_group]
    junction_vertex = ordered_vertex[junction_start]
    junction_arc_parts = [
        incidence["arc"][vertex_order[start:start + degree]]
        for start, degree in zip(junction_start, junction_degree)
    ]
    junction_arc = (
        np.concatenate(junction_arc_parts).astype(np.int32, copy=False)
        if junction_arc_parts
        else np.empty(0, dtype=np.int32)
    )
    junction_offset = np.empty(len(junction_vertex) + 1, dtype=np.int64)
    junction_offset[0] = 0
    np.cumsum(junction_degree, out=junction_offset[1:])

    return {
        "shape": label.shape,
        "edgel": segment,
        "arc": {
            "count": arc_count,
            "cell_first": segment["cell_first"][first_segment],
            "cell_second": segment["cell_second"][first_segment],
            "length": arc_length,
            "closed": endpoint_count == 0,
            "endpoint_offset": endpoint_offset,
            "endpoint_vertex": endpoint_vertex_unique,
            "endpoint_tangent_x": tangent_x,
            "endpoint_tangent_y": tangent_y,
        },
        "junction": {
            "count": len(junction_vertex),
            "vertex": junction_vertex,
            "x": (junction_vertex % stride).astype(np.int32),
            "y": (junction_vertex // stride).astype(np.int32),
            "arc_offset": junction_offset,
            "arc": junction_arc,
        },
    }


def render_arc_ids(topology: dict) -> np.ndarray:
    """Rasterize embedded arc identity for viewer diagnostics."""
    height, width = topology["shape"]
    out = np.zeros((height, width), dtype=np.int32)
    edgel = topology["edgel"]
    if len(edgel["arc"]) == 0:
        return out
    stride = width + 1
    vertex = edgel["vertex_first"]
    x = vertex % stride
    y = vertex // stride
    vertical = edgel["orientation"] == 1
    # Paint each dual segment onto one of its incident source pixels.
    py = np.where(vertical, y, y - 1)
    px = np.where(vertical, x - 1, x)
    valid = (py >= 0) & (py < height) & (px >= 0) & (px < width)
    out[py[valid], px[valid]] = edgel["arc"][valid] + 1
    return out

