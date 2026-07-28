#!/usr/bin/env python3
"""A conservative part-to-parent layer on embedded object interfaces.

The input ``object_id_per_cell`` remains the visible *part* representation.
This module never deletes those parts.  It proposes parent relations only when
the planar embedding supplies a local topological reason:

* an internal seam terminates at a T-like junction while the two outer
  boundaries continue through a common surround;
* a closed part is contained by one surrounding part.

Every accepted seam also emits mutex relations to its common surround.  Thus
the same local event says both "A and B may share a parent" and "neither may
join S".  Appearance similarity cannot create a candidate by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from experiments.embedded_interface_topology import (
    build_embedded_interface_topology,
)
from experiments.object_hierarchy_diagnostics import object_means


@dataclass(frozen=True)
class ParentHierarchyConfig:
    continuation_floor: float = 0.60
    polarity_floor: float = 0.10
    relation_floor: float = 0.48
    minimum_junction_support: int = 2
    tangent_span: int = 8
    junction_attraction: bool = False
    enclosed_seam_dominance: float = 0.72
    containment: bool = True
    containment_dominance: float = 0.97
    surround_completion: bool = True
    completion_gap_scale: float = 3.0
    completion_polarity_floor: float = 0.10
    completion_relation_floor: float = 0.22
    completion_collision_support: int = 2


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


def _endpoint_tangents(
    topology: dict,
    span: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Trace a few exact edgels inward from each arc endpoint."""
    edgel = topology["edgel"]
    arc_data = topology["arc"]
    segment_arc = np.asarray(edgel["arc"], dtype=np.int32)
    first = np.asarray(edgel["vertex_first"], dtype=np.int64)
    second = np.asarray(edgel["vertex_second"], dtype=np.int64)
    shape = topology["shape"]
    stride = shape[1] + 1

    incident: dict[tuple[int, int], list[int]] = {}
    for segment, arc in enumerate(segment_arc):
        incident.setdefault((int(arc), int(first[segment])), []).append(segment)
        incident.setdefault((int(arc), int(second[segment])), []).append(segment)

    tangent: dict[tuple[int, int], np.ndarray] = {}
    endpoint_offset = arc_data["endpoint_offset"]
    endpoint_vertex = arc_data["endpoint_vertex"]
    for arc in range(arc_data["count"]):
        for cursor in range(endpoint_offset[arc], endpoint_offset[arc + 1]):
            origin = int(endpoint_vertex[cursor])
            current = origin
            previous_segment = -1
            for _ in range(max(int(span), 1)):
                candidates = [
                    segment
                    for segment in incident.get((arc, current), ())
                    if segment != previous_segment
                ]
                if not candidates:
                    break
                segment = min(candidates)
                target = (
                    int(second[segment])
                    if int(first[segment]) == current
                    else int(first[segment])
                )
                previous_segment = segment
                current = target
            vector = np.array((
                (current % stride) - (origin % stride),
                (current // stride) - (origin // stride),
            ), dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                tangent[(arc, origin)] = vector / norm
    return tangent


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _junction_relations(
    topology: dict,
    object_lab: np.ndarray,
    config: ParentHierarchyConfig,
) -> dict[str, np.ndarray]:
    arc_data = topology["arc"]
    junction = topology["junction"]
    tangent = _endpoint_tangents(topology, config.tangent_span)
    records: list[tuple] = []

    for junction_id, vertex in enumerate(junction["vertex"]):
        start = int(junction["arc_offset"][junction_id])
        stop = int(junction["arc_offset"][junction_id + 1])
        arcs = np.unique(junction["arc"][start:stop])
        if len(arcs) != 3:
            continue
        pairs = {
            tuple(sorted((
                int(arc_data["cell_first"][arc]),
                int(arc_data["cell_second"][arc]),
            ))): int(arc)
            for arc in arcs
        }
        regions = sorted({region for pair in pairs for region in pair})
        if len(regions) != 3 or len(pairs) != 3:
            continue

        alternatives = []
        for seam_pair, seam_arc in pairs.items():
            a, b = seam_pair
            surround = next(region for region in regions if region not in seam_pair)
            outer_a = pairs.get(tuple(sorted((a, surround))))
            outer_b = pairs.get(tuple(sorted((b, surround))))
            if outer_a is None or outer_b is None:
                continue
            ta = tangent.get((outer_a, int(vertex)))
            tb = tangent.get((outer_b, int(vertex)))
            ts = tangent.get((seam_arc, int(vertex)))
            if ta is None or tb is None or ts is None:
                continue
            continuation = max(0.0, -_cosine(ta, tb))
            orthogonality = max(
                0.0,
                1.0 - 0.5 * (
                    abs(_cosine(ts, ta)) + abs(_cosine(ts, tb))
                ),
            )
            polarity = max(
                0.0,
                _cosine(
                    object_lab[a] - object_lab[surround],
                    object_lab[b] - object_lab[surround],
                ),
            )
            score = (
                continuation * continuation
                * np.sqrt(polarity)
                * (0.5 + 0.5 * orthogonality)
            )
            alternatives.append((
                score, a, b, surround, seam_arc,
                outer_a, outer_b,
                continuation, polarity, orthogonality,
            ))
        if not alternatives:
            continue
        (
            score, a, b, surround, seam_arc,
            outer_a, outer_b,
            continuation, polarity, orthogonality,
        ) = max(alternatives)
        if (
            continuation < config.continuation_floor
            or polarity < config.polarity_floor
        ):
            continue
        records.append((
            a, b, surround, junction_id,
            seam_arc, outer_a, outer_b,
            score, continuation, polarity, orthogonality,
        ))

    names = (
        "first", "second", "surround", "junction",
        "seam_arc", "first_outer_arc", "second_outer_arc",
        "score", "continuation", "polarity", "orthogonality",
    )
    integer_names = {
        "first", "second", "surround", "junction",
        "seam_arc", "first_outer_arc", "second_outer_arc",
    }
    if not records:
        return {
            name: np.empty(
                0,
                dtype=np.int32 if name in integer_names else np.float64,
            )
            for name in names
        }
    columns = list(zip(*records))
    return {
        name: np.asarray(
            columns[index],
            dtype=np.int32 if name in integer_names else np.float64,
        )
        for index, name in enumerate(names)
    }


def _object_interfaces(labels: np.ndarray, count: int) -> dict[str, np.ndarray]:
    first_parts = []
    second_parts = []
    for a, b in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1], labels[1:]),
    ):
        crossing = a != b
        low = np.minimum(a[crossing], b[crossing])
        high = np.maximum(a[crossing], b[crossing])
        first_parts.append(low)
        second_parts.append(high)
    first = np.concatenate(first_parts).astype(np.int32, copy=False)
    second = np.concatenate(second_parts).astype(np.int32, copy=False)
    key = first.astype(np.int64) * count + second
    unique, length = np.unique(key, return_counts=True)
    return {
        "first": (unique // count).astype(np.int32),
        "second": (unique % count).astype(np.int32),
        "length": length.astype(np.float64),
    }


def _frame_geometry(
    labels: np.ndarray,
    count: int,
) -> dict[str, np.ndarray]:
    """Continuous frame exposure; touching the frame is not a Boolean class."""
    label = np.asarray(labels, dtype=np.int32)
    interface = _object_interfaces(label, count)
    interface_length = (
        np.bincount(
            interface["first"],
            weights=interface["length"],
            minlength=count,
        )
        + np.bincount(
            interface["second"],
            weights=interface["length"],
            minlength=count,
        )
    )
    # Count primal boundary edges. Corners legitimately contribute two edges.
    frame_nodes = np.concatenate((
        label[0], label[-1], label[:, 0], label[:, -1],
    ))
    frame_contact = np.bincount(
        frame_nodes, minlength=count).astype(np.float64)
    perimeter = interface_length + frame_contact
    return {
        "frame_contact": frame_contact,
        "interface_length": interface_length,
        "perimeter": perimeter,
        "frame_exposure": frame_contact / np.maximum(perimeter, 1.0),
        "touches_frame": frame_contact > 0.0,
    }


def _containment_relations(
    labels: np.ndarray,
    count: int,
    config: ParentHierarchyConfig,
) -> dict[str, np.ndarray]:
    interface = _object_interfaces(labels, count)
    first, second, length = (
        interface["first"], interface["second"], interface["length"])
    neighbour_length: list[dict[int, float]] = [
        {} for _ in range(count)
    ]
    for a, b, value in zip(first, second, length):
        neighbour_length[int(a)][int(b)] = (
            neighbour_length[int(a)].get(int(b), 0.0) + float(value))
        neighbour_length[int(b)][int(a)] = (
            neighbour_length[int(b)].get(int(a), 0.0) + float(value))
    border = np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    )))
    touches_border = np.zeros(count, dtype=bool)
    touches_border[border] = True
    child = []
    container = []
    dominance = []
    for node, neighbours in enumerate(neighbour_length):
        if touches_border[node] or not neighbours:
            continue
        ordered = sorted(
            neighbours.items(), key=lambda item: (-item[1], item[0]))
        total = sum(value for _, value in ordered)
        fraction = ordered[0][1] / max(total, 1e-12)
        if fraction >= config.containment_dominance:
            child.append(node)
            container.append(ordered[0][0])
            dominance.append(fraction)
    return {
        "child": np.asarray(child, dtype=np.int32),
        "container": np.asarray(container, dtype=np.int32),
        "score": np.asarray(dominance, dtype=np.float64),
    }


def _enclosed_seam_relations(
    labels: np.ndarray,
    junction: dict[str, np.ndarray],
    config: ParentHierarchyConfig,
) -> dict[str, np.ndarray]:
    """Find terminating seams whose pair shares one enclosing exterior.

    A local T can be either an occlusion or a material seam ending at a
    silhouette.  In the latter case, removing the seam leaves a bounded union
    whose remaining boundary faces predominantly one third region.
    """
    label = np.asarray(labels, dtype=np.int32)
    count = int(label.max(initial=-1)) + 1
    interface = _object_interfaces(label, count)
    neighbour_length: list[dict[int, float]] = [
        {} for _ in range(count)
    ]
    for a, b, length in zip(
        interface["first"], interface["second"], interface["length"]
    ):
        ia, ib, value = int(a), int(b), float(length)
        neighbour_length[ia][ib] = value
        neighbour_length[ib][ia] = value

    frame = _frame_geometry(label, count)
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index in range(len(junction["score"])):
        a, b = sorted((
            int(junction["first"][index]),
            int(junction["second"][index]),
        ))
        surround = int(junction["surround"][index])
        grouped.setdefault((a, b, surround), []).append(index)

    records = []
    required = max(int(config.minimum_junction_support), 1)
    for (a, b, surround), indices in grouped.items():
        if len(indices) < required:
            continue
        exterior: dict[int, float] = {}
        for node, other in ((a, b), (b, a)):
            for neighbour, length in neighbour_length[node].items():
                if neighbour == other:
                    continue
                exterior[neighbour] = exterior.get(neighbour, 0.0) + length
        interface_total = float(sum(exterior.values()))
        frame_contact = float(
            frame["frame_contact"][a] + frame["frame_contact"][b])
        total = interface_total + frame_contact
        surround_length = float(exterior.get(surround, 0.0))
        dominance = surround_length / max(total, 1e-12)
        frame_exposure = frame_contact / max(total, 1e-12)
        if dominance < config.enclosed_seam_dominance:
            continue
        ordered = sorted(
            indices, key=lambda index: -float(junction["score"][index]))
        support = ordered[:required]
        junction_score = float(np.prod(
            np.maximum(junction["score"][support], 0.0)
        ) ** (1.0 / required))
        records.append((
            a, b, surround,
            junction_score * dominance,
            dominance,
            frame_exposure,
            len(indices),
            total,
            surround_length,
        ))

    names = (
        "first", "second", "surround", "score", "exterior_dominance",
        "frame_exposure", "witness_count", "exterior_length",
        "surround_length",
    )
    integer = {"first", "second", "surround", "witness_count"}
    if not records:
        return {
            name: np.empty(
                0, dtype=np.int32 if name in integer else np.float64)
            for name in names
        }
    columns = list(zip(*records))
    return {
        name: np.asarray(
            columns[index],
            dtype=np.int32 if name in integer else np.float64,
        )
        for index, name in enumerate(names)
    }


def _signed_parent_forest(
    count: int,
    junction: dict[str, np.ndarray],
    containment: dict[str, np.ndarray],
    config: ParentHierarchyConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Greedy signed forest on the bounded local proposal set."""
    mutex: list[set[int]] = [set() for _ in range(count)]
    attractive: list[tuple[float, int, int, int]] = []
    # relation kind: 0 junction seam, 1 containment
    pair_records: dict[tuple[int, int], list[int]] = {}
    for index in range(len(junction["score"])):
        pair = tuple(sorted((
            int(junction["first"][index]),
            int(junction["second"][index]),
        )))
        pair_records.setdefault(pair, []).append(index)
    required = max(int(config.minimum_junction_support), 1)
    for (a, b), indices in pair_records.items():
        ordered = sorted(
            indices,
            key=lambda index: (
                -float(junction["score"][index]),
                int(junction["junction"][index])
                if "junction" in junction else index,
            ),
        )
        if len(ordered) < required:
            continue
        support = ordered[:required]
        # Requiring both ends is the topological guard.  The geometric mean
        # makes one weak endpoint veto an otherwise excellent isolated corner.
        score = float(np.prod(
            np.maximum(junction["score"][support], 0.0)
        ) ** (1.0 / required))
        if score < config.relation_floor:
            continue
        if config.junction_attraction:
            attractive.append((score, a, b, 0))
            for index in ordered:
                surround = int(junction["surround"][index])
                mutex[a].add(surround)
                mutex[surround].add(a)
                mutex[b].add(surround)
                mutex[surround].add(b)
    if config.containment:
        for child, container, score in zip(
            containment["child"],
            containment["container"],
            containment["score"],
        ):
            attractive.append((
                float(score), int(child), int(container), 1))
    attractive.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, dtype=np.int32)
    component_mutex = [set(values) for values in mutex]

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            previous = int(parent[node])
            parent[node] = root
            node = previous
        return root

    accepted = []
    for score, a, b, kind in attractive:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        blocked = (
            rb in {find(node) for node in component_mutex[ra]}
            or ra in {find(node) for node in component_mutex[rb]}
        )
        if blocked:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        combined = {
            find(node)
            for node in component_mutex[ra] | component_mutex[rb]
            if find(node) not in (ra, rb)
        }
        component_mutex[ra] = combined
        component_mutex[rb] = set()
        for other in combined:
            component_mutex[other].discard(ra)
            component_mutex[other].discard(rb)
            component_mutex[other].add(ra)
        accepted.append((a, b, score, kind))

    roots = np.fromiter(
        (find(node) for node in range(count)),
        dtype=np.int32,
        count=count,
    )
    _, parent_id = np.unique(roots, return_inverse=True)
    if accepted:
        columns = list(zip(*accepted))
        accepted_data = {
            "first": np.asarray(columns[0], dtype=np.int32),
            "second": np.asarray(columns[1], dtype=np.int32),
            "score": np.asarray(columns[2], dtype=np.float64),
            "kind": np.asarray(columns[3], dtype=np.int8),
        }
    else:
        accepted_data = {
            "first": np.empty(0, dtype=np.int32),
            "second": np.empty(0, dtype=np.int32),
            "score": np.empty(0, dtype=np.float64),
            "kind": np.empty(0, dtype=np.int8),
        }
    return parent_id.astype(np.int32), accepted_data


def _csr(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = (
        np.bincount(first, minlength=count)
        + np.bincount(second, minlength=count)
    ).astype(np.int64)
    offset = np.empty(count + 1, dtype=np.int64)
    offset[0] = 0
    np.cumsum(degree, out=offset[1:])
    neighbour = np.empty(offset[-1], dtype=np.int32)
    value = np.empty(offset[-1], dtype=np.float64)
    cursor = offset[:-1].copy()
    for a, b, edge_value in zip(first, second, weight):
        ia = cursor[a]
        neighbour[ia], value[ia] = b, edge_value
        cursor[a] += 1
        ib = cursor[b]
        neighbour[ib], value[ib] = a, edge_value
        cursor[b] += 1
    return offset, neighbour, value


def _aggregate_parent_means(
    objects: dict,
    preliminary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = object_means(objects)
    count = int(preliminary.max(initial=-1)) + 1
    area = np.asarray(means["pixel_area"], dtype=np.float64)
    total = np.bincount(
        preliminary, weights=area, minlength=count)
    lab = np.column_stack([
        np.bincount(
            preliminary,
            weights=area * means["lab"][:, channel],
            minlength=count,
        ) / np.maximum(total, 1e-12)
        for channel in range(means["lab"].shape[1])
    ])
    trace_part = (
        np.asarray(means["qxx"]) + np.asarray(means["qyy"]))
    trace = np.bincount(
        preliminary,
        weights=area * trace_part,
        minlength=count,
    ) / np.maximum(total, 1e-12)
    representative = np.full(count, len(preliminary), dtype=np.int32)
    np.minimum.at(
        representative,
        preliminary,
        np.arange(len(preliminary), dtype=np.int32),
    )
    return lab, trace, representative


def _common_surround_wavefront(
    objects: dict,
    preliminary: np.ndarray,
    config: ParentHierarchyConfig,
) -> dict[str, np.ndarray]:
    """First-arrival collisions inside each surround propose amodal parents."""
    graph = objects["graph"]
    cells = graph["cells"]
    edge = graph["edge"]
    first = np.asarray(edge["first"], dtype=np.int32)
    second = np.asarray(edge["second"], dtype=np.int32)
    part = np.asarray(objects["object_id_per_cell"], dtype=np.int32)
    domain = preliminary[part]
    domain_count = int(domain.max(initial=-1)) + 1
    barrier = np.asarray(objects["evidence"]["barrier"], dtype=np.float64)
    travel = np.hypot(
        graph["node_x"][first] - graph["node_x"][second],
        graph["node_y"][first] - graph["node_y"][second],
    )
    offset, neighbour, edge_travel = _csr(
        cells, first, second, np.maximum(travel, 1e-6))

    source = np.full(cells, -1, dtype=np.int32)
    distance = np.full(cells, np.inf, dtype=np.float64)
    source_strength = np.zeros(cells, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    crossing = domain[first] != domain[second]
    direct_collision: list[tuple[int, int, int, float, float]] = []

    def seed(node: int, outside: int, strength: float) -> None:
        if source[node] < 0:
            source[node] = outside
            distance[node] = 0.0
            source_strength[node] = strength
            heapq.heappush(heap, (0.0, node, outside))
        elif source[node] != outside:
            direct_collision.append((
                int(source[node]), outside, int(domain[node]),
                0.0, min(float(source_strength[node]), strength),
            ))
            if strength > source_strength[node]:
                source[node] = outside
                source_strength[node] = strength
                heapq.heappush(heap, (0.0, node, outside))

    for edge_index in np.flatnonzero(crossing):
        a, b = int(first[edge_index]), int(second[edge_index])
        seed(a, int(domain[b]), float(barrier[edge_index]))
        seed(b, int(domain[a]), float(barrier[edge_index]))

    while heap:
        value, node, owner = heapq.heappop(heap)
        if (
            owner != source[node]
            or value > distance[node] + 1e-12
        ):
            continue
        for cursor in range(offset[node], offset[node + 1]):
            target = int(neighbour[cursor])
            if domain[target] != domain[node]:
                continue
            candidate = value + float(edge_travel[cursor])
            if candidate + 1e-12 < distance[target]:
                distance[target] = candidate
                source[target] = owner
                source_strength[target] = source_strength[node]
                heapq.heappush(heap, (candidate, target, owner))

    collision: list[tuple[int, int, int, float, float]] = direct_collision
    internal = domain[first] == domain[second]
    valid = (
        internal
        & (source[first] >= 0)
        & (source[second] >= 0)
        & (source[first] != source[second])
    )
    for edge_index in np.flatnonzero(valid):
        a, b = int(first[edge_index]), int(second[edge_index])
        collision.append((
            int(source[a]),
            int(source[b]),
            int(domain[a]),
            float(distance[a] + travel[edge_index] + distance[b]),
            min(float(source_strength[a]), float(source_strength[b])),
        ))

    parent_lab, parent_trace, representative = _aggregate_parent_means(
        objects, preliminary)
    spacing = max(
        float(np.sqrt(np.median(graph["area"]))),
        1.0,
    )
    grouped: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    for a, b, surround, gap, strength in collision:
        if a == b or a == surround or b == surround:
            continue
        a, b = sorted((a, b))
        polarity = max(
            0.0,
            _cosine(
                parent_lab[a] - parent_lab[surround],
                parent_lab[b] - parent_lab[surround],
            ),
        )
        if polarity < config.completion_polarity_floor:
            continue
        scale_agreement = np.exp(
            -0.25 * abs(np.log(
                max(float(parent_trace[a]), 1e-12)
                / max(float(parent_trace[b]), 1e-12)
            ))
        )
        gap_score = np.exp(
            -gap / max(
                float(config.completion_gap_scale) * spacing,
                1e-12,
            )
        )
        score = (
            gap_score
            * np.sqrt(polarity)
            * np.sqrt(max(strength, 0.0))
            * scale_agreement
        )
        grouped.setdefault((a, b, surround), []).append((score, gap))

    records = []
    required = max(int(config.completion_collision_support), 1)
    for (a, b, surround), values in grouped.items():
        values.sort(key=lambda item: (-item[0], item[1]))
        if len(values) < required:
            continue
        support = values[:required]
        score = float(np.prod([
            max(item[0], 0.0) for item in support
        ]) ** (1.0 / required))
        if score < config.completion_relation_floor:
            continue
        records.append((
            a, b, surround, score,
            float(np.mean([item[1] for item in support])),
            len(values),
            int(representative[a]), int(representative[b]),
        ))
    names = (
        "first", "second", "surround", "score",
        "gap", "collision_count", "part_first", "part_second",
    )
    if not records:
        return {
            name: np.empty(
                0,
                dtype=np.float64 if name in ("score", "gap") else np.int32,
            )
            for name in names
        }
    columns = list(zip(*records))
    return {
        name: np.asarray(
            columns[index],
            dtype=np.float64
            if name in ("score", "gap")
            else np.int32,
        )
        for index, name in enumerate(names)
    }


def _merge_completion_relations(
    count: int,
    relation: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Signed union of already-aggregated surround-completion proposals."""
    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, dtype=np.int32)
    mutex: list[set[int]] = [set() for _ in range(count)]
    for a, b, surround in zip(
        relation["first"], relation["second"], relation["surround"]
    ):
        a, b, surround = int(a), int(b), int(surround)
        mutex[a].add(surround)
        mutex[b].add(surround)
        mutex[surround].update((a, b))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            previous = int(parent[node])
            parent[node] = root
            node = previous
        return root

    accepted = []
    order = np.argsort(-relation["score"], kind="stable")
    for index in order:
        a = find(int(relation["first"][index]))
        b = find(int(relation["second"][index]))
        if a == b:
            continue
        if (
            b in {find(node) for node in mutex[a]}
            or a in {find(node) for node in mutex[b]}
        ):
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        combined = {
            find(node)
            for node in mutex[a] | mutex[b]
            if find(node) not in (a, b)
        }
        mutex[a] = combined
        mutex[b] = set()
        for other in combined:
            mutex[other].discard(a)
            mutex[other].discard(b)
            mutex[other].add(a)
        accepted.append(int(index))
    roots = np.fromiter(
        (find(node) for node in range(count)),
        dtype=np.int32,
        count=count,
    )
    _, merged = np.unique(roots, return_inverse=True)
    return merged.astype(np.int32), np.asarray(accepted, dtype=np.int32)


def infer_parent_objects(
    objects: dict,
    config: ParentHierarchyConfig = ParentHierarchyConfig(),
) -> dict:
    """Group visible part IDs into conservative parent-object hypotheses."""
    part_labels = np.asarray(objects["object_labels"], dtype=np.int32)
    count = int(part_labels.max(initial=-1)) + 1
    topology = build_embedded_interface_topology(part_labels)
    frame_geometry = _frame_geometry(part_labels, count)
    means = object_means(objects)
    junction = _junction_relations(
        topology, np.asarray(means["lab"]), config)
    enclosed_seam = _enclosed_seam_relations(
        part_labels, junction, config)
    containment = _containment_relations(
        part_labels, count, config)
    preliminary, accepted = _signed_parent_forest(
        count, junction, containment, config)
    if config.surround_completion:
        completion = _common_surround_wavefront(
            objects, preliminary, config)
        completion_map, completion_accepted = _merge_completion_relations(
            int(preliminary.max(initial=-1)) + 1,
            completion,
        )
        parent_id = completion_map[preliminary]
    else:
        completion = _common_surround_wavefront(
            objects, preliminary, config)
        completion_accepted = np.empty(0, dtype=np.int32)
        parent_id = preliminary
    parent_labels = parent_id[part_labels]
    colours = _stable_colours(
        int(parent_id.max(initial=-1)) + 1)
    return {
        "config": config,
        "topology": topology,
        "frame_geometry": frame_geometry,
        "topology_part_labels": part_labels,
        "part_count": count,
        "parent_count": int(parent_id.max(initial=-1)) + 1,
        "parent_id_per_part": parent_id,
        "parent_labels": parent_labels,
        "parent_ids": colours[parent_labels],
        "junction_relations": junction,
        "enclosed_seam_relations": enclosed_seam,
        "containment_relations": containment,
        "accepted_relations": accepted,
        "preliminary_parent_id_per_part": preliminary,
        "completion_relations": completion,
        "accepted_completion_indices": completion_accepted,
    }
