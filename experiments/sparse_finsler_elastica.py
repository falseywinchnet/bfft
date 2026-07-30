"""Sparse orientation-lifted Finsler-elastica paths on intrinsic interfaces.

Chen, Mirebeau, and Cohen lift a curve from ``x`` to ``(x, theta)`` and use

    integral (1 + alpha * curvature**2) / Phi(x, theta) ds.

A dense discretization is H x W x orientations.  The intrinsic Voronoi
partition already supplies a much smaller embedded interface complex with
measured tangents.  Each connected interface arc therefore has two directed
states.  Moving through an arc pays its data-weighted length; changing from
one directed arc to the next pays the discrete Euler-elastica bending term

    alpha * wrapped_angle**2 / local_arclength.

This is the line-graph analogue of the orientation lift.  It is asymmetric,
curvature penalized, and solved by one label-setting Dijkstra march.  It does
not alter either the intrinsic or canonical segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from bfft.effects import srgb_to_lab

try:
    from numba import njit
except ImportError:  # pragma: no cover - project runtime includes numba
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


@dataclass(frozen=True)
class SparseElasticaConfig:
    curvature_scale: float = 8.0
    speed_floor: float = 0.08
    boundary_weight: float = 1.0
    colour_weight: float = 0.65
    support_weight: float = 0.35

    @property
    def alpha(self) -> float:
        return max(float(self.curvature_scale), 0.0) ** 2


@njit(cache=True)
def _elastica_transitions(
    state_start_vertex: np.ndarray,
    state_end_vertex: np.ndarray,
    state_start_tangent: np.ndarray,
    state_end_tangent: np.ndarray,
    state_reverse: np.ndarray,
    state_length: np.ndarray,
    state_travel_cost: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the lifted line graph in linear time."""

    state_count = len(state_start_vertex)
    vertex_count = int(max(
        state_start_vertex.max(),
        state_end_vertex.max(),
    )) + 1
    degree = np.zeros(vertex_count, dtype=np.int64)
    for state in range(state_count):
        degree[state_start_vertex[state]] += 1
    vertex_offset = np.empty(vertex_count + 1, dtype=np.int64)
    vertex_offset[0] = 0
    for vertex in range(vertex_count):
        vertex_offset[vertex + 1] = (
            vertex_offset[vertex] + degree[vertex])
    vertex_state = np.empty(state_count, dtype=np.int32)
    cursor = vertex_offset[:-1].copy()
    for state in range(state_count):
        vertex = state_start_vertex[state]
        vertex_state[cursor[vertex]] = state
        cursor[vertex] += 1

    transition_offset = np.empty(state_count + 1, dtype=np.int64)
    transition_offset[0] = 0
    for state in range(state_count):
        vertex = state_end_vertex[state]
        transition_offset[state + 1] = (
            transition_offset[state]
            + degree[vertex] - 1
        )
    transition_target = np.empty(
        transition_offset[-1], dtype=np.int32)
    transition_cost = np.empty(
        transition_offset[-1], dtype=np.float64)
    for state in range(state_count):
        vertex = state_end_vertex[state]
        write = transition_offset[state]
        for incident in range(
            vertex_offset[vertex], vertex_offset[vertex + 1]
        ):
            candidate = vertex_state[incident]
            if candidate == state_reverse[state]:
                continue
            dot = (
                state_end_tangent[state, 0]
                * state_start_tangent[candidate, 0]
                + state_end_tangent[state, 1]
                * state_start_tangent[candidate, 1]
            )
            cross = (
                state_end_tangent[state, 0]
                * state_start_tangent[candidate, 1]
                - state_end_tangent[state, 1]
                * state_start_tangent[candidate, 0]
            )
            angle = np.arctan2(cross, dot)
            local_length = max(
                0.5 * (
                    state_length[state] + state_length[candidate]),
                1.0,
            )
            transition_target[write] = candidate
            transition_cost[write] = (
                state_travel_cost[candidate]
                + alpha * angle * angle / local_length
            )
            write += 1
    return transition_offset, transition_target, transition_cost


def _robust_unit(values: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    value = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = value[value > 0.0]
    scale = (
        float(np.percentile(positive, percentile))
        if positive.size
        else 1.0
    )
    return value / (value + max(scale, 1e-12))


def _cell_mean(
    labels: np.ndarray,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    flat = np.asarray(labels, dtype=np.int32).ravel()
    area = np.bincount(flat, minlength=count).astype(np.float64)
    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        total = np.bincount(
            flat, weights=field.ravel(), minlength=count)
        return total / np.maximum(area, 1.0)
    return np.column_stack([
        np.bincount(
            flat,
            weights=field[..., channel].ravel(),
            minlength=count,
        ) / np.maximum(area, 1.0)
        for channel in range(field.shape[2])
    ])


def _edgel_pixel_samples(
    topology: dict,
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the pixels on the two sides of every embedded dual segment."""

    height, width = topology["shape"]
    edgel = topology["edgel"]
    vertex = np.asarray(edgel["vertex_first"], dtype=np.int64)
    stride = width + 1
    x = vertex % stride
    y = vertex // stride
    vertical = np.asarray(edgel["orientation"], dtype=np.int8) == 1
    first_y = np.where(vertical, y, y - 1)
    first_x = np.where(vertical, x - 1, x)
    second_y = np.where(vertical, y, y)
    second_x = np.where(vertical, x, x)
    first_y = np.clip(first_y, 0, height - 1)
    first_x = np.clip(first_x, 0, width - 1)
    second_y = np.clip(second_y, 0, height - 1)
    second_x = np.clip(second_x, 0, width - 1)
    value = np.asarray(field)
    return (
        value[first_y, first_x],
        value[second_y, second_x],
    )


def intrinsic_arc_speed(
    topology: dict,
    intrinsic_owner: np.ndarray,
    target_rgb: np.ndarray,
    support_measure: np.ndarray,
    boundary_confidence: np.ndarray,
    config: SparseElasticaConfig = SparseElasticaConfig(),
) -> dict[str, np.ndarray]:
    """Measure the orientation-dependent speed of intrinsic boundary arcs."""

    labels = np.asarray(intrinsic_owner, dtype=np.int32)
    if labels.shape != tuple(topology["shape"]):
        raise ValueError("intrinsic owner and topology must share a shape")
    count = int(labels.max(initial=-1)) + 1
    lab = np.asarray(srgb_to_lab(target_rgb), dtype=np.float64)
    node_lab = _cell_mean(labels, lab, count)
    node_support = _cell_mean(labels, support_measure, count)
    arc = topology["arc"]
    first = np.asarray(arc["cell_first"], dtype=np.int32)
    second = np.asarray(arc["cell_second"], dtype=np.int32)
    colour_jump = _robust_unit(np.linalg.norm(
        node_lab[first] - node_lab[second], axis=1))
    support_jump = _robust_unit(np.abs(np.log(
        np.maximum(node_support[first], 1e-30)
        / np.maximum(node_support[second], 1e-30)
    )))

    boundary_first, boundary_second = _edgel_pixel_samples(
        topology, boundary_confidence)
    boundary_sample = np.maximum(
        np.asarray(boundary_first, dtype=np.float64),
        np.asarray(boundary_second, dtype=np.float64),
    )
    edgel_arc = np.asarray(topology["edgel"]["arc"], dtype=np.int32)
    arc_count = int(arc["count"])
    length = np.bincount(edgel_arc, minlength=arc_count)
    boundary = np.bincount(
        edgel_arc,
        weights=boundary_sample,
        minlength=arc_count,
    ) / np.maximum(length, 1)
    boundary = np.clip(boundary, 0.0, 1.0)

    evidence = (
        max(float(config.boundary_weight), 0.0) * boundary
        + max(float(config.colour_weight), 0.0) * colour_jump
        + max(float(config.support_weight), 0.0) * support_jump
    )
    saliency = 1.0 - np.exp(-evidence)
    floor = float(np.clip(config.speed_floor, 1e-6, 1.0))
    speed = floor + (1.0 - floor) * saliency
    return {
        "speed": np.clip(speed, floor, 1.0),
        "saliency": np.clip(saliency, 0.0, 1.0),
        "boundary": boundary,
        "colour_jump": colour_jump,
        "support_jump": support_jump,
    }


def build_sparse_elastica_graph(
    topology: dict,
    arc_speed: np.ndarray,
    config: SparseElasticaConfig = SparseElasticaConfig(),
) -> dict[str, np.ndarray | int]:
    """Build the directed line graph of two-ended embedded interface arcs."""

    arc = topology["arc"]
    arc_count = int(arc["count"])
    speed = np.asarray(arc_speed, dtype=np.float64)
    if speed.shape != (arc_count,):
        raise ValueError("arc speed must contain one value per arc")
    offset = np.asarray(arc["endpoint_offset"], dtype=np.int64)
    endpoint_count = np.diff(offset)
    ordinary = np.flatnonzero(endpoint_count == 2).astype(np.int32)
    local_count = len(ordinary)
    state_count = 2 * local_count
    if state_count == 0:
        return {
            "state_count": 0,
            "ordinary_arc": ordinary,
            "state_arc": np.empty(0, dtype=np.int32),
            "state_reverse": np.empty(0, dtype=np.int32),
            "state_start_vertex": np.empty(0, dtype=np.int64),
            "state_end_vertex": np.empty(0, dtype=np.int64),
            "state_travel_cost": np.empty(0, dtype=np.float64),
            "transition_offset": np.zeros(1, dtype=np.int64),
            "transition_target": np.empty(0, dtype=np.int32),
            "transition_cost": np.empty(0, dtype=np.float64),
        }

    endpoint_vertex = np.asarray(
        arc["endpoint_vertex"], dtype=np.int64)
    tangent_x = np.asarray(
        arc["endpoint_tangent_x"], dtype=np.float64)
    tangent_y = np.asarray(
        arc["endpoint_tangent_y"], dtype=np.float64)
    first_cursor = offset[ordinary]
    second_cursor = first_cursor + 1
    first_vertex = endpoint_vertex[first_cursor]
    second_vertex = endpoint_vertex[second_cursor]
    first_tangent = np.column_stack((
        tangent_x[first_cursor], tangent_y[first_cursor]))
    second_tangent = np.column_stack((
        tangent_x[second_cursor], tangent_y[second_cursor]))

    state_arc = np.repeat(ordinary, 2)
    state_reverse = np.arange(state_count, dtype=np.int32) ^ 1
    state_start_vertex = np.empty(state_count, dtype=np.int64)
    state_end_vertex = np.empty(state_count, dtype=np.int64)
    state_start_tangent = np.empty((state_count, 2), dtype=np.float64)
    state_end_tangent = np.empty((state_count, 2), dtype=np.float64)
    state_start_vertex[0::2] = first_vertex
    state_end_vertex[0::2] = second_vertex
    state_start_tangent[0::2] = first_tangent
    state_end_tangent[0::2] = -second_tangent
    state_start_vertex[1::2] = second_vertex
    state_end_vertex[1::2] = first_vertex
    state_start_tangent[1::2] = second_tangent
    state_end_tangent[1::2] = -first_tangent

    length = np.asarray(arc["length"], dtype=np.float64)[state_arc]
    state_travel_cost = length / np.maximum(speed[state_arc], 1e-6)
    (
        transition_offset,
        transition_target,
        transition_cost,
    ) = _elastica_transitions(
        state_start_vertex,
        state_end_vertex,
        state_start_tangent,
        state_end_tangent,
        state_reverse,
        length,
        state_travel_cost,
        config.alpha,
    )

    return {
        "state_count": state_count,
        "ordinary_arc": ordinary,
        "state_arc": state_arc,
        "state_reverse": state_reverse,
        "state_start_vertex": state_start_vertex,
        "state_end_vertex": state_end_vertex,
        "state_start_tangent": state_start_tangent,
        "state_end_tangent": state_end_tangent,
        "state_travel_cost": state_travel_cost,
        "transition_offset": transition_offset,
        "transition_target": transition_target,
        "transition_cost": transition_cost,
    }


def sparse_elastica_distance(
    graph: dict,
    sources: np.ndarray,
    *,
    targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run one label-setting march from directed source states."""

    count = int(graph["state_count"])
    source = np.unique(np.asarray(sources, dtype=np.int32))
    if np.any((source < 0) | (source >= count)):
        raise ValueError("source state is outside the elastica graph")
    target_mask = np.zeros(count, dtype=bool)
    remaining = 0
    if targets is not None:
        target = np.unique(np.asarray(targets, dtype=np.int32))
        if np.any((target < 0) | (target >= count)):
            raise ValueError("target state is outside the elastica graph")
        target_mask[target] = True
        remaining = len(target)

    distance = np.full(count, np.inf, dtype=np.float64)
    predecessor = np.full(count, -1, dtype=np.int32)
    source_id = np.full(count, -1, dtype=np.int32)
    heap: list[tuple[float, int]] = []
    initial = np.asarray(
        graph["state_travel_cost"], dtype=np.float64)
    for index, state in enumerate(source):
        distance[state] = initial[state]
        source_id[state] = index
        heapq.heappush(heap, (float(distance[state]), int(state)))

    offset = np.asarray(graph["transition_offset"], dtype=np.int64)
    neighbour = np.asarray(graph["transition_target"], dtype=np.int32)
    cost = np.asarray(graph["transition_cost"], dtype=np.float64)
    settled = np.zeros(count, dtype=bool)
    while heap:
        value, state = heapq.heappop(heap)
        if settled[state] or value > distance[state] + 1e-12:
            continue
        settled[state] = True
        if target_mask[state]:
            target_mask[state] = False
            remaining -= 1
            if remaining == 0:
                break
        for cursor in range(offset[state], offset[state + 1]):
            target_state = int(neighbour[cursor])
            candidate = value + float(cost[cursor])
            if candidate + 1e-12 < distance[target_state]:
                distance[target_state] = candidate
                predecessor[target_state] = state
                source_id[target_state] = source_id[state]
                heapq.heappush(heap, (candidate, target_state))
    return {
        "distance": distance,
        "predecessor": predecessor,
        "source_id": source_id,
        "settled": settled,
    }


def label_intrinsic_arc_parts(
    topology: dict,
    part_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign each intrinsic arc its dominant canonical part interface.

    Arcs inside one canonical part remain unlabeled.  A labeled arc is an
    observed member of an unordered boundary family ``{part, surround}``;
    all unlabeled arcs remain available as completion geometry.
    """

    labels = np.asarray(part_labels, dtype=np.int32)
    if labels.shape != tuple(topology["shape"]):
        raise ValueError("part labels and topology must share a shape")
    arc_count = int(topology["arc"]["count"])
    arc_first = np.full(arc_count, -1, dtype=np.int32)
    arc_second = np.full(arc_count, -1, dtype=np.int32)
    purity = np.zeros(arc_count, dtype=np.float64)
    crossing_fraction = np.zeros(arc_count, dtype=np.float64)
    if arc_count == 0:
        return {
            "first": arc_first,
            "second": arc_second,
            "purity": purity,
            "crossing_fraction": crossing_fraction,
        }

    first_sample, second_sample = _edgel_pixel_samples(topology, labels)
    first_sample = np.asarray(first_sample, dtype=np.int32)
    second_sample = np.asarray(second_sample, dtype=np.int32)
    edgel_arc = np.asarray(topology["edgel"]["arc"], dtype=np.int32)
    total = np.bincount(edgel_arc, minlength=arc_count)
    crossing = first_sample != second_sample
    crossing_count = np.bincount(
        edgel_arc[crossing], minlength=arc_count)
    crossing_fraction = crossing_count / np.maximum(total, 1)
    if not np.any(crossing):
        return {
            "first": arc_first,
            "second": arc_second,
            "purity": purity,
            "crossing_fraction": crossing_fraction,
        }

    part_count = int(labels.max(initial=-1)) + 1
    low = np.minimum(
        first_sample[crossing], second_sample[crossing])
    high = np.maximum(
        first_sample[crossing], second_sample[crossing])
    key = low.astype(np.int64) * part_count + high
    owner_arc = edgel_arc[crossing]
    order = np.lexsort((key, owner_arc))
    ordered_arc = owner_arc[order]
    ordered_key = key[order]
    run_start = np.r_[0, 1 + np.flatnonzero(
        (ordered_arc[1:] != ordered_arc[:-1])
        | (ordered_key[1:] != ordered_key[:-1])
    )]
    run_stop = np.r_[run_start[1:], len(order)]
    run_arc = ordered_arc[run_start]
    run_key = ordered_key[run_start]
    run_count = run_stop - run_start
    winner_order = np.lexsort((run_key, -run_count, run_arc))
    candidate_arc = run_arc[winner_order]
    first_for_arc = np.r_[
        True, candidate_arc[1:] != candidate_arc[:-1]]
    winner = winner_order[first_for_arc]
    selected_arc = run_arc[winner]
    selected_key = run_key[winner]
    selected_count = run_count[winner]
    arc_first[selected_arc] = (
        selected_key // part_count).astype(np.int32)
    arc_second[selected_arc] = (
        selected_key % part_count).astype(np.int32)
    purity[selected_arc] = (
        selected_count
        / np.maximum(crossing_count[selected_arc], 1)
    )
    return {
        "first": arc_first,
        "second": arc_second,
        "purity": purity,
        "crossing_fraction": crossing_fraction,
    }


def sparse_elastica_voronoi(
    graph: dict,
    sources: np.ndarray,
    source_group: np.ndarray,
    *,
    source_potential: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """One multi-source lifted march retaining a boundary-family owner."""

    count = int(graph["state_count"])
    source = np.asarray(sources, dtype=np.int32)
    group = np.asarray(source_group, dtype=np.int32)
    if source.shape != group.shape:
        raise ValueError("source states and groups must share a shape")
    if np.any((source < 0) | (source >= count)):
        raise ValueError("source state is outside the elastica graph")
    if source_potential is None:
        potential = np.zeros(len(source), dtype=np.float64)
    else:
        potential = np.asarray(source_potential, dtype=np.float64)
        if potential.shape != source.shape:
            raise ValueError("source potential must match source states")

    distance = np.full(count, np.inf, dtype=np.float64)
    predecessor = np.full(count, -1, dtype=np.int32)
    owner = np.full(count, -1, dtype=np.int32)
    root = np.full(count, -1, dtype=np.int32)
    heap: list[tuple[float, int]] = []
    for state, label, value in zip(source, group, potential):
        state = int(state)
        if value + 1e-12 < distance[state]:
            distance[state] = float(value)
            owner[state] = int(label)
            root[state] = state
            heapq.heappush(heap, (float(value), state))

    offset = np.asarray(graph["transition_offset"], dtype=np.int64)
    neighbour = np.asarray(graph["transition_target"], dtype=np.int32)
    cost = np.asarray(graph["transition_cost"], dtype=np.float64)
    settled = np.zeros(count, dtype=bool)
    while heap:
        value, state = heapq.heappop(heap)
        if settled[state] or value > distance[state] + 1e-12:
            continue
        settled[state] = True
        for cursor in range(offset[state], offset[state + 1]):
            target = int(neighbour[cursor])
            candidate = value + float(cost[cursor])
            if candidate + 1e-12 < distance[target]:
                distance[target] = candidate
                predecessor[target] = state
                owner[target] = owner[state]
                root[target] = root[state]
                heapq.heappush(heap, (candidate, target))
    return {
        "distance": distance,
        "predecessor": predecessor,
        "owner": owner,
        "root": root,
        "settled": settled,
    }


def elastica_common_surround_relations(
    graph: dict,
    arc_parts: dict[str, np.ndarray],
    arc_saliency: np.ndarray,
) -> dict[str, np.ndarray | dict]:
    """Infer part relations from lifted fronts with a shared surround.

    If fronts from boundary families ``{A,S}`` and ``{B,S}`` meet with low
    Finsler action, the shared label ``S`` falls out algebraically and the
    collision proposes that ``A`` and ``B`` share a parent distinct from
    ``S``.  Every family participates in one geodesic-Voronoi march.
    """

    state_arc = np.asarray(graph["state_arc"], dtype=np.int32)
    first = np.asarray(arc_parts["first"], dtype=np.int32)
    second = np.asarray(arc_parts["second"], dtype=np.int32)
    purity = np.asarray(arc_parts["purity"], dtype=np.float64)
    saliency = np.clip(
        np.asarray(arc_saliency, dtype=np.float64), 0.0, 1.0)
    if not (
        first.shape == second.shape == purity.shape == saliency.shape
    ):
        raise ValueError("arc part and saliency arrays must share a shape")
    observed = (
        (first[state_arc] >= 0)
        & (second[state_arc] >= 0)
        & (first[state_arc] != second[state_arc])
    )
    source = np.flatnonzero(observed).astype(np.int32)
    empty_int = np.empty(0, dtype=np.int32)
    empty_float = np.empty(0, dtype=np.float64)
    if source.size == 0:
        return {
            "first": empty_int,
            "second": empty_int,
            "surround": empty_int,
            "action": empty_float,
            "strength": empty_float,
            "vertex": np.empty(0, dtype=np.int64),
            "state_first": empty_int,
            "state_second": empty_int,
            "wavefront": sparse_elastica_voronoi(
                graph, source, source),
        }

    source_arc = state_arc[source]
    pair = np.column_stack((
        first[source_arc], second[source_arc]))
    unique_pair, source_group = np.unique(
        pair, axis=0, return_inverse=True)
    source_strength = np.clip(
        saliency[source_arc] * purity[source_arc],
        1e-6,
        1.0,
    )
    travel = np.asarray(
        graph["state_travel_cost"], dtype=np.float64)
    finite_travel = travel[np.isfinite(travel) & (travel > 0.0)]
    source_scale = (
        float(np.median(finite_travel))
        if finite_travel.size
        else 1.0
    )
    source_potential = -source_scale * np.log(source_strength)
    wavefront = sparse_elastica_voronoi(
        graph,
        source,
        source_group.astype(np.int32),
        source_potential=source_potential,
    )
    root_strength = np.zeros(int(graph["state_count"]), dtype=np.float64)
    np.maximum.at(root_strength, source, source_strength)

    owner = np.asarray(wavefront["owner"], dtype=np.int32)
    distance = np.asarray(wavefront["distance"], dtype=np.float64)
    root = np.asarray(wavefront["root"], dtype=np.int32)
    reverse = np.asarray(graph["state_reverse"], dtype=np.int32)
    offset = np.asarray(graph["transition_offset"], dtype=np.int64)
    target = np.asarray(graph["transition_target"], dtype=np.int32)
    transition_cost = np.asarray(
        graph["transition_cost"], dtype=np.float64)
    end_vertex = np.asarray(
        graph["state_end_vertex"], dtype=np.int64)

    collision: dict[
        tuple[int, int, int, int],
        tuple[float, float, int, int],
    ] = {}
    for state in range(int(graph["state_count"])):
        group_a = int(owner[state])
        if group_a < 0:
            continue
        a0, a1 = unique_pair[group_a]
        for cursor in range(offset[state], offset[state + 1]):
            next_state = int(target[cursor])
            opposite = int(reverse[next_state])
            group_b = int(owner[opposite])
            if group_b < 0 or group_b == group_a:
                continue
            b0, b1 = unique_pair[group_b]
            if a0 == b0 or a0 == b1:
                common = int(a0)
                part_a = int(a1)
            elif a1 == b0 or a1 == b1:
                common = int(a1)
                part_a = int(a0)
            else:
                continue
            part_b = int(b1 if b0 == common else b0)
            if part_a == part_b or part_a == common or part_b == common:
                continue
            low, high = sorted((part_a, part_b))
            bend = max(
                float(transition_cost[cursor])
                - float(travel[next_state]),
                0.0,
            )
            action = (
                float(distance[state])
                + bend
                + float(distance[opposite])
            )
            strength = min(
                float(root_strength[root[state]]),
                float(root_strength[root[opposite]]),
            )
            vertex = int(end_vertex[state])
            key = (low, high, common, vertex)
            old = collision.get(key)
            if old is None or action < old[0]:
                collision[key] = (
                    action, strength, state, opposite)

    if not collision:
        return {
            "first": empty_int,
            "second": empty_int,
            "surround": empty_int,
            "action": empty_float,
            "strength": empty_float,
            "vertex": np.empty(0, dtype=np.int64),
            "state_first": empty_int,
            "state_second": empty_int,
            "wavefront": wavefront,
        }
    record = sorted(
        (
            key[0], key[1], key[2], value[0], value[1], key[3],
            value[2], value[3],
        )
        for key, value in collision.items()
    )
    column = list(zip(*record))
    return {
        "first": np.asarray(column[0], dtype=np.int32),
        "second": np.asarray(column[1], dtype=np.int32),
        "surround": np.asarray(column[2], dtype=np.int32),
        "action": np.asarray(column[3], dtype=np.float64),
        "strength": np.asarray(column[4], dtype=np.float64),
        "vertex": np.asarray(column[5], dtype=np.int64),
        "state_first": np.asarray(column[6], dtype=np.int32),
        "state_second": np.asarray(column[7], dtype=np.int32),
        "wavefront": wavefront,
    }


def finsler_saliency_closing(
    graph: dict,
    arc_saliency: np.ndarray,
    *,
    continuation_scale: float = 6.0,
) -> dict[str, np.ndarray | float | dict]:
    """Close weak contour gaps by a min-plus Finsler inf-convolution.

    For lifted state ``s`` this computes

    ``D(s) = min_r [-tau log Phi(r) + d_F(r, s)]``

    in one multi-source march.  Combining ``D(s)`` with
    ``D(reverse(s))`` requires support from both contour directions; isolated
    intrinsic cell edges therefore do not receive the same lift as a weak
    interval between two coherent boundary plateaus.
    """

    saliency = np.clip(
        np.asarray(arc_saliency, dtype=np.float64), 1e-6, 1.0)
    state_arc = np.asarray(graph["state_arc"], dtype=np.int32)
    arc_count = len(saliency)
    if np.any(state_arc >= arc_count):
        raise ValueError("arc saliency does not cover the lifted graph")
    state_count = int(graph["state_count"])
    if state_count == 0:
        return {
            "saliency": saliency.copy(),
            "lift": np.zeros_like(saliency),
            "state_saliency": np.empty(0, dtype=np.float64),
            "tau": 1.0,
            "wavefront": sparse_elastica_voronoi(
                graph,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            ),
        }
    travel = np.asarray(
        graph["state_travel_cost"], dtype=np.float64)
    positive = travel[np.isfinite(travel) & (travel > 0.0)]
    local_scale = (
        float(np.median(positive)) if positive.size else 1.0)
    tau = max(float(continuation_scale), 1e-6) * local_scale
    source = np.arange(state_count, dtype=np.int32)
    wavefront = sparse_elastica_voronoi(
        graph,
        source,
        np.zeros(state_count, dtype=np.int32),
        source_potential=-tau * np.log(saliency[state_arc]),
    )
    arrival = np.exp(-np.asarray(
        wavefront["distance"], dtype=np.float64) / tau)
    reverse = np.asarray(graph["state_reverse"], dtype=np.int32)
    state_saliency = np.sqrt(
        np.maximum(arrival * arrival[reverse], 0.0))
    completed = saliency.copy()
    np.maximum.at(completed, state_arc, state_saliency)
    return {
        "saliency": np.clip(completed, 0.0, 1.0),
        "lift": np.maximum(completed - saliency, 0.0),
        "state_saliency": state_saliency,
        "tau": tau,
        "wavefront": wavefront,
    }


def project_pixel_field_to_arcs(
    topology: dict,
    field: np.ndarray,
) -> np.ndarray:
    """Average a pixel field along both sides of every embedded arc."""

    first, second = _edgel_pixel_samples(topology, field)
    sample = np.maximum(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    edgel_arc = np.asarray(topology["edgel"]["arc"], dtype=np.int32)
    arc_count = int(topology["arc"]["count"])
    length = np.bincount(edgel_arc, minlength=arc_count)
    return np.bincount(
        edgel_arc,
        weights=sample,
        minlength=arc_count,
    ) / np.maximum(length, 1)


def trace_sparse_elastica_path(
    result: dict,
    target_state: int,
) -> np.ndarray:
    """Trace one directed minimal path back through a completed march."""

    predecessor = np.asarray(result["predecessor"], dtype=np.int32)
    state = int(target_state)
    if state < 0 or state >= len(predecessor):
        raise ValueError("target state is outside the elastica graph")
    if not np.isfinite(result["distance"][state]):
        return np.empty(0, dtype=np.int32)
    path = [state]
    while predecessor[state] >= 0:
        state = int(predecessor[state])
        path.append(state)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def render_elastica_arcs(
    topology: dict,
    arc_values: np.ndarray,
) -> np.ndarray:
    """Rasterize one scalar per intrinsic arc onto its dual-grid edgels."""

    height, width = topology["shape"]
    out = np.zeros((height, width), dtype=np.float64)
    edgel = topology["edgel"]
    value = np.asarray(arc_values, dtype=np.float64)
    sample = value[np.asarray(edgel["arc"], dtype=np.int32)]
    stride = width + 1
    vertex = np.asarray(edgel["vertex_first"], dtype=np.int64)
    x = vertex % stride
    y = vertex // stride
    vertical = np.asarray(edgel["orientation"], dtype=np.int8) == 1
    for py, px in (
        (np.where(vertical, y, y - 1),
         np.where(vertical, x - 1, x)),
        (np.where(vertical, y, y),
         np.where(vertical, x, x)),
    ):
        valid = (
            (py >= 0) & (py < height)
            & (px >= 0) & (px < width)
        )
        np.maximum.at(out, (py[valid], px[valid]), sample[valid])
    return out
