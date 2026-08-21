"""Relation-lifted incidence bundle over the embedded V3 region complex.

An object continuation does not live on a region name alone.  It arrives at a
region through a particular connected boundary arc with a particular outside
state and tangent.  This module constructs that lifted state and the exact
junction continuations available to it.  It measures; it does not select.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _endpoint_tangent(topology: dict) -> dict[tuple[int, int], np.ndarray]:
    arc = topology["arc"]
    result: dict[tuple[int, int], np.ndarray] = {}
    for identifier in range(int(arc["count"])):
        start = int(arc["endpoint_offset"][identifier])
        stop = int(arc["endpoint_offset"][identifier + 1])
        for cursor in range(start, stop):
            result[(identifier, int(arc["endpoint_vertex"][cursor]))] = (
                np.asarray((
                    arc["endpoint_tangent_x"][cursor],
                    arc["endpoint_tangent_y"][cursor],
                ), dtype=np.float64)
            )
    return result


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-30:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def build_incidence_bundle(complex_: dict) -> dict[str, dict[str, np.ndarray]]:
    """Lift each embedded arc to two region-relative incidence states."""
    arc = complex_["arc"]
    node = complex_["node"]
    topology = complex_["topology"]
    count = len(arc["cell_first"])
    arc_id = np.tile(np.arange(count, dtype=np.int32), 2)
    region = np.concatenate((arc["cell_first"], arc["cell_second"]))
    outside = np.concatenate((arc["cell_second"], arc["cell_first"]))
    side = np.concatenate((
        np.ones(count, dtype=np.float64),
        -np.ones(count, dtype=np.float64),
    ))
    incidence = {
        "arc": arc_id,
        "region": region.astype(np.int32, copy=False),
        "outside": outside.astype(np.int32, copy=False),
        "side": side,
        "length": np.tile(arc["length"], 2),
        "closed": np.tile(arc["closed"], 2),
        "normal_x": np.tile(arc["normal_x"], 2) * side,
        "normal_y": np.tile(arc["normal_y"], 2) * side,
        "target_transition": (
            node["target_mean"][outside] - node["target_mean"][region]
        ),
        "cartoon_transition": (
            node["cartoon_mean"][outside] - node["cartoon_mean"][region]
        ),
        "texture_rms_transition": (
            node["texture_target_rms"][outside]
            - node["texture_target_rms"][region]
        ),
        "boundary_confidence": np.tile(arc["boundary"], 2),
        "literal_target_jump": np.tile(arc["target"], 2),
        "literal_cartoon_jump": np.tile(arc["cartoon"], 2),
        "literal_texture_jump": np.tile(arc["texture_target"], 2),
    }
    for name in ("cartoon", "texture", "residual"):
        node_name = f"fused_{name}_mean"
        arc_name = f"fused_{name}"
        if node_name in node and arc_name in arc:
            incidence[f"fused_{name}_transition"] = (
                node[node_name][outside] - node[node_name][region]
            )
            incidence[f"literal_fused_{name}_jump"] = np.tile(
                arc[arc_name], 2)

    tangent = _endpoint_tangent(topology)
    junction_id: list[int] = []
    vertex_record: list[int] = []
    region_record: list[int] = []
    first_arc_record: list[int] = []
    second_arc_record: list[int] = []
    first_outside_record: list[int] = []
    second_outside_record: list[int] = []
    continuation_record: list[float] = []
    transition_cosine_record: list[float] = []
    junction = topology["junction"]
    for identifier, vertex in enumerate(junction["vertex"]):
        start = int(junction["arc_offset"][identifier])
        stop = int(junction["arc_offset"][identifier + 1])
        arcs = np.unique(junction["arc"][start:stop])
        for first_arc, second_arc in combinations(arcs.tolist(), 2):
            first_regions = {
                int(arc["cell_first"][first_arc]),
                int(arc["cell_second"][first_arc]),
            }
            second_regions = {
                int(arc["cell_first"][second_arc]),
                int(arc["cell_second"][second_arc]),
            }
            for shared in sorted(first_regions & second_regions):
                first_outside = next(iter(first_regions - {shared}))
                second_outside = next(iter(second_regions - {shared}))
                first_tangent = tangent.get((first_arc, int(vertex)))
                second_tangent = tangent.get((second_arc, int(vertex)))
                if first_tangent is None or second_tangent is None:
                    continue
                first_transition = (
                    node["target_mean"][first_outside]
                    - node["target_mean"][shared]
                )
                second_transition = (
                    node["target_mean"][second_outside]
                    - node["target_mean"][shared]
                )
                junction_id.append(identifier)
                vertex_record.append(int(vertex))
                region_record.append(shared)
                first_arc_record.append(first_arc)
                second_arc_record.append(second_arc)
                first_outside_record.append(first_outside)
                second_outside_record.append(second_outside)
                # Endpoint tangents point inward. Opposite tangents are a
                # straight continuation through the junction.
                continuation_record.append(
                    max(0.0, -_cosine(first_tangent, second_tangent)))
                transition_cosine_record.append(
                    _cosine(first_transition, second_transition))

    continuation = {
        "junction": np.asarray(junction_id, dtype=np.int32),
        "vertex": np.asarray(vertex_record, dtype=np.int64),
        "region": np.asarray(region_record, dtype=np.int32),
        "first_arc": np.asarray(first_arc_record, dtype=np.int32),
        "second_arc": np.asarray(second_arc_record, dtype=np.int32),
        "first_outside": np.asarray(first_outside_record, dtype=np.int32),
        "second_outside": np.asarray(second_outside_record, dtype=np.int32),
        "same_outside": np.asarray(first_outside_record, dtype=np.int32)
        == np.asarray(second_outside_record, dtype=np.int32),
        "tangent_continuation": np.asarray(
            continuation_record, dtype=np.float64),
        "transition_cosine": np.asarray(
            transition_cosine_record, dtype=np.float64),
    }
    return {"incidence": incidence, "continuation": continuation}


def summarize_incidence_bundle(bundle: dict) -> dict:
    incidence = bundle["incidence"]
    continuation = bundle["continuation"]
    return {
        "directed_incidences": int(len(incidence["arc"])),
        "closed_incidences": int(np.count_nonzero(incidence["closed"])),
        "junction_continuations": int(len(continuation["junction"])),
        "same_outside_continuations": int(np.count_nonzero(
            continuation["same_outside"])),
        "tangent_continuation_quantiles": (
            [float(value) for value in np.quantile(
                continuation["tangent_continuation"],
                (0.0, 0.25, 0.5, 0.75, 1.0),
            )]
            if len(continuation["junction"]) else [0.0] * 5
        ),
        "transition_cosine_quantiles": (
            [float(value) for value in np.quantile(
                continuation["transition_cosine"],
                (0.0, 0.25, 0.5, 0.75, 1.0),
            )]
            if len(continuation["junction"]) else [0.0] * 5
        ),
    }
