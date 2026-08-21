"""Oriented T-junction evidence on the exact V3 embedded lattice.

The module records sector multiplicity, cyclic adjacency, the two cap-side
arcs, their continuation, and the agreement of their signed content
transitions.  It does not decide an object or collapse the measurements to a
depth partition.
"""

from __future__ import annotations

import numpy as np


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.ravel(np.asarray(first, dtype=np.float64))
    second = np.ravel(np.asarray(second, dtype=np.float64))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-30:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _endpoint_tangent(topology: dict) -> dict[tuple[int, int], np.ndarray]:
    arc = topology["arc"]
    result = {}
    count = int(np.asarray(arc["count"]).item())
    for identifier in range(count):
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


def build_junction_depth(complex_: dict) -> dict[str, np.ndarray]:
    labels = np.asarray(complex_["labels"], dtype=np.int32)
    topology = complex_["topology"]
    junction = topology["junction"]
    tangent = _endpoint_tangent(topology)
    height, width = labels.shape

    junction_record = []
    vertex_record = []
    x_record = []
    y_record = []
    arc_count_record = []
    sector_count_record = []
    tangent_low_record = []
    tangent_high_record = []
    tangent_anisotropy_record = []
    cap_region_record = []
    cap_sector_count_record = []
    cap_adjacent_pairs_record = []
    cap_arc_offset = [0]
    cap_arc_record = []
    cap_tangent_cosine_record = []
    cap_tangent_continuation_record = []
    transition_fields = [
        name for name in (
            "target_mean",
            "cartoon_mean",
            "texture_target_mean",
            "fused_cartoon_mean",
            "fused_texture_mean",
            "fused_residual_mean",
        )
        if name in complex_["node"]
    ]
    transition_cosine_record = {name: [] for name in transition_fields}
    other_region_offset = [0]
    other_region_record = []

    count = int(np.asarray(junction["count"]).item())
    for identifier in range(count):
        vertex = int(junction["vertex"][identifier])
        x = int(junction["x"][identifier])
        y = int(junction["y"][identifier])
        if x <= 0 or x >= width or y <= 0 or y >= height:
            continue
        start = int(junction["arc_offset"][identifier])
        stop = int(junction["arc_offset"][identifier + 1])
        arcs = np.unique(junction["arc"][start:stop])
        directions = [
            tangent[(int(arc), vertex)]
            for arc in arcs
            if (int(arc), vertex) in tangent
        ]
        if len(directions) < 3:
            continue
        direction = np.asarray(directions, dtype=np.float64)
        tensor = direction.T @ direction
        eigenvalue = np.linalg.eigvalsh(tensor)
        tangent_sum = max(float(np.sum(eigenvalue)), 1e-30)
        tangent_anisotropy = float(
            (eigenvalue[-1] - eigenvalue[0]) / tangent_sum)

        # Clockwise sectors around the dual-grid vertex.
        sectors = np.asarray((
            labels[y - 1, x - 1],
            labels[y - 1, x],
            labels[y, x],
            labels[y, x - 1],
        ), dtype=np.int32)
        unique, population = np.unique(sectors, return_counts=True)
        for cap, cap_population in zip(unique, population):
            equal = sectors == cap
            adjacent_pairs = int(np.count_nonzero(
                equal & np.roll(equal, -1)))
            if adjacent_pairs == 0:
                continue
            others = np.unique(sectors[~equal]).astype(np.int32)
            if not len(others):
                continue
            cap_arcs = [
                int(arc_identifier)
                for arc_identifier in arcs
                if int(cap) in (
                    int(topology["arc"]["cell_first"][arc_identifier]),
                    int(topology["arc"]["cell_second"][arc_identifier]),
                )
            ]
            cap_tangents = [
                tangent[(arc_identifier, vertex)]
                for arc_identifier in cap_arcs
                if (arc_identifier, vertex) in tangent
            ]
            if len(cap_arcs) == 2 and len(cap_tangents) == 2:
                tangent_cosine = _cosine(cap_tangents[0], cap_tangents[1])
            else:
                tangent_cosine = 0.0
            transition_cosines = {}
            for name in transition_fields:
                values = []
                for arc_identifier in cap_arcs:
                    first = int(topology["arc"]["cell_first"][arc_identifier])
                    second = int(topology["arc"]["cell_second"][arc_identifier])
                    outside = second if first == int(cap) else first
                    values.append(
                        complex_["node"][name][outside]
                        - complex_["node"][name][int(cap)]
                    )
                transition_cosines[name] = (
                    _cosine(values[0], values[1])
                    if len(values) == 2 else 0.0
                )
            junction_record.append(identifier)
            vertex_record.append(vertex)
            x_record.append(x)
            y_record.append(y)
            arc_count_record.append(len(arcs))
            sector_count_record.append(len(unique))
            tangent_low_record.append(float(eigenvalue[0]))
            tangent_high_record.append(float(eigenvalue[-1]))
            tangent_anisotropy_record.append(tangent_anisotropy)
            cap_region_record.append(int(cap))
            cap_sector_count_record.append(int(cap_population))
            cap_adjacent_pairs_record.append(adjacent_pairs)
            cap_arc_record.extend(cap_arcs)
            cap_arc_offset.append(len(cap_arc_record))
            cap_tangent_cosine_record.append(tangent_cosine)
            cap_tangent_continuation_record.append(max(0.0, -tangent_cosine))
            for name in transition_fields:
                transition_cosine_record[name].append(
                    transition_cosines[name])
            other_region_record.extend(others.tolist())
            other_region_offset.append(len(other_region_record))

    return {
        "junction": np.asarray(junction_record, dtype=np.int32),
        "vertex": np.asarray(vertex_record, dtype=np.int64),
        "x": np.asarray(x_record, dtype=np.int32),
        "y": np.asarray(y_record, dtype=np.int32),
        "arc_count": np.asarray(arc_count_record, dtype=np.int32),
        "sector_count": np.asarray(sector_count_record, dtype=np.int32),
        "tangent_low": np.asarray(tangent_low_record, dtype=np.float64),
        "tangent_high": np.asarray(tangent_high_record, dtype=np.float64),
        "tangent_anisotropy": np.asarray(
            tangent_anisotropy_record, dtype=np.float64),
        "cap_region": np.asarray(cap_region_record, dtype=np.int32),
        "cap_sector_count": np.asarray(
            cap_sector_count_record, dtype=np.int32),
        "cap_adjacent_pairs": np.asarray(
            cap_adjacent_pairs_record, dtype=np.int32),
        "cap_arc_offset": np.asarray(cap_arc_offset, dtype=np.int64),
        "cap_arc": np.asarray(cap_arc_record, dtype=np.int32),
        "cap_tangent_cosine": np.asarray(
            cap_tangent_cosine_record, dtype=np.float64),
        "cap_tangent_continuation": np.asarray(
            cap_tangent_continuation_record, dtype=np.float64),
        "other_region_offset": np.asarray(
            other_region_offset, dtype=np.int64),
        "other_region": np.asarray(other_region_record, dtype=np.int32),
        **{
            f"cap_{name}_transition_cosine": np.asarray(
                transition_cosine_record[name], dtype=np.float64)
            for name in transition_fields
        },
    }


def summarize_junction_depth(depth: dict) -> dict:
    sector_count = depth["sector_count"]
    cap_count = depth["cap_sector_count"]
    anisotropy = depth["tangent_anisotropy"]
    return {
        "cap_records": int(len(depth["junction"])),
        "three_sector_caps": int(np.count_nonzero(sector_count == 3)),
        "two_quadrant_caps": int(np.count_nonzero(cap_count == 2)),
        "classical_t_records": int(np.count_nonzero(
            (sector_count == 3) & (cap_count == 2))),
        "tangent_anisotropy_quantiles": (
            [float(value) for value in np.quantile(
                anisotropy, (0.0, 0.25, 0.5, 0.75, 1.0))]
            if len(anisotropy) else [0.0] * 5
        ),
        "cap_tangent_continuation_quantiles": (
            [float(value) for value in np.quantile(
                depth["cap_tangent_continuation"],
                (0.0, 0.25, 0.5, 0.75, 1.0))]
            if len(depth["cap_tangent_continuation"]) else [0.0] * 5
        ),
    }
