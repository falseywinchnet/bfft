"""Ternary amodal continuation proposals on exact V3 contour ports.

A classical T junction is represented by a continuing cap contour and one
terminating stem.  Ports on the same cap contour propose two corresponding
background-side continuations.  The cap region remains explicit context and
is never inserted as a participant in the resulting background kernel.

All port pairs coexist.  Compatibility is the Gaussian of an atlas-wide
zero-centered covariance metric on geometric and signed-transition residuals;
there is no selected match, threshold, merge, or semantic object rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, QhullError

from experiments.v3_object_transport.junction_depth import _endpoint_tangent


@dataclass(frozen=True)
class ZeroWhitener:
    """Whitening about the physically distinguished zero-residual state."""

    basis: np.ndarray
    names: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        value = np.asarray(values, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != len(self.names):
            raise ValueError("amodal residual matrix does not match whitener")
        return np.ascontiguousarray(value @ self.basis)


def fit_zero_whitener(
    matrices: Iterable[np.ndarray],
    names: Iterable[str],
) -> ZeroWhitener:
    schema = tuple(names)
    values = np.vstack([np.asarray(value, dtype=np.float64) for value in matrices])
    if values.ndim != 2 or values.shape[1] != len(schema) or not len(values):
        raise ValueError("non-empty schema-aligned residual matrices required")
    second_moment = values.T @ values / max(len(values), 1)
    eigenvalue, eigenvector = np.linalg.eigh(second_moment)
    largest = float(np.max(eigenvalue, initial=0.0))
    numerical = (
        np.finfo(np.float64).eps
        * max(values.shape)
        * max(largest, 1.0)
    )
    retained = eigenvalue > numerical
    if not np.any(retained):
        raise ValueError("amodal atlas has no non-null residual mode")
    basis = eigenvector[:, retained] / np.sqrt(eigenvalue[retained])[None, :]
    return ZeroWhitener(np.ascontiguousarray(basis), schema)


def _append(columns, names, name, value) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        columns.append(array[:, None])
        names.append(name)
    else:
        columns.append(array)
        names.extend(f"{name}_{index}" for index in range(array.shape[1]))


def extract_amodal_ports(
    complex_: dict,
    contour: dict,
    junction_depth: dict,
    *,
    focus_arc_margin_first: np.ndarray | None = None,
    focus_arc_reliability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Extract one terminating stem and ordered background sides per T cap."""
    labels = np.asarray(complex_["labels"], dtype=np.int32)
    height, width = labels.shape
    topology = complex_["topology"]
    arc = topology["arc"]
    junction = topology["junction"]
    tangent = _endpoint_tangent(topology)
    node = complex_["node"]

    component_by_arc_owner = {}
    # The contour archive indexes the two arc orientations in the same order
    # as build_incidence_bundle: all first sides, then all second sides.
    arc_count = int(np.asarray(arc["count"]).item())
    component = np.asarray(contour["incidence_component"], dtype=np.int32)
    for identifier in range(arc_count):
        component_by_arc_owner[(
            identifier, int(arc["cell_first"][identifier]))] = int(
                component[identifier])
        component_by_arc_owner[(
            identifier, int(arc["cell_second"][identifier]))] = int(
                component[arc_count + identifier])

    transition_fields = [
        name for name in (
            "target_mean", "cartoon_mean", "texture_target_mean",
            "fused_cartoon_mean", "fused_texture_mean", "fused_residual_mean",
        ) if name in node
    ]
    jump_fields = [
        name for name in (
            "target", "cartoon", "texture_target", "boundary",
            "fused_cartoon", "fused_texture", "fused_residual",
        ) if name in complex_["arc"]
    ]
    record = {
        "junction": [], "vertex": [], "x": [], "y": [],
        "cap_region": [], "cap_component": [], "stem_arc": [],
        "left_region": [], "right_region": [],
        "stem_tangent_x": [], "stem_tangent_y": [],
        "cap_focus_margin": [], "cap_focus_reliability": [],
    }
    transition = {name: [] for name in transition_fields}
    cap_context = {name: [] for name in transition_fields}
    jump = {name: [] for name in jump_fields}
    cap_offset = junction_depth["cap_arc_offset"]
    quadrant_vector = np.asarray((
        (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5),
    ))
    for depth_identifier, (junction_identifier, vertex, cap) in enumerate(zip(
        junction_depth["junction"],
        junction_depth["vertex"],
        junction_depth["cap_region"],
    )):
        start = int(junction["arc_offset"][junction_identifier])
        stop = int(junction["arc_offset"][junction_identifier + 1])
        arcs = np.unique(junction["arc"][start:stop])
        stems = [
            int(identifier) for identifier in arcs
            if int(cap) not in (
                int(arc["cell_first"][identifier]),
                int(arc["cell_second"][identifier]),
            )
        ]
        if len(stems) != 1:
            continue
        stem = stems[0]
        direction = tangent.get((stem, int(vertex)))
        if direction is None or np.linalg.norm(direction) <= 1e-30:
            continue
        direction = direction / np.linalg.norm(direction)
        x = int(junction_depth["x"][depth_identifier])
        y = int(junction_depth["y"][depth_identifier])
        sectors = np.asarray((
            labels[y - 1, x - 1], labels[y - 1, x],
            labels[y, x], labels[y, x - 1],
        ), dtype=np.int32)
        side_regions = np.asarray((
            int(arc["cell_first"][stem]),
            int(arc["cell_second"][stem]),
        ), dtype=np.int32)
        cross = []
        for region in side_regions:
            vectors = quadrant_vector[sectors == region]
            vector = np.mean(vectors, axis=0)
            cross.append(float(
                direction[0] * vector[1] - direction[1] * vector[0]))
        order = np.argsort(cross)
        left = int(side_regions[order[0]])
        right = int(side_regions[order[-1]])

        cap_start = int(cap_offset[depth_identifier])
        cap_stop = int(cap_offset[depth_identifier + 1])
        cap_arcs = junction_depth["cap_arc"][cap_start:cap_stop]
        cap_components = np.unique([
            component_by_arc_owner[(int(identifier), int(cap))]
            for identifier in cap_arcs
        ])
        if len(cap_components) != 1:
            continue
        focus_margin = 0.0
        focus_reliability = 0.0
        if focus_arc_margin_first is not None and focus_arc_reliability is not None:
            signed = []
            reliability = []
            for identifier in cap_arcs:
                identifier = int(identifier)
                sign = 1.0 if int(arc["cell_first"][identifier]) == int(cap) else -1.0
                signed.append(sign * float(focus_arc_margin_first[identifier]))
                reliability.append(float(focus_arc_reliability[identifier]))
            reliability = np.asarray(reliability)
            focus_reliability = float(np.mean(reliability))
            focus_margin = float(np.sum(reliability * signed) / max(
                float(np.sum(reliability)), 1e-30))

        record["junction"].append(int(junction_identifier))
        record["vertex"].append(int(vertex))
        record["x"].append(x)
        record["y"].append(y)
        record["cap_region"].append(int(cap))
        record["cap_component"].append(int(cap_components[0]))
        record["stem_arc"].append(stem)
        record["left_region"].append(left)
        record["right_region"].append(right)
        record["stem_tangent_x"].append(float(direction[0]))
        record["stem_tangent_y"].append(float(direction[1]))
        record["cap_focus_margin"].append(focus_margin)
        record["cap_focus_reliability"].append(focus_reliability)
        for name in transition_fields:
            transition[name].append(node[name][right] - node[name][left])
            cap_context[name].append(node[name][int(cap)])
        for name in jump_fields:
            jump[name].append(float(complex_["arc"][name][stem]))
    result = {
        name: np.asarray(value, dtype=(
            np.float64 if name.startswith("stem_tangent")
            or name.startswith("cap_focus") else np.int32
        )) for name, value in record.items()
    }
    result.update({
        f"transition_{name}": np.asarray(value, dtype=np.float64)
        for name, value in transition.items()
    })
    result.update({
        f"cap_context_{name}": np.asarray(value, dtype=np.float64)
        for name, value in cap_context.items()
    })
    result.update({
        f"jump_{name}": np.asarray(value, dtype=np.float64)
        for name, value in jump.items()
    })
    return result


def amodal_pair_residuals(
    ports: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    candidate_mode: str = "same_component",
    port_depth_agreement: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
    """Enumerate all port pairs on each occluder contour and their residuals."""
    candidates: set[tuple[int, int]] = set()
    for component in np.unique(ports["cap_component"]):
        members = np.flatnonzero(ports["cap_component"] == component)
        for first, second in combinations(members.tolist(), 2):
            candidates.add((first, second))
    if candidate_mode == "contour_delaunay":
        if len(ports["x"]) >= 3:
            point = np.column_stack((ports["x"], ports["y"]))
            try:
                triangulation = Delaunay(point)
                for simplex in triangulation.simplices:
                    for first, second in combinations(simplex.tolist(), 2):
                        candidates.add(tuple(sorted((int(first), int(second)))))
            except QhullError:
                # Degenerate collinear controls retain exact contour pairs.
                pass
    elif candidate_mode != "same_component":
        raise ValueError(f"unknown amodal candidate mode {candidate_mode!r}")
    ordered = sorted(candidates)
    first = np.asarray([value[0] for value in ordered], dtype=np.int32)
    second = np.asarray([value[1] for value in ordered], dtype=np.int32)
    label_field = np.asarray(labels, dtype=np.int32)
    height, width = label_field.shape
    diagonal = max(float(np.hypot(height, width)), 1.0)
    dx = (ports["x"][second] - ports["x"][first]) / diagonal
    dy = (ports["y"][second] - ports["y"][first]) / diagonal
    distance = np.hypot(dx, dy)
    unit_x = np.divide(dx, distance, out=np.zeros_like(dx), where=distance > 0)
    unit_y = np.divide(dy, distance, out=np.zeros_like(dy), where=distance > 0)
    t1 = np.column_stack((
        ports["stem_tangent_x"][first], ports["stem_tangent_y"][first]))
    t2 = np.column_stack((
        ports["stem_tangent_x"][second], ports["stem_tangent_y"][second]))
    unit = np.column_stack((unit_x, unit_y))
    tangent_swap = np.clip(
        0.5 * (1.0 - np.sum(t1 * t2, axis=1)), 0.0, 1.0)
    first_faces_gap = np.clip(
        0.5 * (1.0 - np.sum(t1 * unit, axis=1)), 0.0, 1.0)
    second_faces_gap = np.clip(
        0.5 * (1.0 + np.sum(t2 * unit, axis=1)), 0.0, 1.0)
    orientation_evidence = tangent_swap * first_faces_gap * second_faces_gap
    crossing_fraction = np.zeros(len(first), dtype=np.float64)
    for pair_identifier, (first_port, second_port) in enumerate(zip(first, second)):
        x0, y0 = int(ports["x"][first_port]), int(ports["y"][first_port])
        x1, y1 = int(ports["x"][second_port]), int(ports["y"][second_port])
        sample_count = max(abs(x1 - x0), abs(y1 - y0)) + 1
        if sample_count <= 2:
            continue
        xs = np.rint(np.linspace(x0, x1, sample_count))[1:-1].astype(np.int32)
        ys = np.rint(np.linspace(y0, y1, sample_count))[1:-1].astype(np.int32)
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
        cap = np.asarray((
            ports["cap_region"][first_port],
            ports["cap_region"][second_port],
        ), dtype=np.int32)
        crossing_fraction[pair_identifier] = float(np.mean(
            np.isin(label_field[ys, xs], cap)))
    columns = []
    names = []
    _append(columns, names, "geometry_displacement", np.column_stack((dx, dy)))
    _append(columns, names, "geometry_tangent_opposition", t1 + t2)
    _append(columns, names, "geometry_first_gap_alignment", t1 + unit)
    _append(columns, names, "geometry_second_gap_alignment", t2 - unit)
    _append(
        columns, names, "geometry_cap_crossing_deficit",
        1.0 - crossing_fraction)
    for name in sorted(key for key in ports if key.startswith("transition_")):
        _append(columns, names, f"content_{name}_opposition", ports[name][first] + ports[name][second])
    for name in sorted(key for key in ports if key.startswith("cap_context_")):
        _append(columns, names, f"context_{name}_difference", ports[name][first] - ports[name][second])
    for name in sorted(key for key in ports if key.startswith("jump_")):
        _append(columns, names, f"content_{name}_difference", ports[name][first] - ports[name][second])
    pair = {
        "first_port": first,
        "second_port": second,
        "cap_region_first": ports["cap_region"][first],
        "cap_region_second": ports["cap_region"][second],
        "cap_component_first": ports["cap_component"][first],
        "cap_component_second": ports["cap_component"][second],
        "first_left_region": ports["left_region"][first],
        "first_right_region": ports["right_region"][first],
        "second_left_region": ports["left_region"][second],
        "second_right_region": ports["right_region"][second],
        "distance": distance,
        "cap_crossing_fraction": crossing_fraction,
        "tangent_swap_fraction": tangent_swap,
        "first_faces_gap": first_faces_gap,
        "second_faces_gap": second_faces_gap,
        "orientation_evidence": orientation_evidence,
        "focus_margin_first": ports["cap_focus_margin"][first],
        "focus_margin_second": ports["cap_focus_margin"][second],
        "focus_reliability_first": ports["cap_focus_reliability"][first],
        "focus_reliability_second": ports["cap_focus_reliability"][second],
    }
    if port_depth_agreement is not None:
        agreement = np.asarray(port_depth_agreement, dtype=np.float64)
        pair["depth_agreement_first"] = agreement[first]
        pair["depth_agreement_second"] = agreement[second]
    return pair, np.ascontiguousarray(np.column_stack(columns)), tuple(names)


def build_amodal_transport(
    pair: dict[str, np.ndarray],
    residual: np.ndarray,
    whitener: ZeroWhitener,
    region_count: int,
) -> dict:
    transformed = whitener.transform(residual)
    squared = np.mean(transformed * transformed, axis=1)
    compatibility = np.exp(-0.5 * squared)
    result = build_weighted_amodal_transport(
        pair, compatibility, region_count)
    result["whitened_squared_residual"] = squared
    return result


def build_weighted_amodal_transport(
    pair: dict[str, np.ndarray],
    compatibility: np.ndarray,
    region_count: int,
) -> dict:
    """Materialize background participation from an explicit measured weight."""
    compatibility = np.asarray(compatibility, dtype=np.float64)
    if compatibility.shape != pair["first_port"].shape:
        raise ValueError("one amodal weight is required per port pair")
    row = []
    column = []
    data = []
    correspondence = (
        ("first_left_region", "second_right_region"),
        ("first_right_region", "second_left_region"),
    )
    for pair_identifier in range(len(compatibility)):
        value = float(np.sqrt(compatibility[pair_identifier]))
        for first_name, second_name in correspondence:
            row_identifier = len(row) // 2
            row.extend((row_identifier, row_identifier))
            column.extend((
                int(pair[first_name][pair_identifier]),
                int(pair[second_name][pair_identifier]),
            ))
            data.extend((value, value))
    participation = sparse.csr_matrix(
        (data, (row, column)),
        shape=(2 * len(compatibility), region_count),
    )
    norm = np.sqrt(np.asarray(participation.power(2).sum(axis=0)).ravel())
    normalized = participation @ sparse.diags(np.divide(
        1.0, norm, out=np.zeros_like(norm), where=norm > 0.0))
    kernel = (normalized.T @ normalized).toarray()
    kernel = np.clip(0.5 * (kernel + kernel.T), 0.0, 1.0)
    return {
        "compatibility": compatibility,
        "participation": participation,
        "region_kernel": kernel,
    }


def summarize_amodal_transport(
    ports: dict,
    pair: dict,
    transport: dict,
) -> dict:
    compatibility = transport["compatibility"]
    return {
        "ports": int(len(ports["junction"])),
        "occluder_contours_with_multiple_ports": int(np.count_nonzero(
            np.bincount(ports["cap_component"]) > 1)),
        "port_pairs": int(len(pair["first_port"])),
        "background_correspondences": int(2 * len(pair["first_port"])),
        "compatibility_quantiles": (
            [float(value) for value in np.quantile(
                compatibility, (0.0, 0.25, 0.5, 0.75, 1.0))]
            if len(compatibility) else [0.0] * 5
        ),
    }
