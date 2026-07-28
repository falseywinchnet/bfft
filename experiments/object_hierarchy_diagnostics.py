#!/usr/bin/env python3
"""Read-only diagnostics for failures of the transport-object hierarchy.

The object hierarchy is deliberately kept separate from this module.  These
helpers do not change seeds, barriers, or labels; they expose what the current
hierarchy has decided.

The central diagnostic is the *connected material quotient*.  Cell interfaces
whose measured barrier is below a chosen waterline are contracted, without
looking at object IDs.  The resulting connected components are provisional
pieces of material.  Comparing those components with the inferred object IDs
distinguishes two opposite failures:

* one material component owned by several objects: likely over-segmentation;
* one object containing several material components: likely under-segmentation.

This is not proposed as a replacement segmentation algorithm.  It is an
instrument for finding exactly which interface evidence is missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_NODE_MEAN_FIELDS = (
    "node_lab",
    "node_cartoon",
    "node_glass",
    "node_texture",
    "node_measure",
    "node_energy",
    "node_null",
    "node_qxx",
    "node_qxy",
    "node_qyy",
)


def _dense_object_ids(objects: dict) -> tuple[np.ndarray, int]:
    object_id = np.asarray(
        objects["object_id_per_cell"], dtype=np.int32)
    if object_id.ndim != 1 or np.any(object_id < 0):
        raise ValueError("object_id_per_cell must be a non-negative vector")
    count = int(object_id.max(initial=-1)) + 1
    if count == 0:
        raise ValueError("the hierarchy contains no objects")
    return object_id, count


def _weighted_node_mean(
    object_id: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    denominator = np.bincount(
        object_id, weights=weights, minlength=count)
    if value.ndim == 1:
        numerator = np.bincount(
            object_id, weights=weights * value, minlength=count)
        return numerator / np.maximum(denominator, 1e-30)
    columns = [
        np.bincount(
            object_id,
            weights=weights * value[:, channel],
            minlength=count,
        ) / np.maximum(denominator, 1e-30)
        for channel in range(value.shape[1])
    ]
    return np.column_stack(columns)


def object_means(objects: dict) -> dict[str, np.ndarray]:
    """Return area-weighted summaries for every inferred object."""
    graph = objects["graph"]
    object_id, count = _dense_object_ids(objects)
    area = np.asarray(graph["area"], dtype=np.float64)
    if len(area) != len(object_id):
        raise ValueError("graph cells and object IDs disagree")

    pixel_area = np.bincount(
        object_id, weights=area, minlength=count)
    result: dict[str, np.ndarray] = {
        "object_id": np.arange(count, dtype=np.int32),
        "cell_count": np.bincount(
            object_id, minlength=count).astype(np.int32),
        "pixel_area": pixel_area,
        "area_fraction": pixel_area / max(float(pixel_area.sum()), 1.0),
        "centroid_x": _weighted_node_mean(
            object_id, graph["node_x"], area, count),
        "centroid_y": _weighted_node_mean(
            object_id, graph["node_y"], area, count),
    }
    for name in _NODE_MEAN_FIELDS:
        if name in graph:
            result[name.removeprefix("node_")] = _weighted_node_mean(
                object_id, graph[name], area, count)

    selected = np.asarray(objects.get("selected_seeds", []), dtype=np.int32)
    seed_cell = np.full(count, -1, dtype=np.int32)
    seed_cell[:min(count, len(selected))] = selected[:count]
    result["seed_cell"] = seed_cell
    if "seed_score" in objects:
        seed_score = np.asarray(objects["seed_score"], dtype=np.float64)
        result["mean_seed_score"] = _weighted_node_mean(
            object_id, seed_score, area, count)
        result["seed_score"] = np.where(
            seed_cell >= 0, seed_score[np.maximum(seed_cell, 0)], np.nan)
    return result


def object_adjacency(objects: dict) -> dict[str, np.ndarray]:
    """Aggregate literal inter-object interfaces by unordered object pair.

    Scalar evidence is averaged using literal interface length, so a one-pixel
    spike cannot visually dominate a long quiet boundary.  Min/max barrier are
    retained to reveal a gap hidden by that mean.
    """
    graph = objects["graph"]
    edge = graph["edge"]
    evidence = objects["evidence"]
    object_id, count = _dense_object_ids(objects)
    cell_first = np.asarray(edge["first"], dtype=np.int32)
    cell_second = np.asarray(edge["second"], dtype=np.int32)
    first = object_id[cell_first]
    second = object_id[cell_second]
    crossing = first != second
    low = np.minimum(first[crossing], second[crossing])
    high = np.maximum(first[crossing], second[crossing])
    length = np.asarray(edge["length"], dtype=np.float64)[crossing]
    if not np.any(crossing):
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "first": empty_i,
            "second": empty_i.copy(),
            "interface_count": empty_i.copy(),
            "length": empty_f,
            "barrier_min": empty_f.copy(),
            "barrier_max": empty_f.copy(),
        }

    key = low.astype(np.int64) * count + high
    order = np.argsort(key, kind="stable")
    ordered_key = key[order]
    start = np.r_[0, 1 + np.flatnonzero(
        ordered_key[1:] != ordered_key[:-1])]
    unique_key = ordered_key[start]
    ordered_length = length[order]
    total_length = np.add.reduceat(ordered_length, start)
    result: dict[str, np.ndarray] = {
        "first": (unique_key // count).astype(np.int32),
        "second": (unique_key % count).astype(np.int32),
        "interface_count": np.diff(
            np.r_[start, len(order)]).astype(np.int32),
        "length": total_length,
    }
    for name, values in evidence.items():
        if name == "affinity":
            continue
        value = np.asarray(values, dtype=np.float64)[crossing][order]
        result[name] = (
            np.add.reduceat(value * ordered_length, start)
            / np.maximum(total_length, 1e-30)
        )
        if name == "barrier":
            result["barrier_min"] = np.minimum.reduceat(value, start)
            result["barrier_max"] = np.maximum.reduceat(value, start)
    return result


def connected_material_quotient(
    objects: dict,
    *,
    barrier_threshold: float = 0.35,
) -> dict[str, Any]:
    """Contract low-barrier interfaces and compare material with object IDs.

    ``barrier_threshold`` is a diagnostic waterline, not a segmentation
    control.  A lower value demands stronger evidence that adjacent cells are
    the same material.
    """
    graph = objects["graph"]
    edge = graph["edge"]
    barrier = np.asarray(objects["evidence"]["barrier"], dtype=np.float64)
    cells = int(graph["cells"])
    first = np.asarray(edge["first"], dtype=np.int32)
    second = np.asarray(edge["second"], dtype=np.int32)
    area = np.asarray(graph["area"], dtype=np.float64)
    if len(barrier) != len(first):
        raise ValueError("barrier and graph edge arrays disagree")

    parent = np.arange(cells, dtype=np.int32)
    size = np.ones(cells, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            old = int(parent[node])
            parent[node] = root
            node = old
        return root

    for edge_index in np.flatnonzero(barrier <= float(barrier_threshold)):
        a = find(int(first[edge_index]))
        b = find(int(second[edge_index]))
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    roots = np.fromiter(
        (find(cell) for cell in range(cells)),
        dtype=np.int32,
        count=cells,
    )
    _, material_id = np.unique(roots, return_inverse=True)
    material_id = material_id.astype(np.int32)
    material_count = int(material_id.max(initial=-1)) + 1
    object_id, object_count = _dense_object_ids(objects)

    pair_key = (
        object_id.astype(np.int64) * material_count
        + material_id.astype(np.int64)
    )
    order = np.argsort(pair_key, kind="stable")
    ordered_key = pair_key[order]
    start = np.r_[0, 1 + np.flatnonzero(
        ordered_key[1:] != ordered_key[:-1])]
    unique_key = ordered_key[start]
    overlap_area = np.add.reduceat(area[order], start)
    pair_object = (unique_key // material_count).astype(np.int32)
    pair_material = (unique_key % material_count).astype(np.int32)

    object_area = np.bincount(
        object_id, weights=area, minlength=object_count)
    material_area = np.bincount(
        material_id, weights=area, minlength=material_count)
    object_component_count = np.bincount(
        pair_object, minlength=object_count).astype(np.int32)
    material_object_count = np.bincount(
        pair_material, minlength=material_count).astype(np.int32)
    object_dominant_area = np.zeros(object_count, dtype=np.float64)
    material_dominant_area = np.zeros(material_count, dtype=np.float64)
    np.maximum.at(object_dominant_area, pair_object, overlap_area)
    np.maximum.at(material_dominant_area, pair_material, overlap_area)

    overlap_object_fraction = (
        overlap_area / np.maximum(object_area[pair_object], 1e-30)
    )
    overlap_material_fraction = (
        overlap_area / np.maximum(material_area[pair_material], 1e-30)
    )
    object_entropy = np.bincount(
        pair_object,
        weights=-overlap_object_fraction
        * np.log(np.maximum(overlap_object_fraction, 1e-30)),
        minlength=object_count,
    )
    material_entropy = np.bincount(
        pair_material,
        weights=-overlap_material_fraction
        * np.log(np.maximum(overlap_material_fraction, 1e-30)),
        minlength=material_count,
    )

    return {
        "barrier_threshold": float(barrier_threshold),
        "material_count": material_count,
        "material_id_per_cell": material_id,
        # One means an object is materially connected at this waterline.
        "object_connected_material_quotient": (
            object_dominant_area / np.maximum(object_area, 1e-30)
        ),
        "object_material_component_count": object_component_count,
        "object_material_entropy": object_entropy,
        # One means a material component has not been split among objects.
        "material_object_quotient": (
            material_dominant_area / np.maximum(material_area, 1e-30)
        ),
        "material_object_count": material_object_count,
        "material_object_entropy": material_entropy,
        "object_area": object_area,
        "material_area": material_area,
        "overlap": {
            "object_id": pair_object,
            "material_id": pair_material,
            "pixel_area": overlap_area,
            "object_fraction": overlap_object_fraction,
            "material_fraction": overlap_material_fraction,
        },
    }


def object_boundary_context(
    objects: dict,
    object_id: int,
    *,
    adjacency: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return this object's neighbours, strongest walls, and weakest leaks."""
    adjacency = object_adjacency(objects) if adjacency is None else adjacency
    first = np.asarray(adjacency["first"], dtype=np.int32)
    second = np.asarray(adjacency["second"], dtype=np.int32)
    selected = (first == object_id) | (second == object_id)
    indices = np.flatnonzero(selected)
    neighbour = np.where(
        first[indices] == object_id,
        second[indices],
        first[indices],
    )
    # Weakest mean barrier first: these are the most plausible missing merges.
    order = np.lexsort((
        -np.asarray(adjacency["length"])[indices],
        np.asarray(adjacency["barrier"])[indices],
    ))
    result = {
        "edge_index": indices[order].astype(np.int32),
        "neighbour": neighbour[order].astype(np.int32),
    }
    for name, value in adjacency.items():
        if name not in ("first", "second"):
            result[name] = np.asarray(value)[indices][order]
    return result


def selection_report(
    result: dict,
    objects: dict,
    y: int,
    x: int,
    *,
    quotient: dict[str, Any] | None = None,
    adjacency: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Describe the cell/object under a raster coordinate.

    This function is UI-agnostic so a viewer can call it from a future click
    handler without coupling the diagnostic code to DearPyGui.
    """
    graph = objects["graph"]
    labels = np.asarray(
        graph.get("labels", result["labels"]), dtype=np.int32)
    yy = int(np.clip(y, 0, labels.shape[0] - 1))
    xx = int(np.clip(x, 0, labels.shape[1] - 1))
    cell = int(labels[yy, xx])
    object_id = int(objects["object_id_per_cell"][cell])
    means = object_means(objects)
    context = object_boundary_context(
        objects, object_id, adjacency=adjacency)
    report: dict[str, Any] = {
        "x": xx,
        "y": yy,
        "cell_id": cell,
        "source_site_id": int(
            np.asarray(
                graph.get(
                    "source_site_per_cell",
                    np.arange(graph["cells"], dtype=np.int32),
                ),
            )[cell]
        ),
        "object_id": object_id,
        "second_object_id": int(
            objects["second_object_id_per_cell"][cell]),
        "object_cell_count": int(means["cell_count"][object_id]),
        "object_pixel_area": float(means["pixel_area"][object_id]),
        "object_area_fraction": float(means["area_fraction"][object_id]),
        "object_centroid": (
            float(means["centroid_x"][object_id]),
            float(means["centroid_y"][object_id]),
        ),
        "cell_centroid": (
            float(graph["node_x"][cell]),
            float(graph["node_y"][cell]),
        ),
        "cell_lab": np.asarray(graph["node_lab"][cell]).copy(),
        "object_lab": np.asarray(means["lab"][object_id]).copy(),
        "cell_cartoon": float(graph["node_cartoon"][cell]),
        "object_cartoon": float(means["cartoon"][object_id]),
        "waterline": float(
            objects["propagation"]["best_value"][cell]),
        "saddle_margin": float(
            (
                objects["propagation"]["best_value"][cell]
                - objects["propagation"]["second_value"][cell]
            ) / max(
                float(objects["propagation"]["best_value"][cell]), 1e-12
            )
        ),
        "boundary_context": context,
    }
    if quotient is not None:
        material_id = int(quotient["material_id_per_cell"][cell])
        report.update({
            "material_id": material_id,
            "object_connected_material_quotient": float(
                quotient["object_connected_material_quotient"][object_id]),
            "object_material_component_count": int(
                quotient["object_material_component_count"][object_id]),
            "material_object_quotient": float(
                quotient["material_object_quotient"][material_id]),
            "material_object_count": int(
                quotient["material_object_count"][material_id]),
        })
    return report


def format_selection_report(report: dict[str, Any], neighbours: int = 5) -> str:
    """Compact human-readable rendering of :func:`selection_report`."""
    lines = [
        (
            f"pixel ({report['x']}, {report['y']}) → "
            f"cell {report['cell_id']} → object {report['object_id']} "
            f"(source site {report['source_site_id']}, "
            f"runner-up {report['second_object_id']})"
        ),
        (
            f"object: {report['object_cell_count']} cells, "
            f"{report['object_pixel_area']:.0f} px "
            f"({100.0 * report['object_area_fraction']:.2f}%), "
            f"waterline {report['waterline']:.3f}, "
            f"saddle margin {report['saddle_margin']:.3f}"
        ),
    ]
    if "material_id" in report:
        lines.append(
            f"material {report['material_id']}: object-connectivity "
            f"{report['object_connected_material_quotient']:.3f} across "
            f"{report['object_material_component_count']} components; "
            f"material ownership {report['material_object_quotient']:.3f} "
            f"across {report['material_object_count']} objects"
        )
    context = report["boundary_context"]
    for index in range(min(int(neighbours), len(context["neighbour"]))):
        lines.append(
            f"neighbour {int(context['neighbour'][index])}: "
            f"barrier {float(context['barrier'][index]):.3f} "
            f"[{float(context['barrier_min'][index]):.3f}, "
            f"{float(context['barrier_max'][index]):.3f}], "
            f"length {float(context['length'][index]):.1f}"
        )
    return "\n".join(lines)


def analyze_object_hierarchy(
    result: dict,
    objects: dict,
    *,
    barrier_threshold: float = 0.35,
) -> dict[str, Any]:
    """Build the complete reusable diagnostic bundle."""
    adjacency = object_adjacency(objects)
    return {
        "objects": object_means(objects),
        "adjacency": adjacency,
        "material": connected_material_quotient(
            objects, barrier_threshold=barrier_threshold),
    }
