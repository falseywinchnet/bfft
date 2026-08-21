"""Evidence-preserving region complex over a finished V3 segmentation.

Reconstruction atoms, coherent segments, and objects are different objects.
This module begins with the connected compound regions emitted above V3's
immutable texture atoms.  It measures region interiors and their literal
raster interfaces while keeping content and structural coordinates separate.

There is intentionally no affinity scalar, merge threshold, seed selection,
or object label in this file.  The output is the empirical object on which a
later transport law can be proposed and falsified.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.embedded_interface_topology import (
    build_embedded_interface_topology,
)


def _compact_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    value = np.asarray(labels, dtype=np.int64)
    if value.ndim != 2 or np.any(value < 0):
        raise ValueError("region labels must be a non-negative 2-D array")
    _, compact = np.unique(value, return_inverse=True)
    out = compact.reshape(value.shape).astype(np.int32, copy=False)
    return out, int(out.max(initial=-1)) + 1


def _field_mean(
    labels: np.ndarray,
    values: np.ndarray,
    count: int,
    area: np.ndarray,
) -> np.ndarray:
    flat_label = labels.ravel()
    value = np.asarray(values, dtype=np.float64)
    if value.shape[:2] != labels.shape:
        raise ValueError("region fields must share the label raster shape")
    if value.ndim == 2:
        total = np.bincount(
            flat_label, weights=value.ravel(), minlength=count)
        return total / np.maximum(area, 1.0)
    return np.column_stack([
        np.bincount(
            flat_label,
            weights=value[..., channel].ravel(),
            minlength=count,
        ) / np.maximum(area, 1.0)
        for channel in range(value.shape[2])
    ])


def _field_rms(
    labels: np.ndarray,
    values: np.ndarray,
    count: int,
    area: np.ndarray,
) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    mean_square = _field_mean(labels, value * value, count, area)
    return np.sqrt(np.maximum(mean_square, 0.0))


def _region_ancestry(
    labels: np.ndarray,
    structural_labels: np.ndarray,
    count: int,
) -> dict[str, np.ndarray]:
    structural, structural_count = _compact_labels(structural_labels)
    key = labels.astype(np.int64).ravel() * structural_count + structural.ravel()
    unique, population = np.unique(key, return_counts=True)
    region = (unique // structural_count).astype(np.int32)
    ancestor = (unique % structural_count).astype(np.int32)
    order = np.argsort(region, kind="stable")
    region = region[order]
    ancestor = ancestor[order]
    population = population[order].astype(np.float64)
    starts = np.searchsorted(region, np.arange(count), side="left")
    ends = np.searchsorted(region, np.arange(count), side="right")
    dominant = np.full(count, -1, dtype=np.int32)
    purity = np.zeros(count, dtype=np.float64)
    entropy = np.zeros(count, dtype=np.float64)
    support_count = ends - starts
    for item in range(count):
        start, end = int(starts[item]), int(ends[item])
        if start == end:
            continue
        weight = population[start:end]
        total = float(np.sum(weight))
        winner = start + int(np.argmax(weight))
        dominant[item] = ancestor[winner]
        purity[item] = float(population[winner] / max(total, 1.0))
        probability = weight / max(total, 1.0)
        entropy[item] = float(-np.sum(
            probability * np.log(np.maximum(probability, 1e-30))))
    return {
        "structural_count": np.asarray(structural_count, dtype=np.int32),
        "dominant": dominant,
        "purity": purity,
        "entropy": entropy,
        "support_count": support_count.astype(np.int32),
        "pair_region": region,
        "pair_ancestor": ancestor,
        "pair_population": population,
    }


def _interface_samples(
    labels: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    mean_fields: frozenset[str] = frozenset(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = int(labels.max(initial=-1)) + 1
    keys: list[np.ndarray] = []
    samples: dict[str, list[np.ndarray]] = {
        "normal_x": [],
        "normal_y": [],
    }
    for name in fields:
        samples[name] = []

    for first_slice, second_slice, normal in (
        ((slice(None), slice(0, -1)),
         (slice(None), slice(1, None)), (1.0, 0.0)),
        ((slice(0, -1), slice(None)),
         (slice(1, None), slice(None)), (0.0, 1.0)),
    ):
        first = labels[first_slice]
        second = labels[second_slice]
        crossing = first != second
        a = first[crossing]
        b = second[crossing]
        low = np.minimum(a, b)
        high = np.maximum(a, b)
        sign = np.where(a == low, 1.0, -1.0)
        keys.append(low.astype(np.int64) * count + high)
        samples["normal_x"].append(sign * normal[0])
        samples["normal_y"].append(sign * normal[1])
        for name, values in fields.items():
            value = np.asarray(values, dtype=np.float64)
            x = value[first_slice][crossing]
            y = value[second_slice][crossing]
            if name in mean_fields:
                midpoint = 0.5 * (x + y)
                sample = (
                    midpoint if midpoint.ndim == 1
                    else np.linalg.norm(midpoint, axis=1)
                )
            else:
                difference = x - y
                if difference.ndim == 1:
                    sample = np.abs(difference)
                else:
                    sample = np.linalg.norm(difference, axis=1)
            samples[name].append(sample)

    if not keys:
        return np.empty(0, dtype=np.int64), {
            name: np.empty(0, dtype=np.float64) for name in samples
        }
    return np.concatenate(keys), {
        name: np.concatenate(value) for name, value in samples.items()
    }


def _interface_graph(
    labels: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    mean_fields: frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    count = int(labels.max(initial=-1)) + 1
    key, samples = _interface_samples(
        labels, fields, mean_fields=mean_fields)
    if not len(key):
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "first": empty_i,
            "second": empty_i.copy(),
            "length": empty_f,
        }
    order = np.argsort(key, kind="stable")
    ordered_key = key[order]
    starts = np.flatnonzero(np.r_[
        True, ordered_key[1:] != ordered_key[:-1]])
    unique = ordered_key[starts]
    length = np.diff(np.r_[starts, len(order)]).astype(np.float64)
    graph: dict[str, np.ndarray] = {
        "first": (unique // count).astype(np.int32),
        "second": (unique % count).astype(np.int32),
        "length": length,
    }
    for name, values in samples.items():
        ordered = np.asarray(values, dtype=np.float64)[order]
        if name in ("normal_x", "normal_y"):
            graph[name] = np.add.reduceat(ordered, starts) / length
        else:
            graph[name] = np.add.reduceat(ordered, starts) / length
            graph[f"{name}_rms"] = np.sqrt(
                np.add.reduceat(ordered * ordered, starts) / length)
    graph["orientation_coherence"] = np.hypot(
        graph["normal_x"], graph["normal_y"])
    return graph


def _arc_graph(
    labels: np.ndarray,
    fields: dict[str, np.ndarray],
    topology: dict,
) -> dict[str, np.ndarray]:
    """Aggregate measurements per separately connected embedded arc."""
    _, samples = _interface_samples(
        labels, fields, mean_fields=frozenset(("boundary",)))
    arc_id = np.asarray(topology["edgel"]["arc"], dtype=np.int32)
    count = int(topology["arc"]["count"])
    if len(arc_id) != len(samples["normal_x"]):
        raise RuntimeError("embedded arcs and interface samples disagree")
    length = np.bincount(arc_id, minlength=count).astype(np.float64)
    result = {
        "cell_first": np.asarray(
            topology["arc"]["cell_first"], dtype=np.int32),
        "cell_second": np.asarray(
            topology["arc"]["cell_second"], dtype=np.int32),
        "length": length,
        "closed": np.asarray(topology["arc"]["closed"], dtype=bool),
    }
    for name, values in samples.items():
        value = np.asarray(values, dtype=np.float64)
        total = np.bincount(arc_id, weights=value, minlength=count)
        result[name] = total / np.maximum(length, 1.0)
        if name not in ("normal_x", "normal_y"):
            square = np.bincount(
                arc_id, weights=value * value, minlength=count)
            result[f"{name}_rms"] = np.sqrt(
                square / np.maximum(length, 1.0))
    result["orientation_coherence"] = np.hypot(
        result["normal_x"], result["normal_y"])
    return result


def _perimeter(
    labels: np.ndarray,
    count: int,
    graph: dict[str, np.ndarray],
) -> np.ndarray:
    perimeter = np.zeros(count, dtype=np.float64)
    perimeter += np.bincount(labels[0], minlength=count)
    perimeter += np.bincount(labels[-1], minlength=count)
    perimeter += np.bincount(labels[:, 0], minlength=count)
    perimeter += np.bincount(labels[:, -1], minlength=count)
    np.add.at(perimeter, graph["first"], graph["length"])
    np.add.at(perimeter, graph["second"], graph["length"])
    return perimeter


def _rank_histogram(
    first: np.ndarray,
    second: np.ndarray,
    bins: int = 16,
) -> np.ndarray:
    """Copula histogram: dependence without selecting physical thresholds."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if not len(x):
        return np.zeros((bins, bins), dtype=np.int64)
    x_order = np.argsort(x, kind="stable")
    y_order = np.argsort(y, kind="stable")
    x_rank = np.empty(len(x), dtype=np.int64)
    y_rank = np.empty(len(y), dtype=np.int64)
    x_rank[x_order] = np.arange(len(x), dtype=np.int64)
    y_rank[y_order] = np.arange(len(y), dtype=np.int64)
    xb = np.minimum(x_rank * bins // max(len(x), 1), bins - 1)
    yb = np.minimum(y_rank * bins // max(len(y), 1), bins - 1)
    return np.bincount(
        xb * bins + yb, minlength=bins * bins).reshape(bins, bins)


def _correlation_matrix(channels: dict[str, np.ndarray]) -> dict[str, Any]:
    names = tuple(channels)
    if not names:
        return {"names": (), "matrix": np.empty((0, 0))}
    values = np.column_stack([
        np.asarray(channels[name], dtype=np.float64) for name in names
    ])
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    if len(values) < 2:
        matrix = np.eye(len(names), dtype=np.float64)
    else:
        centered = values - np.mean(values, axis=0, keepdims=True)
        scale = np.sqrt(np.sum(centered * centered, axis=0))
        unit = np.divide(
            centered,
            scale,
            out=np.zeros_like(centered),
            where=scale > 0.0,
        )
        matrix = unit.T @ unit
        diagonal = scale > 0.0
        matrix[np.arange(len(names)), np.arange(len(names))] = diagonal
    return {"names": names, "matrix": matrix}


def build_region_complex(
    result: dict,
    source_rgb: np.ndarray,
    *,
    level: str = "leaves",
    fused_meyer: dict | None = None,
) -> dict[str, Any]:
    """Build a read-only empirical complex from a completed V3 result.

    ``leaves`` are connected one-sided supports before the same-budget
    compound quotient.  ``compounds`` are retained as a comparison only: the
    quotient is allowed to merge leaves and may therefore cross a later
    object boundary.
    """
    compound = result.get("compound_segmentation", {})
    if not bool(compound.get("enabled", False)):
        raise ValueError("V3 compound segmentation must be enabled")
    if level == "leaves":
        source_labels = compound["leaf_labels"]
    elif level == "compounds":
        source_labels = compound["labels"]
    else:
        raise ValueError("region-complex level must be 'leaves' or 'compounds'")
    labels, count = _compact_labels(source_labels)
    source = np.asarray(source_rgb, dtype=np.float64)
    if source.shape[:2] != labels.shape:
        raise ValueError("source image and V3 labels must share a shape")
    area = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]]
    centroid_x = _field_mean(labels, xx, count, area)
    centroid_y = _field_mean(labels, yy, count, area)
    touches_frame = np.zeros(count, dtype=bool)
    touches_frame[np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    )))] = True

    target_lab = np.asarray(result["target_lab"], dtype=np.float64)
    cartoon_lab = np.asarray(result["cartoon_lab"], dtype=np.float64)
    texture_target = np.asarray(
        result["texture_target_lab"], dtype=np.float64)
    texture_fit = np.asarray(result["texture_fit_lab"], dtype=np.float64)
    geometry = result.get("texture_geometry")
    boundary = (
        np.zeros(labels.shape, dtype=np.float64)
        if geometry is None
        else np.asarray(geometry["boundary_confidence"], dtype=np.float64)
    )
    fields = {
        "target": target_lab,
        "cartoon": cartoon_lab,
        "texture_target": texture_target,
        "texture_fit": texture_fit,
        "boundary": boundary,
    }
    if fused_meyer is not None:
        for name in ("cartoon", "texture", "residual"):
            value = np.asarray(fused_meyer[name], dtype=np.float64)
            if value.shape != labels.shape:
                raise ValueError(
                    f"fused Meyer {name} must share the V3 raster shape")
            fields[f"fused_{name}"] = value
    graph = _interface_graph(
        labels, fields, mean_fields=frozenset(("boundary",)))
    perimeter = _perimeter(labels, count, graph)
    ancestry = _region_ancestry(
        labels, np.asarray(result["labels"], dtype=np.int32), count)
    node = {
        "area": area,
        "area_fraction": area / max(float(np.sum(area)), 1.0),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "perimeter": perimeter,
        "thickness": 2.0 * area / np.maximum(perimeter, 1.0),
        "touches_frame": touches_frame,
        "target_mean": _field_mean(labels, target_lab, count, area),
        "target_rms": _field_rms(labels, target_lab, count, area),
        "cartoon_mean": _field_mean(labels, cartoon_lab, count, area),
        "texture_target_mean": _field_mean(
            labels, texture_target, count, area),
        "texture_target_rms": _field_rms(
            labels, texture_target, count, area),
        "texture_fit_mean": _field_mean(labels, texture_fit, count, area),
        "texture_fit_rms": _field_rms(labels, texture_fit, count, area),
        "boundary_mean": _field_mean(labels, boundary, count, area),
        "structural_dominant": ancestry["dominant"],
        "structural_purity": ancestry["purity"],
        "structural_entropy": ancestry["entropy"],
        "structural_support_count": ancestry["support_count"],
    }
    if fused_meyer is not None:
        for name in ("cartoon", "texture", "residual"):
            field = fields[f"fused_{name}"]
            node[f"fused_{name}_mean"] = _field_mean(
                labels, field, count, area)
            node[f"fused_{name}_rms"] = _field_rms(
                labels, field, count, area)
    first, second = graph["first"], graph["second"]
    graph["same_dominant_structure"] = (
        node["structural_dominant"][first]
        == node["structural_dominant"][second]
    )
    graph["region_target_distance"] = np.linalg.norm(
        node["target_mean"][first] - node["target_mean"][second], axis=1)
    graph["region_cartoon_distance"] = np.linalg.norm(
        node["cartoon_mean"][first] - node["cartoon_mean"][second], axis=1)
    graph["region_texture_rms_distance"] = np.linalg.norm(
        node["texture_target_rms"][first]
        - node["texture_target_rms"][second], axis=1)
    if fused_meyer is not None:
        for name in ("cartoon", "texture", "residual"):
            graph[f"region_fused_{name}_distance"] = np.abs(
                node[f"fused_{name}_mean"][first]
                - node[f"fused_{name}_mean"][second])

    relation_channels = {
        "region_target_distance": graph["region_target_distance"],
        "region_cartoon_distance": graph["region_cartoon_distance"],
        "region_texture_rms_distance": graph["region_texture_rms_distance"],
        "literal_target_jump": graph["target"],
        "literal_cartoon_jump": graph["cartoon"],
        "literal_texture_jump": graph["texture_target"],
        "boundary_confidence": graph["boundary"],
        "orientation_coherence": graph["orientation_coherence"],
    }
    if fused_meyer is not None:
        relation_channels.update({
            "region_fused_cartoon_distance": graph[
                "region_fused_cartoon_distance"],
            "region_fused_texture_distance": graph[
                "region_fused_texture_distance"],
            "region_fused_residual_distance": graph[
                "region_fused_residual_distance"],
            "literal_fused_cartoon_jump": graph["fused_cartoon"],
            "literal_fused_texture_jump": graph["fused_texture"],
            "literal_fused_residual_jump": graph["fused_residual"],
        })
    empirical = {
        "relation_correlation": _correlation_matrix(relation_channels),
        "content_boundary_copula": _rank_histogram(
            graph["region_target_distance"], graph["boundary"]),
        "cartoon_texture_copula": _rank_histogram(
            graph["region_cartoon_distance"],
            graph["region_texture_rms_distance"],
        ),
    }
    topology = build_embedded_interface_topology(labels)
    arcs = _arc_graph(labels, fields, topology)
    return {
        "level": level,
        "labels": labels,
        "region_count": count,
        "structural_count": int(ancestry["structural_count"]),
        "node": node,
        "edge": graph,
        "ancestry": ancestry,
        "empirical": empirical,
        "topology": topology,
        "arc": arcs,
    }


def _quantiles(values: np.ndarray) -> list[float]:
    value = np.asarray(values, dtype=np.float64)
    if not len(value):
        return [0.0] * 5
    return [float(item) for item in np.quantile(
        value, (0.0, 0.25, 0.5, 0.75, 1.0))]


def summarize_region_complex(complex_: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-ready audit without inventing object scores."""
    node = complex_["node"]
    edge = complex_["edge"]
    correlation = complex_["empirical"]["relation_correlation"]
    topology = complex_["topology"]
    arc = topology["arc"]
    if int(arc["count"]):
        pair_key = (
            np.asarray(arc["cell_first"], dtype=np.int64)
            * int(complex_["region_count"])
            + np.asarray(arc["cell_second"], dtype=np.int64)
        )
        _, pair_arc_count = np.unique(pair_key, return_counts=True)
    else:
        pair_arc_count = np.empty(0, dtype=np.int64)
    return {
        "regions": int(complex_["region_count"]),
        "level": str(complex_["level"]),
        "structural_regions": int(complex_["structural_count"]),
        "interfaces": int(len(edge["first"])),
        "embedded_arcs": int(arc["count"]),
        "closed_arcs": int(np.count_nonzero(arc["closed"])),
        "junctions": int(topology["junction"]["count"]),
        "region_pairs_with_disconnected_arcs": int(np.count_nonzero(
            pair_arc_count > 1)),
        "arcs_per_region_pair_quantiles": _quantiles(pair_arc_count),
        "frame_regions": int(np.count_nonzero(node["touches_frame"])),
        "area_quantiles": _quantiles(node["area"]),
        "thickness_quantiles": _quantiles(node["thickness"]),
        "structural_purity_quantiles": _quantiles(
            node["structural_purity"]),
        "structural_support_count_quantiles": _quantiles(
            node["structural_support_count"]),
        "interface_length_quantiles": _quantiles(edge["length"]),
        "region_target_distance_quantiles": _quantiles(
            edge["region_target_distance"]),
        "literal_boundary_confidence_quantiles": _quantiles(
            edge["boundary"]),
        "same_dominant_structure_fraction": float(np.mean(
            edge["same_dominant_structure"])) if len(edge["first"]) else 0.0,
        "relation_channels": list(correlation["names"]),
        "relation_correlation": np.asarray(
            correlation["matrix"], dtype=np.float64).tolist(),
        "content_boundary_copula": np.asarray(
            complex_["empirical"]["content_boundary_copula"],
            dtype=np.int64,
        ).tolist(),
        "cartoon_texture_copula": np.asarray(
            complex_["empirical"]["cartoon_texture_copula"],
            dtype=np.int64,
        ).tolist(),
    }
