#!/usr/bin/env python3
"""Lossless-direction forensics for nonlocal transport association.

Nothing in this module changes a segmentation or decides which regions belong
together.  It exposes progressively richer empirical measures carried by the
finished transport graph:

1. region colour (a deliberately weak baseline);
2. the distribution of transport states inside each part;
3. the distribution of signed inside/outside transitions at its boundary,
   expressed in the boundary's own normal frame;
4. the same relational measure with target colour transitions included.

Empirical distributions are represented by deterministic random Fourier
features.  Their Euclidean distance estimates kernel maximum-mean discrepancy:
all moments can contribute, rather than a hand-weighted list of scalar rules.

The anchor-cell readout is deliberately lower-level.  It compares every
connected support fragment directly with one clicked fragment, before any
object waterline or parent hypothesis can erase a distinction.
"""

from __future__ import annotations

import numpy as np

from experiments.object_hierarchy_diagnostics import object_means


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    center = np.median(value, axis=0)
    deviation = np.median(np.abs(value - center), axis=0)
    scale = np.maximum(1.4826 * deviation, 1e-8)
    return (value - center) / scale


def _kernel_features(
    values: np.ndarray,
    dimension: int,
    seed: int,
) -> np.ndarray:
    value = _robust_standardize(values)
    rng = np.random.default_rng(seed)
    frequency = rng.standard_normal(
        (value.shape[1], dimension)
    ) / np.sqrt(max(value.shape[1], 1))
    phase = rng.uniform(0.0, 2.0 * np.pi, dimension)
    return np.sqrt(2.0 / dimension) * np.cos(
        value @ frequency + phase)


def _weighted_embedding(
    owner: np.ndarray,
    features: np.ndarray,
    weight: np.ndarray,
    count: int,
) -> np.ndarray:
    total = np.bincount(
        owner, weights=weight, minlength=count)
    out = np.column_stack([
        np.bincount(
            owner,
            weights=weight * features[:, column],
            minlength=count,
        )
        for column in range(features.shape[1])
    ])
    out /= np.maximum(total[:, None], 1e-30)
    return out


def _similarity_matrix(descriptor: np.ndarray) -> np.ndarray:
    value = np.asarray(descriptor, dtype=np.float64)
    norm = np.sum(value * value, axis=1)
    distance = np.maximum(
        norm[:, None] + norm[None, :] - 2.0 * value @ value.T,
        0.0,
    )
    off_diagonal = distance[
        ~np.eye(len(distance), dtype=bool) & (distance > 1e-12)
    ]
    scale = (
        float(np.median(off_diagonal))
        if off_diagonal.size else 1.0
    )
    similarity = np.exp(-distance / max(scale, 1e-12))
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _anchor_similarity(
    values: np.ndarray,
    anchor: int,
) -> np.ndarray:
    """Robust Gaussian likeness to one empirical support sample.

    Each coordinate is measured in global median-absolute-deviation units.
    The mean squared standardized displacement makes the kernel calibration
    independent of the number of coordinates in a particular readout.
    """
    value = np.clip(_robust_standardize(values), -12.0, 12.0)
    difference = value - value[int(anchor)]
    return np.exp(-0.5 * np.mean(difference * difference, axis=1))


def _transport_node_state(graph: dict) -> np.ndarray:
    qxx = np.asarray(graph["node_qxx"], dtype=np.float64)
    qxy = np.asarray(graph["node_qxy"], dtype=np.float64)
    qyy = np.asarray(graph["node_qyy"], dtype=np.float64)
    trace = np.maximum(qxx + qyy, 1e-30)
    return np.column_stack((
        np.log(np.maximum(graph["node_measure"], 1e-30)),
        np.log(np.maximum(graph["node_energy"], 1e-30)),
        np.log(np.maximum(graph["node_texture"], 1e-12)),
        np.log(trace),
        (qxx - qyy) / trace,
        2.0 * qxy / trace,
        graph["node_cartoon"],
        graph["node_glass"],
        graph["node_null"],
    ))


def _normal_metric(
    graph: dict,
    node: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
) -> np.ndarray:
    qxx = np.asarray(graph["node_qxx"], dtype=np.float64)[node]
    qxy = np.asarray(graph["node_qxy"], dtype=np.float64)[node]
    qyy = np.asarray(graph["node_qyy"], dtype=np.float64)[node]
    trace = np.maximum(qxx + qyy, 1e-30)
    tx, ty = -ny, nx
    qnn = qxx * nx * nx + 2.0 * qxy * nx * ny + qyy * ny * ny
    qtt = qxx * tx * tx + 2.0 * qxy * tx * ty + qyy * ty * ty
    qnt = (
        qxx * nx * tx
        + qxy * (nx * ty + ny * tx)
        + qyy * ny * ty
    )
    return np.column_stack((qnn / trace, qtt / trace, qnt / trace))


def _oriented_boundary_samples(
    objects: dict,
    node_state: np.ndarray,
) -> dict[str, np.ndarray]:
    graph = objects["graph"]
    edge = graph["edge"]
    part = np.asarray(objects["object_id_per_cell"], dtype=np.int32)
    first = np.asarray(edge["first"], dtype=np.int32)
    second = np.asarray(edge["second"], dtype=np.int32)
    crossing = part[first] != part[second]
    a, b = first[crossing], second[crossing]
    owner = np.concatenate((part[a], part[b]))
    inside = np.concatenate((a, b))
    outside = np.concatenate((b, a))
    weight = np.tile(
        np.asarray(edge["length"], dtype=np.float64)[crossing], 2)

    dx = (
        np.asarray(graph["node_x"])[outside]
        - np.asarray(graph["node_x"])[inside]
    )
    dy = (
        np.asarray(graph["node_y"])[outside]
        - np.asarray(graph["node_y"])[inside]
    )
    length = np.maximum(np.hypot(dx, dy), 1e-12)
    nx, ny = dx / length, dy / length
    inside_metric = _normal_metric(graph, inside, nx, ny)
    outside_metric = _normal_metric(graph, outside, nx, ny)

    # Global metric orientation is removed from the boundary representation.
    # The scalar transport state and its signed jump remain, followed by the
    # two tensors expressed in the local normal/tangent frame.
    scalar = np.column_stack((
        np.log(np.maximum(graph["node_measure"], 1e-30)),
        np.log(np.maximum(graph["node_energy"], 1e-30)),
        np.log(np.maximum(graph["node_texture"], 1e-12)),
        graph["node_cartoon"],
        graph["node_glass"],
        graph["node_null"],
    ))
    intrinsic_names = (
        "cartoon_jump", "glass_jump", "transport_action",
        "support_jump", "null_reliability", "decisive_boundary",
    )
    intrinsic = np.column_stack([
        np.tile(
            np.asarray(objects["evidence"][name])[crossing], 2)
        for name in intrinsic_names
    ])
    transport = np.column_stack((
        scalar[inside],
        scalar[inside] - scalar[outside],
        inside_metric,
        outside_metric,
        intrinsic,
    ))

    lab = np.asarray(graph["node_lab"], dtype=np.float64)
    colour_names = ("target_jump", "region_colour_jump")
    colour_intrinsic = np.column_stack([
        np.tile(
            np.asarray(objects["evidence"][name])[crossing], 2)
        for name in colour_names
    ])
    full = np.column_stack((
        transport,
        lab[inside],
        lab[inside] - lab[outside],
        colour_intrinsic,
    ))
    return {
        "owner": owner,
        "weight": weight,
        "transport": transport,
        "full": full,
    }


def transport_relation_forensics(
    objects: dict,
    *,
    feature_dimension: int = 128,
) -> dict:
    """Return independent nonlocal similarity readouts for every hard part."""
    graph = objects["graph"]
    part = np.asarray(objects["object_id_per_cell"], dtype=np.int32)
    count = int(part.max(initial=-1)) + 1
    node_state = _transport_node_state(graph)
    node_features = _kernel_features(
        node_state, feature_dimension, seed=0xBFF7)
    state_descriptor = _weighted_embedding(
        part,
        node_features,
        np.asarray(graph["area"], dtype=np.float64),
        count,
    )
    action_descriptor = _weighted_embedding(
        part,
        _kernel_features(
            node_state[:, 1:2], feature_dimension, seed=0xAC710),
        np.asarray(graph["area"], dtype=np.float64),
        count,
    )
    metric_descriptor = _weighted_embedding(
        part,
        _kernel_features(
            node_state[:, 3:6], feature_dimension, seed=0x6E071),
        np.asarray(graph["area"], dtype=np.float64),
        count,
    )

    boundary = _oriented_boundary_samples(objects, node_state)
    transport_features = _kernel_features(
        boundary["transport"], feature_dimension, seed=0x71A57)
    full_features = _kernel_features(
        boundary["full"], feature_dimension, seed=0xA550C)
    boundary_transport_descriptor = _weighted_embedding(
        boundary["owner"],
        transport_features,
        boundary["weight"],
        count,
    )
    boundary_full_descriptor = _weighted_embedding(
        boundary["owner"],
        full_features,
        boundary["weight"],
        count,
    )

    means = object_means(objects)
    colour_descriptor = _robust_standardize(
        np.asarray(means["lab"], dtype=np.float64))
    return {
        "feature_dimension": feature_dimension,
        "colour_descriptor": colour_descriptor,
        "state_descriptor": state_descriptor,
        "action_descriptor": action_descriptor,
        "metric_descriptor": metric_descriptor,
        "boundary_transport_descriptor": boundary_transport_descriptor,
        "boundary_full_descriptor": boundary_full_descriptor,
        "colour_similarity": _similarity_matrix(colour_descriptor),
        "state_similarity": _similarity_matrix(state_descriptor),
        "action_similarity": _similarity_matrix(action_descriptor),
        "metric_similarity": _similarity_matrix(metric_descriptor),
        "boundary_transport_similarity": _similarity_matrix(
            boundary_transport_descriptor),
        "boundary_full_similarity": _similarity_matrix(
            boundary_full_descriptor),
    }


def transport_anchor_cell_fields(
    objects: dict,
    anchor_cell: int,
) -> dict[str, np.ndarray]:
    """Expose independent cell-level likeness fields around one anchor.

    This is an information audit, not an association rule.  In particular,
    no adjacency, object ID, candidate selection, or region protection enters
    these fields.
    """
    graph = objects["graph"]
    count = int(graph["cells"])
    anchor = int(np.clip(anchor_cell, 0, max(count - 1, 0)))
    state = _transport_node_state(graph)
    lab = np.asarray(graph["node_lab"], dtype=np.float64)
    return {
        "anchor_cell": anchor,
        "colour": _anchor_similarity(lab, anchor),
        "action": _anchor_similarity(state[:, 1:2], anchor),
        "metric": _anchor_similarity(state[:, 3:6], anchor),
        "action_metric": _anchor_similarity(
            state[:, [1, 3, 4, 5]], anchor),
        "full_state": _anchor_similarity(state, anchor),
    }
