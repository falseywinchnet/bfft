#!/usr/bin/env python3
"""Sparse border ownership and transport-depth diagnostics.

This module does not assign semantic objects and does not alter the hard part
segmentation.  It keeps the first genuinely directed information available in
a single image:

* under the occlusion interpretation of a three-region T-junction, the region
  shared by the two continuing arcs is locally in front of the two regions
  separated by the terminating stem;
* each hard part carries signed BFFT support statistics which may, or may not,
  vary consistently with that observed depth order.

The resulting hypotheses form a candidate partial order, not a dense depth
map.  Material seams ending at a silhouette remain a competing explanation.
A single
closed-form accumulation estimates the transport-support direction most
consistent with those arrows.  Its extrapolation is exposed as a research
control and is never fed back into merging.
"""

from __future__ import annotations

import numpy as np

from experiments.object_hierarchy_diagnostics import object_means
from experiments.transport_focus_forensics import (
    transport_focus_interfaces,
)


_FEATURE_NAMES = (
    "log population measure",
    "log transport energy",
    "log metric trace",
    "metric coherence",
    "log texture magnitude",
    "cartoon state",
    "glass state",
    "null reliability",
)


def _robust_standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(values, dtype=np.float64)
    center = np.median(value, axis=0)
    deviation = np.median(np.abs(value - center), axis=0)
    # The normal-consistent Gaussian conversion is useful only as a scale;
    # exact constants have no effect after the direction is normalized.
    scale = np.maximum(1.4826 * deviation, 1e-8)
    return (value - center) / scale, center, scale


def _part_support_features(objects: dict) -> dict[str, np.ndarray]:
    means = object_means(objects)
    trace = np.maximum(
        np.asarray(means["qxx"]) + np.asarray(means["qyy"]), 1e-30)
    coherence = np.hypot(
        np.asarray(means["qxx"]) - np.asarray(means["qyy"]),
        2.0 * np.asarray(means["qxy"]),
    ) / trace
    raw = np.column_stack((
        np.log(np.maximum(means["measure"], 1e-30)),
        np.log(np.maximum(means["energy"], 1e-30)),
        np.log(trace),
        coherence,
        np.log(np.maximum(means["texture"], 1e-12)),
        means["cartoon"],
        means["glass"],
        means["null"],
    ))
    standardized, center, scale = _robust_standardize(raw)
    return {
        "names": np.asarray(_FEATURE_NAMES, dtype=object),
        "raw": raw,
        "standardized": standardized,
        "center": center,
        "scale": scale,
    }


def _directed_junction_votes(
    junction: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Aggregate repeated front/back evidence without forming all pairs."""
    vote: dict[tuple[int, int], list[float]] = {}
    for first, second, front, score in zip(
        junction["first"],
        junction["second"],
        junction["surround"],
        junction["score"],
    ):
        foreground = int(front)
        for background in (int(first), int(second)):
            if foreground == background:
                continue
            vote.setdefault((foreground, background), []).append(float(score))

    records = []
    visited: set[tuple[int, int]] = set()
    for (first, second), values in sorted(vote.items()):
        unordered = tuple(sorted((first, second)))
        if unordered in visited:
            continue
        visited.add(unordered)
        forward = vote.get((first, second), ())
        reverse = vote.get((second, first), ())
        forward_mass = float(np.sum(forward))
        reverse_mass = float(np.sum(reverse))
        net = forward_mass - reverse_mass
        if net >= 0.0:
            front, back = first, second
        else:
            front, back = second, first
        total = forward_mass + reverse_mass
        records.append((
            front,
            back,
            abs(net),
            total,
            abs(net) / max(total, 1e-12),
            len(forward) + len(reverse),
        ))
    names = (
        "front", "back", "net_support", "total_support",
        "direction_confidence", "witness_count",
    )
    if not records:
        return {
            name: np.empty(
                0,
                dtype=np.int32
                if name in ("front", "back", "witness_count")
                else np.float64,
            )
            for name in names
        }
    columns = list(zip(*records))
    return {
        name: np.asarray(
            columns[index],
            dtype=np.int32
            if name in ("front", "back", "witness_count")
            else np.float64,
        )
        for index, name in enumerate(names)
    }


def _arc_ownership_votes(
    junction: dict[str, np.ndarray],
    topology: dict,
) -> dict[str, np.ndarray]:
    """Orient continuing arcs from their literal T-junction witnesses."""
    arc = topology["arc"]
    count = int(arc["count"])
    signed = np.zeros(count, dtype=np.float64)
    total = np.zeros(count, dtype=np.float64)
    witnesses = np.zeros(count, dtype=np.int32)
    for index in range(len(junction["score"])):
        front = int(junction["surround"][index])
        score = float(junction["score"][index])
        for name in ("first_outer_arc", "second_outer_arc"):
            edge = int(junction[name][index])
            first = int(arc["cell_first"][edge])
            second = int(arc["cell_second"][edge])
            if front == first:
                direction = 1.0
            elif front == second:
                direction = -1.0
            else:
                continue
            signed[edge] += direction * score
            total[edge] += score
            witnesses[edge] += 1
    sign = np.sign(signed).astype(np.int8)
    confidence = np.abs(signed) / np.maximum(total, 1e-12)
    front = np.full(count, -1, dtype=np.int32)
    back = np.full(count, -1, dtype=np.int32)
    positive = sign > 0
    negative = sign < 0
    front[positive] = arc["cell_first"][positive]
    back[positive] = arc["cell_second"][positive]
    front[negative] = arc["cell_second"][negative]
    back[negative] = arc["cell_first"][negative]
    return {
        "signed_vote": signed,
        "total_vote": total,
        "sign": sign,
        "confidence": confidence,
        "witness_count": witnesses,
        "front": front,
        "back": back,
    }


def _focus_arc_ownership_votes(
    objects: dict,
    topology: dict,
    labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Orient arcs only where boundary blur matches one measured side."""
    focus = objects.get("focus_forensics")
    count = int(topology["arc"]["count"])
    if focus is None:
        return _empty_arc_ownership(count), None
    interface = transport_focus_interfaces(
        focus,
        labels,
        topology,
    )
    margin = np.asarray(
        interface["first_match_margin"], dtype=np.float64)
    reliability = np.asarray(
        interface["reliability"], dtype=np.float64)
    supported = np.abs(margin)[reliability > 0.0]
    scale = (
        float(np.percentile(supported, 80.0))
        if supported.size else 1.0
    )
    signed = (
        np.tanh(margin / max(scale, 1e-12))
        * np.clip(reliability, 0.0, 1.0)
    )
    return _ownership_from_signed(
        topology,
        signed,
        (reliability > 0.0).astype(np.float64),
        (reliability > 0.0).astype(np.int32),
    ), interface


def _empty_arc_ownership(count: int) -> dict[str, np.ndarray]:
    return {
        "signed_vote": np.zeros(count, dtype=np.float64),
        "total_vote": np.zeros(count, dtype=np.float64),
        "sign": np.zeros(count, dtype=np.int8),
        "confidence": np.zeros(count, dtype=np.float64),
        "witness_count": np.zeros(count, dtype=np.int32),
        "front": np.full(count, -1, dtype=np.int32),
        "back": np.full(count, -1, dtype=np.int32),
    }


def _ownership_from_signed(
    topology: dict,
    signed: np.ndarray,
    total: np.ndarray,
    witnesses: np.ndarray,
) -> dict[str, np.ndarray]:
    arc = topology["arc"]
    signed = np.asarray(signed, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    sign = np.sign(signed).astype(np.int8)
    confidence = np.abs(signed) / np.maximum(total, 1e-12)
    front = np.full(len(signed), -1, dtype=np.int32)
    back = np.full(len(signed), -1, dtype=np.int32)
    positive = sign > 0
    negative = sign < 0
    front[positive] = arc["cell_first"][positive]
    back[positive] = arc["cell_second"][positive]
    front[negative] = arc["cell_second"][negative]
    back[negative] = arc["cell_first"][negative]
    return {
        "signed_vote": signed,
        "total_vote": total,
        "sign": sign,
        "confidence": confidence,
        "witness_count": np.asarray(witnesses, dtype=np.int32),
        "front": front,
        "back": back,
    }


def _combine_arc_ownership(
    topology: dict,
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return _ownership_from_signed(
        topology,
        np.asarray(first["signed_vote"]) + np.asarray(second["signed_vote"]),
        np.asarray(first["total_vote"]) + np.asarray(second["total_vote"]),
        np.asarray(first["witness_count"]) + np.asarray(
            second["witness_count"]),
    )


def _directed_arc_votes(
    ownership: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Collapse separately embedded arcs into directed part-pair evidence."""
    vote: dict[tuple[int, int], list[float]] = {}
    for front, back, signed in zip(
        ownership["front"],
        ownership["back"],
        ownership["signed_vote"],
    ):
        if int(front) < 0 or int(back) < 0:
            continue
        vote.setdefault((int(front), int(back)), []).append(abs(float(signed)))
    records = []
    visited: set[tuple[int, int]] = set()
    for first, second in sorted(vote):
        unordered = tuple(sorted((first, second)))
        if unordered in visited:
            continue
        visited.add(unordered)
        forward = float(np.sum(vote.get((first, second), ())))
        reverse = float(np.sum(vote.get((second, first), ())))
        net = forward - reverse
        front, back = (
            (first, second) if net >= 0.0 else (second, first))
        total = forward + reverse
        records.append((
            front,
            back,
            abs(net),
            total,
            abs(net) / max(total, 1e-12),
            len(vote.get((first, second), ()))
            + len(vote.get((second, first), ())),
        ))
    names = (
        "front", "back", "net_support", "total_support",
        "direction_confidence", "witness_count",
    )
    if not records:
        return {
            name: np.empty(
                0,
                dtype=np.int32
                if name in ("front", "back", "witness_count")
                else np.float64,
            )
            for name in names
        }
    columns = list(zip(*records))
    return {
        name: np.asarray(
            columns[index],
            dtype=np.int32
            if name in ("front", "back", "witness_count")
            else np.float64,
        )
        for index, name in enumerate(names)
    }


def _rasterize_arc_ownership(
    labels: np.ndarray,
    topology: dict,
    ownership: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Paint the two incident pixel sides of every witnessed owned arc."""
    label = np.asarray(labels, dtype=np.int32)
    height, width = label.shape
    stride = width + 1
    edgel = topology["edgel"]
    arc_id = np.asarray(edgel["arc"], dtype=np.int32)
    confidence = ownership["confidence"][arc_id]
    witnessed = ownership["front"][arc_id] >= 0
    vertex = np.asarray(edgel["vertex_first"], dtype=np.int64)
    x = vertex % stride
    y = vertex // stride
    vertical = np.asarray(edgel["orientation"]) == 1
    first_y = np.where(vertical, y, y - 1)
    first_x = np.where(vertical, x - 1, x)
    second_y = np.where(vertical, y, y)
    second_x = np.where(vertical, x, x)
    front_map = np.zeros(label.shape, dtype=np.float64)
    back_map = np.zeros(label.shape, dtype=np.float64)
    for yy, xx in (
        (first_y, first_x),
        (second_y, second_x),
    ):
        valid = (
            witnessed
            & (yy >= 0) & (yy < height)
            & (xx >= 0) & (xx < width)
        )
        if not np.any(valid):
            continue
        region = label[yy[valid], xx[valid]]
        arc = arc_id[valid]
        value = confidence[valid]
        is_front = region == ownership["front"][arc]
        if np.any(is_front):
            np.maximum.at(
                front_map,
                (yy[valid][is_front], xx[valid][is_front]),
                value[is_front],
            )
        if np.any(~is_front):
            np.maximum.at(
                back_map,
                (yy[valid][~is_front], xx[valid][~is_front]),
                value[~is_front],
            )
    return {
        "front_boundary": front_map,
        "back_boundary": back_map,
        "confidence": np.maximum(front_map, back_map),
    }


def infer_transport_border_ownership(
    objects: dict,
    parent_hierarchy: dict,
) -> dict:
    """Return witnessed depth arrows and a falsifiable support extrapolation."""
    part = np.asarray(objects["object_id_per_cell"], dtype=np.int32)
    count = int(part.max(initial=-1)) + 1
    feature = _part_support_features(objects)
    junction_directed = _directed_junction_votes(
        parent_hierarchy["junction_relations"])
    topology = parent_hierarchy.get("topology")
    if (
        topology is not None
        and "first_outer_arc" in parent_hierarchy["junction_relations"]
    ):
        junction_arc_ownership = _arc_ownership_votes(
            parent_hierarchy["junction_relations"], topology)
        focus_arc_ownership, focus_interface = (
            _focus_arc_ownership_votes(
                objects,
                topology,
                parent_hierarchy["topology_part_labels"],
            )
        )
        arc_ownership = _combine_arc_ownership(
            topology,
            junction_arc_ownership,
            focus_arc_ownership,
        )
        directed = _directed_arc_votes(arc_ownership)
        junction_arc_maps = _rasterize_arc_ownership(
            objects["object_labels"], topology, junction_arc_ownership)
        focus_arc_maps = _rasterize_arc_ownership(
            objects["object_labels"], topology, focus_arc_ownership)
        arc_maps = _rasterize_arc_ownership(
            objects["object_labels"], topology, arc_ownership)
    else:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        junction_arc_ownership = {
            "signed_vote": empty_f,
            "total_vote": empty_f.copy(),
            "sign": np.empty(0, dtype=np.int8),
            "confidence": empty_f.copy(),
            "witness_count": empty_i,
            "front": empty_i.copy(),
            "back": empty_i.copy(),
        }
        focus_arc_ownership = dict(junction_arc_ownership)
        arc_ownership = dict(junction_arc_ownership)
        focus_interface = None
        directed = junction_directed
        zero = np.zeros_like(
            objects["object_labels"], dtype=np.float64)
        arc_maps = {
            "front_boundary": zero,
            "back_boundary": zero.copy(),
            "confidence": zero.copy(),
        }
        junction_arc_maps = {
            name: value.copy() for name, value in arc_maps.items()
        }
        focus_arc_maps = {
            name: value.copy() for name, value in arc_maps.items()
        }

    observed = np.zeros(count, dtype=np.float64)
    if len(directed["front"]):
        weight = (
            directed["net_support"] * directed["direction_confidence"])
        np.add.at(observed, directed["front"], weight)
        np.subtract.at(observed, directed["back"], weight)
        normalizer = np.zeros(count, dtype=np.float64)
        np.add.at(normalizer, directed["front"], weight)
        np.add.at(normalizer, directed["back"], weight)
        observed /= np.maximum(normalizer, 1e-12)

        difference = (
            feature["standardized"][directed["front"]]
            - feature["standardized"][directed["back"]]
        )
        direction = np.sum(weight[:, None] * difference, axis=0)
        accumulated_direction = direction.copy()
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction /= norm
        support_score = (
            feature["standardized"] @ direction
            / max(np.sqrt(feature["standardized"].shape[1]), 1.0)
        )
        predicted_difference = (
            support_score[directed["front"]]
            - support_score[directed["back"]]
        )
        agreement = np.sign(predicted_difference)
        # Exact leave-one-relation-out audit.  Each row subtracts its own
        # contribution from the one accumulated direction, so no refit loop
        # or solve is required.
        leave_one_out = (
            accumulated_direction[None, :]
            - weight[:, None] * difference
        )
        leave_norm = np.linalg.norm(leave_one_out, axis=1)
        leave_prediction = np.sum(
            leave_one_out * difference, axis=1
        ) / np.maximum(leave_norm, 1e-12)
        leave_valid = leave_norm > 1e-12
    else:
        direction = np.zeros(
            feature["standardized"].shape[1], dtype=np.float64)
        support_score = np.zeros(count, dtype=np.float64)
        predicted_difference = np.empty(0, dtype=np.float64)
        agreement = np.empty(0, dtype=np.float64)
        leave_prediction = np.empty(0, dtype=np.float64)
        leave_valid = np.empty(0, dtype=bool)

    labels = np.asarray(objects["object_labels"], dtype=np.int32)
    return {
        "feature": feature,
        "directed_relations": directed,
        "junction_directed_relations": junction_directed,
        "junction_arc_ownership": junction_arc_ownership,
        "focus_arc_ownership": focus_arc_ownership,
        "focus_interface": focus_interface,
        "arc_ownership": arc_ownership,
        "arc_maps": arc_maps,
        "junction_arc_maps": junction_arc_maps,
        "focus_arc_maps": focus_arc_maps,
        "observed_frontness_per_part": observed,
        "observed_frontness": observed[labels],
        "support_direction": direction,
        "support_frontness_per_part": support_score,
        "support_frontness": support_score[labels],
        "relation_predicted_difference": predicted_difference,
        "relation_agreement": agreement,
        "relation_agreement_fraction": float(
            np.mean(agreement > 0.0)) if len(agreement) else np.nan,
        "leave_one_out_prediction": leave_prediction,
        "leave_one_out_agreement_fraction": float(
            np.mean(leave_prediction[leave_valid] > 0.0)
        ) if np.any(leave_valid) else np.nan,
    }
