#!/usr/bin/env python3
"""Object support emerging from the finished BFFT transport-cell graph.

This experiment starts *after* the canonical segmenting pipeline.  It does
not revisit pixels as arbitrary candidates and it does not alter the cells.
The hard cell interfaces form a sparse planar graph.  Every graph edge carries
measurements already supplied by the unchanged target and its one frozen BFFT
geometry:

* direct OKLab and cartoon jumps;
* the glass-state jump;
* transport action across the interface normal;
* decisive boundary confidence and cross-scale null confidence.

Local support-density maxima are provisional germs.  A maximum-support forest
then supplies a topological persistence test: nearby germs gather through
permeable interfaces, while a germ separated by a real barrier survives as a
distinct object ID.  Finally an exact two-label widest-path solve on that
forest gives every cell its best and second-best germ.  Their margin is a
smooth object-membership confidence, not a binary post-processing mask.

The expensive image representation is reusable.  Changing the object controls
only rebuilds this O(pixels + interfaces log interfaces) analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np

from bfft.effects import srgb_to_lab
from experiments.embedded_interface_topology import (
    build_embedded_interface_topology,
)
from experiments.embedded_contour_persistence import (
    intrinsic_boundary_alignment,
    maximum_bottleneck_cycle_support,
)
from experiments.transport_focus_forensics import (
    autofocus_cell_score,
    transport_focus_forensics,
    transport_focus_interfaces,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - project runtime includes numba
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


@dataclass(frozen=True)
class ObjectSupportConfig:
    """Controls for the experimental object-support hierarchy."""

    boundary_weight: float = 1.5
    target_jump_weight: float = 0.15
    region_colour_weight: float = 0.05
    cartoon_jump_weight: float = 0.10
    glass_jump_weight: float = 0.25
    transport_weight: float = 0.65
    support_jump_weight: float = 1.0
    focus_jump_weight: float = 0.0
    focus_seed_weight: float = 0.0
    null_suppression: float = 0.90
    fragment_jump_threshold: float = 0.12
    anchored_barriers: bool = False
    material_tolerance: float = 0.012
    material_boundary_ceiling: float = 0.08
    short_contact_scale: float = 0.0
    short_contact_prior: float = 0.5
    contour_cycle_weight: float = 0.0
    intrinsic_contour_weight: float = 0.0
    finsler_contour_weight: float = 0.0
    barrier_scale: float = 1.5
    detail_weight: float = 0.35
    enclosure_weight: float = 0.15
    core_weight: float = 1.0
    peak_prominence: float = 0.30
    confidence_temperature: float = 0.12


def _robust_unit(values: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Positive field -> [0, 1), with a robust, scale-free shoulder."""
    value = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = value[value > 0.0]
    scale = (
        float(np.percentile(positive, percentile))
        if positive.size
        else 1.0
    )
    scale = max(scale, 1e-12)
    return value / (value + scale)


def _robust_excess(
    values: np.ndarray,
    floor_percentile: float = 10.0,
    scale_percentile: float = 90.0,
) -> np.ndarray:
    """Suppress a spatially constant metric floor before normalization."""
    value = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if not value.size:
        return value
    floor = float(np.percentile(value, floor_percentile))
    excess = np.maximum(value - floor, 0.0)
    return _robust_unit(excess, scale_percentile)


def _cell_mean(
    labels: np.ndarray,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    flat_label = np.asarray(labels, dtype=np.int32).ravel()
    area = np.bincount(flat_label, minlength=count).astype(np.float64)
    value = np.asarray(values, dtype=np.float64)
    if value.ndim == 2:
        total = np.bincount(
            flat_label, weights=value.ravel(), minlength=count)
        return total / np.maximum(area, 1.0)
    channels = [
        np.bincount(
            flat_label,
            weights=value[..., channel].ravel(),
            minlength=count,
        ) / np.maximum(area, 1.0)
        for channel in range(value.shape[2])
    ]
    return np.column_stack(channels)


@njit(cache=True)
def _connected_site_roots(
    labels: np.ndarray,
    signal: np.ndarray,
    maximum_jump_squared: float,
) -> np.ndarray:
    """Union four-connected pixels only when their reconstruction site agrees.

    A site is a reconstruction basis function, not necessarily one connected
    region.  Object topology must not be allowed to jump between disconnected
    pieces merely because the renderer assigned them the same site ID.
    """
    height, width = labels.shape
    count = height * width
    parent = np.arange(count, dtype=np.int64)
    size = np.ones(count, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            previous = parent[node]
            parent[node] = root
            node = previous
        return root

    def unite(first: int, second: int) -> None:
        a = find(first)
        b = find(second)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for y in range(height):
        row = y * width
        for x in range(width):
            index = row + x
            value = labels[y, x]
            if x > 0 and labels[y, x - 1] == value:
                jump = 0.0
                for channel in range(signal.shape[2]):
                    difference = (
                        signal[y, x, channel] - signal[y, x - 1, channel])
                    jump += difference * difference
                if jump <= maximum_jump_squared:
                    unite(index, index - 1)
            if y > 0 and labels[y - 1, x] == value:
                jump = 0.0
                for channel in range(signal.shape[2]):
                    difference = (
                        signal[y, x, channel] - signal[y - 1, x, channel])
                    jump += difference * difference
                if jump <= maximum_jump_squared:
                    unite(index, index - width)
    for index in range(count):
        parent[index] = find(index)
    return parent


def connected_site_fragments(
    labels: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    maximum_jump: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Split site IDs into dense four-connected support fragments.

    Returns ``(fragment_labels, source_site_per_fragment)``.  No geometry is
    created or moved: this only restores topology that was lost by quotienting
    all disconnected ownership islands of one reconstruction site together.
    """
    site = np.ascontiguousarray(labels, dtype=np.int32)
    if signal is None:
        value = np.zeros((*site.shape, 1), dtype=np.float64)
    else:
        value = np.ascontiguousarray(signal, dtype=np.float64)
        if value.shape[:2] != site.shape:
            raise ValueError("fragment signal and labels must share a shape")
        if value.ndim == 2:
            value = value[..., None]
    threshold = max(float(maximum_jump), 0.0)
    root = _connected_site_roots(
        site,
        value,
        threshold * threshold,
    ).ravel()
    _, first, fragment = np.unique(
        root, return_index=True, return_inverse=True)
    source_site = site.ravel()[first].astype(np.int32, copy=False)
    return (
        fragment.reshape(site.shape).astype(np.int32, copy=False),
        source_site,
    )


def _interface_samples(
    labels: np.ndarray,
    target_lab: np.ndarray,
    geometry: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Collect one record per horizontal/vertical cell-interface pixel."""
    labels = np.asarray(labels, dtype=np.int32)
    boundary = np.asarray(
        geometry["boundary_confidence"], dtype=np.float64)
    cartoon = np.asarray(geometry["cartoon"], dtype=np.float64)
    glass = np.asarray(geometry["glass"], dtype=np.float64)
    null = np.asarray(geometry["null_confidence"], dtype=np.float64)
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    bxx = np.asarray(geometry["boundary_xx"], dtype=np.float64)
    byy = np.asarray(geometry["boundary_yy"], dtype=np.float64)
    count = int(labels.max(initial=-1)) + 1

    keys: list[np.ndarray] = []
    samples: dict[str, list[np.ndarray]] = {
        "target_jump": [],
        "cartoon_jump": [],
        "glass_jump": [],
        "boundary": [],
        "transport": [],
        "null": [],
        "y": [],
        "x": [],
    }

    def append(
        first_label: np.ndarray,
        second_label: np.ndarray,
        first_lab: np.ndarray,
        second_lab: np.ndarray,
        first_cartoon: np.ndarray,
        second_cartoon: np.ndarray,
        first_glass: np.ndarray,
        second_glass: np.ndarray,
        first_boundary: np.ndarray,
        second_boundary: np.ndarray,
        first_metric: np.ndarray,
        second_metric: np.ndarray,
        first_boundary_metric: np.ndarray,
        second_boundary_metric: np.ndarray,
        first_null: np.ndarray,
        second_null: np.ndarray,
        yy: np.ndarray,
        xx: np.ndarray,
    ) -> None:
        crossing = first_label != second_label
        a = first_label[crossing].astype(np.int64)
        b = second_label[crossing].astype(np.int64)
        low, high = np.minimum(a, b), np.maximum(a, b)
        keys.append(low * np.int64(count) + high)
        difference = second_lab[crossing] - first_lab[crossing]
        samples["target_jump"].append(np.linalg.norm(difference, axis=1))
        samples["cartoon_jump"].append(np.abs(
            second_cartoon[crossing] - first_cartoon[crossing]))
        samples["glass_jump"].append(np.abs(
            second_glass[crossing] - first_glass[crossing]))
        samples["boundary"].append(0.5 * (
            first_boundary[crossing] + second_boundary[crossing]))
        # Both the finite unchanged-target jump action and the BFFT metric
        # normal action are retained.  The former dominates clean silhouettes;
        # the latter still contributes where the target jump is ambiguous.
        samples["transport"].append(np.sqrt(np.maximum(
            0.5 * (
                first_metric[crossing] + second_metric[crossing])
            + 0.5 * (
                first_boundary_metric[crossing]
                + second_boundary_metric[crossing]),
            0.0,
        )))
        samples["null"].append(0.5 * (
            first_null[crossing] + second_null[crossing]))
        samples["y"].append(yy[crossing])
        samples["x"].append(xx[crossing])

    hy, hx = np.mgrid[:labels.shape[0], :labels.shape[1] - 1]
    append(
        labels[:, :-1], labels[:, 1:],
        target_lab[:, :-1], target_lab[:, 1:],
        cartoon[:, :-1], cartoon[:, 1:],
        glass[:, :-1], glass[:, 1:],
        boundary[:, :-1], boundary[:, 1:],
        qxx[:, :-1], qxx[:, 1:],
        bxx[:, :-1], bxx[:, 1:],
        null[:, :-1], null[:, 1:],
        hy, hx,
    )
    vy, vx = np.mgrid[:labels.shape[0] - 1, :labels.shape[1]]
    append(
        labels[:-1], labels[1:],
        target_lab[:-1], target_lab[1:],
        cartoon[:-1], cartoon[1:],
        glass[:-1], glass[1:],
        boundary[:-1], boundary[1:],
        qyy[:-1], qyy[1:],
        byy[:-1], byy[1:],
        null[:-1], null[1:],
        vy, vx,
    )
    if not keys:
        return np.empty(0, dtype=np.int64), {
            name: np.empty(0, dtype=np.float64) for name in samples
        }
    return (
        np.concatenate(keys),
        {name: np.concatenate(parts) for name, parts in samples.items()},
    )


def build_cell_interface_graph(
    result: dict,
    target_rgb: np.ndarray,
    *,
    fragment_jump_threshold: float = 0.12,
) -> dict:
    """Assemble the literal sparse graph induced by hard cell interfaces."""
    source_labels = np.asarray(result["labels"], dtype=np.int32)
    geometry = result["geometry"]
    target_lab = srgb_to_lab(np.asarray(target_rgb, dtype=np.float64))
    labels, source_site = connected_site_fragments(
        source_labels,
        target_lab,
        maximum_jump=fragment_jump_threshold,
    )
    cells = int(labels.max(initial=-1)) + 1
    flat = labels.ravel()
    area = np.bincount(flat, minlength=cells).astype(np.float64)
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]]
    node_x = np.bincount(
        flat, weights=xx.ravel(), minlength=cells) / np.maximum(area, 1.0)
    node_y = np.bincount(
        flat, weights=yy.ravel(), minlength=cells) / np.maximum(area, 1.0)
    touches_border = np.zeros(cells, dtype=bool)
    touches_border[np.unique(np.concatenate((
        labels[0],
        labels[-1],
        labels[:, 0],
        labels[:, -1],
    )))] = True

    key, samples = _interface_samples(labels, target_lab, geometry)
    if key.size == 0:
        raise ValueError("the transport representation has no cell interfaces")
    topology = build_embedded_interface_topology(labels)
    sample_arc = np.asarray(
        topology["edgel"]["arc"], dtype=np.int32)
    if len(sample_arc) != len(key):
        raise RuntimeError(
            "embedded edgels and literal interface samples disagree")
    arc_count = int(topology["arc"]["count"])
    length = np.bincount(
        sample_arc, minlength=arc_count).astype(np.float64)
    edge: dict[str, np.ndarray] = {
        "first": np.asarray(
            topology["arc"]["cell_first"], dtype=np.int32),
        "second": np.asarray(
            topology["arc"]["cell_second"], dtype=np.int32),
        "length": length,
    }
    for name, value in samples.items():
        total = np.bincount(
            sample_arc,
            weights=np.asarray(value, dtype=np.float64),
            minlength=arc_count,
        )
        edge[name] = total / np.maximum(length, 1.0)
        if name not in ("x", "y"):
            square = np.bincount(
                sample_arc,
                weights=np.square(np.asarray(value, dtype=np.float64)),
                minlength=arc_count,
            )
            edge[f"{name}_rms"] = np.sqrt(
                square / np.maximum(length, 1.0))

    node_lab = _cell_mean(labels, target_lab, cells)
    node_cartoon = _cell_mean(labels, geometry["cartoon"], cells)
    node_glass = _cell_mean(labels, geometry["glass"], cells)
    node_texture = _cell_mean(labels, np.abs(geometry["texture"]), cells)
    node_measure = _cell_mean(labels, geometry["measure"], cells)
    node_energy = _cell_mean(labels, geometry["energy"], cells)
    node_null = _cell_mean(labels, geometry["null_confidence"], cells)
    node_qxx = _cell_mean(labels, geometry["precision_xx"], cells)
    node_qxy = _cell_mean(labels, geometry["precision_xy"], cells)
    node_qyy = _cell_mean(labels, geometry["precision_yy"], cells)
    first, second = edge["first"], edge["second"]
    edge["region_colour_jump"] = np.linalg.norm(
        node_lab[first] - node_lab[second], axis=1)
    edge["region_cartoon_jump"] = np.abs(
        node_cartoon[first] - node_cartoon[second])
    trace = np.maximum(node_qxx + node_qyy, 1e-30)
    coherence_x = (node_qxx - node_qyy) / trace
    coherence_y = 2.0 * node_qxy / trace
    support_difference = (
        0.36 * np.abs(np.log(
            np.maximum(node_measure[first], 1e-30)
            / np.maximum(node_measure[second], 1e-30)))
        + 0.22 * np.abs(np.log(
            np.maximum(node_energy[first], 1e-30)
            / np.maximum(node_energy[second], 1e-30)))
        + 0.16 * np.abs(np.log(
            trace[first] / trace[second]))
        + 0.14 * np.hypot(
            coherence_x[first] - coherence_x[second],
            coherence_y[first] - coherence_y[second])
        + 0.12 * np.abs(np.log(
            np.maximum(node_texture[first], 1e-12)
            / np.maximum(node_texture[second], 1e-12)))
    )
    edge["support_jump"] = support_difference

    return {
        "cells": cells,
        "labels": labels,
        "source_site_labels": source_labels,
        "source_site_per_cell": source_site,
        "source_site_count": int(source_labels.max(initial=-1)) + 1,
        "fragment_jump_threshold": float(fragment_jump_threshold),
        "area": area,
        "node_x": node_x,
        "node_y": node_y,
        "touches_border": touches_border,
        "node_lab": node_lab,
        "node_cartoon": node_cartoon,
        "node_glass": node_glass,
        "node_texture": node_texture,
        "node_measure": node_measure,
        "node_energy": node_energy,
        "node_null": node_null,
        "node_qxx": node_qxx,
        "node_qxy": node_qxy,
        "node_qyy": node_qyy,
        "edge": edge,
        "interface_topology": topology,
    }


def _edge_evidence(
    graph: dict,
    config: ObjectSupportConfig,
    focus: dict | None = None,
    intrinsic_owner: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    edge = graph["edge"]
    target = np.maximum(
        _robust_unit(edge["target_jump"]),
        _robust_unit(edge["target_jump_rms"]),
    )
    region_colour = _robust_unit(edge["region_colour_jump"])
    cartoon = _robust_unit(
        0.65 * edge["cartoon_jump"]
        + 0.35 * edge["region_cartoon_jump"])
    cartoon = np.maximum(
        cartoon,
        _robust_unit(edge["cartoon_jump_rms"]),
    )
    glass = np.maximum(
        _robust_unit(edge["glass_jump"]),
        _robust_unit(edge["glass_jump_rms"]),
    )
    # Q contains an intentional isotropic frequency floor.  It controls the
    # maximum support size but is not evidence for an object boundary.
    transport = np.maximum(
        _robust_excess(edge["transport"]),
        _robust_excess(edge["transport_rms"]),
    )
    support = _robust_unit(edge["support_jump"])
    if focus is None or "interface" not in focus:
        focus_jump = np.zeros_like(support)
        focus_reliability = np.zeros_like(support)
    else:
        focus_interface = focus["interface"]
        focus_reliability = np.asarray(
            focus_interface["reliability"], dtype=np.float64)
        focus_jump = (
            _robust_unit(np.abs(
                np.asarray(
                    focus_interface["arc_first_log_effective_scale"],
                    dtype=np.float64,
                )
                - np.asarray(
                    focus_interface["arc_second_log_effective_scale"],
                    dtype=np.float64,
                )
            ))
            * np.clip(focus_reliability, 0.0, 1.0)
        )
    decisive = np.clip(np.maximum(
        edge["boundary"],
        edge["boundary_rms"],
    ), 0.0, 1.0)
    finsler_completion = np.clip(np.asarray(
        graph.get(
            "finsler_contour_completion",
            np.zeros_like(decisive),
        ),
        dtype=np.float64,
    ), 0.0, 1.0)
    if finsler_completion.shape != decisive.shape:
        raise ValueError(
            "Finsler contour completion must match canonical interfaces")
    finsler_amount = np.clip(
        float(config.finsler_contour_weight), 0.0, 1.0)
    # The intrinsic lift can only complete the canonical boundary witness.
    # It cannot lower an observed barrier or directly merge regions.
    completed_decisive = (
        1.0
        - (1.0 - decisive)
        * (1.0 - finsler_amount * finsler_completion)
    )
    focus_jump_raw = focus_jump
    # A difference in local characteristic scale is optical depth evidence
    # only at an observed image boundary. On an arbitrary tessellation seam
    # it is merely different source texture. The frozen BFFT boundary is the
    # analytical witness that distinguishes those cases.
    focus_jump = focus_jump_raw * decisive
    persistent = np.clip(edge["null"], 0.0, 1.0)

    # Unsupported finest-scale activity may not manufacture an object wall.
    # A decisive unchanged-target jump remains decisive even when its
    # neighbourhood is otherwise low-detail.
    local_reliability = (
        1.0
        - np.clip(config.null_suppression, 0.0, 1.0)
        * (1.0 - persistent)
    )
    soft_evidence = local_reliability * (
        max(config.target_jump_weight, 0.0) * target
        + max(config.region_colour_weight, 0.0) * region_colour
        + max(config.cartoon_jump_weight, 0.0) * cartoon
        + max(config.glass_jump_weight, 0.0) * glass
        + max(config.transport_weight, 0.0) * transport
        + max(config.support_jump_weight, 0.0) * support
        + max(config.focus_jump_weight, 0.0) * focus_jump
    )
    additive_evidence = (
        max(config.boundary_weight, 0.0) * completed_decisive
        + soft_evidence
    )
    # Allocation geometry is useful corroboration but is not, by itself, an
    # observed scene discontinuity.  The anchored research control keeps that
    # distinction exact: direct target/cartoon/glass evidence supplies the
    # witness; boundary, region, transport, and support fields may only
    # strengthen an existing witness.  The additive form remains available as
    # a control while the embedded contour topology is being preserved.
    visual_witness = local_reliability * (
        max(config.target_jump_weight, 0.0) * target
        + max(config.cartoon_jump_weight, 0.0) * cartoon
        + max(config.glass_jump_weight, 0.0) * glass
    )
    witness_modulation = (
        1.0
        + max(config.boundary_weight, 0.0) * completed_decisive
        + max(config.region_colour_weight, 0.0) * region_colour
        + max(config.transport_weight, 0.0) * transport
        + max(config.support_jump_weight, 0.0) * support
        + max(config.focus_jump_weight, 0.0) * focus_jump
    )
    anchored_evidence = visual_witness * witness_modulation
    evidence = (
        anchored_evidence
        if config.anchored_barriers
        else additive_evidence
    )
    barrier = 1.0 - np.exp(-np.maximum(evidence, 0.0))
    raw_barrier = barrier.copy()
    contact_scale = max(float(config.short_contact_scale), 0.0)
    if contact_scale > 0.0:
        first, second = edge["first"], edge["second"]
        characteristic_length = contact_scale * np.sqrt(np.minimum(
            graph["area"][first],
            graph["area"][second],
        ))
        contact_reliability = (
            edge["length"]
            / np.maximum(edge["length"] + characteristic_length, 1e-12)
        )
        # A one-pixel permissive contact must not have the same authority as a
        # long quiet interface.  Shrink only short contacts toward a bounded,
        # explicit prior; this is resolution-relative and remains O(E).
        barrier = (
            contact_reliability * barrier
            + (1.0 - contact_reliability)
            * np.clip(float(config.short_contact_prior), 0.0, 1.0)
        )
    else:
        contact_reliability = np.ones_like(barrier)
    tolerance = max(float(config.material_tolerance), 0.0)
    material_join = (
        (edge["target_jump_rms"] <= tolerance)
        & (edge["region_colour_jump"] <= tolerance)
        & (edge["cartoon_jump_rms"] <= tolerance)
        & (
            completed_decisive
            <= max(float(config.material_boundary_ceiling), 0.0)
        )
    )
    # Literal unchanged material is a quotient relation, not merely a high
    # affinity suggestion.  Population/metric variation inside a white panel
    # may control cell shape without manufacturing separate object identity.
    barrier[material_join] = 0.0
    intrinsic_amount = np.clip(
        float(config.intrinsic_contour_weight), 0.0, 1.0)
    if intrinsic_owner is None or intrinsic_amount <= 0.0:
        intrinsic_alignment = np.zeros_like(barrier)
    else:
        intrinsic_alignment = intrinsic_boundary_alignment(
            graph, intrinsic_owner)
    local_contour_barrier = (
        1.0
        - (1.0 - np.clip(barrier, 0.0, 1.0))
        * (1.0 - intrinsic_amount * intrinsic_alignment)
    )
    cycle_amount = np.clip(
        float(config.contour_cycle_weight), 0.0, 1.0)
    if cycle_amount > 0.0:
        cycle_barrier = maximum_bottleneck_cycle_support(
            graph["interface_topology"],
            local_contour_barrier,
            collapse_frame=True,
        )
        barrier = (
            (1.0 - cycle_amount) * barrier
            + cycle_amount * cycle_barrier
        )
    else:
        cycle_barrier = barrier.copy()
    barrier[material_join] = 0.0
    affinity = np.exp(
        -max(float(config.barrier_scale), 1e-6) * barrier)
    return {
        "target_jump": target,
        "region_colour_jump": region_colour,
        "cartoon_jump": cartoon,
        # Keep the visible signal and its actual contribution separate.  A
        # strong cartoon interface can be geometrically informative while
        # contributing little to the scalar barrier because its configured
        # weight is small or its BFFT support is unreliable.
        "cartoon_barrier_contribution": (
            local_reliability
            * max(config.cartoon_jump_weight, 0.0)
            * cartoon
        ),
        "glass_jump": glass,
        "transport_action": transport,
        "support_jump": support,
        "focus_jump": focus_jump,
        "focus_jump_raw": focus_jump_raw,
        "focus_reliability": focus_reliability,
        "visual_witness": visual_witness,
        "latent_support_frontier": np.clip(
            0.5 * transport + 0.5 * support, 0.0, 1.0),
        "additive_barrier": (
            1.0 - np.exp(-np.maximum(additive_evidence, 0.0))
        ),
        "anchored_barrier": (
            1.0 - np.exp(-np.maximum(anchored_evidence, 0.0))
        ),
        "decisive_boundary": decisive,
        "finsler_contour_completion": finsler_completion,
        "completed_decisive_boundary": completed_decisive,
        "null_reliability": persistent,
        "material_join": material_join.astype(np.float64),
        "raw_barrier": raw_barrier,
        "contact_reliability": contact_reliability,
        "intrinsic_boundary_alignment": intrinsic_alignment,
        "local_contour_barrier": local_contour_barrier,
        "cycle_barrier": cycle_barrier,
        "barrier": barrier,
        "affinity": affinity,
    }


def _graph_core_altitude(
    graph: dict,
    barrier: np.ndarray,
) -> np.ndarray:
    """One exact lower envelope of distance from supported interfaces.

    Every interface contributes a source potential rather than passing a
    binary edge threshold.  Strong barriers and the image border start at
    zero; weak interfaces start higher.  The resulting maxima are smooth
    region cores even when the region itself contains no high-detail germ.
    """
    cells = graph["cells"]
    edge = graph["edge"]
    first, second = edge["first"], edge["second"]
    node_wall = np.zeros(cells, dtype=np.float64)
    np.maximum.at(node_wall, first, barrier)
    np.maximum.at(node_wall, second, barrier)
    spacing = np.sqrt(max(float(np.median(graph["area"])), 1.0))
    source_span = max(4.0 * spacing, 1.0)
    distance = source_span * (1.0 - np.clip(node_wall, 0.0, 1.0))
    distance[np.asarray(graph["touches_border"], dtype=bool)] = 0.0
    tree_first = first
    tree_second = second
    travel = np.hypot(
        graph["node_x"][tree_first] - graph["node_x"][tree_second],
        graph["node_y"][tree_first] - graph["node_y"][tree_second],
    )
    offset, neighbour, edge_travel = _tree_csr(
        cells, tree_first, tree_second, np.maximum(travel, 1e-6))
    heap = [(float(value), node) for node, value in enumerate(distance)]
    heapq.heapify(heap)
    tolerance = 1e-12
    while heap:
        value, node = heapq.heappop(heap)
        if value > distance[node] + tolerance:
            continue
        for cursor in range(offset[node], offset[node + 1]):
            target = int(neighbour[cursor])
            candidate = value + float(edge_travel[cursor])
            if candidate + tolerance < distance[target]:
                distance[target] = candidate
                heapq.heappush(heap, (candidate, target))
    scale = max(float(np.percentile(distance, 95.0)), 1e-12)
    return np.clip(distance / scale, 0.0, 1.0)


def _node_seed_score(
    graph: dict,
    evidence: dict,
    config: ObjectSupportConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = graph["cells"]
    edge = graph["edge"]
    first, second = edge["first"], edge["second"]
    length = edge["length"]
    weighted_barrier = evidence["barrier"] * length
    enclosure_total = (
        np.bincount(first, weights=weighted_barrier, minlength=cells)
        + np.bincount(second, weights=weighted_barrier, minlength=cells)
    )
    perimeter = (
        np.bincount(first, weights=length, minlength=cells)
        + np.bincount(second, weights=length, minlength=cells)
    )
    enclosure = enclosure_total / np.maximum(perimeter, 1.0)

    density = _robust_unit(
        graph["node_measure"]
        / np.maximum(np.median(graph["node_measure"]), 1e-30))
    energy = _robust_unit(graph["node_energy"])
    detail = np.sqrt(np.maximum(density * (0.20 + 0.80 * energy), 0.0))
    core = _graph_core_altitude(graph, evidence["barrier"])
    score = (
        max(config.detail_weight, 0.0) * detail
        + max(config.enclosure_weight, 0.0) * enclosure
        + max(config.core_weight, 0.0) * core
    )
    score /= max(
        max(config.detail_weight, 0.0)
        + max(config.enclosure_weight, 0.0)
        + max(config.core_weight, 0.0),
        1e-12,
    )
    return np.clip(score, 0.0, 1.0), enclosure, core


def _material_atoms(
    cells: int,
    first: np.ndarray,
    second: np.ndarray,
    material_join: np.ndarray,
) -> np.ndarray:
    """Connected components of interfaces certified as unchanged material."""
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

    for a, b in zip(first[material_join], second[material_join]):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    root = np.asarray([find(node) for node in range(cells)], dtype=np.int32)
    _, atom = np.unique(root, return_inverse=True)
    return atom.astype(np.int32)


def _local_highpoints(
    score: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """One deterministic graph-local maximum per equal-score plateau edge."""
    cells = len(score)
    # Hash priority is the discrete blue-noise phase.  It breaks exact
    # plateaus without preferring scan order or image axes.
    index = np.arange(cells, dtype=np.uint64)
    priority = (
        (index * np.uint64(11400714819323198485))
        ^ np.uint64(0x9E3779B97F4A7C15)
    )
    peak = np.ones(cells, dtype=bool)
    first_loses = (
        (score[first] < score[second])
        | (
            (score[first] == score[second])
            & (priority[first] < priority[second])
        )
    )
    second_loses = (
        (score[second] < score[first])
        | (
            (score[second] == score[first])
            & (priority[second] < priority[first])
        )
    )
    peak[first[first_loses]] = False
    peak[second[second_loses]] = False
    return np.flatnonzero(peak)


def _support_forest(
    graph: dict,
    affinity: np.ndarray,
    score: np.ndarray,
    peaks: np.ndarray,
) -> dict:
    """Maximum-support forest and zero-dimensional peak persistence."""
    cells = graph["cells"]
    edge = graph["edge"]
    first, second = edge["first"], edge["second"]
    order = np.argsort(-affinity, kind="stable")
    parent = np.arange(cells, dtype=np.int32)
    size = np.ones(cells, dtype=np.int32)
    component_peak = np.full(cells, -1, dtype=np.int32)
    component_peak[peaks] = peaks
    death_affinity = np.zeros(cells, dtype=np.float64)
    tree_first: list[int] = []
    tree_second: list[int] = []
    tree_affinity: list[float] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            old = int(parent[node])
            parent[node] = root
            node = old
        return root

    for edge_index in order:
        a, b = int(first[edge_index]), int(second[edge_index])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        tree_first.append(a)
        tree_second.append(b)
        tree_affinity.append(float(affinity[edge_index]))
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        pa, pb = int(component_peak[ra]), int(component_peak[rb])
        if pa < 0:
            component_peak[ra] = pb
        elif pb >= 0:
            if (
                score[pb] > score[pa]
                or (score[pb] == score[pa] and pb < pa)
            ):
                pa, pb = pb, pa
            death_affinity[pb] = float(affinity[edge_index])
            component_peak[ra] = pa

    prominence = score[peaks] * (1.0 - death_affinity[peaks])
    # The surviving peak in each disconnected component is fully persistent.
    survivors = peaks[death_affinity[peaks] == 0.0]
    prominence[np.isin(peaks, survivors)] = score[survivors]
    return {
        "tree_first": np.asarray(tree_first, dtype=np.int32),
        "tree_second": np.asarray(tree_second, dtype=np.int32),
        "tree_affinity": np.asarray(tree_affinity, dtype=np.float64),
        "peak": peaks,
        "peak_score": score[peaks],
        "peak_death_affinity": death_affinity[peaks],
        "peak_prominence": prominence,
    }


def _tree_csr(
    cells: int,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = (
        np.bincount(first, minlength=cells)
        + np.bincount(second, minlength=cells)
    ).astype(np.int64)
    offset = np.empty(cells + 1, dtype=np.int64)
    offset[0] = 0
    np.cumsum(degree, out=offset[1:])
    neighbour = np.empty(offset[-1], dtype=np.int32)
    edge_weight = np.empty(offset[-1], dtype=np.float64)
    cursor = offset[:-1].copy()
    for a, b, value in zip(first, second, weight):
        ia = cursor[a]
        neighbour[ia], edge_weight[ia] = b, value
        cursor[a] += 1
        ib = cursor[b]
        neighbour[ib], edge_weight[ib] = a, value
        cursor[b] += 1
    return offset, neighbour, edge_weight


def _distance_from_object_interface(
    graph: dict,
    object_id: np.ndarray,
) -> np.ndarray:
    """Geodesic distance inward from the current hard object interfaces."""
    cells = graph["cells"]
    edge = graph["edge"]
    first, second = edge["first"], edge["second"]
    crossing = object_id[first] != object_id[second]
    distance = np.full(cells, np.inf, dtype=np.float64)
    boundary_node = np.unique(np.concatenate((
        first[crossing],
        second[crossing],
    )))
    distance[boundary_node] = 0.0
    travel = np.hypot(
        graph["node_x"][first] - graph["node_x"][second],
        graph["node_y"][first] - graph["node_y"][second],
    )
    offset, neighbour, edge_travel = _tree_csr(
        cells, first, second, np.maximum(travel, 1e-6))
    heap = [(0.0, int(node)) for node in boundary_node]
    heapq.heapify(heap)
    while heap:
        value, node = heapq.heappop(heap)
        if value > distance[node] + 1e-12:
            continue
        for cursor in range(offset[node], offset[node + 1]):
            target = int(neighbour[cursor])
            if object_id[target] != object_id[node]:
                continue
            candidate = value + float(edge_travel[cursor])
            if candidate + 1e-12 < distance[target]:
                distance[target] = candidate
                heapq.heappush(heap, (candidate, target))
    return distance


def _widest_two_on_tree(
    cells: int,
    tree_first: np.ndarray,
    tree_second: np.ndarray,
    tree_affinity: np.ndarray,
    seeds: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> dict:
    """Best and second-best distinct seed under the widest-path semiring."""
    offset, neighbour, edge_weight = _tree_csr(
        cells, tree_first, tree_second, tree_affinity)
    best_value = np.zeros(cells, dtype=np.float64)
    second_value = np.zeros(cells, dtype=np.float64)
    best_seed = np.full(cells, -1, dtype=np.int32)
    second_seed = np.full(cells, -1, dtype=np.int32)
    best_distance = np.full(cells, np.inf, dtype=np.float64)
    second_distance = np.full(cells, np.inf, dtype=np.float64)
    heap: list[tuple[float, float, int, int]] = []
    for seed in np.asarray(seeds, dtype=np.int32):
        best_value[seed] = 1.0
        best_seed[seed] = seed
        best_distance[seed] = 0.0
        heapq.heappush(heap, (-1.0, 0.0, int(seed), int(seed)))

    tolerance = 1e-14
    while heap:
        negative, distance, node, seed = heapq.heappop(heap)
        value = -negative
        valid = (
            (best_seed[node] == seed
             and value + tolerance >= best_value[node]
             and distance <= best_distance[node] + tolerance)
            or (
                second_seed[node] == seed
                and value + tolerance >= second_value[node]
                and distance <= second_distance[node] + tolerance
            )
        )
        if not valid:
            continue
        for cursor in range(offset[node], offset[node + 1]):
            target = int(neighbour[cursor])
            candidate = min(value, float(edge_weight[cursor]))
            candidate_distance = (
                distance
                + float(np.hypot(
                    node_x[target] - node_x[node],
                    node_y[target] - node_y[node],
                ))
            )
            changed = False
            if best_seed[target] == seed:
                if (
                    candidate > best_value[target] + tolerance
                    or (
                        abs(candidate - best_value[target]) <= tolerance
                        and candidate_distance
                        < best_distance[target] - tolerance
                    )
                ):
                    best_value[target] = candidate
                    best_distance[target] = candidate_distance
                    changed = True
            elif second_seed[target] == seed:
                if (
                    candidate > second_value[target] + tolerance
                    or (
                        abs(candidate - second_value[target]) <= tolerance
                        and candidate_distance
                        < second_distance[target] - tolerance
                    )
                ):
                    second_value[target] = candidate
                    second_distance[target] = candidate_distance
                    changed = True
            elif (
                candidate > best_value[target] + tolerance
                or (
                    abs(candidate - best_value[target]) <= tolerance
                    and candidate_distance
                    < best_distance[target] - tolerance
                )
            ):
                second_value[target] = best_value[target]
                second_seed[target] = best_seed[target]
                second_distance[target] = best_distance[target]
                best_value[target] = candidate
                best_seed[target] = seed
                best_distance[target] = candidate_distance
                changed = True
            elif (
                candidate > second_value[target] + tolerance
                or (
                    abs(candidate - second_value[target]) <= tolerance
                    and candidate_distance
                    < second_distance[target] - tolerance
                )
            ):
                second_value[target] = candidate
                second_seed[target] = seed
                second_distance[target] = candidate_distance
                changed = True
            if changed:
                if second_value[target] > best_value[target]:
                    best_value[target], second_value[target] = (
                        second_value[target], best_value[target])
                    best_seed[target], second_seed[target] = (
                        second_seed[target], best_seed[target])
                    best_distance[target], second_distance[target] = (
                        second_distance[target], best_distance[target])
                heapq.heappush(
                    heap,
                    (-candidate, candidate_distance, target, seed),
                )
    return {
        "best_seed": best_seed,
        "second_seed": second_seed,
        "best_value": best_value,
        "second_value": second_value,
        "best_distance": best_distance,
        "second_distance": second_distance,
    }


def _rooted_widest_on_tree(
    cells: int,
    tree_first: np.ndarray,
    tree_second: np.ndarray,
    tree_affinity: np.ndarray,
    seeds: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Connected first-arrival watershed under maximum-support transport.

    The unconstrained two-label solve is useful for measuring competition, but
    it permits a losing seed state to travel through territory already won by
    another seed and reappear in a disconnected branch.  Here a cell is
    settled exactly once by the strongest arriving front.  Every hard basin is
    therefore a connected subtree rooted at its germ.
    """
    offset, neighbour, edge_weight = _tree_csr(
        cells, tree_first, tree_second, tree_affinity)
    winner = np.full(cells, -1, dtype=np.int32)
    value = np.zeros(cells, dtype=np.float64)
    distance = np.full(cells, np.inf, dtype=np.float64)
    predecessor = np.full(cells, -1, dtype=np.int32)
    heap: list[tuple[float, float, int, int, int]] = []
    for seed in np.asarray(seeds, dtype=np.int32):
        heapq.heappush(
            heap, (-1.0, 0.0, int(seed), int(seed), -1))

    while heap:
        negative, path_distance, seed, node, previous = heapq.heappop(heap)
        if winner[node] >= 0:
            continue
        support = -negative
        winner[node] = seed
        value[node] = support
        distance[node] = path_distance
        predecessor[node] = previous
        for cursor in range(offset[node], offset[node + 1]):
            target = int(neighbour[cursor])
            if winner[target] >= 0:
                continue
            candidate = min(support, float(edge_weight[cursor]))
            candidate_distance = (
                path_distance
                + float(np.hypot(
                    node_x[target] - node_x[node],
                    node_y[target] - node_y[node],
                ))
            )
            heapq.heappush(
                heap,
                (
                    -candidate,
                    candidate_distance,
                    seed,
                    target,
                    node,
                ),
            )
    return {
        "best_seed": winner,
        "best_value": value,
        "best_distance": distance,
        "predecessor": predecessor,
    }


def _stable_colours(ids: np.ndarray) -> np.ndarray:
    value = np.asarray(ids, dtype=np.uint32)
    value = value * np.uint32(747796405) + np.uint32(2891336453)
    value = (
        ((value >> ((value >> 28) + 4)) ^ value)
        * np.uint32(277803737)
    )
    value = (value >> 22) ^ value
    return (
        0.10
        + 0.88 * np.column_stack((
            value & 255,
            (value >> 8) & 255,
            (value >> 16) & 255,
        )).astype(np.float64) / 255.0
    )


def _interface_pixel_map(
    labels: np.ndarray,
    graph: dict,
    values: np.ndarray,
) -> np.ndarray:
    """Rasterize one graph-edge scalar back onto its literal interface."""
    topology = graph.get("interface_topology")
    if topology is not None:
        edgel = topology["edgel"]
        arc_value = np.asarray(values, dtype=np.float64)
        sample = arc_value[np.asarray(edgel["arc"], dtype=np.int32)]
        out = np.zeros(labels.shape, dtype=np.float64)
        height, width = labels.shape
        stride = width + 1
        vertex = np.asarray(edgel["vertex_first"], dtype=np.int64)
        x = vertex % stride
        y = vertex // stride
        vertical = np.asarray(edgel["orientation"]) == 1
        coordinates = (
            (np.where(vertical, y, y - 1),
             np.where(vertical, x - 1, x)),
            (np.where(vertical, y, y),
             np.where(vertical, x, x)),
        )
        for py, px in coordinates:
            valid = (
                (py >= 0) & (py < height)
                & (px >= 0) & (px < width)
            )
            np.maximum.at(out, (py[valid], px[valid]), sample[valid])
        return out

    cells = graph["cells"]
    edge = graph["edge"]
    key = (
        edge["first"].astype(np.int64) * cells
        + edge["second"].astype(np.int64)
    )
    order = np.argsort(key)
    ordered_key = key[order]
    ordered_value = np.asarray(values, dtype=np.float64)[order]
    out = np.zeros(labels.shape, dtype=np.float64)

    def paint(first, second, first_slice, second_slice):
        crossing = first != second
        a, b = first[crossing], second[crossing]
        sample_key = (
            np.minimum(a, b).astype(np.int64) * cells
            + np.maximum(a, b).astype(np.int64)
        )
        position = np.searchsorted(ordered_key, sample_key)
        sample = ordered_value[position]
        first_out = out[first_slice]
        second_out = out[second_slice]
        first_out[crossing] = np.maximum(first_out[crossing], sample)
        second_out[crossing] = np.maximum(second_out[crossing], sample)

    paint(
        labels[:, :-1], labels[:, 1:],
        (slice(None), slice(0, -1)),
        (slice(None), slice(1, None)),
    )
    paint(
        labels[:-1], labels[1:],
        (slice(0, -1), slice(None)),
        (slice(1, None), slice(None)),
    )
    return out


def infer_object_support(
    result: dict,
    target_rgb: np.ndarray,
    config: ObjectSupportConfig = ObjectSupportConfig(),
    *,
    graph: dict | None = None,
    intrinsic_owner: np.ndarray | None = None,
) -> dict:
    """Infer hard IDs and soft membership confidence from transport cells."""
    started = time.perf_counter()
    if graph is None:
        graph = build_cell_interface_graph(
            result,
            target_rgb,
            fragment_jump_threshold=config.fragment_jump_threshold,
        )
    focus = graph.get("focus_forensics")
    if focus is None:
        focus = transport_focus_forensics(
            target_rgb,
            graph["labels"],
        )
        focus["interface"] = transport_focus_interfaces(
            focus,
            graph["labels"],
            graph["interface_topology"],
        )
        graph["focus_forensics"] = focus
    graph_ms = 1000.0 * (time.perf_counter() - started)
    analysis_started = time.perf_counter()
    evidence = _edge_evidence(
        graph, config, focus, intrinsic_owner)
    score, enclosure, core = _node_seed_score(graph, evidence, config)
    edge = graph["edge"]
    material_atom = _material_atoms(
        graph["cells"],
        edge["first"],
        edge["second"],
        evidence["material_join"] > 0.5,
    )
    highpoints = _local_highpoints(
        score, edge["first"], edge["second"])
    forest = _support_forest(
        graph, evidence["affinity"], score, highpoints)
    autofocus = autofocus_cell_score(focus)
    focus_seed_weight = float(np.clip(
        config.focus_seed_weight, 0.0, 1.0))
    # Autofocus is a veto, never a source of new objects. Confidently
    # defocused peaks can lose persistence; sharp or unknown peaks retain the
    # exact geometry-derived prominence they already had.
    defocus_veto = np.maximum(
        1.0 - 2.0 * autofocus[highpoints], 0.0)
    peak_selection_prominence = forest["peak_prominence"] * (
        1.0 - focus_seed_weight * defocus_veto)
    selection_prominence = np.zeros(graph["cells"], dtype=np.float64)
    selection_prominence[highpoints] = peak_selection_prominence
    selected = highpoints[
        peak_selection_prominence
        >= max(float(config.peak_prominence), 0.0)
    ]
    if selected.size == 0:
        selected = np.array([int(np.argmax(score))], dtype=np.int32)
    competition = _widest_two_on_tree(
        graph["cells"],
        forest["tree_first"],
        forest["tree_second"],
        forest["tree_affinity"],
        selected,
        graph["node_x"],
        graph["node_y"],
    )
    rooted = _rooted_widest_on_tree(
        graph["cells"],
        forest["tree_first"],
        forest["tree_second"],
        forest["tree_affinity"],
        selected,
        graph["node_x"],
        graph["node_y"],
    )
    rooted_seed = rooted["best_seed"]
    global_best_is_winner = competition["best_seed"] == rooted_seed
    second_seed = np.where(
        global_best_is_winner,
        competition["second_seed"],
        competition["best_seed"],
    ).astype(np.int32)
    second_value = np.where(
        global_best_is_winner,
        competition["second_value"],
        competition["best_value"],
    )
    second_distance = np.where(
        global_best_is_winner,
        competition["second_distance"],
        competition["best_distance"],
    )
    propagation = {
        **rooted,
        "second_seed": second_seed,
        "second_value": second_value,
        "second_distance": second_distance,
        "unconstrained": competition,
    }
    seed_to_object = np.full(graph["cells"], -1, dtype=np.int32)
    seed_to_object[selected] = np.arange(len(selected), dtype=np.int32)
    object_id = seed_to_object[propagation["best_seed"]]
    second_object_id = np.where(
        propagation["second_seed"] >= 0,
        seed_to_object[np.maximum(propagation["second_seed"], 0)],
        -1,
    )
    saddle_margin = (
        propagation["best_value"] - propagation["second_value"]
    ) / np.maximum(propagation["best_value"], 1e-12)
    maximum_distance = np.zeros(len(selected), dtype=np.float64)
    np.maximum.at(
        maximum_distance,
        np.maximum(object_id, 0),
        np.where(
            np.isfinite(propagation["best_distance"]),
            propagation["best_distance"],
            0.0,
        ),
    )
    distance_altitude = 1.0 - (
        propagation["best_distance"]
        / np.maximum(maximum_distance[np.maximum(object_id, 0)], 1e-12)
    )
    distance_altitude = np.clip(distance_altitude, 0.0, 1.0)
    distance_altitude[
        maximum_distance[np.maximum(object_id, 0)] <= 1e-12
    ] = 1.0
    temperature = max(float(config.confidence_temperature), 1e-6)
    # Soft ownership is a statement about competition, not distance from the
    # winning seed.  The old distance blend forced the edge of every large,
    # perfectly certain object toward 50/50 and visually split Pikachu's
    # connected white surround.  The widest-path saddle margin is the actual
    # best-vs-runner ambiguity supplied by the hierarchy.
    first = np.maximum(propagation["best_value"], 1e-300)
    second = np.maximum(propagation["second_value"], 1e-300)
    log_odds = (
        np.log(first) - np.log(second)
    ) / max(float(config.barrier_scale) * temperature, 1e-12)
    first_weight = 0.5 + 0.5 * np.tanh(
        0.5 * np.clip(log_odds, -80.0, 80.0)
    )
    interface_distance = _distance_from_object_interface(graph, object_id)
    uncertainty_span = max(
        2.0 * np.sqrt(float(np.median(graph["area"]))),
        1.0,
    )
    interior_certainty = 1.0 - np.exp(
        -interface_distance / uncertainty_span)
    first_weight = (
        first_weight
        + (1.0 - first_weight) * interior_certainty
    )
    first_weight[propagation["second_seed"] < 0] = 1.0
    colours = _stable_colours(selected)
    hard_colour = colours[np.maximum(object_id, 0)]
    second_colour = hard_colour.copy()
    has_second = second_object_id >= 0
    second_colour[has_second] = colours[second_object_id[has_second]]
    soft_colour = (
        first_weight[:, None] * hard_colour
        + (1.0 - first_weight[:, None]) * second_colour
    )

    labels = np.asarray(graph["labels"], dtype=np.int32)
    object_labels = object_id[labels]
    atom_colours = _stable_colours(
        np.arange(int(material_atom.max(initial=-1)) + 1, dtype=np.int32))
    analysis_ms = 1000.0 * (time.perf_counter() - analysis_started)
    interface_maps = {
        name: _interface_pixel_map(labels, graph, value)
        for name, value in evidence.items()
        if name != "affinity"
    }
    edge_first = graph["edge"]["first"]
    edge_second = graph["edge"]["second"]
    object_crossing = object_id[edge_first] != object_id[edge_second]
    # This is deliberately a diagnostic rather than another object rule.
    # It exposes exactly where a visible cartoon interface failed to become
    # an object cut.  Such an interface may be a material seam inside an
    # object, or—as at Pikachu's black-on-black ear apex—an incomplete
    # silhouette whose ownership must be recovered from contour topology.
    unresolved_cartoon = (
        evidence["cartoon_jump"] * (~object_crossing)
    )
    resolved_cartoon = evidence["cartoon_jump"] * object_crossing
    interface_maps["unresolved_cartoon_jump"] = _interface_pixel_map(
        labels, graph, unresolved_cartoon)
    interface_maps["resolved_cartoon_jump"] = _interface_pixel_map(
        labels, graph, resolved_cartoon)
    return {
        "graph": graph,
        "focus_forensics": focus,
        "autofocus_cell_score": autofocus,
        "autofocus_selection_prominence": selection_prominence,
        "config": config,
        "evidence": evidence,
        "seed_score": score,
        "enclosure": enclosure,
        "core_altitude_per_cell": core,
        "material_atom_per_cell": material_atom,
        "material_atom_labels": material_atom[labels],
        "material_atom_ids": atom_colours[material_atom][labels],
        "highpoints": highpoints,
        "selected_seeds": selected,
        "forest": forest,
        "propagation": propagation,
        "object_id_per_cell": object_id,
        "second_object_id_per_cell": second_object_id,
        "object_labels": object_labels,
        "hard_ids": hard_colour[labels],
        "soft_ids": soft_colour[labels],
        "confidence": saddle_margin[labels],
        "distance_altitude": distance_altitude[labels],
        "object_interface_distance": interface_distance[labels],
        "soft_winner_weight": first_weight[labels],
        "saddle_margin": saddle_margin[labels],
        "waterline": propagation["best_value"][labels],
        "seed_score_map": score[labels],
        "enclosure_map": enclosure[labels],
        "core_altitude_map": core[labels],
        "interface_maps": interface_maps,
        "unresolved_cartoon_jump_per_edge": unresolved_cartoon,
        "resolved_cartoon_jump_per_edge": resolved_cartoon,
        "timing": {
            "graph_ms": graph_ms,
            "analysis_ms": analysis_ms,
            "total_ms": graph_ms + analysis_ms,
        },
    }
