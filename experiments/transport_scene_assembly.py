#!/usr/bin/env python3
"""Compound foreground assemblies above material and object parts.

This layer deliberately does not claim semantic identity.  It asks whether
already lawful geometric relations support a bounded perceptual assembly:

* a material seam may join two parts when their union has one exterior;
* a contained/contacting part may join its container when their measured
  appearance and transport support agree;
* an amodal completion already accepted by the transport hierarchy may join.

Frame exposure is continuous substrate evidence.  A cropped foreground part
is penalized in proportion to its exposed contour rather than rejected merely
for touching the image boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.object_hierarchy_diagnostics import object_means

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


@dataclass(frozen=True)
class SceneAssemblyConfig:
    relation_floor: float = 0.46
    colour_scale: float = 0.16
    support_scale: float = 2.25
    frame_penalty: float = 2.0
    exterior_barrier: float = 0.46
    frame_seed_exposure: float = 0.12
    include_enclosed_seams: bool = True
    include_completions: bool = True


@njit(cache=True)
def _permeable_components(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
    permeable: np.ndarray,
) -> np.ndarray:
    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            previous = parent[node]
            parent[node] = root
            node = previous
        return root

    for index in range(len(first)):
        if not permeable[index]:
            continue
        a, b = find(first[index]), find(second[index])
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    for node in range(count):
        parent[node] = find(node)
    return parent


def _exterior_reachability(
    objects: dict,
    hierarchy: dict,
    config: SceneAssemblyConfig,
) -> dict[str, np.ndarray | int]:
    """Components hidden behind the measured interface waterline.

    Exterior support is not every cell touching the image frame.  Only frame
    cells belonging to a broadly frame-exposed part seed the substrate.  This
    is what lets a short cropped ear remain foreground.
    """
    graph = objects["graph"]
    labels = np.asarray(graph["labels"], dtype=np.int32)
    count = int(graph["cells"])
    edge = graph["edge"]
    first = np.asarray(edge["first"], dtype=np.int32)
    second = np.asarray(edge["second"], dtype=np.int32)
    barrier = np.asarray(objects["evidence"]["barrier"], dtype=np.float64)
    roots = _permeable_components(
        count,
        first,
        second,
        barrier < float(config.exterior_barrier),
    )

    frame_cells = np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    )))
    part = np.asarray(objects["object_id_per_cell"], dtype=np.int32)
    part_exposure = np.asarray(
        hierarchy["frame_geometry"]["frame_exposure"], dtype=np.float64)
    seed = frame_cells[
        part_exposure[part[frame_cells]]
        >= float(config.frame_seed_exposure)
    ]
    if seed.size == 0 and frame_cells.size:
        exposure = part_exposure[part[frame_cells]]
        seed = frame_cells[exposure >= np.max(exposure)]

    root_is_exterior = np.zeros(count, dtype=bool)
    root_is_exterior[roots[seed]] = True
    exterior = root_is_exterior[roots]
    bounded_roots = np.unique(roots[~exterior])
    basin_by_root = np.full(count, -1, dtype=np.int32)
    basin_by_root[bounded_roots] = np.arange(
        len(bounded_roots), dtype=np.int32)
    basin = basin_by_root[roots]
    basin_labels = basin[labels]
    display_labels = basin_labels + 1  # zero is the exterior substrate.
    colours = _stable_colours(len(bounded_roots) + 1)
    colours[0] = np.array([0.06, 0.07, 0.09])
    return {
        "cell_is_exterior": exterior,
        "cell_basin": basin,
        "basin_labels": basin_labels,
        "display_labels": display_labels,
        "basin_ids": colours[display_labels],
        "basin_count": len(bounded_roots),
        "frame_seed_cells": seed,
        "frame_seed_map": np.isin(labels, seed),
    }


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


def _support_signature(means: dict[str, np.ndarray]) -> np.ndarray:
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
    center = np.median(raw, axis=0)
    deviation = np.median(np.abs(raw - center), axis=0)
    scale = np.maximum(1.4826 * deviation, 1e-8)
    return (raw - center) / scale


def _relation_scores(
    objects: dict,
    hierarchy: dict,
    config: SceneAssemblyConfig,
) -> dict[str, np.ndarray]:
    means = object_means(objects)
    support = _support_signature(means)
    lab = np.asarray(means["lab"], dtype=np.float64)
    exposure = np.asarray(
        hierarchy["frame_geometry"]["frame_exposure"], dtype=np.float64)
    records: list[tuple[int, int, float, int, float, float, float]] = []

    # kind 0: two material regions share one compound exterior.  Geometry is
    # the reason; appearance disagreement is expected at a material seam.
    if config.include_enclosed_seams:
        relation = hierarchy["enclosed_seam_relations"]
        for a, b, score, frame in zip(
            relation["first"],
            relation["second"],
            relation["score"],
            relation["frame_exposure"],
        ):
            records.append((
                int(a), int(b), float(score), 0,
                1.0, 1.0, float(frame),
            ))

    # kind 1: nested/contact support.  Geometry creates the candidate;
    # compatible appearance and transport decide whether it is one assembly.
    containment = hierarchy["containment_relations"]
    for child, container, geometry in zip(
        containment["child"],
        containment["container"],
        containment["score"],
    ):
        a, b = int(child), int(container)
        colour_distance = float(np.linalg.norm(lab[a] - lab[b]))
        colour = float(np.exp(-(
            colour_distance / max(config.colour_scale, 1e-8)
        ) ** 2))
        support_distance = float(np.sqrt(np.mean(
            (support[a] - support[b]) ** 2)))
        support_match = float(np.exp(-(
            support_distance / max(config.support_scale, 1e-8)
        ) ** 2))
        substrate = float(np.exp(
            -max(config.frame_penalty, 0.0) * exposure[b]))
        score = float(
            geometry
            * np.sqrt(colour * support_match)
            * substrate
        )
        records.append((
            a, b, score, 1,
            colour, support_match, float(exposure[b]),
        ))

    # kind 2: transport-supported amodal completion.  Only relations accepted
    # by the bounded first-arrival layer enter; there is no appearance search.
    if config.include_completions:
        completion = hierarchy["completion_relations"]
        for index in hierarchy["accepted_completion_indices"]:
            cursor = int(index)
            records.append((
                int(completion["first"][cursor]),
                int(completion["second"][cursor]),
                float(completion["score"][cursor]),
                2, 1.0, 1.0, 0.0,
            ))

    names = (
        "first", "second", "score", "kind", "colour_match",
        "support_match", "frame_exposure",
    )
    integer = {"first", "second", "kind"}
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


def infer_scene_assemblies(
    objects: dict,
    hierarchy: dict,
    config: SceneAssemblyConfig = SceneAssemblyConfig(),
) -> dict:
    """Return simultaneous threshold components of lawful assembly relations."""
    part_labels = np.asarray(objects["object_labels"], dtype=np.int32)
    count = int(part_labels.max(initial=-1)) + 1
    relation = _relation_scores(objects, hierarchy, config)
    accepted = relation["score"] >= config.relation_floor

    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            previous = int(parent[node])
            parent[node] = root
            node = previous
        return root

    # All above-waterline relations act simultaneously.  Their order cannot
    # alter the resulting connected components.
    for a, b in zip(
        relation["first"][accepted],
        relation["second"][accepted],
    ):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    roots = np.fromiter(
        (find(node) for node in range(count)),
        dtype=np.int32,
        count=count,
    )
    _, assembly = np.unique(roots, return_inverse=True)
    assembly = assembly.astype(np.int32)
    labels = assembly[part_labels]
    colours = _stable_colours(int(assembly.max(initial=-1)) + 1)
    exterior = _exterior_reachability(
        objects, hierarchy, config)
    return {
        "config": config,
        "relations": relation,
        "accepted": accepted,
        "assembly_id_per_part": assembly,
        "assembly_labels": labels,
        "assembly_ids": colours[labels],
        "assembly_count": int(assembly.max(initial=-1)) + 1,
        "exterior_reachability": exterior,
    }
