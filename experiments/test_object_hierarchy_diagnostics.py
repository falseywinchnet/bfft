"""Controls for read-only transport-object hierarchy diagnostics."""

from __future__ import annotations

import numpy as np

from experiments.object_hierarchy_diagnostics import (
    analyze_object_hierarchy,
    format_selection_report,
    object_boundary_context,
    selection_report,
)


def _fixture():
    # Six cells in a chain.  Objects split the first continuous material,
    # while object 2 spans both sides of a strong material boundary.
    labels = np.tile(np.arange(6, dtype=np.int32), (2, 1))
    first = np.arange(5, dtype=np.int32)
    second = first + 1
    area = np.full(6, 2.0)
    graph = {
        "cells": 6,
        "area": area,
        "node_x": np.arange(6, dtype=np.float64),
        "node_y": np.full(6, 0.5),
        "node_lab": np.column_stack((
            np.arange(6), np.zeros(6), np.zeros(6))).astype(np.float64),
        "node_cartoon": np.arange(6, dtype=np.float64),
        "node_glass": np.arange(6, dtype=np.float64),
        "node_texture": np.ones(6),
        "node_measure": np.ones(6),
        "node_energy": np.ones(6),
        "node_null": np.ones(6),
        "node_qxx": np.ones(6),
        "node_qxy": np.zeros(6),
        "node_qyy": np.ones(6),
        "edge": {
            "first": first,
            "second": second,
            "length": np.ones(5),
        },
    }
    barrier = np.array([0.1, 0.1, 0.9, 0.1, 0.1])
    evidence = {
        name: barrier.copy()
        for name in (
            "target_jump",
            "region_colour_jump",
            "cartoon_jump",
            "glass_jump",
            "transport_action",
            "support_jump",
            "decisive_boundary",
            "null_reliability",
            "barrier",
        )
    }
    best = np.ones(6)
    objects = {
        "graph": graph,
        "evidence": evidence,
        "object_id_per_cell": np.array([0, 0, 1, 1, 2, 2]),
        "second_object_id_per_cell": np.array([1, 1, 0, 2, 1, 1]),
        "selected_seeds": np.array([0, 2, 4]),
        "seed_score": np.linspace(0.2, 0.8, 6),
        "propagation": {
            "best_value": best,
            "second_value": np.full(6, 0.5),
        },
    }
    return {"labels": labels}, objects


def test_material_quotient_exposes_both_failure_directions():
    result, objects = _fixture()
    diagnostic = analyze_object_hierarchy(
        result, objects, barrier_threshold=0.35)
    material = diagnostic["material"]

    assert material["material_count"] == 2
    # Cells 0..2 are one material split between objects 0 and 1.
    assert material["material_object_count"][0] == 2
    assert material["material_object_quotient"][0] == 2.0 / 3.0
    # Object 1 crosses the strong interface and therefore contains two pieces.
    assert material["object_material_component_count"][1] == 2
    assert material["object_connected_material_quotient"][1] == 0.5


def test_object_means_and_boundary_context_are_area_weighted():
    result, objects = _fixture()
    diagnostic = analyze_object_hierarchy(result, objects)
    means = diagnostic["objects"]
    assert np.allclose(means["centroid_x"], [0.5, 2.5, 4.5])
    assert np.allclose(means["lab"][:, 0], [0.5, 2.5, 4.5])

    context = object_boundary_context(
        objects, 2, adjacency=diagnostic["adjacency"])
    assert context["neighbour"].tolist() == [1]
    assert np.isclose(context["barrier"][0], 0.1)


def test_selection_report_maps_pixel_to_cell_object_and_material():
    result, objects = _fixture()
    diagnostic = analyze_object_hierarchy(
        result, objects, barrier_threshold=0.35)
    report = selection_report(
        result,
        objects,
        1,
        4,
        quotient=diagnostic["material"],
        adjacency=diagnostic["adjacency"],
    )
    assert report["cell_id"] == 4
    assert report["object_id"] == 2
    assert report["material_id"] == 1
    text = format_selection_report(report)
    assert "cell 4 → object 2" in text
    assert "material 1" in text
