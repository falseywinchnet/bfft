"""Regression controls for the experimental transport-object hierarchy."""

from __future__ import annotations

import numpy as np

from experiments.transport_object_support import (
    ObjectSupportConfig,
    _rooted_widest_on_tree,
    _widest_two_on_tree,
    build_cell_interface_graph,
    connected_site_fragments,
    infer_object_support,
)


def _two_panel_fixture():
    height, width = 48, 64
    y, x = np.mgrid[:height, :width]
    labels = (y // 8) * 8 + (x // 8)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    rgb[:, :width // 2] = (0.1, 0.2, 0.8)
    rgb[:, width // 2:] = (0.9, 0.2, 0.1)
    edge = np.zeros((height, width), dtype=np.float64)
    edge[:, width // 2 - 1:width // 2 + 1] = 1.0
    geometry = {
        "boundary_confidence": edge,
        "cartoon": np.mean(rgb, axis=2),
        "texture": np.zeros((height, width), dtype=np.float64),
        "glass": 0.2 * edge,
        "null_confidence": np.ones((height, width), dtype=np.float64),
        "precision_xx": 0.01 + edge,
        "precision_xy": np.zeros((height, width), dtype=np.float64),
        "precision_yy": np.full((height, width), 0.01),
        "boundary_xx": edge,
        "boundary_yy": np.zeros((height, width), dtype=np.float64),
        "measure": np.full((height, width), 1.0 / (height * width)),
        "energy": 0.01 + edge,
    }
    return rgb, {"labels": labels, "geometry": geometry}


def test_literal_interface_graph_is_sparse_and_complete():
    rgb, result = _two_panel_fixture()
    graph = build_cell_interface_graph(result, rgb)
    assert graph["cells"] == 48
    # An 8 by 6 rectangular cell lattice has 7*6 + 8*5 adjacencies.
    assert len(graph["edge"]["first"]) == 82
    assert np.all(graph["edge"]["first"] < graph["edge"]["second"])
    assert np.all(graph["edge"]["length"] > 0)


def test_decisive_interface_preserves_two_objects():
    rgb, result = _two_panel_fixture()
    objects = infer_object_support(
        result,
        rgb,
        ObjectSupportConfig(
            boundary_weight=2.0,
            target_jump_weight=0.0,
            region_colour_weight=0.0,
            cartoon_jump_weight=0.0,
            glass_jump_weight=0.0,
            transport_weight=0.0,
            support_jump_weight=0.0,
            null_suppression=0.0,
            peak_prominence=0.06,
            barrier_scale=3.0,
        ),
    )
    assert len(objects["selected_seeds"]) == 2
    assert np.unique(objects["object_labels"]).size == 2
    assert np.isfinite(objects["soft_ids"]).all()
    assert 0.0 <= float(np.min(objects["confidence"]))
    assert float(np.max(objects["confidence"])) <= 1.0
    assert "unresolved_cartoon_jump" in objects["interface_maps"]
    assert "resolved_cartoon_jump" in objects["interface_maps"]
    assert np.all(
        objects["evidence"]["cartoon_barrier_contribution"] == 0.0)


def test_object_reanalysis_reuses_the_same_graph():
    rgb, result = _two_panel_fixture()
    graph = build_cell_interface_graph(result, rgb)
    first = infer_object_support(
        result,
        rgb,
        ObjectSupportConfig(
            boundary_weight=2.0,
            target_jump_weight=0.0,
            region_colour_weight=0.0,
            cartoon_jump_weight=0.0,
            glass_jump_weight=0.0,
            transport_weight=0.0,
            support_jump_weight=0.0,
            null_suppression=0.0,
            peak_prominence=0.06,
            barrier_scale=3.0,
        ),
        graph=graph,
    )
    second = infer_object_support(
        result,
        rgb,
        ObjectSupportConfig(peak_prominence=0.40, barrier_scale=0.5),
        graph=graph,
    )
    assert first["graph"] is graph
    assert second["graph"] is graph
    assert second["timing"]["graph_ms"] < 1.0
    assert len(second["selected_seeds"]) <= len(first["selected_seeds"])


def test_anchored_mode_cannot_turn_support_only_change_into_a_wall():
    rgb, result = _two_panel_fixture()
    rgb[:] = 0.5
    result["geometry"]["cartoon"][:] = 0.5
    result["geometry"]["glass"][:] = 0.0
    objects = infer_object_support(
        result,
        rgb,
        ObjectSupportConfig(
            anchored_barriers=True,
            boundary_weight=4.0,
            target_jump_weight=3.0,
            cartoon_jump_weight=3.0,
            glass_jump_weight=3.0,
            transport_weight=3.0,
            support_jump_weight=3.0,
            null_suppression=0.0,
        ),
    )
    assert np.all(objects["evidence"]["visual_witness"] == 0.0)
    assert np.all(objects["evidence"]["barrier"] == 0.0)
    assert np.unique(objects["object_labels"]).size == 1


def test_equal_bottlenecks_use_shortest_supported_path_not_seed_order():
    propagation = _widest_two_on_tree(
        5,
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.ones(4, dtype=np.float64),
        np.array([0, 4], dtype=np.int32),
        np.arange(5, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
    )
    assert propagation["best_seed"].tolist() == [0, 0, 0, 4, 4]
    assert propagation["second_seed"].tolist() == [4, 4, 4, 0, 0]


def test_disconnected_and_cross_edge_site_ownership_becomes_fragments():
    labels = np.array([
        [7, 7, 2, 7],
        [7, 7, 2, 7],
    ], dtype=np.int32)
    signal = np.zeros((2, 4, 1), dtype=np.float64)
    signal[:, 1, 0] = 1.0
    fragments, source = connected_site_fragments(
        labels, signal, maximum_jump=0.25)
    seven = np.unique(fragments[labels == 7])
    assert len(seven) == 3
    assert np.all(source[seven] == 7)


def test_rooted_widest_regions_are_connected_first_arrival_subtrees():
    rooted = _rooted_widest_on_tree(
        7,
        np.array([0, 1, 2, 3, 3, 5], dtype=np.int32),
        np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        np.array([0.9, 0.8, 0.7, 0.9, 0.6, 0.9]),
        np.array([0, 4, 6], dtype=np.int32),
        np.arange(7, dtype=np.float64),
        np.zeros(7, dtype=np.float64),
    )
    assert rooted["best_seed"].tolist() == [0, 0, 0, 4, 4, 6, 6]
    for seed in (0, 4, 6):
        positions = np.flatnonzero(rooted["best_seed"] == seed)
        assert np.all(np.diff(positions) == 1)
