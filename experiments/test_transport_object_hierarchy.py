import numpy as np

from experiments.transport_object_hierarchy import (
    ParentHierarchyConfig,
    _common_surround_wavefront,
    _enclosed_seam_relations,
    _junction_relations,
    _signed_parent_forest,
)
from experiments.embedded_interface_topology import (
    build_embedded_interface_topology,
)


def test_t_junction_proposes_internal_seam_not_common_surround():
    labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ], dtype=np.int32)
    topology = build_embedded_interface_topology(labels)
    # Regions 0 and 1 have the same signed contrast relative to surround 2.
    lab = np.array([[0.8, 0.1, 0.0], [0.7, 0.1, 0.0], [0.1, 0.0, 0.0]])
    relation = _junction_relations(
        topology,
        lab,
        ParentHierarchyConfig(
            continuation_floor=0.5,
            polarity_floor=0.0,
            relation_floor=0.2,
            tangent_span=3,
        ),
    )
    assert len(relation["score"]) == 1
    assert {int(relation["first"][0]), int(relation["second"][0])} == {0, 1}
    assert int(relation["surround"][0]) == 2
    arcs = topology["arc"]
    for name, region in (
        ("first_outer_arc", int(relation["first"][0])),
        ("second_outer_arc", int(relation["second"][0])),
    ):
        arc = int(relation[name][0])
        assert {
            int(arcs["cell_first"][arc]),
            int(arcs["cell_second"][arc]),
        } == {region, 2}


def test_signed_forest_keeps_common_surround_outside_parent():
    junction = {
        "first": np.array([0], dtype=np.int32),
        "second": np.array([1], dtype=np.int32),
        "surround": np.array([2], dtype=np.int32),
        "score": np.array([0.9]),
    }
    containment = {
        "child": np.empty(0, dtype=np.int32),
        "container": np.empty(0, dtype=np.int32),
        "score": np.empty(0),
    }
    parent, accepted = _signed_parent_forest(
        3,
        junction,
        containment,
        ParentHierarchyConfig(
            relation_floor=0.5,
            minimum_junction_support=1,
            junction_attraction=True,
        ),
    )
    assert parent[0] == parent[1]
    assert parent[0] != parent[2]
    assert len(accepted["score"]) == 1


def test_bounded_pair_with_one_shared_exterior_is_attachment_alternative():
    labels = np.full((7, 7), 2, dtype=np.int32)
    labels[2:4, 2:5] = 0
    labels[4:6, 2:5] = 1
    junction = {
        "first": np.array([0, 0], dtype=np.int32),
        "second": np.array([1, 1], dtype=np.int32),
        "surround": np.array([2, 2], dtype=np.int32),
        "score": np.array([0.9, 0.8]),
    }
    relation = _enclosed_seam_relations(
        labels,
        junction,
        ParentHierarchyConfig(
            minimum_junction_support=2,
            enclosed_seam_dominance=0.9,
        ),
    )
    assert relation["first"].tolist() == [0]
    assert relation["second"].tolist() == [1]
    assert relation["surround"].tolist() == [2]
    assert np.isclose(relation["exterior_dominance"][0], 1.0)


def test_small_frame_exit_is_penalized_not_vetoed():
    labels = np.full((9, 9), 2, dtype=np.int32)
    labels[:4, 3:6] = 0
    labels[4:7, 3:6] = 1
    junction = {
        "first": np.array([0, 0], dtype=np.int32),
        "second": np.array([1, 1], dtype=np.int32),
        "surround": np.array([2, 2], dtype=np.int32),
        "score": np.array([0.9, 0.8]),
    }
    relation = _enclosed_seam_relations(
        labels,
        junction,
        ParentHierarchyConfig(
            minimum_junction_support=2,
            enclosed_seam_dominance=0.6,
        ),
    )
    assert relation["first"].tolist() == [0]
    assert 0.0 < relation["frame_exposure"][0] < 0.25
    assert relation["exterior_dominance"][0] > 0.6


def test_remote_lookalikes_have_no_relation_without_topology():
    junction = {
        "first": np.empty(0, dtype=np.int32),
        "second": np.empty(0, dtype=np.int32),
        "surround": np.empty(0, dtype=np.int32),
        "score": np.empty(0),
    }
    containment = {
        "child": np.empty(0, dtype=np.int32),
        "container": np.empty(0, dtype=np.int32),
        "score": np.empty(0),
    }
    parent, _ = _signed_parent_forest(
        2, junction, containment, ParentHierarchyConfig())
    assert parent.tolist() == [0, 1]


def test_common_surround_first_arrival_proposes_only_colliding_parts():
    part = np.array([2, 2, 2, 2, 2, 0, 1], dtype=np.int32)
    first = np.array([5, 0, 1, 2, 3, 4], dtype=np.int32)
    second = np.array([0, 1, 2, 3, 4, 6], dtype=np.int32)
    cells = len(part)
    node_lab = np.array([
        [0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.8, 0.1, 0.0],
        [0.7, 0.1, 0.0],
    ])
    objects = {
        "object_id_per_cell": part,
        "selected_seeds": np.empty(0, dtype=np.int32),
        "evidence": {"barrier": np.ones(len(first))},
        "graph": {
            "cells": cells,
            "area": np.ones(cells),
            "node_x": np.arange(cells, dtype=np.float64),
            "node_y": np.zeros(cells),
            "node_lab": node_lab,
            "node_qxx": np.ones(cells),
            "node_qyy": np.ones(cells),
            "edge": {
                "first": first,
                "second": second,
                "length": np.ones(len(first)),
            },
        },
    }
    relation = _common_surround_wavefront(
        objects,
        np.arange(3, dtype=np.int32),
        ParentHierarchyConfig(
            completion_gap_scale=10.0,
            completion_relation_floor=0.0,
            completion_collision_support=1,
        ),
    )
    assert len(relation["score"]) == 1
    assert relation["first"].tolist() == [0]
    assert relation["second"].tolist() == [1]
    assert relation["surround"].tolist() == [2]
