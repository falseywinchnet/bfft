import numpy as np

from experiments.transport_relation_forensics import (
    transport_anchor_cell_fields,
    transport_relation_forensics,
)


def test_signed_boundary_role_distinguishes_opposite_sides():
    objects = {
        "object_id_per_cell": np.array([0, 1], dtype=np.int32),
        "object_labels": np.array([[0, 1]], dtype=np.int32),
        "selected_seeds": np.array([0, 1], dtype=np.int32),
        "evidence": {
            name: np.array([0.5])
            for name in (
                "cartoon_jump", "glass_jump", "transport_action",
                "support_jump", "null_reliability", "decisive_boundary",
                "target_jump", "region_colour_jump",
            )
        },
        "graph": {
            "cells": 2,
            "labels": np.array([[0, 1]], dtype=np.int32),
            "area": np.ones(2),
            "node_x": np.array([0.0, 1.0]),
            "node_y": np.zeros(2),
            "node_lab": np.array([[0.8, 0.1, 0.0], [0.1, 0.0, 0.0]]),
            "node_cartoon": np.array([1.0, 0.0]),
            "node_glass": np.array([1.0, 0.0]),
            "node_texture": np.array([2.0, 1.0]),
            "node_measure": np.array([2.0, 1.0]),
            "node_energy": np.array([2.0, 1.0]),
            "node_null": np.array([1.0, 0.0]),
            "node_qxx": np.array([2.0, 1.0]),
            "node_qxy": np.zeros(2),
            "node_qyy": np.ones(2),
            "edge": {
                "first": np.array([0], dtype=np.int32),
                "second": np.array([1], dtype=np.int32),
                "length": np.ones(1),
            },
        },
    }
    result = transport_relation_forensics(objects, feature_dimension=64)
    assert result["boundary_transport_similarity"].shape == (2, 2)
    assert np.allclose(
        np.diag(result["boundary_transport_similarity"]), 1.0)
    assert result["boundary_transport_similarity"][0, 1] < 1.0


def test_anchor_cell_fields_precede_object_grouping():
    objects = {
        "object_id_per_cell": np.array([0, 0], dtype=np.int32),
        "graph": {
            "cells": 2,
            "node_lab": np.array([[0.8, 0.1, 0.0], [0.1, 0.0, 0.0]]),
            "node_cartoon": np.array([1.0, 0.0]),
            "node_glass": np.array([1.0, 0.0]),
            "node_texture": np.array([2.0, 1.0]),
            "node_measure": np.array([2.0, 1.0]),
            "node_energy": np.array([2.0, 1.0]),
            "node_null": np.array([1.0, 0.0]),
            "node_qxx": np.array([2.0, 1.0]),
            "node_qxy": np.zeros(2),
            "node_qyy": np.ones(2),
        },
    }
    result = transport_anchor_cell_fields(objects, 0)
    assert result["anchor_cell"] == 0
    for name in ("colour", "action", "metric", "action_metric", "full_state"):
        assert result[name].shape == (2,)
        assert result[name][0] == 1.0
        assert result[name][1] < 1.0
