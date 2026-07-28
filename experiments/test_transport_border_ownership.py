import numpy as np

from experiments.transport_border_ownership import (
    _directed_junction_votes,
    infer_transport_border_ownership,
)


def test_opposed_junction_votes_retain_only_net_direction():
    junction = {
        "first": np.array([1, 0], dtype=np.int32),
        "second": np.array([2, 2], dtype=np.int32),
        "surround": np.array([0, 1], dtype=np.int32),
        "score": np.array([0.9, 0.2]),
    }
    directed = _directed_junction_votes(junction)
    pairs = {
        (int(a), int(b)): float(value)
        for a, b, value in zip(
            directed["front"],
            directed["back"],
            directed["net_support"],
        )
    }
    assert np.isclose(pairs[(0, 1)], 0.7)
    assert np.isclose(pairs[(0, 2)], 0.9)
    assert np.isclose(pairs[(1, 2)], 0.2)


def test_transport_direction_is_one_closed_form_weighted_accumulation():
    objects = {
        "object_id_per_cell": np.array([0, 1], dtype=np.int32),
        "object_labels": np.array([[0, 1]], dtype=np.int32),
        "selected_seeds": np.array([0, 1], dtype=np.int32),
        "graph": {
            "cells": 2,
            "area": np.ones(2),
            "node_x": np.array([0.0, 1.0]),
            "node_y": np.zeros(2),
            "node_lab": np.zeros((2, 3)),
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
    parent = {
        "junction_relations": {
            "first": np.array([1], dtype=np.int32),
            "second": np.array([1], dtype=np.int32),
            "surround": np.array([0], dtype=np.int32),
            "score": np.array([1.0]),
        },
    }
    result = infer_transport_border_ownership(objects, parent)
    assert result["observed_frontness_per_part"].tolist() == [1.0, -1.0]
    assert result["support_frontness_per_part"][0] > (
        result["support_frontness_per_part"][1])
    assert result["relation_agreement_fraction"] == 1.0
