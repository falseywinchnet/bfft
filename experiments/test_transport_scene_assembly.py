import numpy as np

from experiments.transport_scene_assembly import (
    SceneAssemblyConfig,
    infer_scene_assemblies,
)


def test_bounded_material_attachment_ignores_appearance_but_substrate_does_not():
    objects = {
        "object_id_per_cell": np.array([0, 1, 2], dtype=np.int32),
        "object_labels": np.array([[0, 1, 2]], dtype=np.int32),
        "selected_seeds": np.arange(3, dtype=np.int32),
        "evidence": {"barrier": np.array([0.9, 0.9])},
        "graph": {
            "cells": 3,
            "labels": np.array([[0, 1, 2]], dtype=np.int32),
            "area": np.ones(3),
            "node_x": np.arange(3, dtype=np.float64),
            "node_y": np.zeros(3),
            "node_lab": np.array([
                [0.8, 0.1, 0.0],
                [0.1, -0.1, 0.0],
                [0.7, 0.1, 0.0],
            ]),
            "node_cartoon": np.ones(3),
            "node_glass": np.ones(3),
            "node_texture": np.ones(3),
            "node_measure": np.ones(3),
            "node_energy": np.ones(3),
            "node_null": np.ones(3),
            "node_qxx": np.ones(3),
            "node_qxy": np.zeros(3),
            "node_qyy": np.ones(3),
            "edge": {
                "first": np.array([0, 1], dtype=np.int32),
                "second": np.array([1, 2], dtype=np.int32),
            },
        },
    }
    hierarchy = {
        "frame_geometry": {
            "frame_exposure": np.array([0.0, 0.0, 1.0]),
        },
        "enclosed_seam_relations": {
            "first": np.array([0], dtype=np.int32),
            "second": np.array([1], dtype=np.int32),
            "score": np.array([0.9]),
            "frame_exposure": np.array([0.0]),
        },
        "containment_relations": {
            "child": np.array([0], dtype=np.int32),
            "container": np.array([2], dtype=np.int32),
            "score": np.array([1.0]),
        },
        "completion_relations": {
            "first": np.empty(0, dtype=np.int32),
            "second": np.empty(0, dtype=np.int32),
            "score": np.empty(0),
        },
        "accepted_completion_indices": np.empty(0, dtype=np.int32),
    }
    result = infer_scene_assemblies(
        objects,
        hierarchy,
        SceneAssemblyConfig(relation_floor=0.5),
    )
    ids = result["assembly_id_per_part"]
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_short_cropped_contact_does_not_seed_the_exterior():
    objects = {
        "object_id_per_cell": np.array([0, 1, 1], dtype=np.int32),
        "object_labels": np.array([[0, 1, 1]], dtype=np.int32),
        "selected_seeds": np.array([0, 1], dtype=np.int32),
        "evidence": {"barrier": np.array([0.9, 0.1])},
        "graph": {
            "cells": 3,
            "labels": np.array([[0, 1, 2]], dtype=np.int32),
            "area": np.ones(3),
            "node_x": np.arange(3, dtype=np.float64),
            "node_y": np.zeros(3),
            "node_lab": np.zeros((3, 3)),
            "node_cartoon": np.ones(3),
            "node_glass": np.ones(3),
            "node_texture": np.ones(3),
            "node_measure": np.ones(3),
            "node_energy": np.ones(3),
            "node_null": np.ones(3),
            "node_qxx": np.ones(3),
            "node_qxy": np.zeros(3),
            "node_qyy": np.ones(3),
            "edge": {
                "first": np.array([0, 1], dtype=np.int32),
                "second": np.array([1, 2], dtype=np.int32),
            },
        },
    }
    hierarchy = {
        "frame_geometry": {
            "frame_exposure": np.array([0.8, 0.05]),
        },
        "enclosed_seam_relations": {
            "first": np.empty(0, dtype=np.int32),
            "second": np.empty(0, dtype=np.int32),
            "score": np.empty(0),
            "frame_exposure": np.empty(0),
        },
        "containment_relations": {
            "child": np.empty(0, dtype=np.int32),
            "container": np.empty(0, dtype=np.int32),
            "score": np.empty(0),
        },
        "completion_relations": {
            "first": np.empty(0, dtype=np.int32),
            "second": np.empty(0, dtype=np.int32),
            "score": np.empty(0),
        },
        "accepted_completion_indices": np.empty(0, dtype=np.int32),
    }
    result = infer_scene_assemblies(objects, hierarchy)
    exterior = result["exterior_reachability"]
    assert exterior["cell_is_exterior"].tolist() == [True, False, False]
    assert exterior["cell_basin"].tolist() == [-1, 0, 0]
