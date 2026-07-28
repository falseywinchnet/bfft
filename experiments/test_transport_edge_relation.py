import numpy as np

from experiments.test_transport_graph_scattering import _cycle_graph
from experiments.transport_edge_relation import (
    aggregate_signed_relations,
    signed_relation_field,
    transport_edge_relation,
)


def test_signed_relation_is_bounded_and_symmetric_when_aggregated():
    objects = _cycle_graph()
    result = transport_edge_relation(objects, scales=3)
    field = signed_relation_field(result, 0, order=2)
    assert field.shape == (4,)
    assert np.all(np.isfinite(field))
    assert np.max(np.abs(field)) <= 1.0

    owner = np.array([0, 0, 1, 1], dtype=np.int32)
    matrix = aggregate_signed_relations(
        result, owner, np.ones(4), order=2)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, matrix.T)
    assert np.max(np.abs(matrix)) <= 1.0


def test_no_relation_modes_are_selected():
    objects = _cycle_graph()
    result = transport_edge_relation(objects, scales=3)
    assert len(result["relation_eigenvalue"]) == result["feature_count"]
    assert result["coordinates"].shape == (
        objects["graph"]["cells"],
        result["feature_count"],
    )
    assert result["whitened_coordinates"].shape == (
        objects["graph"]["cells"],
        result["feature_count"],
    )
