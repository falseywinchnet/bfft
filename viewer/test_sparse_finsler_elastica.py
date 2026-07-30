import numpy as np

from experiments.sparse_finsler_elastica import (
    SparseElasticaConfig,
    build_sparse_elastica_graph,
    elastica_common_surround_relations,
    finsler_saliency_closing,
    sparse_elastica_distance,
    trace_sparse_elastica_path,
)


def _topology(edges, tangents, lengths=None):
    count = len(edges)
    endpoints = np.asarray(edges, dtype=np.int64).ravel()
    tangent = np.asarray(tangents, dtype=np.float64).reshape(-1, 2)
    if lengths is None:
        lengths = np.ones(count, dtype=np.float64)
    return {
        "arc": {
            "count": count,
            "endpoint_offset": np.arange(
                0, 2 * count + 1, 2, dtype=np.int64),
            "endpoint_vertex": endpoints,
            "endpoint_tangent_x": tangent[:, 0],
            "endpoint_tangent_y": tangent[:, 1],
            "length": np.asarray(lengths, dtype=np.float64),
        },
    }


def test_straight_continuation_costs_less_than_right_angle_turn():
    topology = _topology(
        ((0, 1), (1, 2), (1, 3)),
        (
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
            ((0, 1), (0, -1)),
        ),
    )
    graph = build_sparse_elastica_graph(
        topology,
        np.ones(3),
        SparseElasticaConfig(curvature_scale=4.0),
    )
    result = sparse_elastica_distance(
        graph, np.array([0]), targets=np.array([2, 4]))
    assert result["distance"][2] < result["distance"][4]


def test_directional_path_does_not_immediately_reverse_an_arc():
    topology = _topology(
        ((0, 1), (1, 2)),
        (
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
        ),
    )
    graph = build_sparse_elastica_graph(topology, np.ones(2))
    result = sparse_elastica_distance(
        graph, np.array([0]), targets=np.array([2]))
    path = trace_sparse_elastica_path(result, 2)
    np.testing.assert_array_equal(path, [0, 2])
    assert graph["state_reverse"][path[0]] != path[1]


def test_high_speed_arc_has_lower_data_action():
    topology = _topology(
        ((0, 1), (1, 2), (1, 3)),
        (
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
        ),
        lengths=(2.0, 3.0, 3.0),
    )
    graph = build_sparse_elastica_graph(
        topology,
        np.array([1.0, 1.0, 0.25]),
        SparseElasticaConfig(curvature_scale=0.0),
    )
    result = sparse_elastica_distance(
        graph, np.array([0]), targets=np.array([2, 4]))
    assert result["distance"][2] < result["distance"][4]


def test_common_surround_falls_out_of_boundary_pair_intersection():
    topology = _topology(
        ((0, 1), (1, 2), (2, 3)),
        (
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
        ),
    )
    graph = build_sparse_elastica_graph(topology, np.ones(3))
    relation = elastica_common_surround_relations(
        graph,
        {
            "first": np.array([0, -1, 1], dtype=np.int32),
            "second": np.array([2, -1, 2], dtype=np.int32),
            "purity": np.ones(3),
        },
        np.ones(3),
    )
    assert np.any(
        (relation["first"] == 0)
        & (relation["second"] == 1)
        & (relation["surround"] == 2)
    )


def test_two_sided_finsler_closing_lifts_a_weak_straight_gap():
    topology = _topology(
        ((0, 1), (1, 2), (2, 3)),
        (
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
            ((1, 0), (-1, 0)),
        ),
    )
    graph = build_sparse_elastica_graph(
        topology,
        np.ones(3),
        SparseElasticaConfig(curvature_scale=2.0),
    )
    result = finsler_saliency_closing(
        graph,
        np.array([1.0, 0.01, 1.0]),
        continuation_scale=8.0,
    )
    assert result["saliency"][1] > 0.01
    assert result["lift"][1] > result["lift"][0]
