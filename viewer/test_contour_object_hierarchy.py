import numpy as np

from experiments.contour_object_hierarchy import (
    cophenetic_ultrametric,
    labels_at_waterline,
    minimum_barrier_tree,
)


def test_minimum_tree_induces_minimax_ultrametric():
    tree = minimum_barrier_tree(
        4,
        np.array([0, 1, 0, 2], dtype=np.int32),
        np.array([1, 2, 2, 3], dtype=np.int32),
        np.array([0.2, 0.6, 0.9, 0.4]),
    )
    distance = cophenetic_ultrametric(4, tree)
    assert distance[0, 2] == 0.6
    assert distance[0, 3] == 0.6
    assert distance[2, 3] == 0.4
    np.testing.assert_allclose(distance, distance.T)


def test_waterline_is_a_nested_cut_of_the_same_tree():
    tree = {
        "first": np.array([0, 2, 1], dtype=np.int32),
        "second": np.array([1, 3, 2], dtype=np.int32),
        "barrier": np.array([0.2, 0.4, 0.6]),
    }
    fine = labels_at_waterline(4, tree, 0.3)
    coarse = labels_at_waterline(4, tree, 0.5)
    assert len(np.unique(fine)) == 3
    assert len(np.unique(coarse)) == 2
    assert fine[0] == fine[1]
    assert coarse[2] == coarse[3]
