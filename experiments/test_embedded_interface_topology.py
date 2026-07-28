import numpy as np

from experiments.embedded_interface_topology import (
    build_embedded_interface_topology,
)


def test_parallel_interfaces_of_same_pair_remain_distinct_arcs():
    labels = np.array([
        [0, 1, 0],
        [0, 1, 0],
    ], dtype=np.int32)
    topology = build_embedded_interface_topology(labels)
    assert topology["arc"]["count"] == 2
    assert np.all(topology["arc"]["length"] == 2)


def test_t_junction_preserves_three_arcs_and_one_junction():
    labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ], dtype=np.int32)
    topology = build_embedded_interface_topology(labels)
    assert topology["arc"]["count"] == 3
    assert topology["junction"]["count"] == 1
    assert np.diff(topology["junction"]["arc_offset"])[0] == 3


def test_closed_island_has_one_closed_arc_and_no_junction():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2:5, 2:5] = 1
    topology = build_embedded_interface_topology(labels)
    assert topology["arc"]["count"] == 1
    assert topology["arc"]["closed"].tolist() == [True]
    assert topology["junction"]["count"] == 0

