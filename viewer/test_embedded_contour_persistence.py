import numpy as np

from experiments.embedded_contour_persistence import (
    maximum_bottleneck_cycle_support,
)


def _topology(edges, *, shape=(8, 8)):
    endpoints = np.asarray(edges, dtype=np.int64).ravel()
    count = len(edges)
    return {
        "shape": shape,
        "arc": {
            "count": count,
            "endpoint_offset": np.arange(
                0, 2 * count + 1, 2, dtype=np.int64),
            "endpoint_vertex": endpoints,
        },
    }


def test_triangle_cycle_uses_its_weakest_arc():
    topology = _topology(((10, 11), (11, 12), (12, 10)))
    support = maximum_bottleneck_cycle_support(
        topology, np.array([0.9, 0.8, 0.3]))
    np.testing.assert_allclose(support, 0.3)


def test_open_bridge_has_no_closed_contour_support():
    topology = _topology((
        (10, 11), (11, 12), (12, 10), (12, 13),
    ))
    support = maximum_bottleneck_cycle_support(
        topology, np.array([0.9, 0.8, 0.7, 1.0]))
    np.testing.assert_allclose(support, [0.7, 0.7, 0.7, 0.0])


def test_frame_collapse_makes_relative_cycle():
    # Vertices 0 and 8 lie on the top frame of an 8x8 image.
    topology = _topology(((0, 10), (10, 8)))
    relative = maximum_bottleneck_cycle_support(
        topology, np.array([0.8, 0.6]), collapse_frame=True)
    absolute = maximum_bottleneck_cycle_support(
        topology, np.array([0.8, 0.6]), collapse_frame=False)
    np.testing.assert_allclose(relative, 0.6)
    np.testing.assert_allclose(absolute, 0.0)
