"""Deterministic checks for oriented HPWL interval stalks."""

import numpy as np

from .interval_connection import (
    decompose_interval_stalk,
    parallel_transport_interval_detail,
    synthesize_interval_stalk,
)


def main() -> None:
    points = np.asarray(((5.0, 0.0), (5.0, 10.0)))
    weights = np.ones(2)
    diagonal = np.asarray(((1.0, 1.0), (1.0, -1.0)))
    diagonal /= np.linalg.norm(diagonal, axis=1)[:, None]
    even, odd, signs, info = decompose_interval_stalk(
        diagonal, weights, points
    )
    assert np.allclose(even, [np.sqrt(0.5), 0.0])
    assert np.allclose(odd, [0.0, np.sqrt(0.5)])
    assert np.allclose(signs, [[0.0, 1.0], [0.0, -1.0]])
    assert np.allclose(
        synthesize_interval_stalk(even, odd, signs), diagonal
    )
    assert info["paired_axis_count"] == 1
    assert not info["candidate_destinations_materialized"]

    # A fixed opposite face supplies geometry but no relative orientation.
    fixed = np.asarray(((5.0, 20.0),))
    _, one_sided_odd, _, one_sided = decompose_interval_stalk(
        diagonal, weights, points, fixed
    )
    assert np.allclose(one_sided_odd, 0.0)
    assert one_sided["paired_axis_count"] == 0

    active = np.asarray(((0, 1, 2), (0, 1, 2)))
    anchors = np.asarray((1, 1))
    segment_centers = np.asarray(((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)))
    reference = np.asarray(((0.2, 0.6, 0.2), (0.3, 0.4, 0.3)))
    transported = np.asarray(((0.1, 0.6, 0.3), (0.3, 0.4, 0.3)))
    cell_odd = np.asarray(((0.0, 0.75), (0.0, -0.5)))
    masses = np.asarray((2.0, 3.0))
    lifted, balance = parallel_transport_interval_detail(
        transported,
        reference,
        active,
        anchors,
        segment_centers,
        cell_odd,
        masses,
    )
    assert balance["converged"]
    assert balance["row_mass_max_error"] < 1e-12
    assert balance["segment_mass_max_error"] < 0.01
    assert np.all(lifted >= 0.0)
    assert np.allclose(np.sum(lifted, axis=1), 1.0)
    assert np.allclose(
        np.sum(masses[:, None] * lifted, axis=0),
        np.sum(masses[:, None] * transported, axis=0),
        atol=0.01,
    )

    identity, _ = parallel_transport_interval_detail(
        reference,
        reference,
        active,
        anchors,
        segment_centers,
        cell_odd,
        masses,
    )
    assert np.allclose(identity, reference)
    unchanged, _ = parallel_transport_interval_detail(
        transported,
        reference,
        active,
        anchors,
        segment_centers,
        np.zeros_like(cell_odd),
        masses,
    )
    assert np.allclose(unchanged, transported)
    print("oriented interval stalk checks: PASS")


if __name__ == "__main__":
    main()
